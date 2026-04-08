# utils/detect/measurements.py
from __future__ import annotations                          # allows forward references in type hints without quoting types
import csv                                                  # used for writing measurement outputs to CSV files
import math                                                 # used for ratios, gcd reduction, distances, logs, & summary statistics
from collections import defaultdict                         # used for auto-initializing per-class counters & trackers
from datetime import datetime, timedelta, timezone          # used for timestamp normalization & interval math
from pathlib import Path                                    # used for safe cross-platform path handling
from typing import Any                                      # used where row values or config values may vary in type
import yaml                                                 # used for loading & creating measurement configuration YAML files
from shapely.geometry import Polygon                        # used for polygon-based overlap scoring when OBB detections exist
from ..paths import CONFIGS_DIR, MEASURE_CONFIG_YAML        # shared project paths for measurement config storage

# --------- CONFIG ---------
class MeasurementConfig:
    """
    Central configuration for all measurement parameters.
    This class ensures a config file exists, loads any saved overrides, & falls back to built-in defaults when keys are missing.
    """
    DEFAULTS = {
        "avg_group_size": 3,                        # number of snapshot rows grouped together when computing average counts
        "interval_sec": 5,                          # base interval length used by counters, aggregators, interactions, & motion
        "session_sec": 10,                          # larger summary window used for session-level rollups
        "interaction_timeout_sec": 2.0,             # inactivity threshold before an active interaction is closed
        "interaction_max_track_age_frames": 2,      # max allowed tracker age for a track to still count in interactions
        "interaction_min_duration_sec": 0.75,       # minimum sustained overlap duration required before recording an interaction
        "overlap_threshold": 0.1,                   # minimum overlap score required to consider two entities interacting
        "motion_threshold_px": 10.0,                # minimum per-frame displacement before motion persistence is counted
        "motion_min_frames": 3,                     # minimum consecutive qualifying frames before a motion event is recorded
        "track_max_dist_px": 60.0,                  # max center distance allowed when associating tracks to detections
        "track_max_age_frames": 10,                 # max missed frames before a stale track is dropped
    }

    def __init__(self, config_path: str | Path | None = None):
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)                                  # ensure the config directory exists before reading or writing files
        self.config_path = Path(config_path) if config_path else MEASURE_CONFIG_YAML    # use provided config path or fall back to the project default

        if not self.config_path.exists():
            with open(self.config_path, "w") as handle:                                 # create a starter config file the first time measurements are used
                yaml.safe_dump(self.DEFAULTS, handle, sort_keys=False)                  # preserve default key order for readability

        with open(self.config_path, "r") as handle:
            loaded = yaml.safe_load(handle) or {}                                       # load saved config values, defaulting to an empty mapping if file is blank

        for key, default in self.DEFAULTS.items():
            setattr(self, key, loaded.get(key, default))                                # expose every config key as an attribute with fallback to the built-in default

# --------- TIME HELPERS ---------
def _to_utc(dt: datetime | None) -> datetime | None:
    """
    Normalize datetimes to UTC-aware values.
    This keeps timestamp arithmetic consistent across all measurement classes.
    """
    if dt is None:
        return None                                                         # preserve None so callers can continue using optional timestamps safely
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)             # attach UTC when a datetime is naive, otherwise leave timezone-aware values intact

def _fmt_time(dt: datetime) -> str:
    """
    Format a datetime as HH:MM:SS.
    All exported measurement timestamps use this compact display format.
    """
    return _to_utc(dt).strftime("%H:%M:%S")                                 # normalize first so string formatting is consistent across all outputs

# --------- CSV WRITER ---------
class CSVTableWriter:
    """
    Small shared CSV writer for measurement outputs.
    This keeps CSV-writing behavior consistent across counters, interactions, summaries, & motion outputs.
    """

    def __init__(self, out_folder: str | Path | None):
        self.out_folder = Path(out_folder) if out_folder else None          # store a normalized output folder or None when writing is disabled

    def write_rows(
        self,
        filename: str,
        headers: list[str],
        rows: list[dict[str, Any]],
    ) -> Path | None:
        """
        Write a list of row dictionaries to CSV using a fixed header order.
        Returns the written file path, or None when nothing should be written.
        """
        if not self.out_folder or not rows:
            return None                                                                 # skip writes when no destination exists or there is no data to save

        self.out_folder.mkdir(parents=True, exist_ok=True)                              # ensure the destination folder exists before opening the file
        out_path = self.out_folder / filename                                           # resolve the final CSV output path

        with open(out_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)                         # enforce a stable column order instead of relying on dict insertion order
            writer.writeheader()
            for row in rows:
                writer.writerow({header: row.get(header, "") for header in headers})    # fill missing keys with blanks so every row stays aligned

        return out_path                                                                 # return the written file path so callers can collect saved artifacts

# --------- INTERVAL CLOCK ---------
class IntervalClock:
    """
    Grid-aligned interval clock.
    All consumers map timestamps to:
        anchor + k * interval_sec
    This gives every measurement system the same interval boundaries for a given source.
    """
    def __init__(self, interval_sec: float, anchor_ts: datetime | None = None):
        self.interval_sec = float(interval_sec)                                         # store the interval size as float so fractional intervals remain valid
        self.anchor_ts = _to_utc(anchor_ts) if anchor_ts else None                      # optional fixed anchor keeps all later interval starts aligned

    def interval_start(self, ts: datetime) -> datetime:
        """
        Return the aligned interval start for a timestamp.
        The first timestamp becomes the anchor when no anchor has been set yet.
        """
        ts = _to_utc(ts)                                                                # normalize before computing elapsed seconds from the anchor

        if self.anchor_ts is None:
            self.anchor_ts = ts                                                         # first seen timestamp becomes the interval anchor
            return ts

        delta_sec = (ts - self.anchor_ts).total_seconds()                               # compute offset from the anchor in seconds
        step = int(delta_sec // self.interval_sec)                                      # floor to the current interval bucket
        return self.anchor_ts + timedelta(seconds=step * self.interval_sec)             # reconstruct the aligned interval start from the anchor

    def tick(self, ts: datetime, current_start: datetime | None):
        """
        Compare a timestamp against the current interval state.

        Returns:
            rolled, new_start, old_start
        """
        ts = _to_utc(ts)                                                        # normalize incoming timestamp before any interval logic
        current_start = _to_utc(current_start) if current_start else None       # normalize current interval boundary when present

        new_start = self.interval_start(ts)                                     # compute which interval the current timestamp belongs to
        if current_start is None:
            return False, new_start, None                                       # first timestamp initializes the interval without a rollover

        if new_start != current_start:
            return True, new_start, current_start                               # interval boundary changed, so caller should finalize the previous bucket

        return False, current_start, None                                       # still inside the same interval, so nothing rolls over yet

# --------- COUNT HELPERS ---------
def _normalize_class_lists(class_config=None, focus_classes=None, context_classes=None):
    """
    Normalize class inputs so every measurement class can share the same logic.
    Explicit lists are accepted, but class_config overrides them when provided.
    """
    focus = list(focus_classes or [])                                  # begin with direct focus class input when supplied
    context = list(context_classes or [])                              # begin with direct context class input when supplied

    if class_config is not None:
        focus = list(getattr(class_config, "focus", []))               # prefer the centralized class configuration when available
        context = list(getattr(class_config, "context", []))

    return focus, context                                              # return normalized plain lists for consistent downstream use

def measurement_columns(focus_classes: list[str], context_classes: list[str], include_ratio: bool = True) -> list[str]:
    """
    Return the standard measurement column order.
    Focus classes always come first, followed by OBJECTS & optional RATIO when context classes are enabled.
    """
    columns = list(focus_classes)                                      # preserve caller-provided focus class order in all CSV exports
    if context_classes:
        columns.append("OBJECTS")                                      # combined context-object count is exported as a single column
        if include_ratio:
            columns.append("RATIO")                                    # append readable class ratio only when requested
    return columns

def add_ratio_to_counts(
    counts: dict[str, Any],
    focus_classes: list[str],
    context_classes: list[str],
    *,
    allow_float: bool = False,
) -> dict[str, Any]:
    """
    Add a readable ratio string for focus classes when context classes are enabled.
    Float averages are scaled before reduction so ratios remain readable after averaging.
    """
    if not context_classes:
        return counts                                                  # ratios only make sense in the focus-vs-context workflow

    values: list[int] = []
    for cls in focus_classes:
        value = counts.get(cls, 0)                                     # fetch count for each focus class in the configured display order

        if allow_float:
            try:
                value = float(value)                                   # averages may come in as floats during aggregation
            except Exception:
                value = 0.0
            value = int(round(value * 10.0))                           # scale floats so small non-integer averages still influence the ratio
        else:
            try:
                value = int(value)                                     # normal count tables should reduce directly as integers
            except Exception:
                value = 0

        values.append(value)                                           # preserve ordered ratio components in the same class order

    non_zero = [v for v in values if v != 0]                           # ignore zeros when computing gcd so empty classes do not collapse the ratio
    if len(non_zero) > 1:
        gcd_value = abs(non_zero[0])                                   # seed gcd reduction from the first non-zero value
        for value in non_zero[1:]:
            gcd_value = math.gcd(gcd_value, abs(value))                # reduce all non-zero components to the smallest readable whole-number ratio
        if gcd_value > 1:
            values = [int(v // gcd_value) for v in values]

    counts["RATIO"] = ":".join(str(v) for v in values)                 # store the final readable ratio string directly in the counts dict
    return counts

def compute_counts_from_boxes(
    boxes_list,
    names,
    focus_classes=None,
    context_classes=None,
):
    """
    Count detections from YOLO-format boxes:
        [x1, y1, x2, y2, conf, cls_id, ...optional obb corners...]
    Returns a frame-level count dictionary matching the configured focus/context class layout.
    """
    focus_classes = list(focus_classes or [])                          # preserve focus class order for the resulting count mapping
    context_classes = set(context_classes or [])                       # use a set for fast context-membership checks during counting

    counts = {cls: 0 for cls in focus_classes}                         # initialize all focus counts at zero so outputs stay structurally consistent
    if context_classes:
        counts["OBJECTS"] = 0                                          # combined context-object count is tracked only when context classes exist

    for box in boxes_list:
        if not box or len(box) < 6:
            continue                                                   # skip malformed detections that do not include class information

        cls_id = box[5]                                                # YOLO-format detections store class id at index 5
        cls_name = names.get(cls_id)                                   # resolve class id to its human-readable class label

        if cls_name in counts:
            counts[cls_name] += 1                                      # increment exact focus-class matches directly
        elif context_classes and cls_name in context_classes:
            counts["OBJECTS"] += 1                                     # collapse all context detections into one combined object count

    return add_ratio_to_counts(counts, focus_classes, list(context_classes))  # attach ratio string when context classes are enabled

def build_context_anchors(boxes, names, context_classes):
    """
    Build one stable anchor box per context class from the current frame.

    Simplest rule:
        - collect all detections for each context class
        - keep the largest box for that class

    This works well for static environmental objects like feeders, perches, or boxes.
    """
    context_set = set(context_classes or [])                           # normalize context classes to a set for fast membership checks
    anchors: dict[str, list[float | int]] = {}                         # stores one chosen anchor box per context class

    for box in boxes:
        if not box or len(box) < 6:
            continue                                                   # ignore malformed detections that cannot be classified reliably

        cls_name = names.get(box[5])                                   # resolve the detection class label from the class id
        if cls_name not in context_set:
            continue                                                   # only context-class detections are eligible to become anchors

        if cls_name not in anchors:
            anchors[cls_name] = box                                    # first seen box for a context class becomes the provisional anchor
            continue

        old_area = _box_area(anchors[cls_name])                        # compare current anchor size against the new candidate size
        new_area = _box_area(box)
        if new_area > old_area:
            anchors[cls_name] = box                                    # prefer the largest visible anchor for more stable overlap checks

    return anchors                                                     # return one anchor box per context class present in the frame

def _box_area(box) -> float:
    """
    Compute axis-aligned box area from the first four coordinates.
    This helper is used when selecting the largest anchor candidate.
    """
    x1, y1, x2, y2 = box[:4]                                           # only the first four coordinates are needed for AABB area
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)                       # clamp width & height to zero so malformed boxes do not produce negative areas

# --------- TRACKER ---------
class Tracker:
    """
    Lightweight measurement tracker.

    Design goals:
        - stay class-aware
        - preserve a center-distance gate
        - prefer detections with better IoU
        - blend IoU, center distance, & box-size similarity into one match score
    """

    def __init__(self, config=None, focus_classes=None, context_classes=None):
        self.config = config or MeasurementConfig()                                 # use shared measurement config unless one was provided explicitly
        self.focus_classes = set(focus_classes or [])                               # tracked focus classes eligible for downstream biological measurements
        self.context_classes = set(context_classes or [])                           # optional tracked context classes such as perches or feeders

        self.max_dist = float(getattr(self.config, "track_max_dist_px", 60.0))      # hard distance gate for track/detection association
        self.max_age = int(getattr(self.config, "track_max_age_frames", 10))        # stale tracks older than this are removed

        self.tracks = defaultdict(dict)                                             # stores active tracks as tracks[class_name][track_id] = state dict
        self._next_id = defaultdict(int)                                            # stores the next integer id to assign for each class independently

    def reset(self):
        """
        Clear all active tracking state.
        Useful when starting a fresh sequence or when a full tracker reset is needed.
        """
        self.tracks.clear()                                            # remove all active per-class track records
        self._next_id.clear()                                          # reset id counters so new sessions start from zero again

    @staticmethod
    def _center(box):
        """
        Return the center point of an axis-aligned box.
        """
        x1, y1, x2, y2 = box[:4]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)                      # center-based tracking uses midpoint rather than top-left coordinates

    @staticmethod
    def _size(box):
        """
        Return width & height for an axis-aligned box.
        """
        x1, y1, x2, y2 = box[:4]
        return max(0.0, x2 - x1), max(0.0, y2 - y1)                    # clamp to zero so malformed boxes do not produce negative dimensions

    @staticmethod
    def _area(box):
        """
        Return box area for an axis-aligned box.
        """
        w, h = Tracker._size(box)                                      # reuse the shared width/height helper for consistency
        return w * h

    @staticmethod
    def _dist(a, b):
        """
        Return Euclidean distance between two center points.
        """
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5        # direct center-to-center distance is the basic spatial association signal

    @staticmethod
    def _aabb_iou(box_a, box_b) -> float:
        """
        Return axis-aligned IoU between two boxes.
        """
        ax1, ay1, ax2, ay2 = box_a[:4]
        bx1, by1, bx2, by2 = box_b[:4]

        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))              # overlap width between the two boxes
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))              # overlap height between the two boxes
        inter = inter_w * inter_h                                      # raw overlapping area

        if inter <= 0:
            return 0.0                                                 # no overlap means IoU is zero immediately

        area_a = Tracker._area(box_a)
        area_b = Tracker._area(box_b)
        union = area_a + area_b - inter                                # standard IoU union area

        if union <= 0:
            return 0.0                                                 # guard against malformed zero-area boxes

        return inter / union

    @staticmethod
    def _size_similarity(box_a, box_b) -> float:
        """
        Return relative box-size similarity between two detections.
        Larger values indicate more similar area.
        """
        area_a = Tracker._area(box_a)
        area_b = Tracker._area(box_b)

        if area_a <= 0 or area_b <= 0:
            return 0.0                                                 # invalid or zero-area boxes cannot provide meaningful size similarity

        return min(area_a, area_b) / max(area_a, area_b)               # ratio stays within 0..1 where 1 means identical area

    def _new_id(self, cls_name: str) -> int:
        """
        Allocate the next track id for a class.
        Class-specific counters keep ids simple while avoiding cross-class collisions.
        """
        self._next_id[cls_name] += 1                                   # advance the per-class id counter
        return self._next_id[cls_name] - 1                             # return the previous value so ids begin at zero

    def _match_score(self, track: dict[str, Any], detection: dict[str, Any]) -> float:
        """
        Compute a combined association score for one track/detection pair.
        Higher is better.

        Score combines:
            - IoU as the strongest same-object signal
            - normalized center-distance score
            - box-size similarity
        """
        distance = self._dist(track["center"], detection["center"])             # compute center distance between previous track state & new detection
        if distance > self.max_dist:
            return -1.0                                                         # hard reject detections that move too far to plausibly be the same object

        iou = self._aabb_iou(track["box"], detection["box"])                    # strong overlap signal across consecutive frames
        size_sim = self._size_similarity(track["box"], detection["box"])        # penalize drastic box-size changes
        dist_score = max(0.0, 1.0 - (distance / max(self.max_dist, 1e-6)))      # convert distance into a bounded 0..1 proximity score

        return (0.60 * iou) + (0.25 * dist_score) + (0.15 * size_sim)           # weighted blend tuned to favor overlap most strongly

    def _associate(self, cls_name: str, detections: list[dict[str, Any]]):
        """
        Greedy bipartite-style matching for one class.

        Steps:
            - compute all viable track/detection pairs
            - sort by descending score
            - assign best non-conflicting pairs first
        """
        candidates: list[tuple[float, int, int]] = []                  # each row is (score, track_id, detection_index)

        track_items = list(self.tracks[cls_name].items())              # snapshot current tracks so matching operates on a stable list
        for track_id, track in track_items:
            for det_index, detection in enumerate(detections):
                score = self._match_score(track, detection)            # compute association quality for every track/detection pair
                if score >= 0:
                    candidates.append((score, track_id, det_index))    # keep only viable candidates that passed the distance gate

        candidates.sort(reverse=True, key=lambda row: row[0])          # match highest-scoring pairs first in greedy order

        assigned: dict[int, int] = {}                                  # maps track_id -> detection_index for accepted matches
        used_tracks: set[int] = set()                                  # prevents one track from being matched multiple times
        used_detections: set[int] = set()                              # prevents one detection from being reused

        for score, track_id, det_index in candidates:
            if track_id in used_tracks or det_index in used_detections:
                continue                                               # skip any candidate that conflicts with an earlier higher-scoring assignment
            assigned[track_id] = det_index
            used_tracks.add(track_id)
            used_detections.add(det_index)

        return assigned, used_detections                               # caller updates matched tracks & spawns new tracks for unmatched detections

    def update(self, boxes, names, allow_classes=None):
        """
        Update tracker with current-frame detections.

        Returns:
            dict[class_name, list[track_dict]]
        """
        allowed = set(allow_classes) if allow_classes is not None else None  # optional allowlist further limits which detections are tracked

        detections_by_class = defaultdict(list)                             # group current detections by class before association
        for box in boxes:
            if not box or len(box) < 6:
                continue                                                    # skip malformed detections that do not include class id info

            cls_name = names.get(box[5])                                    # resolve class name from the detection class id
            if not cls_name:
                continue                                                    # skip detections whose class id cannot be resolved

            if allowed is not None and cls_name not in allowed:
                continue                                                    # optional caller-level allowlist excludes irrelevant classes early

            if (self.focus_classes or self.context_classes) and (
                cls_name not in self.focus_classes and cls_name not in self.context_classes
            ):
                continue                                                    # tracker-level focus/context filter excludes classes outside measurement scope

            detections_by_class[cls_name].append(
                {
                    "center": self._center(box),                            # precompute center once so later scoring is cheaper
                    "box": box,                                             # retain original detection box for overlap, motion, & export logic
                }
            )

        touched = defaultdict(set)                                          # records tracks updated this frame so untouched tracks can be aged later

        for cls_name, detections in detections_by_class.items():
            assigned, used_detections = self._associate(cls_name, detections)   # match current detections against existing tracks of the same class

            for track_id, det_index in assigned.items():
                detection = detections[det_index]
                track = self.tracks[cls_name].get(track_id)
                if track is None:
                    continue                                                    # guard against unlikely desync if a track disappeared mid-update

                track["center"] = detection["center"]                           # refresh track position to the current matched detection
                track["box"] = detection["box"]                                 # refresh stored box for overlap/motion calculations
                track["age"] = 0                                                # matched tracks are considered fresh again this frame
                touched[cls_name].add(track_id)

            for det_index, detection in enumerate(detections):
                if det_index in used_detections:
                    continue                                                    # only unmatched detections should create brand-new tracks

                track_id = self._new_id(cls_name)                               # allocate a new class-local id for this previously unseen object
                self.tracks[cls_name][track_id] = {
                    "center": detection["center"],
                    "box": detection["box"],
                    "age": 0,
                }
                touched[cls_name].add(track_id)

        for cls_name in list(self.tracks.keys()):
            for track_id in list(self.tracks[cls_name].keys()):
                track = self.tracks[cls_name][track_id]
                if track_id not in touched.get(cls_name, set()):
                    track["age"] += 1                                           # unmatched tracks age by one frame when not refreshed this update
                if track["age"] > self.max_age:
                    del self.tracks[cls_name][track_id]                         # retire stale tracks once they have been missing too long

        stable_tracks = {}
        for cls_name, class_tracks in self.tracks.items():
            stable_tracks[cls_name] = [
                {
                    "id": track_id,                                    # stable per-class track id used by motion & interactions
                    "cls": cls_name,                                   # class label kept alongside each track for downstream convenience
                    "center": track["center"],                         # most recent center position
                    "box": track["box"],                               # most recent box geometry
                    "age": track["age"],                               # freshness indicator where age==0 means current-frame detection
                }
                for track_id, track in class_tracks.items()
            ]

        return stable_tracks                                           # return a normalized track structure for all downstream measurement consumers

# --------- COUNTER ---------
class Counter:
    """
    Snapshot counts at interval rollovers plus grouped averages.
    This class records tracker-based count snapshots & exports both raw interval counts & averaged grouped counts.
    """
    def __init__(
        self,
        out_folder=None,
        config=None,
        start_time=None,
        class_config=None,
        focus_classes=None,
        context_classes=None,
        tracker=None,
    ):
        self.out_folder = Path(out_folder) if out_folder else None     # optional destination folder for exported count tables
        self.writer = CSVTableWriter(self.out_folder)                  # shared CSV helper used by both snapshot & average exports
        self.config = config or MeasurementConfig()                    # shared measurement config for interval lengths & grouping size
        self.start_time = _to_utc(start_time) if start_time else None  # normalize starting timestamp for consistent interval anchoring

        self.focus_classes, self.context_classes = _normalize_class_lists(
            class_config=class_config,
            focus_classes=focus_classes,
            context_classes=context_classes,
        )                                                                                   # resolve which classes should contribute to counting outputs

        self.clock = IntervalClock(self.config.interval_sec, anchor_ts=self.start_time)     # align snapshots to the global interval grid
        self._interval_start = None                                                         # tracks the currently active interval boundary
        self.snapshot_buffer: list[tuple[datetime, dict[str, Any]]] = []                    # stores timestamped count snapshots taken at interval rollovers
        self.group_number = 1                                                               # running group id used when exporting averaged count blocks

    def _count_recent_tracks(self, tracks_by_class: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        """
        Count only fresh tracks from the current frame.
        Age==0 tracks represent detections actually seen in the current interval boundary frame.
        """
        counts = {cls: 0 for cls in self.focus_classes}                                             # initialize focus-class counts so output columns stay stable

        for cls in self.focus_classes:
            counts[cls] = sum(1 for track in tracks_by_class.get(cls, []) if track["age"] == 0)     # count only active current-frame tracks

        if self.context_classes:
            counts["OBJECTS"] = 0                                                                   # collapse all current context detections into one combined object count
            for cls in self.context_classes:
                counts["OBJECTS"] += sum(1 for track in tracks_by_class.get(cls, []) if track["age"] == 0)

        return add_ratio_to_counts(counts, self.focus_classes, self.context_classes)                # attach ratio when context classes are part of the analysis

    def update_counts(self, tracks_by_class, timestamp=None):
        """
        Capture a new tracker-based snapshot when a fresh interval begins.
        Only one snapshot is stored per interval boundary.
        """
        now = _to_utc(timestamp) if timestamp else _to_utc(datetime.now())                              # use provided frame timestamp or fall back to current wall time
        interval_start = self.clock.interval_start(now)                                                 # align the timestamp to the shared interval grid

        if self._interval_start is None:
            self._interval_start = interval_start                                                       # initialize the first active interval
            self.snapshot_buffer.append((interval_start, self._count_recent_tracks(tracks_by_class)))   # store the first snapshot immediately
            return

        if interval_start != self._interval_start:
            self._interval_start = interval_start                                                       # advance to the newly entered interval boundary
            self.snapshot_buffer.append((interval_start, self._count_recent_tracks(tracks_by_class)))   # record one new snapshot for the new interval

    def _compute_averages(self):
        """
        Compute grouped average counts across snapshot blocks.
        Group size is controlled by avg_group_size in the measurement config.
        """
        if not self.snapshot_buffer:
            return []                                                      # no snapshots means there is nothing to average

        group_size = max(int(self.config.avg_group_size or 1), 1)          # guard against invalid zero or negative grouping sizes
        rows = []

        for start in range(0, len(self.snapshot_buffer), group_size):
            block = self.snapshot_buffer[start : start + group_size]       # take one contiguous block of snapshots for averaging
            summed = defaultdict(float)                                    # accumulate numeric values across the block

            for _, counts in block:
                for key, value in counts.items():
                    if key == "RATIO":
                        continue                                           # ratio is recomputed from averages rather than averaged directly as a string
                    summed[key] += float(value)

            divisor = len(block)                                           # use actual block length so the final partial block still averages correctly
            avg_counts = {key: summed[key] / divisor for key in summed}
            avg_counts = add_ratio_to_counts(
                avg_counts,
                self.focus_classes,
                self.context_classes,
                allow_float=True,                                          # allow float-aware ratio formatting for averaged values
            )

            midpoint = block[0][0] + (block[-1][0] - block[0][0]) / 2      # place the averaged row at the midpoint of the grouped snapshot window
            row = {
                "GROUP": self.group_number,
                "TIME": _fmt_time(midpoint),
                **avg_counts,
            }
            rows.append(row)
            self.group_number += 1                                         # increment exported group number for the next averaged block
        return rows

    def save_results(self):
        """
        Save raw snapshot counts & grouped average counts.
        Returns a list of written file paths, or None when no output is available.
        """
        if not self.out_folder or not self.snapshot_buffer:
            return None                                                                                 # skip saving when no destination exists or no snapshots were collected

        columns = measurement_columns(self.focus_classes, self.context_classes)                         # compute the standard count-column layout once

        snapshot_rows = [
            {"TIME": _fmt_time(timestamp), **counts}
            for timestamp, counts in self.snapshot_buffer
        ]                                                                                               # convert buffered snapshots into CSV-ready rows

        averages = self._compute_averages()                                                             # compute grouped averages from the stored snapshot history

        saved = []
        snap_file = self.writer.write_rows("counts.csv", ["TIME", *columns], snapshot_rows)             # write raw interval snapshots first
        if snap_file:
            saved.append(snap_file)

        avg_file = self.writer.write_rows("average_counts.csv", ["GROUP", "TIME", *columns], averages)  # then write grouped average rows
        if avg_file:
            saved.append(avg_file)
        return saved

# --------- INTERACTIONS ---------
class Interactions:
    """
    Detect sustained overlaps between tracked entities.
    Focus-vs-context mode measures overlaps between moving focus tracks & static context anchors,
    while focus-only mode measures overlaps between focus tracks themselves.
    """
    def __init__(
        self,
        out_folder=None,
        config=None,
        start_time=None,
        is_obb=False,
        class_config=None,
        focus_classes=None,
        context_classes=None,
        tracker=None,
    ):
        self.out_folder = Path(out_folder) if out_folder else None     # optional output folder for interaction CSV export
        self.writer = CSVTableWriter(self.out_folder)                  # shared CSV writer for interaction records
        self.config = config or MeasurementConfig()                    # measurement config controls thresholds, timeouts, & age limits
        self.start_time = _to_utc(start_time) if start_time else None  # normalized session start used for exported interaction timing
        self.is_obb = bool(is_obb)                                     # toggles polygon-based overlap scoring for oriented boxes

        self.focus_classes, self.context_classes = _normalize_class_lists(
            class_config=class_config,
            focus_classes=focus_classes,
            context_classes=context_classes,
        )                                                                                       # resolve which classes should participate in interaction logic

        self.max_track_age = int(getattr(self.config, "interaction_max_track_age_frames", 2))   # stale tracks above this age are ignored in overlap checks

        self.active: dict[tuple, dict[str, Any]] = {}                                           # active overlapping pairs keyed by pair identity
        self.records: list[dict[str, Any]] = []                                                 # finalized interaction rows ready for export
        self.clock = IntervalClock(self.config.interval_sec, anchor_ts=self.start_time)         # shared interval alignment for video-time conversion
        self.ref_time = None                                                                    # remembers the first aligned interval boundary seen during processing

    def _video_time(self, ts: datetime) -> datetime:
        """
        Convert a processing timestamp back into source-relative video time.
        This keeps exported interaction times aligned to the original source timeline.
        """
        ts = _to_utc(ts)                                               # normalize timestamp before any offset calculation

        if not self.start_time:
            return ts                                                  # without a known start time, just return the normalized timestamp directly

        if self.ref_time is None:
            self.ref_time = self.clock.interval_start(ts)              # anchor the first seen processing timestamp to the interval grid

        delta_sec = (ts - self.ref_time).total_seconds()               # compute elapsed seconds from the first interval-aligned reference point
        return self.start_time + timedelta(seconds=delta_sec)          # map elapsed processing time back onto the original source start time

    @staticmethod
    def _aabb_area(box) -> float:
        """
        Return area for an axis-aligned box.
        """
        x1, y1, x2, y2 = box[:4]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)                   # clamp width & height to zero so invalid boxes do not produce negative area

    def _polygon_points(self, box):
        """
        Extract polygon corner points from an OBB box when available, otherwise fall back to an axis-aligned rectangle.
        """
        if len(box) >= 14:
            return [
                (box[6], box[7]),
                (box[8], box[9]),
                (box[10], box[11]),
                (box[12], box[13]),
            ]                                                           # OBB detections append four corner points after the basic YOLO fields
        x1, y1, x2, y2 = box[:4]
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]                 # AABB fallback lets the same polygon pipeline work for normal boxes too

    def _polygon_from_box(self, box):
        """
        Build a valid Shapely polygon from a detection box.
        Invalid polygons are repaired when possible using buffer(0).
        """
        poly = Polygon(self._polygon_points(box))                      # build polygon geometry from the detection corner points
        if not poly.is_valid:
            poly = poly.buffer(0)                                      # common geometry repair trick for self-intersections or invalid winding
        if poly.is_empty or not poly.is_valid:
            return None                                                # discard polygons that still cannot be used safely
        return poly

    def _aabb_intersection_metrics(self, a, b):
        """
        Return intersection area & individual areas using axis-aligned boxes.
        """
        ax1, ay1, ax2, ay2 = a[:4]
        bx1, by1, bx2, by2 = b[:4]

        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))             # overlap width between the two boxes
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))             # overlap height between the two boxes
        inter = inter_w * inter_h                                     # axis-aligned intersection area

        area_a = self._aabb_area(a)
        area_b = self._aabb_area(b)

        return inter, area_a, area_b                                  # caller decides whether to use IoU, containment, or both

    def _polygon_intersection_metrics(self, a, b):
        """
        Return intersection area & polygon areas using polygon geometry.
        Used when oriented boxes are available for both entities.
        """
        poly_a = self._polygon_from_box(a)
        poly_b = self._polygon_from_box(b)

        if poly_a is None or poly_b is None:
            return 0.0, 0.0, 0.0                                      # invalid polygon geometry cannot contribute to overlap scoring

        inter = poly_a.intersection(poly_b).area                      # true polygon overlap area for OBB-aware interaction scoring
        if inter <= 0:
            return 0.0, poly_a.area, poly_b.area                      # preserve individual areas even when there is no overlap

        return inter, poly_a.area, poly_b.area

    def _overlap_score(self, a, b) -> float:
        """
        Compute overlap score between two boxes.
        Score is the max of IoU & containment so both partial overlap & near-full enclosure can count as interaction.
        """
        if self.is_obb and len(a) >= 14 and len(b) >= 14:
            inter, area_a, area_b = self._polygon_intersection_metrics(a, b)            # use polygon-aware overlap for true oriented boxes
        else:
            inter, area_a, area_b = self._aabb_intersection_metrics(a, b)               # otherwise fall back to axis-aligned overlap

        if inter <= 0 or area_a <= 0 or area_b <= 0:
            return 0.0                                                                  # invalid or non-overlapping geometry produces no interaction score

        union = area_a + area_b - inter                                                 # union area for standard IoU computation
        iou = inter / union if union > 0 else 0.0
        containment = inter / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0   # containment rewards one object sitting inside another

        return max(iou, containment)                                                    # allow either overlap style to trigger an interaction

    def _tracks_for_interaction(self, tracks_by_class, classes):
        """
        Filter tracks down to those recent enough to participate in interaction checks.
        """
        rows = []
        for cls in classes:
            for track in tracks_by_class.get(cls, []):
                if track["age"] <= self.max_track_age:
                    rows.append(track)                                   # include fresh or very recently seen tracks only
        return rows

    def _activate(self, pair_key, pair_labels, ts):
        """
        Start or refresh an active interaction pair.
        """
        if pair_key not in self.active:
            self.active[pair_key] = {
                "start": ts,                                             # first timestamp when the overlap became active
                "last": ts,                                              # most recent timestamp the overlap was observed
                "labels": pair_labels,                                   # human-readable class labels used when recording the interaction
            }
        else:
            self.active[pair_key]["last"] = ts                           # refresh the last-seen time for an already active overlap

    def _finalize_inactive(self, active_now, ts):
        """
        Finalize interactions that are no longer active & have exceeded the timeout threshold.
        """
        timeout = float(self.config.interaction_timeout_sec)                # tolerated gap before an inactive interaction is considered finished
        ended = []

        for pair_key, info in list(self.active.items()):
            if pair_key in active_now:
                continue                                                    # still active this frame, so do not finalize it yet

            if (ts - info["last"]).total_seconds() >= timeout:
                self._record(info["labels"], info["start"], info["last"])   # record the completed interaction once the inactivity gap is long enough
                ended.append(pair_key)

        for pair_key in ended:
            del self.active[pair_key]                                       # remove finalized interactions from the active state map

    def _record(self, labels, start, end):
        """
        Convert an interaction span into an exportable row when it meets the minimum duration requirement.
        """
        duration = round((end - start).total_seconds(), 2)                                  # compute interaction duration in seconds with readable precision
        min_duration = float(getattr(self.config, "interaction_min_duration_sec", 0.75))

        if duration <= 0 or duration < min_duration:
            return                                                                          # ignore zero-length or very brief overlaps that do not meet the threshold

        if self.context_classes:
            row = {
                "TIME0": _fmt_time(start),
                "TIME1": _fmt_time(end),
                "FOCUS": labels[0],
                "CONTEXT": labels[1],
                "DURATION": duration,
            }                                                                               # focus-vs-context export layout
        else:
            row = {
                "TIME0": _fmt_time(start),
                "TIME1": _fmt_time(end),
                "CLASS1": labels[0],
                "CLASS2": labels[1],
                "DURATION": duration,
            }                                                                               # focus-vs-focus export layout

        self.records.append(row)                                                            # store the completed interaction row for later CSV export

    def process_tracks(self, tracks_by_class, boxes, names, ts):
        """
        Process one frame of tracks & detections for interaction updates.
        Active interactions are refreshed, newly overlapping pairs are activated, & timed-out inactive pairs are finalized.
        """
        ts = _to_utc(ts)
        video_ts = self._video_time(ts)                                                     # convert processing time to source-relative video time for exported records

        focus_tracks = self._tracks_for_interaction(tracks_by_class, self.focus_classes)    # collect recent focus tracks eligible for overlap checks
        active_now = set()                                                                  # tracks which interaction keys were seen active in this frame
        threshold = float(self.config.overlap_threshold)                                    # minimum overlap score required to count as an interaction

        if self.context_classes:
            context_anchors = build_context_anchors(
                boxes=boxes,
                names=names,
                context_classes=self.context_classes,
            )                                                                               # build one stable anchor box per context object class for this frame

            for focus_track in focus_tracks:
                for context_cls, context_box in context_anchors.items():
                    score = self._overlap_score(focus_track["box"], context_box)    # compare each focus track against each available context anchor
                    if score < threshold:
                        continue                                                    # ignore weak overlaps below the configured interaction threshold

                    pair_key = (
                        focus_track["cls"],
                        focus_track["id"],
                        context_cls,
                    )                                                               # include focus class/id plus context class so the pair remains stable across frames
                    pair_labels = (focus_track["cls"], context_cls)

                    active_now.add(pair_key)
                    self._activate(pair_key, pair_labels, video_ts)

        else:
            for i, track_a in enumerate(focus_tracks):
                for j in range(i + 1, len(focus_tracks)):
                    track_b = focus_tracks[j]                                       # compare each focus track against later tracks only to avoid duplicate pairs

                    if track_a["id"] == track_b["id"] and track_a["cls"] == track_b["cls"]:
                        continue                                                    # skip the impossible self-pair case just to be safe

                    score = self._overlap_score(track_a["box"], track_b["box"])
                    if score < threshold:
                        continue                                                    # ignore weak overlaps below threshold

                    pair_key = tuple(
                        sorted(
                            [
                                (track_a["cls"], track_a["id"]),
                                (track_b["cls"], track_b["id"]),
                            ]
                        )
                    )                                                               # sort pair identity so A/B & B/A collapse to the same key
                    pair_labels = tuple(sorted((track_a["cls"], track_b["cls"])))   # sorted labels keep output ordering stable too

                    active_now.add(pair_key)
                    self._activate(pair_key, pair_labels, video_ts)

        self._finalize_inactive(active_now, video_ts)                               # close any interaction pairs that are no longer active & have timed out

    def finalize(self):
        """
        Finalize any still-active interactions at shutdown.
        Returns the full record list for convenience.
        """
        for _, info in list(self.active.items()):
            self._record(info["labels"], info["start"], info["last"])           # flush any still-open interactions using their last observed timestamp
        self.active.clear()                                                     # clear runtime state now that all remaining interactions have been recorded
        return self.records

    def save_results(self):
        """
        Save finalized interaction records to CSV.
        Returns the written file path, or None when no records are available.
        """
        if not self.records or not self.out_folder:
            return None                                                        # skip writing when there is no destination or no interaction data

        headers = (
            ["TIME0", "TIME1", "FOCUS", "CONTEXT", "DURATION"]
            if self.context_classes
            else ["TIME0", "TIME1", "CLASS1", "CLASS2", "DURATION"]
        )                                                                      # choose export schema based on whether context classes are enabled

        rows = sorted(self.records, key=lambda row: row["TIME0"])              # sort chronologically by interaction start time for readability
        return self.writer.write_rows("interactions.csv", headers, rows)

# --------- AGGREGATOR ---------
class Aggregator:
    """
    Frame-level accumulation -> interval rollups -> session summary.
    Interval values are averaged per frame rather than summed raw across unequal frame counts.
    """
    def __init__(self, out_folder, config=None, start_time=None, class_config=None):
        self.out_folder = Path(out_folder)                                 # aggregation always expects a valid output folder
        self.writer = CSVTableWriter(self.out_folder)                      # shared CSV writer for interval & session summary outputs
        self.config = config or MeasurementConfig()                        # shared measurement configuration object
        self.start_time = _to_utc(start_time) if start_time else None      # normalized source start time for aligned interval anchoring

        if class_config is None:
            raise ValueError("class_config is required.")                  # aggregation needs explicit class structure to produce stable tables

        self.focus_classes, self.context_classes = _normalize_class_lists(class_config=class_config)  # resolve class lists directly from class config

        self.clock = IntervalClock(self.config.interval_sec, anchor_ts=self.start_time)  # shared interval grid for frame accumulation
        self._interval_start = None                                       # start time for the currently accumulating interval
        self._interval_sums = defaultdict(float)                          # running numeric sums within the current interval
        self._interval_frames = 0                                         # number of frames accumulated into the current interval
        self.intervals: list[dict[str, Any]] = []                         # finalized interval-average rows ready for export

    def push_frame_data(self, timestamp, current_boxes_list=None, names=None, counts_dict=None):
        """
        Add one frame of counts into the current interval bucket.
        Counts may be provided directly or derived from raw detections.
        """
        timestamp = _to_utc(timestamp)                                    # normalize frame timestamp before interval alignment

        if counts_dict is None and current_boxes_list is not None and names is not None:
            counts_dict = compute_counts_from_boxes(
                current_boxes_list,
                names,
                focus_classes=self.focus_classes,
                context_classes=self.context_classes,
            )                                                              # derive per-frame counts on the fly when raw detections were provided instead

        counts_dict = dict(counts_dict or {})                              # normalize missing count input to an empty dict
        counts_dict.pop("RATIO", None)                                     # ratio is recomputed later from interval averages, not accumulated directly

        rolled, new_start, old_start = self.clock.tick(timestamp, self._interval_start)     # check whether this frame entered a new interval
        if self._interval_start is None:
            self._interval_start = new_start                                                # initialize the current interval on the first frame

        if rolled and old_start is not None:
            self.intervals.append(
                self._finalize_interval(old_start, dict(self._interval_sums), self._interval_frames)
            )                                                              # finalize the previous interval before starting the next one
            self._interval_sums.clear()                                    # reset accumulation state for the new interval
            self._interval_frames = 0
            self._interval_start = new_start

        for cls, value in counts_dict.items():
            self._interval_sums[cls] += float(value)                       # accumulate per-class counts across frames in the active interval

        self._interval_frames += 1                                         # count this frame toward the interval average denominator

    def _finalize_interval(self, start_ts, summed: dict[str, float], frames: int):
        """
        Convert one accumulated interval into an averaged export row.
        """
        frames = max(int(frames or 0), 1)                                                   # guard against divide-by-zero in case of malformed interval state
        averages = {cls: float(value) / frames for cls, value in (summed or {}).items()}    # convert summed counts to per-frame averages
        averages = add_ratio_to_counts(
            averages,
            self.focus_classes,
            self.context_classes,
            allow_float=True,                                                               # averaged values need float-aware ratio formatting
        )

        midpoint = _to_utc(start_ts) + timedelta(seconds=float(self.config.interval_sec) / 2.0)  # place the exported interval row at the interval midpoint
        return {"TIME": _fmt_time(midpoint), **averages}

    def finalize(self):
        """
        Flush the currently accumulating interval into the finalized interval list.
        Safe to call multiple times.
        """
        if self._interval_frames > 0 and self._interval_start is not None:
            self.intervals.append(
                self._finalize_interval(self._interval_start, dict(self._interval_sums), self._interval_frames)
            )                                                              # finalize the last partially open interval at shutdown/export time
            self._interval_sums.clear()
            self._interval_frames = 0

    def save_interval_results(self):
        """
        Save per-interval averaged frame counts.
        Returns the written file path, or None when no intervals exist.
        """
        self.finalize()                                                    # ensure the current open interval is flushed before export
        if not self.intervals:
            return None

        headers = ["TIME", *measurement_columns(self.focus_classes, self.context_classes)]  # use the standard measurement column order
        return self.writer.write_rows("frame_counts.csv", headers, self.intervals)

    def save_session_summary(self):
        """
        Save session-level summary statistics computed across blocks of intervals.
        Summary includes total count, mean rate, rate standard deviation, & focus-class proportion.
        """
        self.finalize()                                                    # ensure interval data is complete before building session summaries
        if not self.intervals:
            return None

        interval_sec = float(getattr(self.config, "interval_sec", 1) or 1)                      # base interval duration in seconds
        session_sec = float(getattr(self.config, "session_sec", interval_sec) or interval_sec)  # larger session window length
        intervals_per_session = max(1, int(round(session_sec / interval_sec)))                  # number of interval rows grouped into one session block

        session_totals = defaultdict(float)                                # accumulates total counts across all session blocks
        session_rates = defaultdict(list)                                  # stores per-session-block rates for mean/std-dev calculations

        for start in range(0, len(self.intervals), intervals_per_session):
            block = self.intervals[start : start + intervals_per_session]  # take one session-sized chunk of interval rows
            if not block:
                continue

            block_duration = max(1e-9, len(block) * interval_sec)          # protect against divide-by-zero even in degenerate cases
            block_totals = defaultdict(float)                              # accumulates totals within just this session block

            for interval in block:
                for cls, value in interval.items():
                    if cls == "TIME" or cls == "RATIO":
                        continue                                           # ignore presentation-only fields when building numeric summaries
                    block_totals[cls] += float(value)

            for cls, total in block_totals.items():
                session_totals[cls] += total                                                        # add block total into the overall session total
                session_rates[cls].append(total / block_duration if block_duration > 0 else 0.0)    # also store normalized rate for mean/std-dev later

        if self.context_classes:
            object_total = session_totals.pop("OBJECTS", 0.0)                                       # keep OBJECTS at the end of the summary for cleaner presentation
            session_totals["OBJECTS"] = object_total
            session_rates.setdefault("OBJECTS", [])

        focus_total = sum(session_totals.get(cls, 0.0) for cls in self.focus_classes) or 1.0        # denominator for focus-class proportion values

        rows = []
        for cls, total in session_totals.items():
            rates = session_rates.get(cls, [])
            mean_rate = sum(rates) / len(rates) if rates else 0.0                                           # average per-session rate for this class
            std_dev = math.sqrt(sum((r - mean_rate) ** 2 for r in rates) / len(rates)) if rates else 0.0    # population-style std dev across session rates
            prop = (total / focus_total) if cls in self.focus_classes else "n/a"                            # only focus classes receive within-focus proportion values

            rows.append(
                {
                    "CLASS": cls,
                    "TOTAL_COUNT": round(total, 3),
                    "AVG_RATE": round(mean_rate, 3),
                    "STD_DEV": round(std_dev, 3),
                    "PROP": prop if isinstance(prop, str) else round(prop, 3),
                }
            )
        return self.writer.write_rows(
            "session_summary.csv",
            ["CLASS", "TOTAL_COUNT", "AVG_RATE", "STD_DEV", "PROP"],
            rows,
        )

# --------- MOTION ---------
class Motion:
    """
    Interval-based, per-track motion analysis.

    A motion event is counted when:
        - displacement >= threshold
        - sustained for >= min_frames
        - counted once per track per interval
    """
    def __init__(
        self,
        paths,
        frame_width,
        frame_height,
        config=None,
        start_time=None,
        class_config=None,
        tracker=None,
    ):
        self.out_folder = Path(paths["motion"])                           # motion outputs always write into the dedicated motion subfolder
        self.writer = CSVTableWriter(self.out_folder)                     # shared CSV writer for motion-related output tables
        self.config = config or MeasurementConfig()                       # config supplies thresholds, interval size, & persistence requirements
        self.start_time = _to_utc(start_time) if start_time else None     # normalized source start time for interval anchoring

        if class_config is None:
            raise ValueError("class_config is required.")                 # motion output needs an explicit class layout to build stable rows

        self.focus_classes, self.context_classes = _normalize_class_lists(class_config=class_config)    # motion tracks focus classes but keeps context structure available
        self.clock = IntervalClock(self.config.interval_sec, anchor_ts=self.start_time)                 # align motion summaries to the shared interval grid
        self._interval_start = None                                                                     # current active motion-summary interval boundary

        self.motion_threshold_px = float(self.config.motion_threshold_px)                               # per-frame displacement needed before motion persistence increments
        self.min_frames = int(getattr(self.config, "motion_min_frames", 3))                             # required consecutive qualifying frames for one motion event

        self.prev_center = defaultdict(dict)                              # stores previous center position per class & track id
        self.persist = defaultdict(dict)                                  # stores consecutive-above-threshold frame counts per class & track id
        self.locked = defaultdict(set)                                    # prevents repeated counting of the same track within one interval

        self.motion_events = defaultdict(int)                             # number of counted motion events per focus class in the current interval
        self.interval_displacement = defaultdict(float)                   # summed displacement contributing to intensity per class
        self.frames_with_motion = defaultdict(int)                        # number of frames containing qualifying motion per class
        self.interval_frames = 0                                          # total processed frames within the current motion interval

        self.rows_counts = []                                             # exported motion event counts by interval
        self.rows_intensity = []                                          # exported motion intensity rows by interval
        self.rows_prevalence = []                                         # exported motion prevalence rows by interval

    @staticmethod
    def _dist(a, b):
        """
        Return Euclidean distance between two center points.
        """
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5          # direct center displacement is the base motion signal

    @staticmethod
    def log_transform(x):
        """
        Log-transform a value using log1p for stable compression.
        This keeps large displacement totals readable while preserving zero.
        """
        return round(math.log1p(x), 3)

    def process_tracks(self, tracks_by_class, ts):
        """
        Update motion state from one frame of tracker output.
        Motion is event-based per track & summarized once per interval.
        """
        ts = _to_utc(ts)
        self.interval_frames += 1                                         # every processed frame contributes to prevalence denominator

        for cls in self.focus_classes:
            for track in tracks_by_class.get(cls, []):
                if track["age"] != 0:
                    continue                                              # only current-frame tracks contribute to motion updates

                track_id = track["id"]
                center = track["center"]
                prev = self.prev_center[cls].get(track_id)

                if prev is None:
                    self.prev_center[cls][track_id] = center              # initialize history the first time this track is seen
                    self.persist[cls][track_id] = 0
                    continue

                displacement = self._dist(prev, center)                   # compute frame-to-frame center displacement for this track
                self.prev_center[cls][track_id] = center                  # always update stored center so the next frame compares correctly

                if displacement >= self.motion_threshold_px:
                    self.persist[cls][track_id] = self.persist[cls].get(track_id, 0) + 1    # qualifying movement extends persistence streak
                else:
                    self.persist[cls][track_id] = 0                                         # falling below threshold breaks the persistence streak

                if self.persist[cls][track_id] >= self.min_frames and track_id not in self.locked[cls]:
                    self.motion_events[cls] += 1                          # count one motion event once the persistence requirement is met
                    self.interval_displacement[cls] += displacement       # accumulate displacement into interval intensity
                    self.frames_with_motion[cls] += 1                     # record that this class had qualifying motion in this frame
                    self.locked[cls].add(track_id)                        # lock the track so it is not counted again until the next interval

        rolled, new_start, old_start = self.clock.tick(ts, self._interval_start)  # check whether this frame crossed into a new interval
        self._interval_start = new_start

        if rolled and old_start is not None:
            self._finalize_interval(old_start)                            # write summarized rows for the interval that just ended
            self._reset_interval()                                        # clear interval-specific state before continuing into the next one

    def _finalize_interval(self, ts):
        """
        Convert current interval motion state into export rows.
        """
        time_str = _fmt_time(ts)                                          # export interval timestamp in the same HH:MM:SS format as other measurement tables

        counts = add_ratio_to_counts(
            {cls: self.motion_events.get(cls, 0) for cls in self.focus_classes},
            self.focus_classes,
            self.context_classes,
        )                                                                 # motion event counts use the standard ratio formatting when context classes exist

        intensity = {
            cls: self.log_transform(self.interval_displacement.get(cls, 0.0))
            for cls in self.focus_classes
        }                                                                 # intensity compresses displacement totals so large values remain readable

        prevalence = {
            cls: round(self.frames_with_motion.get(cls, 0) / max(self.interval_frames, 1), 3)
            for cls in self.focus_classes
        }                                                                 # prevalence reports how often motion appeared during the interval

        self.rows_counts.append({"TIME": time_str, **counts})             # store motion event counts for CSV export
        self.rows_intensity.append({"TIME": time_str, **intensity})       # store motion intensity row for CSV export
        self.rows_prevalence.append({"TIME": time_str, **prevalence})     # store motion prevalence row for CSV export

    def _reset_interval(self):
        """
        Clear interval-specific motion accumulators after an interval is finalized.
        """
        self.motion_events.clear()                                        # reset motion event counts for the next interval
        self.interval_displacement.clear()                                # reset displacement totals for the next interval
        self.frames_with_motion.clear()                                   # reset motion-presence counters for the next interval
        self.interval_frames = 0                                          # reset frame denominator for the next interval

        for cls in list(self.locked.keys()):
            self.locked[cls].clear()                                      # unlock all tracks so they can contribute a new event next interval
        for cls in list(self.persist.keys()):
            self.persist[cls].clear()                                     # reset persistence streaks at the interval boundary

    def save_results(self):
        """
        Save motion count, intensity, & prevalence tables.
        Returns a list of written file paths, or None when no motion rows were collected.
        """
        if not self.rows_counts:
            return None                                                     # motion export is skipped entirely when no interval rows exist

        outputs = []

        motion_count_headers = ["TIME", *self.focus_classes]                # base headers for motion event counts; ratio is added later when context classes are present
        if self.context_classes:
            motion_count_headers.append("RATIO")                            # ratio is included in the counts table when context classes exist since it is a simple derivative of the counts themselves and provides useful normalized perspective on motion frequency relative to overall object presence

        count_file = self.writer.write_rows(                                # motion event counts are the primary motion output, so they are written first using a clear naming convention that emphasizes the event-based nature of this table 
            "motion_counts.csv",
            motion_count_headers,
            self.rows_counts,
        )                                                                   # write interval motion-event counts using the standard count column layout
        if count_file:
            outputs.append(count_file)

        intensity_file = self.writer.write_rows(
            "motion_intensity.csv",
            ["TIME", *self.focus_classes],
            self.rows_intensity,
        )                                                                 # write per-class motion intensity rows
        if intensity_file:
            outputs.append(intensity_file)

        prevalence_file = self.writer.write_rows(
            "motion_prevalence.csv",
            ["TIME", *self.focus_classes],
            self.rows_prevalence,
        )                                                                 # write per-class motion prevalence rows
        if prevalence_file:
            outputs.append(prevalence_file)

        return outputs