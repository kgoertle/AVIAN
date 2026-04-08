# utils/detect/video_util.py
from __future__ import annotations                          # allows forward references in type hints without quoting types
import json                                                 # used for metadata serialization & ffprobe JSON parsing
import os                                                   # used for filesystem stat fallback timestamps
import platform                                             # used for platform-specific camera backend & timestamp handling
import queue                                                # used for passing frames/results safely between worker threads
import subprocess                                           # used to call ffprobe for file metadata extraction
import threading                                            # used to run the frame reader in a background thread
import time                                                 # used for short sleeps when live cameras temporarily fail to read
from dataclasses import dataclass                           # used to define the resolved runtime container
from datetime import datetime, timezone                     # used for parsing, normalizing, & storing timestamps
from pathlib import Path                                    # used for safe cross-platform path handling
from typing import Any                                      # used where metadata values may vary in type
import cv2                                                  # used for video capture, frame reads, & video writing
from .runtime_util import EOF_MARKER                        # sentinel used to signal end-of-file for completed video sources

# --------- CONSTANTS & PLATFORM FLAGS ---------
DEFAULT_CAMERA_WIDTH = 640                                  # safe fallback width when camera metadata cannot be resolved
DEFAULT_CAMERA_HEIGHT = 480                                 # safe fallback height when camera metadata cannot be resolved
DEFAULT_CAMERA_FPS = 30.0                                   # safe fallback FPS when camera metadata cannot be resolved

IS_MAC = platform.system() == "Darwin"                      # macOS flag used for backend selection & timestamp preference logic
IS_LINUX = platform.system() == "Linux"                     # Linux flag used when handling Raspberry Pi-specific behavior
IS_PI = IS_LINUX and ("arm" in platform.machine() or "aarch64" in platform.machine())  # Raspberry Pi / ARM Linux detection

# --------- RUNTIME CONTAINERS ---------
@dataclass
class SourceRuntime:
    """
    Fully resolved runtime information for one source.

    This is the finalized capture metadata that detect.py needs in order to:
        - create output paths
        - build the writer
        - calculate timestamps / ETA
        - initialize measurement systems
    """
    src_info: "VideoSourceInfo"                             # normalized metadata wrapper for the original source
    start_time: datetime                                    # resolved start/creation time used to anchor all later timestamps
    metadata: dict[str, Any]                                # exportable metadata dictionary saved alongside results
    fps: float                                              # finalized FPS used by writer, ETA logic, & measurement timing
    total_frames: int | None                                # total frame count for file sources, or None for open-ended cameras
    frame_width: int                                        # finalized frame width used by output writer & motion logic
    frame_height: int                                       # finalized frame height used by output writer & motion logic

class VideoSourceInfo:
    """
    Lightweight wrapper around raw metadata extracted from either:
        - a video file
        - a live camera source
    """

    def __init__(self, metadata: dict[str, Any], is_camera: bool, display_name: str):
        self.metadata = metadata                             # raw metadata dictionary extracted from file/camera inspection
        self.is_camera = is_camera                           # distinguishes open-ended live sources from finite file sources
        self.display_name = display_name                     # human-readable label used by UI messages higher in the pipeline

        self.width = metadata.get("width")                  # optional width pulled from extracted metadata
        self.height = metadata.get("height")                # optional height pulled from extracted metadata
        self.fps = metadata.get("fps")                      # optional FPS pulled from extracted metadata
        self.duration = metadata.get("duration")            # optional duration used as a frame-count fallback for files
        self.creation_time = metadata.get("creation_time_used")  # final timestamp string selected by metadata extraction logic
        self.creation_dt: datetime | None = None            # parsed datetime cached after first resolution

    def parse_creation_time(self) -> datetime:
        """
        Parse the resolved creation/start timestamp for this source.

        Falls back to current system time when parsing fails.
        """
        parsed = parse_timestamp(self.metadata.get("creation_time_used"))  # parse the chosen source timestamp using shared timestamp rules
        if parsed is None:
            self.creation_dt = datetime.now()               # final fallback is local system time so runtime never lacks a starting point
            return self.creation_dt

        self.creation_dt = parsed                           # cache the parsed datetime for any later consumers
        return parsed

# --------- TIMESTAMP HELPERS ---------
def parse_timestamp(value: str | None) -> datetime | None:
    """
    Parse a timestamp string using several common formats.

    Returns:
        datetime | None
    """
    if not value:
        return None                                         # empty or missing timestamp cannot be parsed

    text = value.strip().replace("Z", "+00:00")             # normalize trailing Z into explicit UTC offset for strptime/fromisoformat

    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(text, fmt)               # try common embedded/media timestamp formats first
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)        # naive timestamps are treated as UTC for consistency
            return dt
        except ValueError:
            continue                                        # move to the next known format when one fails

    try:
        dt = datetime.fromisoformat(text)                   # final fallback handles broader ISO-style timestamp strings
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)            # normalize naive ISO strings the same way as manual strptime formats
        return dt
    except Exception:
        return None                                         # parsing failure is allowed so caller can choose a safe fallback time

def parse_filename_time(video_path: str | Path):
    """
    Attempt to extract an HH-MM-SS time from a filename stem.

    Example:
        13-45-02.mp4 -> 13:45:02
    """
    stem = Path(video_path).stem                            # use only the filename stem so extension never affects parsing
    parts = stem.split("-")                                 # expected format is exactly three dash-separated time components

    if len(parts) != 3:
        return None                                         # filenames that do not match HH-MM-SS are ignored entirely

    try:
        h, m, s = map(int, parts)                           # parse hour, minute, & second directly from the filename parts
        return datetime.strptime(f"{h:02d}:{m:02d}:{s:02d}", "%H:%M:%S").time()
    except Exception:
        return None                                         # malformed numeric ranges or parse failures simply mean no usable filename time

# --------- METADATA EXTRACTION ---------
def extract_video_metadata(video_path: str | Path) -> dict[str, Any]:
    """
    Extract metadata for a video file using ffprobe plus filesystem fallbacks.
    """
    video_path = Path(video_path)                           # normalize path once so later filesystem/subprocess use stays consistent

    metadata: dict[str, Any] = {
        "type": "video",                                    # identifies this metadata block as file-based rather than live-camera based
        "source": str(video_path),                          # original source path stored for traceability in exported metadata
        "width": None,
        "height": None,
        "fps": None,
        "duration": None,
        "codec": None,
    }

    # --------- FFPROBE EXTRACTION ---------
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name,avg_frame_rate,duration",
            "-show_entries",
            "format_tags=creation_time",
            "-of",
            "json",
            str(video_path),
        ]                                                   # ffprobe pulls core video-stream metadata plus embedded creation_time when available

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        data = json.loads(result.stdout or "{}")            # parse ffprobe JSON output, tolerating blank stdout as empty metadata

        streams = data.get("streams", [])
        if streams:
            stream = streams[0]                             # only the first video stream is needed for detection/runtime setup
            metadata["width"] = stream.get("width")
            metadata["height"] = stream.get("height")
            metadata["codec"] = stream.get("codec_name")

            fps_str = stream.get("avg_frame_rate", "0/1")  # ffprobe commonly returns FPS as a fraction string like 30000/1001
            try:
                num, den = map(float, fps_str.split("/"))
                metadata["fps"] = round(num / den, 3) if den not in (0, 0.0) and num not in (0, 0.0) else None
            except Exception:
                metadata["fps"] = None                      # malformed FPS fractions fall back to unresolved FPS

            try:
                duration = stream.get("duration")
                metadata["duration"] = float(duration) if duration is not None else None
            except Exception:
                metadata["duration"] = None                 # duration is optional, so parse failures should not abort metadata extraction

        format_tags = data.get("format", {}).get("tags", {})
        metadata["creation_time_embedded"] = format_tags.get("creation_time")  # embedded media timestamp is often the best file-origin timestamp
    except Exception as exc:
        metadata["ffprobe_error"] = str(exc)                # retain ffprobe failure details for later debugging without blocking the pipeline

    # --------- FILESYSTEM TIMESTAMP ---------
    try:
        stat = os.stat(video_path)
        creation_ts = getattr(stat, "st_birthtime", stat.st_mtime)  # prefer creation time when available, otherwise fall back to modification time
        metadata["creation_time_filesystem"] = datetime.fromtimestamp(creation_ts).isoformat()
    except Exception as exc:
        metadata["creation_time_filesystem"] = None
        metadata["fs_time_error"] = str(exc)                # preserve filesystem timestamp failure info for debugging

    # --------- FILENAME TIME ---------
    filename_time = parse_filename_time(video_path)         # attempt to recover a time-of-day directly from filenames like HH-MM-SS.mp4
    metadata["creation_time_filename"] = filename_time.strftime("%H:%M:%S") if filename_time else None

    # --------- SELECT BEST TIMESTAMP ---------
    creation_fs = metadata.get("creation_time_filesystem")
    creation_file = metadata.get("creation_time_filename")
    creation_emb = metadata.get("creation_time_embedded")

    if creation_file:
        if IS_MAC and creation_fs:
            date_part = creation_fs.split("T")[0]           # on macOS, pair filename time with the filesystem date when available
        elif IS_PI:
            date_part = datetime.now().strftime("%Y-%m-%d") # Raspberry Pi often lacks reliable file birthtimes, so use current date instead
        elif creation_fs and isinstance(creation_fs, str) and "T" in creation_fs:
            date_part = creation_fs.split("T")[0]           # generic fallback: use filesystem date when an ISO-like timestamp exists
        else:
            date_part = datetime.now().strftime("%Y-%m-%d") # final fallback date when only a filename time is available

        metadata["creation_time_used"] = f"{date_part}T{creation_file}"  # combine chosen date with recovered filename time

    elif IS_MAC and creation_fs:
        metadata["creation_time_used"] = creation_fs        # macOS filesystem time is often reliable enough to use directly

    elif creation_emb:
        metadata["creation_time_used"] = creation_emb       # otherwise prefer embedded media timestamp when present

    elif creation_fs:
        metadata["creation_time_used"] = creation_fs        # final structured fallback is the filesystem timestamp

    else:
        metadata["creation_time_used"] = datetime.now().isoformat()  # absolute last resort is current system time
        metadata["creation_time_error"] = "No usable timestamp from filename/fs/embedded. Using system time."

    metadata["extracted_at"] = datetime.now().isoformat()   # records when metadata extraction itself occurred
    return metadata

def extract_camera_metadata(cap, source_id: int) -> dict[str, Any]:
    """
    Extract basic metadata from a live camera capture.
    """
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)               # query current capture width from OpenCV
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)             # query current capture height from OpenCV
    fps = cap.get(cv2.CAP_PROP_FPS)                         # query current capture FPS from OpenCV

    now = datetime.now().isoformat()                        # live camera sessions use current system time as their start reference

    return {
        "type": "camera",                                   # identifies this metadata block as a live camera source
        "source": f"usb{source_id}",                        # store source label in the same display style used elsewhere in the pipeline
        "width": int(width) if width else DEFAULT_CAMERA_WIDTH,
        "height": int(height) if height else DEFAULT_CAMERA_HEIGHT,
        "fps": round(fps if fps and fps > 0 else DEFAULT_CAMERA_FPS, 3),
        "duration": None,                                   # live cameras are treated as open-ended sources
        "started_at": now,
        "creation_time_used": now,                          # camera sources begin at the current time because no file-origin timestamp exists
    }

def parse_creation_time(metadata: dict[str, Any]):
    """
    Backward-compatible standalone timestamp parser.

    Existing callers elsewhere in the pipeline can still use this.
    """
    return parse_timestamp(metadata.get("creation_time_used"))  # thin wrapper retained so older code paths do not have to change

# --------- CAPTURE HELPERS ---------
def get_camera_backend() -> int:
    """
    Return the best OpenCV backend for USB camera capture on this platform.
    """
    return cv2.CAP_AVFOUNDATION if IS_MAC else cv2.CAP_V4L2   # macOS uses AVFoundation while Linux-based systems prefer V4L2

def open_video_capture(source: int | str, is_camera: bool, source_display_name: str, ui):
    """
    Open a cv2.VideoCapture for either a USB camera or a video file.

    Returns:
        cv2.VideoCapture | None
    """
    try:
        if is_camera:
            cap = cv2.VideoCapture(int(source), get_camera_backend())  # camera sources use the platform-appropriate capture backend
        else:
            cap = cv2.VideoCapture(str(source))                        # file sources are opened directly from path
    except Exception:
        ui.open_capture_fail(source_display_name)
        return None

    if cap is None or not cap.isOpened():
        ui.open_capture_fail(source_display_name)                      # failed capture opens are reported uniformly through the UI
        return None

    return cap

def build_source_info(cap, source: int | str, is_camera: bool, source_display_name: str) -> VideoSourceInfo:
    """
    Build a VideoSourceInfo wrapper for the requested source.
    """
    if is_camera:
        try:
            source_id = int(source)                        # normalize the camera source into an integer index when possible
        except ValueError:
            source_id = 0                                  # fallback index keeps metadata generation from failing entirely

        metadata = extract_camera_metadata(cap, source_id)
        return VideoSourceInfo(metadata, is_camera=True, display_name=source_display_name)

    metadata = extract_video_metadata(source)
    return VideoSourceInfo(metadata, is_camera=False, display_name=source_display_name)

def resolve_source_dimensions(cap, src_info: VideoSourceInfo, is_camera: bool, ui) -> tuple[int, int] | None:
    """
    Resolve width/height for the source.

    For files, read the first frame to guarantee actual dimensions.
    For cameras, rely on metadata/capture properties.
    """
    if is_camera:
        width = src_info.width or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or DEFAULT_CAMERA_WIDTH)    # trust metadata first, then live capture properties
        height = src_info.height or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or DEFAULT_CAMERA_HEIGHT)
        return width, height

    ok, frame = cap.read()
    if not ok or frame is None:
        ui.read_frame_fail(src_info.display_name)                # file dimensions are only trusted after a real frame can be read
        return None

    height, width = frame.shape[:2]                              # actual decoded frame shape is the most reliable file-dimension source
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)                          # rewind so the real processing loop still starts from frame zero
    return width, height

def resolve_source_fps(cap, src_info: VideoSourceInfo) -> float:
    """
    Resolve FPS for the source with safe fallback behavior.
    """
    if src_info.fps:
        return src_info.fps                                      # prefer extracted metadata when it already contains a usable FPS

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        return DEFAULT_CAMERA_FPS                                # fallback protects timing/UI/writer code from invalid FPS values

    try:
        return DEFAULT_CAMERA_FPS if src_fps != src_fps else src_fps  # NaN-safe check without relying on math.isnan
    except Exception:
        return DEFAULT_CAMERA_FPS

def resolve_total_frames(cap, src_info: VideoSourceInfo, is_camera: bool, fps: float) -> int | None:
    """
    Resolve total frame count for file sources.

    Cameras are treated as open-ended sources.
    """
    if is_camera:
        return None                                              # live cameras do not have a meaningful terminal frame count

    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if total is not None and total >= 2:
        return int(total)                                        # trust capture-reported frame count when it looks valid

    if src_info.duration:
        return int(src_info.duration * fps)                      # duration*fps gives a useful estimate when direct frame count is missing
    return None

def resolve_source_runtime(cap, source: int | str, is_camera: bool, source_display_name: str, ui) -> SourceRuntime | None:
    """
    Resolve all runtime metadata for a source in one place.
    """
    src_info = build_source_info(cap, source, is_camera, source_display_name)  # build normalized source metadata wrapper first
    start_time = src_info.parse_creation_time()                                 # resolve & parse the best available starting timestamp

    metadata = dict(src_info.metadata)
    metadata["creation_time_str"] = start_time.strftime("%H:%M:%S")             # store compact display time used by downstream outputs/UI

    dimensions = resolve_source_dimensions(cap, src_info, is_camera, ui)
    if dimensions is None:
        return None
    frame_width, frame_height = dimensions

    fps = resolve_source_fps(cap, src_info)                                     # finalize FPS with safe fallback behavior
    total_frames = resolve_total_frames(cap, src_info, is_camera, fps)          # resolve total frame count only for finite file sources

    return SourceRuntime(
        src_info=src_info,
        start_time=start_time,
        metadata=metadata,
        fps=fps,
        total_frames=total_frames,
        frame_width=frame_width,
        frame_height=frame_height,
    )

def write_metadata_json(path: Path, metadata: dict[str, Any]) -> None:
    """
    Save metadata to a JSON file.
    """
    with open(path, "w") as handle:
        json.dump(metadata, handle, indent=2)                  # formatted JSON keeps saved metadata readable for later inspection

# --------- VIDEO READER ---------
class VideoReader:
    """
    Frame reader for a single source.

    Responsibilities:
        - read frames from cv2.VideoCapture
        - push frames into the frame queue
        - emit EOF for video files when the source ends
        - drop stale camera frames when the input queue backs up
    """

    def __init__(
        self,
        cap,
        frame_queue: queue.Queue,
        infer_queue: queue.Queue,
        is_camera: bool,
        source_display_name: str,
        global_stop_event,
        ui,
    ):
        self.cap = cap                                              # active cv2 capture object for this source
        self.frame_queue = frame_queue                              # queue feeding frames into the inference worker
        self.infer_queue = infer_queue                              # queue used to emit EOF when a file source finishes
        self.is_camera = is_camera                                  # controls low-latency behavior vs complete file playback behavior
        self.source_display_name = source_display_name              # human-readable source label used in UI/error messages
        self.global_stop_event = global_stop_event                  # shared shutdown signal for the full detection pipeline
        self.ui = ui                                                # retained for future UI hooks even though this reader stays mostly quiet

        self._stop_local = threading.Event()                        # local stop flag lets the reader exit independently when requested
        self._thread = None                                         # background reader thread created in start()

    def start(self):
        """
        Start the background frame-reader thread if it is not already running.
        """
        if not self._thread or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, daemon=True)  # daemon thread avoids blocking interpreter shutdown in crash cases
            self._thread.start()

    def stop(self):
        """
        Signal the reader loop to stop.
        """
        self._stop_local.set()

    def join(self, timeout=None):
        """
        Wait for the reader thread to exit.
        """
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run(self):
        """
        Main frame-reader loop.
        Live cameras favor freshness by dropping stale queued frames.
        File sources favor completeness by blocking until frames can be queued.
        """
        while not self._stop_local.is_set():
            ret, frame = self.cap.read()

            if not ret or frame is None:
                if self.is_camera:
                    time.sleep(0.01)                              # camera reads can fail transiently, so pause briefly & retry instead of ending
                    continue

                try:
                    self.infer_queue.put((EOF_MARKER, None), timeout=0.1)  # finite file sources emit EOF once the stream truly ends
                except Exception:
                    pass
                break

            try:
                self.frame_queue.put(frame, timeout=0.02)         # fast path: queue frame immediately when there is room
            except queue.Full:
                if self.is_camera:
                    try:
                        self.frame_queue.get_nowait()             # drop the oldest queued frame so live camera mode stays current
                    except queue.Empty:
                        pass

                    try:
                        self.frame_queue.put(frame, timeout=0.02)  # retry with newest frame after removing one stale frame
                    except queue.Full:
                        pass                                      # if still full, skip this frame rather than increasing latency
                else:
                    while not self.global_stop_event.is_set():
                        try:
                            self.frame_queue.put(frame, timeout=0.1)  # file mode waits so every frame is preserved in order
                            break
                        except queue.Full:
                            continue

# --------- VIDEO WRITING ---------
def create_video_writer(
    out_file: Path,
    fps: float,
    frame_width: int,
    frame_height: int,
    source_display_name: str,
    ui,
    cap,
    source_type: str,
):
    """
    Create and register a cv2.VideoWriter for annotated output.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")                    # mp4v provides a widely supported annotated-output codec choice
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (frame_width, frame_height))

    if not writer.isOpened():
        ui.warn(f"VideoWriter failed: {source_display_name}")
        return None                                             # refuse writer use if OpenCV failed to initialize the destination file

    ui.register_writer(
        out_file.name,
        writer,
        cap,
        source_type,
        out_file,
        display_name=source_display_name,
    )                                                           # register writer with the UI/runtime tracking layer before returning it
    return writer

def write_annotated_frame(writer, frame):
    """
    Write one annotated frame if the writer exists.
    """
    if writer:
        writer.write(frame)                                     # small guard keeps downstream code from checking writer existence repeatedly