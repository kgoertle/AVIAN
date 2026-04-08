# utils/detect/inference_util.py
from __future__ import annotations              # allows forward references in type hints without quoting types
import queue                                    # used for passing frames/results safely between worker threads
import threading                                # used to run inference in a background worker thread
from typing import Any                          # used where payload contents or model-derived values may vary in type
import cv2                                      # used for frame resizing & annotation rendering
import numpy as np                              # used for synthetic probe frames & OBB point normalization
from ultralytics import YOLO                    # Ultralytics model wrapper used for loading & running inference
from .runtime_util import EOF_MARKER            # sentinel used to signal end-of-stream or worker startup failure
DEFAULT_IMGSZ = 416                             # default inference image size used when no override is supplied

# --------- MODEL HELPERS ---------
def load_model(weights_path):
    """
    Load a YOLO model from a weights path.

    The original path is preserved on the model object for convenience so downstream
    systems can still reference which weights were loaded.
    """
    model = YOLO(str(weights_path))             # load the Ultralytics model from the provided weights path
    model.weights_path = weights_path           # preserve the original path on the model object for later reference/debugging
    return model

def detect_is_obb_model(model) -> tuple[bool, str | None]:
    """
    Probe the model once to determine whether it produces OBB results.

    Returns:
        (is_obb_model, obb_corner_attr)

    obb_corner_attr is the best available attribute name for corner extraction,
    if one exists on the current Ultralytics version.
    """
    try:
        test_frame = np.zeros((640, 640, 3), dtype=np.uint8)   # build a blank probe frame so model capability can be tested safely
        results = model.predict(test_frame, verbose=False, show=False)
        if not results:
            return False, None                                 # no results means OBB capability could not be confirmed

        result = results[0]
        if not (hasattr(result, "obb") and result.obb is not None):
            return False, None                                 # standard detection result with no OBB payload

        candidate_attrs = [
            "xyxyxyxy",                                        # common OBB corner attribute in some Ultralytics versions
            "xyxyxyxyn",                                       # normalized OBB corner attribute variant
            "xyxyxyxyxyxyxyxy",                                # extra-expanded corner attribute variant seen in some builds
        ]

        for attr in candidate_attrs:
            if hasattr(result.obb, attr):
                try:
                    value = getattr(result.obb, attr)          # probe the attribute dynamically because version support varies
                    if value is not None:
                        return True, attr                      # first usable corner attribute wins
                except Exception:
                    continue                                   # keep trying alternate attribute names if access fails

        return True, None                                      # model appears to be OBB-capable even if no corner attribute was usable
    except Exception:
        return False, None                                     # any probe failure falls back to standard non-OBB behavior

# --------- FRAME PREP & PREDICTION ---------
def prepare_inference_frame(frame, is_camera: bool, frame_width: int, frame_height: int):
    """
    Prepare a frame for model inference.

    Camera frames are resized to the configured writer/output dimensions.
    Video-file frames are passed through unchanged.
    """
    if is_camera:
        return cv2.resize(frame, (frame_width, frame_height))  # camera streams are normalized to the configured runtime dimensions
    return frame                                               # file frames are already tied to source dimensions & can pass through unchanged


def predict_frame(model, frame, imgsz: int):
    """
    Run YOLO prediction on a single frame.

    Returns:
        Ultralytics results object or None on failure.
    """
    try:
        return model.predict(
            frame,
            verbose=False,                                     # suppress Ultralytics console noise during per-frame inference
            show=False,                                        # never open model preview windows inside the worker thread
            imgsz=imgsz,                                       # use configured inference image size for all frame predictions
        )
    except Exception:
        return None                                            # inference failures are surfaced as None so the caller can degrade gracefully

def render_annotated_frame(results, fallback_frame, frame_width: int, frame_height: int):
    """
    Render an annotated frame from prediction results.

    If plotting fails or results are empty, return the fallback frame.
    The output is resized if needed so it always matches the target dimensions.
    """
    if not results:
        return fallback_frame                                  # if prediction failed or returned nothing, preserve the original prepared frame

    try:
        annotated = results[0].plot()                          # ask Ultralytics to render the plotted detections for the first result
        height, width = annotated.shape[:2]

        if (width, height) != (frame_width, frame_height):
            return cv2.resize(
                annotated,
                (frame_width, frame_height),
                interpolation=cv2.INTER_LINEAR,                # resize plotted frame back to the expected output dimensions if needed
            )

        return annotated                                       # already matches the target output size, so no extra resize is needed
    except Exception:
        return fallback_frame                                  # plotting failures should never break the detection pipeline

# --------- BOX EXTRACTION HELPERS ---------
def _extract_axis_aligned_boxes(result) -> tuple[dict, list[list[float | int]]]:
    """
    Extract standard axis-aligned boxes from a YOLO result.

    Row format:
        [x1, y1, x2, y2, conf, class_id]
    """
    names = result.names                                       # class-id-to-name lookup table returned by Ultralytics for this result

    try:
        xyxy = result.boxes.xyxy.cpu().numpy()                 # axis-aligned box coordinates in pixel space
        conf = result.boxes.conf.cpu().numpy()                 # confidence values for each detection
        cls = result.boxes.cls.cpu().numpy()                   # class ids for each detection
    except Exception:
        return names, []                                       # malformed or missing box payload falls back to an empty detection list

    boxes_list = [
        [float(x1), float(y1), float(x2), float(y2), float(cf), int(class_id)]
        for (x1, y1, x2, y2), cf, class_id in zip(xyxy, conf, cls)
    ]                                                          # normalize every row to the shared downstream box schema
    return names, boxes_list

def _normalize_obb_points(points, obb_corner_attr: str | None, frame_width: int | None, frame_height: int | None):
    """
    Normalize raw OBB corner arrays into a flat 8-value sequence.

    If normalized coordinates are provided, convert them back into pixel space.
    """
    pts = np.asarray(points).reshape(-1)                       # flatten incoming corner structure so all later checks work on one-dimensional arrays

    if obb_corner_attr == "xyxyxyxyn" and frame_width and frame_height:
        pts = pts.copy()                                       # avoid mutating the original array if normalized coordinates must be rescaled
        pts[0::2] *= float(frame_width)                        # x coordinates are every other value beginning at index 0
        pts[1::2] *= float(frame_height)                       # y coordinates are every other value beginning at index 1

    if pts.size == 8:
        return [float(v) for v in pts.tolist()]               # already a flat 4-corner sequence in x1,y1,...,x4,y4 form

    if pts.size == 4 * 2:
        return [float(v) for v in pts.tolist()]               # equivalent explicit check kept for readability around corner-count expectations

    return None                                                # unsupported corner payload layout cannot be used safely downstream

def _extract_obb_boxes(
    result,
    obb_corner_attr: str | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> tuple[dict, list[list[float | int]]]:
    """
    Extract OBB detections while always including axis-aligned xyxy values first.

    Row format:
        [x1, y1, x2, y2, conf, class_id, <optional 8 OBB corner floats>]
    """
    names = result.names                                       # class-id-to-name mapping is still shared with standard detection output

    try:
        xyxy = result.obb.xyxy.cpu().numpy()                   # axis-aligned bounds derived from the oriented boxes
        conf = result.obb.conf.cpu().numpy()                   # confidence values for each OBB detection
        cls = result.obb.cls.cpu().numpy()                     # class ids for each OBB detection
    except Exception:
        return names, []                                       # broken OBB payload yields no boxes rather than crashing the worker

    corners = None
    if obb_corner_attr and hasattr(result.obb, obb_corner_attr):
        try:
            raw_corners = getattr(result.obb, obb_corner_attr)  # access the best-supported corner attribute detected during model probing
            if raw_corners is not None:
                corners = raw_corners.cpu().numpy()             # move corners to CPU/numpy for uniform downstream processing
        except Exception:
            corners = None                                      # corner extraction failure should not block basic OBB xyxy export

    boxes_list: list[list[float | int]] = []

    for idx, ((x1, y1, x2, y2), cf, class_id) in enumerate(zip(xyxy, conf, cls)):
        row: list[float | int] = [
            float(x1),
            float(y1),
            float(x2),
            float(y2),
            float(cf),
            int(class_id),
        ]                                                       # every OBB row begins with the same shared xyxy/conf/class layout

        if corners is not None and idx < len(corners):
            flattened = _normalize_obb_points(
                corners[idx],
                obb_corner_attr=obb_corner_attr,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if flattened:
                row.extend(flattened)                           # append the optional 8-value corner sequence when it can be normalized safely

        boxes_list.append(row)

    return names, boxes_list

def extract_boxes_from_results(
    results,
    is_obb_model: bool,
    obb_corner_attr: str | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
):
    """
    Extract names and boxes from the first prediction result.

    Supports both:
        - standard axis-aligned models
        - OBB models with optional polygon corner extraction
    """
    if not results:
        return {}, []                                           # empty predictions produce no names or boxes

    result = results[0]

    try:
        if is_obb_model and hasattr(result, "obb") and result.obb is not None:
            return _extract_obb_boxes(
                result,
                obb_corner_attr=obb_corner_attr,
                frame_width=frame_width,
                frame_height=frame_height,
            )                                                   # OBB-capable models prefer the richer OBB extraction path

        return _extract_axis_aligned_boxes(result)              # otherwise use standard xyxy extraction
    except Exception:
        return result.names, []                                 # preserve class-name lookup even if actual box extraction fails

# --------- PAYLOAD BUILDING ---------
def build_inference_payload(
    results,
    frame_for_inference,
    is_obb_model: bool,
    obb_corner_attr: str | None,
    frame_width: int,
    frame_height: int,
):
    """
    Build the standardized inference payload consumed by detect.py.

    Payload format:
        (annotated_frame, names, boxes_list)
    """
    annotated = render_annotated_frame(
        results=results,
        fallback_frame=frame_for_inference,
        frame_width=frame_width,
        frame_height=frame_height,
    )                                                           # always build an annotated frame first so downstream video writing stays stable

    names, boxes_list = extract_boxes_from_results(
        results=results,
        is_obb_model=is_obb_model,
        obb_corner_attr=obb_corner_attr,
        frame_width=frame_width,
        frame_height=frame_height,
    )                                                           # extract structured detections in the same pass used by measurement systems

    return annotated, names, boxes_list                         # packaged in the exact format expected by the main detection loop

def push_inference_result(infer_queue: queue.Queue, payload, is_camera: bool) -> None:
    """
    Push an inference payload into the output queue.

    Camera mode prefers low latency:
        - if full, drop the oldest result and keep the newest one

    File mode prefers completeness:
        - block until the result is queued
    """
    if is_camera:
        try:
            infer_queue.put_nowait(payload)                     # fast path for live cameras where keeping up matters more than completeness
            return
        except queue.Full:
            try:
                infer_queue.get_nowait()                        # discard the oldest queued result to make room for the newest frame
            except queue.Empty:
                pass

            try:
                infer_queue.put_nowait(payload)                 # retry immediately after dropping one stale result
            except queue.Full:
                pass                                            # if the queue is still full, silently skip this frame to preserve low latency
        return

    infer_queue.put(payload)                                    # file playback mode blocks so every frame result is preserved in order

# --------- INFERENCE WORKER ---------
class InferenceWorker:
    """
    Background worker that performs YOLO inference for one source.

    Responsibilities:
        - load the model
        - detect whether the model is OBB-capable
        - read frames from the frame queue
        - run prediction
        - render annotations
        - extract boxes
        - push structured results into the inference queue
    """
    def __init__(
        self,
        weights_path,
        frame_queue: queue.Queue,
        infer_queue: queue.Queue,
        is_camera: bool,
        frame_width: int,
        frame_height: int,
        source_display_name: str,
        global_stop_event,
        ui,
        imgsz: int = DEFAULT_IMGSZ,
    ):
        self.weights_path = weights_path                          # model weights file used for this worker's inference session
        self.frame_queue = frame_queue                            # queue supplying raw/prepared frames from the reader
        self.infer_queue = infer_queue                            # queue receiving standardized inference payloads for the main loop
        self.is_camera = is_camera                                # controls low-latency behavior & frame preparation logic
        self.frame_width = frame_width                            # target frame width used for resizing & annotation output consistency
        self.frame_height = frame_height                          # target frame height used for resizing & annotation output consistency
        self.source_display_name = source_display_name            # human-readable source label used in UI error reporting
        self.global_stop_event = global_stop_event                # shared stop signal used across the full detection pipeline
        self.ui = ui                                              # UI object used for reporting inference failures
        self.imgsz = imgsz                                        # configured YOLO inference image size

        self._stop_local = threading.Event()                      # local stop flag for this worker instance only
        self._thread = None                                       # background thread handle created in start()

        self.model = None                                         # loaded Ultralytics model instance
        self.is_obb_model = False                                 # cached capability flag set during startup probing
        self.obb_corner_attr: str | None = None                   # best-supported OBB corner attribute name, if available

    @property
    def is_obb(self) -> bool:
        """
        Public compatibility property used by measurement systems.
        """
        return bool(self.is_obb_model)                            # expose worker OBB status through a stable public property name

    def start(self):
        """
        Load the model and start the background inference thread.
        """
        if self._thread and self._thread.is_alive():
            return                                                # avoid creating duplicate worker threads for the same instance

        try:
            self.model = load_model(self.weights_path)            # load the configured YOLO weights before the thread begins consuming frames
            self.is_obb_model, self.obb_corner_attr = detect_is_obb_model(self.model)  # probe OBB capability once during startup instead of every frame
        except Exception as exc:
            if self.ui:
                self.ui.inference_fail(self.source_display_name, exc)  # surface startup failure through the terminal UI when available
            try:
                self.infer_queue.put((EOF_MARKER, "INFER_INIT_FAIL"))  # signal the consumer that inference could not initialize
            except Exception:
                pass
            return

        self._thread = threading.Thread(target=self._run, daemon=True)  # daemon thread prevents inference worker from blocking interpreter shutdown
        self._thread.start()

    def stop(self):
        """
        Signal the worker loop to stop.
        """
        self._stop_local.set()                                    # local stop flag lets the worker exit even without a global shutdown

    def join(self, timeout=None):
        """
        Wait for the worker thread to exit.
        """
        if self._thread:
            self._thread.join(timeout=timeout)                    # join only when the worker thread was actually started

    def _run(self):
        """
        Main inference loop.
        Frames are consumed from the frame queue, passed through YOLO, converted into
        a standardized payload, & pushed into the inference queue for detect.py.
        """
        while not self._stop_local.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.05)        # short timeout keeps the worker responsive to global shutdown
            except queue.Empty:
                if self.global_stop_event.is_set():
                    break                                         # stop waiting once the overall detection pipeline is shutting down
                continue

            frame_for_inference = prepare_inference_frame(
                frame,
                is_camera=self.is_camera,
                frame_width=self.frame_width,
                frame_height=self.frame_height,
            )                                                     # normalize live-camera frames before inference while leaving file frames alone

            results = predict_frame(
                model=self.model,
                frame=frame_for_inference,
                imgsz=self.imgsz,
            )

            if results is None and self.ui:
                self.ui.inference_fail(
                    self.source_display_name,
                    RuntimeError("Model prediction failed."),
                )                                                 # surface runtime inference failure but keep the worker alive for later frames

            payload = build_inference_payload(
                results=results,
                frame_for_inference=frame_for_inference,
                is_obb_model=self.is_obb_model,
                obb_corner_attr=self.obb_corner_attr,
                frame_width=self.frame_width,
                frame_height=self.frame_height,
            )                                                     # build the shared payload format consumed by the main detection loop

            push_inference_result(
                infer_queue=self.infer_queue,
                payload=payload,
                is_camera=self.is_camera,
            )                                                     # queue result using latency-first or completeness-first behavior based on source type