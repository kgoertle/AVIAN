# detect.py
from __future__ import annotations          # allows forward references in type hints without quoting types
import queue                                # used for thread-safe frame & inference queues
import sys                                  # used for clean CLI exits
import time                                 # used for timing, FPS tracking, & idle waits
import threading                            # used to run one detection instance per source/model pair
from dataclasses import dataclass           # used to create lightweight immutable info containers
from datetime import datetime, timedelta    # used for timestamping sessions & per-frame video time
from pathlib import Path                    # used for safe cross-platform path handling
from typing import Any                      # used where a value may come from multiple runtime types
from .utils.ui import DetectUI, fmt_bold    # terminal UI manager & helper for bold terminal text
from .utils.detect import (
    MeasurementConfig,                      # central measurement settings shared across helpers
    ClassConfig,                            # stores class labels, focus classes, & context classes
    Counter,                                # tracks per-class counts over time
    Aggregator,                             # aggregates per-frame counts into interval/session summaries
    Interactions,                           # measures class interactions through tracked detections
    Motion,                                 # computes motion-based measurements from tracked detections
    Tracker,                                # persistent object tracking across frames
    InferenceWorker,                        # background worker that runs model inference on frames
    VideoReader,                            # background reader that pulls frames from capture into queue
    SourceRuntime,                          # resolved runtime metadata for the current source
    DEFAULT_IMGSZ,                          # default image size passed into inference worker
    EOF_MARKER,                             # sentinel value used to indicate end-of-stream
    compute_counts_from_boxes,              # converts raw detections into per-class count dictionary
    create_video_writer,                    # creates the annotated output video writer
    open_video_capture,                     # opens cv2 video capture for file or camera source
    drain_queue,                            # safely empties remaining items from a queue during shutdown
    is_eof_marker,                          # checks whether a pulled queue item represents end-of-stream
    load_or_create_classes,                 # loads cached class config or derives it from model weights
    parse_arguments,                        # parses CLI arguments for detection
    release_quietly,                        # safely releases cv2-like resources without noisy errors
    resolve_source_runtime,                 # resolves FPS, frame size, total frames, start time, & metadata
    safe_extend,                            # appends results to a list only when values are valid
    write_annotated_frame,                  # writes a processed frame to the output video
    write_metadata_json,                    # saves runtime/source metadata to JSON
)
from .utils.paths import WEIGHTS_DIR, get_output_folder, get_runs_dir   # shared path helpers
from .utils.train.io import ensure_weights                              # resolves official weights & downloads them if needed

# --------- CONSTANTS & GLOBALS ---------
stop_event = threading.Event()      # shared stop flag seen by all workers & processors in this script

FRAME_QUEUE_SIZE = 50               # max queued raw frames per processor before reader must slow down
INFER_QUEUE_SIZE = 20               # max queued inference outputs per processor before infer loop backs up


# --------- SOURCE & MODEL INFO POOL ---------
@dataclass(frozen=True)
class SourceInfo:
    """
    Holds resolved information for one input source.
    This keeps source parsing logic separate from the heavier detection manager so that every later step can rely on a single normalized representation of a source.
    """
    source: int | str       # original normalized source value: camera index int or file path str
    source_type: str        # resolved source type, e.g. "usb" for camera or "video" for file source

    @property
    def is_camera(self) -> bool:
        """
        Return True when this source represents a USB camera.
        This property makes later branching easier so the rest of the file does not need to repeatedly compare raw source_type strings everywhere.
        """
        return self.source_type == "usb"  # camera sources are stored using the "usb" source type

    @property
    def short_name(self) -> str:
        """
        Return a short display-safe name for UI & output labeling.
        Camera sources keep a readable name like 'usb0', while file sources are reduced to the filename stem so the UI stays compact.
        """
        return f"usb{self.source}" if self.is_camera else Path(str(self.source)).stem   # examples:
                                                                                            # source=0 & source_type="usb" -> "usb0"
                                                                                            # source="video.mp4"           -> "video"
@dataclass(frozen=True)
class ModelInfo:
    """
    Holds resolved information for one model.
    Once a model is validated, this object bundles together its display name, weights path, & class configuration so the rest of setup can stay clean & predictable.
    """
    model_name: str                 # display/run name for this model
    weights_path: Path              # resolved .pt file path for this model
    class_config: ClassConfig       # loaded classes, focus classes, & context classes for this model

# --------- PROCESSOR SETUP & MANAGEMENT UTILITIES ---------

def _parse_source_arg(raw_source: Any, ui: DetectUI) -> SourceInfo | None:
    """
    Normalize one user-supplied source argument into a SourceInfo object.
    
    Supported inputs include:
        - USB camera strings like 'usb0', 'usb1', etc.
        - video file paths like 'video.mp4', '/path/to/video.avi', etc.
    Returns None when the provided source cannot be interpreted safely.
    """
    source_text = str(raw_source).strip()                                               # normalize input to trimmed string for consistent parsing

    if source_text.lower().startswith("usb"):                                           # camera sources are expected in the form "usb0"
        try:
            source_id = int(source_text[3:])                                            # pull numeric camera index from the text after "usb"
        except ValueError:
            ui.warn(f"Invalid USB source '{source_text}' — must be like usb0, usb1")
            return None                                                                 # fail early so an invalid source does not create a broken processor

        return SourceInfo(source=source_id, source_type="usb")                          # normalized USB camera source

    return SourceInfo(source=source_text, source_type="video")                          # everything else is treated as a video path


def _model_select_prompt(runs_dir: Path, args: Any, ui: DetectUI) -> tuple[str, Path] | None:
    """
    Resolve a default model when '--model' was omitted entirely, which checks for:
        - if model runs exist in '/runs', auto-select the only run or prompt the user to choose
        - if no runs exist, fall back to official YOLO11n weights
    
    Returns:
        (model_name, weights_path) on success
        None on failure or user cancellation
    """
    model_dirs = sorted(
        [
            directory
            for directory in runs_dir.iterdir()
            if directory.is_dir() and (args.test or directory.name.lower() != "test")
        ],
        reverse=True,  # newest-looking folders usually appear first when names are timestamped
    )

    if not model_dirs:                                          # if no local trained runs were found, fall back to the official YOLO11n model
        placeholder = WEIGHTS_DIR / "yolo11n.pt"                    # use a real path in the weights directory so fallback resolution follows the same path logic
        resolved = ensure_weights(placeholder, "yolo11n")           # ensure_weights handles locating or downloading the model into the expected weights folder
    
        if resolved is None or not resolved.exists():
            ui.error("Failed to download or resolve YOLO11n fallback model.")
            ui.exit("Detection aborted due to missing fallback model.")
            return None                                                             # abort if even the fallback model could not be prepared
        return resolved.stem, resolved                                              # return the resolved fallback model information

    if len(model_dirs) == 1:
        selected = model_dirs[0]                                                    # streamline the common case where only one trained model exists
    else:
        selected = ui.prompt_model_selection(runs_dir, exclude_test=not args.test)  # prompt user only when there is an actual choice to make
        if not selected:
            return None                                                             # user declined or prompt returned no valid selection
        selected = Path(selected)                                                   # normalize returned selection to Path for downstream path operations

    best = selected / "weights" / "best.pt"                                         # standard Ultralytics best-checkpoint location
    if best.exists():
        return selected.name, best                                                  # prefer best.pt when available because it is the expected final model

    pt_files = sorted(                                                              # if best.pt is missing, fall back to the most recently modified .pt file in the run folder
        selected.rglob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pt_files:
        ui.warn(f"No best.pt found — using: {pt_files[0].name}")
        return selected.name, pt_files[0]

    ui.missing_weights(selected)                                                    # no usable weights were found anywhere in the selected run directory
    return None


def _resolve_selected_model(model_arg: str, runs_dir: Path, ui: DetectUI) -> tuple[str, Path] | None:
    """
    Resolve a user-supplied model argument into '(model_name, weights_path)'.
    
    Supported forms:
        - a run directory name inside '/runs'
        - an explicit '.pt' file path
        - an official YOLO model name handled through ensure_weights()
    Returns None when the model cannot be resolved safely.
    """
    model_arg = str(model_arg).strip()                  # normalize user input before any checks
    if not model_arg:
        return None                                     # ignore empty model arguments entirely

    candidate_dir = runs_dir / model_arg                # first treat the argument like a run directory name
    if candidate_dir.is_dir():
        best = candidate_dir / "weights" / "best.pt"    # expected best checkpoint inside that run
        if best.exists():
            return candidate_dir.name, best
        pt_files = sorted(                              # when best.pt is missing, use the newest checkpoint in that run folder
            candidate_dir.rglob("*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if pt_files:
            ui.warn(f"No best.pt found for {candidate_dir.name} — using: {pt_files[0].name}")
            return candidate_dir.name, pt_files[0]
        ui.missing_weights(candidate_dir)               # run directory exists but contains no usable weights
        return None

    model_path = Path(model_arg)                        # next, interpret the argument as a direct filesystem path
    if model_path.suffix == ".pt":
        if not model_path.exists():
            ui.error(f"Model file does not exist: {fmt_bold(model_path)}")
            return None                                 # fail early when an explicit weights path is invalid

        return model_path.stem, model_path              # direct .pt file paths are accepted as-is
    placeholder = WEIGHTS_DIR / f"{model_arg}.pt"       # finally, interpret the argument as an official model name to be resolved in the weights folder
    resolved = ensure_weights(placeholder, model_arg)

    if resolved is None or not resolved.exists():
        ui.error(f"Could NOT resolve or download model '{model_arg}'.")
        return None                                                         # official model name also failed to resolve
    return resolved.stem, resolved                                          # successful official-model resolution

def _resolve_model_info(args: Any, runs_dir: Path, ui: DetectUI) -> list[ModelInfo]:
    """
    Resolve all requested models into validated ModelInfo objects.
    This function centralizes all model-selection behavior so later setup code can assume every returned model already has:
        - a valid weights file
        - a usable model name
        - a loaded class configuration
    """
    requested_models = list(getattr(args, "model", []) or [])   # normalize parsed CLI model argument into a plain list
    
    if not requested_models:
        resolved = _model_select_prompt(runs_dir, args, ui)     # no explicit model passed, use default flow
        if resolved is None:
            return []                                           # return empty list so caller can abort cleanly
        requested_models = [resolved]                           # wrap single resolved model into a list for uniform downstream logic
    else:
        explicit_specs: list[tuple[str, Path]] = []             # temporary list of validated explicit model specs
        for model_arg in requested_models:
            resolved = _resolve_selected_model(model_arg, runs_dir, ui)
            if resolved is None:
                ui.exit("Detection aborted due to invalid model selection.")
                return []                                                       # any invalid explicit selection aborts the detection request
            explicit_specs.append(resolved)
        requested_models = explicit_specs                                       # replace raw CLI values with resolved model tuples
    model_specs: list[ModelInfo] = []                                           # final list of validated model specifications

    for model_name, weights_path in requested_models:
        if not weights_path.exists() or weights_path.stat().st_size == 0:
            ui.error(f"Invalid weights file: {fmt_bold(weights_path)}")
            ui.exit("Detection aborted due to missing weights file.")
            return []                                                           # refuse to continue with missing or empty model file
        class_config = load_or_create_classes(                                  # load cached class metadata or derive it from the model if not already cached
            model_name=model_name,
            weights_path=weights_path,
            force_reload=False,
            ui=ui,
        )
        model_specs.append(                                                     # bundle validated model info into an immutable container for later setup steps
            ModelInfo(
                model_name=model_name,
                weights_path=Path(weights_path),
                class_config=class_config,
            )
        )
    return model_specs

def _register_source_with_ui(
    *,
    ui: DetectUI,
    idx: int,
    model_name: str,
    source_spec: SourceInfo,
    class_config: ClassConfig,
) -> None:
    """
    Register one source/model pair with the UI before processing begins.
    The UI needs both source identity & class ordering up front so it can build stable terminal rows before live processing starts.
    """
    full_display_name = f"{model_name} | {source_spec.short_name}"  # create a readable combined label so each processor line is unique in the terminal
    ui.register_source_identity(                                    # stores the display labels used in status updates & completion messages
        idx=idx,
        model_name=model_name,
        short_source_name=source_spec.short_name,
        full_display_name=full_display_name,
    )
    ui.register_source_classes(                                     # stores class order so count displays remain consistent for this processor
        idx=idx,
        classes_list=class_config.display_classes,
    )
    
def _build_processors(args: Any, model_specs: list[ModelInfo], ui: DetectUI) -> list[DetectInstance]:
    """
    Create & initialize one DetectInstance for every model/source pair.
    Each combination becomes its own independent processor so multiple sources & multiple models can run in parallel.
    """
    processors: list[DetectInstance] = []                           # holds only processors that initialized successfully
    idx = 1                                                         # UI/source index is 1-based for cleaner display

    for model_spec in model_specs:
        for raw_source in args.sources:
            source_spec = _parse_source_arg(raw_source, ui)         # normalize each requested source
            if source_spec is None:
                continue                                            # skip invalid sources without crashing the whole run

            _register_source_with_ui(                               # register processor identity before initialization so UI rows exist from the start
                ui=ui,
                idx=idx,
                model_name=model_spec.model_name,
                source_spec=source_spec,
                class_config=model_spec.class_config,
            )

            manager = DetectInstance(                               # build one manager responsible for this exact model/source pair
                weights_path=model_spec.weights_path,
                source_spec=source_spec,
                idx=idx,
                ui=ui,
                test=args.test,
                model_name=model_spec.model_name,
                class_config=model_spec.class_config,
            )
            if manager.initialize():
                processors.append(manager)                                                          # only keep processors that completed setup successfully
            else:
                ui.warn(f"Processor failed to initialize: {model_spec.model_name} | {raw_source}")  # warn instead of hard-exiting so other valid processors can still run
            idx += 1                                                                                # increment UI/source index for the next processor, ensuring each has a unique identifier
    return processors

def _initiate_threads(processors: list[DetectInstance]) -> list[threading.Thread]:
    """
    Launch one thread per initialized processor.
    Each processor owns its own source/model pair, so they can be run independently in daemon threads while the main thread just supervises overall lifecycle.
    """
    threads: list[threading.Thread] = []                                # stores started processor threads for later join/shutdown

    for processor in processors:
        thread = threading.Thread(target=processor.run, daemon=True)    # daemon=True ensures threads do not block interpreter shutdown in unexpected crash cases
        thread.start()                                                  # begin processing immediately
        threads.append(thread)                                          # retain handle so the main thread can monitor/join later
    return threads

# --------- CORE DETECTION INSTANCE ---------
class DetectInstance:
    """
    Manage the complete detection pipeline for one model/source pair.
    Main responsibilities include:
        - open & validate the source
        - resolve source metadata
        - prepare output paths
        - create video reader & inference worker
        - initialize measurement helpers
        - run the main frame-processing loop
        - shut everything down cleanly & save outputs
    """
    def __init__(
        self,
        *,
        weights_path: str | Path,
        source_spec: SourceInfo,
        idx: int,
        ui: DetectUI,
        test: bool = False,
        model_name: str | None = None,
        class_config: ClassConfig | None = None,
    ) -> None:
        """
        Store the core runtime state for one processor.
        Only lightweight attribute setup happens here; expensive setup like opening the source or building workers is deferred to initialize().
        """
        self.weights_path = Path(weights_path)                                                      # normalize weights path to Path for safe reuse
        self.model_name = str(model_name) if model_name is not None else self.weights_path.stem     # prefer provided model name so UI/output naming stays consistent with selection logic
        self.class_config = class_config                                                            # preloaded class configuration for this model
        self.source = source_spec.source                                                            # normalized source value: int camera index or file path str
        self.source_type = source_spec.source_type                                                  # source category string used in branching & paths
        self.is_camera = source_spec.is_camera                                                      # convenience boolean used throughout runtime logic
        self.idx = idx                                                                              # unique UI/display index for this processor
        self.ui = ui                                                                                # shared terminal UI manager
        self.test = test                                                                            # whether detection is running in test mode
        self.source_display_name = f"{self.model_name} | {source_spec.short_name}"                  # compact unique label used in warnings, UI rows, & writer setup messages
        self.frame_queue: queue.Queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)                       # reader pushes raw frames here & inference worker consumes them
        self.infer_queue: queue.Queue = queue.Queue(maxsize=INFER_QUEUE_SIZE)                       # inference worker pushes processed results here & main run loop consumes them
        self.cap = None                                                                             # cv2 capture object created during initialization
        self.out_writer = None                                                                      # cv2 video writer for annotated output
        
        self.reader: VideoReader | None = None                                                      # background frame reader thread wrapper
        self.infer_worker: InferenceWorker | None = None                                            # background inference thread wrapper
        
        self.config = MeasurementConfig()                                                           # shared measurement settings for helper objects
        self.tracker: Tracker | None = None                                                         # tracker used by counting, motion, & interactions
        self.counter: Counter | None = None                                                         # per-class counting helper
        self.aggregator: Aggregator | None = None                                                   # interval/session aggregation helper
        self.interactions: Interactions | None = None                                               # class interaction measurement helper
        self.motion: Motion | None = None                                                           # motion measurement helper
        
        self.start_time: datetime | None = None                                                     # session start timestamp for this source
        self.fps_video: float | None = None                                                         # resolved video FPS used for timing & ETA
        self.total_frames: int | None = None                                                        # resolved total frames for file sources when available
        self.frame_width: int | None = None                                                         # resolved frame width from runtime metadata
        self.frame_height: int | None = None                                                        # resolved frame height from runtime metadata
        
        self.paths: dict[str, Any] | None = None                                                    # output path collection returned by get_output_folder
        self.out_file: Path | None = None                                                           # final annotated video path
        self.metadata_file: Path | None = None                                                      # metadata JSON path

    def initialize(self) -> bool:
        """
        Build & validate everything needed before processing begins.

        Initialization includes:
            - opening the capture source
            - resolving runtime metadata
            - preparing output folders & metadata files
            - creating the video writer
            - preparing reader/inference workers
            - preparing measurement helpers

        Returns True only when the processor is fully ready to run.
        """
        self.cap = open_video_capture(                          # open the requested camera or video file as a cv2 capture source
            source=self.source,
            is_camera=self.is_camera,
            source_display_name=self.source_display_name,
            ui=self.ui,
        )
    
        if self.cap is None:
            return False                                        # fail immediately if source could not be opened

        runtime = resolve_source_runtime(                       # resolve FPS, size, frame count, timestamps, & metadata once up front
            cap=self.cap,
            source=self.source,
            is_camera=self.is_camera,
            source_display_name=self.source_display_name,
            ui=self.ui,
        )
        
        if runtime is None:
            return False                                            # fail if source metadata could not be determined safely

        self._resolve_metadata(runtime)                             # copy resolved metadata onto instance attributes
        self._resolve_output_paths(runtime)                         # build all output file/folder paths for this session
        write_metadata_json(self.metadata_file, runtime.metadata)   # persist source metadata immediately

        if not self._build_video_writer():
            return False                                            # fail if annotated output writer could not be created

        self._prepare_workers()                                     # create reader & inference workers after core runtime is known
        self._build_measurement_workers()                           # create helper objects that consume tracking/detection data

        return True                                                 # processor is fully initialized & ready to run

    def _resolve_metadata(self, runtime: SourceRuntime) -> None:
        """
        Copy resolved runtime metadata onto this instance.
        Keeping these values as direct attributes avoids repeatedly reaching into the runtime object during the hot path of frame processing.
        """
        self.start_time = runtime.start_time                            # base timestamp used to derive per-frame video time
        self.fps_video = runtime.fps                                    # FPS used for timestamps, UI, & output writer
        self.total_frames = runtime.total_frames                        # total frames if known, mainly useful for video files
        self.frame_width = runtime.frame_width                          # width used to configure writer & motion helper
        self.frame_height = runtime.frame_height                        # height used to configure writer & motion helper

    def _resolve_output_paths(self, runtime: SourceRuntime) -> None:
        """
        Create all output paths for this processor.
        Output folders are separated by model/source/session so results remain organized & do not overwrite prior runs.
        """
        source_name = self.source if not self.is_camera else f"usb{self.source}"    # camera sources are converted back to readable labels like 'usb0' for folder naming
        self.paths = get_output_folder(                                             # central helper returns all needed folders/files for this detection run
            self.weights_path,
            self.source_type,
            source_name,
            test_detect=self.test,
            base_time=runtime.start_time,
        )
        
        self.out_file = self.paths["video_folder"] / f"{self.paths['safe_name']}.mp4"   # final annotated video file path for this processor
        self.metadata_file = self.paths["metadata"]                                     # metadata JSON path stored alongside the rest of the run outputs
        
    def _build_video_writer(self) -> bool:
        """
        Create the annotated output video writer.
        Returns True when the writer is ready, otherwise False.
        """
        self.out_writer = create_video_writer(      # writer setup depends on resolved FPS, frame size, source type, & destination path
            self.out_file,
            self.fps_video,
            self.frame_width,
            self.frame_height,
            self.source_display_name,
            self.ui,
            self.cap,
            self.source_type,
        )
        

        return self.out_writer is not None          # simple readiness check for caller

    def _prepare_workers(self) -> None:
        """
        Create the inference worker & frame reader.
        The inference worker is started immediately so it is ready to consume frames as soon as the reader begins pushing them.
        """
        self.infer_worker = InferenceWorker(                    # inference worker owns the loaded model & turns frames into annotated detections
            weights_path=self.weights_path,
            frame_queue=self.frame_queue,
            infer_queue=self.infer_queue,
            is_camera=self.is_camera,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            source_display_name=self.source_display_name,
            global_stop_event=stop_event,
            ui=self.ui,
            imgsz=DEFAULT_IMGSZ,
        )
        

        self.infer_worker.start()                               # start inference thread before reader begins feeding frames

        self.reader = VideoReader(                              # reader is created after inference worker so frame production has an active consumer
            cap=self.cap,
            frame_queue=self.frame_queue,
            infer_queue=self.infer_queue,
            is_camera=self.is_camera,
            source_display_name=self.source_display_name,
            global_stop_event=stop_event,
            ui=self.ui,
        )
        
    def _build_measurement_workers(self) -> None:
        """
        Create all measurement helpers that consume tracked detections.
        These helpers are split by responsibility so counting, aggregation, motion, & interactions stay modular & easier to maintain.
        """
        self.tracker = Tracker(                         # tracker maintains stable object identities across frames for downstream measurements
            config=self.config,
            focus_classes=self.class_config.focus,
            context_classes=self.class_config.context,
        )
        self.counter = Counter(                         # counter records tracked counts over time & writes per-class measurement outputs
            out_folder=self.paths["counts"],
            class_config=self.class_config,
            config=self.config,
            start_time=self.start_time,
            tracker=self.tracker,
        ) 
        self.aggregator = Aggregator(                   # aggregator turns frame-level counts into interval & whole-session summaries
            out_folder=self.paths["counts"],
            class_config=self.class_config,
            config=self.config,
            start_time=self.start_time,
        )
        self.interactions = Interactions(               # interactions measures contact/overlap-style relationships between tracked detections 
            out_folder=self.paths["interactions"],
            class_config=self.class_config,
            config=self.config,
            start_time=self.start_time,
            is_obb=self.infer_worker.is_obb,
            tracker=self.tracker,
        )
        self.motion = Motion(                           # motion helper uses tracked positions across frames to compute movement-based metrics
            paths=self.paths,
            class_config=self.class_config,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            config=self.config,
            start_time=self.start_time,
            tracker=self.tracker,
        )
        

    def run(self) -> None:
        """
        Run the main frame-processing loop for this processor.

        Processing flow per frame:
            - pull next inference result
            - stop on EOF
            - derive video timestamp
            - compute counts from detections
            - update tracker
            - feed all measurement helpers
            - refresh UI status
            - write annotated frame
        """
        frame_count = 0                                 # total processed frames for this processor
        prev_time = time.time()                         # last timestamp used in UI FPS smoothing
        loop_start = time.time()                        # wall-clock start time for elapsed/ETA calculations
        self.reader.start()                             # begin pulling frames into the queue once everything is ready

        try:
            while not stop_event.is_set():
                item = self._pull_inference_result()    # fetch next processed result from inference worker
                if item is None:
                    continue                            # queue timed out briefly, so keep looping unless a stop is triggered
                if is_eof_marker(item):
                    break                               # reader/inference signaled end-of-stream, so exit cleanly
                annotated, names, boxes_list = item     # inference worker returns:
                                                            #   annotated  -> frame with drawn boxes/labels
                                                            #   names      -> class names for detections
                                                            #   boxes_list -> raw detection boxes
                video_ts = self.start_time + timedelta(seconds=frame_count / self.fps_video) # derive the logical timestamp for this frame within the source timeline
                counts = compute_counts_from_boxes(                             # produce a frame-level count dictionary that matches the model's class configuration
                    boxes_list,
                    names,
                    focus_classes=self.class_config.focus,
                    context_classes=self.class_config.context,
                )
                tracks_by_class = self.tracker.update(                          # update object tracks using only classes relevant to this model's configured analysis
                    boxes_list,
                    names,
                    allow_classes=[*self.class_config.focus, *self.class_config.context],
                )
                self.counter.update_counts(tracks_by_class, video_ts)           # record tracked counts at this timestamp
                self.aggregator.push_frame_data(video_ts, counts_dict=counts)   # feed frame-level counts into interval/session aggregation
                self.interactions.process_tracks(                               # detect & record interaction events for this frame
                    tracks_by_class=tracks_by_class,
                    boxes=boxes_list,
                    names=names,
                    ts=video_ts,
                )
                self.motion.process_tracks(tracks_by_class, video_ts)   # update movement measurements from current tracked positions
                frame_count, prev_time = self._update_ui_status(        # refresh terminal row at a throttled cadence & carry updated counters back
                    frame_count=frame_count,
                    prev_time=prev_time,
                    loop_start=loop_start,
                    counts=counts,
                )
                write_annotated_frame(self.out_writer, annotated)       # persist the annotated frame to the output video
        finally:
            self._shutdown()                                            # always clean up, even if an exception or interrupt occurs

    def _pull_inference_result(self) -> Any | None:
        """
        Pull the next inference result while respecting global stop requests.

        Returns:
            - an inference result tuple when available
            - EOF marker tuple when a stop was requested during queue wait
            - None on normal short timeout so the run loop can stay responsive
        """
        try:
            return self.infer_queue.get(timeout=0.1)            # short timeout keeps the loop responsive to stop_event instead of blocking indefinitely
            
        except queue.Empty:
            if stop_event.is_set():
                return (EOF_MARKER, None)                       # synthesize an EOF-like signal when global stop is active
            return None                                         # no result yet, but not an error

    def _update_ui_status(                                      
        self,
        *,
        frame_count: int,
        prev_time: float,
        loop_start: float,
        counts: dict[str, int],
    ) -> tuple[int, float]:
        """
        UI helper computes smoothed FPS, elapsed time string, updated timing state, & ETA that refreshes terminal status for a processor on a throttled cadence.
        Returns:
            updated frame_count & updated prev_time for reuse in the next loop iteration
        """
        fps_smooth, time_str, prev_time, eta = self.ui.format_time_fps(
            frame_count,
            prev_time,
            loop_start,
            fps_video=self.fps_video,
            total_frames=self.total_frames,
            source_type=self.source_type,
            source_idx=self.idx,
        )
        frame_count += 1                                    # increment after successful processing of the current frame
        if frame_count % 5 == 0:                            # update terminal row every 5 frames to reduce flicker & terminal overhead
            self.ui.update_frame_status(
                idx=self.idx,
                display_name=self.source_display_name,
                frame_count=frame_count,
                fps_smooth=fps_smooth,
                counts=counts,
                time_str=time_str,
                eta=eta,
            )
        return frame_count, prev_time                       # feed updated state back into the main loop

    def _shutdown(self) -> None:
        """
        Stop workers, release resources, finalize measurement outputs, & log completion.
        Note that the shutdown order matters here:
            - stop worker loops
            - join threads
            - drain queues
            - release file/capture resources
            - finalize & save measurement outputs
            - update the UI
        """
        if self.reader:
            self.reader.stop()                                                                      # request frame reader shutdown first so no new frames are produced

        if self.infer_worker:
            self.infer_worker.stop()                                                                # request inference worker shutdown next

        if self.reader:
            self.reader.join(timeout=1.0)                                                           # give the reader a moment to finish since it may be mid-read when stopped
            
        drain_queue(self.frame_queue)                                                               # remove any leftover unread frames so shutdown is not blocked by stale queue contents
        
        if self.infer_worker:
            self.infer_worker.join(timeout=2.0)                                                     # give the inference worker a bit more time to finish since it may be mid-processing when stopped
            
        release_quietly(self.out_writer)                                                            # safely close annotated output video
        release_quietly(self.cap)                                                                   # safely release capture source

        saved: list[Path] = [self.out_file, self.metadata_file]                                     # start the saved files list with the most important outputs that should always be present, even if measurements fail or are skipped
        safe_extend(saved, self.counter.save_results())                                             # save counting results next since they are usually the most critical measurement output for users, so they should be included in the saved list even if aggregation or interactions fail

        self.aggregator.finalize()                                                                  # finalize aggregation before interactions since it may rely on interval boundaries that should be set before interaction processing runs

        safe_extend(saved, self.aggregator.save_interval_results())                                 # save interval-level aggregation outputs, which are usually more critical than the session summary, so they should be included in the saved list even if the final summary fails
        safe_extend(saved, self.aggregator.save_session_summary())                                  # save session summary after interactions since it may rely on interaction-derived metrics that should be included in the summary if possible

        self.interactions.finalize()                                                                # finalize interactions before saving since it may set important state like interaction end times that should be reflected in the saved outputs
        safe_extend(saved, self.interactions.save_results())                                        # save interactions after finalization since it may rely on finalized interaction state to produce correct outputs

        safe_extend(saved, self.motion.save_results())                                              # motion is usually the least critical output, so it is saved last and only included in the saved list if it succeeds, since motion measurement may be less important to some users and more likely to fail on edge-case sources where tracking is unstable

        self.ui.mark_source_complete(self.idx)                                                      # update UI status to show this source is done after all processing is finished but before measurements are saved since saving can take a moment & it's good to show completion as soon as the main processing is done, even if measurements are still being finalized/saved
        self.ui.save_measurements(self.paths["scores_folder"], saved, model_name=self.model_name)   # trigger UI to copy saved measurement outputs to the central scores folder for easier user access, passing the model name for better organization of multi-model runs
        
# --------- MAIN ENTRY POINT ---------
def main() -> None:
    """
    CLI entry point for detection.
    The order of operations mostly goes as follows:
        - clear prior stop state
        - parse arguments
        - resolve models
        - initialize UI
        - build processors
        - launch one thread per processor
        - supervise until completion or Ctrl+C
    """
    stop_event.clear()                                                      # ensure no stale stop flag leaks in from a prior invocation

    args = parse_arguments()                                                # parse CLI flags, sources, model selections, & test mode
    runs_dir = get_runs_dir(test=args.test)                                 # resolve which runs directory should be searched

    bootstrap_ui = DetectUI(total_sources=0)                                # lightweight temporary UI used during model resolution before final source count is known
    model_specs = _resolve_model_info(args, runs_dir, bootstrap_ui)
    if not model_specs:
        sys.exit(1)                                                         # abort immediately if no usable models could be resolved

    total_processors = len(model_specs) * len(args.sources)                 # one processor is created for every model/source pair
    ui = DetectUI(total_sources=total_processors)                           # build final UI only after total processor count is known

    for model_spec in model_specs:
        ui.model_classes_loaded(model_spec.model_name, model_spec.class_config.display_classes)     # announce loaded classes for each selected model before processing begins
    
    processors = _build_processors(args, model_specs, ui)                                           # initialize all valid model/source processors
    threads = _initiate_threads(processors)                                                         # launch one thread per initialized processor
    try:
        while any(thread.is_alive() for thread in threads):                                         # keep the main thread lightweight while child processor threads do the real work
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        ui.stop_signal_received(single_thread=(len(threads) == 1))      # show a cleaner interrupt message tailored to single vs multi-thread runs
        stop_event.set()                                                # signal all processors & workers to stop as soon as possible
        
    finally:
        for thread in threads:
            while thread.is_alive():
                try:
                    thread.join(timeout=0.5)                            # keep joining in short intervals so repeated Ctrl+C does not break cleanup
                except KeyboardInterrupt:                               # ignore further interrupts during shutdown & keep joining cleanly
                    continue  
        ui.all_threads_terminated()                                     # final UI message confirming all processor threads have exited
        
if __name__ == "__main__":                                              # run CLI entry point only when file is executed directly
    main() 