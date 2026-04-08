# utils/ui.py
from __future__ import annotations                  # allows forward references in type hints without quoting types
import logging                                      # used for filtering & patching noisy Ultralytics log output
import os                                           # used for terminal clearing, size detection, & platform shell calls
import re                                           # used for log filtering patterns & safe writer-name normalization
import threading                                    # used to protect shared UI state during concurrent updates
import time                                         # used for redraw throttling & smoothed FPS timing
from datetime import datetime                       # used for writer registration timestamps
from pathlib import Path                            # used for safe cross-platform path formatting
from ultralytics.utils import LOGGER                # Ultralytics logger patched to suppress noisy runtime output

# --------- COLORS ---------
RESET = "\033[0m"                                   # reset all ANSI formatting
BOLD = "\033[1m"                                    # bold ANSI style
DIM = "\033[2m"                                     # dim ANSI style
ITALIC = "\033[3m"                                  # italic ANSI style

RED = "\033[31m"                                    # standard red
BRIGHT_RED = "\033[91m"                             # brighter red used for exits/errors

GREEN = "\033[32m"                                  # standard green
BRIGHT_GREEN = "\033[92m"                           # brighter green used for save messages

YELLOW = "\033[33m"                                 # standard yellow
BRIGHT_YELLOW = "\033[93m"                          # brighter yellow used for model labels

BLUE = "\033[34m"                                   # standard blue
BRIGHT_BLUE = "\033[94m"                            # brighter blue
BLUE_006B = "\033[38;5;33m"                         # custom blue tone used for source labels

MAGENTA = "\033[35m"                                # standard magenta
BRIGHT_MAGENTA = "\033[95m"                         # brighter magenta used for paths

CYAN = "\033[36m"                                   # standard cyan
BRIGHT_CYAN = "\033[96m"                            # brighter cyan used for info labels

WHITE = "\033[37m"                                  # standard white for main text

# --------- SHARED FORMATTING HELPERS ---------
def fmt_info(msg): return f"{BRIGHT_CYAN}{BOLD}info{RESET}: {WHITE}{msg}{RESET}"           # formatted info line
def fmt_model(msg): return f"{BRIGHT_YELLOW}{BOLD}model{RESET}: {WHITE}{msg}{RESET}"       # formatted model line
def fmt_dataset(msg): return f"{BLUE}{BOLD}dataset{RESET}: {WHITE}{msg}{RESET}"             # formatted dataset line
def fmt_train(msg): return f"{GREEN}{BOLD}train{RESET}: {WHITE}{msg}{RESET}"                # formatted train line
def fmt_exit(msg): return f"{BRIGHT_RED}{BOLD}exit{RESET}: {WHITE}{msg}{RESET}"             # formatted exit line
def fmt_warn(msg): return f"{YELLOW}{BOLD}warn{RESET}: {WHITE}{msg}{RESET}"                 # formatted warning line
def fmt_error(msg): return f"{RED}{BOLD}error{RESET}: {WHITE}{msg}{RESET}"                  # formatted error line
def fmt_save(msg): return f"{BRIGHT_GREEN}{BOLD}save{RESET}: {WHITE}{msg}{RESET}"           # formatted save line
def fmt_source(txt): return f"{BLUE_006B}{BOLD}{txt}{RESET}"                                # formatted source label
def fmt_label(txt): return f"{WHITE}{BOLD}{txt}{RESET}"                                     # formatted inline label
def fmt_header(txt): return f"{WHITE}{BOLD}{txt}{RESET}"                                    # formatted section/header text
def fmt_num(n): return f"{WHITE}{BOLD}{n}{RESET}"                                           # formatted numeric display
def fmt_path(path: str | Path) -> str: return f"{BRIGHT_MAGENTA}{BOLD}{path}{RESET}"        # formatted path string
def fmt_bold(txt: str | Path) -> str: return f"{WHITE}{BOLD}{txt}{RESET}"                   # generic bold formatter

# --------- SHARED TERMINAL HELPERS ---------
def clear_terminal() -> None:
    os.system("cls" if os.name == "nt" else "clear")   # use the platform-appropriate terminal clear command

def print_divider(width: int = 120) -> None:
    print("-" * width)                                 # shared solid divider used by training output blocks

def print_static_separator(width: int = 120) -> None:
    print("\n" + "-" * width + "\n")                   # inserts visual separation between static terminal sections

# --------- ULTRALYTICS / WANDB LOG FILTERING ---------
class UltralyticsFilter(logging.Filter):
    noise_patterns = [
        r"Ultralytics.*Python",                        # startup banner noise
        r"New .* available",                          # update notices
        r"engine/trainer",                            # trainer-path chatter
        r"Transferred .* items",                      # transfer summary noise
        r"optimizer:",                                # optimizer details
        r"anchors:",                                  # anchor summary lines
        r"Scanning .*labels",                         # label scan spam
        r"duplicate labels removed",                  # dataset cleanup notes
        r"ignoring corrupt",                          # corrupt-file notices
        r"Freezing layer",                            # layer freezing chatter
        r"summary:",                                  # model summary noise
        r"Layers:",                                   # layer count output
        r"Model.*parameters",                         # parameter-count lines
        r"Using 0 dataloader workers",                # worker count chatter
        r"Image sizes",                               # image size summary
        r"Creating cache",                            # cache generation notices
        r"YOLO.*layers",                              # architecture dump noise
    ]

    def filter(self, record):
        msg = record.getMessage()                     # read the current log message text from the record
        return not any(re.search(pattern, msg) for pattern in self.noise_patterns)  # suppress lines matching known noisy patterns

def quiet_ultralytics_logs() -> None:
    logger = logging.getLogger("ultralytics")        # target the named Ultralytics logger directly
    logger.setLevel(logging.INFO)                    # keep informational severity while still applying the custom filter
    logger.addFilter(UltralyticsFilter())            # attach the noise filter so repeated startup/runtime chatter is hidden

def quiet_wandb_logs() -> None:
    pass                                             # reserved hook for future W&B suppression without changing caller code later

_original_ultralytics_info = LOGGER.info             # preserve the original logger method so patching can be reversed cleanly

def patched_ultralytics_info(msg, *args, **kwargs):
    if getattr(patched_ultralytics_info, "interrupted", False):
        return                                       # once interrupted, stop forwarding ordinary Ultralytics info messages

    lower = msg.lower()

    if "exit" in lower or "interrupt" in lower or "error" in lower:
        return _original_ultralytics_info(msg, *args, **kwargs)  # always allow serious/termination-related lines through

    if msg.strip() == "-" * 120:
        return                                       # skip bare divider lines so the UI controls separators itself

    suppress = (
        "Overriding model.yaml" in msg
        or "Fast image access" in msg
        or "ultralytics.nn" in msg
        or "Conv " in msg
        or "C2" in msg
        or "C3" in msg
        or "SPPF" in msg
        or "Concat" in msg
        or "Upsample" in msg
        or "parameters" in msg
        or msg.strip().startswith("from")
    )                                                # suppress noisy architecture/configuration chatter that clutters the terminal

    if suppress:
        return

    return _original_ultralytics_info(msg, *args, **kwargs)  # forward all remaining acceptable lines to the original logger

def apply_ultralytics_patch() -> None:
    patched_ultralytics_info.interrupted = False     # reset patch state before reapplying it
    LOGGER.info = patched_ultralytics_info           # replace Ultralytics info logger with the patched filtering wrapper

def remove_ultralytics_patch() -> None:
    LOGGER.info = _original_ultralytics_info         # restore the original Ultralytics logger method

# --------- TRAINING UI ---------
class TrainingUI:
    """
    UI wrapper for the training side of the pipeline.
    """
    def info(self, msg: str) -> None:
        print(fmt_info(msg))                         # print a formatted informational training message

    def warn(self, msg: str) -> None:
        print(fmt_warn(msg))                         # print a formatted warning message

    def error(self, msg: str) -> None:
        print(fmt_error(msg))                        # print a formatted error message

    def exit(self, msg: str) -> None:
        print(fmt_exit(msg))                         # print a formatted exit/status message

    def model(self, msg: str) -> None:
        print(fmt_model(msg))                        # print a formatted model-related message

    def dataset(self, msg: str) -> None:
        print(fmt_dataset(msg))                      # print a formatted dataset-related message

    def train(self, msg: str) -> None:
        print(fmt_train(msg))                        # print a formatted training-progress message

    def show_training_header(self) -> None:
        clear_terminal()                             # clear the terminal so training starts with a clean screen
        print(fmt_bold("Training"))
        print("----------------\n")

    def show_training_header_static(self) -> None:
        print_static_separator()                     # print a standalone separator when a full terminal clear is not needed

    def show_training_context(
        self,
        *,
        model_source: str | Path,
        dataset_name: str,
        batch_size: int,
        workers: int,
        epochs: int,
    ) -> None:
        print(fmt_model(f"Model initializing: {fmt_path(model_source)}"))  # show the source model/weights being used
        print(fmt_dataset(fmt_bold(dataset_name)))                         # show dataset name in a compact bold format
        print(
            fmt_train(
                f"{WHITE}{BOLD}Epochs -{RESET} {epochs}    "
                f"{WHITE}{BOLD}Batch -{RESET} {batch_size}    "
                f"{WHITE}{BOLD}Workers -{RESET} {workers}"
            )
        )                                              # print one compact training-context line for key runtime settings
        print()
        print_divider()
        print()

    def configure_external_logs(self) -> None:
        quiet_ultralytics_logs()                       # apply standard Ultralytics logging suppression
        quiet_wandb_logs()                            # apply W&B suppression hook when implemented

    def patch_ultralytics_output(self) -> None:
        apply_ultralytics_patch()                     # patch Ultralytics info output for cleaner terminal display

    def unpatch_ultralytics_output(self) -> None:
        remove_ultralytics_patch()                    # restore original Ultralytics logging behavior

# --------- DETECTION UI ---------
class DetectUI:
    """
    UI wrapper for the detection side of the pipeline.

    Multi-model behavior:
        - sources grouped by model
        - source panes show short source names
        - class lists are tracked per source
        - final save sections grouped by model
    """
    def __init__(self, total_sources: int = 0):
        self.total_sources = total_sources                         # total number of source panes the UI is expected to manage
        self.lock = threading.Lock()                               # protects shared UI state across multiple detection threads
        self.sources = [
            {
                "name": None,                                      # full display name such as "model | source"
                "short_name": None,                                # short source-only display label
                "model_name": None,                                # owning model label for source grouping
                "frame_count": 0,                                  # latest processed frame count
                "fps": 0.0,                                        # latest smoothed FPS value
                "time_str": "--:--",                               # elapsed/progress time display string
                "counts": {},                                      # latest class count snapshot for this source
                "completed": False,                                # source completion state used to freeze display values
                "source_type": None,                               # "usb" or "video" once known
                "eta": "--",                                       # ETA display for finite video sources
                "classes": None,                                   # ordered display classes for this specific source
            }
            for _ in range(total_sources)
        ]

        self.log_lines: list[str] = []                             # rolling log lines rendered beneath the source panes
        cols, rows = self._get_term_size()                         # snapshot terminal size at init so max log count can be estimated
        self.max_logs = max(15, rows // 2)                         # reserve a sensible lower half of the terminal for rolling logs

        self._fps_smooth: dict[int, float] = {}                    # per-source smoothed FPS cache keyed by source index
        self.last_redraw_time = 0.0                                # timestamp of the last full UI redraw
        self.redraw_interval = 0.15                                # throttle redraw frequency to reduce flicker & terminal overhead

        self.freeze_ui = False                                     # when True, live redraws are temporarily suppressed
        self.in_shutdown = False                                   # when True, the UI transitions into shutdown/save mode
        self.final_exit_events: list[str] = []                     # final exit messages rendered in the shutdown block
        self.final_save_blocks_by_model: dict[str, list[list[str]]] = {}  # grouped save summaries keyed by model name

        self._recording_initialized_once = False                   # ensures recording initialization is only logged once
        self._classes_logged_for_model: set[str] = set()           # avoids duplicate per-model class-load messages
        self.all_classes: list[str] = []                           # fallback global class ordering when a source-specific one is absent

        self.active_writers = {}                                   # tracks registered output writers so they can be safely released later

    # --------- BASIC LOGGING ---------
    def info(self, msg: str) -> None:
        self._append_log(fmt_info(msg))                            # add a formatted info line to the rolling log buffer

    def warn(self, msg: str) -> None:
        self._append_log(fmt_warn(msg))                            # add a formatted warning line to the rolling log buffer

    def error(self, msg: str) -> None:
        self._append_log(fmt_error(msg))                           # add a formatted error line to the rolling log buffer

    def exit(self, msg: str) -> None:
        self._append_log(fmt_exit(msg))                            # add a formatted exit/status line to the rolling log buffer

    def save(self, msg: str) -> None:
        self._append_log(fmt_save(msg))                            # add a formatted save line to the rolling log buffer

    # --------- INTERNAL HELPERS ---------
    def _get_term_size(self):
        try:
            size = os.get_terminal_size()
            return size.columns, size.lines                        # prefer the live terminal size when available
        except OSError:
            return 120, 40                                         # safe fallback size for nonstandard terminals or redirected output

    def _append_log(self, formatted: str) -> None:
        with self.lock:
            self.log_lines.append(formatted)                       # append the newest formatted line to the rolling log buffer
            if len(self.log_lines) > self.max_logs:
                self.log_lines = self.log_lines[-self.max_logs:]   # keep only the newest visible portion of the log buffer
        self._maybe_redraw()

    def _extract_model_and_short(self, display_name: str):
        if not display_name:
            return None, None                                      # missing display names cannot be split meaningfully
        if " | " in display_name:
            model_name, short_name = display_name.split(" | ", 1)  # split full labels into grouped model/source pieces
            return model_name.strip() or None, short_name.strip() or None
        return None, display_name.strip()                          # plain labels are treated as source-only names

    def _group_sources_by_model(self):
        order = []                                                 # preserves first-seen model grouping order for display stability
        buckets = {}

        for src in self.sources:
            model_name = src.get("model_name")

            if not model_name and src.get("name"):
                parsed_model, parsed_short = self._extract_model_and_short(src["name"])  # backfill grouped names from full display strings if needed
                if parsed_model and not src.get("model_name"):
                    src["model_name"] = parsed_model
                if parsed_short and not src.get("short_name"):
                    src["short_name"] = parsed_short
                model_name = src.get("model_name")

            model_name = model_name or "unknown"                   # sources without a resolved model still need a display bucket
            if model_name not in buckets:
                buckets[model_name] = []
                order.append(model_name)
            buckets[model_name].append(src)

        return [(model_name, buckets[model_name]) for model_name in order]  # return grouped sources in stable first-seen order

    def _format_class_lines(self, counts: dict, width: int, classes_list=None):
        display_entries = []                                       # flat formatted class/value entries before wrapping into columns
        is_placeholder = bool(counts) and all(str(value) == "--" for value in counts.values())  # completed sources display placeholder values

        if classes_list:
            ordered = list(classes_list)                           # prefer source-specific class ordering when available
        elif self.all_classes:
            ordered = list(self.all_classes)                       # otherwise fall back to the shared class ordering
        else:
            ordered = list(counts.keys())                          # final fallback uses whatever order exists in the counts mapping

        if not ordered:
            return ["  (no detections yet)"]                       # render a friendly placeholder before any class structure is known

        for cls in ordered:
            value = "--" if is_placeholder else counts.get(cls, 0) # completed panes show placeholders instead of frozen numeric values
            display_entries.append(f"{fmt_label(cls)}: {value}")

        count = len(display_entries)
        if count <= 5:
            cols = 1                                               # few classes are most readable in a single column
        elif count <= 8:
            cols = 2
        elif count <= 15:
            cols = 3 if width > 90 else 2                          # medium class counts use width-aware wrapping
        else:
            cols = max(1, width // 18)                             # dense class layouts scale to terminal width automatically

        max_len = max(len(entry) for entry in display_entries) + 4 # padded width keeps columns visually aligned

        lines = []
        for start in range(0, count, cols):
            row = display_entries[start : start + cols]
            row = [entry.ljust(max_len) for entry in row]          # pad each entry so multi-column rows line up cleanly
            lines.append("  " + "".join(row).rstrip())

        return lines

    def _redraw_locked(self) -> None:
        width, _ = self._get_term_size()

        print("\033[3J", end="")                                   # clear scrollback-visible terminal region for cleaner redraws
        clear_terminal()
        print("\033[H", end="")                                    # return cursor to home position before repainting the UI

        print(fmt_bold("Detection"))
        print("-" * width)
        print()

        grouped = self._group_sources_by_model()

        for model_name, srcs in grouped:
            print(fmt_model(model_name))
            print()

            for src in srcs:
                display_name = src.get("short_name")
                if not display_name:
                    _, display_name = self._extract_model_and_short(src.get("name") or "")
                display_name = display_name or "source"            # final fallback label keeps every pane printable

                name_fmt = fmt_source(display_name)

                if src["completed"]:
                    header = (
                        f"{name_fmt}: "
                        f"{fmt_label('Frames')}: -- | "
                        f"{fmt_label('FPS')}: -- | "
                        f"{fmt_label('Time')}: -- | "
                        f"{fmt_label('ETA')}: --"
                    )                                              # completed sources show a final static placeholder header
                else:
                    eta_display = src.get("eta", "--")
                    show_eta = src.get("source_type") == "video"   # ETA is only meaningful for finite video-file sources

                    if show_eta:
                        header = (
                            f"{name_fmt}: "
                            f"{fmt_label('Frames')}: {src['frame_count']} | "
                            f"{fmt_label('FPS')}: {src['fps']:.1f} | "
                            f"{fmt_label('Time')}: {src['time_str']} | "
                            f"{fmt_label('ETA')}: {WHITE}{eta_display}{RESET}"
                        )
                    else:
                        header = (
                            f"{name_fmt}: "
                            f"{fmt_label('Frames')}: {src['frame_count']} | "
                            f"{fmt_label('FPS')}: {src['fps']:.1f} | "
                            f"{fmt_label('Time')}: {src['time_str']}"
                        )

                print(header)

                for line in self._format_class_lines(src["counts"], width, classes_list=src.get("classes")):
                    print(line)                                     # render the class-count grid beneath the source header
                print()

            print("-" * width)
            print()

        for line in self.log_lines[-self.max_logs:]:
            print(line[:width])                                     # clip rolling log lines to the current terminal width

        print("", flush=True)

    def _maybe_redraw(self) -> None:
        if self.freeze_ui or self.in_shutdown:
            return                                                  # suppress live redraws during model prompts or shutdown output

        with self.lock:
            now = time.time()
            if now - self.last_redraw_time < self.redraw_interval:
                return                                              # throttle redraw frequency to reduce flicker & CPU churn
            self.last_redraw_time = now
            self._redraw_locked()

    # --------- SOURCE / CLASS REGISTRATION ---------
    def register_source_identity(
        self,
        *,
        idx: int,
        model_name: str,
        short_source_name: str,
        full_display_name: str | None = None,
    ) -> None:
        i = idx - 1
        if i < 0 or i >= self.total_sources:
            return                                                  # ignore out-of-range source registrations safely

        with self.lock:
            src = self.sources[i]
            src["model_name"] = model_name                          # record owning model name for later grouping
            src["short_name"] = short_source_name                   # store compact source label for pane headers
            src["name"] = full_display_name or f"{model_name} | {short_source_name}"  # always keep a full display label available

        self._maybe_redraw()

    def register_source_classes(self, *, idx: int, classes_list) -> None:
        if not classes_list:
            return                                                  # no class order means nothing should be stored

        i = idx - 1
        if i < 0 or i >= self.total_sources:
            return                                                  # ignore invalid source indices safely

        with self.lock:
            self.sources[i]["classes"] = list(classes_list)         # preserve source-specific class ordering for count displays

    def model_classes_loaded(self, model_name: str, classes_list) -> None:
        if not model_name or not classes_list:
            return

        key = str(model_name)
        if key in self._classes_logged_for_model:
            return                                                  # avoid duplicate class-loaded log messages for the same model

        self._classes_logged_for_model.add(key)
        self.info(f"Loaded {len(classes_list)} classes: {classes_list}")

    def classes_loaded(self, classes_list) -> None:
        if classes_list is None:
            return
        self.all_classes = list(classes_list)                       # store shared class ordering for displays lacking source-specific order

    # --------- FPS / TIME HELPERS ---------
    def _format_time_str(
        self,
        *,
        frame_count: int,
        prev_time: float,
        start_time: float,
        fps_video: float = None,
        total_frames: int = None,
        source_type: str = "video",
        source_idx: int = 0,
    ):
        now = time.time()
        instantaneous = 1.0 / (now - prev_time + 1e-6)              # estimate instantaneous FPS from time since last update
        instantaneous = min(instantaneous, 60.0)                    # clamp unrealistic spikes for more stable display behavior

        prev_smooth = self._fps_smooth.get(source_idx, instantaneous)
        fps_smooth = 0.9 * prev_smooth + 0.1 * instantaneous        # exponential smoothing makes FPS readout less jittery
        fps_smooth = min(fps_smooth, 60.0)
        self._fps_smooth[source_idx] = fps_smooth

        eta_str = None
        if source_type == "video" and fps_video and total_frames:
            elapsed = (frame_count + 1) / float(fps_video)          # convert processed frames into source-time progress for file inputs
            total = total_frames / float(fps_video)
            remaining = max(0.0, total - elapsed)

            e_m, e_s = divmod(int(elapsed), 60)
            t_m, t_s = divmod(int(total), 60)
            r_m, r_s = divmod(int(remaining), 60)

            time_str = f"{e_m:02d}:{e_s:02d}/{t_m:02d}:{t_s:02d}"   # file sources show elapsed/total progress
            eta_str = f"{r_m:02d}:{r_s:02d}"
        else:
            elapsed = int(now - start_time)                          # live sources fall back to wall-clock runtime only
            e_m, e_s = divmod(elapsed, 60)
            time_str = f"{e_m:02d}:{e_s:02d}"

        return fps_smooth, time_str, now, eta_str

    def format_time_fps(
        self,
        frame_count,
        prev_time,
        start_time,
        fps_video=None,
        total_frames=None,
        source_type="video",
        source_idx=None,
    ):
        if source_idx is None:
            source_idx = 0                                           # fallback source index ensures FPS smoothing always has a key

        return self._format_time_str(
            frame_count=frame_count,
            prev_time=prev_time,
            start_time=start_time,
            fps_video=fps_video,
            total_frames=total_frames,
            source_type=source_type,
            source_idx=source_idx,
        )

    # --------- LIVE STATUS ---------
    def update_frame_status(
        self,
        *,
        idx,
        display_name,
        frame_count,
        fps_smooth,
        counts,
        time_str,
        eta=None,
    ) -> None:
        i = idx - 1
        if i < 0 or i >= self.total_sources:
            return                                                  # ignore invalid status updates safely

        with self.lock:
            src = self.sources[i]
            src["name"] = display_name

            model_name, short_name = self._extract_model_and_short(display_name)
            if model_name and not src.get("model_name"):
                src["model_name"] = model_name                      # backfill model grouping when not already registered
            if short_name and not src.get("short_name"):
                src["short_name"] = short_name                      # backfill compact source label from full display string

            src["frame_count"] = frame_count                        # latest processed frame count
            src["fps"] = fps_smooth                                 # latest smoothed FPS
            src["time_str"] = time_str                              # latest elapsed/progress time string
            src["counts"] = dict(counts)                            # snapshot class counts for the current UI refresh

            if eta is not None:
                src["eta"] = eta                                    # update ETA only when one is supplied

            if src.get("source_type") is None:
                low = (display_name or "").lower()
                src["source_type"] = "usb" if "| usb" in low or low.startswith("usb") else "video"  # infer source type lazily when not already known

        self._maybe_redraw()

    def mark_source_complete(self, idx: int) -> None:
        i = idx - 1
        if i < 0 or i >= self.total_sources:
            return                                                  # ignore invalid completion updates safely

        with self.lock:
            src = self.sources[i]
            src["completed"] = True                                 # marks the pane as complete so values stop updating
            src["frame_count"] = "--"
            src["fps"] = "--"
            src["time_str"] = "--"
            src["eta"] = "--"

            ordered_classes = src.get("classes") or (list(self.all_classes) if self.all_classes else [])
            src["counts"] = {cls: "--" for cls in ordered_classes}  # completed sources display placeholder counts in stable class order

        self.info(f"Source '{fmt_bold(src['name'] or 'unknown')}' completed.")
        self._redraw_locked()                                       # force one immediate redraw so the completed pane freezes visibly

    # --------- DETECTION ERRORS / MODEL HELPERS ---------
    def model_fail(self, error) -> None:
        self.error(f"Could NOT initialize model: {error}")          # standard model initialization failure message

    def missing_weights(self, runs_dir) -> None:
        runs_dir = Path(runs_dir)
        self.error("YOLO model weights NOT found.")
        self.warn(f"Expected to find at least one model directory inside: {fmt_path(runs_dir)}")
        self.warn("Run training first OR copy a model into the runs folder.")
        self.exit("Detection aborted due to missing weights.")      # structured multi-line guidance when no usable model weights exist

    def model_init(self, weights_path) -> None:
        weights_path = Path(weights_path)
        try:
            idx = weights_path.parts.index("runs")
            short = Path(*weights_path.parts[idx : idx + 3])        # shorten runs-path output to a cleaner relative-looking path when possible
        except ValueError:
            short = weights_path
        self._append_log(fmt_model(f"Initializing model: {fmt_path(short)}"))

    def open_capture_fail(self, src) -> None:
        self.error(f"Could not open source: {fmt_path(src)}")       # standard source-open failure message

    def read_frame_fail(self, src) -> None:
        self.error(f"Could not read frame from {fmt_path(src)}")    # standard first-frame read failure message

    def inference_fail(self, src, error) -> None:
        self.error(f"Inference failed for {fmt_path(src)}: {error}")  # standard inference runtime failure message

    # --------- VIDEO WRITER REGISTRATION ---------
    def recording_initialized(self, ts: str) -> None:
        if not self._recording_initialized_once:
            self._recording_initialized_once = True                 # only log recording initialization once across all sources
            self.info(f"Recording initialized at {ts}")

    def register_writer(
        self,
        raw_name,
        writer,
        cap,
        source_type,
        out_file,
        display_name=None,
    ):
        safe_name = re.sub(r"[^\w\-]", "_", Path(out_file.name).stem) + out_file.suffix  # normalize writer key so it is safe & deterministic
        self.active_writers[safe_name] = {
            "writer": writer,                                       # active cv2.VideoWriter instance
            "cap": cap,                                             # paired capture object for cleanup symmetry
            "source_type": source_type,                             # source type retained for any future cleanup/reporting needs
            "out_file": out_file,                                   # actual destination file path
            "source_name": raw_name,                                # original source label
            "display_name": display_name or raw_name,               # preferred display label used by UI/reporting
        }
        ts = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        self.recording_initialized(ts)
        return safe_name

    def safe_release_writer(self, name) -> None:
        entry = self.active_writers.get(name)
        if not entry:
            return                                                  # silently ignore unknown writer keys during cleanup

        try:
            entry["writer"].release()                               # release the writer first so file output is finalized
        except Exception:
            pass

        try:
            entry["cap"].release()                                  # release paired capture too in case writer registration owns it
        except Exception:
            pass

        self.active_writers.pop(name, None)                         # remove the writer entry whether or not release calls succeeded

    def release_all_writers(self) -> None:
        for name in list(self.active_writers.keys()):
            self.safe_release_writer(name)                          # iterate over a copy so entries can be removed safely during cleanup

    # --------- SAVE BLOCKS / SHUTDOWN ---------
    def add_final_save_block(
        self,
        title,
        base_dir: Path,
        files: list,
        model_name: str | None = None,
    ) -> None:
        base_dir = Path(base_dir)
        try:
            idx = base_dir.parts.index("measurements")
            short = Path(*base_dir.parts[idx:])                     # shorten saved path display beginning at the measurements folder when possible
        except ValueError:
            short = base_dir

        block = [
            fmt_save(f"{fmt_bold(title)}"),
            fmt_save(f"Saved to: {fmt_path(short)}"),
        ]

        for file_path in files:
            block.append(f"      - {Path(file_path).name}")         # list only filenames inside the save block to keep output compact

        key = model_name or "unknown"
        self.final_save_blocks_by_model.setdefault(key, []).append(block)  # group save summaries by model for final shutdown rendering

    def save_measurements(self, base_dir, files, model_name: str | None = None) -> None:
        base_dir = Path(base_dir)
        try:
            source_name = base_dir.parent.parent.name               # derive readable source name from the measurements directory hierarchy
        except Exception:
            source_name = base_dir.name
        self.add_final_save_block(
            title=f"Measurements for {source_name}",
            base_dir=base_dir,
            files=files,
            model_name=model_name,
        )

    def stop_signal_received(self, single_thread=True) -> None:
        msg = (
            "Stop signal received. Terminating pipeline..."
            if single_thread
            else "Stop signal received. Terminating pipelines..."
        )                                                           # pluralize shutdown message for multi-source/multi-thread runs
        self.final_exit_events.append(msg)
        self.final_exit_events.append("Saving CSV spreadsheets...")
        self.in_shutdown = True                                     # freezes live redraw behavior so the final exit block can take over cleanly

    def render_final_exit_block(self) -> None:
        width, _ = self._get_term_size()

        print("\n" + "-" * width + "\n")

        if self.final_exit_events:
            for line in self.final_exit_events:
                print(fmt_exit(line))                               # print queued shutdown/status lines first
            print()
            print("-" * width)
            print()

        for model_name, blocks in self.final_save_blocks_by_model.items():
            print(fmt_model(model_name))
            print()

            for block in blocks:
                for line in block:
                    print(line)                                     # render each grouped save block under its model heading
                print()

            print("-" * width)
            print()

        print(fmt_exit("All detection threads safely terminated.") + "\n")

    def all_threads_terminated(self) -> None:
        self.freeze_ui = False                                      # ensure final rendering is not suppressed by prompt-time UI freeze
        self.render_final_exit_block()

    # --------- MODEL SELECTION ---------
    def prompt_model_selection(self, runs_dir, exclude_test=False):
        runs_dir = Path(runs_dir)
        dirs = sorted(
            [
                d
                for d in runs_dir.iterdir()
                if d.is_dir() and (not exclude_test or d.name.lower() != "test")
            ],
            reverse=True,
        )                                                           # newest-looking run folders are shown first for convenience

        if not dirs:
            self.missing_weights(runs_dir)
            return None

        self.freeze_ui = True                                       # pause live redraws while the terminal is used for manual input
        self.info(f"{len(dirs)} models found:")

        print("\n" + fmt_bold("Available models") + ":")
        for i, d in enumerate(dirs, 1):
            print(f"   {fmt_bold(i)}. {d.name}")                    # display a numbered list for simple keyboard selection
        print()

        try:
            while True:
                try:
                    choice = input(f"Select model (1-{len(dirs)}) or Ctrl+C: ").strip()
                except KeyboardInterrupt:
                    print("\n" + fmt_exit("Model selection cancelled") + "\n")
                    return None

                if choice.isdigit():
                    choice = int(choice)
                    if 1 <= choice <= len(dirs):
                        return dirs[choice - 1]                     # return the selected run directory directly

                self.warn("Invalid selection.")                     # keep prompting until a valid index is entered or cancelled
        finally:
            self.freeze_ui = False                                  # always unfreeze UI redraws after model-selection prompt exits
            self._maybe_redraw()

# --------- SHARED TRAINING UI INSTANCE ---------
training_ui = TrainingUI()                                         # shared singleton-style training UI used across the training pipeline