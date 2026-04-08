# utils/paths.py
from __future__ import annotations              # allows forward references in type hints without quoting types
import re                                       # used for sanitizing source names into filesystem-safe strings
import shutil                                   # used for copying bundled example assets into the user workspace
from datetime import datetime                   # used for timestamped detection output folder names
from importlib.resources import files           # used for locating packaged example assets inside the installed project
from pathlib import Path                        # used for safe cross-platform path handling

BASE_DIR = Path.home() / "avian"               # root workspace directory where all user data, runs, logs, & configs live

# --------- STANDARD FOLDER LAYOUT ---------
DATA_DIR = BASE_DIR / "data"                    # root dataset directory
RUNS_DIR = BASE_DIR / "runs"                    # root training runs directory
LOGS_DIR = BASE_DIR / "logs"                    # root detection/training logs directory
MODELS_DIR = BASE_DIR / "models"                # bundled/custom model YAML directory
WEIGHTS_DIR = BASE_DIR / "weights"              # downloaded or copied model weights directory
CONFIGS_DIR = BASE_DIR / "configs"              # config storage directory for measurement/W&B/model configs
LS_ROOT = BASE_DIR / "labelstudio-projects"     # root Label Studio projects directory
WANDB_ROOT = BASE_DIR / "wandb"                 # root W&B output/cache directory

# --------- DEFAULT CONFIG FILES ---------
MEASURE_CONFIG_YAML = CONFIGS_DIR / "measure_config.yaml"  # default measurement config YAML path
WANDB_CONFIG_YAML = CONFIGS_DIR / "wandb_config.yaml"      # default W&B config YAML path

# --------- DIRECTORY BOOTSTRAP ---------
def _ensure_dir(path: Path) -> Path:
    """
    Create a directory if it does not already exist, then return it.
    """
    path.mkdir(parents=True, exist_ok=True)      # recursive safe mkdir keeps repeated setup calls harmless
    return path

def _ensure_standard_dirs() -> None:
    """
    Ensure the standard avian folder layout exists.
    """
    for path in (
        BASE_DIR,
        DATA_DIR,
        RUNS_DIR,
        LOGS_DIR,
        MODELS_DIR,
        WEIGHTS_DIR,
        CONFIGS_DIR,
        LS_ROOT,
        WANDB_ROOT,
    ):
        _ensure_dir(path)                        # create every expected top-level workspace folder before anything else uses them

def _install_examples() -> None:
    """
    Install bundled example assets into the user's avian workspace.

    This only copies examples when the target does not already exist, so
    user data is not overwritten.
    """
    try:
        pkg_root = files("avian")               # resolve the installed package root so bundled example assets can be found

        pkg_example_ls = pkg_root / "labelstudio-projects" / "example"  # packaged example Label Studio project
        target_example_ls = LS_ROOT / "example"
        if pkg_example_ls.is_dir() and not target_example_ls.exists():
            shutil.copytree(pkg_example_ls, target_example_ls, dirs_exist_ok=True)  # copy only when the user has no existing example project

        pkg_example_run = pkg_root / "runs" / "sparrows"  # packaged example trained run
        target_example_run = RUNS_DIR / "sparrows"
        if pkg_example_run.is_dir() and not target_example_run.exists():
            shutil.copytree(pkg_example_run, target_example_run, dirs_exist_ok=True)  # seed example run only when absent

        pkg_models = pkg_root / "models"         # packaged model YAML directory
        if pkg_models.is_dir() and not any(MODELS_DIR.iterdir()):
            shutil.copytree(pkg_models, MODELS_DIR, dirs_exist_ok=True)  # populate model YAMLs only when the target models folder is empty
    except Exception:
        pass                                     # example installation should never block normal application startup

_ensure_standard_dirs()                          # create the standard workspace structure at import time
_install_examples()                             # seed example assets once the folder layout exists

# --------- ROOT DIRECTORY HELPERS ---------
def get_runs_dir(test: bool = False) -> Path:
    """
    Return the training runs root directory.
    """
    return _ensure_dir(RUNS_DIR / "test") if test else RUNS_DIR  # test mode isolates runs beneath runs/test

def get_logs_dir(test: bool = False) -> Path:
    """
    Return the detection/training logs root directory.
    """
    return _ensure_dir(LOGS_DIR / "test") if test else LOGS_DIR  # test mode isolates logs beneath logs/test

# --------- TRAINING PATH HELPERS ---------
def get_training_paths(dataset_folder: str | Path, test: bool = False) -> dict[str, Path]:
    """
    Build the standard path bundle used by the training pipeline.
    """
    dataset_folder = Path(dataset_folder)        # normalize dataset path once for all downstream training path construction

    return {
        "runs_root": get_runs_dir(test),         # destination root for model run folders
        "logs_root": get_logs_dir(test),         # destination root for training logs/artifacts
        "train_folder": dataset_folder / "train" / "images",  # expected train image folder
        "val_folder": dataset_folder / "val" / "images",      # expected validation image folder
        "weights_folder": WEIGHTS_DIR,           # shared weights directory
        "models_folder": MODELS_DIR,             # shared model YAML directory
        "dataset_folder": dataset_folder,        # original normalized dataset root
    }

# --------- MODEL CONFIG HELPERS ---------
def get_model_config_dir(model_name: str) -> Path:
    """
    Return the config directory for a model, creating it if needed.
    """
    return _ensure_dir(CONFIGS_DIR / str(model_name).strip())  # each model gets its own config subdirectory

# --------- DETECTION OUTPUT HELPERS ---------
def _resolve_model_name_from_weights(weights_path: str | Path) -> str:
    """
    Resolve the model name from a weights path.
    """
    weights_path = Path(weights_path)             # normalize weights path before inspecting its folder structure

    if "runs" in weights_path.parts:
        return weights_path.parent.parent.name    # trained run weights usually live at runs/<model>/weights/<file>.pt

    return weights_path.stem                      # public/pretrained weights fall back to the file stem as model name

def _sanitize_source_name(source_name: str | Path, source_type: str) -> str:
    """
    Convert a source name into a filesystem-safe folder/file stem.
    """
    raw_name = Path(source_name).stem if source_type == "video" else str(source_name)  # file sources use stem while cameras preserve the raw label
    return re.sub(r"[^\w\-.]", "_", raw_name)  # replace unsafe filesystem characters with underscores

def _unique_run_folder(base_folder: Path) -> Path:
    """
    Prevent output folder collisions by appending a numeric suffix.
    """
    if not base_folder.exists():
        return base_folder                        # first-choice timestamp folder is fine when it does not already exist

    original = base_folder
    suffix = 1

    while base_folder.exists():
        base_folder = original.parent / f"{original.name}_{suffix}"  # append incrementing suffixes until a free folder name is found
        suffix += 1

    return base_folder

def get_detection_output_paths(
    weights_path: str | Path,
    source_type: str,
    source_name: str,
    test_detect: bool = False,
    base_time: datetime | None = None,
) -> dict[str, Path | str]:
    """
    Build the full detection output folder structure for one model/source pair.
    """
    model_name = _resolve_model_name_from_weights(weights_path)  # derive owning model name from the weights path
    logs_root = get_logs_dir(test_detect) / model_name / "measurements"  # group measurement outputs beneath model-specific log folders

    folder_time = base_time or datetime.now()      # prefer source-based start time when available so folder names stay meaningful
    run_timestamp = folder_time.strftime("%m-%d-%Y_%H-%M-%S")
    safe_name = _sanitize_source_name(source_name, source_type)  # normalize source name for safe use in folders/files

    if source_type == "video":
        base_folder = logs_root / "video-in" / safe_name / run_timestamp  # file sources are grouped beneath video-in
    else:
        base_folder = logs_root / "camera-feeds" / safe_name / run_timestamp  # live sources are grouped beneath camera-feeds

    base_folder = _unique_run_folder(base_folder)  # protect against collisions when timestamps/source names repeat

    video_folder = base_folder / "recordings"      # annotated output video folder
    scores_folder = base_folder / "scores"         # root measurement-output folder
    counts_folder = scores_folder / "counts"       # count-related CSV folder
    interactions_folder = scores_folder / "interactions"  # interaction CSV folder
    motion_folder = scores_folder / "motion"       # motion CSV folder

    for path in (
        video_folder,
        scores_folder,
        counts_folder,
        interactions_folder,
        motion_folder,
    ):
        _ensure_dir(path)                          # ensure all expected detection output subfolders exist before use

    return {
        "video_folder": video_folder,
        "scores_folder": scores_folder,
        "counts": counts_folder,
        "interactions": interactions_folder,
        "motion": motion_folder,
        "metadata": scores_folder / f"{safe_name}_metadata.json",  # metadata JSON saved alongside measurement outputs
        "safe_name": safe_name,
    }

get_output_folder = get_detection_output_paths    # backward-compatible alias retained for older callers