# utils/train/wandb_util.py
from __future__ import annotations                 # keeps annotation behavior consistent with the rest of the project
import os                                          # used for configuring W&B runtime through environment variables
import warnings                                    # reserved for future warning control around W&B/runtime integration
import yaml                                        # used for reading & writing the W&B YAML config
import wandb                                       # imported so the W&B SDK is available when Ultralytics uses it
from ultralytics import settings as yolo_settings  # Ultralytics settings object used to toggle internal W&B integration
from ..ui import fmt_info, fmt_warn, fmt_path      # shared UI formatters for readable terminal messages
from ..paths import WANDB_ROOT, WANDB_CONFIG_YAML  # shared W&B directory & config file paths

DEFAULT_WANDB_CONFIG = {
    "enabled": False,                              # master on/off switch for W&B integration
    "project": "<your_project_name>",              # placeholder project name written into the starter config
    "entity": None,                                # optional profile/team/entity name
    "mode": "online",                              # supported modes: online | offline | disabled
}

# --------- CONFIG HELPERS ---------
def _write_default_wandb_config():
    """
    Write the default W&B config YAML to disk.
    """
    WANDB_CONFIG_YAML.parent.mkdir(parents=True, exist_ok=True)  # ensure the config directory exists before writing the YAML file
    with open(WANDB_CONFIG_YAML, "w") as f:
        yaml.safe_dump(DEFAULT_WANDB_CONFIG, f, sort_keys=False)  # preserve key order so the default config stays readable

def load_wandb_config() -> dict:
    """
    Load, normalize, and validate the W&B config YAML.
    """
    if not WANDB_CONFIG_YAML.exists():
        _write_default_wandb_config()                 # create a starter config automatically the first time W&B settings are requested
        print(fmt_info(f"Created default W&B config: {fmt_path(WANDB_CONFIG_YAML)}"))
        print(fmt_info("Edit this file to set your own W&B project and profile/entity."))

    try:
        with open(WANDB_CONFIG_YAML, "r") as f:
            data = yaml.safe_load(f) or {}            # blank or minimal YAML files normalize to an empty mapping
    except Exception as e:
        print(fmt_warn(f"Failed reading W&B config, using defaults: {e}"))
        data = {}                                     # fallback to defaults when config loading fails

    config = DEFAULT_WANDB_CONFIG.copy()
    config.update(data)                               # YAML values override defaults while unspecified keys inherit them

    mode = str(config.get("mode", "online")).strip().lower()
    if mode not in {"online", "offline", "disabled"}:
        print(fmt_warn(f"Invalid W&B mode '{mode}', falling back to 'online'."))
        mode = "online"                               # invalid mode strings are corrected rather than aborting training
    config["mode"] = mode

    config["enabled"] = bool(config.get("enabled", True))  # coerce enabled flag to a real boolean

    project = config.get("project")
    if project is not None:
        project = str(project).strip()
    config["project"] = project or "yolo4r"          # fallback project name keeps runtime configuration usable even when YAML is blank

    entity = config.get("entity")
    if entity is not None:
        entity = str(entity).strip()
    config["entity"] = entity or None                # blank entity strings normalize to None

    return config

# --------- RUNTIME CONFIGURATION ---------
def configure_wandb_runtime() -> dict:
    """
    Apply both W&B SDK settings and Ultralytics settings before training starts.

    Returns:
        normalized config
    """
    config = load_wandb_config()

    if not config["enabled"] or config["mode"] == "disabled":
        os.environ["WANDB_MODE"] = "disabled"        # disables SDK-side logging behavior explicitly
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ.pop("WANDB_DISABLED", None)       # remove hard disable flag when W&B should actually run
        os.environ["WANDB_MODE"] = config["mode"]    # pass through online/offline mode directly to the SDK

    WANDB_ROOT.mkdir(parents=True, exist_ok=True)    # ensure the W&B root folder exists before assigning it as WANDB_DIR
    os.environ["WANDB_DIR"] = str(WANDB_ROOT)
    os.environ["WANDB_PROJECT"] = config["project"]  # set project name for Ultralytics/W&B runtime usage

    if config["entity"]:
        os.environ["WANDB_ENTITY"] = config["entity"]  # set entity only when explicitly configured
    else:
        os.environ.pop("WANDB_ENTITY", None)           # remove stale entity values from earlier sessions/configurations

    try:
        yolo_settings.update({"wandb": bool(config["enabled"] and config["mode"] != "disabled")})  # keep Ultralytics' internal W&B toggle in sync
    except Exception as e:
        print(fmt_warn(f"Could not update Ultralytics W&B setting: {e}"))

    return config

# --------- PUBLIC TRAINING ENTRY POINT ---------
def init_wandb(run_name: str):
    """
    Configure W&B/Ultralytics runtime only.

    Do NOT manually create a W&B run here, otherwise duplicate runs can occur.
    """
    config = configure_wandb_runtime()              # normalize config & apply SDK/Ultralytics runtime settings first

    if not config["enabled"] or config["mode"] == "disabled":
        print(fmt_info("W&B logging disabled within the config."))
        return None                                 # fully disabled mode stops here so training proceeds without W&B integration

    if config["entity"]:
        return config                               # return normalized config when entity exists so callers can inspect/use it if needed

    return config                                   # return config even without entity so caller behavior stays consistent