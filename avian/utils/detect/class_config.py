# utils/detect/classes_config.py
from __future__ import annotations                  # allows forward references in type hints without quoting types
from dataclasses import dataclass                   # used to define the per-model class configuration container
from pathlib import Path                            # used for safe cross-platform config path handling
from typing import Any                              # used where YAML-loaded values may vary in type
import yaml                                         # used for reading & writing class configuration YAML files
from ultralytics import YOLO                        # used to load model weights & discover embedded class names
from ..paths import get_model_config_dir            # shared helper that resolves the config folder for a given model

# --------- CLASS CONFIG CONTAINER ---------
@dataclass
class ClassConfig:
    """
    Per-model class configuration.

    This replaces global FOCUS_CLASSES / CONTEXT_CLASSES and keeps each model's
    class layout isolated in its own config file.
    """
    model_name: str                                 # logical model name used for config folder resolution & UI display
    config_path: Path                               # full path to this model's saved class config YAML
    focus: list[str]                                # biologically relevant tracked classes used directly in counting/ratio outputs
    context: list[str]                              # environmental/context classes grouped separately from focus classes

    @property
    def all_classes(self) -> list[str]:
        """
        Return all configured classes in display order:
        focus classes first, then context classes.
        """
        return [*self.focus, *self.context]         # preserves the intended display/processing order across the full pipeline

    @property
    def display_classes(self) -> list[str]:
        """
        Return the class labels that should appear in the UI.

        Context classes are grouped under a single 'OBJECTS' label when present.
        """
        display = list(self.focus)                  # UI always shows focus classes individually
        if self.context:
            display.append("OBJECTS")               # context classes collapse into one display label for cleaner count output
        return display

    def to_dict(self) -> dict[str, list[str]]:
        """
        Convert the config into the YAML storage format.
        """
        return {
            "FOCUS_CLASSES": list(self.focus),      # saved key name kept explicit for user-facing YAML readability
            "CONTEXT_CLASSES": list(self.context),
        }

    def save(self, ui=None) -> None:
        """
        Save this class configuration to disk.
        """
        self.config_path.parent.mkdir(parents=True, exist_ok=True)  # ensure the model-specific config directory exists before writing

        with open(self.config_path, "w") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)  # preserve key order so saved YAML stays readable & predictable

        if ui:
            ui.info(f"Class configuration saved to: {_short_config_path(self.model_name, self.config_path)}")  # show a cleaner relative-style path in UI messages

# --------- PATH HELPERS ---------
def _model_config_path(model_name: str) -> Path:
    """
    Return the canonical class-config path for a model.

    Example:
        configs/<model_name>/classes_config.yaml
    """
    return get_model_config_dir(model_name) / "classes_config.yaml"  # every model keeps its own isolated class config file

def _short_config_path(model_name: str, fallback_path: Path) -> str:
    """
    Return a user-friendly config path string for console messages.
    """
    try:
        model_dir = get_model_config_dir(model_name)
        return f"configs/{model_dir.name}/classes_config.yaml"  # shorter relative-style path is easier to read in terminal output
    except Exception:
        return str(fallback_path)                               # fallback preserves a usable path string even if helper resolution fails

# --------- YAML NORMALIZATION HELPERS ---------
def _normalize_class_list(value: Any) -> list[str]:
    """
    Normalize a YAML-loaded class field into a clean list of strings.
    """
    if not isinstance(value, list):
        return []                                               # non-list YAML fields are treated as invalid/empty class lists

    normalized: list[str] = []
    for item in value:
        text = str(item).strip()                                # normalize all class names to trimmed strings
        if text:
            normalized.append(text)                             # skip blank entries so saved configs stay clean

    return normalized

def _read_yaml_mapping(path: Path) -> dict[str, Any] | None:
    """
    Read a YAML file and return its top-level mapping.

    Returns None when the file is missing, unreadable, or not a mapping.
    """
    if not path.exists():
        return None                                             # missing config is not an error because generation may happen later

    with open(path, "r") as handle:
        loaded = yaml.safe_load(handle) or {}                   # blank YAML files normalize to an empty mapping

    if not isinstance(loaded, dict):
        raise ValueError("Class config YAML must contain a top-level mapping.")  # callers rely on mapping semantics for key lookup

    return loaded

def _load_yaml_config(path: Path, ui=None, model_name: str | None = None) -> dict[str, list[str]] | None:
    """
    Load and normalize an existing class config YAML file.

    Returns:
        {
            "focus": [...],
            "context": [...],
        }

    Returns None when the file is missing or invalid.
    """
    if not path.exists():
        return None                                             # missing file simply means a new config may need to be generated
    try:
        saved = _read_yaml_mapping(path)
        if saved is None:
            return None

        return {
            "focus": _normalize_class_list(saved.get("FOCUS_CLASSES", [])),       # normalize focus class list to clean strings
            "context": _normalize_class_list(saved.get("CONTEXT_CLASSES", [])),   # normalize context class list to clean strings
        }
    except Exception:
        if ui and model_name:
            ui.error(f"Class config YAML is corrupted for model '{model_name}'.")
        elif ui:
            ui.error("Class config YAML is corrupted.")
        return None                                            # invalid YAML is recoverable because config can be regenerated from weights

# --------- MODEL CLASS DISCOVERY ---------
def _detect_classes_from_weights(weights_path: Path, ui=None) -> list[str]:
    """
    Load the YOLO model and detect class names from its weights.
    """
    try:
        model = YOLO(str(weights_path))                        # load model weights directly so class names can be read from the model object
        names = model.names or {}                              # Ultralytics may return either a dict or a list depending on version/setup

        if isinstance(names, dict):
            return [str(names[index]).strip() for index in sorted(names.keys())]  # preserve numeric class-id ordering when names is a mapping

        if isinstance(names, list):
            return [str(name).strip() for name in names if str(name).strip()]     # keep non-blank class labels in original model order

        return []                                               # unsupported names structures collapse to an empty class list
    except Exception as exc:
        if ui:
            ui.error(f"Failed to load classes: {exc}")
        return []                                               # caller can still generate/save an empty config rather than crashing

# --------- CONFIG BUILDERS ---------
def _build_class_config(
    model_name: str,
    config_path: Path,
    focus: list[str],
    context: list[str],
) -> ClassConfig:
    """
    Construct a ClassConfig object from normalized class lists.
    """
    return ClassConfig(
        model_name=model_name,
        config_path=config_path,
        focus=list(focus),                                     # copy lists so later outside mutations do not affect stored config state
        context=list(context),
    )

def _load_existing_class_config(
    model_name: str,
    config_path: Path,
    ui=None,
) -> ClassConfig | None:
    """
    Load an existing class config from YAML if it is valid.
    """
    loaded = _load_yaml_config(config_path, ui=ui, model_name=model_name)
    if loaded is None:
        return None                                            # caller will decide whether to regenerate when no valid YAML exists

    class_config = _build_class_config(
        model_name=model_name,
        config_path=config_path,
        focus=loaded["focus"],
        context=loaded["context"],
    )

    if ui:
        ui.info(f"Loaded {len(class_config.all_classes)} classes: {class_config.all_classes}")  # report class load success with the effective class order

    return class_config

def _generate_class_config_from_weights(
    model_name: str,
    weights_path: Path,
    config_path: Path,
    ui=None,
) -> ClassConfig:
    """
    Generate a new class config by reading class names directly from model weights.
    """
    detected_classes = _detect_classes_from_weights(weights_path, ui=ui)  # discover classes directly from the YOLO weights file

    class_config = _build_class_config(
        model_name=model_name,
        config_path=config_path,
        focus=detected_classes,                           # newly generated configs treat all discovered classes as focus by default
        context=[],                                       # context classes start empty until user edits the YAML intentionally
    )

    class_config.save(ui=ui)                              # persist generated config immediately so future runs can reuse it

    if ui:
        ui.info(f"Generated class config YAML with {len(detected_classes)} classes: {detected_classes}")

    return class_config

# --------- PUBLIC ENTRY POINT ---------
def load_or_create_classes(
    model_name: str,
    weights_path: str | Path,
    force_reload: bool = False,
    ui=None,
) -> ClassConfig:
    """
    Load or generate a per-model class configuration.

    Behavior:
        - If a valid YAML config exists and force_reload is False, load it.
        - Otherwise regenerate from YOLO(weights).names.
        - Newly generated configs use:
            focus   = detected classes
            context = []
        - Always returns a ClassConfig.
    """
    model_name = str(model_name).strip()                  # normalize model name before using it in config folder/path resolution
    weights_path = Path(weights_path)                     # normalize weights path to Path for downstream file/model loading logic
    config_path = _model_config_path(model_name)          # canonical config location for this specific model

    if not force_reload:
        existing = _load_existing_class_config(
            model_name=model_name,
            config_path=config_path,
            ui=ui,
        )
        if existing is not None:
            return existing                               # valid existing YAML is reused as-is unless force_reload was requested

        if config_path.exists() and ui:
            ui.warn("Class config YAML is invalid or unreadable. Regenerating...")  # warn only when a file existed but could not be used

    return _generate_class_config_from_weights(
        model_name=model_name,
        weights_path=weights_path,
        config_path=config_path,
        ui=ui,
    )                                                     # final fallback always regenerates config directly from model weights