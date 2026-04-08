# utils/train/io.py
from __future__ import annotations                  # keeps annotation behavior consistent with the rest of the project
import json                                         # used for reading training metadata JSON files
from pathlib import Path                            # used for safe cross-platform path handling
from typing import Optional                         # used for optional Path return annotations
import requests                                     # used for downloading missing model YAMLs & pretrained weights
from ..ui import fmt_exit, fmt_info, fmt_model, fmt_warn, fmt_error, fmt_dataset  # shared terminal formatting helpers

# --------- MODEL FAMILY REGISTRIES ---------
FAMILY_TO_YAML = {
    "yolov8": "yolov8.yaml",                        # official YOLOv8 HBB architecture YAML
    "yolov8-obb": "yolov8-obb.yaml",               # official YOLOv8 OBB architecture YAML
    "yolo11": "yolo11.yaml",                       # official YOLO11 HBB architecture YAML
    "yolo11-obb": "yolo11-obb.yaml",               # official YOLO11 OBB architecture YAML
    "yolo12": "yolo12.yaml",                       # official YOLO12 HBB architecture YAML
    "yolo12-obb": "yolo12-obb.yaml",               # official YOLO12 OBB architecture YAML
}

FAMILY_TO_WEIGHTS = {
    "yolov8": "yolov8n.pt",                        # default pretrained YOLOv8 nano weights
    "yolov8-obb": "yolov8n-obb.pt",               # default pretrained YOLOv8 nano OBB weights
    "yolo11": "yolo11n.pt",                       # default pretrained YOLO11 nano weights
    "yolo11-obb": "yolo11n-obb.pt",               # default pretrained YOLO11 nano OBB weights
    "yolo12": "yolo12n.pt",                       # default pretrained YOLO12 nano weights
    # NOTE: no official "yolo12n-obb.pt", so special-cased fallback logic is used below.
}

# --------- MODEL NAME NORMALIZATION ---------
def normalize_model_name(name: str) -> tuple[str, str | None]:
    """
    Normalize a model/architecture name into (family, variant).

    Examples:
        yolo11n.pt     -> ("yolo11", "n")
        yolo11n-obb.pt -> ("yolo11-obb", "n")
        yolo11.yaml    -> ("yolo11", None)
    """
    base = name.lower().replace(".pt", "").replace(".yaml", "")  # strip common model file extensions before family parsing

    is_obb = base.endswith("-obb")                    # OBB families are represented with a trailing -obb suffix
    core = base[:-4] if is_obb else base             # remove the OBB suffix temporarily so variant parsing stays simple

    variant = None
    if core and core[-1] in {"n", "s", "m", "l", "x"}:
        family_core = core[:-1]                       # trailing size letter belongs to the variant, not the family name
        variant = core[-1]
    else:
        family_core = core                            # architecture-only names like yolo11 have no explicit variant

    family = family_core + ("-obb" if is_obb else "")  # rebuild normalized family name with OBB suffix when needed
    return family, variant

def is_custom_yaml(arch: str, models_dir: Path) -> bool:
    """
    Return True when the requested architecture should be treated as a
    custom YAML instead of an official YOLO family.

    Rules:
        - explicit YAML names that match official family YAMLs are NOT custom
        - any other YAML path/name IS custom
        - non-YAML names are official only if their normalized family is known
    """
    arch_lower = str(arch).lower()

    if arch_lower.endswith(".yaml"):
        yaml_name = Path(arch_lower).name             # compare only filename for explicit YAML requests
        return yaml_name not in FAMILY_TO_YAML.values()  # official family YAML filenames are not treated as custom

    family, _ = normalize_model_name(arch_lower)
    return family not in FAMILY_TO_YAML               # non-YAML names are custom when they do not map to a known official family

def family_is_obb(family: str | None) -> bool:
    """
    Return True when a normalized model family is an OBB family.
    """
    return bool(family and family.endswith("-obb"))   # normalized OBB families always carry the -obb suffix

def yaml_is_obb(yaml_path: Path) -> bool:
    """
    Return True when a model YAML appears to define an OBB architecture.
    """
    try:
        with open(yaml_path, "r") as handle:
            return "obb" in handle.read().lower()     # lightweight heuristic scans YAML text for OBB references
    except Exception:
        return False                                  # unreadable YAML is treated as non-OBB for safety

# --------- FILE DOWNLOADS ---------
def download_file(url: str, dest_path: Path) -> Optional[Path]:
    """
    Download a file to disk and return the saved path on success.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)  # ensure the destination directory exists before streaming the download
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)                           # stream chunks to disk so large model files do not load fully into memory
        print(fmt_info(f"Downloaded {dest_path}"))
        return dest_path
    except Exception as e:
        print(fmt_error(f"Failed downloading {url}: {e}"))
        return None                                     # failed downloads degrade to None so callers can handle resolution failure cleanly

# --------- ARCHITECTURE YAML RESOLUTION ---------
def ensure_yolo_yaml(yolo_yaml_path: Path, model_type: str) -> Optional[Path]:
    """
    Ensure the official YOLO architecture YAML exists locally.

    Returns:
        YAML path or None on failure
    """
    family, _ = normalize_model_name(model_type)

    if family not in FAMILY_TO_YAML:
        print(fmt_error(f"Unsupported architecture family: '{model_type}' → '{family}'"))
        print(fmt_error(f"Supported families: {list(FAMILY_TO_YAML.keys())}"))
        return None                                     # unsupported official family names are rejected early

    if yolo_yaml_path.exists():
        return yolo_yaml_path                           # existing local YAML is always preferred over re-downloading

    yaml_urls = {
        "yolov8": "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/v8/yolov8.yaml",
        "yolov8-obb": "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/v8/yolov8-obb.yaml",
        "yolo11": "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/11/yolo11.yaml",
        "yolo11-obb": "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/11/yolo11-obb.yaml",
        "yolo12": "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/12/yolo12.yaml",
        "yolo12-obb": "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/12/yolo12-obb.yaml",
    }                                                   # official upstream YAML download locations keyed by normalized family

    print(fmt_info(f"Model architecture YAML not found, downloading '{family}' → {yolo_yaml_path}"))
    return download_file(yaml_urls[family], yolo_yaml_path)

# --------- WEIGHTS RESOLUTION ---------
def ensure_weights(yolo_weights_path: Path, model_type: str) -> Optional[Path]:
    """
    Ensure pretrained weights exist locally for the requested model type.

    Returns:
        weights path or None on failure
    """
    if yolo_weights_path.suffix == ".pt":
        yolo_weights_path = yolo_weights_path.parent  # tolerate callers passing either a weights folder or a placeholder .pt path

    family, variant = normalize_model_name(model_type)
    is_obb = family.endswith("-obb")
    family_base = family[:-4] if is_obb else family   # remove OBB suffix temporarily when composing weight filename
    variant = variant or "n"                          # default to nano when no size variant was explicitly provided

    correct_name = f"{family_base}{variant}{'-obb' if is_obb else ''}.pt"
    dest_path = yolo_weights_path / correct_name

    if dest_path.is_file():
        return dest_path                              # reuse already downloaded weights immediately

    if family not in FAMILY_TO_WEIGHTS:
        if family == "yolo12-obb":
            print(fmt_warn(f"Pretrained OBB weights not found for '{family}'. Falling back to 'yolo12'."))
            correct_name = f"yolo12{variant}.pt"     # special-case fallback because no official yolo12 OBB pretrained weights exist
            dest_path = yolo_weights_path / correct_name
            family = "yolo12"

            if dest_path.is_file():
                return dest_path                      # fallback file may already exist locally even though OBB weights do not
        else:
            print(fmt_error(f"No registered default weights for '{family}'"))
            return None                              # unknown families cannot be auto-resolved to official pretrained weights

    weight_urls = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt",
        "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt",
        "yolov8l.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt",
        "yolov8x.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt",
        "yolov8n-obb.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-obb.pt",
        "yolov8s-obb.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-obb.pt",
        "yolov8m-obb.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-obb.pt",
        "yolov8l-obb.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-obb.pt",
        "yolov8x-obb.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-obb.pt",
        "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "yolo11s.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
        "yolo11m.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
        "yolo11l.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt",
        "yolo11x.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
        "yolo11n-obb.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.pt",
        "yolo11s-obb.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-obb.pt",
        "yolo11m-obb.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-obb.pt",
        "yolo11l-obb.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-obb.pt",
        "yolo11x-obb.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-obb.pt",
        "yolo12n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt",
        "yolo12s.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12s.pt",
        "yolo12m.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12m.pt",
        "yolo12l.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12l.pt",
        "yolo12x.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12x.pt",
    }                                                   # official pretrained weight URLs keyed by full weight filename

    if correct_name not in weight_urls:
        print(fmt_error(f"No URL for {correct_name}"))
        return None                                     # weight family may be known but unavailable for the requested size/variant

    print(fmt_info(f"Model weights not found, downloading '{model_type}' ({correct_name}) → {dest_path}"))
    return download_file(weight_urls[correct_name], dest_path)

# --------- IMAGE COUNTING ---------
def count_images(folder: Path) -> int:
    """
    Count image files inside a folder using common image extensions.
    """
    if not folder.exists():
        return 0                                        # missing folders simply count as zero images

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sum(len(list(folder.glob(f"*{e}"))) for e in exts)  # count each supported extension separately for a simple flat image total

# --------- METADATA LOADING ---------
def load_latest_metadata(logs_root: Path) -> Optional[dict]:
    """
    Load the newest metadata.json found directly beneath a logs root.
    """
    if not logs_root.exists():
        return None                                     # no logs root means no metadata to scan

    latest, meta = 0, None
    for run in logs_root.iterdir():
        if not run.is_dir():
            continue                                    # only run folders can contain metadata.json
        p = run / "metadata.json"
        if p.exists() and (mtime := p.stat().st_mtime) > latest:
            latest = mtime                              # keep only the newest metadata file seen so far
            try:
                meta = json.load(open(p, "r"))
            except Exception as e:
                print(fmt_warn(f"Failed to load metadata JSON file: {e}"))

    return meta