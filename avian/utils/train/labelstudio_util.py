# utils/train/labelstudio_util.py
from __future__ import annotations                  # keeps annotation behavior consistent with the rest of the project
import json                                         # used for reading/writing dataset metadata JSON files
import random                                       # used for shuffling images before train/val split
import shutil                                       # used for copying image/label pairs into YOLO dataset folders
from datetime import datetime                       # used for timestamp-based dataset naming & metadata
from pathlib import Path                            # used for safe cross-platform dataset/project path handling
import yaml                                         # used for writing YOLO data.yaml files
from ..ui import fmt_info, fmt_warn, fmt_error, fmt_save, fmt_path, fmt_bold  # shared formatted console helpers

# --------- LABEL MODE HELPERS ---------
def _detect_label_mode(lbl_folder: Path) -> str:
    """
    Detect whether a label folder appears to contain HBB or OBB annotations.
    """
    for lbl_file in lbl_folder.glob("*.txt"):
        try:
            with open(lbl_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue                      # skip blank lines when probing label structure
                    if len(parts) >= 6:
                        return "obb"                 # OBB rows generally contain more coordinates than plain HBB YOLO rows
        except Exception:
            continue                                 # unreadable label files are ignored so probing can continue
    return "hbb"                                     # default to HBB when no clear OBB signature is found

def get_dataset_label_mode(dataset_folder: Path) -> str | None:
    """
    Read the processed dataset metadata and return its label mode.

    Returns:
        "hbb", "obb", or None when metadata is missing/unreadable.
    """
    meta_path = dataset_folder / "metadata.json"     # processed datasets store their label mode in metadata.json
    if not meta_path.exists():
        return None                                  # missing metadata means label mode cannot be determined here

    try:
        with open(meta_path, "r") as handle:
            metadata = json.load(handle)
        return metadata.get("label_mode")
    except Exception:
        return None                                  # unreadable metadata falls back to unknown label mode

# --------- LABEL STUDIO PROJECT VALIDATION ---------
def is_labelstudio_project(project_folder: Path) -> bool:
    """
    Return True when a folder matches the expected minimal Label Studio project layout.
    """
    project_folder = Path(project_folder)
    return (
        project_folder.is_dir()
        and (project_folder / "images").is_dir()     # Label Studio image folder must exist
        and (project_folder / "labels").is_dir()     # Label Studio label folder must exist
        and (project_folder / "classes.txt").exists()  # Label Studio class list must exist
    )

# --------- EXISTING DATASET LOOKUP ---------
def find_processed_dataset_for_project(project_folder: Path, data_root: Path):
    """
    Search for an already processed YOLO dataset linked to a Label Studio project.

    Returns:
        (dataset_folder, data_yaml) or (None, None)
    """
    project_folder = Path(project_folder).resolve()  # resolve to absolute path so comparisons against saved metadata are stable
    data_root = Path(data_root).resolve()

    if not data_root.exists():
        return None, None                            # no dataset root means no processed dataset can exist yet

    for existing in data_root.iterdir():
        if not existing.is_dir():
            continue                                 # only folders can represent processed datasets

        meta_path = existing / "metadata.json"
        if not meta_path.exists():
            continue                                 # skip folders that do not look like processed dataset outputs

        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)

            if (
                meta.get("processed")
                and meta.get("source_type") == "labelstudio"
                and Path(meta.get("original_project", "")).resolve() == project_folder
            ):
                data_yaml = existing / "data.yaml"
                if data_yaml.exists():
                    return existing, data_yaml       # return the matching processed dataset only when its data.yaml also exists
        except Exception:
            continue                                 # corrupted metadata is ignored so the search can continue

    return None, None

# --------- PROJECT RESOLUTION ---------
def resolve_labelstudio_project(labelstudio_root: Path, data_root: Path, project_name: str | None = None):
    """
    Resolve which Label Studio project should be used.

    Behavior:
        - if project_name is provided, validate & return that project
        - otherwise choose the most recent valid unprocessed project
        - returns None when only already-processed valid projects exist
    """
    labelstudio_root = Path(labelstudio_root).resolve()
    data_root = Path(data_root).resolve()

    if not labelstudio_root.exists():
        raise FileNotFoundError(
            fmt_error(f"Label-Studio root folder NOT found: {fmt_path(labelstudio_root)}")
        )

    if project_name:
        project_folder = (labelstudio_root / project_name).resolve()  # resolve the explicitly requested project inside the LS root

        if not project_folder.exists():
            raise FileNotFoundError(
                fmt_error(f"Label-Studio project NOT found: {fmt_path(project_folder)}")
            )

        if not is_labelstudio_project(project_folder):
            raise FileNotFoundError(
                fmt_error(
                    f"Label-Studio project must contain "
                    f"{fmt_path('images/')}, {fmt_path('labels/')}, {fmt_path('classes.txt')}: "
                    f"{fmt_path(project_folder)}"
                )
            )

        return project_folder                         # explicit project requests return immediately once validated

    valid_projects = [p for p in labelstudio_root.iterdir() if is_labelstudio_project(p)]  # collect only structurally valid LS projects

    if not valid_projects:
        raise FileNotFoundError(
            fmt_error(f"No valid Label-Studio projects found in: {fmt_path(labelstudio_root)}")
        )

    unprocessed = []
    for project_folder in valid_projects:
        existing_dataset, existing_yaml = find_processed_dataset_for_project(project_folder, data_root)
        if existing_dataset is None and existing_yaml is None:
            unprocessed.append(project_folder)       # only projects without an existing processed dataset remain candidates

    if not unprocessed:
        return None                                  # caller can interpret None as "nothing new to process"

    unprocessed.sort(key=lambda p: p.stat().st_mtime, reverse=True)  # newest project is preferred when auto-selecting
    return unprocessed[0]

# --------- MAIN LS -> YOLO DATASET PROCESSOR ---------
def process_labelstudio_project(
    project_folder: Path,
    data_root: Path,
    train_pct: float = 0.8,
    dataset_name: str | None = None,
):
    """
    Convert a Label Studio project into a YOLO-style train/val dataset.

    Returns:
        (dataset_folder, data_yaml)
    """
    project_folder = Path(project_folder).resolve()
    data_root = Path(data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)    # ensure dataset root exists before any processing begins

    if not project_folder.exists():
        raise FileNotFoundError(
            fmt_error(f"Label-Studio project folder NOT found: {fmt_path(project_folder)}")
        )

    existing_dataset, existing_yaml = find_processed_dataset_for_project(project_folder, data_root)
    if existing_dataset is not None and existing_yaml is not None:
        print(fmt_info(f"Found existing processed dataset: {fmt_path(existing_dataset)}"))
        return existing_dataset, existing_yaml      # reuse already processed datasets instead of duplicating work

    img_folder = project_folder / "images"
    lbl_folder = project_folder / "labels"
    classes_file = project_folder / "classes.txt"

    if not img_folder.is_dir() or not lbl_folder.is_dir() or not classes_file.exists():
        raise FileNotFoundError(
            fmt_error(
                f"Label-Studio project must contain "
                f"{fmt_path('images/')}, {fmt_path('labels/')}, {fmt_path('classes.txt')}: "
                f"{fmt_path(project_folder)}"
            )
        )

    label_mode = _detect_label_mode(lbl_folder)     # determine whether incoming labels look like HBB or OBB
    print(fmt_info(f"Detected label mode: {fmt_bold(label_mode.upper())}"))

    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    base = dataset_name or timestamp                # user-specified dataset name wins, otherwise use a timestamp
    dataset_folder = data_root / base

    if dataset_folder.exists():
        i = 1
        while (data_root / f"{base}{i}").exists():
            i += 1
        dataset_folder = data_root / f"{base}{i}"  # append numeric suffixes when the base dataset folder name already exists

    dataset_folder.mkdir(parents=True, exist_ok=True)

    train_img = dataset_folder / "train/images"
    train_lbl = dataset_folder / "train/labels"
    val_img = dataset_folder / "val/images"
    val_lbl = dataset_folder / "val/labels"

    for p in (train_img, train_lbl, val_img, val_lbl):
        p.mkdir(parents=True, exist_ok=True)        # create the standard YOLO dataset folder layout

    all_imgs = [p for p in img_folder.iterdir() if p.is_file()]
    if not all_imgs:
        raise RuntimeError(fmt_error(f"Images NOT found in: {fmt_path(img_folder)}"))

    random.shuffle(all_imgs)                        # shuffle before splitting so train/val are not ordered by filename or import order
    split_idx = int(len(all_imgs) * train_pct)
    train_imgs = all_imgs[:split_idx]
    val_imgs = all_imgs[split_idx:]

    if len(all_imgs) > 1 and not val_imgs:
        val_imgs = [train_imgs.pop()]               # preserve at least one validation image when possible

    def _copy_pairs(img_list, out_img_dir, out_lbl_dir):
        for img_path in img_list:
            shutil.copy2(img_path, out_img_dir / img_path.name)  # preserve image metadata while copying into YOLO structure
            lbl_src = lbl_folder / f"{img_path.stem}.txt"
            if lbl_src.exists():
                shutil.copy2(lbl_src, out_lbl_dir / lbl_src.name)  # copy label only when a matching txt exists

    _copy_pairs(train_imgs, train_img, train_lbl)
    _copy_pairs(val_imgs, val_img, val_lbl)

    with open(classes_file, "r") as f:
        names = [x.strip() for x in f.readlines() if x.strip()]  # normalize class names by trimming blanks & empty lines

    if not names:
        raise RuntimeError(fmt_error(f"Class names NOT found in: {fmt_path(classes_file)}"))

    data_yaml = dataset_folder / "data.yaml"
    with open(data_yaml, "w") as f:
        yaml.dump(
            {
                "path": str(dataset_folder.resolve()),  # dataset root used by Ultralytics
                "train": str(train_img.resolve()),      # explicit train image folder
                "val": str(val_img.resolve()),          # explicit validation image folder
                "nc": len(names),                       # class count
                "names": names,                         # ordered class names
            },
            f,
            sort_keys=False,
        )

    print(fmt_save(f"Created dataset YAML: {fmt_path(data_yaml)}"))

    metadata = {
        "processed": True,                              # marks this folder as a generated processed dataset
        "original_project": str(project_folder),        # links dataset back to the LS project it came from
        "timestamp": timestamp,                         # records dataset creation time
        "label_mode": label_mode,                       # stores detected HBB/OBB mode
        "project_name": project_folder.name,            # friendly LS project name
        "dataset_name": dataset_folder.name,            # final dataset folder name
        "source_type": "labelstudio",                   # identifies dataset origin for later lookup/reuse
    }

    with open(dataset_folder / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)                # save structured metadata for future reuse & dataset lookup

    print(fmt_save(f"Saved metadata: {fmt_path(dataset_folder / 'metadata.json')}"))
    return dataset_folder, data_yaml