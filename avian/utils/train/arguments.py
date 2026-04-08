# utils/train/arguments.py
from __future__ import annotations                  # allows forward references in type hints without quoting types
import argparse                                     # used to build the training & Label Studio CLI interfaces
import sys                                          # used for raw CLI access & explicit termination on invalid setup
from datetime import datetime                       # used for fallback timestamped dataset/run names
from pathlib import Path                            # used for safe cross-platform dataset/model path handling
from ..help_text import print_train_help            # custom training help printer used instead of default argparse help
from ..paths import BASE_DIR, LS_ROOT, MODELS_DIR   # shared workspace paths used during dataset/model resolution
from ..ui import fmt_bold, fmt_dataset, fmt_error, fmt_exit, fmt_model, fmt_path, fmt_warn  # shared formatted console helpers
from .io import (
    FAMILY_TO_WEIGHTS,                              # official family -> default weights registry
    FAMILY_TO_YAML,                                 # official family -> YAML registry
    ensure_weights,                                 # resolves/downloads pretrained weights as needed
    ensure_yolo_yaml,                               # resolves/downloads official architecture YAMLs as needed
    family_is_obb,                                  # tests normalized family names for OBB status
    is_custom_yaml,                                 # distinguishes official family names from custom YAML requests
    normalize_model_name,                           # normalizes model/arch names into (family, variant)
    yaml_is_obb,                                    # probes custom YAML files for OBB architecture hints
)
from .labelstudio_util import (
    get_dataset_label_mode,                         # reads processed dataset metadata to determine HBB/OBB mode
    process_labelstudio_project,                    # converts a Label Studio project into a YOLO dataset
    resolve_labelstudio_project,                    # resolves which Label Studio project should be used
)

# --------- LABEL STUDIO COMMAND HELPERS ---------
def get_labelstudio_args():
    """
    Build & parse the standalone Label Studio processing CLI.
    """
    parser = argparse.ArgumentParser(
        prog="AVIAN-labelstudio",
        description="Process a Label Studio project into a YOLO dataset.",
        add_help=True,                              # standalone LS command uses normal argparse help behavior
    )

    parser.add_argument(
        "--labelstudio",
        "--labelstudio-project",
        "--project",
        "-ls",
        nargs="?",
        const=True,
        default=None,
        help="Optional Label Studio project name. If omitted, the most recent unprocessed project is used.",
    )

    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Optional output dataset name.",
    )

    return parser.parse_args()

def run_labelstudio_command():
    """
    Process a Label Studio project into a YOLO dataset as a standalone command.
    """
    args = get_labelstudio_args()

    data_root = BASE_DIR / "data"
    data_root.mkdir(parents=True, exist_ok=True)    # ensure dataset root exists before project processing begins
    LS_ROOT.mkdir(parents=True, exist_ok=True)      # ensure Label Studio root exists before project resolution begins

    project_name = args.labelstudio if isinstance(args.labelstudio, str) else None  # bare --labelstudio means auto-select newest unprocessed project

    try:
        project_folder = resolve_labelstudio_project(
            labelstudio_root=LS_ROOT,
            data_root=data_root,
            project_name=project_name,
        )

        if project_folder is None:
            print(fmt_exit("No unprocessed Label-Studio projects found. Nothing to do."))
            return

        dataset_name = args.name.strip() if args.name else None  # normalize custom dataset name when one was provided

        dataset_folder, data_yaml = process_labelstudio_project(
            project_folder=project_folder,
            data_root=data_root,
            dataset_name=dataset_name,
        )

        print(fmt_dataset(f"Project used: {fmt_path(project_folder)}"))
        print(fmt_dataset(f"Dataset created: {fmt_path(dataset_folder)}"))
        print(fmt_exit(f"Dataset YAML ready: {fmt_path(data_yaml)}"))

    except FileNotFoundError as exc:
        print(fmt_error(str(exc)))
        sys.exit(1)
    except Exception as exc:
        print(fmt_error(f"Label-Studio processing failed: {exc}"))
        sys.exit(1)

# --------- MAIN TRAINING ARGUMENTS ---------
def get_args():
    """
    Parse, validate, and normalize all training CLI arguments.

    Returns:
        (args, mode)
    """
    parser = argparse.ArgumentParser(
        description="YOLO Training Script",
        add_help=False,                              # custom help flow is handled manually so project-specific help text can be shown
    )

    if any(arg in sys.argv for arg in ("--help", "-h", "help")):
        print_train_help()
        sys.exit(0)

    mode_group = parser.add_mutually_exclusive_group(required=False)  # scratch/train flags are mutually exclusive while update is handled separately

    mode_group.add_argument(
        "--train",
        "--transfer-learning",
        "-t",
        action="store_true",
        help="Train using transfer learning.",
    )

    parser.add_argument(
        "--update",
        "--upgrade",
        "-u",
        type=str,
        nargs="?",
        const=True,
        help="Update an existing model run if new dataset images are available.",
    )

    mode_group.add_argument(
        "--scratch",
        "-s",
        action="store_true",
        help="Train from scratch using an architecture YAML.",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Model weights to use for transfer learning.",
    )

    parser.add_argument(
        "--arch",
        "--architecture",
        "--backbone",
        "-a",
        "-b",
        type=str,
        help="Architecture family or custom YAML to use.",
    )

    parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="Resume from the latest last.pt checkpoint.",
    )

    parser.add_argument(
        "--test",
        "-T",
        action="store_true",
        help="Use test-mode runs/logs locations and reduced training settings.",
    )

    parser.add_argument(
        "--dataset",
        "--data",
        "-d",
        type=str,
        default=None,
        help="Dataset folder name inside ~/AVIAN/data.",
    )

    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Optional run name.",
    )

    parser.add_argument(
        "--labelstudio",
        "--labelstudio-project",
        "--project",
        "-ls",
        nargs="?",
        const=True,
        default=None,
        help="Optional Label Studio project to process before training.",
    )

    args = parser.parse_args()

    if not hasattr(args, "weights"):
        args.weights = None                           # keeps downstream callers compatible with code expecting args.weights

    # --------- DETERMINE MODE ---------
    if args.update:
        mode = "update"                               # explicit update takes priority because it changes checkpoint logic
    elif args.arch and not args.model:
        mode = "scratch"                              # architecture-only requests imply scratch training
    elif args.model and not args.arch:
        mode = "train"                                # model-only requests imply transfer learning
    elif args.scratch:
        mode = "scratch"
    elif args.train:
        mode = "train"
    else:
        mode = "train"                                # default mode is transfer-learning style training

    # --------- VALIDATE MODEL / ARCH INPUTS ---------
    custom_arch = bool(args.arch and is_custom_yaml(args.arch, MODELS_DIR))  # custom YAML mode changes later architecture resolution rules

    if args.model:
        family, variant = normalize_model_name(args.model.lower())

        if family not in FAMILY_TO_YAML:
            print(fmt_error(f"Model family NOT recognized: '{fmt_bold(family)}'."))
            sys.exit(1)

        if variant not in {None, "n", "s", "m", "l", "x"}:
            print(fmt_error(f"Model variant NOT recognized '{fmt_bold(args.model)}'."))
            sys.exit(1)

    if args.arch and not custom_arch:
        family, _ = normalize_model_name(args.arch.lower())
        if family not in FAMILY_TO_YAML:
            print(fmt_error(f"Model architecture NOT recognized '{fmt_bold(args.arch)}'."))
            sys.exit(1)

    if args.update and args.arch:
        print(fmt_error("Update cannot be used with architecture selection."))
        sys.exit(1)                                   # update mode always works from existing runs/weights rather than arbitrary arch selection

    # --------- FINAL NAME ---------
    base_name = args.name.strip() if args.name else datetime.now().strftime("%d-%m-%Y_%H-%M-%S")  # timestamp fallback keeps run names unique enough by default

    data_root = BASE_DIR / "data"
    data_root.mkdir(exist_ok=True)
    LS_ROOT.mkdir(exist_ok=True)

    final_name = base_name
    suffix = 1
    while (data_root / final_name).exists():
        final_name = f"{base_name}{suffix}"           # avoid naming collisions with existing dataset folders
        suffix += 1

    args.final_name = final_name                      # preserve unique resolved name separately for downstream LS/dataset creation
    args.name = final_name

    # --------- DATASET RESOLUTION ---------
    if args.dataset:
        dataset_folder = data_root / args.dataset
        if not dataset_folder.exists():
            print(fmt_error(f"Dataset folder NOT found: {fmt_bold(dataset_folder)}"))
            sys.exit(1)

        data_yaml = dataset_folder / "data.yaml"
        if not data_yaml.exists():
            print(fmt_error(f"Data YAML NOT found in dataset folder: {fmt_bold(data_yaml)}"))
            sys.exit(1)

    elif args.labelstudio is not None:
        project_name = args.labelstudio if isinstance(args.labelstudio, str) else None  # bare --labelstudio again means auto-select newest project

        try:
            project_folder = resolve_labelstudio_project(
                labelstudio_root=LS_ROOT,
                data_root=data_root,
                project_name=project_name,
            )
        except FileNotFoundError as exc:
            print(fmt_error(str(exc)))
            sys.exit(1)

        if project_folder is None:
            print(fmt_exit("No unprocessed Label-Studio projects found. Nothing to train on."))
            sys.exit(0)

        print(fmt_dataset(f"Using Label-Studio project: {fmt_path(project_folder)}"))
        dataset_folder, data_yaml = process_labelstudio_project(
            project_folder=project_folder,
            data_root=data_root,
            dataset_name=args.final_name,
        )                                              # auto-generated LS datasets inherit the final unique run/dataset name

    else:
        all_datasets = [d for d in data_root.iterdir() if d.is_dir()]

        if not all_datasets:
            print(fmt_error("No datasets exist. Provide dataset or use Label Studio processing first."))
            sys.exit(1)

        if len(all_datasets) == 1:
            dataset_folder = all_datasets[0]
            data_yaml = dataset_folder / "data.yaml"
            print(fmt_dataset(f"Auto-selected dataset: {fmt_path(dataset_folder.name)}"))  # safe only when exactly one dataset exists
        else:
            print(fmt_error("Multiple datasets detected; specify one with --dataset."))
            print("Available datasets:", [d.name for d in all_datasets])
            sys.exit(1)

    label_mode = get_dataset_label_mode(dataset_folder)  # metadata-driven dataset mode helps enforce HBB/OBB compatibility
    dataset_is_obb = (label_mode == "obb") if label_mode is not None else None

    # --------- REQUESTED FAMILIES ---------
    requested_model_family = None
    if args.model:
        requested_model_family, _ = normalize_model_name(args.model)

    if args.arch:
        requested_arch_family = None if custom_arch else normalize_model_name(args.arch)[0]  # custom YAMLs bypass official-family normalization
    elif requested_model_family:
        requested_arch_family = requested_model_family  # reuse model family when no explicit architecture family was supplied
    else:
        requested_arch_family = "yolo11"               # default official architecture family

    if not custom_arch and args.model and requested_model_family and requested_arch_family:
        if requested_model_family != "yolo12-obb" and requested_arch_family != requested_model_family:
            print(
                fmt_error(
                    f"Model architecture '{fmt_bold(requested_arch_family)}' does NOT match model "
                    f"'{fmt_bold(requested_model_family)}'."
                )
            )
            sys.exit(1)                                # model/architecture family mismatches are rejected to prevent invalid training setups

    arch_family = requested_arch_family
    weight_family = None if mode == "scratch" else (requested_model_family or "yolo11")  # scratch mode does not begin from pretrained weights

    # --------- DATASET OBB/HBB ENFORCEMENT ---------
    if dataset_is_obb is not None and not custom_arch:
        arch_is_obb = family_is_obb(arch_family)
        weight_is_obb = family_is_obb(weight_family) if weight_family else None

        fallback_family = None
        if dataset_is_obb:
            if (arch_family and not arch_is_obb) or (weight_family and weight_is_obb is False):
                fallback_family = "yolo11-obb"         # OBB datasets require OBB-capable official families
        else:
            if (arch_family and arch_is_obb) or (weight_family and weight_is_obb):
                fallback_family = "yolo11"             # HBB datasets must not be paired with OBB official families

        if fallback_family:
            if arch_family != fallback_family:
                print(fmt_warn(f"Dataset is {label_mode.upper()}; overriding architecture family to {fmt_bold(fallback_family)}."))
                arch_family = fallback_family

            if mode != "scratch" and weight_family != fallback_family:
                print(fmt_warn(f"Dataset is {label_mode.upper()}; overriding weight family to {fmt_bold(fallback_family)}."))
                weight_family = fallback_family

    # --------- RESOLVE MODEL YAML ---------
    if custom_arch:
        arch_text = args.arch.lower()
        candidates = []

        if arch_text.endswith(".yaml"):
            candidates.extend([Path(arch_text), MODELS_DIR / arch_text])  # exact YAML requests may be absolute/local or inside MODELS_DIR
        else:
            candidates.extend([
                Path(f"{arch_text}.yaml"),
                MODELS_DIR / f"{arch_text}.yaml",
                Path(arch_text),
                MODELS_DIR / arch_text,
            ])                                       # looser resolution supports names both with & without explicit .yaml suffix

        model_yaml = next((candidate for candidate in candidates if candidate.exists()), None)

        if model_yaml is None:
            print(fmt_error(f"Custom model architecture YAML NOT found for '{args.arch}'."))
            sys.exit(1)

    else:
        yaml_name = FAMILY_TO_YAML.get(arch_family)
        if yaml_name is None:
            print(fmt_error(f"Model architecture YAML NOT registered to '{fmt_bold(arch_family)}'."))
            sys.exit(1)

        model_yaml = ensure_yolo_yaml(MODELS_DIR / yaml_name, model_type=arch_family)
        if model_yaml is None:
            print(fmt_error(f"Failed to resolve model architecture YAML to '{fmt_bold(arch_family)}'."))
            sys.exit(1)

    print(fmt_model(f"Using model architecture YAML: {fmt_path(model_yaml)}"))

    if custom_arch and dataset_is_obb is not None:
        arch_is_obb = yaml_is_obb(model_yaml)

        if dataset_is_obb and not arch_is_obb:
            print(fmt_error("OBB dataset requires an OBB-capable architecture."))
            sys.exit(1)

        if not dataset_is_obb and arch_is_obb:
            print(fmt_error("HBB dataset cannot be trained with an OBB architecture."))
            sys.exit(1)                               # custom YAMLs must also obey dataset-mode compatibility even without official family names

    # --------- RESOLVE WEIGHTS ---------
    if mode != "scratch":
        if weight_family not in FAMILY_TO_WEIGHTS and weight_family != "yolo12-obb":
            print(fmt_error(f"Default weights are NOT registered for '{fmt_bold(weight_family)}'."))
            sys.exit(1)

        model_type_for_weights = args.model if args.model else weight_family  # explicit model request wins, otherwise use resolved weight family
        args.weights = ensure_weights(BASE_DIR / "weights", model_type=model_type_for_weights)
    else:
        args.weights = None                            # scratch mode intentionally trains without a starting pretrained checkpoint

    if isinstance(args.weights, str) and args.weights.endswith(".pt"):
        args.weights = Path(args.weights)              # normalize any string path returned from older/helper code into Path

    args.model_yaml = model_yaml
    args.DATA_YAML = data_yaml
    args.train_folder = dataset_folder / "train" / "images"
    args.val_folder = dataset_folder / "val" / "images"
    args.dataset_folder = dataset_folder
    args.custom_arch = custom_arch                    # expose final normalized values back on args for downstream training code

    return args, mode