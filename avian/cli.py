# cli.py
import argparse
import sys

from .utils.help_text import print_train_help, print_detect_help, print_labelstudio_help
from .train import main as train_main
from .detect import main as detect_main
from .utils.train.arguments import run_labelstudio_command
from .version import AVIAN_VERSION

def print_global_help():
    print(
        """
AVIAN - Automated Visual Identification & Analysis Network
============================================================

Available Commands:
  avian train         Train, update, or resume a YOLO model.
  avian detect        Run YOLO detection on one or more video/camera sources.
  avian label-studio  Process a Label Studio project into a YOLO dataset.
  avian version       Show the avian version.
  avian help          Show this help menu.

----------------------------------------------

Command Specific Help:
  avian train help
  avian detect help
  avian label-studio help

----------------------------------------------

Examples:
  avian train model=yolo11n architecture=custom_arch dataset=birds
  avian train architecture=yolo12
  avian train model=yolov8x name="best run ever!!" test

  # label-studio
  avian label-studio
  avian label-studio=example
  avian label-studio name=example
  avian train label-studio
  avian train label-studio=example model=yolo11n arch=yolo11n

  # detect (single model)
  avian detect model=sparrow-v2 sources=usb0 usb1
  avian detect test trailcam.mp4 trailcam2.mov

  # detect (multi-model)
  avian detect model=sparrow-v2 model=yolo11n sources=usb0
  avian detect models=sparrow-v2,yolo11n usb0

AVIAN Documentation & Support:
  https://github.com/kgoertle/avian
"""
    )

def expand_key_value_args(argv):
    """
    Expands friendly CLI forms into argparse-friendly flags.

    Supports:
      model=..., weights=...              -> --model <value> (repeatable)
      models=a,b,c                        -> repeat --model a --model b --model c
      models=yolo11n yolov8-obb           -> --models yolo11n yolov8-obb
      sources=video.mp4 usb0              -> --sources video.mp4 usb0
      test / test=true                    -> --test
    """
    expanded = []

    mappings = {
        # naming (train / label-studio)
        "name": "--name",
        "run": "--name",
        "run_name": "--name",

        # models (detect/train)
        "model": "--model",
        "weights": "--model",

        # train
        "update": "--update",
        "arch": "--arch",
        "architecture": "--arch",
        "backbone": "--arch",
        "data": "--dataset",
        "dataset": "--dataset",
        "labelstudio": "--labelstudio",
        "label-studio": "--labelstudio",
        "project": "--labelstudio",

        # detect
        "sources": "--sources",
        "source": "--sources",

        # shared
        "test": "--test",
    }

    boolean_true = {"1", "true", "yes", "on", ""}

    collecting_models = False
    collecting_sources = False

    for arg in argv:
        low = arg.lower()

        # stop collection if user explicitly starts a flag
        if arg.startswith("--"):
            collecting_models = False
            # if they started --sources, treat following positionals as sources (argparse will handle)
            if arg == "--sources":
                collecting_sources = True
            expanded.append(arg)
            continue

        # ---------- special case: plain "test" ----------
        if low == "test":
            expanded.append("--test")
            continue

        # ---------- special case: plain "label-studio" / "labelstudio" ----------
        if low in {"label-studio", "labelstudio"}:
            expanded.append("--labelstudio")
            continue

        # ---------- key=value pattern ----------
        if "=" in arg:
            key, value = arg.split("=", 1)
            key = key.lower().strip()

            # entering a new key=value stops previous collection modes
            collecting_models = False
            collecting_sources = False

            # test=true / test=1 / test=
            if key == "test":
                if value.lower().strip() in boolean_true:
                    expanded.append("--test")
                continue

            # sources=...  (start sources mode)
            if key in ("sources", "source"):
                expanded.append("--sources")
                if value.strip():
                    expanded.append(value.strip())
                collecting_sources = True
                continue

            # models=... (comma or space form)
            if key == "models":
                # if comma-separated, expand to repeat --model
                if "," in value:
                    parts = [p.strip() for p in value.split(",") if p.strip()]
                    for p in parts:
                        expanded.append("--model")
                        expanded.append(p)
                    continue

                # space-separated: use --models + keep collecting additional bare args as models
                expanded.append("--models")
                if value.strip():
                    expanded.append(value.strip())
                collecting_models = True
                continue

            # normal key=value
            if key in mappings:
                expanded.append(mappings[key])
                if value.strip():
                    expanded.append(value.strip())
                continue

        # ---------- collection modes ----------
        if collecting_models:
            # bare token following models=... is another model
            expanded.append(arg)
            continue

        if collecting_sources:
            # bare token following sources=... is another source
            expanded.append(arg)
            continue

        # default passthrough
        expanded.append(arg)

    return expanded

def main():
    parser = argparse.ArgumentParser(
        prog="avian",
        description="Automated Visual Identification & Analysis Network",
        add_help=True,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- TRAIN ----
    train_parser = subparsers.add_parser("train", help="Train or update a model.")
    train_parser.set_defaults(func="train")

    # ---- DETECT ----
    detect_parser = subparsers.add_parser("detect", help="Run YOLO detection.")
    detect_parser.set_defaults(func="detect")

    # ---- LABEL-STUDIO ----
    ls_parser = subparsers.add_parser("label-studio", help="Process a Label Studio project into a YOLO dataset.")
    ls_parser.set_defaults(func="labelstudio")

    # ---- VERSION ----
    version_parser = subparsers.add_parser("version", help="Show the current version of AVIAN.")
    version_parser.set_defaults(func="version")

    # ---- HELP ----
    help_parser = subparsers.add_parser("help", help="Show all AVIAN commands.")
    help_parser.set_defaults(func="help")

    argv = sys.argv[1:]

    # allow friendly "label-studio=example" top-level usage
    if argv and argv[0].lower().startswith("label-studio="):
        _, value = argv[0].split("=", 1)
        argv = ["label-studio", "--labelstudio", value] + argv[1:]
    elif argv and argv[0].lower().startswith("labelstudio="):
        _, value = argv[0].split("=", 1)
        argv = ["label-studio", "--labelstudio", value] + argv[1:]

    # ---- Parse command (not sub-arguments) ----
    args, unknown = parser.parse_known_args(argv)

    # Expand key=value into standard flags
    unknown = expand_key_value_args(unknown)

    # ROUTING
    if args.func == "train":
        if "--help" in unknown or "-h" in unknown or "help" in unknown:
            print_train_help()
            return

        sys.argv = ["avian-train"] + unknown
        return train_main()

    elif args.func == "detect":
        if "--help" in unknown or "-h" in unknown or "help" in unknown:
            print_detect_help()
            return

        sys.argv = ["avian-detect"] + unknown
        return detect_main()

    elif args.func == "labelstudio":
        if "--help" in unknown or "-h" in unknown or "help" in unknown:
            print_labelstudio_help()
            return

        sys.argv = ["avian-labelstudio"] + unknown
        return run_labelstudio_command()

    elif args.func == "version":
        print(f"avian {AVIAN_VERSION}")
        return

    elif args.func == "help":
        return print_global_help()

    else:
        parser.print_help()