# train.py
from __future__ import annotations                          # allows forward references in type hints without quoting types
import shutil                                               # used for copying data.yaml into the final run folder
import sys                                                  # used for explicit CLI termination on unrecoverable setup errors
import time                                                 # used for overall training duration measurement
from datetime import datetime                               # used for timestamped fallback run names
from pathlib import Path                                    # used for safe cross-platform path handling
import wandb                                                # used for clean W&B shutdown after training
from ultralytics import YOLO                                # Ultralytics model wrapper used for training
from ultralytics.utils import LOGGER                        # Ultralytics logger patched during training for cleaner console output
from .utils.train.arguments import get_args                 # parses & normalizes all training CLI arguments
from .utils.paths import get_training_paths                 # builds the standard path bundle for training outputs
from .utils.train import (
    configure_wandb_runtime,                                # applies W&B runtime environment configuration
    count_images,                                           # counts training/validation images for summaries & update checks
    get_checkpoint_and_resume,                              # resolves checkpoint/pretrained starting point for training
    init_wandb,                                             # configures W&B integration without manually starting duplicate runs
    load_latest_metadata,                                   # reads the newest metadata.json for update-mode image comparison
    parse_results,                                          # parses final Ultralytics metrics from results.csv
    save_metadata,                                          # writes structured metadata.json after training
    save_quick_summary,                                     # writes a human-readable quick training summary
    select_device,                                          # chooses training device plus batch/worker defaults
)
from .utils.ui import fmt_path, training_ui as ui           # shared training UI instance & formatted path helper

# --------- TRAINING ORCHESTRATION ---------
def train_yolo(args, mode: str = "train", checkpoint=None, resume_flag: bool = False) -> None:
    """
    Orchestrate YOLO training based on the selected mode.
    """
    # --------- VALIDATE DATASET YAML ---------
    if not args.DATA_YAML.exists():
        ui.error(f"DATA_YAML not found: {args.DATA_YAML}")   # training cannot start without a resolved dataset YAML
        return

    # --------- PATH SETUP ---------
    paths = get_training_paths(args.DATA_YAML.parent, test=args.test)  # derive standard runs/logs/data paths from the resolved dataset root

    # --------- TRAINING PARAMETER SETUP ---------
    reset_weights = mode == "scratch"                         # scratch mode begins from architecture only rather than pretrained weights
    epochs, imgsz = (10, 640) if args.test else (120, 640)   # test mode uses a short lightweight run while normal mode uses full defaults

    if reset_weights and not args.test:
        epochs = 150                                          # scratch training gets extra epochs in full mode because it starts from zero

    total_imgs = count_images(args.train_folder) + count_images(args.val_folder)  # total dataset size used for metadata/update tracking
    new_imgs = 0                                              # populated only when update mode compares against a previous run

    # --------- UPDATE MODE: CHECK FOR NEW IMAGES ---------
    if mode == "update":
        logs_root = paths["logs_root"] / args.dataset_folder.name  # update mode looks at prior logs for this dataset name

        prev_meta = load_latest_metadata(logs_root)
        prev_total = prev_meta.get("total_images_used", 0) if prev_meta else 0
        new_imgs = total_imgs - prev_total                         # difference determines whether retraining is actually needed

        if new_imgs <= 0:
            ui.exit("No new images detected. Skipping training.")
            return                                                 # skip update runs entirely when dataset size did not increase

        ui.info(f"{new_imgs} new images detected. Proceeding with update.")

    # --------- MODEL SOURCE SELECTION ---------
    use_pretrained = False                                         # controls Ultralytics' pretrained flag during model.train()
    model_source = None                                            # final source string passed into YOLO initialization/training

    if mode == "scratch":
        model_source = str(args.model_yaml)                        # scratch mode always starts from the architecture YAML itself
        use_pretrained = False
        checkpoint = None                                          # scratch mode ignores checkpoint/pretrained startup state entirely
    else:
        if checkpoint:
            model_source = str(Path(checkpoint))                   # resume/update mode starts directly from a resolved checkpoint path
            use_pretrained = True
        elif getattr(args, "weights", None):
            model_source = str(args.weights)                       # transfer-learning mode starts from explicit pretrained weights
            use_pretrained = True
        else:
            model_source = str(args.model_yaml)                    # fallback lets Ultralytics resolve official pretrained behavior from the YAML
            use_pretrained = True

    if model_source is None:
        ui.error("Could not resolve a model source (weights or architecture).")
        return                                                     # guard against impossible/invalid startup state

    # --------- DEVICE + RUN NAME ---------
    device, batch_size, workers = select_device()                  # choose runtime device plus default batch/worker settings
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    run_name = args.name or timestamp                              # caller-provided name wins, otherwise use a timestamp fallback

    ui.model(f"Saved as: {fmt_path(run_name)}")

    # --------- LOGGER / UI SETUP ---------
    ui.configure_external_logs()                                   # suppress noisy external logs before training begins
    ui.show_training_header()                                      # clear/prepare terminal for training output

    # --------- WANDB INIT ---------
    try:
        configure_wandb_runtime()                                  # apply environment/runtime config first
        init_wandb(run_name)                                       # then initialize integration behavior without forcing a duplicate run
    except Exception as exc:
        ui.warn(f"Failed to initialize W&B: {exc}")                # W&B failure should not block training itself

    ui.show_training_context(
        model_source=model_source,
        dataset_name=args.dataset_folder.name,
        batch_size=batch_size,
        workers=workers,
        epochs=epochs,
    )                                                              # print compact training context before model initialization

    # --------- MODEL INIT ---------
    model = YOLO(model_source, task="detect")                      # initialize the Ultralytics model using the resolved source
    ui.patch_ultralytics_output()                                  # patch Ultralytics logging so epoch output stays cleaner in the terminal

    start_time = time.time()                                       # used later to report total training duration
    skip_completion_message = False                                # set when keyboard interruption already printed a custom completion state

    # --------- TRAINING CALL ---------
    try:
        model.train(
            data=str(args.DATA_YAML),
            model=model_source,
            epochs=epochs,
            resume=resume_flag,
            patience=10,
            imgsz=imgsz,
            batch=batch_size,
            workers=workers,
            project=str(paths["runs_root"]),
            name=run_name,
            exist_ok=False,
            pretrained=use_pretrained,
            device=device,
            augment=False,
            mosaic=False,
            mixup=True,
            fliplr=0.5,
            flipud=0.0,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            plots=False,
            verbose=False,
            show=False,
            show_labels=True,
            show_conf=True,
        )                                                          # all core training hyperparameters & runtime options are passed in one centralized call

    except KeyboardInterrupt:
        try:
            LOGGER.info.interrupted = True                         # hint patched logger to suppress extra noise after user interruption
        except Exception:
            pass

        ui.unpatch_ultralytics_output()
        skip_completion_message = True                             # prevents normal completion message from printing after an interrupt
        ui.show_training_header_static()
        ui.exit("Training interrupted by user. Partial results preserved.")

    except Exception as exc:
        try:
            LOGGER.info.interrupted = True                         # same post-failure suppression helps avoid extra noisy logger output
        except Exception:
            pass

        ui.unpatch_ultralytics_output()
        ui.show_training_header()
        ui.error(f"Training failed: {exc}")
        return                                                     # unrecoverable training failure stops the rest of the export pipeline

    # --------- AFTER TRAINING ---------
    elapsed_minutes = (time.time() - start_time) / 60.0            # compute total wall-clock training time in minutes
    ui.unpatch_ultralytics_output()                                # always restore original Ultralytics logging once training has ended

    if not skip_completion_message:
        ui.show_training_header_static()
        ui.exit(f"Training completed in {elapsed_minutes:.2f} minutes.")  # standard completion message for uninterrupted runs

    # --------- RESOLVE RUN DIRECTORY ---------
    try:
        run_folder = Path(model.trainer.save_dir)                  # Ultralytics trainer exposes the final run output folder here
        run_name = run_folder.name
        log_dir = paths["logs_root"] / run_name
        log_dir.mkdir(parents=True, exist_ok=True)                 # create matching log folder for summaries/metadata outside the run folder
    except Exception:
        return                                                     # if run directory cannot be resolved, metric export cannot continue safely

    # --------- SAVE METRICS + METADATA ---------
    try:
        metrics = parse_results(run_folder) or {}                  # results.csv is parsed into a compact metrics dictionary

        save_quick_summary(
            log_dir=log_dir,
            mode=mode,
            epochs=epochs,
            metrics=metrics,
            new_imgs=new_imgs,
            total_imgs=total_imgs,
            weights_used=args.weights.name if args.weights else "n/a",
            arch_used=args.model_yaml.name if args.model_yaml else "n/a",
        )

        save_metadata(log_dir, mode, epochs, new_imgs, total_imgs)
    except Exception as exc:
        ui.warn(f"Failed to save metadata JSON: {exc}")            # summary/metadata export failures are warned but do not crash shutdown

    # --------- COPY DATA.YAML INTO RUN FOLDER ---------
    try:
        dst_yaml = run_folder / "data.yaml"

        if not dst_yaml.exists():
            shutil.copy(args.DATA_YAML, dst_yaml)                  # preserve the exact dataset YAML used for this run inside the run folder itself
            ui.exit(f"Copied dataset YAML to: {fmt_path(dst_yaml)}")
    except Exception as exc:
        ui.warn(f"Could not copy dataset YAML: {exc}")

    # --------- WANDB SHUTDOWN ---------
    try:
        if wandb.run:
            wandb.finish()                                         # explicitly finish the active W&B run when one exists
    except Exception as exc:
        ui.warn(f"Could not close W&B cleanly: {exc}")

# --------- MAIN ENTRY POINT ---------
def main() -> None:
    """
    Parse arguments, resolve checkpoint behavior, and launch training.
    """
    args, mode = get_args()

    try:
        checkpoint, resume_flag = get_checkpoint_and_resume(
            mode=mode,
            resume_flag=args.resume,
            runs_dir=get_training_paths(args.DATA_YAML.parent, test=args.test)["runs_root"],  # reuse standard training path resolution for checkpoint lookup
            default_weights=args.weights,
            custom_weights=args.weights,
            update_folder=args.update if isinstance(args.update, str) else None,
        )

        if mode == "update" and checkpoint:
            ui.model(f"Updating model from: {fmt_path(checkpoint)}")  # update mode reports the checkpoint being refined
        elif mode == "train":
            if args.weights:
                ui.model(f"Training model from transferred weights: {fmt_path(args.weights)}")
            else:
                ui.model(f"Training model using architecture/default weights: {fmt_path(args.model_yaml)}")
        elif mode == "scratch":
            ui.model(f"Training from scratch using architecture: {fmt_path(args.model_yaml)}")
    except FileNotFoundError as exc:
        ui.error(str(exc))
        sys.exit(1)

    train_yolo(args, mode=mode, checkpoint=checkpoint, resume_flag=resume_flag)

if __name__ == "__main__":
    main()