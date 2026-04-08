# utils/train/results.py
from __future__ import annotations                  # keeps annotation behavior consistent with the rest of the project
import csv                                          # used for reading Ultralytics results.csv outputs
import json                                         # used for writing structured metadata JSON files
from datetime import datetime                       # used for timestamping quick summaries & metadata
from pathlib import Path                            # used for safe cross-platform result/log path handling
from ..ui import fmt_exit, fmt_info, fmt_model, fmt_warn, fmt_error, fmt_dataset, fmt_path  # shared terminal formatting helpers

# --------- RESULTS PARSING ---------
def parse_results(run_dir: Path) -> dict:
    """
    Parse the final row of Ultralytics results.csv into a normalized metrics dictionary.
    """
    csv_path = run_dir / "results.csv"              # Ultralytics stores epoch metrics in results.csv inside the run directory
    if not csv_path.exists():
        return {}                                   # missing results.csv means there is nothing to parse yet

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))            # read all rows so the last epoch row can be selected cleanly
        if not reader:
            return {}                               # empty CSV means no usable metrics are available

        row = reader[-1]                            # most recent row represents the final training state
        try:
            p = float(row.get("metrics/precision(B)", 0))
            r = float(row.get("metrics/recall(B)", 0))
            f1 = 2 * p * r / (p + r) if p + r > 0 else 0  # compute F1 manually from precision/recall when possible

            return {
                "F1": f1,
                "Precision": p,
                "Recall": r,
                "mAP50": float(row.get("metrics/mAP50(B)", 0)),
                "mAP50-95": float(row.get("metrics/mAP50-95(B)", 0)),
                "Box Loss": float(row.get("train/box_loss", 0)),
                "Class Loss": float(row.get("train/cls_loss", 0)),
                "DFL Loss": float(row.get("train/dfl_loss", 0)),
            }                                       # normalize the most relevant metrics/losses into one shared summary dict
        except Exception as e:
            print(fmt_warn(f"Failed to parse results.csv: {e}"))
            return {}                               # parse failures degrade to an empty metrics dict instead of breaking the pipeline

# --------- QUICK SUMMARY EXPORT ---------
def save_quick_summary(
    log_dir: Path,
    mode: str,
    epochs: int,
    metrics: dict,
    new_imgs=0,
    total_imgs=0,
    weights_used="n/a",
    arch_used="n/a",
):
    """
    Save a compact human-readable training summary text file.
    """
    log_dir.mkdir(parents=True, exist_ok=True)      # ensure the destination log directory exists before writing output files
    path = log_dir / "quick-summary.txt"

    with open(path, "w") as f:
        f.write("Quick Training Summary\n=======================\n")
        f.write(f"Date: {datetime.now():%m-%d-%Y %H-%M-%S}\n")
        f.write(f"Training Type: {mode}\n")
        f.write(f"Epochs Run: {epochs}\n")
        f.write(f"Model Weights: {weights_used}\n")
        f.write(f"Model Architecture: {arch_used}\n\n")
        f.write("Best Metrics:\n-------------\n")
        for k in ["F1", "Precision", "Recall", "mAP50", "mAP50-95"]:
            f.write(f"{k}: {metrics.get(k, 0):.3f}\n")  # write primary evaluation metrics in a compact readable block
        f.write("\nLosses:\n-------\n")
        for k in ["Box Loss", "Class Loss", "DFL Loss"]:
            f.write(f"{k}: {metrics.get(k, 0):.4f}\n")  # write final loss values with slightly higher precision
        f.write(f"\nNew Images Added: {new_imgs}\n")
        f.write(f"Total Images Used: {total_imgs}\n")

    print(fmt_exit(f"Quick summary saved to {fmt_path(path)}"))

# --------- METADATA EXPORT ---------
def save_metadata(log_dir: Path, mode: str, epochs: int, new_imgs: int, total_imgs: int):
    """
    Save structured metadata.json after training.
    """
    path = log_dir / "metadata.json"
    log_dir.mkdir(parents=True, exist_ok=True)      # ensure metadata destination exists before writing JSON

    meta = {
        "timestamp": datetime.now().isoformat(),    # records when this training metadata file was produced
        "train_type": mode,                         # stores run mode such as test, force, or auto
        "epochs": epochs,                           # stores how many epochs were executed
        "new_images_added": new_imgs,               # stores how many new images triggered or participated in retraining
        "total_images_used": total_imgs,            # stores total dataset size used during this run
    }

    with open(path, "w") as f:
        json.dump(meta, f, indent=4)                # save metadata in readable indented JSON form

    print(fmt_exit(f"Metadata JSON saved to {fmt_path(path)}"))