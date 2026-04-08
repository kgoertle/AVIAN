# utils/train/checkpoints.py
from __future__ import annotations      # keeps annotation behavior consistent with the rest of the project
from pathlib import Path                # used for checkpoint path discovery & composition
from .io import ensure_weights          # used for fallback pretrained-weight resolution when no checkpoint is available

# --------- CHECKPOINT DISCOVERY ---------
def check_checkpoint(runs_dir: Path, prefer_last=True):
    """
    Find the newest usable checkpoint beneath a runs directory.

    Returns:
        checkpoint path or None
    """
    if not runs_dir.exists():
        return None                                # missing runs directory means no checkpoints exist yet

    subfolders = sorted(
        [f for f in runs_dir.iterdir() if f.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )                                              # newest modified run folders are checked first

    for folder in subfolders:
        weights_dir = folder / "weights"
        if weights_dir.exists():
            filename = "last.pt" if prefer_last else "best.pt"  # resume mode prefers last.pt while update mode prefers best.pt
            candidate = weights_dir / filename
            if candidate.exists():
                return candidate                   # return the first matching checkpoint found in newest-first order

    return None

# --------- CHECKPOINT / RESUME RESOLUTION ---------
def get_checkpoint_and_resume(
    mode,
    resume_flag,
    runs_dir: Path,
    default_weights=None,
    custom_weights=None,
    update_folder=None,
):
    """
    Resolve which checkpoint or pretrained weights should seed training.

    Returns:
        (checkpoint_path, resume_flag)
    """
    checkpoint = None

    if resume_flag:
        checkpoint = check_checkpoint(runs_dir, prefer_last=True)  # resume mode explicitly continues from the newest last.pt
        if not checkpoint:
            raise FileNotFoundError(f"No last.pt found for resuming in {runs_dir}")
        resume_flag = True

    elif mode == "update":
        if update_folder and isinstance(update_folder, str):
            target = runs_dir / update_folder / "weights" / "best.pt"  # targeted update mode can pin to a specific run folder
            if target.exists():
                checkpoint = target
            else:
                raise FileNotFoundError(f"[ERROR] best.pt not found in runs/{update_folder}/weights/")
        else:
            checkpoint = check_checkpoint(runs_dir, prefer_last=False)  # otherwise use the newest best.pt found globally

        if not checkpoint:
            print("[WARN] No best.pt found. Falling back to default weights.")
            if default_weights:
                checkpoint = ensure_weights(
                    Path(default_weights),
                    model_type=str(default_weights),
                )                                  # update mode falls back to official/default weights when no trained best.pt exists

    elif mode == "train" and custom_weights:
        checkpoint = ensure_weights(
            Path(custom_weights),
            model_type=str(custom_weights),
        )                                          # transfer-learning mode can start from explicit custom/pretrained weights

    if checkpoint is None and default_weights:
        checkpoint = ensure_weights(
            Path(default_weights),
            model_type=str(default_weights),
        )                                          # final fallback ensures a default pretrained starting point when nothing else resolved

    return checkpoint, resume_flag