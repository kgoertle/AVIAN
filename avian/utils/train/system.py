# utils/train/system.py
from __future__ import annotations      # keeps annotation behavior consistent with the rest of the project
import os                               # used for CPU core counting when selecting worker defaults
import torch                            # used to detect available training backends such as MPS, CUDA, or CPU

# --------- DEVICE SELECTION ---------
def select_device():
    """
    Select the best available training device and return default runtime settings.

    Returns:
        (device, batch_size, workers)
    """
    if torch.backends.mps.is_available():
        device = "mps"                  # prefer Apple Metal when available on supported macOS hardware
        batch_size = 4                  # conservative default batch size for MPS stability
        workers = 0                     # MPS setups often behave best with zero dataloader workers
    elif torch.cuda.is_available():
        device = "cuda"                 # prefer CUDA when an NVIDIA GPU is available
        batch_size = 32                 # GPU training can usually support a much larger batch size
        workers = 16                    # higher worker count helps keep the GPU fed during training
    else:
        device = "cpu"                  # final fallback when no accelerator backend is available
        batch_size = 2                  # CPU training uses a small batch size to stay lightweight
        workers = min(4, os.cpu_count())  # cap workers so CPU fallback does not oversubscribe the system

    print(f"[INFO] Using device: {device}, batch_size={batch_size}, workers={workers}")
    return device, batch_size, workers