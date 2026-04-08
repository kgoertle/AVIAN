# utils/detect/runtime_util.py
from __future__ import annotations      # allows forward references in type hints without quoting types
import queue                            # used for queue draining & empty-queue handling during shutdown
from pathlib import Path                # used for typing saved output path collections
from typing import Any                  # used where helper inputs may vary in type

EOF_MARKER = "EOF"                      # shared sentinel used to signal end-of-stream across reader, inference, & manager layers

# --------- EOF HELPERS ---------
def is_eof_marker(item: Any) -> bool:
    """
    Return True when a queue item matches the shared EOF sentinel format.

    Expected EOF payload format:
        (EOF_MARKER, <optional detail>)
    """
    return (
        isinstance(item, tuple)         # EOF payloads are always communicated as tuples
        and len(item) == 2              # current EOF contract expects exactly two tuple entries
        and isinstance(item[0], str)    # first element must be the sentinel string itself
        and item[0] == EOF_MARKER
    )

# --------- PATH COLLECTION HELPERS ---------
def safe_extend(saved: list[Path], value: Any) -> None:
    """
    Append one path or extend with many paths while ignoring falsey values.

    Supported inputs:
        - None / False / empty -> ignored
        - single Path-like object -> appended
        - list of Path-like objects -> extended
    """
    if not value:
        return                          # ignore empty save results so callers do not need repeated None checks

    if isinstance(value, list):
        saved.extend(value)             # multi-file save helpers can contribute all returned paths at once
    else:
        saved.append(value)             # single output paths are appended directly

# --------- RESOURCE CLEANUP HELPERS ---------
def release_quietly(obj: Any) -> None:
    """
    Call .release() on a cv2-like object without allowing cleanup failures
    to interrupt shutdown.
    """
    try:
        if obj is not None:
            obj.release()               # release capture/writer-like objects only when they actually exist
    except Exception:
        pass                            # cleanup failures should never block the rest of shutdown

def drain_queue(q: queue.Queue) -> None:
    """
    Remove all currently queued items from a queue.
    """
    while True:
        try:
            q.get_nowait()              # keep pulling queued items until the queue is fully empty
        except queue.Empty:
            return                      # empty queue means draining is complete