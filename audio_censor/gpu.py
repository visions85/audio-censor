"""One film at a time on the GPU, when several audio-censor runs share the machine.

Speech recognition (Whisper) and the context AI (Ollama) each want several GB of VRAM;
two runs scanning at once could run the card out of memory. Scanning therefore takes
an exclusive lock, while copying files and encoding (CPU) overlap freely.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import time
from pathlib import Path

from .media import eprint


def _lock_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "audio-censor" / "gpu.lock"


@contextlib.contextmanager
def turn(what: str = "scanning"):
    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            eprint(f"  gpu: another run is using it; waiting to start {what} ...")
            t0 = time.time()
            fcntl.flock(fh, fcntl.LOCK_EX)
            eprint(f"  gpu: free after {int(time.time() - t0)} s")
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
