"""Relax the kernel's MGLRU thrashing guard while a run is working, and put it back after.

/sys/kernel/mm/lru_gen/min_ttl_ms makes kswapd kill a process when the recently used
pages cannot be kept for that many milliseconds. Set low (CachyOS ships 100 ms on some
setups), it fires during the big sequential reads and writes of a remux and its copy
back over the network, even with gigabytes of RAM free. With [system]
relax_thrash_guard = true the guard is set to 0 for the duration of process, scan and
render, through `sudo -n` (no prompt; when that is not allowed the run just warns).

Several runs can overlap, and a run killed outright cannot clean up, so a state file
records the original value and the PIDs of the runs holding the guard down; the last
live holder restores it, and a later run restores it for holders that died.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import subprocess
from pathlib import Path
from typing import Iterator

from .media import eprint

KNOB = Path("/sys/kernel/mm/lru_gen/min_ttl_ms")


def _state_path() -> Path:
    base = os.environ.get("XDG_RUNTIME_DIR") or os.environ.get("XDG_CACHE_HOME") \
        or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "audio-censor-thrash-guard.json"


def _read() -> int | None:
    try:
        return int(KNOB.read_text().strip())
    except (OSError, ValueError):
        return None


def _write(value: int) -> bool:
    try:
        proc = subprocess.run(["sudo", "-n", "tee", str(KNOB)], input=f"{value}\n", text=True,
                              capture_output=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and _read() == value


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextlib.contextmanager
def _locked_state() -> Iterator[dict]:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        try:
            state = json.loads(fh.read() or "{}")
        except ValueError:
            state = {}
        yield state
        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps(state))


def _release(state: dict, pid: int | None) -> None:
    """Drop one hold of `pid` (a process may hold it twice: a run, and a step inside it) and any
    dead holders; restore the original when nobody holds it."""
    holders = list(state.get("holders", []))
    if pid in holders:
        holders.remove(pid)
    holders = [p for p in holders if _alive(p)]
    state["holders"] = holders
    if not holders and "original" in state:
        original = int(state["original"])
        if _read() != original and not _write(original):
            eprint(f"  thrash guard: could not restore {KNOB} to {original}")
        state.clear()


@contextlib.contextmanager
def relaxed(enabled: bool) -> Iterator[None]:
    """Hold the thrashing guard at 0 inside the block (when enabled and possible)."""
    if not enabled or _read() is None:
        yield
        return
    pid = os.getpid()
    with _locked_state() as state:
        _release(state, None)                       # clean up after runs that were killed
        current = _read()
        if not state.get("holders"):
            if current == 0:
                yield_ok = True                     # already off (set by hand): nothing to restore
                state.clear()
            else:
                yield_ok = _write(0)
                if yield_ok:
                    state["original"] = current
                    eprint(f"  thrash guard: {KNOB.name} {current} -> 0 for this run")
                else:
                    eprint(f"  thrash guard: could not lower {KNOB} (sudo -n not allowed?); "
                           "large copies may be killed under memory pressure")
        else:
            yield_ok = True
        if yield_ok and "original" in state:
            state.setdefault("holders", []).append(pid)
    try:
        yield
    finally:
        with _locked_state() as state:
            if pid in state.get("holders", []):
                _release(state, pid)
                if not state:
                    eprint(f"  thrash guard: {KNOB.name} restored")
