"""Work on a local copy of a film that lives on a network share.

A full `process` reads the film three times (audio for Whisper, the subtitle track,
the remux) and writes the clean file once. Over NFS/SMB each of those is a full
transfer. Staging copies the film (and its small sidecar files) to a local folder
once, runs everything there, then copies only what changed or is new back beside the
original and deletes the local copy.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .media import eprint

NETWORK_FS = {"nfs", "nfs4", "cifs", "smb3", "smbfs", "fuse.sshfs", "fuse.rclone", "9p", "afs", "ceph",
              "glusterfs", "fuse.glusterfs", "davfs", "fuse.s3fs", "fuse.mergerfs", "fuse.shfs"}
SIDECAR_LIMIT = 200 * 1024 * 1024      # copy files sharing the film's name up to this size (subs, caches)
MARGIN = 2 * 1024 ** 3                  # keep this much free on the staging disk
FLUSH_EVERY = 256 * 1024 * 1024         # big copies are synced and dropped from the page cache this often


@dataclass
class Staged:
    local: Path        # the path to work on
    origin: Path       # the film's real location (for Plex/NFO lookups and reporting)

    @property
    def staged(self) -> bool:
        return self.local != self.origin


def filesystem_type(path: Path) -> str:
    """The type of the filesystem holding `path`, from /proc/mounts ("" if unknown)."""
    try:
        target = os.path.realpath(path)
        best, kind = "", ""
        with open("/proc/mounts", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mnt = parts[1].replace("\\040", " ")
                inside = target == mnt or target.startswith(mnt.rstrip("/") + "/")
                if inside and len(mnt) >= len(best):
                    best, kind = mnt, parts[2]
        return kind
    except OSError:
        return ""


def stage_dir(cfg: dict, args=None) -> Path | None:
    """The staging folder if staging is on for this run, else None."""
    if getattr(args, "no_stage", False):
        return None
    raw = getattr(args, "stage_dir", None) or cfg.get("staging", {}).get("dir", "")
    return Path(raw).expanduser() if raw else None


def should_stage(media: Path, root: Path, mode: str) -> bool:
    if mode == "never":
        return False
    try:
        if media.stat().st_dev == root.stat().st_dev:
            return False                        # already on the staging disk
    except OSError:
        pass
    return mode == "always" or filesystem_type(media) in NETWORK_FS


def _related(media: Path) -> list[Path]:
    """Small files beside the film that belong to it (Movie.en.srt, Movie.censor.json ...)."""
    out = []
    prefix = media.stem + "."
    for p in media.parent.iterdir():
        if p == media or not p.name.startswith(prefix) or not p.is_file():
            continue
        try:
            if p.stat().st_size <= SIDECAR_LIMIT:
                out.append(p)
        except OSError:
            pass
    return out


def _fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def _copy_with_progress(src: Path, dst: Path) -> None:
    total = src.stat().st_size
    done, t0, last = 0, time.time(), 0.0
    live = sys.stderr.isatty()
    synced = 0
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        while chunk := fin.read(16 * 1024 * 1024):
            fout.write(chunk)
            done += len(chunk)
            if done - synced >= FLUSH_EVERY or done == total:
                # Write out and drop what is copied: a 50 GB copy to a slow share otherwise
                # piles up unwritten pages until the kernel's out-of-memory killer steps in.
                fout.flush()
                os.fsync(fout.fileno())
                for fd in (fin.fileno(), fout.fileno()):
                    try:
                        os.posix_fadvise(fd, 0, done, os.POSIX_FADV_DONTNEED)
                    except (AttributeError, OSError):
                        pass
                synced = done
            now = time.time()
            if now - last >= (1 if live else 30) or done == total:
                last = now
                rate = done / max(now - t0, 0.001)
                eta = (total - done) / rate if rate else 0
                line = (f"  staging: {done * 100 / total:5.1f}%  {_fmt_size(done)} of {_fmt_size(total)}"
                        f"  {_fmt_size(rate)}/s  {int(eta // 60)}:{int(eta % 60):02d} left")
                print(f"\r{line}\033[K" if live else line, end="" if live else "\n", file=sys.stderr, flush=True)
    if live:
        print(file=sys.stderr)
    shutil.copystat(src, dst)


def _copy(src: Path, dst: Path) -> None:
    """Copy with metadata via a temporary name, so a half-copied file never has the real name."""
    tmp = dst.with_name(f".{dst.name}.partial")
    try:
        if src.stat().st_size > SIDECAR_LIMIT:
            _copy_with_progress(src, tmp)
        else:
            shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _signature(p: Path) -> tuple[int, int]:
    st = p.stat()
    return st.st_size, int(st.st_mtime)


def _local_folder(root: Path, media: Path) -> Path:
    key = hashlib.sha1(str(media.parent.resolve()).encode()).hexdigest()[:10]
    name = "".join(c if c.isalnum() or c in " ._-()" else "_" for c in media.parent.name)[:60]
    return root / f"{name} [{key}]"


@contextmanager
def staged(media: Path, cfg: dict, args=None, copies: int = 2) -> Iterator[Staged]:
    """Yield a local working path for `media`, copying it in first and changes back after.

    `copies` is how many film-sized files the work needs room for locally (the film
    plus a clean output = 2). Without the room, or when staging is off or pointless
    (a local file), the original path is used as is.
    """
    root = stage_dir(cfg, args)
    mode = (cfg.get("staging", {}).get("mode") or "network").lower()
    if getattr(args, "dry_run", False) or root is None or not should_stage(media, root, mode):
        yield Staged(media, media)
        return
    root.mkdir(parents=True, exist_ok=True)
    folder = _local_folder(root, media)
    local = folder / media.name
    size = media.stat().st_size
    reuse = local.exists() and _signature(local) == _signature(media)
    need = size * copies + MARGIN - (local.stat().st_size if local.exists() else 0)
    free = shutil.disk_usage(root).free
    if need > free:
        eprint(f"  staging: needs {_fmt_size(need)} free in {root}, has {_fmt_size(free)}; working over the network")
        yield Staged(media, media)
        return

    folder.mkdir(parents=True, exist_ok=True)
    # Leftovers from a run that was killed outright (half-written outputs, stale caches)
    # must not count as results: start from the film alone plus fresh sidecar copies.
    for p in folder.iterdir():
        if p.name != local.name:
            p.unlink(missing_ok=True) if not p.is_dir() else shutil.rmtree(p, ignore_errors=True)
    if reuse:
        eprint(f"  staging: reusing local copy in {folder}")
    else:
        eprint(f"  staging: copying {_fmt_size(size)} to {folder} ...")
        t0 = time.time()
        _copy(media, local)
        secs = max(time.time() - t0, 0.001)
        eprint(f"  staging: copied in {secs / 60:.1f} min ({_fmt_size(size / secs)}/s)")
    for p in _related(media):
        _copy(p, folder / p.name)
    before = {p.name: _signature(p) for p in folder.iterdir() if p.is_file()}

    ok = False
    try:
        yield Staged(local, media)
        ok = True
    finally:
        try:
            _copy_back(folder, media.parent, before, ok)
        finally:
            if ok:
                shutil.rmtree(folder, ignore_errors=True)
            else:
                # Keep the film for a quick retry (it is reused when unchanged), drop the rest.
                for p in folder.iterdir():
                    if p.name != local.name:
                        p.unlink(missing_ok=True) if p.is_file() else shutil.rmtree(p, ignore_errors=True)
                eprint(f"  staging: kept the local copy in {folder} for a retry")


def _copy_back(folder: Path, dest: Path, before: dict[str, tuple[int, int]], ok: bool) -> None:
    """Copy new and changed files back beside the original (after a failure, only the .json files)."""
    changed = []
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.name.startswith(".") or p.name.endswith(".partial"):
            continue
        if before.get(p.name) == _signature(p):
            continue
        if not ok and p.suffix.lower() != ".json":
            continue                # after a failure keep only scan results and caches, never half outputs
        changed.append(p)
    big = [p for p in changed if p.stat().st_size > SIDECAR_LIMIT]
    if big:
        total = sum(p.stat().st_size for p in big)
        eprint(f"  staging: copying {_fmt_size(total)} back to {dest} ...")
    # Order matters if the run is killed part-way: the films go first, then subtitles and
    # caches, and the span file (which records "rendered") last, so it never claims a
    # result that has not reached the share.
    def order(p: Path) -> tuple[int, str]:
        if p.name.endswith(".censor.json"):
            return 2, p.name
        return (0 if p.stat().st_size > SIDECAR_LIMIT else 1), p.name
    changed.sort(key=order)
    for p in changed:
        _copy(p, dest / p.name)
    if changed:
        eprint(f"  staging: copied back {', '.join(p.name for p in changed)}")
