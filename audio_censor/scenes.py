"""Scenes to skip: stored beside the film and written into it as "Skip" chapters.

Jellyfin's Chapter Segments Provider plugin turns a chapter named exactly "Skip" into a
skippable media segment (configure its Commercial mapping as ^Skip$), and players set to
skip that segment type jump over the scene. The film itself is untouched apart from its
chapter list, which mkvpropedit rewrites in place (no remux).

The first time a film gets scenes, its own chapters are saved in <file>.skip.json, so
the chapter list can always be rebuilt as: original chapters + one "Skip" chapter at
each scene start + a chapter at each scene end that resumes the chapter it interrupted.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from .media import MediaError, run

SKIP_TITLE = "Skip"
_TIME_RE = re.compile(r"^(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d+)?)$")


@dataclass
class Chapter:
    start: float
    title: str


def parse_time(text: str) -> float:
    """'1:02:03.5', '62:03', '3723' -> seconds."""
    m = _TIME_RE.match(text.strip())
    if not m:
        raise ValueError(f"not a time: {text!r} (use h:mm:ss, mm:ss or seconds)")
    a, b, c = m.groups()
    parts = [x for x in (a, b) if x is not None]
    seconds = float(c)
    for i, p in enumerate(reversed(parts)):
        seconds += int(p) * (60 if i == 0 else 3600)
    return seconds


def parse_range(text: str) -> tuple[float, float]:
    """'1:02:03-1:02:40' -> (3723.0, 3760.0)."""
    if "-" not in text:
        raise ValueError(f"not a range: {text!r} (use start-end, e.g. 1:02:03-1:02:40)")
    a, b = text.split("-", 1)
    start, end = parse_time(a), parse_time(b)
    if end <= start:
        raise ValueError(f"range {text!r} ends before it starts")
    return start, end


def fmt_time(seconds: float) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:06.3f}"


def sidecar(media: Path) -> Path:
    return media.with_name(media.stem + ".skip.json")


def load(media: Path) -> dict:
    try:
        return json.loads(sidecar(media).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save(media: Path, doc: dict) -> None:
    sidecar(media).write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")


def read_chapters(media: Path) -> list[Chapter]:
    proc = run(["ffprobe", "-v", "error", "-of", "json", "-show_chapters", str(media)])
    out = []
    for c in json.loads(proc.stdout or "{}").get("chapters", []):
        out.append(Chapter(float(c.get("start_time", 0)), (c.get("tags") or {}).get("title", "")))
    return sorted(out, key=lambda c: c.start)


def merge_scenes(original: list[Chapter], scenes: list[tuple[float, float]], duration: float) -> list[Chapter]:
    """Original chapters, a "Skip" chapter at each scene start, and the interrupted
    chapter resumed at each scene end. Original chapters inside a scene are dropped
    (they would end the skip early)."""
    scenes = merge_overlaps(scenes)
    base = original or [Chapter(0.0, "Chapter 1")]

    def covering(t: float) -> Chapter:
        cur = base[0]
        for c in base:
            if c.start <= t + 1e-6:
                cur = c
        return cur

    inside = lambda t: any(s - 1e-6 <= t < e for s, e in scenes)
    chapters = [c for c in base if not inside(c.start)]
    for s, e in scenes:
        chapters.append(Chapter(s, SKIP_TITLE))
        if e < duration - 0.5:
            chapters.append(Chapter(e, covering(e).title or "Chapter"))
    chapters.sort(key=lambda c: c.start)
    if chapters[0].start > 0:
        chapters.insert(0, Chapter(0.0, base[0].title if base[0].start == 0 else "Start"))
    return chapters


def merge_overlaps(scenes: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for s, e in sorted(scenes):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def chapters_xml(chapters: list[Chapter], language: str = "eng") -> str:
    atoms = []
    for c in chapters:
        atoms.append(
            "    <ChapterAtom>\n"
            f"      <ChapterTimeStart>{fmt_time(c.start)}</ChapterTimeStart>\n"
            "      <ChapterDisplay>\n"
            f"        <ChapterString>{escape(c.title)}</ChapterString>\n"
            f"        <ChapterLanguage>{language}</ChapterLanguage>\n"
            "      </ChapterDisplay>\n"
            "    </ChapterAtom>")
    return ('<?xml version="1.0"?>\n<!DOCTYPE Chapters SYSTEM "matroskachapters.dtd">\n<Chapters>\n'
            "  <EditionEntry>\n" + "\n".join(atoms) + "\n  </EditionEntry>\n</Chapters>\n")


def restore_chapters(media: Path, note: str) -> None:
    """Put the film's own chapters back (no "Skip" chapters) but keep the scene list,
    marked with `note` (e.g. that an edition now has them cut out)."""
    doc = load(media)
    if "original_chapters" not in doc:
        return
    original = [Chapter(float(s), t) for s, t in doc["original_chapters"]]
    with tempfile.TemporaryDirectory(prefix="audio-censor-") as tmp:
        if original:
            xml = Path(tmp) / "chapters.xml"
            xml.write_text(chapters_xml(original), encoding="utf-8")
            run(["mkvpropedit", "-q", str(media), "--chapters", str(xml)])
        else:
            run(["mkvpropedit", "-q", str(media), "--chapters", ""])
    doc["skip_chapters"] = False
    doc["edition"] = note
    save(media, doc)


def apply(media: Path, scenes: list[tuple[float, float, str]], duration: float) -> list[Chapter]:
    """Record `scenes` for `media` and rewrite its chapters to match. Returns the chapters."""
    if media.suffix.lower() != ".mkv":
        raise MediaError("skip scenes need an .mkv file (its chapters are rewritten in place)")
    if shutil.which("mkvpropedit") is None:
        raise MediaError("mkvpropedit not found; install mkvtoolnix (e.g. pacman -S mkvtoolnix-cli)")
    doc = load(media)
    if "original_chapters" not in doc:
        doc["original_chapters"] = [[c.start, c.title] for c in read_chapters(media)]
    original = [Chapter(float(s), t) for s, t in doc["original_chapters"]]
    doc["scenes"] = [[round(s, 3), round(e, 3), note] for s, e, note in sorted(scenes)]
    if scenes:
        chapters = merge_scenes(original, [(s, e) for s, e, _ in scenes], duration)
    else:
        chapters = original
    with tempfile.TemporaryDirectory(prefix="audio-censor-") as tmp:
        if chapters:
            xml = Path(tmp) / "chapters.xml"
            xml.write_text(chapters_xml(chapters), encoding="utf-8")
            run(["mkvpropedit", "-q", str(media), "--chapters", str(xml)])
        else:
            run(["mkvpropedit", "-q", str(media), "--chapters", ""])
    if scenes:
        save(media, doc)
    else:
        sidecar(media).unlink(missing_ok=True)
    return chapters
