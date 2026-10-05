"""Subtitle discovery, parsing and scanning."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pysubs2

from .media import MediaInfo, Stream, extract_subtitle, language_matches
from .spans import Hit
from .wordlist import Matcher, tokenize

TEXT_SUB_CODECS = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text", "tx3g"}
EXTERNAL_EXTS = (".srt", ".ass", ".ssa", ".vtt")
_BRACKETS_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")   # [GRUNTS], (laughs) are not speech
_SPEAKER_RE = re.compile(r"^\s*-?\s*[A-Z][A-Z .'-]{1,24}:\s*", re.M)   # "JOHN: ..."


@dataclass
class Cue:
    start: float
    end: float
    text: str


def find_external(media: Path, languages: list[str]) -> Path | None:
    """Look for <stem>.srt, <stem>.en.srt, <stem>.eng.srt ... next to the media file."""
    stem = media.stem
    candidates = []
    for p in media.parent.iterdir():
        if p.suffix.lower() not in EXTERNAL_EXTS or not p.name.startswith(stem):
            continue
        rest = p.name[len(stem):-len(p.suffix)].strip(".").lower()
        tags = [t for t in rest.split(".") if t]
        if any(t in ("forced", "sdh") for t in tags) and tags != ["sdh"]:
            score = 0  # forced subs only carry foreign lines; skip unless nothing else
        elif not tags:
            score = 2
        elif any(language_matches(t, languages) for t in tags):
            score = 3
        else:
            score = 1
        candidates.append((score, p))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (-c[0], c[1].name))
    return candidates[0][1] if candidates[0][0] > 0 else None


def pick_embedded(info: MediaInfo, languages: list[str]) -> Stream | None:
    subs = [s for s in info.of_type("subtitle") if s.codec_name in TEXT_SUB_CODECS]
    if not subs:
        return None
    is_forced = lambda s: "forced" in (s.title or "").lower()
    for s in subs:
        if language_matches(s.language, languages) and not is_forced(s):
            return s
    for s in subs:
        if not is_forced(s):
            return s
    return None


def load_cues(path: Path) -> list[Cue]:
    subs = pysubs2.load(str(path), encoding="utf-8", errors="replace")
    cues = []
    for ev in subs.events:
        if ev.is_comment:
            continue
        text = ev.plaintext.replace("\n", " ").strip()
        if not text:
            continue
        cues.append(Cue(ev.start / 1000.0, ev.end / 1000.0, text))
    cues.sort(key=lambda c: c.start)
    return cues


def obtain_cues(media: Path, info: MediaInfo, languages: list[str], workdir: Path,
                explicit: Path | None = None) -> tuple[list[Cue], str]:
    """Return (cues, description-of-source) or ([], reason)."""
    if explicit is not None:
        return load_cues(explicit), f"file {explicit.name}"
    ext = find_external(media, languages)
    if ext is not None:
        return load_cues(ext), f"file {ext.name}"
    stream = pick_embedded(info, languages)
    if stream is not None:
        out = workdir / f"{media.stem}.s{stream.type_index}.srt"
        extract_subtitle(media, stream.type_index, out)
        return load_cues(out), f"embedded track {stream.describe()}"
    return [], "no subtitles found"


def _blank_non_speech(text: str) -> str:
    """Replace sound cues and speaker labels with spaces, keeping offsets aligned."""
    def blank(m):
        return " " * (m.end() - m.start())
    text = _BRACKETS_RE.sub(blank, text)
    return _SPEAKER_RE.sub(blank, text)


def scan_cues(cues: list[Cue], matcher: Matcher, mode: str = "estimate") -> list[Hit]:
    hits: list[Hit] = []
    for cue in cues:
        text = _blank_non_speech(cue.text)
        tokens = tokenize(text)
        if not tokens:
            continue
        length = max(1, len(text))
        dur = max(0.0, cue.end - cue.start)
        for m in matcher.find(tokens):
            if mode == "cue":
                start, end = cue.start, cue.end
            else:
                c0 = tokens[m.start].start
                c1 = tokens[m.end - 1].end
                start = cue.start + dur * (c0 / length)
                end = cue.start + dur * (c1 / length)
            hits.append(Hit(start=start, end=end, word=m.text, pattern=m.pattern.raw,
                            tier=m.pattern.tier, source="subtitle", confidence=0.6,
                            cue_start=cue.start, cue_end=cue.end))
    return hits
