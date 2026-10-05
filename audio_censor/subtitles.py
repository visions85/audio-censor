"""Subtitle discovery, parsing and scanning."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path

import pysubs2

from .media import MediaInfo, Stream, extract_subtitle, language_matches
from .names import NameDetector
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


@dataclass
class SubtitleSource:
    path: Path            # file on disk (external file, or an extracted embedded track)
    language: str         # "eng", "en", ... or ""
    description: str      # for log lines
    embedded: bool


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16"):
        try:
            text = raw.decode(enc)
            if enc == "utf-16" and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
                raise UnicodeDecodeError(enc, raw, 0, 1, "no BOM")
            return text
        except UnicodeDecodeError:
            continue
    return raw.decode("cp1252", errors="replace")


def load_subs(path: Path) -> pysubs2.SSAFile:
    """Parse a subtitle file, tolerating the usual encoding mess."""
    return pysubs2.SSAFile.from_string(_read_text(path))


def cues_from_subs(subs: pysubs2.SSAFile) -> list[Cue]:
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


def load_cues(path: Path) -> list[Cue]:
    return cues_from_subs(load_subs(path))


def _language_tag(media: Path, sub: Path) -> str:
    rest = sub.name[len(media.stem):-len(sub.suffix)].strip(".")
    tags = [t for t in rest.split(".") if t and t.lower() not in ("forced", "sdh")]
    return tags[-1] if tags else ""


def find_subtitle_source(media: Path, info: MediaInfo, languages: list[str], workdir: Path,
                         explicit: Path | None = None) -> SubtitleSource | None:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"subtitle file not found: {explicit}")
        return SubtitleSource(explicit, _language_tag(media, explicit), f"file {explicit.name}", False)
    ext = find_external(media, languages)
    if ext is not None:
        return SubtitleSource(ext, _language_tag(media, ext), f"file {ext.name}", False)
    stream = pick_embedded(info, languages)
    if stream is not None:
        out = workdir / f"{media.stem}.s{stream.type_index}.srt"
        extract_subtitle(media, stream.type_index, out)
        return SubtitleSource(out, stream.language, f"embedded track {stream.describe()}", True)
    return None


def obtain_cues(media: Path, info: MediaInfo, languages: list[str], workdir: Path,
                explicit: Path | None = None) -> tuple[list[Cue], str]:
    """Return (cues, description-of-source) or ([], reason)."""
    src = find_subtitle_source(media, info, languages, workdir, explicit)
    if src is None:
        return [], "no subtitles found"
    return load_cues(src.path), src.description


# ----------------------------------------------------------------------------- clean subtitles

STYLES = ("asterisks", "first-letter", "bleep", "remove")
_PIECE_RE = re.compile(r"(\{[^}]*\}|\\[Nnh])")   # ASS override tags and line breaks


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for s, e in sorted(set(ranges)):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _censor_plain(text: str, matcher: Matcher, style: str, replacement: str,
                  detector: "NameDetector | None" = None) -> tuple[str, int]:
    tokens = tokenize(text)
    matches = matcher.find(tokens)
    if detector:
        matches = [m for m in matches if not detector.is_name_use(text, tokens, m)]
    if not matches:
        return text, 0
    if style in ("bleep", "remove"):
        ranges = [(tokens[m.start].start, tokens[m.end - 1].end) for m in matches]
    else:
        ranges = [(tokens[k].start, tokens[k].end) for m in matches for k in range(m.start, m.end)]
    ranges = _merge_ranges(ranges)
    out, pos = [], 0
    for s, e in ranges:
        if style == "remove":
            # also eat one adjacent space so "a damn shame" -> "a shame"
            if e < len(text) and text[e] == " ":
                e += 1
            elif s > 0 and text[s - 1] == " " and pos < s:
                s -= 1
            out.append(text[pos:s])
        else:
            out.append(text[pos:s])
            if style == "asterisks":
                out.append("*" * (e - s))
            elif style == "first-letter":
                out.append(text[s] + "*" * (e - s - 1))
            else:
                out.append(replacement)
        pos = e
    out.append(text[pos:])
    result = "".join(out)
    if style == "remove":
        result = re.sub(r" {2,}", " ", result).strip()
    return result, len(ranges)


def censor_text(text: str, matcher: Matcher, style: str = "asterisks", replacement: str = "[BLEEP]",
                detector: "NameDetector | None" = None) -> tuple[str, int]:
    """Censor the visible words of an event's text, leaving tags and line breaks alone."""
    if style not in STYLES:
        raise ValueError(f"subtitle style must be one of {', '.join(STYLES)}")
    out, count = [], 0
    for piece in _PIECE_RE.split(text):
        if not piece or _PIECE_RE.fullmatch(piece):
            out.append(piece)
            continue
        new, n = _censor_plain(piece, matcher, style, replacement, detector)
        out.append(new)
        count += n
    return "".join(out), count


def censor_subs(subs: pysubs2.SSAFile, matcher: Matcher, style: str, replacement: str,
                detector: "NameDetector | None" = None) -> tuple[pysubs2.SSAFile, int]:
    clean = copy.deepcopy(subs)
    total = 0
    for ev in clean.events:
        if ev.is_comment:
            continue
        ev.text, n = censor_text(ev.text, matcher, style, replacement, detector)
        total += n
    return clean, total


def write_clean_subtitles(source: SubtitleSource, matcher: Matcher, out_path: Path,
                          style: str, replacement: str, detector: "NameDetector | None" = None) -> int:
    """Write a censored copy of `source` to out_path (format follows the extension)."""
    subs = load_subs(source.path)
    if detector is not None:
        detector.learn([c.text for c in cues_from_subs(subs)], matcher)
    clean, count = censor_subs(subs, matcher, style, replacement, detector)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.save(str(out_path), encoding="utf-8")
    return count


def _blank_non_speech(text: str) -> str:
    """Replace sound cues and speaker labels with spaces, keeping offsets aligned."""
    def blank(m):
        return " " * (m.end() - m.start())
    text = _BRACKETS_RE.sub(blank, text)
    return _SPEAKER_RE.sub(blank, text)


def scan_cues(cues: list[Cue], matcher: Matcher, mode: str = "estimate",
              detector: "NameDetector | None" = None) -> list[Hit]:
    """Find flagged words in cues. Hits judged to be names are returned with name_use=True
    (build_spans drops them and uses them to exempt the matching ASR words)."""
    hits: list[Hit] = []
    for cue in cues:
        text = _blank_non_speech(cue.text)
        tokens = tokenize(text)
        if not tokens:
            continue
        length = max(1, len(text))
        dur = max(0.0, cue.end - cue.start)
        for m in matcher.find(tokens):
            name_use = detector.is_name_use(text, tokens, m) if detector else False
            if mode == "cue":
                start, end = cue.start, cue.end
            else:
                c0 = tokens[m.start].start
                c1 = tokens[m.end - 1].end
                start = cue.start + dur * (c0 / length)
                end = cue.start + dur * (c1 / length)
            hits.append(Hit(start=start, end=end, word=m.text, pattern=m.pattern.raw,
                            tier=m.pattern.tier, source="subtitle", confidence=0.6,
                            cue_start=cue.start, cue_end=cue.end, name_use=name_use))
    return hits
