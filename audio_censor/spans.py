"""Hits (single detections) and Spans (merged, renderable time ranges)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from .wordlist import ASTERISK_PATTERN, TIERS

SIDECAR_VERSION = 1


@dataclass
class Hit:
    start: float
    end: float
    word: str
    pattern: str
    tier: str
    source: str              # "subtitle" | "asr"
    confidence: float = 1.0
    cue_start: float | None = None
    cue_end: float | None = None
    name_use: bool = False   # judged to be a character's name, not a swear


@dataclass
class Span:
    start: float
    end: float
    words: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    tier: str = "mild"
    confidence: float = 1.0
    estimated: bool = False   # timing came from subtitle position only

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = round(self.start, 3)
        d["end"] = round(self.end, 3)
        d["confidence"] = round(self.confidence, 3)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Span":
        return cls(
            start=float(d["start"]), end=float(d["end"]),
            words=list(d.get("words", [])), sources=list(d.get("sources", [])),
            tier=d.get("tier", "mild"), confidence=float(d.get("confidence", 1.0)),
            estimated=bool(d.get("estimated", False)),
        )


def _worst_tier(a: str, b: str) -> str:
    return a if TIERS.index(a) <= TIERS.index(b) else b


def _same_pattern(a: Hit, b: Hit) -> bool:
    return a.pattern == b.pattern or ASTERISK_PATTERN in (a.pattern, b.pattern)


def build_spans(sub_hits: list[Hit], asr_hits: list[Hit], cfg: dict, duration: float | None = None) -> list[Span]:
    """Combine subtitle and ASR hits into merged spans.

    A subtitle hit is "confirmed" when a matching ASR word falls inside its cue window;
    the tighter ASR timing wins and the subtitle hit is dropped. Unconfirmed subtitle
    hits are kept with their estimated timing. ASR hits without subtitle support are
    kept when scan.asr_only is true.
    """
    scan = cfg.get("scan", {})
    sub_pad = float(scan.get("subtitle_pad", 0.35))
    asr_pad = float(scan.get("asr_pad", 0.10))
    merge_gap = float(scan.get("merge_gap", 0.25))
    keep_asr_only = bool(scan.get("asr_only", True))
    slack = 0.6  # subtitle timing is loose; allow ASR words slightly outside the cue
    drift = float(scan.get("subtitle_drift", 2.0))  # fallback window for badly synced subtitles

    confirmed_asr: set[int] = set()
    spans: list[Span] = []

    # Name exemptions: a subtitle occurrence judged to be a name also exempts the ASR
    # word for the same pattern inside that cue; ASR words judged names on their own
    # capitalization are dropped too.
    exempt_windows = [(sh.cue_start if sh.cue_start is not None else sh.start,
                       sh.cue_end if sh.cue_end is not None else sh.end, sh)
                      for sh in sub_hits if sh.name_use]
    sub_hits = [sh for sh in sub_hits if not sh.name_use]
    kept_asr = []
    for ah in asr_hits:
        if ah.name_use:
            continue
        if any(ah.start >= cs - slack and ah.end <= ce + slack and _same_pattern(sh, ah)
               for cs, ce, sh in exempt_windows):
            continue
        kept_asr.append(ah)
    asr_hits = kept_asr

    def claim(sh: Hit, cue_start: float, cue_end: float, window: float) -> int | None:
        """Index of an ASR hit for the same word within the cue window (+/- window)."""
        best = None
        for i, ah in enumerate(asr_hits):
            if i in confirmed_asr or not _same_pattern(sh, ah):
                continue
            if ah.start >= cue_start - window and ah.end <= cue_end + window:
                dist = max(0.0, cue_start - ah.start, ah.end - cue_end)
                if best is None or dist < best[0]:
                    best = (dist, i)
        return None if best is None else best[1]

    for sh in sub_hits:
        cue_start = sh.cue_start if sh.cue_start is not None else sh.start
        cue_end = sh.cue_end if sh.cue_end is not None else sh.end
        match_idx = claim(sh, cue_start, cue_end, slack)
        if match_idx is None and drift > slack:
            match_idx = claim(sh, cue_start, cue_end, drift)
        if match_idx is not None:
            confirmed_asr.add(match_idx)
            continue
        spans.append(Span(
            start=max(0.0, sh.start - sub_pad), end=sh.end + sub_pad,
            words=[sh.word], sources=["subtitle"], tier=sh.tier,
            confidence=sh.confidence, estimated=True,
        ))

    for i, ah in enumerate(asr_hits):
        confirmed = i in confirmed_asr
        if not confirmed and not keep_asr_only and sub_hits:
            continue
        spans.append(Span(
            start=max(0.0, ah.start - asr_pad), end=ah.end + asr_pad,
            words=[ah.word], sources=["asr", "subtitle"] if confirmed else ["asr"],
            tier=ah.tier, confidence=ah.confidence, estimated=False,
        ))

    if duration:
        for s in spans:
            s.end = min(s.end, duration)
    return merge_spans(spans, merge_gap)


def merge_spans(spans: list[Span], gap: float) -> list[Span]:
    spans = sorted((s for s in spans if s.end > s.start), key=lambda s: (s.start, s.end))
    merged: list[Span] = []
    for s in spans:
        if merged and s.start <= merged[-1].end + gap:
            m = merged[-1]
            m.end = max(m.end, s.end)
            for w in s.words:
                if w not in m.words:
                    m.words.append(w)
            for src in s.sources:
                if src not in m.sources:
                    m.sources.append(src)
            m.tier = _worst_tier(m.tier, s.tier)
            m.confidence = min(m.confidence, s.confidence)
            m.estimated = m.estimated and s.estimated
        else:
            merged.append(Span(s.start, s.end, list(s.words), list(s.sources), s.tier, s.confidence, s.estimated))
    return merged


def sidecar_path(media: Path) -> Path:
    return media.with_name(media.stem + ".censor.json")


def save_sidecar(path: Path, media: Path, spans: list[Span], meta: dict) -> Path:
    doc = {
        "version": SIDECAR_VERSION,
        "source_file": media.name,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **meta,
        "spans": [s.to_dict() for s in spans],
    }
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def load_sidecar(path: Path) -> tuple[list[Span], dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    spans = [Span.from_dict(d) for d in doc.get("spans", [])]
    meta = {k: v for k, v in doc.items() if k != "spans"}
    return spans, meta


def fmt_time(t: float) -> str:
    h, rem = divmod(max(0.0, t), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def format_table(spans: list[Span]) -> str:
    if not spans:
        return "(no spans)"
    lines = [f"{'#':>3}  {'start':>12}  {'end':>12}  {'len':>5}  {'tier':8}  {'src':8}  words"]
    for i, s in enumerate(spans, 1):
        src = "+".join(sorted(x[:3] for x in s.sources)) + ("~" if s.estimated else "")
        lines.append(f"{i:>3}  {fmt_time(s.start):>12}  {fmt_time(s.end):>12}  {s.duration:5.2f}  "
                     f"{s.tier:8}  {src:8}  {', '.join(s.words)}")
    return "\n".join(lines)
