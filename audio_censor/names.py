"""Tell a character called Dick from an insult.

Subtitles (and Whisper) capitalize proper nouns. A flagged word written with a capital
in the middle of a sentence is almost always a name; written in lowercase it is almost
always the swear. Ambiguous spots (start of a sentence, all-caps lines) fall back to a
film-level verdict: a word seen capitalized mid-sentence often enough, or used as an
SDH speaker label ("DICK:"), is a name for that film.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .wordlist import Match, Matcher, Token, tokenize

# "you're a dick", "what a dick", "such a Dick" -> insult even when capitalized
DETERMINERS = {
    "a", "an", "the", "such", "total", "complete", "utter", "real", "little", "big", "huge",
    "giant", "stupid", "fucking", "goddamn", "damn", "some", "that", "this", "what",
    "his", "her", "my", "your", "their", "our", "its",
}
_SENTENCE_BREAK = re.compile(r"[.!?…:;]|--|—|^\s*-")
_SPEAKER_LABEL = re.compile(r"(?:^|\n)\s*-?\s*([A-Z][A-Z'\-]{1,24}):", re.M)


@dataclass
class Occurrence:
    cls: str          # cap-mid | cap-initial | lower | allcaps
    word: str
    after_determiner: bool


@dataclass
class NameStats:
    cap_mid: int = 0
    lower: int = 0
    other: int = 0
    speaker_label: int = 0

    @property
    def total(self) -> int:
        return self.cap_mid + self.lower + self.other


@dataclass
class NameDetector:
    enabled: bool = True
    min_occurrences: int = 2
    ignore: set[str] = field(default_factory=set)
    stats: dict[str, NameStats] = field(default_factory=dict)
    names: set[str] = field(default_factory=set)

    @classmethod
    def from_config(cls, cfg: dict) -> "NameDetector":
        n = cfg.get("names", {})
        return cls(enabled=bool(n.get("detect", True)), min_occurrences=int(n.get("min_occurrences", 2)),
                   ignore={w.lower() for w in n.get("ignore", [])})

    # ------------------------------------------------------------------ classification

    @staticmethod
    def classify(text: str, tokens: list[Token], m: Match) -> Occurrence:
        """Look at how the (single-token) match is written in the original text."""
        tok = tokens[m.start]
        orig = text[tok.start:tok.end]
        word = tok.text.removesuffix("'s")
        prev = tokens[m.start - 1] if m.start > 0 else None
        gap = text[prev.end:tok.start] if prev else text[:tok.start]
        after_det = bool(prev) and prev.text in DETERMINERS and "," not in gap

        line_alpha = [c for c in text if c.isalpha()]
        line_is_caps = len(line_alpha) >= 6 and sum(c.isupper() for c in line_alpha) / len(line_alpha) > 0.9
        if orig.isupper() and (len(orig) > 1 or line_is_caps):
            cls = "allcaps" if line_is_caps else "cap-mid"
            if cls == "cap-mid" and (prev is None or _SENTENCE_BREAK.search(gap)):
                cls = "cap-initial"
        elif orig[0].isupper():
            cls = "cap-initial" if prev is None or _SENTENCE_BREAK.search(gap) else "cap-mid"
        else:
            cls = "lower"
        return Occurrence(cls, word, after_det)

    # ------------------------------------------------------------------ learning

    def learn(self, texts: list[str], matcher: Matcher) -> None:
        """Gather film-level evidence from every subtitle line."""
        self.stats = {}
        self.names = set()
        if not self.enabled:
            return
        labels = Counter()
        for text in texts:
            for lab in _SPEAKER_LABEL.findall(text):
                labels[lab.lower()] += 1
            tokens = tokenize(text)
            for m in matcher.find(tokens):
                if m.end - m.start != 1:
                    continue
                occ = self.classify(text, tokens, m)
                st = self.stats.setdefault(occ.word, NameStats())
                if occ.cls == "cap-mid" and not occ.after_determiner:
                    st.cap_mid += 1
                elif occ.cls == "lower":
                    st.lower += 1
                else:
                    st.other += 1
        for word, st in self.stats.items():
            if word in self.ignore:
                continue
            st.speaker_label = labels.get(word, 0)
            if st.speaker_label or st.cap_mid >= self.min_occurrences:
                self.names.add(word)

    # ------------------------------------------------------------------ per-occurrence verdict

    def is_name_use(self, text: str, tokens: list[Token], m: Match) -> bool:
        if not self.enabled or m.end - m.start != 1:
            return False
        occ = self.classify(text, tokens, m)
        if occ.word in self.ignore or occ.after_determiner:
            return False
        if occ.cls == "cap-mid":
            return True
        if occ.cls == "lower":
            return False
        return occ.word in self.names          # sentence-initial or all-caps: film-level verdict

    def summary(self) -> str:
        if not self.names:
            return ""
        parts = []
        for w in sorted(self.names):
            st = self.stats[w]
            why = f"{st.cap_mid} capitalized" + (f", {st.speaker_label} speaker label(s)" if st.speaker_label else "")
            parts.append(f"{w.capitalize()} ({why})")
        return ", ".join(parts)
