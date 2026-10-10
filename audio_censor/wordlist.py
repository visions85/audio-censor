"""Profanity patterns and the word matcher."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from importlib import resources

TIERS = ("strong", "moderate", "mild")
ASTERISK_PATTERN = "<asterisked>"

# Characters replaced 1:1 so that token offsets still line up with the source text.
_CHAR_MAP = str.maketrans({
    "’": "'", "‘": "'", "`": "'", "´": "'",
    "-": " ", "‐": " ", "‑": " ", "–": " ", "—": " ", "_": " ", "/": " ",
})
_TOKEN_RE = re.compile(r"[a-z0-9*']+")


@dataclass(frozen=True)
class Token:
    text: str      # normalized token (lowercase, apostrophes trimmed)
    start: int     # char offset into the normalized text
    end: int


@dataclass(frozen=True)
class Pattern:
    raw: str
    tier: str
    parts: tuple   # ("exact", str) | ("prefix", str) | ("regex", re.Pattern)

    def __len__(self) -> int:
        return len(self.parts)


@dataclass(frozen=True)
class Match:
    start: int     # token index (inclusive)
    end: int       # token index (exclusive)
    pattern: Pattern
    text: str      # the matched words joined by spaces


def normalize_text(text: str) -> str:
    """Lowercase and map punctuation 1:1, preserving length."""
    return text.lower().translate(_CHAR_MAP)


def tokenize(text: str) -> list[Token]:
    """Split text into normalized tokens with character offsets."""
    norm = normalize_text(text)
    tokens = []
    for m in _TOKEN_RE.finditer(norm):
        raw = m.group(0)
        stripped = raw.strip("'")
        if not stripped:
            continue
        lead = len(raw) - len(raw.lstrip("'"))
        start = m.start() + lead
        tokens.append(Token(stripped, start, start + len(stripped)))
    return tokens


def compile_pattern(raw: str, tier: str) -> Pattern:
    raw = raw.strip().lower()
    if not raw:
        raise ValueError("empty pattern")
    if raw.startswith("re:"):
        return Pattern(raw, tier, (("regex", re.compile(raw[3:])),))
    parts = []
    for word in normalize_text(raw).split():
        if word.endswith("*") and len(word) > 1:
            parts.append(("prefix", word[:-1].strip("'")))
        else:
            parts.append(("exact", word.strip("'")))
    return Pattern(raw, tier, tuple(parts))


def _part_matches(part, token: str) -> bool:
    kind, value = part
    if kind == "exact":
        return token == value or (token.endswith("'s") and token[:-2] == value)
    if kind == "prefix":
        return token.startswith(value)
    return value.fullmatch(token) is not None


def load_default_tiers() -> dict[str, list[str]]:
    data = resources.files("audio_censor").joinpath("data/default_words.toml").read_text("utf-8")
    return tomllib.loads(data)["tiers"]


def load_default_allow() -> list[str]:
    data = resources.files("audio_censor").joinpath("data/default_words.toml").read_text("utf-8")
    return list(tomllib.loads(data).get("allow", []))


def active_tiers(level: str) -> tuple[str, ...]:
    """Tiers censored at a given level: 'mild' censors everything, 'strong' only the worst."""
    if level not in TIERS:
        raise ValueError(f"unknown level {level!r}; expected one of {', '.join(TIERS)}")
    return TIERS[: TIERS.index(level) + 1]


def is_asterisked(token: str) -> bool:
    """Whisper sometimes self-censors ('f***ing'); treat that as a hit."""
    return "*" in token and len(token) >= 3 and re.search(r"[a-z]", token) is not None


class Matcher:
    def __init__(self, patterns: list[Pattern], allow: list[Pattern] | None = None,
                 asterisks_as_hit: bool = True):
        self.patterns = patterns
        self.allow = allow or []
        self.asterisks_as_hit = asterisks_as_hit

    @classmethod
    def from_config(cls, cfg: dict) -> "Matcher":
        wcfg = cfg.get("words", {})
        tiers = load_default_tiers()
        for tier, extra in wcfg.get("tiers", {}).items():
            if tier not in TIERS:
                raise ValueError(f"unknown tier {tier!r} in words.tiers")
            tiers[tier] = list(tiers.get(tier, [])) + list(extra)
        level = cfg.get("level", "moderate")
        patterns = []
        for tier in active_tiers(level):
            patterns += [compile_pattern(p, tier) for p in tiers.get(tier, [])]
        patterns += [compile_pattern(p, "strong") for p in wcfg.get("extra", [])]
        allow = [compile_pattern(p, "allow") for p in load_default_allow() + list(wcfg.get("allow", []))]
        # Longest phrases first so "son of a bitch" wins over "bitch" at the same position.
        patterns.sort(key=len, reverse=True)
        return cls(patterns, allow, cfg.get("scan", {}).get("treat_asterisks_as_hit", True))

    def _try(self, pattern: Pattern, words: list[str], i: int) -> bool:
        if i + len(pattern) > len(words):
            return False
        return all(_part_matches(part, words[i + k]) for k, part in enumerate(pattern.parts))

    def _allowed(self, words: list[str], i: int, j: int) -> bool:
        """True when an allow pattern covers words[i:j]; a longer phrase ("cum laude") exempts the hit inside it."""
        for a in self.allow:
            for start in range(max(0, j - len(a)), i + 1):
                if self._try(a, words, start):
                    return True
        return False

    def find(self, tokens: list[Token]) -> list[Match]:
        words = [t.text for t in tokens]
        matches: list[Match] = []
        for i in range(len(words)):
            for pattern in self.patterns:
                if self._try(pattern, words, i):
                    j = i + len(pattern)
                    if not self._allowed(words, i, j):
                        matches.append(Match(i, j, pattern, " ".join(words[i:j])))
            if self.asterisks_as_hit and is_asterisked(words[i]) and not self._allowed(words, i, i + 1):
                matches.append(Match(i, i + 1, Pattern(ASTERISK_PATTERN, "strong", ()), words[i]))
        return matches

    def find_in_text(self, text: str) -> list[Match]:
        return self.find(tokenize(text))
