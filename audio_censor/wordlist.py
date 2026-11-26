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
        if word == "*":
            parts.append(("any", None))       # any one word: "bust* * balls"
        elif word.endswith("*") and len(word) > 1:
            parts.append(("prefix", word[:-1].strip("'")))
        else:
            parts.append(("exact", word.strip("'")))
    return Pattern(raw, tier, tuple(parts))


def _part_matches(part, token: str) -> bool:
    kind, value = part
    if kind == "any":
        return True
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


@dataclass
class ContextRule:
    """Words censored only when, in context, they mean `meaning` (see context.py)."""
    patterns: list[Pattern]
    meaning: str
    fallback: list[Pattern]
    innocent: str = ""
    ask_twice: bool = False
    unsure_mute: bool = False   # without the AI: mute every time (True) or only on a fallback phrase


# Everyday words never muted on their own inside a listed phrase ("son of a bitch").
_PHRASE_FILLER = frozenset("""a an the of you your yours my his her him me it its off on in up out to at for and
or oh so get gets got getting make makes made making give gives gave giving kiss kisses go goes
all some such what this that one""".split())


class Matcher:
    def __init__(self, patterns: list[Pattern], allow: list[Pattern] | None = None,
                 asterisks_as_hit: bool = True, rules: list[ContextRule] | None = None, judge=None,
                 vocab: list[Pattern] | None = None):
        self.patterns = patterns
        self.vocab = vocab or []      # single listed words of every tier, to find a phrase's rude word
        self.allow = allow or []
        self.asterisks_as_hit = asterisks_as_hit
        self.rules = rules or []
        self.judge = judge

    def prepare(self, texts: list[str]) -> None:
        """Hand the film's subtitle lines to the context judge (neighbouring lines help it)."""
        if self.judge is not None:
            self.judge.prepare(texts)

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
        ccfg = cfg.get("context", {})
        rules = [ContextRule(patterns=[compile_pattern(w, r.get("tier", "moderate")) for w in r.get("words", [])],
                             meaning=str(r.get("meaning", "")),
                             fallback=[compile_pattern(f, "allow") for f in r.get("fallback", [])],
                             unsure_mute=str(r.get("when_unsure", "keep")).lower() == "mute",
                             innocent=str(r.get("innocent", "")),
                             ask_twice=bool(r.get("ask_twice", False)))
                 for r in ccfg.get("rules", []) if r.get("words") and r.get("meaning")]
        # A word a rule names is judged in context instead of always censored ("bitch*"
        # in a rule replaces "bitch*" in the list; "son of a bitch" stays as it is).
        judged = {p.raw for r in rules for p in r.patterns}
        patterns = [p for p in patterns if p.raw not in judged]
        patterns.sort(key=len, reverse=True)
        judge = None
        if rules:
            from .context import ContextJudge
            judge = ContextJudge(cfg)
        vocab = [compile_pattern(w, "vocab") for ws in tiers.values() for w in ws if len(w.split()) == 1]
        vocab += [compile_pattern(w, "vocab") for w in wcfg.get("extra", []) if len(w.split()) == 1]
        vocab += [p for r in rules for p in r.patterns if len(p) == 1]
        return cls(patterns, allow, cfg.get("scan", {}).get("treat_asterisks_as_hit", True), rules, judge,
                   vocab)

    def core(self, words: list[str], i: int, j: int) -> tuple[int, int]:
        """The part of a listed phrase to mute: its rude words ("son of a [bitch]",
        "god [damn]"), or failing that, its words other than everyday fillers."""
        if j - i < 2:
            return i, j
        bad = [k for k in range(i, j) if any(self._try(v, words, k) for v in self.vocab)]
        if not bad:
            bad = [k for k in range(i, j) if words[k] not in _PHRASE_FILLER]
        return (bad[0], bad[-1] + 1) if bad else (i, j)

    @staticmethod
    def _try(pattern: Pattern, words: list[str], i: int) -> bool:
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
                        a, b = self.core(words, i, j)
                        matches.append(Match(a, b, pattern, " ".join(words[i:j])))
            if self.judge is not None and not any(m.start == i for m in matches):
                for rule in self.rules:
                    for pattern in rule.patterns:
                        j = i + len(pattern)
                        if self._try(pattern, words, i) and not self._allowed(words, i, j) \
                                and self.judge.accept(words, i, j, rule):
                            matches.append(Match(i, j, pattern, " ".join(words[i:j])))
                            break
            if self.asterisks_as_hit and is_asterisked(words[i]) and not self._allowed(words, i, i + 1):
                matches.append(Match(i, i + 1, Pattern(ASTERISK_PATTERN, "strong", ()), words[i]))
        return matches

    def find_in_text(self, text: str) -> list[Match]:
        return self.find(tokenize(text))
