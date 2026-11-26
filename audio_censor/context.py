"""Words that are only rude in one sense ("balls", "nuts"), judged in context by a local AI.

A rule in the config names the words and the meaning to censor:

    [[context.rules]]
    words = ["balls", "nuts"]
    meaning = "testicles, or crude slang built on them"
    innocent = "a literal ball or nut, or meaning crazy"
    fallback = ["bust* * balls", "in the balls", "balls off", "kick* * nuts", "your nuts"]

Each time one of the words turns up, the subtitle line and the lines around it (or the
surrounding transcript words) go to a model served by Ollama on this machine, which
answers whether the word means `meaning` there. Only a "yes" is censored. Answers are
cached (by model, meaning and context) in the user cache directory, so re-running a film
costs nothing. While a game is running, or when Ollama is not reachable, the fallback
phrases decide instead: a word is censored only when one of them covers it.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from .media import eprint

PROMPT_RUDE = (
    "You check film dialogue for a family audience. The word marked [[like this]] is on "
    "the line marked >>>. Is it used there for {meaning}? It is not when it means {innocent}. "
    "Answer with one word: yes or no."
)
# For words that are censored unless shown innocent (when_unsure = "mute"): the model is
# asked to recognise the innocent sense, so anything unclear stays censored.
PROMPT_INNOCENT = (
    "You check film dialogue for a family audience. The word marked [[like this]] is on "
    "the line marked >>>. Is it clearly used there in one of these innocent senses: "
    "{innocent}? Answer no if it is a swear, an insult, slang for a body part, or anything "
    "vulgar. Answer with one word: yes or no."
)


def find_game() -> str | None:
    """A running game (Steam, Lutris, umu, Wine/Proton), so the GPU is left alone."""
    me = str(os.getpid())
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or pid == me:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                argv = fh.read().split(b"\0")
        except OSError:
            continue
        if not argv or not argv[0]:
            continue
        names = [os.path.basename(a) for a in argv[:4]]
        for i, name in enumerate(names[:-1]):
            if name == b"reaper" and argv[i + 1] == b"SteamLaunch":
                return "a Steam game"
        if argv[0].startswith(b"lutris-wrapper") or b"umu-run" in names[:2] \
                or names[0] in (b"wineserver", b"wineserver64"):
            return "a game"
    return None


def corrections_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "audio-censor" / "corrections.json"


def load_corrections() -> list[dict]:
    """[{"phrase": "a private [[dick]]", "mute": false, "note": "..."}], oldest first."""
    try:
        data = json.loads(corrections_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [c for c in data if isinstance(c, dict) and "[[" in c.get("phrase", "")]


def save_corrections(items: list[dict]) -> None:
    path = corrections_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def parse_phrase(phrase: str) -> tuple[list[str], int, int]:
    """'a private [[dick]], ma'am' -> (normalized words, start, end of the marked words)."""
    from .wordlist import tokenize
    if phrase.count("[[") != 1 or phrase.count("]]") != 1 or phrase.index("[[") > phrase.index("]]"):
        raise ValueError("mark the word once, like: a private [[dick]]")
    before, rest = phrase.split("[[", 1)
    word, after = rest.split("]]", 1)
    b, w, a = ([t.text for t in tokenize(x)] for x in (before, word, after))
    if not w:
        raise ValueError("nothing between [[ and ]]")
    return b + w + a, len(b), len(b) + len(w)


def anchors(phrase: str) -> tuple[bool, bool]:
    """'^[[nuts]]$': the correction applies only where the line starts (^) / ends ($) with it."""
    t = phrase.strip()
    return t.startswith("^"), t.endswith("$")


def _cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "audio-censor" / "context.json"


class ContextDown(RuntimeError):
    """The AI cannot answer, and [context] when_down = "stop": end the run rather than
    let the simpler fallback phrases decide a film's mutes unseen."""


class ContextJudge:
    def __init__(self, cfg: dict):
        ccfg = cfg.get("context", {})
        self.url = str(ccfg.get("url", "http://127.0.0.1:11434")).rstrip("/")
        if not self.url.startswith("http"):
            self.url = "http://" + self.url
        self.model = ccfg.get("model", "gemma4:12b")
        self.enabled = bool(ccfg.get("enabled", True))
        self.timeout = float(ccfg.get("timeout", 120))
        # "stop": end the run when the AI cannot answer; "fallback": use the fallback phrases
        self.when_down = str(ccfg.get("when_down", "stop")).lower()
        self.retries = int(ccfg.get("retries", 3))
        self.retry_wait = float(ccfg.get("retry_wait", 20))
        self.lines: list[list[str]] = []      # prepared document: words of each subtitle line
        self.captions: list[list[bool]] = []  # per word: inside a [sound caption]
        self.in_caption = False
        self.cur = None
        self._index: dict[str, int] = {}
        self._cache: dict[str, bool] | None = None
        self._down: str | None = None          # why the model cannot be used this run
        self._used = False
        self._told: set[str] = set()
        self.history: list[dict] = []                            # this film's decisions, for its .censor.json
        self.corrections = []
        for c in load_corrections():
            try:
                self.corrections.append((*parse_phrase(c["phrase"]), bool(c.get("mute")), c["phrase"],
                                         *anchors(c["phrase"])))
            except ValueError:
                continue

    # ------------------------------------------------------------------ document

    def prepare(self, texts: list[str]) -> None:
        """Remember a film's subtitle lines, so a word in one of them is judged with its neighbours."""
        import re
        from .wordlist import tokenize
        self.lines, self.captions = [], []
        for x in texts:
            toks = tokenize(x)
            # [BANGING], (door slams): a caption of a sound, not something anyone says
            spans = [(m.start(), m.end()) for m in re.finditer(r"\[[^\]]*\]|\([^)]*\)|♪[^♪]*♪?", x)]
            self.lines.append([t.text for t in toks])
            self.captions.append([any(a <= t.start < b for a, b in spans) for t in toks])
        self.history = []
        self._told = set()
        self._index = {}
        for i, words in enumerate(self.lines):
            self._index.setdefault(" ".join(words), i)

    def _line_of(self, words: list[str]) -> int | None:
        key = " ".join(words)
        if key in self._index:
            return self._index[key]
        if key:
            for i, line in enumerate(self.lines):
                if key in " ".join(line):
                    return i
        return None

    def context(self, words: list[str], i: int, j: int) -> tuple[str, str]:
        """(text sent to the model, the line itself) for words[i:j]."""
        line = self._line_of(words)
        if line is not None:
            # `words` may be one piece of the line (subtitle tags split it): mark the word
            # in the whole line, so the question is the same however the line was reached.
            full = self.lines[line]
            off = next((k for k in range(len(full) - len(words) + 1) if full[k:k + len(words)] == words), 0)
            i, j, words = i + off, j + off, full
            flags = self.captions[line] if line < len(self.captions) else []
            self.in_caption = bool(flags[i:j]) and all(flags[i:j])
            self.cur = (full, i, j)          # the whole subtitle line, for ^ / $ corrections
        else:
            self.in_caption = False
            self.cur = None
        marked = words[:i] + ["[[" + " ".join(words[i:j]) + "]]"] + words[j:]
        if line is not None:
            before = [" ".join(w) for w in self.lines[max(0, line - 3):line]]
            after = [" ".join(w) for w in self.lines[line + 1:line + 4]]
            shown = " ".join(marked)
            return "\n".join(before + [">>> " + shown] + after), shown
        # a transcript: the words around it
        # (in `marked` the word is at index i; 30 words either side)
        text = ">>> " + " ".join(marked[max(0, i - 30):i + 31])
        return text, " ".join(marked[max(0, i - 8):i + 9])

    # ------------------------------------------------------------------ deciding

    def accept(self, words: list[str], i: int, j: int, rule) -> bool:
        text, shown = self.context(words, i, j)
        if self.in_caption:
            return self._note(words, i, j, shown, False, "sound caption")
        fixed = self._corrected(words, i, j)
        if fixed is not None:
            return self._note(words, i, j, shown, fixed, "your correction")
        if not self.enabled:
            return self._note(words, i, j, shown, self._fallback(words, i, j, rule), "fallback phrases")
        # With ask_twice, asked with the neighbouring lines and on the line alone: a small
        # model's answer shifts with the context it is shown, so either one judging it rude
        # mutes it. (Not for everyday words: alone, "just come on" can look rude.)
        yes, how = False, "cached"
        questions = (text, ">>> " + shown) if rule.ask_twice else (text,)
        for question in dict.fromkeys(questions):
            answer, source = self._judge(question, rule)
            if answer is None:
                if self.when_down != "fallback":
                    raise ContextDown(f"{self._down or 'the AI did not answer'}; stopping so the fallback "
                                      "phrases do not decide this film (start Ollama and run again, or set "
                                      '[context] when_down = "fallback")')
                yes, how = self._fallback(words, i, j, rule), "fallback phrases"
                break
            if source != "cached":
                how = source
            if answer:
                yes = True
                break
        return self._note(words, i, j, shown, yes, how)

    def _note(self, words, i, j, shown, yes: bool, how: str) -> bool:
        tag = f"{shown}:{yes}"
        if tag not in self._told:
            self._told.add(tag)
            self.history.append({"word": " ".join(words[i:j]), "line": shown, "mute": yes, "how": how})
            # printed as it is decided, so a run can be watched live
            eprint(f"  context: {'MUTE' if yes else 'keep'} {' '.join(words[i:j])!r} in \"{shown[:90]}\" ({how})")
        return yes

    def _corrected(self, words: list[str], i: int, j: int) -> bool | None:
        """Your answer for this use of the word, when a correction's phrase surrounds it.
        When corrections disagree ("wants to [[come]] in" keep, "[[come]] in her" mute),
        mute wins: a family filter errs toward silence."""
        found = set()
        for cw, cs, ce, mute, _, start, end in self.corrections:
            w, a, b = (self.cur if (start or end) and getattr(self, "cur", None) else (words, i, j))
            if (start or end) and not getattr(self, "cur", None):
                continue                    # a line-anchored correction needs a subtitle line
            if ce - cs != b - a or w[a:b] != cw[cs:ce]:
                continue
            lo = a - cs
            if lo >= 0 and w[lo:lo + len(cw)] == cw and (not start or lo == 0) and (not end or lo + len(cw) == len(w)):
                found.add(mute)
        return (True in found) if found else None

    def _judge(self, text: str, rule) -> tuple[bool | None, str]:
        key = hashlib.sha1(f"{self.model}\0{self.prompt(rule)}\0{text}".encode()).hexdigest()
        cache = self._load()
        if key in cache:
            return cache[key], "cached"
        answer = self._ask(text, rule)
        if answer is not None:
            cache[key] = answer
            self._save()
        return answer, self.model

    def _fallback(self, words: list[str], i: int, j: int, rule) -> bool:
        from .wordlist import Matcher
        if rule.unsure_mute:
            return True
        for p in rule.fallback:
            for start in range(max(0, i - len(p) + 1), i + 1):
                if start + len(p) >= j and Matcher._try(p, words, start):
                    return True
        return False

    def prompt(self, rule) -> str:
        innocent = rule.innocent or "any other, ordinary meaning"
        if rule.unsure_mute:
            text = PROMPT_INNOCENT.format(innocent=innocent)
        else:
            text = PROMPT_RUDE.format(meaning=rule.meaning, innocent=innocent)
        return text + self._examples(rule)

    def _examples(self, rule, limit: int = 12) -> str:
        """Your corrections for this rule's words, as worked examples (newest first)."""
        from .wordlist import Matcher
        shots = []
        for cw, cs, ce, mute, phrase, *_ in reversed(self.corrections):
            if not any(len(p) == ce - cs and Matcher._try(p, cw, cs) for p in rule.patterns):
                continue
            answer = ("no" if mute else "yes") if rule.unsure_mute else ("yes" if mute else "no")
            marked = " ".join(cw[:cs] + ["[[" + " ".join(cw[cs:ce]) + "]]"] + cw[ce:])
            shots.append(f">>> {marked}\nAnswer: {answer}")
            if len(shots) == limit:
                break
        if not shots:
            return ""
        return ("\n\nThe family has already decided these; answer alike in similar cases:\n\n"
                + "\n\n".join(shots))

    def _ask(self, text: str, rule) -> bool | None:
        if self._down:
            return None
        game = find_game()
        if game:
            self._down = f"{game} is running"
            eprint(f"  context: {self._down}"
                   + (f"; using the fallback phrases instead of {self.model}" if self.when_down == "fallback" else ""))
            return None
        body = {"model": self.model, "stream": False, "think": False, "keep_alive": "2m",
                "messages": [{"role": "system", "content": self.prompt(rule)},
                             {"role": "user", "content": text}],
                "options": {"temperature": 0, "num_predict": 3, "num_ctx": 4096}}
        req = urllib.request.Request(self.url + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        import time
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    reply = json.load(r)["message"]["content"].strip().lower()
                break
            except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
                if attempt < self.retries:             # a restart of Ollama takes a few seconds
                    eprint(f"  context: Ollama at {self.url} did not answer ({exc}); "
                           f"trying again in {self.retry_wait:.0f} s")
                    time.sleep(self.retry_wait)
                    continue
                self._down = f"Ollama at {self.url} did not answer ({exc})"
                eprint(f"  context: {self._down}"
                       + ("; using the fallback phrases" if self.when_down == "fallback" else ""))
                return None
        self._used = True
        yes = reply.lstrip("*( ").startswith("yes")
        return not yes if rule.unsure_mute else yes

    def release(self) -> None:
        """Unload the model, so it does not sit in VRAM beside Whisper or the encode."""
        if not self._used:
            return
        body = {"model": self.model, "keep_alive": 0}
        req = urllib.request.Request(self.url + "/api/generate", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=30).read()
        except (urllib.error.URLError, OSError):
            pass
        self._used = False

    def report(self) -> None:
        """Decisions are printed as they are made (see _note)."""

    # ------------------------------------------------------------------ cache

    def _load(self) -> dict[str, bool]:
        if self._cache is None:
            try:
                self._cache = json.loads(_cache_path().read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._cache = {}
        return self._cache

    def _save(self) -> None:
        path = _cache_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._cache), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            pass
