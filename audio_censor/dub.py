"""Dub mode: say a family word in the actor's voice instead of muting the rude one.

For each muted word with a replacement in the word table (see DEFAULT_WORDS, [dub.words]):

1. the centre channel around it is cut out (the dialogue channel of a 5.1 / 7.1 track);
2. the dub worker (dub_worker.py, run with the Python of a separate model environment:
   [dub] python) separates the voice and regenerates just that word inside the actor's own
   line, once per replacement word and seed;
3. Whisper listens to every take: it must clearly hear the new word and no trace of the old;
4. among the takes that pass, the replacement used least so far in the film wins, so a
   film does not say "dang" forty times; a word with no passing take is muted as before.

The chosen takes become one "change track" (front left, front right, centre) that the
render adds to the dialogue (see beep.dub_chain): only the dubbed words change. When the
front left/right also carry the voice (an upmix of a stereo film), the voice is separated
there too and the same new word is put in at each side's share of it, so the music on
those speakers keeps playing; takes are checked against the front mix, so a leak fails.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .media import MediaError, eprint
from .spans import Span

# word -> replacements (tried in turn; the first that sounds right is used most).
# A multi-word key replaces only its last word, and only after the other words:
# "son of a bitch" -> "son of a gun", while "bitch" alone stays muted.
DEFAULT_WORDS: dict[str, list[str]] = {
    "shit": ["sugar", "shoot", "crud"],
    "shitty": ["crummy", "cruddy"],
    "bullshit": ["baloney", "bull"],
    "damn": ["dang", "darn"],
    "dammit": ["dang it", "darn it"],
    "goddamn": ["doggone", "gosh darn"],
    "goddamnit": ["doggone it", "dang it"],
    "hell": ["heck"],
    "ass": ["bum", "butt", "rear"],
    "asshole": ["numbskull", "doofus", "jerk"],
    "assholes": ["numbskulls", "doofuses", "jerks"],
    "jackass": ["dingbat", "turkey", "jerk"],
    "dumbass": ["dummy", "dingbat"],
    "son of a bitch": ["gun", "biscuit"],
    "fuck": ["frick", "fudge"],
    "fucking": ["freaking", "frickin'", "flipping"],
    "fuckin": ["freakin'", "frickin'"],
    "fucked": ["frigged", "fried"],
    "motherfucker": ["lowly-trucker", "mother trucker"],
    "fart": ["toot"],
    "pissed": ["ticked", "mad", "sore"],
    "bastard": ["buzzard", "rascal"],
}

PAD = 6.0          # window cut around each word, seconds either side
BEFORE = 4.0       # context the model hears before the word ...
AFTER = 2.5        # ... and after it
LEAK = 0.25        # front L/R louder than this share of the centre on the word: duck them
ROOM = 0.35        # a longer swap may run this far into the pause after the word, seconds
GAP = 0.06         # ... leaving at least this much of the pause before the next word
SWALLOW = 0.03     # an edit this far into a neighbouring word takes the whole word
CLOSE = 0.05       # takes this close to the most natural one count as just as good
VERSION = 3        # bump when the worker's output changes, so cached takes are remade


class DubError(RuntimeError):
    pass


def norm(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


@dataclass
class Item:
    index: int                 # position in the film's span list
    span: Span
    key: str                   # word-table entry matched
    word: str                  # the transcript word replaced
    alts: list[str]
    win_start: float           # film time of the window's first sample
    crop: tuple[float, float]  # seconds within the window
    edit: tuple[float, float]  # seconds within the crop
    texts: dict[str, str] = field(default_factory=dict)
    edits: dict[str, tuple[float, float]] = field(default_factory=dict)   # per replacement, when longer


@dataclass
class DubResult:
    delta: Path | None                                   # the change track (FL, FR, FC), or None
    dubbed: dict[int, str] = field(default_factory=dict)  # span index -> replacement said
    notes: list[dict] = field(default_factory=list)       # per word, for the .censor.json


def table(cfg: dict) -> dict[tuple[str, ...], list[str]]:
    # [dub.words] adds to and overrides the built-in table; an empty list drops a word
    words = {**DEFAULT_WORDS, **(cfg.get("dub", {}).get("words") or {})}
    out = {}
    for k, v in words.items():
        alts = [v] if isinstance(v, str) else [str(x) for x in v]
        key = tuple(norm(w) for w in str(k).split() if norm(w))
        if key and alts:
            out[key] = alts
    return out


def match(span: Span, words: list, table_: dict) -> tuple[int, tuple[str, ...]] | None:
    """(index of the transcript word to replace, word-table key) for a span, or None."""
    hits = [i for i, w in enumerate(words) if w.end > span.start and w.start < span.end]
    best = None
    for i in hits:
        n = norm(words[i].text)
        for key in sorted(table_, key=len, reverse=True):
            if key[-1] != n:
                continue
            if len(key) > 1 and [norm(w.text) for w in words[max(0, i - len(key) + 1):i]] != list(key[:-1]):
                continue
            overlap = min(words[i].end, span.end) - max(words[i].start, span.start)
            if best is None or overlap > best[2]:
                best = (i, key, overlap)
            break
    return (best[0], best[1]) if best else None


def plan(spans: list[Span], words: list, cfg: dict) -> list[Item]:
    """The muted words that can be dubbed: heard by Whisper (so their timing is known)
    and in the word table."""
    table_ = table(cfg)
    items = []
    for idx, s in enumerate(spans):
        if s.estimated or "asr" not in s.sources:
            continue
        m = match(s, words, table_)
        if m is None:
            continue
        hit, key = m
        w = words[hit]
        win_start = max(0.0, s.start - PAD)
        crop_start = max(win_start, s.start - BEFORE)
        crop_end = s.end + AFTER
        crop = [x for x in words if x.end > crop_start and x.start < crop_end]
        lead, _, trail = re.match(r"^(\W*)(.*?)(\W*)$", w.text.strip()).groups()   # keep "Oh, shit." 's full stop
        texts = {alt: " ".join(f"{lead}{alt}{trail}" if x is w else x.text.strip() for x in crop) for alt in table_[key]}
        lo, hi = _edit_bounds(w, s, crop)
        e0, e1 = max(0.0, lo - crop_start), hi - crop_start
        # a longer word ("shit" -> "sugar") gets room in the pause after it, at the actor's
        # own pace, instead of being squeezed into the old word's time
        nxt = min([x.start for x in words if x.start >= hi - SWALLOW] + [crop_end])
        room = max(0.0, min(ROOM, nxt - GAP - (crop_start + e1), crop_end - 0.5 - (crop_start + e1)))
        pace = _pace([x for x in crop if x is not w])
        edits = {}
        for alt in table_[key]:
            extra = min(room, len(norm(alt)) / pace + 0.08 - (e1 - e0))
            if extra > 0.02:
                edits[alt] = (e0, round(e1 + extra, 3))
        items.append(Item(idx, s, " ".join(key), w.text.strip(), table_[key], win_start,
                          (crop_start - win_start, crop_end - win_start), (e0, e1), texts, edits))
    return items


def _edit_bounds(w, s: Span, crop: list) -> tuple[float, float]:
    """(start, end) film time the model rewrites for word `w` muted by span `s`.

    The mute span is padded, and the transcript's word edges are rough, so the old word may
    run past them: the edit covers both. But a padded span often reaches into the words on
    either side, and a model asked to rebuild half of "class" or "me" mangles it ("ticks me
    off", "whose guts"): a neighbour the edit reaches well into is taken whole (the model
    says it again, as in the line), and one it only grazes stops the edit at its edge."""
    lo, hi = min(w.start - 0.04, s.start), max(w.end + 0.04, s.end)
    others = [x for x in crop if x is not w]
    for x in others:                       # neighbours well inside the edit: say them again
        ov = min(x.end, hi) - max(x.start, lo)
        if ov >= SWALLOW:
            lo, hi = min(lo, x.start - 0.02), max(hi, x.end + 0.02)
    for x in others:                       # neighbours it only grazes: stop at their edge
        ov = min(x.end, hi) - max(x.start, lo)
        if 0 < ov < SWALLOW:
            if x.start >= w.end - 1e-6:
                hi = max(w.end, min(hi, x.start))
            else:
                lo = min(w.start, max(lo, x.end))
    return lo, hi


def _pace(words: list) -> float:
    """The speaker's letters per second over the line (a typical pace when it is too short)."""
    letters = sum(len(norm(x.text)) for x in words)
    secs = sum(max(0.0, x.end - x.start) for x in words)
    if secs < 0.5 or letters < 6:
        return 14.0
    return min(22.0, max(8.0, letters / secs))


def _windows(media: Path, stream: int, items: list[Item], rate: int, chunk_s: float = 10.0):
    """FL, FR, FC of the dialogue track around every item: {index: array (frames, 3)}.

    Read in one pass from the start, as the render reads the track: a seek into the middle
    (ffmpeg -ss) can land a few samples off with some codecs (TrueHD, DTS), and a take made on
    a window even 8 samples out of step no longer cancels the old word's highs, so both words
    are heard at once."""
    import numpy as np
    spans = []
    for it in items:
        a = int(round(it.win_start * rate))
        spans.append((a, a + int((it.span.end + PAD - it.win_start) * rate), it.index))
    out = {idx: np.zeros((b - a, 3), np.float32) for a, b, idx in spans}
    end = max(b for _, b, _ in spans)
    cmd = ["ffmpeg", "-v", "error", "-i", str(media), "-map", f"0:a:{stream}", "-t", f"{end / rate + 1:.3f}",
           "-af", "pan=3c|c0=FL|c1=FR|c2=FC", "-ar", str(rate), "-f", "f32le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    step = int(chunk_s * rate) * 12
    pos = 0
    try:
        while pos < end:
            buf = proc.stdout.read(step)
            if not buf:
                break
            x = np.frombuffer(buf[:len(buf) // 12 * 12], np.float32).reshape(-1, 3)
            for a, b, idx in spans:
                lo, hi = max(a, pos), min(b, pos + len(x))
                if lo < hi:
                    out[idx][lo - a:hi - a] = x[lo - pos:hi - pos]
            pos += len(x)
    finally:
        proc.stdout.close()
        proc.kill()
        err = proc.stderr.read().decode(errors="replace")
        proc.wait()
    if pos < max(a for a, _, _ in spans):
        raise MediaError(f"could not read the dialogue audio: {err[-300:]}")
    return out


def _to16k(x, rate: int):
    import numpy as np
    r = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(rate), "-ac", "1", "-i", "-",
                        "-ar", "16000", "-f", "f32le", "-"], input=x.astype(np.float32).tobytes(),
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32)


def score(heard: list, alt: str, old: str, t0: float, t1: float, min_prob: float) -> tuple[bool, float]:
    """Whisper's words (start, end, text, probability) near the dubbed word -> (passes, score)."""
    near = [w for w in heard if w[1] > t0 - 0.3 and w[0] < t1 + 0.3]
    target = norm(alt)
    joined = "".join(norm(w[2]) for w in near)
    probs = [w[3] for w in near if norm(w[2]) and norm(w[2]) in target]
    p = (sum(probs) / len(probs)) if probs and target in joined else 0.0
    o = norm(old)
    old_heard = any(norm(w[2]).startswith(o) or (len(o) > 3 and o[:4] in norm(w[2])) for w in near)
    return (p >= min_prob and not old_heard), p - (1.0 if old_heard else 0.0)


def natural(take: dict) -> float:
    """How natural a take sounds (higher is better): its match to the actor's voice, less
    a penalty for a loudness step at its edges the original does not have, or a word far
    louder or softer than the one it replaces."""
    return (float(take.get("voice", 0.0)) - 0.03 * max(0.0, float(take.get("edge_db", 0.0)) - 3.0)
            - 0.03 * max(0.0, abs(float(take.get("level_db", 0.0))) - 4.0))


def choose(items: list[Item], passing: dict[int, list[dict]]) -> dict[int, dict]:
    """One take per word. The most natural takes come first; among the replacements with a
    take about as natural (within CLOSE), the one used least so far for that word wins, so a
    film varies its swaps."""
    used: dict[str, dict[str, int]] = {}
    chosen = {}
    for it in sorted(items, key=lambda x: x.span.start):
        takes = passing.get(it.index) or []
        if not takes:
            continue
        top = max(natural(t) for t in takes)
        takes = [t for t in takes if natural(t) >= top - CLOSE]
        counts = used.setdefault(it.key, {})
        best = min(takes, key=lambda t: (counts.get(t["said"], 0), it.alts.index(t["said"]) if t["said"] in it.alts else 99,
                                         -natural(t), -t["score"]))
        counts[best["said"]] = counts.get(best["said"], 0) + 1
        chosen[it.index] = best
    return chosen


# the speech-editing engines: worker script, and the settings that change their takes
ENGINES = {
    "f5": "dub_worker.py",              # F5-TTS ([dub] python, steps, seeds, cfg)
    "playdiffusion": "dub_pd.py",       # PlayHT PlayDiffusion ([dub.playdiffusion] python, seeds, steps)
    "auk": "dub_auk.py",                # Tencent AuK ([dub.auk] python, ckpts, seeds)
}


def engine_settings(d: dict, name: str) -> dict:
    """An engine's settings: its Python and what it is asked (F5's live in [dub] itself)."""
    if name == "f5":
        return {"python": str(d.get("python") or ""), "steps": int(d.get("steps", 32)),
                "seeds": list(d.get("seeds", [1, 2])), "cfg": [float(x) for x in d.get("cfg", [2.0, 3.0])]}
    e = d.get(name) or {}
    out = {"python": str(e.get("python") or ""), "seeds": list(e.get("seeds", [1, 2, 3, 4] if name == "playdiffusion" else [1, 2]))}
    if name == "playdiffusion":
        out["steps"] = int(e.get("steps", 30))
    if name == "auk":
        out["ckpts"] = str(e.get("ckpts") or "")
    return out


def retry_seeds(d: dict, name: str) -> list[int]:
    """New seeds for a second round of takes, for words no first take could do ([dub] retry)."""
    n = int(d.get("retry", 4))
    if n <= 0:
        return []
    first = max(engine_settings(d, name)["seeds"] or [0])
    return list(range(first + 1, first + 1 + n))


def engines(d: dict) -> tuple[list[str], list[str]]:
    """(engines every word is tried with, engines only for words those could not do)."""
    main = [str(x) for x in (d.get("engines") or ["f5"])]
    fall = [str(x) for x in (d.get("fallback") or []) if str(x) not in main]
    return main, fall


def available(cfg: dict) -> str | None:
    """Why dubbing cannot run, or None when it can."""
    d = cfg.get("dub", {})
    if not d.get("enabled"):
        return "off"
    py = Path(str(d.get("python") or "")).expanduser()
    if not str(d.get("python") or "") or not py.exists():       # needed by every engine: it separates the voice
        return f"[dub] python {str(py)!r} not found (the F5-TTS model environment's Python)"
    main, fall = engines(d)
    for name in main + fall:
        if name not in ENGINES:
            return f"[dub] engine {name!r} is unknown (one of {', '.join(ENGINES)})"
        epy = engine_settings(d, name)["python"]
        if not epy or not Path(epy).expanduser().exists():
            return f"[dub.{name}] python {epy!r} not found"
        if name == "auk" and not Path(engine_settings(d, name)["ckpts"]).expanduser().is_dir():
            return "[dub.auk] ckpts: the folder with AuK/ and Qwen2.5-Omni-3B/ was not found"
    return None


def _run_worker(py: str, script: str, job: dict, work: Path, name: str) -> list[dict]:
    path = work / f"job-{name}.json"
    path.write_text(json.dumps(job))
    out = Path(job["out"])
    proc = subprocess.run([str(Path(py).expanduser()), str(Path(__file__).with_name(script)), str(path)],
                          stdout=sys.stderr, stderr=None, check=False)
    if proc.returncode != 0 or not (out / "results.json").exists():
        raise DubError(f"the {name} dub worker failed (exit {proc.returncode})")
    return json.loads((out / "results.json").read_text())


class _Takes:
    """The takes of every word for one film, kept per word and engine in the take store and
    reused while that word's job is the same: a stopped run, or one after adding swaps, only
    makes the takes it does not have yet."""

    def __init__(self, media: Path, d: dict, work: Path, base: dict, entries: list[dict]):
        self.d, self.work, self.base = d, work, base
        self.store = _store(media, d, work)
        self.entries = {e["id"]: e for e in entries}
        common = {k: base[k] for k in ("rate", "tone", "v")}
        f5 = engine_settings(d, "f5")
        # F5's key is the word's key: the separated voice is kept with it
        self.keys = {i: _item_key(e, {**common, **{k: f5[k] for k in ("steps", "seeds", "cfg")}}, media)
                     for i, e in self.entries.items()}
        self.takes: dict[int, list[dict]] = {}

    def key(self, i: int, name: str, seeds: list[int] | None = None) -> str:
        if name == "f5" and seeds is None:
            return self.keys[i]
        import hashlib
        st = {k: v for k, v in engine_settings(self.d, name).items() if k != "python"}
        if seeds is not None:
            st["seeds"] = seeds
        return hashlib.sha1(json.dumps([self.keys[i], name, st], sort_keys=True).encode()).hexdigest()[:16]

    def voice(self, i: int) -> dict:
        here = self.store / self.keys[i]
        v = {"voice": str(here / "voice.npy")}
        if self.entries[i].get("sides"):
            v["voice_sides"] = str(here / "voice_sides.npy")
        return v if all(Path(p).exists() for p in v.values()) else {}

    def _keep(self, r: dict, k: str, name: str) -> list[dict]:
        """Move a worker's takes into the store under key `k`."""
        import shutil
        dest = self.store / k
        dest.mkdir(parents=True, exist_ok=True)
        takes = []
        for t in r["takes"]:
            to = dest / Path(t["delta"]).name
            shutil.move(t["delta"], to)
            takes.append({**t, "delta": str(to), "engine": name, "key": k})
        for f, to in (("voice", "voice.npy"), ("voice_sides", "voice_sides.npy")):
            if r.get(f):
                (self.store / self.keys[r["id"]]).mkdir(parents=True, exist_ok=True)
                shutil.move(r[f], self.store / self.keys[r["id"]] / to)
        self._save(k, takes)
        return takes

    def _save(self, k: str, takes: list[dict]) -> None:
        (self.store / f"{k}.json").write_text(json.dumps([{a: b for a, b in t.items() if a not in ("engine", "key")}
                                                          for t in takes]))

    def make(self, name: str, ids: list[int], seeds: list[int] | None = None) -> dict[int, list[dict]]:
        """Takes of `name` for the words `ids` (from the store, or made now), with the
        engine's own seeds or `seeds` (a second round); returns them per word."""
        import shutil
        got: dict[int, list[dict]] = {}
        todo = []
        for i in ids:
            k = self.key(i, name, seeds)
            f = self.store / f"{k}.json"
            takes = json.loads(f.read_text()) if f.exists() else None
            if takes is not None and all(Path(t["delta"]).exists() for t in takes):
                got[i] = [{**t, "engine": name, "key": k} for t in takes]
            else:
                todo.append(i)
        if len(todo) < len(ids):
            eprint(f"  dub: {name}: reusing the takes made earlier for {len(ids) - len(todo)} word(s)")
        if todo:
            got.update(self._run(name, todo, seeds))
        for i, ts in got.items():
            self.takes.setdefault(i, []).extend(ts)
        return got

    def _run(self, name: str, todo: list[int], seeds: list[int] | None) -> dict[int, list[dict]]:
        import shutil
        st = engine_settings(self.d, name)
        if seeds is not None:
            st["seeds"] = seeds
        if name != "f5":            # the voice is separated by the F5 environment (Demucs lives there)
            self.separate(todo)
        items = [{**self.entries[i], **(self.voice(i) if name != "f5" else {})} for i in todo]
        run = self.store / f"run-{name}"
        shutil.rmtree(run, ignore_errors=True)
        eprint(f"  dub: {name}: making {'more ' if seeds is not None else ''}takes for {len(todo)} word(s)")
        job = {**self.base, **{k: v for k, v in st.items() if k != "python"}, "items": items, "out": str(run)}
        got = {}
        for r in _run_worker(st["python"], ENGINES[name], job, self.work, name):
            got[r["id"]] = self._keep(r, self.key(r["id"], name, seeds), name)
        shutil.rmtree(run, ignore_errors=True)
        return got

    def separate(self, ids: list[int]) -> None:
        import shutil
        need = [i for i in ids if not self.voice(i)]
        if not need:
            return
        run = self.store / "run-separate"
        shutil.rmtree(run, ignore_errors=True)
        job = {**self.base, "mode": "separate", "items": [self.entries[i] for i in need], "out": str(run)}
        for r in _run_worker(str(self.d["python"]), ENGINES["f5"], job, self.work, "separate"):
            for f, to in (("voice", "voice.npy"), ("voice_sides", "voice_sides.npy")):
                if r.get(f):
                    (self.store / self.keys[r["id"]]).mkdir(parents=True, exist_ok=True)
                    shutil.move(r[f], self.store / self.keys[r["id"]] / to)
        shutil.rmtree(run, ignore_errors=True)

    def score_voices(self, ids: list[int]) -> None:
        """Voice match for the takes that lack one (engines without a speaker model), so takes
        from every engine are ranked alike; saved with the takes."""
        import shutil
        req = []
        for i in ids:
            miss = [t for t in self.takes.get(i, []) if "voice" not in t]
            if miss and self.voice(i):
                req.append((i, miss))
        if not req:
            return
        run = self.store / "run-voice"
        shutil.rmtree(run, ignore_errors=True)
        items = [{**self.entries[i], **self.voice(i), "takes": [{"delta": t["delta"], "start": t["start"]} for t in miss]}
                 for i, miss in req]
        job = {**self.base, "mode": "voice", "items": items, "out": str(run)}
        res = {r["id"]: r["voices"] for r in _run_worker(str(self.d["python"]), ENGINES["f5"], job, self.work, "voice")}
        shutil.rmtree(run, ignore_errors=True)
        for i, miss in req:
            for t, v in zip(miss, res.get(i, [])):
                if v is not None:
                    t["voice"] = v
            for k in {t["key"] for t in miss}:              # keep the scores for the next run
                self._save(k, [t for t in self.takes[i] if t.get("key") == k])

    def tidy(self, names: list[str]) -> None:
        """Forget the takes of words this film no longer has (and of settings no longer used)."""
        import shutil
        live = {self.key(i, n, s) for i in self.entries for n in names
                for s in (None, retry_seeds(self.d, n) or None)}
        for f in self.store.glob("*.json"):
            if f.stem not in live:
                f.unlink(missing_ok=True)
                shutil.rmtree(self.store / f.stem, ignore_errors=True)


def _check(items: list[Item], wins: dict, takes: dict[int, list[dict]], cfg: dict, rate: int) -> dict[int, list[dict]]:
    """Whisper listens to every take: the new word clearly, nothing of the old one."""
    import numpy as np
    from . import asr
    model = asr._get_model(cfg)
    min_prob = float(cfg.get("dub", {}).get("min_prob", 0.5))
    passing: dict[int, list[dict]] = {}
    for it in items:
        win = wins[it.index]
        t0 = it.span.start - it.win_start
        a, b = max(0, int((t0 - 3) * rate)), int((t0 + it.span.duration + 3) * rate)
        for take in takes.get(it.index, []):
            delta = np.load(take["delta"])
            x = apply(win, delta, take["start"])
            # heard as the front speakers play it: a leak from any of them fails the take
            mix = 0.5 * (x[:, 0] + x[:, 1]) + 0.707 * x[:, 2]
            segs, _ = model.transcribe(_to16k(mix[a:b], rate), language="en", word_timestamps=True,
                                       condition_on_previous_text=False)
            heard = [(w.start, w.end, w.word, w.probability) for sg in segs for w in (sg.words or [])]
            off = a / rate
            t1 = max(t0 + it.span.duration, (take["start"] + len(delta)) / rate - 0.03)
            ok, sc = score(heard, take["said"], it.word, t0 - off, t1 - off, min_prob)
            if ok:
                passing.setdefault(it.index, []).append({**take, "score": sc})
    return passing


def prepare(media: Path, stream: int, rate: int, duration: float, spans: list[Span], words: list,
            cfg: dict, work: Path) -> DubResult:
    """Make the change track for `media`; words that cannot be dubbed are left to the mute."""
    import numpy as np
    from . import asr
    d = cfg.get("dub", {})
    items = plan(spans, words, cfg)
    if not items:
        eprint("  dub: no muted word has a replacement in the word table; muting as usual")
        return DubResult(None)
    main, fall = engines(d)
    eprint(f"  dub: {len(items)} of {len(spans)} muted word(s) can be dubbed; making takes with "
           f"{' + '.join(main)}" + (f" ({' + '.join(fall)} for the rest)" if fall else ""))
    wins = _windows(media, stream, items, rate)
    entries = []
    for it in items:
        w = wins[it.index]
        np.save(work / f"w{it.index}.npy", w[:, 2])
        entry = {"id": it.index, "window": str(work / f"w{it.index}.npy"), "crop": list(it.crop),
                 "edit": list(it.edit), "texts": it.texts, "edits": {a: list(e) for a, e in it.edits.items()},
                 "word": it.word, "label": f"{it.span.start:.0f}s {it.word!r} -> {'/'.join(it.alts)}"}
        if side_share(w, it) > LEAK:          # the voice plays from the front left/right too
            np.save(work / f"s{it.index}.npy", w[:, :2])
            entry["sides"] = str(work / f"s{it.index}.npy")
        entries.append(entry)
    base = {"rate": rate, "cache": str(d.get("cache") or ""), "tone": bool(d.get("tone", True)), "v": VERSION}
    store = _Takes(media, d, work, base, entries)
    ids = [it.index for it in items]
    for name in main:
        store.make(name, ids)
    store.score_voices(ids)
    passing = _check(items, wins, store.takes, cfg, rate)
    left = [i for i in ids if not passing.get(i)]

    def more(name: str, seeds: list[int] | None = None) -> None:
        nonlocal left
        new = store.make(name, left, seeds)
        store.score_voices(left)
        for i, ts in _check([it for it in items if it.index in left], wins, new, cfg, rate).items():
            passing.setdefault(i, []).extend(ts)
        left = [i for i in left if not passing.get(i)]

    if left and int(d.get("retry", 4)) > 0:       # a second round with new seeds, for those words only
        eprint(f"  dub: {len(left)} word(s) with no good take; more takes with {' + '.join(main)}")
        for name in main:
            if left:
                more(name, retry_seeds(d, name))
    for name in fall:
        if not left:
            break
        eprint(f"  dub: {len(left)} word(s) with no good take; trying {name}")
        asr.release_models()                  # the GPU memory goes to the bigger model
        from . import thrash
        with thrash.relaxed(name == "auk"):   # its 12 GB load trips the kernel's thrashing guard
            more(name)
    store.tidy(main + fall)
    chosen = choose(items, passing)

    # the change track: zero everywhere except the dubbed words
    notes = []
    for it in items:
        take = chosen.get(it.index)
        made = store.takes.get(it.index, [])
        notes.append({"start": round(it.span.start, 3), "word": it.word, "said": take["said"] if take else None,
                      "takes": len(made), "passed": len(passing.get(it.index, [])),
                      **({"natural": round(natural(take), 3), "engine": take.get("engine", "f5")} if take else {})})
        if take:
            eprint(f"  dub: {it.span.start:7.1f}s {it.word!r} -> {take['said']!r}"
                   f"  ({len(passing[it.index])}/{len(made)} takes passed; {take.get('engine', 'f5')},"
                   f" voice {take.get('voice', '?')}, edges {take.get('edge_db', '?')} dB)")
        else:
            eprint(f"  dub: {it.span.start:7.1f}s {it.word!r}: no take sounded right; muted instead")
    if not chosen:
        return DubResult(None, {}, notes)
    patches = []
    for it in items:
        take = chosen.get(it.index)
        if not take:
            continue
        patch = as3(np.load(take["delta"]))
        patches.append((int(round(it.win_start * rate)) + take["start"], patch))
    out = work / "dub-delta.flac"
    write_delta(out, patches, int((duration + 2) * rate), rate)
    return DubResult(out, {i: t["said"] for i, t in chosen.items()}, notes)


def _store(media: Path, d: dict, work: Path) -> Path:
    """This film's take folder: [dub] cache/takes/<film> (the work folder without a cache)."""
    if not d.get("cache"):
        return work / "takes"
    root = Path(str(d["cache"])).expanduser() / "takes"
    stem = re.sub(r"[^A-Za-z0-9]+", "-", media.stem)[:60]
    if root.exists():                       # takes kept per film by an earlier version
        import shutil
        for old in root.glob(f"{stem}-*"):
            if old.is_dir() and re.fullmatch(r"[0-9a-f]{12}", old.name[len(stem) + 1:]):
                shutil.rmtree(old, ignore_errors=True)
    here = root / stem
    here.mkdir(parents=True, exist_ok=True)
    return here


def _item_key(entry: dict, common: dict, media: Path) -> str:
    """Names one word's job: what the worker is asked, and the audio it is asked about (the
    window itself, not the film file, whose size changes whenever a clean track is written
    into it), not where its temporary files are."""
    import hashlib
    h = hashlib.sha1()
    for f in ("window", "sides"):
        if entry.get(f) and Path(entry[f]).exists():
            h.update(Path(entry[f]).read_bytes())
    job = {**{k: v for k, v in entry.items() if k not in ("window", "sides", "id", "label", "word")},
           "sides": bool(entry.get("sides")), **common, "film": media.name, "audio": h.hexdigest()}
    return hashlib.sha1(json.dumps(job, sort_keys=True).encode()).hexdigest()[:16]


def side_share(win, it: Item) -> float:
    """How loud the front left/right are against the centre on the word (an upmix: ~1)."""
    import numpy as np
    rate = len(win) / max(1e-9, it.span.end + PAD - it.win_start)
    on = win[int((it.span.start - it.win_start) * rate):int((it.span.end - it.win_start) * rate)]
    if not len(on):
        return 0.0
    return float(np.sqrt(np.mean(on[:, :2] ** 2))) / max(float(np.sqrt(np.mean(on[:, 2] ** 2))), 1e-9)


def as3(delta):
    """A take's change as (frames, 3) FL, FR, FC (a centre-only take is FC alone)."""
    import numpy as np
    if delta.ndim == 2:
        return delta.astype(np.float32)
    out = np.zeros((len(delta), 3), np.float32)
    out[:, 2] = delta
    return out


def apply(win, delta, start: int):
    """The (frames, 3) window with a take's change added at `start`."""
    x = win.copy()
    d = as3(delta)[:max(0, len(x) - start)]
    x[start:start + len(d)] += d
    return x


def write_delta(path: Path, patches: list, frames: int, rate: int, chunk_s: float = 30.0) -> None:
    """Stream a 3-channel track of `frames` samples, silent except the patches, to FLAC."""
    import numpy as np
    patches = sorted(patches, key=lambda p: p[0])
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(rate), "-ac", "3", "-i", "-",
                             "-c:a", "flac", "-sample_fmt", "s32", str(path)], stdin=subprocess.PIPE)
    step = int(chunk_s * rate)
    try:
        for c0 in range(0, frames, step):
            c1 = min(frames, c0 + step)
            buf = np.zeros((c1 - c0, 3), np.float32)
            for s0, p in patches:
                a, b = max(c0, s0), min(c1, s0 + len(p))
                if a < b:
                    buf[a - c0:b - c0] += p[a - s0:b - s0]
            proc.stdin.write(buf.tobytes())
    finally:
        proc.stdin.close()
        if proc.wait() != 0:
            raise MediaError("could not write the dub change track")


def work_dir(root: Path | None = None) -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix="audio-censor-dub-", dir=str(root) if root else None)
