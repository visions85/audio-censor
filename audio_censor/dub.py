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
        e0 = min(w.start - 0.04, s.start) - crop_start
        e1 = max(w.end + 0.04, s.end) - crop_start
        items.append(Item(idx, s, " ".join(key), w.text.strip(), table_[key], win_start,
                          (crop_start - win_start, crop_end - win_start), (max(0.0, e0), e1), texts))
    return items


def _windows(media: Path, stream: int, items: list[Item], rate: int):
    """FL, FR, FC of the dialogue track around every item: {index: array (frames, 3)}."""
    import numpy as np
    out = {}
    for it in items:
        dur = it.span.end + PAD - it.win_start
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{it.win_start:.3f}", "-t", f"{dur:.3f}", "-i", str(media),
               "-map", f"0:a:{stream}", "-af", "pan=3c|c0=FL|c1=FR|c2=FC", "-ar", str(rate), "-f", "f32le", "-"]
        raw = subprocess.run(cmd, capture_output=True, check=False)
        if raw.returncode != 0:
            raise MediaError(f"could not read the audio around {it.span.start:.1f} s: {raw.stderr.decode()[-300:]}")
        out[it.index] = np.frombuffer(raw.stdout, np.float32).reshape(-1, 3).copy()
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


def choose(items: list[Item], passing: dict[int, list[dict]]) -> dict[int, dict]:
    """One take per word: the replacement used least so far for that word wins, so a film
    varies its swaps; among its takes, Whisper's surest."""
    used: dict[str, dict[str, int]] = {}
    chosen = {}
    for it in sorted(items, key=lambda x: x.span.start):
        takes = passing.get(it.index) or []
        if not takes:
            continue
        counts = used.setdefault(it.key, {})
        best = min(takes, key=lambda t: (counts.get(t["said"], 0), it.alts.index(t["said"]) if t["said"] in it.alts else 99,
                                         -t["score"]))
        counts[best["said"]] = counts.get(best["said"], 0) + 1
        chosen[it.index] = best
    return chosen


def available(cfg: dict) -> str | None:
    """Why dubbing cannot run, or None when it can."""
    d = cfg.get("dub", {})
    if not d.get("enabled"):
        return "off"
    py = Path(str(d.get("python") or "")).expanduser()
    if not str(d.get("python") or "") or not py.exists():
        return f"[dub] python {str(py)!r} not found (the model environment's Python)"
    return None


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
    eprint(f"  dub: {len(items)} of {len(spans)} muted word(s) can be dubbed; making takes")
    wins = _windows(media, stream, items, rate)
    job_items = []
    for it in items:
        w = wins[it.index]
        np.save(work / f"w{it.index}.npy", w[:, 2])
        entry = {"id": it.index, "window": str(work / f"w{it.index}.npy"), "crop": list(it.crop),
                 "edit": list(it.edit), "texts": it.texts,
                 "label": f"{it.span.start:.0f}s {it.word!r} -> {'/'.join(it.alts)}"}
        if side_share(w, it) > LEAK:          # the voice plays from the front left/right too
            np.save(work / f"s{it.index}.npy", w[:, :2])
            entry["sides"] = str(work / f"s{it.index}.npy")
        job_items.append(entry)
    job = {"rate": rate, "items": job_items, "cache": str(d.get("cache") or ""),
           "steps": int(d.get("steps", 32)), "seeds": list(d.get("seeds", [1, 2])),
           "cfg": [float(x) for x in d.get("cfg", [2.0, 3.0])]}
    # takes are kept per word and reused while that word's job is the same, so a stopped
    # run, or one after adding swaps, only makes the takes it does not have yet
    import shutil
    store = _store(media, d, work)
    common = {k: job[k] for k in ("rate", "steps", "seeds", "cfg")}
    keys = {e["id"]: _item_key(e, common, media) for e in job_items}
    results, todo = {}, []
    for e in job_items:
        f = store / f"{keys[e['id']]}.json"
        takes = json.loads(f.read_text()) if f.exists() else None
        if takes and all(Path(t["delta"]).exists() for t in takes):
            results[e["id"]] = takes
        else:
            todo.append(e)
    if results:
        eprint(f"  dub: reusing the takes made earlier for {len(results)} word(s)")
    if todo:
        run = store / "run"
        shutil.rmtree(run, ignore_errors=True)
        (work / "job.json").write_text(json.dumps({**job, "items": todo, "out": str(run)}))
        py = str(Path(str(d["python"])).expanduser())
        worker = Path(__file__).with_name("dub_worker.py")
        proc = subprocess.run([py, str(worker), str(work / "job.json")], stdout=sys.stderr, stderr=None, check=False)
        if proc.returncode != 0 or not (run / "results.json").exists():
            raise DubError(f"the dub worker failed (exit {proc.returncode})")
        for r in json.loads((run / "results.json").read_text()):
            k = keys[r["id"]]
            dest = store / k
            dest.mkdir(parents=True, exist_ok=True)
            takes = []
            for t in r["takes"]:
                to = dest / Path(t["delta"]).name
                shutil.move(t["delta"], to)
                takes.append({**t, "delta": str(to)})
            (store / f"{k}.json").write_text(json.dumps(takes))
            results[r["id"]] = takes
        shutil.rmtree(run, ignore_errors=True)
    for f in store.glob("*.json"):            # words this film no longer has
        if f.stem not in keys.values():
            f.unlink(missing_ok=True)
            shutil.rmtree(store / f.stem, ignore_errors=True)

    # Whisper checks every take: the new word clearly, nothing of the old one
    model = asr._get_model(cfg)
    min_prob = float(d.get("min_prob", 0.5))
    passing: dict[int, list[dict]] = {}
    for it in items:
        win = wins[it.index]
        t0 = it.span.start - it.win_start
        a, b = max(0, int((t0 - 3) * rate)), int((t0 + it.span.duration + 3) * rate)
        for take in results.get(it.index, []):
            x = apply(win, np.load(take["delta"]), take["start"])
            # heard as the front speakers play it: a leak from any of them fails the take
            mix = 0.5 * (x[:, 0] + x[:, 1]) + 0.707 * x[:, 2]
            segs, _ = model.transcribe(_to16k(mix[a:b], rate), language="en", word_timestamps=True,
                                       condition_on_previous_text=False)
            heard = [(w.start, w.end, w.word, w.probability) for sg in segs for w in (sg.words or [])]
            off = a / rate
            ok, sc = score(heard, take["said"], it.word, t0 - off, t0 - off + it.span.duration, min_prob)
            if ok:
                passing.setdefault(it.index, []).append({**take, "score": sc})
    chosen = choose(items, passing)

    # the change track: zero everywhere except the dubbed words
    notes, starts = [], {}
    for it in items:
        take = chosen.get(it.index)
        notes.append({"start": round(it.span.start, 3), "word": it.word, "said": take["said"] if take else None,
                      "takes": len(results.get(it.index, [])), "passed": len(passing.get(it.index, []))})
        if take:
            eprint(f"  dub: {it.span.start:7.1f}s {it.word!r} -> {take['said']!r}"
                   f"  ({len(passing[it.index])}/{len(results.get(it.index, []))} takes passed)")
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
    """Names one word's job: what the worker is asked, not where its temporary files are."""
    import hashlib
    job = {**{k: v for k, v in entry.items() if k not in ("window", "sides", "id", "label")},
           "sides": bool(entry.get("sides")), **common, "film": [media.name, media.stat().st_size]}
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
