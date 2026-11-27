import numpy as np
from pathlib import Path

from audio_censor import dub
from audio_censor.asr import Word
from audio_censor.beep import BeepSpec, build_filtergraph
from audio_censor.media import probe
from audio_censor.spans import Span


def _words(text, start=10.0, step=0.4):
    return [Word(start + i * step, start + i * step + 0.3, w) for i, w in enumerate(text.split())]


def test_your_words_add_to_the_table_and_an_empty_list_drops_one():
    t = dub.table({"dub": {"words": {"fart": ["toot", "trump"], "hell": []}}})
    assert t[("fart",)] == ["toot", "trump"]
    assert ("hell",) not in t
    assert t[("damn",)] == ["dang", "darn"]


def test_a_phrase_key_replaces_only_its_last_word_and_only_in_the_phrase():
    t = dub.table({})
    words = _words("all right you son of a bitch this is my cab")
    bitch = words[6]
    span = Span(bitch.start, bitch.end, ["son of a bitch"], ["asr"])
    assert dub.match(span, words, t) == (6, ("son", "of", "a", "bitch"))
    alone = _words("you little bitch")
    assert dub.match(Span(alone[2].start, alone[2].end, ["bitch"], ["asr"]), alone, t) is None


def test_plan_gives_the_model_the_line_with_the_word_swapped():
    words = _words("God you're a tight ass How'd you like a mouth full of teeth")
    ass = words[4]
    spans = [Span(ass.start - 0.1, ass.end + 0.1, ["ass"], ["asr", "subtitle"]),
             Span(30, 31, ["ass"], ["subtitle"], estimated=True)]          # no word timing: muted
    items = dub.plan(spans, words, {})
    assert len(items) == 1
    it = items[0]
    assert it.texts["bum"].startswith("God you're a tight bum How'd")
    crop0 = it.crop[0] + it.win_start
    assert abs(crop0 + it.edit[0] - (ass.start - 0.1)) < 1e-6         # the whole muted span is redone
    assert abs(crop0 + it.edit[1] - (ass.end + 0.1)) < 1e-6


def test_a_take_passes_only_when_the_new_word_is_heard_and_the_old_one_is_not():
    good = [(0.0, 0.3, " tight", 0.9), (0.4, 0.7, " bum.", 0.8)]
    assert dub.score(good, "bum", "ass", 0.4, 0.7, 0.5)[0]
    mumbled = [(0.0, 0.3, " tight", 0.9), (0.4, 0.7, " bus", 0.8)]
    assert not dub.score(mumbled, "bum", "ass", 0.4, 0.7, 0.5)[0]
    leftover = [(0.4, 0.6, " doof", 0.9), (0.6, 0.8, "-hole", 0.9)]
    assert not dub.score(leftover, "doofus", "asshole", 0.4, 0.8, 0.5)[0]
    assert not dub.score([(0.4, 0.7, " asshole", 0.9)], "jerk", "asshole", 0.4, 0.7, 0.5)[0]
    two = [(0.4, 0.6, " lowly", 0.9), (0.6, 0.9, " trucker", 0.8)]
    assert dub.score(two, "lowly-trucker", "motherfucker", 0.4, 0.9, 0.5)[0]


def test_choices_vary_through_a_film():
    words = _words("damn it damn it damn it damn it")
    spans = [Span(w.start, w.end, ["damn"], ["asr"]) for w in words[::2]]
    items = dub.plan(spans, words, {})
    passing = {it.index: [{"said": "dang", "score": 0.9}, {"said": "darn", "score": 0.6}] for it in items}
    passing[items[2].index] = [{"said": "darn", "score": 0.7}]
    said = [dub.choose(items, passing)[it.index]["said"] for it in items]
    assert said == ["dang", "darn", "darn", "dang"]


def test_the_change_track_is_added_before_the_left_over_mutes():
    spec = BeepSpec(mode="mute", duck=0.0)
    g = build_filtergraph([Span(5, 6)], spec, audio_label="0:a:0", layout="5.1(side)", channels=6,
                          rate=48000, dub_input=1)
    assert g.startswith("[0:a:0][1:a]amerge=inputs=2,pan=5.1(side)|FL=c0+c6|FR=c1+c7|FC=c2+c8|LFE=c3|SL=c4|SR=c5[dubbed]")
    assert "[dubbed]" in g.split(";")[1] and "between(t,5,6)" in g
    only = build_filtergraph([], spec, audio_label="0:a:0", layout="7.1", channels=8, rate=48000, dub_input=2)
    assert only.endswith("[dubbed]anull[clean]") and "FC=c2+c10" in only


def test_write_delta_is_silent_except_the_patches(tmp_path):
    p = np.zeros((480, 3), np.float32)
    p[:, 2] = 0.5
    out = tmp_path / "d.flac"
    dub.write_delta(out, [(48000, p)], 48000 * 3, 48000, chunk_s=1.0)
    assert abs(probe(out).duration - 3.0) < 0.05


def test_subtitles_show_the_dubbed_word_instead_of_a_mask():
    import pysubs2
    from audio_censor.config import load_config
    from audio_censor.subtitles import censor_subs
    from audio_censor.wordlist import Matcher
    subs = pysubs2.SSAFile()
    subs.events = [pysubs2.SSAEvent(start=10_000, end=12_000, text="You son of a bitch!"),
                   pysubs2.SSAEvent(start=20_000, end=22_000, text="Shit. Oh, shit."),
                   pysubs2.SSAEvent(start=30_000, end=32_000, text="Bullshit.")]
    m = Matcher.from_config(load_config())
    clean, _ = censor_subs(subs, m, "asterisks", "", dubs=[(11.0, "bitch", "gun"), (21.5, "shit", "sugar")])
    t = [e.text for e in clean.events]
    assert t[0] == "You son of a gun!"
    assert t[1] == "****. Oh, sugar." or t[1] == "Sugar. Oh, ****."     # one dubbed, one still muted
    assert t[2].startswith("****")                                     # not dubbed: masked as before


def test_a_take_can_change_the_front_left_and_right_too():
    win = np.zeros((100, 3), np.float32)
    centre_only = np.ones(10, np.float32)
    x = dub.apply(win, centre_only, 20)
    assert x[25, 2] == 1 and x[25, 0] == 0 and x[35, 2] == 0
    sides = np.ones((10, 3), np.float32) * [0.5, 0.25, 1.0]
    y = dub.apply(win, sides, 95)                       # clipped at the window's end
    assert y[97].tolist() == [0.5, 0.25, 1.0] and y.shape == (100, 3)
    it = dub.Item(0, Span(6.0, 7.0), "x", "x", ["y"], 0.0, (0, 1), (0, 1))
    w = np.zeros((48000 * 13, 3), np.float32)
    w[48000 * 6:48000 * 7] = [0.3, 0.3, 0.3]            # same level on every speaker: an upmix
    assert abs(dub.side_share(w, it) - 1.0) < 0.01


def test_a_longer_swap_gets_room_in_the_pause_after_the_word():
    from audio_censor.asr import Word
    words = [Word(10.0, 10.3, "Oh,"), Word(10.4, 10.55, "shit."), Word(11.4, 11.7, "Look"), Word(11.75, 12.0, "out")]
    s = Span(10.4, 10.55, ["shit"], ["asr"])
    it = dub.plan([s], words, {})[0]
    e0, e1 = it.edits["sugar"]
    assert e0 == it.edit[0] and e1 > it.edit[1]
    crop0 = it.win_start + it.crop[0]
    assert crop0 + e1 <= 11.4 - dub.GAP + 1e-6           # the next word is left alone
    tight = [Word(10.4, 10.55, "shit"), Word(10.6, 10.9, "happens")]
    it2 = dub.plan([Span(10.4, 10.55, ["shit"], ["asr"])], tight, {})[0]
    assert not any(e[1] > it2.edit[1] + 0.02 for e in it2.edits.values())   # no pause: no room


def test_the_most_natural_take_wins_and_variety_only_among_equals():
    words = _words("damn it damn it")
    spans = [Span(w.start, w.end, ["damn"], ["asr"]) for w in words[::2]]
    items = dub.plan(spans, words, {})
    poor = {"said": "dang", "score": 0.9, "voice": 0.40, "edge_db": 9.0}
    good = {"said": "darn", "score": 0.6, "voice": 0.70, "edge_db": 1.0}
    passing = {it.index: [poor, good] for it in items}
    said = [dub.choose(items, passing)[it.index]["said"] for it in items]
    assert said == ["darn", "darn"]                      # a far better take beats variety
    close = {"said": "dang", "score": 0.9, "voice": 0.68, "edge_db": 1.0}
    passing = {it.index: [close, good] for it in items}
    said = [dub.choose(items, passing)[it.index]["said"] for it in items]
    assert said == ["dang", "darn"]


def test_windows_line_up_with_the_film_sample_for_sample(tmp_path):
    import subprocess
    rate = 48000
    x = np.zeros((rate * 20, 6), np.float32)
    clicks = [rate * 7 + 123, rate * 15 + 4567]
    for c in clicks:
        x[c, 2] = 0.9                                   # a click on the centre channel
    src = tmp_path / "a.flac"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(rate), "-ac", "6", "-channel_layout",
                    "5.1(side)", "-i", "-", "-c:a", "flac", "-sample_fmt", "s32", str(src)], input=x.tobytes(), check=True)
    items = [dub.Item(i, Span(c / rate, c / rate + 0.2), "x", "x", ["y"], c / rate - 1.5, (0, 1), (0, 1))
             for i, c in enumerate(clicks)]
    wins = dub._windows(src, 0, items, rate, chunk_s=1.0)
    for it, c in zip(items, clicks):
        w = wins[it.index][:, 2]
        assert int(np.argmax(np.abs(w))) == c - int(round(it.win_start * rate))


def test_a_second_round_uses_new_seeds():
    d = {"seeds": [1, 2], "playdiffusion": {"seeds": [1, 2, 3, 4]}}
    assert dub.retry_seeds(d, "f5") == [3, 4, 5, 6]
    assert dub.retry_seeds(d, "playdiffusion") == [5, 6, 7, 8]
    assert dub.retry_seeds({**d, "retry": 0}, "f5") == []


def test_engines_and_their_settings():
    d = {"python": "/x/f5", "engines": ["f5", "playdiffusion"], "fallback": ["auk", "f5"],
         "playdiffusion": {"python": "/x/pd"}, "auk": {"python": "/x/auk", "ckpts": "/c"}}
    assert dub.engines(d) == (["f5", "playdiffusion"], ["auk"])
    assert dub.engines({}) == (["f5"], [])
    assert dub.engine_settings(d, "playdiffusion") == {"python": "/x/pd", "seeds": [1, 2, 3, 4], "steps": 30}
    assert dub.engine_settings(d, "auk")["ckpts"] == "/c"
    assert "not found" in dub.available({"dub": {**d, "enabled": True}})


def test_the_fallback_engine_only_gets_the_words_the_others_could_not_do(tmp_path, monkeypatch):
    words = _words("oh shit look out you jackass run")
    spans = [Span(words[1].start, words[1].end, ["shit"], ["asr"]), Span(words[5].start, words[5].end, ["jackass"], ["asr"])]
    media = tmp_path / "film.mkv"
    media.write_bytes(b"x")
    rate = 1000
    monkeypatch.setattr(dub, "_windows", lambda m, s, items, r: {it.index: np.zeros((20000, 3), np.float32) for it in items})
    asked = {}

    def worker(py, script, job, work, name):
        out = Path(job["out"])
        out.mkdir(parents=True, exist_ok=True)
        asked.setdefault(name, []).append([i["id"] for i in job["items"]])
        res = []
        for it in job["items"]:
            r = {"id": it["id"], "takes": []}
            if name in ("separate", "f5"):
                np.save(out / f"v{it['id']}.npy", np.zeros(20000, np.float32))
                r["voice"] = str(out / f"v{it['id']}.npy")
            if name == "voice":
                r = {"id": it["id"], "voices": [0.5 for _ in it["takes"]]}
            else:
                for alt in it["texts"]:
                    f = out / f"{name}{it['id']}{alt}.npy"
                    np.save(f, np.ones(10, np.float32))
                    r["takes"].append({"said": alt, "seed": 1, "delta": str(f), "start": 0, "edge_db": 1.0,
                                       **({"voice": 0.6} if name == "f5" else {})})
            res.append(r)
        return res
    monkeypatch.setattr(dub, "_run_worker", worker)

    def check(items, wins, takes, cfg, r):          # f5 can do "shit"; only auk can do "jackass"
        return {i: [{**t, "score": 0.9} for t in ts if (i == 0) == (t["engine"] in ("f5", "playdiffusion"))]
                for i, ts in takes.items() if any((i == 0) == (t["engine"] in ("f5", "playdiffusion")) for t in ts)}
    monkeypatch.setattr(dub, "_check", check)
    from audio_censor import asr
    monkeypatch.setattr(asr, "release_models", lambda: None)
    cfg = {"dub": {"python": "py", "cache": str(tmp_path / "c"), "engines": ["f5", "playdiffusion"],
                   "fallback": ["auk"], "playdiffusion": {"python": "pd"}, "auk": {"python": "auk", "ckpts": "c"}}}
    res = dub.prepare(media, 0, rate, 30.0, spans, words, cfg, tmp_path)
    assert asked["f5"] == [[0, 1], [1]] and asked["playdiffusion"] == [[0, 1], [1]]   # a second round for "jackass"
    assert asked["auk"] == [[1]]                        # only the word nobody else could do
    assert {n["engine"] for n in res.notes} == {"f5", "auk"} or {n["engine"] for n in res.notes} == {"playdiffusion", "auk"}
    assert set(res.dubbed) == {0, 1}
    asked.clear()
    media.write_bytes(b"x" * 100)                      # a clean track written into the film
    dub.prepare(media, 0, rate, 30.0, spans, words, cfg, tmp_path)
    assert "f5" not in asked and "playdiffusion" not in asked and "auk" not in asked   # all reused


def test_an_edit_takes_a_neighbour_it_reaches_into_and_stops_at_one_it_grazes():
    # "you piss me off": the padded mute runs well into "me" -> the model says "tick me" again
    piss, me, off = Word(2670.21, 2670.47, "piss"), Word(2670.47, 2670.67, "me"), Word(2670.67, 2670.95, "off")
    lo, hi = dub._edit_bounds(piss, Span(2670.11, 2670.57, ["piss"], ["asr"]), [Word(2669.9, 2670.1, "You"), piss, me, off])
    assert hi == off.start and lo < piss.start      # all of "me", up to "off"
    # a mute that only grazes the next word stops at its edge
    shit, look = Word(10.0, 10.3, "shit"), Word(10.32, 10.6, "look")
    lo, hi = dub._edit_bounds(shit, Span(9.95, 10.34, ["shit"], ["asr"]), [shit, look])
    assert hi == look.start and lo == 9.95
