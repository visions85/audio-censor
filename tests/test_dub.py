import numpy as np

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
