from audio_censor.asr import Word, scan_words
from audio_censor.config import DEFAULTS, deep_merge
from audio_censor.names import NameDetector
from audio_censor.spans import build_spans
from audio_censor.subtitles import Cue, censor_text, scan_cues
from audio_censor.wordlist import Matcher

CFG = deep_merge(DEFAULTS, {"level": "mild"})
M = Matcher.from_config(CFG)

LINES = [
    "Hey Dick, pass the ball.",
    "I told Dick we'd be late.",
    "Dick! Over here!",
    "Don't be such a dick about it.",
    "You're a Dick, you know that?",
    "Oh, shit. That is a damn shame.",
    "DICK: We should go.",
    "WHAT THE HELL IS DICK DOING?",
]


def detector(lines=LINES, **names):
    d = NameDetector.from_config(deep_merge(CFG, {"names": names}))
    d.learn(lines, M)
    return d


def verdicts(d, line):
    from audio_censor.wordlist import tokenize
    toks = tokenize(line)
    return [(m.text, d.is_name_use(line, toks, m)) for m in M.find(toks)]


def test_learns_name_from_capitalized_uses_and_speaker_label():
    d = detector()
    assert d.names == {"dick"}
    assert d.stats["dick"].cap_mid >= 2 and d.stats["dick"].speaker_label == 1
    assert "shit" not in d.names and "hell" not in d.names


def test_per_occurrence_verdicts():
    d = detector()
    assert verdicts(d, "Hey Dick, pass the ball.") == [("dick", True)]
    assert verdicts(d, "Don't be such a dick about it.") == [("dick", False)]
    assert verdicts(d, "You're a Dick, you know that?") == [("dick", False)]    # determiner wins
    assert verdicts(d, "Dick! Over here!") == [("dick", True)]                   # sentence-initial, film says name
    assert verdicts(d, "WHAT THE HELL IS DICK DOING?") == [("hell", False), ("dick", True)]
    assert verdicts(d, "Shit happens.") == [("shit", False)]                     # capitalized but never a name here


def test_unknown_film_keeps_sentence_initial_as_swear():
    d = detector(["Dick move, man.", "That was lame."])      # one capitalized use, below threshold
    assert d.names == set()
    assert verdicts(d, "Dick move, man.") == [("dick", False)]


def test_ignore_and_disable():
    assert detector(ignore=["dick"]).names == set()
    d = detector(detect=False)
    assert verdicts(d, "Hey Dick, pass the ball.") == [("dick", False)]


def test_possessive_and_phrase_not_affected():
    d = detector()
    assert verdicts(d, "That is Dick's car.") == [("dick's", True)]
    assert verdicts(d, "You son of a Bitch.") == [("son of a bitch", False), ("bitch", False)]


def test_scan_cues_marks_name_uses_and_build_spans_drops_them():
    d = detector()
    cues = [Cue(1, 3, "Hey Dick, pass the ball."), Cue(4, 6, "Don't be such a dick about it.")]
    hits = scan_cues(cues, M, detector=d)
    assert [h.name_use for h in hits] == [True, False]
    spans = build_spans(hits, [], CFG)
    assert len(spans) == 1 and 4 <= spans[0].start <= spans[0].end <= 6.5


def test_asr_hits_inside_exempted_cue_are_dropped():
    d = detector()
    cues = [Cue(1, 3, "Hey Dick, pass the ball.")]
    sub_hits = scan_cues(cues, M, detector=d)
    asr = [Word(1.2, 1.5, " Hey"), Word(1.5, 1.9, " Dick,"), Word(1.9, 2.2, " pass")]
    asr_hits = scan_words(asr, M)                  # no detector: ASR alone would flag it
    assert asr_hits and not asr_hits[0].name_use
    assert build_spans(sub_hits, asr_hits, CFG) == []


def test_asr_classifies_names_from_whisper_capitalization():
    d = detector()
    words = [Word(0, 0.3, " Hey"), Word(0.3, 0.6, " Dick,"), Word(0.6, 0.9, " you're"), Word(0.9, 1.1, " a"),
             Word(1.1, 1.4, " dick."), Word(1.4, 1.7, " Dick"), Word(1.7, 2.0, " left.")]
    hits = scan_words(words, M, d)
    assert [(h.word, h.name_use) for h in hits] == [("dick", True), ("dick", False), ("dick", True)]


def test_clean_subtitles_keep_the_name():
    d = detector()
    assert censor_text("Hey Dick, don't be a dick.", M, detector=d) == ("Hey Dick, don't be a ****.", 1)
