import pysubs2

from audio_censor.config import DEFAULTS, deep_merge
from audio_censor.subtitles import censor_subs, censor_text, load_subs, _language_tag
from audio_censor.wordlist import Matcher
from pathlib import Path

M = Matcher.from_config(deep_merge(DEFAULTS, {"level": "mild"}))


def test_asterisks_keep_length_and_punctuation():
    assert censor_text("Oh, shit. That is a damn shame.", M) == ("Oh, ****. That is a **** shame.", 2)


def test_first_letter_keeps_case():
    assert censor_text("SHIT happens", M, "first-letter") == ("S*** happens", 1)


def test_bleep_replaces_whole_phrase_once():
    text, n = censor_text("You son of a bitch!", M, "bleep", "[BLEEP]")
    assert text == "You [BLEEP]!" and n == 1


def test_remove_collapses_spaces():
    assert censor_text("That is a damn shame", M, "remove") == ("That is a shame", 1)
    assert censor_text("Damn, hello", M, "remove") == (", hello", 1)


def test_tags_and_line_breaks_untouched():
    text, n = censor_text(r"{\i1}Oh shit{\i0}\NDamn it, Nancy", M)
    assert text == r"{\i1}Oh ****{\i0}\N**** it, Nancy" and n == 2


def test_censor_subs_preserves_timing_and_formatting(tmp_path):
    srt = "1\n00:00:01,000 --> 00:00:02,000\n<i>Oh shit</i>\nhello\n\n2\n00:00:03,000 --> 00:00:04,000\nfine\n"
    path = tmp_path / "x.srt"
    path.write_text(srt)
    subs = load_subs(path)
    clean, n = censor_subs(subs, M, "asterisks", "")
    assert n == 1
    assert clean.events[0].start == 1000 and clean.events[1].text == "fine"
    out = tmp_path / "x.clean.srt"
    clean.save(str(out))
    saved = out.read_text()
    assert "<i>Oh ****</i>" in saved and "hello" in saved
    assert subs.events[0].text.startswith("{\\i1}Oh shit")   # original untouched


def test_cp1252_subtitle_loads(tmp_path):
    path = tmp_path / "latin.srt"
    path.write_bytes("1\n00:00:01,000 --> 00:00:02,000\ncaf\xe9 shit\n".encode("cp1252"))
    subs = load_subs(path)
    assert "café" in subs.events[0].text


def test_language_tag_from_filename():
    media = Path("/x/Movie.mkv")
    assert _language_tag(media, Path("/x/Movie.en.srt")) == "en"
    assert _language_tag(media, Path("/x/Movie.en.sdh.srt")) == "en"
    assert _language_tag(media, Path("/x/Movie.srt")) == ""
