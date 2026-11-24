import argparse
from pathlib import Path

from audio_censor.cli import decide_audio_language
from audio_censor.config import DEFAULTS, deep_merge
from audio_censor.media import Stream


def stream(lang):
    return Stream(index=1, type_index=0, codec_type="audio", codec_name="aac", language=lang, title="", default=True)


def args(**kw):
    return argparse.Namespace(**kw)


def decide(lang, cfg_over=None, **kw):
    cfg = deep_merge(DEFAULTS, cfg_over or {})
    cfg["scan"]["use_asr"] = False          # keep Whisper out of unit tests
    return decide_audio_language(Path("x.mkv"), None, stream(lang), cfg, args(**kw), Path("."))


def test_tagged_english_is_censored():
    assert decide("eng") == (True, "eng", "tag")
    assert decide("en") == (True, "en", "tag")


def test_tagged_foreign_is_skipped():
    ok, lang, how = decide("fre")
    assert (ok, lang, how) == (False, "fre", "tag")


def test_untagged_assumed_english_without_detection():
    assert decide("") == (True, "eng", "untagged, assumed")
    assert decide("und") == (True, "eng", "untagged, assumed")
    ok, lang, how = decide("", {"scan": {"assume_untagged": ""}})
    assert (ok, how) == (False, "untagged")


def test_overrides():
    assert decide("fre", audio_language="eng") == (True, "eng", "forced")
    assert decide("fre", any_language=True)[0] is False      # flag is applied in apply_overrides, not here
    assert decide("fre", {"scan": {"audio_only_languages": False}}) == (True, "fre", "any")
    assert decide("jpn", {"languages": ["jpn"]}) == (True, "jpn", "tag")


def test_verbose_and_config_accepted_after_subcommand():
    from audio_censor.cli import build_parser
    a = build_parser().parse_args(["scan", "x.mkv", "-v"])
    b = build_parser().parse_args(["-v", "scan", "x.mkv"])
    c = build_parser().parse_args(["scan", "x.mkv"])
    assert a.verbose is True and b.verbose is True and c.verbose is False
    assert build_parser().parse_args(["render", "x.mkv", "-c", "my.toml"]).config == "my.toml"
    assert build_parser().parse_args(["render", "x.mkv"]).config is None



def test_own_clean_track_is_never_the_dialogue_source():
    from pathlib import Path as P
    from audio_censor.media import MediaInfo, Stream, pick_audio_stream
    def a(i, title, default):
        return Stream(index=i, type_index=i, codec_type="audio", codec_name="ac3", language="eng",
                      title=title, default=default, channels=6)
    info = MediaInfo(P("x.mkv"), 1.0, [a(0, "DTS 5.1", False), a(1, "Commentary", False), a(2, "Clean (muted)", True)])
    assert pick_audio_stream(info, ["eng"]).type_index == 0
    assert pick_audio_stream(info, ["eng"], requested=2).type_index == 2     # explicit choice still wins
