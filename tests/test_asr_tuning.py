from types import SimpleNamespace

from audio_censor import asr
from audio_censor.config import load_config
from audio_censor.media import Stream, has_center
from audio_censor.spans import Hit
from audio_censor.wordlist import Matcher


def _audio(channels, layout):
    return Stream(index=1, type_index=0, codec_type="audio", codec_name="dts", language="eng", title="", default=True, channels=channels,
                  channel_layout=layout)


def test_has_center():
    assert has_center(_audio(6, "5.1(side)")) and has_center(_audio(8, "7.1")) and has_center(_audio(6, ""))
    assert not has_center(_audio(2, "stereo")) and not has_center(_audio(4, "quad")) and not has_center(None)


def test_transcript_from_a_downmix_is_stale_for_centre_listening(tmp_path):
    path = tmp_path / "f.transcript.json"
    asr.save_transcript(path, [asr.Word(1, 2, " hi")], "large-v3", 0)          # old: no channel = mix
    assert asr.load_transcript(path, "large-v3", "mix") is not None
    assert asr.load_transcript(path, "large-v3", "center") is None
    asr.save_transcript(path, [asr.Word(1, 2, " hi")], "large-v3", 0, "center")
    assert asr.load_transcript(path, "large-v3", "center")[0].text == " hi"


def test_recheck_finds_a_word_the_transcript_missed(tmp_path, monkeypatch):
    cfg = load_config()
    matcher = Matcher.from_config(cfg)
    clips = []

    def fake_clip(media, track, out, start, length, center=False):
        clips.append((start, length, center))
        return out

    word = SimpleNamespace(start=1.6, end=2.0, word=" shit", probability=0.9)
    model = SimpleNamespace(transcribe=lambda *a, **k: ([SimpleNamespace(words=[
        SimpleNamespace(start=1.0, end=1.5, word=" Holy", probability=0.9), word])], None))
    monkeypatch.setattr("audio_censor.media.extract_audio_clip", fake_clip)
    monkeypatch.setattr(asr, "_get_model", lambda cfg: model)
    monkeypatch.setattr(asr, "load_wav", lambda p: None)
    pattern = matcher.find(__import__("audio_censor.wordlist", fromlist=["tokenize"]).tokenize("shit"))[0].pattern
    missed = Hit(10.5, 10.9, "shit", pattern.raw, pattern.tier, "subtitle", 0.6, cue_start=10.0, cue_end=12.0)
    heard = Hit(30.1, 30.4, "shit", pattern.raw, pattern.tier, "subtitle", 0.6, cue_start=30.0, cue_end=31.0)
    asr_hit = Hit(30.2, 30.5, "shit", pattern.raw, pattern.tier, "asr", 0.9)
    found, checked = asr.recheck_subtitle_hits(tmp_path / "f.mkv", 0, [missed, heard], [asr_hit], matcher, cfg,
                                               tmp_path, center=True)
    assert checked == 1 and clips == [(8.5, 5.0, True)]
    assert [(h.word, h.source, round(h.start, 1)) for h in found] == [("shit", "asr", 10.1)]
