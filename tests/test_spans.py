from audio_censor.config import DEFAULTS, deep_merge
from audio_censor.spans import Hit, Span, build_spans, merge_spans, load_sidecar, save_sidecar
from audio_censor.subtitles import Cue, scan_cues
from audio_censor.wordlist import Matcher

CFG = deep_merge(DEFAULTS, {"level": "mild"})
M = Matcher.from_config(CFG)


def test_subtitle_estimate_places_word_within_cue():
    cues = [Cue(10.0, 12.0, "Hello there you damn fool")]
    (hit,) = scan_cues(cues, M)
    assert hit.word == "damn"
    assert 10.0 < hit.start < hit.end < 12.0
    assert hit.start > 11.0                     # "damn" is in the second half of the line
    assert (hit.cue_start, hit.cue_end) == (10.0, 12.0)


def test_subtitle_cue_mode_uses_whole_cue():
    cues = [Cue(10.0, 12.0, "damn")]
    (hit,) = scan_cues(cues, M, mode="cue")
    assert (hit.start, hit.end) == (10.0, 12.0)


def test_sound_cues_and_speaker_labels_ignored():
    cues = [Cue(0, 2, "[HELL BREAKS LOOSE] JOHN: fine"), Cue(3, 4, "(damn) ok")]
    assert scan_cues(cues, M) == []


def test_asr_confirms_subtitle_hit_and_wins_timing():
    sub = [Hit(10.8, 11.2, "damn", "damn*", "mild", "subtitle", 0.6, cue_start=10.0, cue_end=12.0)]
    asr = [Hit(11.0, 11.3, "damn", "damn*", "mild", "asr", 0.9)]
    (span,) = build_spans(sub, asr, CFG)
    assert span.sources == ["asr", "subtitle"]
    assert not span.estimated
    assert abs(span.start - (11.0 - 0.10)) < 1e-6 and abs(span.end - (11.3 + 0.10)) < 1e-6


def test_unconfirmed_subtitle_hit_kept_with_padding():
    sub = [Hit(10.8, 11.2, "damn", "damn*", "mild", "subtitle", 0.6, cue_start=10.0, cue_end=12.0)]
    (span,) = build_spans(sub, [], CFG)
    assert span.estimated and span.sources == ["subtitle"]
    assert abs(span.start - (10.8 - 0.35)) < 1e-6


def test_asr_only_hit_dropped_when_disabled():
    cfg = deep_merge(CFG, {"scan": {"asr_only": False}})
    sub = [Hit(1.0, 1.2, "damn", "damn*", "mild", "subtitle", 0.6, cue_start=0.5, cue_end=2.0)]
    asr = [Hit(50.0, 50.3, "shit", "shit*", "moderate", "asr", 0.9)]
    spans = build_spans(sub, asr, cfg)
    assert [s.words for s in spans] == [["damn"]]
    spans = build_spans(sub, asr, CFG)
    assert [s.words for s in spans] == [["damn"], ["shit"]]


def test_merge_close_spans_keeps_worst_tier():
    spans = [Span(1.0, 1.5, ["damn"], ["asr"], "mild", 0.9),
             Span(1.6, 2.0, ["shit"], ["subtitle"], "moderate", 0.6, estimated=True),
             Span(5.0, 5.5, ["hell"], ["asr"], "mild", 0.8)]
    merged = merge_spans(spans, gap=0.25)
    assert len(merged) == 2
    assert merged[0].words == ["damn", "shit"] and merged[0].tier == "moderate"
    assert merged[0].sources == ["asr", "subtitle"] and not merged[0].estimated
    assert merged[0].confidence == 0.6


def test_sidecar_roundtrip(tmp_path):
    media = tmp_path / "film.mkv"
    spans = [Span(1.0, 1.5, ["damn"], ["asr"], "mild", 0.9)]
    side = save_sidecar(tmp_path / "film.censor.json", media, spans, {"audio_track": 0})
    loaded, meta = load_sidecar(side)
    assert loaded == spans and meta["audio_track"] == 0 and meta["source_file"] == "film.mkv"
