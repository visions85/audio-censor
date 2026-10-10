import pytest

from audio_censor.beep import BeepError, BeepSpec, build_filtergraph, gate_filters, pan_filter, parse_volume
from audio_censor.config import DEFAULTS, deep_merge
from audio_censor.spans import Span

SPANS = [Span(1.0, 1.5, ["a"]), Span(10.25, 11.0, ["b"])]


def spec(**beep):
    return BeepSpec.from_config(deep_merge(DEFAULTS, {"beep": beep}))


def test_parse_volume():
    assert parse_volume(0.5) == 0.5
    assert abs(parse_volume("-6dB") - 0.501) < 0.001
    with pytest.raises(BeepError):
        parse_volume("9")


def test_pan_center_and_all():
    assert pan_filter("5.1(side)", 6, "center") == "pan=5.1(side)|FC=c0"
    assert pan_filter("stereo", 2, "center") == "pan=stereo|FL=c0|FR=c0"
    assert pan_filter("5.1", 6, "all") == "pan=5.1|FL=c0|FR=c0|FC=c0|BL=c0|BR=c0"
    assert pan_filter("mono", 1, "all") == "anull"
    assert pan_filter("", 6, "center") == "pan=5.1(side)|FC=c0"      # layout inferred from count
    assert pan_filter("weird", 9, "center") == "pan=9c|c2=c0"


def test_gate_chunks():
    g = gate_filters([Span(i, i + 0.5) for i in range(90)], 0.0, chunk=40)
    assert g.count("volume=volume=0:enable=") == 3
    assert "between(t,0,0.5)" in g


def test_default_mode_is_duck():
    d = spec()
    assert d.mode == "duck" and d.duck == 0.1
    assert spec(mode="mute").duck == 0.0
    assert spec(mode="duck", duck=0).mode == "mute"
    g = build_filtergraph(SPANS, d, audio_label="0:a:0", layout="stereo", channels=2, rate=48000)
    assert "volume=volume=0.1:enable=" in g and "aevalsrc" not in g


def test_beep_graph_shape():
    g = build_filtergraph(SPANS, spec(mode="beep", duck=0), audio_label="0:a:0", layout="5.1(side)", channels=6, rate=48000)
    assert g.startswith("[0:a:0]asetnsamples=n=240:p=0,volume=volume=0:enable=")
    assert "aevalsrc=exprs='0.4*sin(2*PI*1000*t)':s=48000:c=mono:d=0.5" in g
    assert "adelay=48000S:all=1[b0]" in g and "adelay=492000S:all=1[b1]" in g
    assert "[b0][b1]amix=inputs=2:duration=longest:normalize=0,pan=5.1(side)|FC=c0[beeps]" in g
    assert g.endswith("[dlg][beeps]amix=inputs=2:duration=first:normalize=0[clean]")


def test_mute_graph_has_no_beeps():
    g = build_filtergraph(SPANS, spec(mode="duck", duck=0.2), audio_label="0:a:1", layout="stereo", channels=2, rate=44100)
    assert "aevalsrc" not in g and "amix" not in g
    assert "volume=volume=0.2:enable=" in g and g.endswith("[clean]")
    g = build_filtergraph(SPANS, spec(mode="mute", duck=0.2), audio_label="0:a:1", layout="stereo", channels=2, rate=44100)
    assert "volume=volume=0:enable=" in g


def test_file_beep_graph(tmp_path):
    snd = tmp_path / "quack.wav"
    snd.write_bytes(b"RIFF")
    s = spec(mode="beep", wave="file", file=str(snd))
    s.file_duration = 0.6
    g = build_filtergraph(SPANS, s, audio_label="0:a:0", layout="stereo", channels=2, rate=48000, file_input=1)
    assert "[1:a]asplit=2[src0][src1]" in g
    # 0.6 s file is shorter than the 0.75 s span: looped (0.6*48000 + 4800 samples buffered)
    assert "[src1]aformat=channel_layouts=mono,aresample=48000,aloop=loop=-1:size=33600,atrim=duration=0.75,volume=0.4" in g
    # ... but longer than the 0.5 s span: no loop
    assert "[src0]aformat=channel_layouts=mono,aresample=48000,atrim=duration=0.5,volume=0.4" in g


def test_no_spans_passthrough():
    assert build_filtergraph([], spec(mode="beep"), audio_label="0:a:0", layout="stereo", channels=2, rate=48000) == "[0:a:0]anull[clean]"


def test_config_validation():
    with pytest.raises(BeepError):
        spec(mode="beep", wave="file", file="")
    with pytest.raises(BeepError):
        spec(wave="klaxon")
    with pytest.raises(BeepError):
        spec(duck=2)
