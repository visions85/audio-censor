"""Beep sound definition and ffmpeg filtergraph construction."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from .spans import Span

WAVES = {
    "sine":     "{A}*sin(2*PI*{f}*t)",
    "square":   "{A}*(2*gt(sin(2*PI*{f}*t),0)-1)",
    "triangle": "{A}*(2*abs(2*({f}*t-floor({f}*t+0.5)))-1)",
    "sawtooth": "{A}*(2*({f}*t-floor({f}*t))-1)",
    "noise":    "{A}*(2*random(0)-1)",
}

# Channel names ffmpeg uses for common layouts (for the pan filter).
LAYOUT_CHANNELS = {
    "mono": ["FC"],
    "stereo": ["FL", "FR"],
    "2.1": ["FL", "FR", "LFE"],
    "3.0": ["FL", "FR", "FC"],
    "3.0(back)": ["FL", "FR", "BC"],
    "4.0": ["FL", "FR", "FC", "BC"],
    "quad": ["FL", "FR", "BL", "BR"],
    "quad(side)": ["FL", "FR", "SL", "SR"],
    "3.1": ["FL", "FR", "FC", "LFE"],
    "5.0": ["FL", "FR", "FC", "BL", "BR"],
    "5.0(side)": ["FL", "FR", "FC", "SL", "SR"],
    "4.1": ["FL", "FR", "FC", "LFE", "BC"],
    "5.1": ["FL", "FR", "FC", "LFE", "BL", "BR"],
    "5.1(side)": ["FL", "FR", "FC", "LFE", "SL", "SR"],
    "6.0": ["FL", "FR", "FC", "BC", "SL", "SR"],
    "6.1": ["FL", "FR", "FC", "LFE", "BC", "SL", "SR"],
    "7.0": ["FL", "FR", "FC", "BL", "BR", "SL", "SR"],
    "7.1": ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"],
    "7.1(wide)": ["FL", "FR", "FC", "LFE", "BL", "BR", "FLC", "FRC"],
    "7.1(wide-side)": ["FL", "FR", "FC", "LFE", "FLC", "FRC", "SL", "SR"],
}
_DEFAULT_LAYOUT_FOR_CHANNELS = {1: "mono", 2: "stereo", 3: "2.1", 4: "quad", 5: "5.0", 6: "5.1(side)", 7: "6.1", 8: "7.1"}


class BeepError(ValueError):
    pass


def parse_volume(value) -> float:
    """Accept 0.4, "0.4", "-8dB" -> linear gain."""
    if isinstance(value, (int, float)):
        gain = float(value)
    else:
        s = str(value).strip().lower()
        if s.endswith("db"):
            gain = 10 ** (float(s[:-2]) / 20.0)
        else:
            gain = float(s)
    if gain < 0 or gain > 4:
        raise BeepError(f"beep volume {value!r} out of range (0..1 linear, or dB like '-6dB')")
    return gain


@dataclass
class BeepSpec:
    mode: str = "beep"
    wave: str = "sine"
    frequency: float = 1000.0
    volume: float = 0.4
    file: Path | None = None
    loop_file: bool = True
    fade: float = 0.01
    channel: str = "center"
    duck: float = 0.0
    file_duration: float = 0.0   # filled in by the renderer (seconds), for aloop sizing

    @classmethod
    def from_config(cls, cfg: dict) -> "BeepSpec":
        b = cfg.get("beep", {})
        mode = b.get("mode", "beep")
        if mode not in ("beep", "mute"):
            raise BeepError(f"beep.mode must be 'beep' or 'mute', got {mode!r}")
        wave = b.get("wave", "sine")
        if wave not in WAVES and wave != "file":
            raise BeepError(f"beep.wave must be one of {', '.join(WAVES)}, file; got {wave!r}")
        file = None
        if wave == "file":
            raw = str(b.get("file") or "").strip()
            if not raw:
                raise BeepError("beep.wave is 'file' but beep.file is empty")
            file = Path(os.path.expanduser(raw))
            if not file.exists():
                raise BeepError(f"beep.file not found: {file}")
        channel = b.get("channel", "center")
        if channel not in ("center", "all"):
            raise BeepError("beep.channel must be 'center' or 'all'")
        duck = float(b.get("duck", 0.0))
        if not 0 <= duck <= 1:
            raise BeepError("beep.duck must be between 0 and 1")
        return cls(mode=mode, wave=wave, frequency=float(b.get("frequency", 1000)),
                   volume=parse_volume(b.get("volume", 0.4)), file=file,
                   loop_file=bool(b.get("loop_file", True)), fade=float(b.get("fade", 0.01)),
                   channel=channel, duck=duck)


def _f(x: float) -> str:
    return f"{x:.4f}".rstrip("0").rstrip(".") if x != int(x) else str(int(x))


def gate_filters(spans: list[Span], level: float, chunk: int = 40) -> str:
    """volume filters that silence (or duck) the dialogue inside every span."""
    if not spans:
        return "anull"
    parts = []
    for i in range(0, len(spans), chunk):
        expr = "+".join(f"between(t,{_f(s.start)},{_f(s.end)})" for s in spans[i:i + chunk])
        parts.append(f"volume=volume={_f(level)}:enable='{expr}'")
    return ",".join(parts)


def pan_filter(layout: str, channels: int, where: str) -> str:
    """Place a mono signal into the dialogue track's layout."""
    names = LAYOUT_CHANNELS.get(layout)
    if names is None:
        layout = _DEFAULT_LAYOUT_FOR_CHANNELS.get(channels, "")
        names = LAYOUT_CHANNELS.get(layout)
    if names is None:
        # Unknown layout: address channels by number, skipping index 3 (usually LFE).
        idx = [i for i in range(channels) if not (channels >= 6 and i == 3)] if where == "all" else [min(2, channels - 1)]
        return f"pan={channels}c|" + "|".join(f"c{i}=c0" for i in idx)
    if len(names) == 1:
        return "anull"
    if where == "center" and "FC" in names:
        targets = ["FC"]
    elif where == "center":
        targets = [n for n in names if n in ("FL", "FR")] or names[:1]
    else:
        targets = [n for n in names if n != "LFE"]
    return f"pan={layout}|" + "|".join(f"{n}=c0" for n in targets)


def beep_source(spec: BeepSpec, duration: float, rate: int, input_label: str | None = None) -> str:
    """Filter chain producing one mono beep of `duration` seconds (before delay/pan)."""
    fade = min(spec.fade, duration / 2)
    chain = []
    if spec.wave == "file":
        if input_label is None:
            raise BeepError("file beep needs an input label")
        chain.append(f"[{input_label}]aformat=channel_layouts=mono,aresample={rate}")
        if spec.loop_file and spec.file_duration > 0 and spec.file_duration < duration:
            size = int(math.ceil(spec.file_duration * rate)) + rate // 10
            chain.append(f"aloop=loop=-1:size={size}")
        chain.append(f"atrim=duration={_f(duration)}")
        chain.append(f"volume={_f(spec.volume)}")
    else:
        expr = WAVES[spec.wave].format(A=_f(spec.volume), f=_f(spec.frequency))
        chain.append(f"aevalsrc=exprs='{expr}':s={rate}:c=mono:d={_f(duration)}")
    if fade > 0:
        chain.append(f"afade=t=in:st=0:d={_f(fade)}")
        chain.append(f"afade=t=out:st={_f(max(0.0, duration - fade))}:d={_f(fade)}")
    return ",".join(chain)


def build_filtergraph(spans: list[Span], spec: BeepSpec, *, audio_label: str, layout: str,
                      channels: int, rate: int, file_input: int | None = None,
                      out_label: str = "clean") -> str:
    """Full filter_complex script: gated dialogue mixed with positioned beeps -> [out_label]."""
    if not spans:
        return f"[{audio_label}]anull[{out_label}]"
    lines = []
    gate = gate_filters(spans, spec.duck)
    # asetnsamples gives the timeline gate ~5 ms resolution instead of one decoder frame.
    if spec.mode == "mute":
        lines.append(f"[{audio_label}]asetnsamples=n=240:p=0,{gate}[{out_label}]")
        return ";\n".join(lines)

    lines.append(f"[{audio_label}]asetnsamples=n=240:p=0,{gate}[dlg]")
    n = len(spans)
    if spec.wave == "file":
        if file_input is None:
            raise BeepError("file beep needs the index of the sound file input")
        split = "".join(f"[src{i}]" for i in range(n))
        lines.append(f"[{file_input}:a]asplit={n}{split}")
    for i, s in enumerate(spans):
        src = beep_source(spec, s.duration, rate, input_label=f"src{i}" if spec.wave == "file" else None)
        delay = int(round(s.start * rate))
        lines.append(f"{src},adelay={delay}S:all=1[b{i}]")
    inputs = "".join(f"[b{i}]" for i in range(n))
    pan = pan_filter(layout, channels, spec.channel)
    lines.append(f"{inputs}amix=inputs={n}:duration=longest:normalize=0,{pan}[beeps]")
    lines.append(f"[dlg][beeps]amix=inputs=2:duration=first:normalize=0[{out_label}]")
    return ";\n".join(lines)


def preview_graph(spec: BeepSpec, duration: float, rate: int = 48000) -> str:
    src = beep_source(spec, duration, rate, input_label="0:a" if spec.wave == "file" else None)
    return f"{src}[out]"
