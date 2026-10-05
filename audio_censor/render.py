"""Build and run the ffmpeg remux that adds the clean audio track."""

from __future__ import annotations

import tempfile
from pathlib import Path

from .beep import BeepSpec, build_filtergraph, preview_graph
from .media import MediaError, MediaInfo, Stream, eprint, probe, run
from .spans import Span

TEXT_SUBS_NEEDING_CONVERSION = {"mov_text", "tx3g"}


def choose_codec(stream: Stream, cfg: dict) -> tuple[str, str | None]:
    out = cfg.get("output", {})
    codec = (out.get("codec") or "auto").lower()
    bitrate = out.get("bitrate") or "auto"
    ch = stream.channels or 2
    if codec == "auto":
        if ch <= 2:
            codec = "aac"
        elif ch <= 6:
            codec = "ac3"
        else:
            codec = "aac"
    if bitrate == "auto":
        if codec in ("flac", "pcm_s16le", "pcm_s24le"):
            bitrate = None
        elif codec == "aac":
            bitrate = {1: "96k", 2: "192k"}.get(ch, f"{min(96 * ch, 768)}k")
        elif codec == "ac3":
            bitrate = "640k" if ch > 2 else "256k"
        elif codec == "eac3":
            bitrate = "768k" if ch > 2 else "256k"
        elif codec in ("libopus", "opus"):
            bitrate = f"{min(64 * ch, 510)}k"
            codec = "libopus"
        else:
            bitrate = None
    return codec, (None if bitrate in (None, "", "none") else str(bitrate))


def default_output(media: Path, cfg: dict, output_dir: Path | None = None) -> Path:
    out = cfg.get("output", {})
    suffix = out.get("suffix", ".clean")
    container = (out.get("container") or "mkv").lstrip(".")
    name = f"{media.stem}{suffix}.{container}"
    return (output_dir or media.parent) / name


def build_command(media: Path, info: MediaInfo, dialogue: Stream, spans: list[Span], spec: BeepSpec,
                  cfg: dict, output: Path, script_path: Path,
                  clean_subs: Path | None = None, clean_subs_lang: str = "") -> tuple[list[str], str]:
    out_cfg = cfg.get("output", {})
    keep_original = bool(out_cfg.get("keep_original", True))
    set_default = bool(out_cfg.get("set_default", True))

    inputs = ["-i", str(media)]
    file_input = None
    if spec.mode == "beep" and spec.wave == "file":
        file_input = 1
        spec.file_duration = probe(spec.file).duration
        inputs += ["-i", str(spec.file)]

    subs_input = None
    if clean_subs is not None:
        subs_input = len(inputs) // 2
        inputs += ["-i", str(clean_subs)]

    rate = dialogue.sample_rate or 48000
    graph = build_filtergraph(spans, spec, audio_label=f"0:a:{dialogue.type_index}",
                              layout=dialogue.channel_layout, channels=dialogue.channels or 2,
                              rate=rate, file_input=file_input)
    script_path.write_text(graph + "\n", encoding="utf-8")

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-stats", *inputs,
           "-filter_complex_script", str(script_path)]

    audio_streams = info.of_type("audio")
    kept_audio = audio_streams if keep_original else []
    cmd += ["-map", "0:v?"]
    for s in kept_audio:
        cmd += ["-map", f"0:a:{s.type_index}"]
    cmd += ["-map", "[clean]", "-map", "0:s?"]
    if subs_input is not None:
        cmd += ["-map", f"{subs_input}:s:0"]
    cmd += ["-map", "0:t?"]
    # Per-stream codecs (rather than a blanket "-c copy") so ffmpeg does not warn
    # about the clean track having two codec options.
    cmd += ["-c:v", "copy", "-c:t", "copy"]
    new_idx = len(kept_audio)
    for i in range(new_idx):
        cmd += [f"-c:a:{i}", "copy"]
    codec, bitrate = choose_codec(dialogue, cfg)
    cmd += [f"-c:a:{new_idx}", codec]
    if bitrate:
        cmd += [f"-b:a:{new_idx}", bitrate]
    if codec == "aac" and (dialogue.channels or 2) > 2:
        cmd += [f"-ac:a:{new_idx}", str(dialogue.channels)]
    title = out_cfg.get("title") or "Clean (beeped)"
    if spec.mode == "mute" and title == "Clean (beeped)":
        title = "Clean (muted)"
    cmd += [f"-metadata:s:a:{new_idx}", f"title={title}"]
    if dialogue.language:
        cmd += [f"-metadata:s:a:{new_idx}", f"language={dialogue.language}"]
    if set_default:
        for i in range(new_idx):
            cmd += [f"-disposition:a:{i}", "0"]
        cmd += [f"-disposition:a:{new_idx}", "default"]
    else:
        cmd += [f"-disposition:a:{new_idx}", "0"]

    cmd += _subtitle_args(info, output, cfg, subs_input is not None, clean_subs_lang)
    cmd += ["-max_muxing_queue_size", "4096", str(output)]
    return cmd, graph


def _subtitle_args(info: MediaInfo, output: Path, cfg: dict, has_clean: bool, clean_lang: str) -> list[str]:
    """Codec / metadata options for the copied subtitle streams and the appended clean one."""
    container = output.suffix.lower()
    is_mp4 = container in (".mp4", ".m4v", ".mov")
    args: list[str] = []
    for s in info.of_type("subtitle"):
        if container in (".mkv", ".mka", ".webm") and s.codec_name in TEXT_SUBS_NEEDING_CONVERSION:
            args += [f"-c:s:{s.type_index}", "srt"]
        elif is_mp4 and s.codec_name in ("subrip", "ass", "ssa", "webvtt"):
            args += [f"-c:s:{s.type_index}", "mov_text"]
        else:
            args += [f"-c:s:{s.type_index}", "copy"]
    if has_clean:
        sub_cfg = cfg.get("subtitles", {})
        idx = len(info.of_type("subtitle"))
        args += [f"-c:s:{idx}", "mov_text" if is_mp4 else "copy", f"-metadata:s:s:{idx}", "title=Clean"]
        if clean_lang:
            args += [f"-metadata:s:s:{idx}", f"language={clean_lang}"]
        if sub_cfg.get("set_default", False):
            for i in range(idx):
                args += [f"-disposition:s:{i}", "0"]
            args += [f"-disposition:s:{idx}", "default"]
        else:
            args += [f"-disposition:s:{idx}", "0"]
    return args


def remux_with_subtitles(media: Path, info: MediaInfo, clean_subs: Path, lang: str, cfg: dict, output: Path,
                         dry_run: bool = False, verbose: bool = False) -> Path:
    """Copy every stream and append the clean subtitle track (used when audio is not censored)."""
    if output.resolve() == media.resolve():
        raise MediaError("output path equals the input; refusing to overwrite the source")
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-stats", "-i", str(media), "-i", str(clean_subs),
           "-map", "0:v?", "-map", "0:a?", "-map", "0:s?", "-map", "1:s:0", "-map", "0:t?",
           "-c:v", "copy", "-c:a", "copy", "-c:t", "copy"]
    cmd += _subtitle_args(info, output, cfg, True, lang)
    cmd += ["-max_muxing_queue_size", "4096", str(output)]
    if verbose or dry_run:
        eprint("command:\n  " + " ".join(_quote(c) for c in cmd))
    if not dry_run:
        eprint(f"  remuxing with clean subtitles -> {output.name}")
        run(cmd, quiet=False)
    return output


def render(media: Path, info: MediaInfo, dialogue: Stream, spans: list[Span], spec: BeepSpec, cfg: dict,
           output: Path, dry_run: bool = False, verbose: bool = False,
           clean_subs: Path | None = None, clean_subs_lang: str = "") -> Path:
    if output.resolve() == media.resolve():
        raise MediaError("output path equals the input; refusing to overwrite the source")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="audio-censor-") as tmp:
        script = Path(tmp) / "filter.txt"
        cmd, graph = build_command(media, info, dialogue, spans, spec, cfg, output, script,
                                   clean_subs=clean_subs, clean_subs_lang=clean_subs_lang)
        if verbose or dry_run:
            eprint("filter graph:\n" + graph)
            eprint("command:\n  " + " ".join(_quote(c) for c in cmd))
        if dry_run:
            return output
        eprint(f"  encoding clean track -> {output.name}")
        run(cmd, quiet=False)
    return output


def render_preview(spec: BeepSpec, output: Path, duration: float = 1.0, rate: int = 48000) -> Path:
    inputs = []
    if spec.wave == "file":
        spec.file_duration = probe(spec.file).duration
        inputs = ["-i", str(spec.file)]
    graph = preview_graph(spec, duration, rate)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs,
           "-filter_complex", graph, "-map", "[out]", str(output)]
    run(cmd)
    return output


def _quote(s: str) -> str:
    return s if all(c.isalnum() or c in "-_./:=,[]" for c in s) else "'" + s.replace("'", "'\\''") + "'"
