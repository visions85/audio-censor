"""Build and run the ffmpeg remux that adds the clean audio track."""

from __future__ import annotations

import functools
import os
import shutil
import tempfile
from pathlib import Path

from .beep import BeepSpec, build_filtergraph, preview_graph
from .media import MediaError, MediaInfo, Stream, eprint, probe, run
from .spans import Span

TEXT_SUBS_NEEDING_CONVERSION = {"mov_text", "tx3g"}


@functools.lru_cache(maxsize=None)
def filter_script_option() -> str:
    """The ffmpeg option that reads a complex filtergraph from a file.

    -filter_complex_script was deprecated in ffmpeg 7.0 and later removed; its
    replacement, the generic "-/option file" syntax, does not exist before 7.0.
    """
    proc = run(["ffmpeg", "-hide_banner", "-h", "full"], check=False)
    return "-filter_complex_script" if "-filter_complex_script" in (proc.stdout or "") else "-/filter_complex"


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


CLEAN_SUB_TITLE = "Clean"


def clean_title(spec: BeepSpec, cfg: dict) -> str:
    return cfg.get("output", {}).get("title") or {"beep": "Clean (beeped)", "mute": "Clean (muted)",
                                                   "duck": "Clean (ducked)"}[spec.mode]


def is_clean_track(stream: Stream) -> bool:
    """A track this tool added earlier (so an in-place re-render replaces it)."""
    t = (stream.title or "").strip()
    return t == CLEAN_SUB_TITLE or (t.startswith("Clean (") and t.endswith(")"))


def build_command(media: Path, info: MediaInfo, dialogue: Stream, spans: list[Span], spec: BeepSpec,
                  cfg: dict, output: Path, script_path: Path,
                  clean_subs: Path | None = None, clean_subs_lang: str = "",
                  drop_clean: bool = False) -> tuple[list[str], str]:
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
           filter_script_option(), str(script_path)]

    audio_streams = [a for a in info.of_type("audio") if not (drop_clean and is_clean_track(a))]
    kept_audio = audio_streams if keep_original else []
    kept_subs = [x for x in info.of_type("subtitle") if not (drop_clean and is_clean_track(x))]
    cmd += ["-map", "0:v?"]
    for s in kept_audio:
        cmd += ["-map", f"0:a:{s.type_index}"]
    cmd += ["-map", "[clean]"]
    for x in kept_subs:
        cmd += ["-map", f"0:s:{x.type_index}"]
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
    cmd += [f"-metadata:s:a:{new_idx}", f"title={clean_title(spec, cfg)}"]
    if dialogue.language:
        cmd += [f"-metadata:s:a:{new_idx}", f"language={dialogue.language}"]
    if set_default:
        for i in range(new_idx):
            cmd += [f"-disposition:a:{i}", "0"]
        cmd += [f"-disposition:a:{new_idx}", "default"]
    else:
        cmd += [f"-disposition:a:{new_idx}", "0"]

    cmd += _subtitle_args(kept_subs, output, cfg, subs_input is not None, clean_subs_lang)
    cmd += ["-max_muxing_queue_size", "4096", str(output)]
    return cmd, graph


def _subtitle_args(kept_subs: list[Stream], output: Path, cfg: dict, has_clean: bool, clean_lang: str) -> list[str]:
    """Codec / metadata options for the copied subtitle streams (in output order) and the appended clean one."""
    container = output.suffix.lower()
    is_mp4 = container in (".mp4", ".m4v", ".mov")
    args: list[str] = []
    for i, s in enumerate(kept_subs):
        if container in (".mkv", ".mka", ".webm") and s.codec_name in TEXT_SUBS_NEEDING_CONVERSION:
            args += [f"-c:s:{i}", "srt"]
        elif is_mp4 and s.codec_name in ("subrip", "ass", "ssa", "webvtt"):
            args += [f"-c:s:{i}", "mov_text"]
        else:
            args += [f"-c:s:{i}", "copy"]
    if has_clean:
        sub_cfg = cfg.get("subtitles", {})
        idx = len(kept_subs)
        args += [f"-c:s:{idx}", "mov_text" if is_mp4 else "copy", f"-metadata:s:s:{idx}", f"title={CLEAN_SUB_TITLE}"]
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
                         dry_run: bool = False, verbose: bool = False, drop_clean: bool = False) -> Path:
    """Copy every stream and append the clean subtitle track (used when audio is not censored)."""
    if output.resolve() == media.resolve():
        raise MediaError("output path equals the input; refusing to overwrite the source")
    output.parent.mkdir(parents=True, exist_ok=True)
    kept_subs = [x for x in info.of_type("subtitle") if not (drop_clean and is_clean_track(x))]
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-stats", "-i", str(media), "-i", str(clean_subs),
           "-map", "0:v?", "-map", "0:a?"]
    for x in kept_subs:
        cmd += ["-map", f"0:s:{x.type_index}"]
    cmd += ["-map", "1:s:0", "-map", "0:t?", "-c:v", "copy", "-c:a", "copy", "-c:t", "copy"]
    cmd += _subtitle_args(kept_subs, output, cfg, True, lang)
    cmd += ["-max_muxing_queue_size", "4096", str(output)]
    if verbose or dry_run:
        eprint("command:\n  " + " ".join(_quote(c) for c in cmd))
    if not dry_run:
        eprint(f"  remuxing with clean subtitles -> {output.name}")
        run(cmd, quiet=False)
    return output


def temp_output_for(media: Path) -> Path:
    """Where an in-place render is written before it replaces the original (same directory, same disk)."""
    return media.with_name(f"{media.stem}.clean.tmp.mkv")


def check_disk_space(media: Path) -> None:
    free = shutil.disk_usage(media.parent).free
    need = media.stat().st_size * 1.05 + 64 * 1024 * 1024
    if free < need:
        raise MediaError(f"not enough free space beside {media.name} for an in-place rewrite "
                         f"({free / 1e9:.1f} GB free, {need / 1e9:.1f} GB needed)")


def finalize_in_place(media: Path, tmp: Path, expect_audio: int, expect_subs: int, backup: bool = False) -> Path:
    """Verify the rewritten file, then swap it over the original. Returns the final path (.mkv)."""
    try:
        orig = probe(media)
        new = probe(tmp)
        problems = []
        if abs(new.duration - orig.duration) > 2.0:
            problems.append(f"duration {new.duration:.1f}s vs {orig.duration:.1f}s")
        if len(new.of_type("video")) != len(orig.of_type("video")):
            problems.append("video stream count differs")
        if len(new.of_type("audio")) != expect_audio:
            problems.append(f"expected {expect_audio} audio streams, got {len(new.of_type('audio'))}")
        if len(new.of_type("subtitle")) < expect_subs:
            problems.append(f"expected at least {expect_subs} subtitle streams, got {len(new.of_type('subtitle'))}")
        if tmp.stat().st_size < media.stat().st_size * 0.5:
            problems.append("output is less than half the size of the original")
        if problems:
            raise MediaError("in-place verification failed, original left untouched: " + "; ".join(problems))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    try:
        shutil.copystat(media, tmp)
        st = media.stat()
        os.chown(tmp, st.st_uid, st.st_gid)
    except OSError:
        pass
    final = media.with_suffix(".mkv")
    if backup:
        media.replace(media.with_name(f"{media.stem}.orig{media.suffix}"))
    os.replace(tmp, final)
    if final != media and media.exists():
        media.unlink()
    return final


def render(media: Path, info: MediaInfo, dialogue: Stream, spans: list[Span], spec: BeepSpec, cfg: dict,
           output: Path, dry_run: bool = False, verbose: bool = False,
           clean_subs: Path | None = None, clean_subs_lang: str = "", drop_clean: bool = False) -> Path:
    if output.resolve() == media.resolve():
        raise MediaError("output path equals the input; refusing to overwrite the source")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="audio-censor-") as tmp:
        script = Path(tmp) / "filter.txt"
        cmd, graph = build_command(media, info, dialogue, spans, spec, cfg, output, script,
                                   clean_subs=clean_subs, clean_subs_lang=clean_subs_lang, drop_clean=drop_clean)
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
