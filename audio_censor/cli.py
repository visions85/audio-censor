"""audio-censor command-line interface."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from . import __version__, asr
from .beep import BeepError, BeepSpec, WAVES
from .config import default_config_path, load_config, write_template
from .media import VIDEO_EXTS, MediaError, eprint, pick_audio_stream, probe
from .render import default_output, render, render_preview
from .spans import Span, build_spans, format_table, load_sidecar, save_sidecar, sidecar_path
from .subtitles import obtain_cues, scan_cues
from .wordlist import TIERS, Matcher, active_tiers, load_default_tiers


class UserError(Exception):
    pass


# ----------------------------------------------------------------------------- helpers

def expand_inputs(paths: list[str], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            it = p.rglob("*") if recursive else p.iterdir()
            files += sorted(f for f in it if f.is_file() and f.suffix.lower() in VIDEO_EXTS
                            and ".clean." not in f.name)
        elif p.exists():
            files.append(p)
        else:
            raise UserError(f"not found: {p}")
    return files


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """CLI flags win over the config file."""
    g = lambda name: getattr(args, name, None)
    if g("level"):
        cfg["level"] = args.level
    if g("no_asr"):
        cfg["scan"]["use_asr"] = False
    if g("no_subtitles"):
        cfg["scan"]["use_subtitles"] = False
    if g("model"):
        cfg["asr"]["model"] = args.model
    if g("device"):
        cfg["asr"]["device"] = args.device
    if g("mode"):
        cfg["beep"]["mode"] = args.mode
    if g("wave"):
        cfg["beep"]["wave"] = args.wave
    if g("frequency") is not None:
        cfg["beep"]["frequency"] = args.frequency
    if g("volume") is not None:
        cfg["beep"]["volume"] = args.volume
    if g("beep_file"):
        cfg["beep"]["wave"] = "file"
        cfg["beep"]["file"] = args.beep_file
    if g("beep_channel"):
        cfg["beep"]["channel"] = args.beep_channel
    if g("duck") is not None:
        cfg["beep"]["duck"] = args.duck
    if g("codec"):
        cfg["output"]["codec"] = args.codec
    if g("replace_audio"):
        cfg["output"]["keep_original"] = False
    if g("no_default"):
        cfg["output"]["set_default"] = False
    if g("title"):
        cfg["output"]["title"] = args.title
    if cfg["level"] not in TIERS:
        raise UserError(f"level must be one of {', '.join(TIERS)}")
    return cfg


def scan_file(media: Path, cfg: dict, args: argparse.Namespace) -> tuple[list[Span], dict]:
    """Run subtitle + ASR scans and return merged spans plus metadata."""
    info = probe(media)
    dialogue = pick_audio_stream(info, cfg["languages"], getattr(args, "audio_track", None))
    matcher = Matcher.from_config(cfg)
    scan_cfg = cfg["scan"]
    eprint(f"{media.name}: {info.duration / 60:.1f} min, dialogue track {dialogue.describe()}")

    sub_hits, asr_hits = [], []
    sub_source = "disabled"
    with tempfile.TemporaryDirectory(prefix="audio-censor-") as tmp:
        work = Path(tmp)
        if scan_cfg["use_subtitles"]:
            explicit = Path(args.subtitles).expanduser() if getattr(args, "subtitles", None) else None
            cues, sub_source = obtain_cues(media, info, cfg["languages"], work, explicit)
            if cues:
                sub_hits = scan_cues(cues, matcher, scan_cfg.get("subtitle_mode", "estimate"))
                eprint(f"  subtitles: {sub_source}, {len(cues)} cues, {len(sub_hits)} hits")
            else:
                eprint(f"  subtitles: {sub_source}")

        asr_source = "disabled"
        if scan_cfg["use_asr"]:
            if not asr.available():
                eprint(f"  asr: skipped ({asr.INSTALL_HINT})")
                asr_source = "unavailable"
            else:
                words = None
                tpath = asr.transcript_path(media)
                if scan_cfg.get("cache_transcript", True) and not getattr(args, "rescan", False):
                    words = asr.load_transcript(tpath, cfg["asr"]["model"])
                    if words is not None:
                        eprint(f"  asr: using cached transcript {tpath.name} ({len(words)} words)")
                if words is None:
                    wav = work / "dialogue.wav"
                    eprint("  extracting audio for speech recognition ...")
                    from .media import extract_audio_wav
                    extract_audio_wav(media, dialogue.type_index, wav)
                    words = asr.transcribe(wav, cfg, info.duration)
                    if scan_cfg.get("cache_transcript", True):
                        asr.save_transcript(tpath, words, cfg["asr"]["model"], dialogue.type_index)
                asr_hits = asr.scan_words(words, matcher)
                asr_source = f"whisper {cfg['asr']['model']}"
                eprint(f"  asr: {len(asr_hits)} hits")

    spans = build_spans(sub_hits, asr_hits, cfg, info.duration)
    meta = {
        "audio_track": dialogue.type_index,
        "level": cfg["level"],
        "subtitles": sub_source,
        "asr": asr_source,
        "duration": round(info.duration, 3),
    }
    return spans, meta


def render_file(media: Path, spans: list[Span], cfg: dict, args: argparse.Namespace,
                audio_track: int | None) -> Path | None:
    info = probe(media)
    dialogue = pick_audio_stream(info, cfg["languages"], audio_track)
    spec = BeepSpec.from_config(cfg)
    if getattr(args, "output", None):
        output = Path(args.output).expanduser()
    else:
        out_dir = Path(args.output_dir).expanduser() if getattr(args, "output_dir", None) else None
        output = default_output(media, cfg, out_dir)
    if not spans and not getattr(args, "force", False):
        eprint(f"  no spans to censor in {media.name}; skipping render (use --force to remux anyway)")
        return None
    if output.exists() and not getattr(args, "overwrite", False) and not getattr(args, "dry_run", False):
        raise UserError(f"{output} exists (use --overwrite)")
    return render(media, info, dialogue, spans, spec, cfg, output,
                  dry_run=getattr(args, "dry_run", False), verbose=getattr(args, "verbose", False))


# ----------------------------------------------------------------------------- commands

def cmd_scan(args, cfg) -> int:
    for media in expand_inputs(args.files, args.recursive):
        spans, meta = scan_file(media, cfg, args)
        side = sidecar_path(media)
        save_sidecar(side, media, spans, meta)
        print(f"\n{media.name}: {len(spans)} span(s) -> {side.name}")
        print(format_table(spans))
    return 0


def cmd_render(args, cfg) -> int:
    for media in expand_inputs(args.files, args.recursive):
        side = Path(args.spans).expanduser() if args.spans else sidecar_path(media)
        if not side.exists():
            raise UserError(f"no span file {side.name}; run 'audio-censor scan' first")
        spans, meta = load_sidecar(side)
        track = args.audio_track if args.audio_track is not None else meta.get("audio_track")
        eprint(f"{media.name}: {len(spans)} span(s) from {side.name}")
        out = render_file(media, spans, cfg, args, track)
        if out and not args.dry_run:
            print(f"wrote {out}")
    return 0


def cmd_process(args, cfg) -> int:
    failures = 0
    for media in expand_inputs(args.files, args.recursive):
        try:
            spans, meta = scan_file(media, cfg, args)
            side = sidecar_path(media)
            save_sidecar(side, media, spans, meta)
            print(format_table(spans))
            out = render_file(media, spans, cfg, args, meta["audio_track"])
            if out and not args.dry_run:
                print(f"wrote {out}")
        except (MediaError, BeepError, UserError, RuntimeError) as exc:
            failures += 1
            eprint(f"error: {media.name}: {exc}")
            if not args.keep_going:
                raise
    return 1 if failures else 0


def cmd_review(args, cfg) -> int:
    for media in expand_inputs(args.files, args.recursive):
        side = sidecar_path(media)
        if not side.exists():
            print(f"{media.name}: no span file")
            continue
        spans, meta = load_sidecar(side)
        print(f"{media.name}: {len(spans)} span(s)  level={meta.get('level')}  "
              f"subtitles={meta.get('subtitles')}  asr={meta.get('asr')}")
        print(format_table(spans))
    return 0


def cmd_info(args, cfg) -> int:
    for media in expand_inputs(args.files, args.recursive):
        info = probe(media)
        print(f"{media.name}  ({info.duration / 60:.1f} min)")
        for s in info.streams:
            print(f"  {s.describe()}")
    return 0


def cmd_words(args, cfg) -> int:
    matcher = Matcher.from_config(cfg)
    tiers = active_tiers(cfg["level"])
    print(f"level={cfg['level']}  censoring tiers: {', '.join(tiers)}")
    default = load_default_tiers()
    for tier in tiers:
        pats = sorted(p.raw for p in matcher.patterns if p.tier == tier and p.raw not in cfg["words"]["extra"])
        print(f"\n[{tier}] ({len(pats)})")
        print("  " + ", ".join(pats))
    if cfg["words"]["extra"]:
        print("\n[extra]\n  " + ", ".join(cfg["words"]["extra"]))
    if cfg["words"]["allow"]:
        print("\n[allow]\n  " + ", ".join(cfg["words"]["allow"]))
    unused = [t for t in default if t not in tiers]
    if unused:
        print(f"\n(not censored at this level: {', '.join(unused)})")
    return 0


def cmd_init_config(args, cfg) -> int:
    path = Path(args.path).expanduser() if args.path else default_config_path()
    write_template(path, force=args.force)
    print(f"wrote {path}")
    return 0


def cmd_preview_beep(args, cfg) -> int:
    spec = BeepSpec.from_config(cfg)
    if spec.mode == "mute":
        raise UserError("beep.mode is 'mute'; nothing to preview")
    out = Path(args.output).expanduser()
    render_preview(spec, out, duration=args.duration)
    print(f"wrote {out}  ({spec.wave}"
          + (f" {spec.frequency:g} Hz" if spec.wave != "file" else f" from {spec.file.name}")
          + f", volume {spec.volume:.2f})")
    return 0


# ----------------------------------------------------------------------------- parser

def add_scan_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("scanning")
    g.add_argument("--level", choices=TIERS, help="strictness: strong < moderate < mild (mild censors most)")
    g.add_argument("--no-asr", action="store_true", help="skip speech recognition")
    g.add_argument("--no-subtitles", action="store_true", help="skip subtitle scanning")
    g.add_argument("--subtitles", metavar="FILE", help="use this subtitle file instead of auto-detecting")
    g.add_argument("--audio-track", type=int, metavar="N", help="audio track to scan/censor (see 'info')")
    g.add_argument("--model", help="whisper model (tiny, base, small, medium, large-v3, ...)")
    g.add_argument("--device", choices=["auto", "cpu", "cuda"], help="whisper device")
    g.add_argument("--rescan", action="store_true", help="ignore a cached transcript")


def add_render_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("beep sound")
    g.add_argument("--mode", choices=["beep", "mute"])
    g.add_argument("--wave", choices=list(WAVES) + ["file"], help="beep waveform")
    g.add_argument("--frequency", type=float, metavar="HZ")
    g.add_argument("--volume", metavar="V", help="0..1 or decibels, written as --volume=-8dB")
    g.add_argument("--beep-file", metavar="SOUND", help="custom sound file (sets --wave file)")
    g.add_argument("--beep-channel", choices=["center", "all"])
    g.add_argument("--duck", type=float, metavar="LEVEL", help="residual dialogue level under the beep (0..1)")
    o = p.add_argument_group("output")
    o.add_argument("-o", "--output", metavar="FILE", help="output file (single input only)")
    o.add_argument("--output-dir", metavar="DIR")
    o.add_argument("--codec", help="clean track codec (default: aac for stereo, ac3 for 5.1)")
    o.add_argument("--title", help="clean track title")
    o.add_argument("--replace-audio", action="store_true", help="drop the original audio tracks")
    o.add_argument("--no-default", action="store_true", help="do not mark the clean track as default")
    o.add_argument("--overwrite", action="store_true")
    o.add_argument("--force", action="store_true", help="remux even when nothing was found")
    o.add_argument("--dry-run", action="store_true", help="print the ffmpeg command and stop")


def add_input_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("files", nargs="+", metavar="FILE", help="media files or directories")
    p.add_argument("-r", "--recursive", action="store_true", help="descend into directories")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="audio-censor",
                                description="Find profanity in films via subtitles and speech recognition, "
                                            "then remux a beeped 'Clean' audio track.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-c", "--config", metavar="FILE", help=f"config file (default {default_config_path()})")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("process", help="scan and render in one go (the usual command)")
    add_input_arg(sp); add_scan_options(sp); add_render_options(sp)
    sp.add_argument("--keep-going", action="store_true", help="continue with the next file after an error")
    sp.set_defaults(func=cmd_process)

    sp = sub.add_parser("scan", help="detect profanity and write <file>.censor.json for review")
    add_input_arg(sp); add_scan_options(sp)
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("render", help="remux using an existing (possibly hand-edited) <file>.censor.json")
    add_input_arg(sp); add_render_options(sp)
    sp.add_argument("--spans", metavar="JSON", help="span file to use instead of <file>.censor.json")
    sp.add_argument("--audio-track", type=int, metavar="N")
    sp.set_defaults(func=cmd_render)

    sp = sub.add_parser("review", help="print the spans recorded for a file")
    add_input_arg(sp)
    sp.set_defaults(func=cmd_review)

    sp = sub.add_parser("info", help="list a file's streams (to pick --audio-track)")
    add_input_arg(sp)
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("words", help="show the active word list")
    sp.add_argument("--level", choices=TIERS)
    sp.set_defaults(func=cmd_words)

    sp = sub.add_parser("init-config", help="write a commented config template")
    sp.add_argument("path", nargs="?", help=f"destination (default {default_config_path()})")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_init_config)

    sp = sub.add_parser("preview-beep", help="render the configured beep to a short audio file")
    sp.add_argument("-o", "--output", default="beep-preview.wav")
    sp.add_argument("-d", "--duration", type=float, default=1.0)
    for name, kw in (("--wave", dict(choices=list(WAVES) + ["file"])), ("--frequency", dict(type=float)),
                     ("--volume", {}), ("--beep-file", {}), ("--mode", dict(choices=["beep", "mute"]))):
        sp.add_argument(name, **kw)
    sp.set_defaults(func=cmd_preview_beep)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        cfg = apply_overrides(cfg, args)
        return args.func(args, cfg)
    except (MediaError, BeepError, UserError, FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        eprint(f"error: {exc}")
        return 1
    except KeyboardInterrupt:
        eprint("\ninterrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
