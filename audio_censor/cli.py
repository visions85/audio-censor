"""audio-censor command-line interface."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

from . import __version__, asr
from .beep import BeepError, BeepSpec, WAVES
from .config import default_config_path, load_config, write_template
from .media import VIDEO_EXTS, MediaError, eprint, language_matches, pick_audio_stream, probe
from .names import NameDetector
from .plex import ORDERS, PlexError, PlexRatings
from .ratings import find_rating, should_skip
from .render import default_output, remux_with_subtitles, render, render_preview
from .spans import Span, build_spans, format_table, load_sidecar, save_sidecar, sidecar_path, veto_by_subtitles
from .subtitles import STYLES, find_subtitle_source, load_cues, scan_cues, write_clean_subtitles
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
    if g("no_clean_subs"):
        cfg["subtitles"]["clean"] = False
    if g("no_names"):
        cfg["names"]["detect"] = False
    if g("any_language"):
        cfg["scan"]["audio_only_languages"] = False
    if g("standalone_subs"):
        cfg["subtitles"]["standalone"] = args.standalone_subs
    if cfg["subtitles"].get("standalone", "sidecar") not in ("sidecar", "remux", "skip"):
        raise UserError("subtitles.standalone must be sidecar, remux or skip")
    if g("sub_style"):
        cfg["subtitles"]["style"] = args.sub_style
    if g("sub_replacement") is not None:
        cfg["subtitles"]["replacement"] = args.sub_replacement
    if cfg["subtitles"]["style"] not in STYLES:
        raise UserError(f"subtitles.style must be one of {', '.join(STYLES)}")
    if cfg["level"] not in TIERS:
        raise UserError(f"level must be one of {', '.join(TIERS)}")
    return cfg


def scan_file(media: Path, cfg: dict, args: argparse.Namespace, position: str = "") -> tuple[list[Span], dict]:
    """Run subtitle + ASR scans and return merged spans plus metadata."""
    info = probe(media)
    dialogue = pick_audio_stream(info, cfg["languages"], getattr(args, "audio_track", None))
    matcher = Matcher.from_config(cfg)
    detector = NameDetector.from_config(cfg)
    scan_cfg = cfg["scan"]
    eprint(f"{position}{media.name}: {info.duration / 60:.1f} min, dialogue track {dialogue.describe()}")

    rating, rating_src = find_rating(media, info.tags, _plex_lookup(cfg, args))
    if rating and should_skip(rating, scan_cfg.get("skip_ratings", [])) and not getattr(args, "ignore_rating", False):
        eprint(f"  rated {rating} ({rating_src}); skipping")
        return [], {"audio_track": dialogue.type_index, "level": cfg["level"], "duration": round(info.duration, 3),
                    "rating": rating, "rating_source": rating_src, "skipped": f"rated {rating}",
                    "audio_censored": False, "subtitles": "n/a", "asr": "skipped", "names": []}
    if rating:
        eprint(f"  rated {rating} ({rating_src})")

    sub_hits, asr_hits, cues = [], [], []
    sub_source = "disabled"
    with tempfile.TemporaryDirectory(prefix="audio-censor-") as tmp:
        work = Path(tmp)
        censor_audio, lang, how = decide_audio_language(media, info, dialogue, cfg, args, work)
        base_meta = {
            "audio_track": dialogue.type_index, "level": cfg["level"], "duration": round(info.duration, 3),
            "audio_language": lang, "language_source": how, "audio_censored": censor_audio,
            "rating": rating, "rating_source": rating_src,
        }
        if not censor_audio:
            eprint(f"  audio language: {lang or 'unknown'} ({how}); not in {', '.join(cfg['languages'])}, "
                   "audio censoring skipped")
            return [], {**base_meta, "subtitles": "n/a", "asr": "skipped", "names": []}
        if how not in ("tag", "any"):
            eprint(f"  audio language: {lang} ({how})")

        if scan_cfg["use_subtitles"]:
            explicit = Path(args.subtitles).expanduser() if getattr(args, "subtitles", None) else None
            src = find_subtitle_source(media, info, cfg["languages"], work, explicit, _ignore_tags(cfg))
            cues = load_cues(src.path) if src else []
            sub_source = src.description if src else "no subtitles found"
            if cues:
                detector.learn([c.text for c in cues], matcher)
                sub_hits = scan_cues(cues, matcher, scan_cfg.get("subtitle_mode", "estimate"), detector)
                names = sum(h.name_use for h in sub_hits)
                eprint(f"  subtitles: {sub_source}, {len(cues)} cues, {len(sub_hits) - names} hits"
                       + (f", {names} name use(s) exempted" if names else ""))
                if detector.names:
                    eprint(f"  names detected: {detector.summary()}")
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
                asr_hits = asr.scan_words(words, matcher, detector)
                names = sum(h.name_use for h in asr_hits)
                asr_hits = veto_by_subtitles(asr_hits, cues, scan_cfg.get("homophones", {}))
                vetoed = sum(h.name_use for h in asr_hits) - names
                asr_source = f"whisper {cfg['asr']['model']}"
                eprint(f"  asr: {len(asr_hits) - names - vetoed} hits"
                       + (f", {names} name use(s) exempted" if names else "")
                       + (f", {vetoed} contradicted by subtitles (sound-alikes)" if vetoed else ""))

    spans = build_spans(sub_hits, asr_hits, cfg, info.duration)
    meta = {**base_meta, "subtitles": sub_source, "asr": asr_source, "names": sorted(detector.names)}
    return spans, meta


_PLEX: dict = {}


def _plex(cfg: dict, args) -> PlexRatings | None:
    if "server" not in _PLEX:
        try:
            _PLEX["server"] = PlexRatings.from_config(cfg, refresh=getattr(args, "refresh_plex", False))
        except PlexError as exc:
            eprint(f"  plex: {exc}")
            _PLEX["server"] = None
    return _PLEX["server"]


def order_files(files: list[Path], cfg: dict, args) -> list[Path]:
    """--order rating|added|watched sorts a batch with Plex's data; name is plain path order."""
    by = getattr(args, "order", None) or "name"
    if by == "name" or len(files) < 2:
        return files
    plex = _plex(cfg, args)
    if plex is None:
        raise UserError(f"--order {by} needs a Plex server: set [plex] url/token or PLEX_URL/PLEX_TOKEN")
    ordered, matched = plex.order(files, by)
    label = {"rating": "Plex rating, best first", "added": "most recently added first",
             "watched": "most watched first"}[by]
    eprint(f"ordering {len(files)} file(s) by {label} ({matched} matched in Plex, the rest last)\n")
    return ordered


def _plex_lookup(cfg: dict, args):
    """A rating_for callable backed by the configured Plex server, built once per run (or None)."""
    if "lookup" not in _PLEX:
        _PLEX["lookup"] = None
        if not cfg["scan"].get("skip_ratings"):
            return None
        plex = _plex(cfg, args)
        if plex is not None:
            def lookup(media: Path) -> str:
                try:
                    return plex.rating_for(media)
                except PlexError as exc:
                    eprint(f"  plex: {exc}; ratings unavailable for this run")
                    _PLEX["lookup"] = None
                    return ""
            _PLEX["lookup"] = lookup
    return _PLEX["lookup"]


def decide_audio_language(media: Path, info, dialogue, cfg: dict, args, work: Path) -> tuple[bool, str, str]:
    """Is this track in a language we censor? -> (yes/no, language code, how we know)."""
    scan = cfg["scan"]
    forced = getattr(args, "audio_language", None)
    lang = (forced or dialogue.language or "").lower()
    if lang == "und":
        lang = ""
    how = "forced" if forced else ("tag" if lang else "untagged")
    if not scan.get("audio_only_languages", True):
        return True, lang, "any"
    if not lang:
        if scan.get("detect_language", True) and scan.get("use_asr", True) and asr.available():
            lang, prob = asr.detect_language(media, dialogue.type_index, info.duration, cfg, work)
            how = f"whisper, {prob:.0%} confidence"
        else:
            lang = (scan.get("assume_untagged") or "").lower()
            how = "untagged, assumed" if lang else "untagged"
    return language_matches(lang, cfg["languages"]), lang, how


def _ignore_tags(cfg: dict) -> tuple[str, ...]:
    tag = (cfg["subtitles"].get("standalone_suffix") or ".clean").strip(".").lower()
    return tuple({tag, "clean"})


def resolve_output(media: Path, cfg: dict, args) -> Path:
    if getattr(args, "output", None):
        return Path(args.output).expanduser()
    out_dir = Path(args.output_dir).expanduser() if getattr(args, "output_dir", None) else None
    return default_output(media, cfg, out_dir)


def standalone_outputs(media: Path, cfg: dict, args) -> list[Path]:
    """Clean subtitle files already written beside a foreign-language film (any language/format)."""
    suffix = (cfg["subtitles"].get("standalone_suffix") or ".clean").strip(".")
    return sorted(p for p in media.parent.glob(f"{media.stem}.*")
                  if p.suffix.lower() in (".srt", ".ass", ".ssa", ".vtt")
                  and suffix in p.name[len(media.stem):].lower().split("."))


def render_subtitles_only(media: Path, cfg: dict, args, meta: dict) -> tuple[Path | None, str]:
    """Foreign-language audio: censor the English subtitles without touching the audio.

    Returns (path written, status) with status one of written / skipped / nothing.
    """
    sub_cfg = cfg["subtitles"]
    mode = sub_cfg.get("standalone", "sidecar")
    if not sub_cfg.get("clean", True) or mode == "skip":
        eprint("  clean subtitles: disabled for non-English audio, nothing to do")
        return None, "nothing"
    info = probe(media)
    dry_run = getattr(args, "dry_run", False)
    with tempfile.TemporaryDirectory(prefix="audio-censor-") as tmp:
        work = Path(tmp)
        explicit = Path(args.subtitles).expanduser() if getattr(args, "subtitles", None) else None
        src = find_subtitle_source(media, info, cfg["languages"], work, explicit, _ignore_tags(cfg))
        if src is None:
            eprint("  clean subtitles: no subtitles found, nothing to do")
            return None, "nothing"
        if src.language and not language_matches(src.language, cfg["languages"]):
            eprint(f"  clean subtitles: only {src.language} subtitles found, nothing to do")
            return None, "nothing"
        lang = src.language
        suffix = sub_cfg.get("standalone_suffix", ".clean")
        if mode == "sidecar":
            target = media.parent / (f"{media.stem}.{lang}{suffix}{src.path.suffix}" if lang
                                     else f"{media.stem}{suffix}{src.path.suffix}")
        else:
            output = resolve_output(media, cfg, args)
            name = f"{output.stem}.{lang}{src.path.suffix}" if lang else f"{output.stem}{src.path.suffix}"
            target = output.parent / name if sub_cfg.get("sidecar", True) else work / name
        if dry_run:
            eprint(f"  clean subtitles: would write {target}")
            return target, "written"
        if mode == "sidecar" and target.exists() and not getattr(args, "overwrite", False):
            eprint(f"  clean subtitles: {target.name} exists, skipping (--overwrite to redo)")
            return None, "skipped"
        count = write_clean_subtitles(src, Matcher.from_config(cfg), target, sub_cfg.get("style", "asterisks"),
                                      sub_cfg.get("replacement", "[BLEEP]"), NameDetector.from_config(cfg))
        eprint(f"  clean subtitles: {count} word(s) masked from {src.description} -> {target.name}")
        if mode == "remux":
            return remux_with_subtitles(media, info, target, lang, cfg, output, dry_run=dry_run,
                                        verbose=getattr(args, "verbose", False)), "written"
        return target, "written"


def render_file(media: Path, spans: list[Span], cfg: dict, args: argparse.Namespace,
                audio_track: int | None) -> Path | None:
    info = probe(media)
    dialogue = pick_audio_stream(info, cfg["languages"], audio_track)
    spec = BeepSpec.from_config(cfg)
    output = resolve_output(media, cfg, args)
    if not spans and not getattr(args, "force", False):
        eprint(f"  no spans to censor in {media.name}; skipping render (use --force to remux anyway)")
        return None
    if output.exists() and not getattr(args, "overwrite", False) and not getattr(args, "dry_run", False):
        raise UserError(f"{output} exists (use --overwrite)")
    dry_run = getattr(args, "dry_run", False)
    with tempfile.TemporaryDirectory(prefix="audio-censor-") as tmp:
        clean_subs, lang = prepare_clean_subtitles(media, info, cfg, args, output, Path(tmp), dry_run)
        return render(media, info, dialogue, spans, spec, cfg, output, dry_run=dry_run,
                      verbose=getattr(args, "verbose", False), clean_subs=clean_subs, clean_subs_lang=lang)


def prepare_clean_subtitles(media: Path, info, cfg: dict, args, output: Path, work: Path,
                            dry_run: bool) -> tuple[Path | None, str]:
    """Write censored subtitles (sidecar and/or temp file for embedding). Returns (path, language)."""
    sub_cfg = cfg["subtitles"]
    if not sub_cfg.get("clean", True) or not (sub_cfg.get("sidecar", True) or sub_cfg.get("embed", True)):
        return None, ""
    explicit = Path(args.subtitles).expanduser() if getattr(args, "subtitles", None) else None
    src = find_subtitle_source(media, info, cfg["languages"], work, explicit, _ignore_tags(cfg))
    if src is None:
        eprint("  clean subtitles: no subtitles found, skipping")
        return None, ""
    lang = src.language
    name = f"{output.stem}.{lang}{src.path.suffix}" if lang else f"{output.stem}{src.path.suffix}"
    target = output.parent / name if sub_cfg.get("sidecar", True) else work / name
    if dry_run:
        eprint(f"  clean subtitles: would write {target}")
        return (target if sub_cfg.get("embed", True) else None), lang
    count = write_clean_subtitles(src, Matcher.from_config(cfg), target,
                                  sub_cfg.get("style", "asterisks"), sub_cfg.get("replacement", "[BLEEP]"),
                                  NameDetector.from_config(cfg))
    where = target.name if sub_cfg.get("sidecar", True) else "embedded only"
    eprint(f"  clean subtitles: {count} word(s) masked from {src.description} -> {where}")
    return (target if sub_cfg.get("embed", True) else None), lang


# ----------------------------------------------------------------------------- commands

def _is_batch(args, files: list[Path]) -> bool:
    """A directory argument means batch semantics even if it holds a single file."""
    return len(files) > 1 or any(Path(f).expanduser().is_dir() for f in args.files)


def _batch_guard(files: list[Path], args, counts: dict, media: Path, exc: Exception) -> None:
    """Report a per-file failure and keep going, unless told (or a single file) to stop.

    Any exception counts: a crash inside a library on one odd file must not end a
    5000-file overnight run. -v prints the traceback.
    """
    if getattr(args, "stop_on_error", False) or not _is_batch(args, files):
        raise exc
    counts["failed"] += 1
    known = isinstance(exc, (MediaError, BeepError, UserError, RuntimeError, OSError))
    eprint(f"error: {media.name}: {exc if known else f'{type(exc).__name__}: {exc}'}")
    if getattr(args, "verbose", False) or not known:
        import traceback
        tb = traceback.format_exception(exc)
        eprint("".join(tb[-3:]).rstrip() if not getattr(args, "verbose", False) else "".join(tb).rstrip())


def _summary(files: list[Path], counts: dict, args) -> None:
    if _is_batch(args, files):
        print(f"{len(files)} file(s): " + ", ".join(f"{v} {k}" for k, v in counts.items() if v))


def cmd_scan(args, cfg) -> int:
    """Phase one: write <file>.censor.json (and the transcript cache) beside every film."""
    files = order_files(expand_inputs(args.files, args.recursive), cfg, args)
    counts = {"scanned": 0, "subtitles only": 0, "skipped (rating)": 0, "already scanned": 0, "failed": 0}
    started = time.time()
    for n, media in enumerate(files, 1):
        side = sidecar_path(media)
        pos = f"[{n}/{len(files)}] " if _is_batch(args, files) else ""
        try:
            if side.exists() and not args.overwrite and not args.rescan:
                spans, meta = load_sidecar(side)
                eprint(f"{pos}{media.name}: already scanned ({len(spans)} span(s), level {meta.get('level')}); "
                       "--overwrite to redo")
                counts["already scanned"] += 1
                continue
            spans, meta = scan_file(media, cfg, args, pos)
            save_sidecar(side, media, spans, meta)
            if meta.get("skipped"):
                print(f"{media.name}: {meta['skipped']}, skipped -> {side.name}")
                counts["skipped (rating)"] += 1
            elif meta.get("audio_censored", True):
                print(f"{media.name}: {len(spans)} span(s) -> {side.name}")
                print(format_table(spans))
                counts["scanned"] += 1
            else:
                print(f"{media.name}: audio not censored ({meta.get('audio_language') or 'unknown'}) -> {side.name}")
                counts["subtitles only"] += 1
        except Exception as exc:  # noqa: BLE001 - see _batch_guard
            _batch_guard(files, args, counts, media, exc)
        if _is_batch(args, files):
            done = counts["scanned"] + counts["subtitles only"] + counts["failed"]
            if done and n < len(files):
                per = (time.time() - started) / done
                eprint(f"  {len(files) - n} file(s) left, ~{int(per * (len(files) - n) / 60)} min at this pace\n")
            else:
                eprint("")
    _summary(files, counts, args)
    return 1 if counts["failed"] else 0


def cmd_render(args, cfg) -> int:
    """Phase two: remux from the span files written by scan. Unscanned files are skipped."""
    files = expand_inputs(args.files, args.recursive)
    if args.spans and len(files) > 1:
        raise UserError("--spans works with a single input")
    if args.output and len(files) > 1:
        raise UserError("-o/--output works with a single input; use --output-dir for batches")
    counts = {"rendered": 0, "subtitles only": 0, "clean already": 0, "not scanned": 0, "skipped": 0, "failed": 0}
    base_level = cfg["level"]
    for media in files:
        try:
            side = Path(args.spans).expanduser() if args.spans else sidecar_path(media)
            if not side.exists():
                if not _is_batch(args, files):
                    raise UserError(f"no span file {side.name}; run 'audio-censor scan' first")
                eprint(f"{media.name}: not scanned yet, skipping")
                counts["not scanned"] += 1
                continue
            output = resolve_output(media, cfg, args)
            if output.exists() and not args.overwrite and not args.dry_run:
                eprint(f"{media.name}: {output.name} exists, skipping (--overwrite to redo)")
                counts["skipped"] += 1
                continue
            spans, meta = load_sidecar(side)
            if meta.get("skipped"):
                eprint(f"{media.name}: {meta['skipped']}, nothing to render")
                counts["clean already"] += 1
                continue
            track = args.audio_track if args.audio_track is not None else meta.get("audio_track")
            # mask the same words in subtitles that the scan found, unless --level says otherwise
            cfg["level"] = meta["level"] if not args.level and meta.get("level") in TIERS else base_level
            eprint(f"{media.name}: {len(spans)} span(s) from {side.name}, level {cfg['level']}")
            if meta.get("audio_censored") is False:
                eprint(f"  audio language {meta.get('audio_language') or 'unknown'}: subtitles only")
                out, status = render_subtitles_only(media, cfg, args, meta)
                counts[{"written": "subtitles only", "skipped": "skipped", "nothing": "clean already"}[status]] += 1
            else:
                out = render_file(media, spans, cfg, args, track)
                counts["rendered" if spans else "clean already"] += 1
            if out and not args.dry_run:
                print(f"wrote {out}")
        except Exception as exc:  # noqa: BLE001 - see _batch_guard
            _batch_guard(files, args, counts, media, exc)
        if _is_batch(args, files):
            eprint("")
    _summary(files, counts, args)
    return 1 if counts["failed"] else 0


def cmd_status(args, cfg) -> int:
    """Where every file stands: unscanned, scanned (N spans), or rendered."""
    files = expand_inputs(args.files, args.recursive)
    rows = []
    totals = {"unscanned": 0, "scanned": 0, "rendered": 0, "nothing to do": 0}
    for media in files:
        side = sidecar_path(media)
        output = resolve_output(media, cfg, args)
        if not side.exists():
            state, detail = "unscanned", ""
        else:
            spans, meta = load_sidecar(side)
            if meta.get("skipped"):
                state, detail = "nothing to do", meta["skipped"]
            elif meta.get("audio_censored") is False:
                detail = f"audio {meta.get('audio_language') or '?'}, subtitles only"
                done = output.exists() or bool(standalone_outputs(media, cfg, args))
                state = "rendered" if done else "scanned"
            else:
                detail = f"{len(spans)} span(s), level {meta.get('level')}"
                if meta.get("asr") in ("skipped", "unavailable", "disabled"):
                    detail += ", no asr"
                state = "rendered" if output.exists() else ("scanned" if spans else "nothing to do")
        totals[state] += 1
        rows.append((media, state, detail))
    def label(m: Path) -> str:
        return str(m.relative_to(Path.cwd())) if m.is_relative_to(Path.cwd()) else m.name
    width = max((len(label(m)) for m, _, _ in rows), default=10)
    for media, state, detail in rows:
        print(f"{label(media):<{width}}  {state:<13}  {detail}")
    if _is_batch(args, files):
        print(f"\n{len(files)} file(s): " + ", ".join(f"{v} {k}" for k, v in totals.items() if v))
    return 0


def cmd_process(args, cfg) -> int:
    files = order_files(expand_inputs(args.files, args.recursive), cfg, args)
    if args.output and len(files) > 1:
        raise UserError("-o/--output works with a single input; use --output-dir for batches")
    counts = {"censored": 0, "subtitles only": 0, "clean already": 0, "skipped": 0, "skipped (rating)": 0, "failed": 0}
    for n, media in enumerate(files, 1):
        pos = f"[{n}/{len(files)}] " if _is_batch(args, files) else ""
        try:
            output = resolve_output(media, cfg, args)
            if output.exists() and not args.overwrite and not args.dry_run:
                eprint(f"{pos}{media.name}: {output.name} exists, skipping (--overwrite to redo)")
                counts["skipped"] += 1
                continue
            spans, meta = scan_file(media, cfg, args, pos)
            save_sidecar(sidecar_path(media), media, spans, meta)
            if meta.get("skipped"):
                counts["skipped (rating)"] += 1
                out = None
            elif not meta.get("audio_censored", True):
                out, status = render_subtitles_only(media, cfg, args, meta)
                counts[{"written": "subtitles only", "skipped": "skipped", "nothing": "clean already"}[status]] += 1
            else:
                print(format_table(spans))
                out = render_file(media, spans, cfg, args, meta["audio_track"])
                counts["censored" if spans else "clean already"] += 1
            if out and not args.dry_run:
                print(f"wrote {out}")
        except Exception as exc:  # noqa: BLE001 - see _batch_guard
            _batch_guard(files, args, counts, media, exc)
        if _is_batch(args, files):
            eprint("")
    _summary(files, counts, args)
    return 1 if counts["failed"] else 0


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
    g.add_argument("--no-names", action="store_true", help="censor Dick even when it is a character's name")
    g.add_argument("--any-language", action="store_true", help="censor audio whatever language it is in")
    g.add_argument("--ignore-rating", action="store_true", help="scan even films rated G / TV-Y / TV-G")
    g.add_argument("--refresh-plex", action="store_true", help="re-fetch ratings from Plex instead of the 6h cache")
    g.add_argument("--order", choices=ORDERS, help="batch order: name (default), or via Plex: rating (best first), "
                                                   "added (newest first), watched (most viewed first)")
    g.add_argument("--audio-language", metavar="CODE", help="treat the dialogue track as this language (eng, fre ...)")


def add_render_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("beep sound")
    g.add_argument("--mode", choices=["beep", "mute"])
    g.add_argument("--wave", choices=list(WAVES) + ["file"], help="beep waveform")
    g.add_argument("--frequency", type=float, metavar="HZ")
    g.add_argument("--volume", metavar="V", help="0..1 or decibels, written as --volume=-8dB")
    g.add_argument("--beep-file", metavar="SOUND", help="custom sound file (sets --wave file)")
    g.add_argument("--beep-channel", choices=["center", "all"])
    g.add_argument("--duck", type=float, metavar="LEVEL", help="residual dialogue level under the beep (0..1)")
    c = p.add_argument_group("clean subtitles")
    c.add_argument("--no-clean-subs", action="store_true", help="do not write censored subtitles")
    c.add_argument("--sub-style", choices=STYLES, help="how masked words look (default asterisks)")
    c.add_argument("--sub-replacement", metavar="TEXT", help="replacement text for --sub-style bleep")
    if not any(a.dest == "no_names" for a in p._actions):
        c.add_argument("--no-names", action="store_true", help="censor Dick even when it is a character's name")
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
    o.add_argument("--standalone-subs", choices=["sidecar", "remux", "skip"],
                   help="what to do with non-English films: censored subtitle file beside the original, "
                        "a remuxed copy with the clean subtitle track, or nothing")


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
    sp.add_argument("--stop-on-error", action="store_true", help="abort the batch at the first failing file")
    sp.set_defaults(func=cmd_process)

    sp = sub.add_parser("scan", help="phase one: detect profanity and write <file>.censor.json beside each file")
    add_input_arg(sp); add_scan_options(sp)
    sp.add_argument("--overwrite", action="store_true", help="re-scan files that already have a span file")
    sp.add_argument("--stop-on-error", action="store_true", help="abort the batch at the first failing file")
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("render", help="phase two: remux from the span files written by scan (unscanned files are skipped)")
    add_input_arg(sp); add_render_options(sp)
    sp.add_argument("--spans", metavar="JSON", help="span file to use instead of <file>.censor.json")
    sp.add_argument("--audio-track", type=int, metavar="N")
    sp.add_argument("--subtitles", metavar="FILE", help="subtitle file to censor instead of auto-detecting")
    sp.add_argument("--level", choices=TIERS, help="word level for clean subtitles (default: the scan's level)")
    sp.add_argument("--stop-on-error", action="store_true", help="abort the batch at the first failing file")
    sp.set_defaults(func=cmd_render)

    sp = sub.add_parser("status", help="show which files are unscanned, scanned or rendered")
    add_input_arg(sp)
    sp.add_argument("--output-dir", metavar="DIR", help="where rendered files live, if not beside the originals")
    sp.set_defaults(func=cmd_status)

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
