"""A re-encoded "Family" edition with scenes cut out, next to the original.

Only for films that have scenes to remove (`audio-censor skip`): muting alone never needs
a re-encode. The edition is cut frame-exactly while decoding (select/aselect), encoded
with the GPU (hevc_nvenc, 10-bit, HDR10 metadata carried through from the source
frames), and gets the clean audio track and the clean subtitles shifted to match the
cuts. Plex shows a file named `... {edition-Family}.mkv` as an edition of the same film.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pysubs2

from .media import MediaError, MediaInfo, Stream, eprint, probe, run
from .scenes import Chapter, merge_overlaps

DEFAULT_CQ = 22


@dataclass
class Cut:
    keep: list[tuple[float, float]]      # spans of the original that stay
    removed: list[tuple[float, float]]   # merged scenes

    def map_time(self, t: float) -> float | None:
        """Original time -> edition time, or None if `t` falls inside a removed scene."""
        shift = 0.0
        for s, e in self.removed:
            if t >= e:
                shift += e - s
            elif t >= s:
                return None
        return t - shift

    @property
    def removed_seconds(self) -> float:
        return sum(e - s for s, e in self.removed)


def plan_cut(scenes: list[tuple[float, float]], duration: float) -> Cut:
    removed = [(max(0.0, s), min(duration, e)) for s, e in merge_overlaps(scenes) if e > 0 and s < duration]
    keep, t = [], 0.0
    for s, e in removed:
        if s > t:
            keep.append((t, s))
        t = e
    if t < duration:
        keep.append((t, duration))
    return Cut(keep, removed)


def select_expr(cut: Cut) -> str:
    """ffmpeg select/aselect expression dropping the removed scenes."""
    parts = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in cut.removed)
    return f"not({parts})"


def shift_subtitles(src: Path, dst: Path, cut: Cut) -> int:
    """Drop cues inside removed scenes, move later ones up. Returns cues kept."""
    subs = pysubs2.load(str(src), encoding="utf-8")
    kept = pysubs2.SSAFile()
    for ev in subs.events:
        start, end = cut.map_time(ev.start / 1000), cut.map_time(max(ev.start, ev.end - 1) / 1000)
        if start is None or end is None:
            continue
        ev.start, ev.end = int(start * 1000), int(end * 1000) + 1
        kept.append(ev)
    kept.save(str(dst), encoding="utf-8")
    return len(kept)


def shift_chapters(chapters: list[Chapter], cut: Cut) -> list[Chapter]:
    out = []
    for c in chapters:
        t = cut.map_time(c.start)
        if t is not None and (not out or t > out[-1].start + 0.5):
            out.append(Chapter(t, c.title))
    return out


def output_path(media: Path, name: str) -> Path:
    """`<folder name> {edition-Name}.mkv` beside the film (Plex editions; Jellyfin versions)."""
    return media.with_name(f"{media.parent.name} {{edition-{name}}}.mkv")


def has_dolby_vision(media: Path) -> bool:
    proc = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream_side_data",
                "-of", "json", str(media)])
    data = json.loads(proc.stdout or "{}")
    return any("DOVI" in (d.get("side_data_type") or "").upper() or "dolby" in (d.get("side_data_type") or "").lower()
               for s in data.get("streams", []) for d in s.get("side_data_list", []))


def clean_audio(info: MediaInfo) -> Stream:
    for a in info.of_type("audio"):
        t = (a.title or "").strip()
        if t.startswith("Clean (") and t.endswith(")"):
            return a
    raise MediaError("no clean audio track in the film; run 'audio-censor process' on it first")


def build_command(media: Path, info: MediaInfo, cut: Cut, out: Path, subs: Path | None, cq: int) -> list[str]:
    audio = clean_audio(info)
    expr = select_expr(cut)
    graph = (f"[0:v:0]select='{expr}',setpts=N/FRAME_RATE/TB[v];"
             f"[0:a:{audio.type_index}]aselect='{expr}',asetpts=N/SR/TB[a]")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-stats", "-i", str(media)]
    if subs is not None:
        cmd += ["-i", str(subs)]
    cmd += ["-filter_complex", graph, "-map", "[v]", "-map", "[a]"]
    if subs is not None:
        cmd += ["-map", "1:s:0"]
    cmd += [
        "-c:v", "hevc_nvenc", "-preset", "p7", "-tune", "hq", "-multipass", "fullres",
        "-spatial-aq", "1", "-temporal-aq", "1", "-rc-lookahead", "32", "-b_ref_mode", "middle",
        "-profile:v", "main10", "-pix_fmt", "p010le",
        "-rc", "vbr", "-cq", str(cq), "-b:v", "0", "-maxrate", "150M", "-bufsize", "300M",
        "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc",
        "-color_range", "tv", "-chroma_sample_location", "topleft",
        "-c:a", "ac3", "-b:a", "640k" if (audio.channels or 2) > 2 else "256k",
        "-metadata:s:a:0", f"title={audio.title}", "-metadata:s:a:0", f"language={audio.language or 'eng'}",
        "-disposition:a:0", "default",
    ]
    if subs is not None:
        cmd += ["-c:s", "srt", "-metadata:s:s:0", "title=Clean", "-metadata:s:s:0", "language=eng",
                "-disposition:s:0", "default"]
    cmd += ["-map_chapters", "-1", "-max_muxing_queue_size", "4096", str(out)]
    return cmd


def verify(out: Path, expected: float) -> None:
    info = probe(out)
    if abs(info.duration - expected) > 2.0:
        raise MediaError(f"edition is {info.duration:.1f} s long, expected {expected:.1f} s")
    if not info.of_type("video") or not info.of_type("audio"):
        raise MediaError("edition is missing its video or audio")
    eprint(f"  edition: {info.duration / 60:.1f} min, {out.stat().st_size / 1e9:.1f} GB")
