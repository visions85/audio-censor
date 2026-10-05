"""ffprobe / ffmpeg helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTS = {".mkv", ".mp4", ".m4v", ".mov", ".avi", ".webm", ".ts", ".mpg", ".mpeg", ".wmv", ".flv"}


class MediaError(RuntimeError):
    pass


@dataclass
class Stream:
    index: int            # absolute stream index
    type_index: int       # index among streams of the same type (what ffmpeg's a:N means)
    codec_type: str
    codec_name: str
    language: str
    title: str
    default: bool
    channels: int = 0
    channel_layout: str = ""
    sample_rate: int = 0

    def describe(self) -> str:
        bits = [f"{self.codec_type[0]}:{self.type_index}", self.codec_name]
        if self.language:
            bits.append(self.language)
        if self.codec_type == "audio":
            bits.append(self.channel_layout or f"{self.channels}ch")
        if self.title:
            bits.append(f'"{self.title}"')
        if self.default:
            bits.append("(default)")
        return " ".join(bits)


@dataclass
class MediaInfo:
    path: Path
    duration: float
    streams: list[Stream]

    def of_type(self, kind: str) -> list[Stream]:
        return [s for s in self.streams if s.codec_type == kind]


def require_tools() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise MediaError(f"{tool} not found on PATH; install ffmpeg first")


def run(cmd: list[str], *, quiet: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE if quiet else None,
                          stderr=subprocess.PIPE if quiet else None, text=True)
    if check and proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()
        tail = "\n".join(err[-15:]) if err else f"exit code {proc.returncode}"
        raise MediaError(f"command failed: {' '.join(cmd[:3])} ...\n{tail}")
    return proc


def probe(path: str | Path) -> MediaInfo:
    require_tools()
    path = Path(path)
    if not path.exists():
        raise MediaError(f"file not found: {path}")
    proc = run(["ffprobe", "-v", "error", "-of", "json", "-show_streams", "-show_format", str(path)])
    data = json.loads(proc.stdout or "{}")
    counters: dict[str, int] = {}
    streams = []
    for raw in data.get("streams", []):
        kind = raw.get("codec_type", "unknown")
        tags = raw.get("tags", {}) or {}
        type_index = counters.get(kind, 0)
        counters[kind] = type_index + 1
        streams.append(Stream(
            index=raw["index"], type_index=type_index, codec_type=kind,
            codec_name=raw.get("codec_name", "?"),
            language=(tags.get("language") or "").lower(),
            title=tags.get("title") or "",
            default=bool((raw.get("disposition") or {}).get("default", 0)),
            channels=int(raw.get("channels", 0) or 0),
            channel_layout=raw.get("channel_layout", "") or "",
            sample_rate=int(raw.get("sample_rate", 0) or 0),
        ))
    duration = float((data.get("format") or {}).get("duration") or 0.0)
    if duration <= 0:
        for raw in data.get("streams", []):
            try:
                duration = max(duration, float(raw.get("duration") or 0))
            except (TypeError, ValueError):
                pass
    return MediaInfo(path, duration, streams)


def language_matches(lang: str, preferred: list[str]) -> bool:
    lang = (lang or "").lower()
    return bool(lang) and any(lang == p.lower() or lang.startswith(p.lower()) for p in preferred)


def pick_audio_stream(info: MediaInfo, languages: list[str], requested: int | None = None) -> Stream:
    audio = info.of_type("audio")
    if not audio:
        raise MediaError(f"{info.path.name} has no audio streams")
    if requested is not None:
        for s in audio:
            if s.type_index == requested:
                return s
        raise MediaError(f"audio track {requested} not found (have 0..{len(audio) - 1})")
    # Prefer: a default track in a preferred language, then any preferred language,
    # then the default track, then the first.
    for s in audio:
        if s.default and language_matches(s.language, languages):
            return s
    for s in audio:
        if language_matches(s.language, languages):
            return s
    for s in audio:
        if s.default:
            return s
    return audio[0]


def extract_audio_wav(path: Path, audio_type_index: int, out: Path, rate: int = 16000) -> Path:
    """Mono 16 kHz PCM for speech recognition."""
    run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-map", f"0:a:{audio_type_index}",
         "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", str(out)])
    return out


def extract_audio_clip(path: Path, audio_type_index: int, out: Path, start: float, length: float,
                       rate: int = 16000) -> Path:
    """A short mono clip, for language detection."""
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(path),
         "-map", f"0:a:{audio_type_index}", "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(rate),
         "-c:a", "pcm_s16le", str(out)])
    return out


def extract_subtitle(path: Path, sub_type_index: int, out: Path) -> Path:
    run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-map", f"0:s:{sub_type_index}",
         "-c:s", "srt", str(out)])
    return out


def media_duration_of(path: Path) -> float:
    return probe(path).duration


def eprint(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, **kwargs)
