"""Configuration: defaults, TOML loading, deep merge."""

from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path

DEFAULT_PROMPT = (
    "Transcribe every word exactly as spoken, including profanity such as "
    "fuck, fucking, shit, bitch, asshole, goddamn and damn."
)

DEFAULTS: dict = {
    "level": "moderate",          # strong | moderate | mild  (mild censors the most)
    "languages": ["eng", "en"],   # preferred audio / subtitle languages
    "scan": {
        "use_subtitles": True,
        "use_asr": True,
        "subtitle_mode": "estimate",   # estimate (word position within cue) | cue (whole cue)
        "subtitle_pad": 0.35,          # seconds added either side of an estimated subtitle hit
        "asr_pad": 0.10,               # seconds added either side of a recognized word
        "merge_gap": 0.25,             # spans closer than this are merged
        "subtitle_drift": 2.0,         # an ASR word this far outside its cue still claims the subtitle hit
        "asr_only": True,              # keep ASR hits that no subtitle confirms
        "cache_transcript": True,      # save <file>.transcript.json so re-scans skip ASR
        "treat_asterisks_as_hit": True,
    },
    "asr": {
        "model": "small",              # tiny, base, small, medium, large-v3, distil-large-v3 ...
        "device": "auto",              # auto | cpu | cuda
        "compute_type": "auto",        # auto | int8 | float16 | float32
        "language": "en",
        "beam_size": 5,
        "vad": True,
        "initial_prompt": DEFAULT_PROMPT,
    },
    "beep": {
        "mode": "beep",                # beep | mute
        "wave": "sine",                # sine | square | triangle | sawtooth | noise | file
        "frequency": 1000,
        "volume": 0.4,                 # linear 0..1, or a string like "-8dB"
        "file": "",                    # sound file used when wave = "file"
        "loop_file": True,             # loop a short file to fill long spans
        "fade": 0.01,                  # fade in/out seconds for each beep
        "channel": "center",           # center | all
        "duck": 0.0,                   # residual dialogue level under the beep (0 = silent)
    },
    "output": {
        "suffix": ".clean",
        "container": "mkv",
        "codec": "auto",               # auto | aac | ac3 | eac3 | flac | opus ...
        "bitrate": "auto",
        "title": "Clean (beeped)",
        "set_default": True,           # make the clean track the default audio
        "keep_original": True,         # keep the original audio tracks in the output
    },
    "words": {
        "extra": [],                   # extra patterns, censored at every level
        "allow": [],                   # patterns never censored (e.g. "dick" if it's a name)
        "tiers": {},                   # {"strong": [...], "moderate": [...], "mild": [...]}
    },
}

TEMPLATE = '''# audio-censor configuration
# Every key is optional; anything missing falls back to the built-in default.

level = "moderate"          # strong | moderate | mild   (mild censors the most words)
languages = ["eng", "en"]

[scan]
use_subtitles = true
use_asr = true
subtitle_mode = "estimate"  # estimate | cue
subtitle_pad = 0.35
asr_pad = 0.10
merge_gap = 0.25
subtitle_drift = 2.0
asr_only = true
cache_transcript = true

[asr]
model = "small"             # small is a fine CPU default; medium or large-v3 on a GPU
device = "auto"
compute_type = "auto"
language = "en"

[beep]
mode = "beep"               # beep | mute
wave = "sine"               # sine | square | triangle | sawtooth | noise | file
frequency = 1000
volume = 0.4                # 0..1 or "-8dB"
file = ""                   # e.g. "~/sounds/quack.wav" when wave = "file"
loop_file = true
fade = 0.01
channel = "center"          # center | all
duck = 0.0

[output]
suffix = ".clean"
container = "mkv"
codec = "auto"
bitrate = "auto"
title = "Clean (beeped)"
set_default = true
keep_original = true

[words]
extra = []                  # e.g. ["moist", "stupid*"]
allow = []                  # e.g. ["dick"] when Dick is a character's name
[words.tiers]
# strong = ["some* new* pattern*"]
'''


def default_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "audio-censor" / "config.toml"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load defaults merged with the user's TOML file (if it exists)."""
    cfg_path = Path(path) if path else default_config_path()
    user: dict = {}
    if cfg_path.exists():
        with open(cfg_path, "rb") as fh:
            user = tomllib.load(fh)
    elif path:
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    return deep_merge(DEFAULTS, user)


def write_template(path: Path, force: bool = False) -> Path:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists (use --force to overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE, encoding="utf-8")
    return path
