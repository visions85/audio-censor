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
        "audio_only_languages": True,  # only censor audio whose dialogue is in `languages`
        "detect_language": True,       # use Whisper language ID when the audio track is untagged
        "assume_untagged": "eng",      # language assumed for untagged tracks when detection is unavailable ("" = skip)
        "skip_ratings": ["G", "TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G"],   # ratings (from NFO / tags / Plex) to skip
        # An ASR-only hit is dropped when the subtitle line at that moment contains one of
        # these innocent sound-alikes and not the flagged word itself ("Hoover Dam").
        "homophones": {
            "damn": ["dam", "dams"], "hell": ["he'll"], "bitch": ["beach", "beaches"],
            "shit": ["sheet", "sheets", "shoot"], "cock": ["caulk", "caulking"], "ass": ["as"],
            "dick": ["dic", "dyk"], "tits": ["tit's"], "whore": ["hoar", "hoare"],
        },
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
        "mode": "duck",                # duck (lower the dialogue) | mute | beep
        "duck": 0.1,                   # dialogue level inside a span: 0.1 = -20 dB; also under a beep
        "wave": "sine",                # beep mode: sine | square | triangle | sawtooth | noise | file
        "frequency": 1000,
        "volume": 0.4,                 # beep level, linear 0..1 or a string like "-8dB"
        "file": "",                    # sound file used when wave = "file"
        "loop_file": True,             # loop a short file to fill long spans
        "fade": 0.01,                  # fade in/out seconds for each beep
        "channel": "center",           # center | all
    },
    "output": {
        "suffix": ".clean",
        "container": "mkv",
        "codec": "auto",               # auto | aac | ac3 | eac3 | flac | opus ...
        "bitrate": "auto",
        "title": "",                   # clean track title; empty = "Clean (ducked|muted|beeped)"
        "set_default": True,           # make the clean track the default audio
        "keep_original": True,         # keep the original audio tracks in the output
        "in_place": False,             # rewrite the original file with the clean track added (no second copy)
        "backup": False,               # in-place: keep the original as <name>.orig.<ext>
    },
    "subtitles": {
        "clean": True,                 # also produce censored subtitles
        "style": "asterisks",          # asterisks (s***) | first-letter (s***) | bleep ([BLEEP]) | remove
        "replacement": "[BLEEP]",      # text used by style = bleep
        "sidecar": True,               # write <output>.<lang>.srt next to the clean video
        "embed": True,                 # add a "Clean" subtitle track to the output file
        "set_default": False,          # make the clean subtitle track default
        "standalone": "sidecar",       # non-English audio: sidecar (<file>.<lang>.clean.srt) | remux | skip
        "standalone_suffix": ".clean",
    },
    "plex": {
        "url": "",                     # e.g. "https://plex.example.com:32400"; or env PLEX_URL
        "token": "",                   # X-Plex-Token; prefer env PLEX_TOKEN over writing it here
        "path_map": {},                # {"/media/movies": "/data/movies"} when Plex sees other paths
    },
    "names": {
        "detect": True,                # treat capitalized mid-sentence uses (Dick, Fagin) as names
        "min_occurrences": 2,          # capitalized uses needed before ambiguous spots count as the name
        "ignore": [],                  # words never treated as names
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
audio_only_languages = true # skip the beeped track when the dialogue is not in `languages`
detect_language = true      # Whisper language ID for untagged audio tracks
assume_untagged = "eng"     # when detection is unavailable; "" treats untagged as foreign
skip_ratings = ["G", "TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G"]   # read from Movie.nfo, tvshow.nfo, container tags, or Plex
cache_transcript = true

[asr]
model = "small"             # small is a fine CPU default; medium or large-v3 on a GPU
device = "auto"
compute_type = "auto"
language = "en"

[beep]
mode = "duck"               # duck | mute | beep
duck = 0.1                  # dialogue level inside a span (0.1 = -20 dB, 0.03 = -30 dB)
wave = "sine"               # beep mode only: sine | square | triangle | sawtooth | noise | file
frequency = 1000
volume = 0.4                # 0..1 or "-8dB"
file = ""                   # e.g. "~/sounds/quack.wav" when wave = "file"
loop_file = true
fade = 0.01
channel = "center"          # center | all

[output]
suffix = ".clean"
container = "mkv"
codec = "auto"
bitrate = "auto"
title = ""                  # default "Clean (ducked)" / "Clean (muted)" / "Clean (beeped)"
set_default = true
keep_original = true
in_place = false            # true: add the clean track to the original file instead of Movie.clean.mkv
backup = false              # in-place only: keep Movie.orig.mkv

[subtitles]
clean = true                # write censored subtitles alongside the clean audio
style = "asterisks"         # asterisks | first-letter | bleep | remove
replacement = "[BLEEP]"     # used when style = "bleep"
sidecar = true              # Movie.clean.en.srt next to Movie.clean.mkv
embed = true                # also add a "Clean" subtitle track inside the file
set_default = false
standalone = "sidecar"      # foreign-language audio: sidecar | remux | skip
standalone_suffix = ".clean"

[plex]
# Ratings for the G-rated skip, when your library has no NFO files (Plex never writes them).
# url = "https://plex.example.com:32400"
# token = ""                # or export PLEX_TOKEN=... ; keep this file private (chmod 600)
# [plex.path_map]
# "/media/movies" = "/data/movies"   # local path -> the path Plex sees, if they differ

[names]
detect = true               # "Dick" mid-sentence is a character, "dick" is not
min_occurrences = 2
ignore = []                 # words never treated as names

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
