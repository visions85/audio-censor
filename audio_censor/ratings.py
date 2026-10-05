"""Find a film's age rating from local metadata, so G-rated titles can be skipped.

Sources, in order: a Kodi / Jellyfin NFO beside the file (Movie.nfo or movie.nfo, <mpaa>
or <certification>), the show's tvshow.nfo for episodes, container tags (iTunes-style
"iTunEXTC", "content_rating" ...), and a Plex server when one is configured.
"""

from __future__ import annotations

import re
from pathlib import Path

KNOWN = {"G", "PG", "PG-13", "R", "NC-17", "NR", "TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G", "TV-PG", "TV-14", "TV-MA",
         "U", "12", "12A", "15", "18"}
DEFAULT_SKIP = ["G", "TV-Y", "TV-Y7", "TV-Y7-FV", "TV-G"]

_NFO_TAG_RE = re.compile(r"<(mpaa|certification|rating_mpaa)>\s*(.*?)\s*</\1>", re.I | re.S)
_TAG_KEYS = ("itunextc", "content_rating", "contentrating", "rating", "rtng", "mpaa", "certification")


def normalize(raw: str) -> str:
    """'Rated PG-13' / 'US:PG-13' / 'mpaa|PG-13|300|' / 'TV-14' -> 'PG-13'."""
    s = (raw or "").strip()
    if not s:
        return ""
    if "|" in s:                                  # iTunEXTC: scheme|rating|score|reasons
        parts = [p for p in s.split("|") if p]
        s = parts[1] if len(parts) > 1 else parts[0]
    s = s.upper().strip()
    aliases = {"NOT RATED": "NR", "UNRATED": "NR", "N/A": "", "NONE": "", "TV-Y7 FV": "TV-Y7-FV"}
    if s in aliases:
        return aliases[s]
    s = s.replace("RATED", "").strip(" :")
    s = re.sub(r"^(US|USA|GB|UK|AU|CA|DE|FR|NZ)\s*[:\-]\s*", "", s)
    s = s.split("/")[0].split(",")[0].strip()
    return aliases.get(s, s)


def from_nfo(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    for _tag, value in _NFO_TAG_RE.findall(text):
        value = re.sub(r"<[^>]+>", "", value)
        norm = normalize(value)
        if norm:
            return norm
    return ""


def nfo_candidates(media: Path) -> list[Path]:
    here = media.parent
    paths = [media.with_suffix(".nfo"), here / "movie.nfo"]
    # episodes: Show/Season 01/ep.mkv -> Show/tvshow.nfo (or one level up again)
    for parent in (here, here.parent):
        paths.append(parent / "tvshow.nfo")
    return [p for p in paths if p.is_file()]


def from_tags(tags: dict) -> str:
    for key, value in (tags or {}).items():
        k = key.lower().split(";")[-1]            # "com.apple.iTunes;iTunEXTC"
        if k in _TAG_KEYS:
            norm = normalize(str(value))
            if norm in KNOWN:
                return norm
    return ""


def find_rating(media: Path, tags: dict | None = None, lookup=None) -> tuple[str, str]:
    """-> (rating or "", where it came from). `lookup(media) -> rating` is an extra source (Plex)."""
    for nfo in nfo_candidates(media):
        r = from_nfo(nfo)
        if r:
            return r, nfo.name
    r = from_tags(tags or {})
    if r:
        return r, "container tags"
    if lookup is not None:
        r = normalize(lookup(media) or "")
        if r:
            return r, "plex"
    return "", ""


def should_skip(rating: str, skip_list: list[str]) -> bool:
    return bool(rating) and rating.upper() in {s.upper() for s in skip_list}
