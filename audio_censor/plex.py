"""Content ratings from a Plex Media Server.

Plex keeps ratings in its own database and never writes NFO files, so we ask the server
once per run for every movie and episode it knows, with its file paths and
contentRating, and match our files against that index. The token is read from
[plex] token in the config or the PLEX_TOKEN environment variable; never store it in
the repository.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath, PureWindowsPath

from .media import eprint
from .ratings import normalize

CACHE_TTL = 6 * 3600


class PlexError(RuntimeError):
    pass


def _cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "audio-censor" / "plex-ratings.json"


def _fetch_json(url: str, token: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers={"X-Plex-Token": token, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        hint = " (bad token?)" if exc.code == 401 else ""
        raise PlexError(f"Plex returned HTTP {exc.code} for {url.split('?')[0]}{hint}") from exc
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise PlexError(f"could not reach Plex at {url.split('/library')[0]}: {exc}") from exc


def _parts(path: str) -> tuple[str, ...]:
    """Case-folded path components, whatever the separator Plex used."""
    pure = PureWindowsPath(path) if "\\" in path else PurePosixPath(path)
    return tuple(p.lower() for p in pure.parts if p not in ("/", "\\") and not p.endswith(":"))


class PlexRatings:
    def __init__(self, url: str, token: str, path_map: dict[str, str] | None = None, use_cache: bool = True):
        if not url or not token:
            raise PlexError("plex.url and plex.token (or PLEX_URL / PLEX_TOKEN) are required")
        if "://" not in url:
            url = "https://" + url
        self.url = url.rstrip("/")
        self.token = token
        self.path_map = {k.rstrip("/\\"): v.rstrip("/\\") for k, v in (path_map or {}).items()}
        self.use_cache = use_cache
        self._index: dict[str, str] | None = None     # plex file path -> rating
        self._by_tail: dict[tuple[str, ...], list[str]] = {}

    @classmethod
    def from_config(cls, cfg: dict, refresh: bool = False) -> "PlexRatings | None":
        pcfg = cfg.get("plex", {})
        url = os.environ.get("PLEX_URL") or pcfg.get("url") or ""
        token = os.environ.get("PLEX_TOKEN") or pcfg.get("token") or ""
        if not (url and token):
            return None
        return cls(url, token, pcfg.get("path_map") or {}, use_cache=not refresh)

    # ------------------------------------------------------------------ index

    def _build(self) -> dict[str, str]:
        cache = _cache_path()
        if self.use_cache and cache.exists() and time.time() - cache.stat().st_mtime < CACHE_TTL:
            try:
                doc = json.loads(cache.read_text(encoding="utf-8"))
                if doc.get("url") == self.url:
                    return doc["ratings"]
            except (OSError, ValueError, KeyError):
                pass
        eprint(f"  fetching ratings from Plex ({self.url}) ...")
        sections = _fetch_json(f"{self.url}/library/sections", self.token)
        index: dict[str, str] = {}
        for sec in (sections.get("MediaContainer") or {}).get("Directory", []):
            kind = sec.get("type")
            if kind not in ("movie", "show"):
                continue
            item_type = 1 if kind == "movie" else 4           # movies, or episodes
            data = _fetch_json(f"{self.url}/library/sections/{sec['key']}/all?type={item_type}", self.token)
            show_ratings: dict[str, str] = {}
            if kind == "show":                               # episodes inherit the show's rating
                shows = _fetch_json(f"{self.url}/library/sections/{sec['key']}/all?type=2", self.token)
                for sh in (shows.get("MediaContainer") or {}).get("Metadata", []):
                    if sh.get("contentRating"):
                        show_ratings[str(sh.get("ratingKey"))] = sh["contentRating"]
            for item in (data.get("MediaContainer") or {}).get("Metadata", []):
                rating = item.get("contentRating") or show_ratings.get(str(item.get("grandparentRatingKey")), "")
                rating = normalize(rating)
                if not rating:
                    continue
                for media in item.get("Media", []):
                    for part in media.get("Part", []):
                        if part.get("file"):
                            index[part["file"]] = rating
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"url": self.url, "ratings": index}), encoding="utf-8")
        except OSError:
            pass
        return index

    def index(self) -> dict[str, str]:
        if self._index is None:
            self._index = self._build()
            for path, rating in self._index.items():
                parts = _parts(path)
                for n in (1, 2):
                    if len(parts) >= n:
                        self._by_tail.setdefault(parts[-n:], []).append(rating)
        return self._index

    # ------------------------------------------------------------------ lookup

    def _mapped(self, local: Path) -> str:
        s = str(local)
        for local_prefix, plex_prefix in self.path_map.items():
            if s.startswith(local_prefix + "/") or s == local_prefix:
                return plex_prefix + s[len(local_prefix):]
        return s

    def rating_for(self, media: Path) -> str:
        """Exact path (after path_map), else parent-dir + filename, else a unique filename."""
        index = self.index()
        mapped = self._mapped(media.resolve() if media.exists() else media)
        if mapped in index:
            return index[mapped]
        if str(media) in index:
            return index[str(media)]
        parts = _parts(str(media))
        for n in (2, 1):
            hits = self._by_tail.get(parts[-n:], []) if len(parts) >= n else []
            if len(set(hits)) == 1:
                return hits[0]
        return ""
