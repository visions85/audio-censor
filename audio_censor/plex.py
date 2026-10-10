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
CACHE_VERSION = 2
ORDERS = ("name", "rating", "added", "watched")


class PlexError(RuntimeError):
    pass


def _cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "audio-censor" / "plex-ratings.json"


def _fetch_json(url: str, token: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers={"X-Plex-Token": token, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            # Plex passes file names through as raw bytes, and old ones are not always UTF-8;
            # surrogateescape keeps them comparable with what the local filesystem reports.
            return json.loads(resp.read().decode("utf-8", errors="surrogateescape"))
    except urllib.error.HTTPError as exc:
        hint = " (bad token?)" if exc.code == 401 else ""
        raise PlexError(f"Plex returned HTTP {exc.code} for {url.split('?')[0]}{hint}") from exc
    except ValueError as exc:
        raise PlexError(f"unreadable response from Plex for {url.split('?')[0]}: {exc}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise PlexError(f"could not reach Plex at {url.split('/library')[0]}: {exc}") from exc


def _printable(text: str) -> str:
    return text.encode("utf-8", errors="replace").decode("utf-8")


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
        self._index: dict[str, dict] | None = None    # plex file path -> {"content", "score", "added", "views"}
        self._by_tail: dict[tuple[str, ...], list[dict]] = {}

    @classmethod
    def from_config(cls, cfg: dict, refresh: bool = False) -> "PlexRatings | None":
        pcfg = cfg.get("plex", {})
        url = os.environ.get("PLEX_URL") or pcfg.get("url") or ""
        token = os.environ.get("PLEX_TOKEN") or pcfg.get("token") or ""
        if not (url and token):
            return None
        return cls(url, token, pcfg.get("path_map") or {}, use_cache=not refresh)

    # ------------------------------------------------------------------ index

    @staticmethod
    def _score(item: dict) -> float:
        """Critic rating, else audience rating, else the user's own stars; -1 when Plex has none."""
        for key in ("rating", "audienceRating", "userRating"):
            try:
                v = float(item.get(key))
                if v > 0:
                    return v
            except (TypeError, ValueError):
                continue
        return -1.0

    def _build(self) -> dict[str, dict]:
        cache = _cache_path()
        if self.use_cache and cache.exists() and time.time() - cache.stat().st_mtime < CACHE_TTL:
            try:
                doc = json.loads(cache.read_text(encoding="utf-8"))
                if doc.get("url") == self.url and doc.get("version") == CACHE_VERSION:
                    return doc["items"]
            except (OSError, ValueError, KeyError):
                pass
        eprint(f"  fetching library from Plex ({self.url}) ...")
        sections = _fetch_json(f"{self.url}/library/sections", self.token)
        index: dict[str, dict] = {}
        for sec in (sections.get("MediaContainer") or {}).get("Directory", []):
            kind = sec.get("type")
            if kind not in ("movie", "show"):
                continue
            item_type = 1 if kind == "movie" else 4           # movies, or episodes
            data = _fetch_json(f"{self.url}/library/sections/{sec['key']}/all?type={item_type}", self.token)
            shows: dict[str, dict] = {}
            if kind == "show":                               # episodes inherit the show's ratings
                sdata = _fetch_json(f"{self.url}/library/sections/{sec['key']}/all?type=2", self.token)
                for sh in (sdata.get("MediaContainer") or {}).get("Metadata", []):
                    shows[str(sh.get("ratingKey"))] = sh
            for item in (data.get("MediaContainer") or {}).get("Metadata", []):
                show = shows.get(str(item.get("grandparentRatingKey")), {})
                content = normalize(item.get("contentRating") or show.get("contentRating") or "")
                score = self._score(item)
                if score < 0:
                    score = self._score(show)
                entry = {"content": content, "score": score,
                         "added": int(item.get("addedAt") or 0), "views": int(item.get("viewCount") or 0),
                         "title": _printable(item.get("grandparentTitle") or item.get("title") or "")}
                for media in item.get("Media", []):
                    for part in media.get("Part", []):
                        if part.get("file"):
                            index[part["file"]] = entry
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"url": self.url, "version": CACHE_VERSION, "items": index}), encoding="utf-8")
        except OSError:
            pass
        return index

    def index(self) -> dict[str, dict]:
        if self._index is None:
            self._index = self._build()
            for path, entry in self._index.items():
                parts = _parts(path)
                for n in (1, 2):
                    if len(parts) >= n:
                        self._by_tail.setdefault(parts[-n:], []).append(entry)
        return self._index

    # ------------------------------------------------------------------ lookup

    def _mapped(self, local: Path) -> str:
        s = str(local)
        for local_prefix, plex_prefix in self.path_map.items():
            if s.startswith(local_prefix + "/") or s == local_prefix:
                return plex_prefix + s[len(local_prefix):]
        return s

    def info_for(self, media: Path) -> dict | None:
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
            if hits and all(h is hits[0] or h == hits[0] for h in hits):
                return hits[0]
        return None

    def rating_for(self, media: Path) -> str:
        info = self.info_for(media)
        return info["content"] if info else ""

    def order(self, files: list[Path], by: str) -> tuple[list[Path], int]:
        """Sort files by a Plex field, best first; unknown files last. -> (sorted, matched count)."""
        key = {"rating": "score", "added": "added", "watched": "views"}[by]
        decorated, matched = [], 0
        for i, f in enumerate(files):
            info = self.info_for(f)
            value = info[key] if info and info.get(key) is not None else -1
            matched += info is not None and value >= 0
            decorated.append((-value, i, f))
        decorated.sort()
        return [f for _, _, f in decorated], matched
