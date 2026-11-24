"""Tell a Jellyfin server about a film's new clean track.

Jellyfin picks a film's audio track from the user's preferred language and then the
file's default flag, so once it has re-read the file the clean track (marked default)
plays. What it needs is a refresh after the file changes: this finds the film by its
path, asks Jellyfin to refresh it, and waits until Jellyfin lists the clean track.

Configure [jellyfin] url and api_key (Dashboard -> API Keys), or JELLYFIN_URL and
JELLYFIN_API_KEY in the environment; never store the key in the repository.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .plex import _parts, is_clean_title


class JellyfinError(RuntimeError):
    pass


def _request(url: str, key: str, method: str = "GET", timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, method=method, data=b"" if method == "POST" else None, headers={
        # Jellyfin 12 accepts only this header (X-Emby-Token and ?api_key= get 401).
        "Authorization": f'MediaBrowser Client="audio-censor", Token="{key}"', "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        hint = " (bad API key?)" if exc.code in (401, 403) else ""
        raise JellyfinError(f"Jellyfin returned HTTP {exc.code} for {url.split('?')[0]}{hint}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise JellyfinError(f"could not reach Jellyfin at {url.split('/Items')[0]}: {exc}") from exc
    if not body.strip():
        return {}
    try:
        return json.loads(body.decode("utf-8", errors="surrogateescape"))
    except ValueError as exc:
        raise JellyfinError(f"unreadable response from Jellyfin for {url.split('?')[0]}") from exc


class Jellyfin:
    def __init__(self, url: str, key: str, path_map: dict[str, str] | None = None):
        if not url or not key:
            raise JellyfinError("jellyfin.url and jellyfin.api_key (or JELLYFIN_URL / JELLYFIN_API_KEY) are required")
        if "://" not in url:
            url = "http://" + url
        self.url = url.rstrip("/")
        self.key = key
        self.path_map = {k.rstrip("/"): v.rstrip("/") for k, v in (path_map or {}).items()}
        self._by_tail: dict[tuple[str, ...], list[str]] | None = None

    @classmethod
    def from_config(cls, cfg: dict) -> "Jellyfin | None":
        jcfg = cfg.get("jellyfin", {})
        url = os.environ.get("JELLYFIN_URL") or jcfg.get("url") or ""
        key = os.environ.get("JELLYFIN_API_KEY") or jcfg.get("api_key") or ""
        if not (url and key):
            return None
        return cls(url, key, jcfg.get("path_map") or {})

    def _get(self, path: str, **query) -> dict:
        q = urllib.parse.urlencode(query)
        return _request(f"{self.url}{path}" + (f"?{q}" if q else ""), self.key)

    # ------------------------------------------------------------------ lookup

    def _index(self) -> dict[tuple[str, ...], list[str]]:
        """Folder + file name (and file name alone) -> item ids, for every movie and episode."""
        if self._by_tail is None:
            data = self._get("/Items", Recursive="true", IncludeItemTypes="Movie,Episode", Fields="Path")
            self._by_tail = {}
            for item in data.get("Items", []):
                parts = _parts(item.get("Path") or "")
                for n in (1, 2):
                    if len(parts) >= n:
                        self._by_tail.setdefault(parts[-n:], []).append(item["Id"])
        return self._by_tail

    def item_id(self, media: Path) -> str | None:
        s = str(media)
        for local, remote in self.path_map.items():
            if s.startswith(local + "/"):
                s = remote + s[len(local):]
        parts = _parts(s)
        index = self._index()
        for n in (2, 1):
            ids = set(index.get(parts[-n:], [])) if len(parts) >= n else set()
            if len(ids) == 1:
                return ids.pop()
        return None

    def _audio_streams(self, item_id: str, target: Path) -> list[dict]:
        data = self._get("/Items", Ids=item_id, Fields="MediaSources,MediaStreams,Path")
        want = _parts(str(target))[-1:]
        for item in data.get("Items", []):
            sources = item.get("MediaSources") or [{"Path": item.get("Path"), "MediaStreams": item.get("MediaStreams")}]
            for src in sources:
                if _parts(src.get("Path") or "")[-1:] == want:
                    return [s for s in src.get("MediaStreams") or [] if s.get("Type") == "Audio"]
        return []

    # ------------------------------------------------------------------ clean track

    def refresh_clean(self, media: Path, target: Path | None = None, wait: float = 90.0, poll: float = 3.0) -> str:
        """Refresh the film and wait for Jellyfin to list its clean track. Returns a log line."""
        target = target or media
        item = self.item_id(media) or self.item_id(target)
        if item is None:
            return "not in the Jellyfin library (yet); scan the library, then run 'audio-censor media-select'"
        q = urllib.parse.urlencode({"metadataRefreshMode": "Default", "imageRefreshMode": "Default",
                                    "replaceAllMetadata": "false", "replaceAllImages": "false"})
        _request(f"{self.url}/Items/{item}/Refresh?{q}", self.key, method="POST")
        deadline = time.time() + wait
        while True:
            clean = [s for s in self._audio_streams(item, target) if is_clean_title(s.get("Title"))]
            if clean:
                break
            if time.time() >= deadline:
                return (f"Jellyfin has not picked up the clean track after {int(wait)} s; "
                        "run 'audio-censor media-select' once it has")
            time.sleep(poll)
        s = clean[0]
        if s.get("IsDefault"):
            return f"refreshed; '{s.get('Title')}' is the default track"
        return f"refreshed, but Jellyfin does not see '{s.get('Title')}' as default"
