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
import re
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

    def _get_slow(self, path: str, **query) -> dict:
        q = urllib.parse.urlencode(query)
        return _request(f"{self.url}{path}" + (f"?{q}" if q else ""), self.key, timeout=180.0)

    # ------------------------------------------------------------------ lookup

    def _index(self) -> dict[tuple[str, ...], list[str]]:
        """Folder + file name (and file name alone) -> item ids, for every movie and episode."""
        if self._by_tail is None:
            data = self._get_slow("/Items", Recursive="true", IncludeItemTypes="Movie,Episode", Fields="Path")
            self._by_tail = {}
            for item in data.get("Items", []):
                parts = _parts(item.get("Path") or "")
                for n in (1, 2):
                    if len(parts) >= n:
                        self._by_tail.setdefault(parts[-n:], []).append(item["Id"])
        return self._by_tail

    def item_id(self, media: Path) -> str | None:
        s = str(media)
        for local, remote in sorted(self.path_map.items(), key=lambda kv: -len(kv[0])):
            if s.startswith(local + "/"):
                s = remote + s[len(local):]
                break
        parts = _parts(s)
        # A title search is quick; the full listing (every movie and episode, 60,000+ on a
        # big server) is the fallback for titles the search misses.
        found = self._search(s, parts)
        if found:
            return found
        index = self._index()
        for n in (2, 1):
            ids = set(index.get(parts[-n:], [])) if len(parts) >= n else set()
            if len(ids) == 1:
                return ids.pop()
        return None

    def _search(self, s: str, parts: tuple[str, ...]) -> str | None:
        p = Path(s)
        folder = p.parent.name
        # an episode's show is the folder above its "Season NN" folder
        if re.match(r"(?i)^(season|series|specials)\b", folder) and len(p.parents) > 1:
            folder = p.parent.parent.name
        title = re.sub(r"\s*[\[({].*$", "", folder).strip() or p.stem
        for kind in ("Movie", "Episode"):
            query = {"Recursive": "true", "IncludeItemTypes": kind, "Fields": "Path,MediaSources",
                     "SearchTerm": title} if kind == "Movie" else None
            if query is None:
                series = self._get("/Items", Recursive="true", IncludeItemTypes="Series", SearchTerm=title,
                                   Fields="Path").get("Items", [])
                items = []
                for sr in series[:5]:
                    items += self._get(f"/Shows/{sr['Id']}/Episodes", Fields="Path,MediaSources").get("Items", [])
            else:
                items = self._get("/Items", **query).get("Items", [])
            # A film merged with its edition is one item whose Path is only one of its
            # versions; the others are among its MediaSources.
            paths = {i["Id"]: [i.get("Path")] + [m.get("Path") for m in i.get("MediaSources") or []] for i in items}
            exact = {k for k, ps in paths.items() if s in ps}          # same full path (path_map set)
            ids = exact or {k for k, ps in paths.items() if any(_parts(x or "")[-2:] == parts[-2:] for x in ps)}
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
        # FullRefresh re-reads the file's tracks even when its date is unchanged (an in-place
        # rewrite keeps the old date); replaceAllMetadata=false keeps titles, art and edits.
        q = urllib.parse.urlencode({"metadataRefreshMode": "FullRefresh", "imageRefreshMode": "Default",
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

    # ------------------------------------------------------------------ skip segments

    def skip_segments(self, item: str) -> list[tuple[float, float]]:
        data = self._get(f"/MediaSegments/{item}")
        return [(s["StartTicks"] / 1e7, s["EndTicks"] / 1e7) for s in data.get("Items", [])
                if s.get("Type") == "Commercial"]

    def refresh_skips(self, media: Path, expected: int) -> str:
        """Re-read the film's chapters and start Jellyfin's segment scan; does not wait.

        Jellyfin only builds segments in its library-wide "Media Segment Scan" (about ten
        minutes on a large library) and has no per-film call, so the result shows up later:
        `audio-censor skip FILM` (no ranges) reports what Jellyfin has by then."""
        item = self.item_id(media)
        if item is None:
            return "not in the Jellyfin library (yet)"
        q = urllib.parse.urlencode({"metadataRefreshMode": "FullRefresh", "imageRefreshMode": "Default",
                                    "replaceAllMetadata": "false", "replaceAllImages": "false"})
        _request(f"{self.url}/Items/{item}/Refresh?{q}", self.key, method="POST")
        time.sleep(15)                                   # let the refresh read the new chapters
        how = self.start_segment_scan()
        return (f"chapters refreshed and {how}; the {expected} segment(s) appear when it finishes "
                "(about 15 min); check with: audio-censor skip FILE")

    def start_segment_scan(self) -> str:
        """Start the library-wide Media Segment Scan (a refresh clears a film's segments;
        only this scan rebuilds them)."""
        tasks = self._get("/ScheduledTasks")
        task = next((t for t in (tasks if isinstance(tasks, list) else [])
                     if t.get("Key") == "TaskExtractMediaSegments"), None)
        if task is None:
            return "found no Media Segment Scan task (is the Chapter Segments Provider plugin installed?)"
        if task.get("State") == "Running":
            return "its Media Segment Scan is already running"
        _request(f"{self.url}/ScheduledTasks/Running/{task['Id']}", self.key, method="POST")
        return "started its Media Segment Scan"

    def skip_status(self, media: Path) -> str:
        item = self.item_id(media)
        if item is None:
            return "not in the Jellyfin library"
        found = self.skip_segments(item)
        spans = ", ".join(f"{int(a // 60)}:{int(a % 60):02d}-{int(b // 60)}:{int(b % 60):02d}" for a, b in found)
        return f"{len(found)} skip segment(s) in Jellyfin" + (f": {spans}" if spans else "")

    # ------------------------------------------------------------------ editions

    def add_version(self, original: Path, new_file: Path, wait: float = 180.0, poll: float = 5.0) -> str:
        """Announce `new_file` (beside `original`) and merge it into the original's item as
        a second version. Jellyfin groups versions by itself only when every file name
        starts with the folder name, which download-client names rarely do."""
        orig = self.item_id(original)
        if orig is None:
            return "the original is not in the Jellyfin library"
        data = self._get("/Items", Ids=orig, Fields="Path,MediaSources")
        items = data.get("Items", [])
        paths = [s.get("Path") or "" for i in items for s in i.get("MediaSources") or [{"Path": i.get("Path")}]]
        if any(_parts(p)[-1:] == _parts(str(new_file))[-1:] for p in paths):
            return "already a version of the film"
        jf_path = str(Path(items[0].get("Path") or "").with_name(new_file.name)) if items else ""
        body = json.dumps({"Updates": [{"Path": jf_path, "UpdateType": "Created"}]}).encode()
        req = urllib.request.Request(f"{self.url}/Library/Media/Updated", data=body, method="POST", headers={
            "Authorization": f'MediaBrowser Client="audio-censor", Token="{self.key}"',
            "Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=30).read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise JellyfinError(f"could not notify Jellyfin of {new_file.name}: {exc}") from exc
        deadline = time.time() + wait
        while time.time() < deadline:
            time.sleep(poll)
            self._by_tail = None
            new = self.item_id(new_file)
            if new and new != orig:
                _request(f"{self.url}/Videos/MergeVersions?ids={orig},{new}", self.key, method="POST")
                return "added as a second version of the film"
        return f"Jellyfin has not picked up {new_file.name} yet; merge it with the film by hand"
