"""A web page for the library: search Jellyfin, make clean versions, see what was muted.

    audio-censor serve                 # http://<this machine>:8787

Search finds movies and shows through Jellyfin. Each title shows whether it has a clean
version (from the .censor.json and clean audio files beside it); a film's page lists
every mute and every word the context AI judged. "Make clean version" queues a job that
runs `audio-censor process` on the film or the chosen episodes (two jobs at once by
default; the GPU lock keeps their scans apart). Stats count the muted words across every
processed title.

The page talks to this server only: the Jellyfin API key stays here, and posters are
fetched through it. It has no login unless [web] password is set; it is meant for a
home network.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path

from .jellyfin import Jellyfin, JellyfinError, _request
from .media import VIDEO_EXTS, eprint

SEASON_RE = re.compile(r"(?i)^(season|series|specials)\b")


def _state_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "audio-censor"


# --------------------------------------------------------------------------- status

class Library:
    """Jellyfin paths <-> local paths, and what has been done to each file."""

    def __init__(self, cfg: dict, jf: Jellyfin):
        self.cfg = cfg
        self.jf = jf
        # Jellyfin path -> local path, most specific first
        self.to_local = sorted(((r, l) for l, r in jf.path_map.items()), key=lambda rl: -len(rl[0]))
        self._meta: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()
        self.index: set[str] = set()          # every .censor.json found under the library roots
        self.indexed_at = 0.0

    def local(self, remote: str | None) -> Path | None:
        if not remote:
            return None
        for r, l in self.to_local:
            if remote == r or remote.startswith(r + "/"):
                return Path(l + remote[len(r):])
        return Path(remote)

    def remote(self, local: str | Path) -> str:
        s = str(local)
        for l, r in sorted(self.jf.path_map.items(), key=lambda lr: -len(lr[0])):
            if s == l or s.startswith(l + "/"):
                return r + s[len(l):]
        return s

    def roots(self) -> list[Path]:
        """The local folders of Jellyfin's libraries (not every folder on the share)."""
        try:
            libs = self.jf._get("/Library/VirtualFolders")
        except JellyfinError:
            libs = []
        found = {self.local(loc) for lib in libs or [] if lib.get("CollectionType") in ("movies", "tvshows", None)
                 for loc in lib.get("Locations") or []}
        found = sorted(p for p in found if p is not None and p.is_dir())
        # drop folders inside another root
        return [p for p in found if not any(q != p and q in p.parents for q in found)]

    def _sidecar(self, path: Path) -> dict | None:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        key = str(path)
        with self._lock:
            hit = self._meta.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        with self._lock:
            self._meta[key] = (mtime, doc)
            self.index.add(key)
        return doc

    def status(self, media: Path | None, names: set[str] | None = None) -> dict:
        """{"state": clean|nothing|foreign|scanned|none|missing, "mutes": n, "where": beside|in file}"""
        if media is None:
            return {"state": "missing"}
        try:
            names = names if names is not None else set(os.listdir(media.parent))
        except OSError:
            return {"state": "missing"}
        if media.name not in names:
            return {"state": "missing"}
        stem = media.stem
        beside = any(n.startswith(stem + ".Clean (") and n.endswith(".mka") for n in names)
        side = stem + ".censor.json"
        if side not in names:
            return {"state": "clean" if beside else "none", "where": "beside" if beside else None}
        doc = self._sidecar(media.parent / side) or {}
        spans = doc.get("spans", [])
        out = {"mutes": len(spans), "level": doc.get("level"), "scanned": doc.get("generated")}
        if beside or doc.get("rendered_in_place") or (doc.get("rendered_file") in names):
            out.update(state="clean", where="beside" if beside else "in file")
        elif doc.get("audio_censored") is False:
            out.update(state="foreign")
        elif not spans:
            out.update(state="nothing")
        else:
            out.update(state="scanned")
        return out

    def details(self, media: Path) -> dict:
        doc = self._sidecar(media.with_name(media.stem + ".censor.json")) or {}
        return {
            "spans": doc.get("spans", []),
            "context": doc.get("context", []),
            "dub": doc.get("dub", []),
            "names": doc.get("names", []),
            "subtitles": doc.get("subtitles"),
            "asr": doc.get("asr"),
            "rating": doc.get("rating"),
            "language": doc.get("audio_language"),
            "duration": doc.get("duration"),
        }

    # ---- index of everything processed, for stats and the "clean" list

    def reindex(self) -> None:
        found: set[str] = set()
        for root in self.roots():
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if not d.startswith(".") and not d.endswith(".trickplay")]
                for f in filenames:
                    if f.endswith(".censor.json"):
                        found.add(os.path.join(dirpath, f))
        with self._lock:
            self.index = found
            self.indexed_at = time.time()

    def processed(self) -> list[dict]:
        out = []
        with self._lock:
            paths = sorted(self.index)
        for p in paths:
            side = Path(p)
            doc = self._sidecar(side)
            if doc is None:
                continue
            media = side.with_name(side.name[:-len(".censor.json")])
            names = None
            try:
                names = set(os.listdir(side.parent))
            except OSError:
                continue
            video = next((media.with_name(n) for n in names
                          if n.startswith(media.name + ".") and Path(n).suffix.lower() in VIDEO_EXTS
                          and n.count(".") == media.name.count(".") + 1), None)
            st = self.status(video, names) if video else {"state": "missing"}
            show = _tidy(side.parent.parent.name if SEASON_RE.match(side.parent.name) else side.parent.name)
            out.append({"title": show, "file": media.name, "path": str(video or media), "status": st,
                        "spans": doc.get("spans", []), "episode": bool(SEASON_RE.match(side.parent.name))
                        or bool(re.search(r"(?i)\bS\d+E\d+\b", media.name))})
        return out

    def stats(self) -> dict:
        words: Counter = Counter()
        sources: Counter = Counter()
        titles = []
        clean = mutes = estimated = 0
        for item in self.processed():
            if item["status"].get("state") not in ("clean", "nothing"):
                continue                      # scanned but never cleaned: nothing was muted
            clean += 1
            n = 0
            for s in item["spans"]:
                for w in s.get("words", [])[:1]:          # a span's first word names it
                    words[w.lower()] += 1
                for src in s.get("sources", []):
                    sources[src] += 1
                estimated += bool(s.get("estimated"))
                n += 1
            mutes += n
            titles.append({"title": item["title"], "file": item["file"], "mutes": n,
                           "state": item["status"].get("state")})
        titles.sort(key=lambda t: -t["mutes"])
        return {"titles_processed": len(self.index), "titles_clean": clean, "mutes": mutes,
                "estimated": estimated, "words": words.most_common(), "sources": dict(sources),
                "top_titles": titles[:25], "indexed_at": self.indexed_at}


# --------------------------------------------------------------------------- jobs

class Pending:
    """Films corrected since their clean version was made: they need a re-render."""

    def __init__(self):
        self.file = _state_dir() / "web-needs-rerender.json"
        self.lock = threading.Lock()
        try:
            self.paths = set(json.loads(self.file.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            self.paths = set()

    def _save(self) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text(json.dumps(sorted(self.paths)), encoding="utf-8")

    def add(self, path: str) -> None:
        with self.lock:
            self.paths.add(path)
            self._save()

    def done(self, paths: list[str]) -> None:
        with self.lock:
            self.paths -= set(paths)
            self._save()


class Jobs:
    """A queue of `audio-censor process` runs, kept across restarts."""

    def __init__(self, cfg: dict, lib: Library, pending: "Pending | None" = None):
        self.cfg = cfg
        self.lib = lib
        self.pending = pending
        self.dir = _state_dir() / "web-jobs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "jobs.json"
        self.lock = threading.Lock()
        self.jobs: list[dict] = []
        self.procs: dict[str, subprocess.Popen] = {}
        try:
            self.jobs = json.loads(self.file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.jobs = []
        for j in self.jobs:
            if j["state"] == "running":               # the server stopped mid-job: run it again
                j["state"] = "queued"
        self._save()
        workers = int(cfg.get("web", {}).get("workers", 2))
        for _ in range(max(1, workers)):
            threading.Thread(target=self._worker, daemon=True).start()

    def _save(self) -> None:
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.jobs, indent=1), encoding="utf-8")
        tmp.replace(self.file)

    def add(self, title: str, files: list[str], mode: str, redo: bool) -> dict:
        job = {"id": uuid.uuid4().hex[:10], "title": title, "files": files, "mode": mode, "redo": redo,
               "state": "queued", "added": time.time(), "started": None, "finished": None, "exit": None}
        with self.lock:
            self.jobs.append(job)
            self._save()
        return job

    def cancel(self, job_id: str) -> bool:
        with self.lock:
            for j in self.jobs:
                if j["id"] == job_id and j["state"] in ("queued", "running"):
                    if j["state"] == "running" and job_id in self.procs:
                        self.procs[job_id].terminate()
                    j["state"] = "cancelled"
                    j["finished"] = time.time()
                    self._save()
                    return True
        return False

    def clear(self) -> None:
        with self.lock:
            self.jobs = [j for j in self.jobs if j["state"] in ("queued", "running")]
            self._save()

    def log_path(self, job: dict) -> Path:
        return self.dir / f"{job['id']}.log"

    def progress(self, job: dict) -> str:
        """The latest thing the run said (the last line, progress updates included)."""
        try:
            with open(self.log_path(job), "rb") as fh:
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - 4096))
                tail = fh.read().decode("utf-8", "replace")
        except OSError:
            return ""
        lines = [l.strip() for l in re.split(r"[\r\n]", tail) if l.strip()
                 and not re.search(r"Could not find codec|Lossless check|Consider increasing", l)]
        last = lines[-1] if lines else ""
        film = next((l for l in reversed(lines) if re.match(r"^\[\d+/\d+\]", l)), "")
        m = re.search(r"time=([0-9:]+)\S*.*?speed=\s*([0-9.]+x)", last)
        if m:
            last = f"encoding {m.group(1)} ({m.group(2)})"
        return (film.split(":")[0] + " - " if film and len(job["files"]) > 1 else "") + last[:160]

    def snapshot(self) -> list[dict]:
        with self.lock:
            jobs = [dict(j) for j in self.jobs]
        for j in jobs:
            if j["state"] == "running":
                j["progress"] = self.progress(j)
        return jobs[::-1][:100]

    def _next(self) -> dict | None:
        with self.lock:
            for j in self.jobs:
                if j["state"] == "queued":
                    j["state"] = "running"
                    j["started"] = time.time()
                    self._save()
                    return j
        return None

    def _worker(self) -> None:
        while True:
            job = self._next()
            if job is None:
                time.sleep(2)
                continue
            # the same Python as this server (it has Whisper and its CUDA libraries)
            cmd = [sys.executable, "-c", "import sys; from audio_censor.cli import main; sys.exit(main())",
                   "process"]
            if job["mode"] == "beside":
                cmd.append("--audio-sidecar")
            if job["redo"]:
                cmd.append("--overwrite")
            cmd += job["files"]
            with open(self.log_path(job), "w", encoding="utf-8") as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        start_new_session=True)
                with self.lock:
                    self.procs[job["id"]] = proc
                code = proc.wait()
            with self.lock:
                self.procs.pop(job["id"], None)
                if job["state"] == "running":
                    job["state"] = "done" if code == 0 else "failed"
                    if code == 0 and self.pending is not None:
                        self.pending.done(job["files"])
                job["exit"] = code
                job["finished"] = time.time()
                self._save()
            for f in job["files"]:                  # pick the new .censor.json up for stats
                for side in Path(f).parent.glob("*.censor.json") if Path(f).is_file() else Path(f).rglob("*.censor.json"):
                    self.lib._sidecar(side)


# --------------------------------------------------------------------------- HTTP

class App:
    def __init__(self, cfg: dict):
        jf = Jellyfin.from_config(cfg)
        if jf is None:
            raise JellyfinError("the web page needs [jellyfin] url and api_key in the config")
        if not jf.path_map:
            raise JellyfinError("set [jellyfin] path_map (local path -> Jellyfin path) so files can be found")
        self.cfg = cfg
        self.jf = jf
        self.lib = Library(cfg, jf)
        self.pending = Pending()
        self.jobs = Jobs(cfg, self.lib, self.pending)
        self.images: dict[str, tuple[str, bytes]] = {}
        self._libraries: list[tuple[str, str]] | None = None
        threading.Thread(target=self._index_loop, daemon=True).start()

    def _index_loop(self) -> None:
        while True:
            try:
                self.lib.reindex()
            except Exception as exc:  # a share gone away: try again later
                eprint(f"  web: indexing failed: {exc}")
            time.sleep(1800)

    # ---- Jellyfin

    def search(self, q: str, kind: str) -> list[dict]:
        types = {"movie": "Movie", "series": "Series"}.get(kind, "Movie,Series")
        query = {"Recursive": "true", "IncludeItemTypes": types, "Fields": "Path,ProductionYear,OriginalTitle",
                 "Limit": "60", "SortBy": "SortName"}
        if q:
            query["SearchTerm"] = q
        items = self.jf._get("/Items", **query).get("Items", [])
        out = []
        for it in items:
            local = self.lib.local(it.get("Path"))
            entry = {"id": it["Id"], "name": _name(it), "year": it.get("ProductionYear"),
                     "type": it.get("Type"), "library": self._library_of(it.get("Path"))}
            if it.get("Type") == "Movie":
                entry["status"] = self.lib.status(local)
            else:
                entry["status"] = self._series_summary(local)
            out.append(entry)
        return out

    def _library_of(self, remote: str | None) -> str:
        """Its Jellyfin library ("Kids - Weekend"), from the library folders."""
        if self._libraries is None:
            try:
                libs = self.jf._get("/Library/VirtualFolders")
            except JellyfinError:
                libs = []
            self._libraries = sorted(((loc.rstrip("/"), lib.get("Name", "")) for lib in libs or []
                                      for loc in lib.get("Locations") or []), key=lambda x: -len(x[0]))
        remote = remote or ""
        return next((name for loc, name in self._libraries if remote.startswith(loc + "/")), "")

    def _series_summary(self, folder: Path | None) -> dict:
        """Clean episodes / episodes, from the files alone (fast enough for a page of results)."""
        if folder is None or not folder.is_dir():
            return {"state": "missing"}
        total = clean = scanned = 0
        dirs = [folder] + [folder / d for d in os.listdir(folder) if (folder / d).is_dir() and SEASON_RE.match(d)]
        for d in dirs:
            try:
                names = set(os.listdir(d))
            except OSError:
                continue
            for n in names:
                p = d / n
                if p.suffix.lower() in VIDEO_EXTS and ".clean." not in n:
                    total += 1
                    st = self.lib.status(p, names)["state"]
                    clean += st in ("clean", "nothing")
                    scanned += st != "none"
        state = "none" if not scanned else ("clean" if clean == total else "partial")
        return {"state": state, "episodes": total, "clean": clean}

    def item(self, item_id: str) -> dict:
        data = self.jf._get("/Items", Ids=item_id, Fields="Path,ProductionYear,Overview,OfficialRating,OriginalTitle")
        items = data.get("Items", [])
        if not items:
            raise KeyError(item_id)
        it = items[0]
        out = {"id": it["Id"], "name": _name(it), "year": it.get("ProductionYear"), "type": it.get("Type"),
               "overview": it.get("Overview"), "rating": it.get("OfficialRating"), "path": it.get("Path")}
        if it.get("Type") == "Movie":
            local = self.lib.local(it.get("Path"))
            out["local"] = str(local)
            out["status"] = self.lib.status(local)
            out["details"] = self.lib.details(local) if local else {}
            out["needs_rerender"] = str(local) in self.pending.paths
        else:
            eps = self.jf._get(f"/Shows/{item_id}/Episodes", Fields="Path").get("Items", [])
            listings: dict[Path, set[str]] = {}
            seasons: dict[str, list] = {}
            for e in eps:
                local = self.lib.local(e.get("Path"))
                if local is not None and local.parent not in listings:
                    try:
                        listings[local.parent] = set(os.listdir(local.parent))
                    except OSError:
                        listings[local.parent] = set()
                st = self.lib.status(local, listings.get(local.parent) if local else None)
                season = e.get("SeasonName") or f"Season {e.get('ParentIndexNumber') or '?'}"
                seasons.setdefault(season, []).append({
                    "id": e["Id"], "name": e.get("Name"), "season": e.get("ParentIndexNumber"),
                    "episode": e.get("IndexNumber"), "local": str(local) if local else None, "status": st})
            out["seasons"] = [{"name": k, "episodes": v} for k, v in seasons.items()]
        return out

    def processed(self) -> list[dict]:
        """Everything scanned, with what the Cleaned page filters on."""
        out = []
        for p in self.lib.processed():
            spans = p.pop("spans")
            words = sorted({(sp.get("words") or ["?"])[0].lower() for sp in spans})
            out.append(p | {"mutes": len(spans), "words": words, "needs_rerender": p["path"] in self.pending.paths,
                            "estimated": sum(bool(sp.get("estimated")) for sp in spans),
                            "library": self._library_of(self.lib.remote(p["path"]))})
        return out

    def episode(self, local: str) -> dict:
        p = Path(local)
        return {"status": self.lib.status(p), "details": self.lib.details(p),
                "needs_rerender": local in self.pending.paths}

    def correct(self, body: dict) -> dict:
        """Save a correction from the page and flag the film for a re-render."""
        from . import context
        phrase, mute, path = body["phrase"].strip(), bool(body["mute"]), body.get("path") or ""
        try:
            cw, cs, ce = context.parse_phrase(phrase)
        except ValueError as exc:
            raise ValueError(str(exc))
        items = [c for c in context.load_corrections() if context.parse_phrase(c["phrase"]) != (cw, cs, ce)]
        items.append({"phrase": phrase, "mute": mute, "note": body.get("note") or Path(path).parent.name})
        context.save_corrections(items)
        if path:
            self.pending.add(path)
        return {"ok": True, "phrase": phrase, "mute": mute}

    def rerender(self, body: dict) -> dict:
        path = body["path"]
        if not Path(path).exists():
            raise ValueError("that file is no longer there")
        where = self.lib.status(Path(path)).get("where")
        mode = body.get("mode") or ("in file" if where == "in file" else "beside")
        title = _tidy(Path(path).parent.parent.name if SEASON_RE.match(Path(path).parent.name) else Path(path).parent.name)
        return self.jobs.add(f"{title} (re-render)", [path], mode, True)

    def image(self, item_id: str) -> tuple[str, bytes] | None:
        if item_id in self.images:
            return self.images[item_id]
        import urllib.request
        req = urllib.request.Request(f"{self.jf.url}/Items/{item_id}/Images/Primary?fillHeight=360&quality=85",
                                     headers={"Authorization": f'MediaBrowser Client="audio-censor", Token="{self.jf.key}"'})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                found = (r.headers.get("Content-Type", "image/jpeg"), r.read())
        except Exception:
            return None
        if len(self.images) > 600:
            self.images.clear()
        self.images[item_id] = found
        return found

    def make_clean(self, body: dict) -> dict:
        item = self.item(body["id"])
        mode = body.get("mode") or self.cfg.get("web", {}).get("mode", "beside")
        if mode not in ("beside", "in file"):
            raise ValueError("mode must be 'beside' or 'in file'")
        if item["type"] == "Movie":
            files, title = [item["local"]], f"{item['name']} ({item.get('year') or '?'})"
        else:
            want = set(body.get("episodes") or [])
            eps = [e for s in item["seasons"] for e in s["episodes"] if not want or e["id"] in want]
            files = [e["local"] for e in eps if e["local"]]
            title = f"{item['name']}: {len(files)} episode(s)"
        files = [f for f in files if f and Path(f).exists()]
        if not files:
            raise ValueError("no files found for that title on this machine (check [jellyfin] path_map)")
        return self.jobs.add(title, files, mode, bool(body.get("redo")))


def _tidy(folder: str) -> str:
    """'Hook (1991) {imdb-tt0102057}' -> 'Hook (1991)'."""
    return re.sub(r"\s*[\[{](?:imdb|tvdb|tmdb)(?:id)?-[^\]}]*[\]}]", "", folder).strip()


def _name(item: dict) -> str:
    """Its title; an item renamed "#DUPE#" in Jellyfin shows its original title, flagged."""
    name = item.get("Name") or ""
    if name.startswith("#DUPE#") and item.get("OriginalTitle"):
        return f"{item['OriginalTitle']} (marked #DUPE#)"
    return name


def make_handler(app: App, password: str | None):
    page = resources.files("audio_censor").joinpath("data/web/index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "audio-censor"

        def log_message(self, fmt, *args):   # quiet: the page polls every few seconds
            pass

        def _authorized(self) -> bool:
            if not password:
                return True
            got = self.headers.get("Authorization", "")
            if got.startswith("Basic "):
                try:
                    user, _, pw = base64.b64decode(got[6:]).decode().partition(":")
                    if pw == password:
                        return True
                except ValueError:
                    pass
            self.send_response(HTTPStatus.UNAUTHORIZED)
            self.send_header("WWW-Authenticate", 'Basic realm="audio-censor"')
            self.end_headers()
            return False

        def _send(self, body: bytes, ctype: str = "application/json", status: int = 200, cache: bool = False):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200):
            self._send(json.dumps(obj).encode(), status=status)

        def do_GET(self):
            if not self._authorized():
                return
            url = urllib.parse.urlparse(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    return self._send(page, "text/html; charset=utf-8")
                if url.path == "/api/search":
                    return self._json(app.search(q.get("q", "").strip(), q.get("type", "all")))
                if url.path == "/api/item":
                    return self._json(app.item(q["id"]))
                if url.path == "/api/episode":
                    return self._json(app.episode(q["path"]))
                if url.path == "/api/jobs":
                    return self._json(app.jobs.snapshot())
                if url.path == "/api/stats":
                    return self._json(app.lib.stats())
                if url.path == "/api/processed":
                    return self._json(app.processed())
                if url.path == "/img":
                    found = app.image(q.get("id", ""))
                    if not found:
                        return self._send(b"", "image/png", 404)
                    return self._send(found[1], found[0], cache=True)
                self._json({"error": "not found"}, 404)
            except KeyError as exc:
                self._json({"error": f"not found: {exc}"}, 404)
            except JellyfinError as exc:
                self._json({"error": str(exc)}, 502)
            except Exception as exc:
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def do_POST(self):
            if not self._authorized():
                return
            url = urllib.parse.urlparse(self.path)
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                if url.path == "/api/correct":
                    return self._json(app.correct(body))
                if url.path == "/api/rerender":
                    return self._json(app.rerender(body))
                if url.path == "/api/clean":
                    return self._json(app.make_clean(body))
                if url.path == "/api/cancel":
                    return self._json({"ok": app.jobs.cancel(body.get("id", ""))})
                if url.path == "/api/clear":
                    app.jobs.clear()
                    return self._json({"ok": True})
                if url.path == "/api/reindex":
                    threading.Thread(target=app.lib.reindex, daemon=True).start()
                    return self._json({"ok": True})
                self._json({"error": "not found"}, 404)
            except (ValueError, KeyError) as exc:
                self._json({"error": str(exc)}, 400)
            except JellyfinError as exc:
                self._json({"error": str(exc)}, 502)

    return Handler


def serve(cfg: dict, host: str, port: int) -> None:
    app = App(cfg)
    password = cfg.get("web", {}).get("password") or None
    httpd = ThreadingHTTPServer((host, port), make_handler(app, password))
    eprint(f"audio-censor web page on http://{host if host != '0.0.0.0' else _lan_ip()}:{port}"
           + ("" if password else "  (no password: anyone on your network can use it)"))
    httpd.serve_forever()


def _lan_ip() -> str:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "localhost"
    finally:
        s.close()
