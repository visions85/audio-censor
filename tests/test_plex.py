import json
import urllib.parse
from pathlib import Path

import pytest

import audio_censor.plex as plex
from audio_censor.plex import PlexError, PlexRatings
from audio_censor.ratings import find_rating

SECTIONS = {"MediaContainer": {"Directory": [{"key": "1", "type": "movie"}, {"key": "2", "type": "show"},
                                             {"key": "3", "type": "artist"}]}}
MOVIES = {"MediaContainer": {"Metadata": [
    {"title": "Up", "contentRating": "G", "rating": 9.8, "addedAt": 100, "viewCount": 5,
     "Media": [{"Part": [{"file": "/data/Movies/Up (2009)/Up (2009).mkv"}]}]},
    {"title": "Heat", "contentRating": "R", "audienceRating": 9.4, "addedAt": 300,
     "Media": [{"Part": [{"file": "/data/Movies/Heat (1995)/Heat (1995).mkv"}]}]},
    {"title": "Unrated thing", "rating": 4.0, "addedAt": 200, "viewCount": 9,
     "Media": [{"Part": [{"file": "/data/Movies/x/x.mkv"}]}]},
]}}
SHOWS = {"MediaContainer": {"Metadata": [{"ratingKey": "77", "title": "Bluey", "contentRating": "TV-Y", "rating": 9.9}]}}
EPISODES = {"MediaContainer": {"Metadata": [
    {"title": "Magic Xylophone", "grandparentRatingKey": "77",
     "Media": [{"Part": [{"file": "D:\\TV\\Bluey\\Season 01\\Bluey S01E01.mkv"}]}]},
]}}


@pytest.fixture
def server(monkeypatch, tmp_path):
    calls = []

    def fake_fetch(url, token, timeout=30.0):
        calls.append(url)
        assert token == "secret"
        if url.endswith("/library/sections"):
            return SECTIONS
        if "sections/1/all?type=1" in url:
            return MOVIES
        if "sections/2/all?type=2" in url:
            return SHOWS
        if "sections/2/all?type=4" in url:
            return EPISODES
        raise AssertionError(url)

    monkeypatch.setattr(plex, "_fetch_json", fake_fetch)
    monkeypatch.setattr(plex, "_cache_path", lambda: tmp_path / "cache.json")
    return calls


def test_index_and_lookup(server):
    p = PlexRatings("plex.example.com", "secret")
    assert p.url == "https://plex.example.com"
    assert p.rating_for(Path("/data/Movies/Up (2009)/Up (2009).mkv")) == "G"          # exact
    assert p.rating_for(Path("/media/films/Heat (1995)/Heat (1995).mkv")) == "R"      # parent + name
    assert p.rating_for(Path("/elsewhere/Bluey S01E01.mkv")) == "TV-Y"                # unique filename, show rating
    assert p.rating_for(Path("/data/Movies/x/x.mkv")) == ""                           # no rating in Plex
    assert p.rating_for(Path("/nowhere/Unknown.mkv")) == ""
    assert len(server) == 4                                                           # one fetch per endpoint


def test_path_map_and_cache(server, tmp_path):
    p = PlexRatings("https://plex.example.com/", "secret", {"/media/movies": "/data/Movies"})
    assert p.rating_for(Path("/media/movies/Up (2009)/Up (2009).mkv")) == "G"
    cached = json.loads((tmp_path / "cache.json").read_text())
    assert cached["items"]["/data/Movies/Heat (1995)/Heat (1995).mkv"]["content"] == "R"
    # a second instance uses the cache: no new fetches
    n = len(server)
    assert PlexRatings("https://plex.example.com", "secret").rating_for(Path("/data/Movies/Heat (1995)/Heat (1995).mkv")) == "R"
    assert len(server) == n


def test_find_rating_uses_plex_last(server, tmp_path):
    p = PlexRatings("https://plex.example.com", "secret")
    media = tmp_path / "Heat (1995).mkv"
    assert find_rating(media, {}, p.rating_for) == ("R", "plex")
    media.with_suffix(".nfo").write_text("<movie><mpaa>PG-13</mpaa></movie>")
    assert find_rating(media, {}, p.rating_for) == ("PG-13", "Heat (1995).nfo")      # local NFO wins


def test_config_and_env(monkeypatch):
    from audio_censor.config import DEFAULTS, deep_merge
    assert PlexRatings.from_config(DEFAULTS) is None
    monkeypatch.setenv("PLEX_URL", "plex.example.com")
    monkeypatch.setenv("PLEX_TOKEN", "secret")
    p = PlexRatings.from_config(DEFAULTS)
    assert p is not None and p.url == "https://plex.example.com" and p.token == "secret"
    with pytest.raises(PlexError):
        PlexRatings("", "")


def test_order_by_plex_fields(server):
    p = PlexRatings("https://plex.example.com", "secret")
    files = [Path("/data/Movies/x/x.mkv"), Path("/nowhere/Unknown.mkv"), Path("/data/Movies/Heat (1995)/Heat (1995).mkv"),
             Path("/tv/Bluey S01E01.mkv"), Path("/data/Movies/Up (2009)/Up (2009).mkv")]
    ordered, matched = p.order(files, "rating")
    assert [f.name for f in ordered] == ["Bluey S01E01.mkv", "Up (2009).mkv", "Heat (1995).mkv", "x.mkv", "Unknown.mkv"]
    assert matched == 4                                   # Bluey via the show's rating, Heat via audience rating
    ordered, _ = p.order(files, "added")
    assert [f.name for f in ordered][:3] == ["Heat (1995).mkv", "x.mkv", "Up (2009).mkv"]
    ordered, _ = p.order(files, "watched")
    assert [f.name for f in ordered][:2] == ["x.mkv", "Up (2009).mkv"]


def test_order_requires_plex(tmp_path):
    import argparse
    from audio_censor.cli import UserError, order_files, _PLEX
    from audio_censor.config import DEFAULTS
    _PLEX.clear()
    files = [tmp_path / "a.mkv", tmp_path / "b.mkv"]
    assert order_files(files, DEFAULTS, argparse.Namespace(order="name")) == files
    with pytest.raises(UserError):
        order_files(files, DEFAULTS, argparse.Namespace(order="rating", refresh_plex=False))
    _PLEX.clear()


def _item_with_streams(streams, file="/data/Movies/Heat (1995)/Heat (1995).mkv"):
    return {"MediaContainer": {"Metadata": [{"ratingKey": "42", "title": "Heat", "Media": [
        {"Part": [{"id": 900, "file": file, "Stream": streams}]}]}]}}


ORIG = {"id": 1, "streamType": 2, "title": "5.1 Surround", "selected": True}
CLEAN = {"id": 2, "streamType": 2, "title": "Clean (muted)", "default": True}


@pytest.fixture
def plex_api(monkeypatch, tmp_path):
    """A fake Plex: the index, an item whose clean track appears after one refresh poll, and PUTs."""
    movies = json.loads(json.dumps(MOVIES))
    movies["MediaContainer"]["Metadata"][1]["ratingKey"] = "42"
    state = {"polls": 0, "sent": [], "streams_after": [ORIG, CLEAN]}

    def fake_fetch(url, token, timeout=30.0):
        if url.endswith("/library/sections"):
            return {"MediaContainer": {"Directory": [{"key": "1", "type": "movie"}]}}
        if "sections/1/all?type=1" in url:
            return movies
        if url.endswith("/library/metadata/42"):
            state["polls"] += 1
            return _item_with_streams([ORIG] if state["polls"] == 1 else state["streams_after"])
        raise AssertionError(url)

    monkeypatch.setattr(plex, "_fetch_json", fake_fetch)
    monkeypatch.setattr(plex, "_send", lambda url, token, method="PUT", timeout=30.0: state["sent"].append(url))
    monkeypatch.setattr(plex, "_cache_path", lambda: tmp_path / "cache.json")
    monkeypatch.setattr(plex.time, "sleep", lambda s: None)
    return state


def test_select_clean_audio_refreshes_waits_and_selects(plex_api):
    p = PlexRatings("http://plex:32400", "secret")
    msg = p.select_clean_audio(Path("/data/Movies/Heat (1995)/Heat (1995).mkv"))
    assert "selected 'Clean (muted)'" in msg
    assert plex_api["sent"][0] == "http://plex:32400/library/metadata/42/refresh"
    assert plex_api["sent"][1] == "http://plex:32400/library/metadata/42/analyze"
    assert plex_api["sent"][2] == "http://plex:32400/library/parts/900?audioStreamID=2&allParts=1"
    assert plex_api["polls"] == 2          # waited one round for Plex to notice the new track


def test_select_clean_audio_gives_up_and_skips_when_selected(plex_api):
    p = PlexRatings("http://plex:32400", "secret")
    plex_api["streams_after"] = [ORIG]
    msg = p.select_clean_audio(Path("/data/Movies/Heat (1995)/Heat (1995).mkv"), wait=0)
    assert "has not picked up" in msg and len(plex_api["sent"]) == 2   # refresh + analyze only
    plex_api["streams_after"] = [dict(ORIG, selected=False), dict(CLEAN, selected=True)]
    plex_api["polls"] = 1
    assert "already selected" in p.select_clean_audio(Path("/data/Movies/Heat (1995)/Heat (1995).mkv"))


def test_select_clean_audio_unknown_file(plex_api):
    p = PlexRatings("http://plex:32400", "secret")
    assert "not in the Plex library" in p.select_clean_audio(Path("/elsewhere/Nope.mkv"), wait=0)
    assert plex_api["sent"] == []


def test_unknown_file_triggers_a_folder_scan_then_is_found(monkeypatch, tmp_path):
    """A fresh download Plex has not indexed: scan its folder, re-read the index, select."""
    state = {"scanned": [], "indexed": False, "sent": []}
    new_file = "/data/Movies/Uncle Buck (1989)/Uncle Buck NEW.mkv"

    def fake_fetch(url, token, timeout=30.0):
        if url.endswith("/library/sections"):
            return {"MediaContainer": {"Directory": [{"key": "1", "type": "movie",
                                                      "Location": [{"path": "/data/Movies"}]}]}}
        if "sections/1/all?type=1" in url:
            items = [{"ratingKey": "7", "title": "Uncle Buck", "Media": [{"Part": [{"file": new_file}]}]}]
            return {"MediaContainer": {"Metadata": items if state["indexed"] else []}}
        if url.endswith("/library/metadata/7"):
            return _item_with_streams([ORIG, CLEAN], file=new_file)
        raise AssertionError(url)

    monkeypatch.setattr(plex, "_fetch_json", fake_fetch)
    def fake_send(url, token, method="PUT", timeout=30.0):
        if "/refresh?path=" in url:
            assert method == "GET"            # Plex answers a scan with an empty body
            state["scanned"].append(urllib.parse.unquote(url.split("path=")[1]))
            state["indexed"] = True
        state["sent"].append(url)

    monkeypatch.setattr(plex, "_send", fake_send)
    monkeypatch.setattr(plex, "_cache_path", lambda: tmp_path / "cache.json")
    monkeypatch.setattr(plex.time, "sleep", lambda s: None)
    p = PlexRatings("http://plex:32400", "secret")
    msg = p.select_clean_audio(Path("/mnt/share/Movies/Uncle Buck (1989)/Uncle Buck NEW.mkv"))
    assert state["scanned"] == ["/data/Movies/Uncle Buck (1989)"]
    assert "selected 'Clean (muted)'" in msg
