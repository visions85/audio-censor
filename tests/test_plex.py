import json
from pathlib import Path

import pytest

import audio_censor.plex as plex
from audio_censor.plex import PlexError, PlexRatings
from audio_censor.ratings import find_rating

SECTIONS = {"MediaContainer": {"Directory": [{"key": "1", "type": "movie"}, {"key": "2", "type": "show"},
                                             {"key": "3", "type": "artist"}]}}
MOVIES = {"MediaContainer": {"Metadata": [
    {"title": "Up", "contentRating": "G",
     "Media": [{"Part": [{"file": "/data/Movies/Up (2009)/Up (2009).mkv"}]}]},
    {"title": "Heat", "contentRating": "R",
     "Media": [{"Part": [{"file": "/data/Movies/Heat (1995)/Heat (1995).mkv"}]}]},
    {"title": "Unrated thing", "Media": [{"Part": [{"file": "/data/Movies/x/x.mkv"}]}]},
]}}
SHOWS = {"MediaContainer": {"Metadata": [{"ratingKey": "77", "title": "Bluey", "contentRating": "TV-Y"}]}}
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
    assert cached["ratings"]["/data/Movies/Heat (1995)/Heat (1995).mkv"] == "R"
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
