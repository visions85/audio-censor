from pathlib import Path

import pytest

import audio_censor.jellyfin as jf
from audio_censor.jellyfin import Jellyfin

FILM = "/movies/Heat (1995)/Heat (1995).mkv"
ORIG = {"Type": "Audio", "Title": "5.1 Surround", "IsDefault": False}
CLEAN = {"Type": "Audio", "Title": "Clean (muted)", "IsDefault": True}


@pytest.fixture
def server(monkeypatch):
    state = {"polls": 0, "posts": [], "after": [ORIG, CLEAN], "items": [
        {"Id": "a1", "Path": FILM}, {"Id": "b2", "Path": "/movies/Up (2009)/Up (2009).mkv"}]}

    def fake(url, key, method="GET", timeout=30.0):
        assert key == "k3y"
        if method == "POST":
            state["posts"].append(url)
            return {}
        if "SearchTerm=" in url:
            return {"Items": state.get("search", [])}
        if "IncludeItemTypes" in url:
            return {"Items": state["items"]}
        if "Ids=a1" in url:
            state["polls"] += 1
            streams = [ORIG] if state["polls"] == 1 else state["after"]
            return {"Items": [{"Id": "a1", "MediaSources": [{"Path": FILM, "MediaStreams": streams}]}]}
        raise AssertionError(url)

    monkeypatch.setattr(jf, "_request", fake)
    monkeypatch.setattr(jf.time, "sleep", lambda s: None)
    return state


def test_refresh_waits_for_the_clean_track(server):
    j = Jellyfin("10.0.0.5:8096", "k3y")
    msg = j.refresh_clean(Path("/mnt/share/Movies/Heat (1995)/Heat (1995).mkv"))
    assert msg == "refreshed; 'Clean (muted)' is the default track"
    assert server["posts"][0].startswith("http://10.0.0.5:8096/Items/a1/Refresh?metadataRefreshMode=FullRefresh")
    assert server["polls"] == 2


def test_reports_when_not_default_or_not_seen(server):
    j = Jellyfin("http://jf", "k3y")
    server["after"] = [ORIG, dict(CLEAN, IsDefault=False)]
    assert "does not see 'Clean (muted)' as default" in j.refresh_clean(Path(FILM))
    server["after"], server["polls"] = [ORIG], 0
    assert "has not picked up" in j.refresh_clean(Path(FILM), wait=0)


def test_unknown_or_ambiguous_film(server):
    j = Jellyfin("http://jf", "k3y")
    assert "not in the Jellyfin library" in j.refresh_clean(Path("/x/Nope/Nope.mkv"))
    server["items"].append({"Id": "c3", "Path": "/other/Heat (1995)/Heat (1995).mkv"})
    j = Jellyfin("http://jf", "k3y")
    assert j.item_id(Path("/mnt/share/Heat (1995)/Heat (1995).mkv")) is None   # two matches: refuse to guess
    assert server["posts"] == []


def test_config_and_env(monkeypatch):
    assert Jellyfin.from_config({"jellyfin": {"url": "", "api_key": ""}}) is None
    monkeypatch.setenv("JELLYFIN_URL", "http://jf:8096")
    monkeypatch.setenv("JELLYFIN_API_KEY", "abc")
    j = Jellyfin.from_config({})
    assert j.url == "http://jf:8096" and j.key == "abc"


def test_falls_back_to_search_when_the_listing_misses_the_film(server):
    hidden = "/movies/Home Alone (1990) {imdb-tt0099785}/Home Alone (1990).mkv"
    server["search"] = [{"Id": "h1", "Path": hidden}, {"Id": "h2", "Path": "/movies/Home Alone 2/x.mkv"}]
    j = Jellyfin("http://jf", "k3y")
    assert j.item_id(Path("/mnt/share/Home Alone (1990) {imdb-tt0099785}/Home Alone (1990).mkv")) == "h1"


def test_add_version_announces_then_merges(monkeypatch):
    calls = {"merged": None, "announced": None}
    orig = "/movies/Film (1989)/Film.1989.Remux.mkv"
    new = "/movies/Film (1989)/Film (1989) {edition-Family}.mkv"
    state = {"listed": False}

    def fake(url, key, method="GET", timeout=30.0):
        if method == "POST" and "MergeVersions" in url:
            calls["merged"] = url.split("ids=")[1]
            return {}
        if "Ids=o1" in url:
            return {"Items": [{"Id": "o1", "Path": orig, "MediaSources": [{"Path": orig}]}]}
        if "IncludeItemTypes" in url:
            items = [{"Id": "o1", "Path": orig}] + ([{"Id": "n2", "Path": new}] if state["listed"] else [])
            return {"Items": items}
        raise AssertionError(url)

    class Resp:
        def read(self):
            return b""

    def fake_urlopen(req, timeout=30):
        calls["announced"] = json.loads(req.data)["Updates"][0]["Path"]
        state["listed"] = True
        return Resp()

    import json
    monkeypatch.setattr(jf, "_request", fake)
    monkeypatch.setattr(jf.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(jf.time, "sleep", lambda s: None)
    j = Jellyfin("http://jf", "k3y")
    msg = j.add_version(Path("/mnt/x/Film (1989)/Film.1989.Remux.mkv"), Path("/mnt/x/Film (1989)/Film (1989) {edition-Family}.mkv"))
    assert msg == "added as a second version of the film"
    assert calls["announced"] == new and calls["merged"] == "o1,n2"
