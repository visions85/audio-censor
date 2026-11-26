import json

from audio_censor.jellyfin import Jellyfin
from audio_censor.web import Library, _name, _tidy


def _lib():
    jf = Jellyfin("http://jf", "k", {"/mnt/share/Movies": "/movies", "/mnt/share": "/shares"})
    return Library({}, jf)


def test_paths_map_back_to_this_machine():
    lib = _lib()
    assert str(lib.local("/movies/Up (2009)/Up.mkv")) == "/mnt/share/Movies/Up (2009)/Up.mkv"
    assert str(lib.local("/shares/Kids/Show/ep.mkv")) == "/mnt/share/Kids/Show/ep.mkv"


def test_status_from_the_files_beside_a_film(tmp_path):
    lib = _lib()
    film = tmp_path / "Up.mkv"
    film.write_bytes(b"")
    assert lib.status(film)["state"] == "none"
    side = tmp_path / "Up.censor.json"
    side.write_text(json.dumps({"spans": [{"start": 1, "end": 2, "words": ["damn"]}], "audio_censored": True}))
    assert lib.status(film) | {} == lib.status(film) and lib.status(film)["state"] == "scanned"
    (tmp_path / "Up.Clean (muted).eng.default.mka").write_bytes(b"")
    st = lib.status(film)
    assert st["state"] == "clean" and st["where"] == "beside" and st["mutes"] == 1
    side.write_text(json.dumps({"spans": [], "audio_censored": True}))
    (tmp_path / "Up.Clean (muted).eng.default.mka").unlink()
    assert lib.status(film)["state"] == "nothing"
    assert lib.status(tmp_path / "Gone.mkv")["state"] == "missing"


def test_stats_count_words(tmp_path):
    lib = _lib()
    d = tmp_path / "Up (2009) {imdb-tt1049413}"
    d.mkdir()
    (d / "Up.mkv").write_bytes(b"")
    (d / "Up.Clean (muted).eng.default.mka").write_bytes(b"")
    (d / "Up.censor.json").write_text(json.dumps({"spans": [
        {"start": 1, "end": 2, "words": ["damn"], "sources": ["asr"]},
        {"start": 5, "end": 6, "words": ["son of a bitch", "bitch"], "sources": ["asr", "subtitle"], "estimated": True}]}))
    lib.index = {str(d / "Up.censor.json")}
    s = lib.stats()
    assert s["titles_clean"] == 1 and s["mutes"] == 2 and s["estimated"] == 1
    assert dict(s["words"]) == {"damn": 1, "son of a bitch": 1}
    assert s["top_titles"][0]["title"] == "Up (2009)"


def test_names():
    assert _tidy("Dragon Ball (1986) [tvdb-76666]") == "Dragon Ball (1986)"
    assert _name({"Name": "#DUPE#", "OriginalTitle": "Pingu"}) == "Pingu (marked #DUPE#)"
