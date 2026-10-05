from pathlib import Path

from audio_censor.ratings import DEFAULT_SKIP, find_rating, from_nfo, normalize, should_skip


def test_normalize():
    assert normalize("Rated PG-13") == "PG-13"
    assert normalize("US:G") == "G"
    assert normalize("mpaa|TV-Y7|100|") == "TV-Y7"
    assert normalize("Not Rated") == "NR"
    assert normalize("UK:12A") == "12A"
    assert normalize("") == ""


def test_filename_is_not_a_source(tmp_path):
    assert find_rating(tmp_path / "Toy Story (1995) [G].mkv") == ("", "")


def test_nfo_sources(tmp_path):
    (tmp_path / "Up (2009).nfo").write_text("<movie><title>Up</title><mpaa>Rated G</mpaa></movie>")
    assert find_rating(tmp_path / "Up (2009).mkv") == ("G", "Up (2009).nfo")
    show = tmp_path / "Bluey"
    (show / "Season 01").mkdir(parents=True)
    (show / "tvshow.nfo").write_text("<tvshow><mpaa>TV-Y</mpaa></tvshow>")
    assert find_rating(show / "Season 01" / "Bluey S01E01.mkv") == ("TV-Y", "tvshow.nfo")
    # malformed xml with a trailing url line, Kodi style
    nfo = tmp_path / "x.nfo"
    nfo.write_text("<movie><certification>US:PG-13</certification></movie>\nhttps://www.themoviedb.org/movie/1")
    assert from_nfo(nfo) == "PG-13"


def test_container_tags(tmp_path):
    assert find_rating(tmp_path / "m.m4v", {"com.apple.iTunes;iTunEXTC": "mpaa|G|100|"}) == ("G", "container tags")
    assert find_rating(tmp_path / "m.mkv", {"title": "G"}) == ("", "")   # a title is not a rating


def test_should_skip():
    assert should_skip("G", DEFAULT_SKIP) and should_skip("tv-y7", DEFAULT_SKIP)
    assert not should_skip("PG", DEFAULT_SKIP) and not should_skip("", DEFAULT_SKIP)
