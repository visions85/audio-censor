import os
import shutil
from collections import namedtuple
from types import SimpleNamespace

import pytest

from audio_censor import staging


@pytest.fixture
def share(tmp_path, monkeypatch):
    """A 'network' folder holding a film and its sidecars, and a local staging root."""
    remote = tmp_path / "share" / "Film (1999)"
    remote.mkdir(parents=True)
    film = remote / "Film.mkv"
    film.write_bytes(b"x" * 1000)
    (remote / "Film.en.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n")
    (remote / "Film.transcript.json").write_text("{}")
    (remote / "Other.mkv").write_bytes(b"y")
    os.utime(film, (1_600_000_000, 1_600_000_000))
    root = tmp_path / "local"
    monkeypatch.setattr(staging, "should_stage", lambda media, root, mode: True)
    cfg = {"staging": {"dir": str(root), "mode": "always"}}
    return film, root, cfg


def test_stages_copies_back_new_and_changed_files_and_cleans_up(share):
    film, root, cfg = share
    with staging.staged(film, cfg) as st:
        assert st.staged and st.origin == film
        assert st.local.parent.parent == root and st.local.read_bytes() == film.read_bytes()
        assert (st.local.parent / "Film.en.srt").exists()           # sidecars come along
        assert not (st.local.parent / "Other.mkv").exists()         # other films do not
        (st.local.parent / "Film.clean.mkv").write_bytes(b"clean")  # new output
        (st.local.parent / "Film.transcript.json").write_text('{"words": []}')   # changed cache
    assert (film.parent / "Film.clean.mkv").read_bytes() == b"clean"
    assert (film.parent / "Film.transcript.json").read_text() == '{"words": []}'
    assert film.read_bytes() == b"x" * 1000                         # untouched original stays
    assert not any(root.iterdir())                                  # local copy removed
    assert not list(film.parent.glob(".*partial"))


def test_in_place_rewrite_replaces_the_original(share):
    film, root, cfg = share
    with staging.staged(film, cfg) as st:
        st.local.write_bytes(b"rewritten with a clean track")
    assert film.read_bytes() == b"rewritten with a clean track"


def test_failure_copies_back_small_files_and_keeps_the_film_for_a_retry(share):
    film, root, cfg = share
    with pytest.raises(RuntimeError):
        with staging.staged(film, cfg) as st:
            (st.local.parent / "Film.censor.json").write_text("{}")
            (st.local.parent / "Film.clean.mkv").write_bytes(b"half")     # interrupted render
            raise RuntimeError("boom")
    assert (film.parent / "Film.censor.json").exists()
    assert not (film.parent / "Film.clean.mkv").exists()
    kept = list(root.glob("*/Film.mkv"))
    assert len(kept) == 1
    # a retry reuses the unchanged local copy instead of fetching it again
    calls = []
    real = staging._copy
    staging._copy = lambda s, d: (calls.append(s.name), real(s, d))
    try:
        with staging.staged(film, cfg):
            pass
    finally:
        staging._copy = real
    assert "Film.mkv" not in calls


def test_not_enough_room_works_over_the_network(share, monkeypatch):
    film, root, cfg = share
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: Usage(10, 10, 0))
    with staging.staged(film, cfg) as st:
        assert st.local == film and not st.staged


def test_off_and_dry_run_use_the_original(share):
    film, root, cfg = share
    with staging.staged(film, {"staging": {"dir": ""}}) as st:
        assert st.local == film
    with staging.staged(film, cfg, SimpleNamespace(dry_run=True)) as st:
        assert st.local == film
    with staging.staged(film, cfg, SimpleNamespace(no_stage=True, dry_run=False)) as st:
        assert st.local == film


def test_filesystem_type_reads_proc_mounts(tmp_path):
    assert isinstance(staging.filesystem_type(tmp_path), str)
    assert staging.filesystem_type(tmp_path) not in staging.NETWORK_FS


def test_leftovers_from_a_killed_run_are_cleared(share):
    film, root, cfg = share
    folder = staging._local_folder(root, film)
    folder.mkdir(parents=True)
    import shutil as sh
    sh.copy2(film, folder / film.name)
    (folder / "Film.clean.mkv").write_bytes(b"half-written")
    with staging.staged(film, cfg) as st:
        assert not (st.local.parent / "Film.clean.mkv").exists()
        assert (st.local.parent / "Film.en.srt").exists()
