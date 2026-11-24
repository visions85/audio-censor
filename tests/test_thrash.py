import json
import os

import pytest

from audio_censor import thrash


@pytest.fixture
def knob(tmp_path, monkeypatch):
    """A fake min_ttl_ms file, written directly instead of through sudo."""
    f = tmp_path / "min_ttl_ms"
    f.write_text("100\n")
    monkeypatch.setattr(thrash, "KNOB", f)
    monkeypatch.setattr(thrash, "_state_path", lambda: tmp_path / "state.json")
    writes = []

    def write(value):
        writes.append(value)
        f.write_text(f"{value}\n")
        return True

    monkeypatch.setattr(thrash, "_write", write)
    return f, writes, tmp_path / "state.json"


def test_lowers_then_restores(knob):
    f, writes, _ = knob
    with thrash.relaxed(True):
        assert f.read_text().strip() == "0"
    assert f.read_text().strip() == "100" and writes == [0, 100]


def test_restores_after_an_error(knob):
    f, _, _ = knob
    with pytest.raises(RuntimeError):
        with thrash.relaxed(True):
            raise RuntimeError("boom")
    assert f.read_text().strip() == "100"


def test_disabled_or_already_off_changes_nothing(knob):
    f, writes, _ = knob
    with thrash.relaxed(False):
        pass
    f.write_text("0\n")
    with thrash.relaxed(True):
        pass
    assert writes == [] and f.read_text().strip() == "0"


def test_another_live_run_keeps_it_down(knob):
    f, _, state = knob
    with thrash.relaxed(True):
        doc = json.loads(state.read_text())
        doc["holders"].append(os.getppid())            # a second run, still alive
        state.write_text(json.dumps(doc))
    assert f.read_text().strip() == "0"                 # the other run will restore it


def test_a_killed_run_is_cleaned_up_by_the_next(knob):
    f, _, state = knob
    f.write_text("0\n")
    state.write_text(json.dumps({"original": 100, "holders": [2 ** 22 + 12345]}))   # dead pid
    with thrash.relaxed(True):
        assert f.read_text().strip() == "0"
    assert f.read_text().strip() == "100" and json.loads(state.read_text()) == {}


def test_without_sudo_it_warns_and_carries_on(knob, monkeypatch, capsys):
    f, _, _ = knob
    monkeypatch.setattr(thrash, "_write", lambda v: False)
    with thrash.relaxed(True):
        pass
    assert "could not lower" in capsys.readouterr().err and f.read_text().strip() == "100"
