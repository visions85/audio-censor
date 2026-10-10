import re
import shutil
import subprocess

import pytest

from audio_censor.cli import main, mpv_script_path

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_script_is_packaged():
    text = mpv_script_path().read_text()
    assert "options.read_options(o, \"audio-censor\")" in text
    assert "amix=inputs=2:normalize=0" in text


def test_play_dry_run_scans_and_builds_command(tmp_path, capsys):
    SRT = "1\n00:00:01,000 --> 00:00:03,000\nHello there.\n\n2\n00:00:04,000 --> 00:00:06,000\nOh, shit. That is a damn shame.\n"
    path = tmp_path / "movie.mkv"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=10:d=8",
                    "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000:d=8", "-map", "0:v", "-map", "1:a",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-metadata:s:a:0", "language=eng",
                    "-shortest", str(path)], check=True)
    (tmp_path / "movie.en.srt").write_text(SRT)
    assert main(["play", str(path), "--no-asr", "--level", "mild", "--dry-run", "--frequency", "800"]) == 0
    err = capsys.readouterr().err
    assert "no span file yet, scanning first" in err
    assert (tmp_path / "movie.censor.json").exists()
    cmd = err.split("command:")[1]
    assert "--script=" in cmd and "audio-censor.lua" in cmd
    assert "--script-opts-append=audio-censor-mode=duck" in cmd
    assert "--script-opts-append=audio-censor-duck=0.1" in cmd
    assert "--script-opts-append=audio-censor-frequency=800" in cmd
    assert re.search(r"audio-censor-subs=\S*movie\.en\.clean\.srt", cmd)
    assert "clean subtitles: 2 word(s) masked" in err


def test_install_mpv(tmp_path, capsys):
    assert main(["install-mpv", "--dir", str(tmp_path / "scripts")]) == 0
    assert (tmp_path / "scripts" / "audio-censor.lua").read_text() == mpv_script_path().read_text()
