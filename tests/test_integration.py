"""End-to-end test against a synthetic film. Needs ffmpeg on PATH."""

import re
import shutil
import subprocess

import pytest

from audio_censor.cli import main
from audio_censor.media import probe

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

SRT = """1
00:00:01,000 --> 00:00:03,000
Hello there, how are you today?

2
00:00:04,000 --> 00:00:06,000
Oh, shit. That is a damn shame.
"""


def rms_db(path, track, start, end, channel):
    out = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(path), "-map", f"0:a:{track}",
         "-af", f"atrim={start}:{end},pan=mono|c0={channel},astats=measure_perchannel=RMS_level:measure_overall=none",
         "-f", "null", "-"], capture_output=True, text=True).stderr
    m = re.search(r"RMS level dB: (-?[\d.]+|-inf)", out)
    return float(m.group(1)) if m else float("-inf")


@pytest.fixture
def movie(tmp_path):
    path = tmp_path / "movie.mkv"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=10:d=8",
         "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000:d=8",
         "-filter_complex", "[1:a]pan=5.1(side)|FL=0.3*c0|FR=0.3*c0|FC=c0|SL=0.2*c0|SR=0.2*c0[a]",
         "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "ac3",
         "-metadata:s:a:0", "language=eng", "-shortest", str(path)], check=True)
    (tmp_path / "movie.en.srt").write_text(SRT)
    return path


def test_process_adds_beeped_clean_track(movie, tmp_path):
    rc = main(["process", str(movie), "--no-asr", "--level", "mild", "--mode", "beep", "--duck", "0",
               "--wave", "sine", "--frequency", "1000"])
    assert rc == 0
    out = movie.with_name("movie.clean.mkv")
    assert out.exists()
    info = probe(out)
    audio = info.of_type("audio")
    assert len(audio) == 2
    assert audio[1].title == "Clean (beeped)" and audio[1].default and not audio[0].default
    assert audio[1].channel_layout == "5.1(side)"

    # untouched region matches the original
    assert abs(rms_db(out, 1, 1.0, 3.0, "FC") - rms_db(movie, 0, 1.0, 3.0, "FC")) < 0.5
    # inside the span: original 220 Hz tone is gone from the center, beep present, sides silent
    assert rms_db(out, 1, 4.3, 5.5, "SL") < -60
    assert rms_db(out, 1, 4.3, 5.5, "FC") > -15

    sidecar = movie.with_name("movie.censor.json")
    assert sidecar.exists()
    assert "shit" in sidecar.read_text() and "damn" in sidecar.read_text()

    # clean subtitles: sidecar file next to the output and an embedded "Clean" track
    clean_srt = movie.with_name("movie.clean.en.srt")
    assert clean_srt.exists()
    text = clean_srt.read_text()
    assert "Oh, ****. That is a **** shame." in text and "shit" not in text
    subs = info.of_type("subtitle")
    assert len(subs) == 1 and subs[0].title == "Clean" and subs[0].language == "en" and not subs[0].default


def test_mute_mode_and_render_from_sidecar(movie, tmp_path):
    assert main(["scan", str(movie), "--no-asr", "--level", "mild"]) == 0
    out = tmp_path / "muted.mkv"
    assert main(["render", str(movie), "--mode", "mute", "-o", str(out), "--sub-style", "bleep",
                 "--sub-replacement", "[bleep]"]) == 0
    assert "Oh, [bleep]. That is a [bleep] shame." in (tmp_path / "muted.en.srt").read_text()
    audio = probe(out).of_type("audio")
    assert audio[1].title == "Clean (muted)"
    assert rms_db(out, 1, 4.3, 5.5, "FC") < -60
    assert rms_db(out, 1, 1.0, 3.0, "FC") > -25


def test_nothing_found_skips_render(movie):
    assert main(["process", str(movie), "--no-asr", "--level", "strong"]) == 0
    assert not movie.with_name("movie.clean.mkv").exists()


def test_no_clean_subs_flag(movie, tmp_path):
    out = tmp_path / "nosubs.mkv"
    assert main(["process", str(movie), "--no-asr", "--level", "mild", "--no-clean-subs", "-o", str(out)]) == 0
    assert not (tmp_path / "nosubs.en.srt").exists()
    assert probe(out).of_type("subtitle") == []


@pytest.fixture
def french_movie(movie, tmp_path):
    path = tmp_path / "french.mkv"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(movie), "-map", "0", "-c", "copy",
                    "-metadata:s:a:0", "language=fre", str(path)], check=True)
    (tmp_path / "french.en.srt").write_text(SRT)
    return path


def test_foreign_audio_gets_subtitles_only(french_movie, tmp_path):
    assert main(["process", str(french_movie), "--no-asr", "--level", "mild"]) == 0
    assert not (tmp_path / "french.clean.mkv").exists()
    side = tmp_path / "french.en.clean.srt"
    assert side.exists() and "Oh, ****. That is a **** shame." in side.read_text()
    meta = (tmp_path / "french.censor.json").read_text()
    assert '"audio_censored": false' in meta and '"audio_language": "fre"' in meta
    # second run: the clean sidecar must not be mistaken for the source, and is skipped
    assert main(["process", str(french_movie), "--no-asr", "--level", "mild"]) == 0
    assert sorted(p.name for p in tmp_path.glob("french*.srt")) == ["french.en.clean.srt", "french.en.srt"]


def test_foreign_audio_remux_mode(french_movie, tmp_path):
    assert main(["process", str(french_movie), "--no-asr", "--level", "mild", "--standalone-subs", "remux"]) == 0
    out = tmp_path / "french.clean.mkv"
    info = probe(out)
    assert [a.title for a in info.of_type("audio")] == [""]                    # original audio only, untouched
    subs = info.of_type("subtitle")
    assert len(subs) == 1 and subs[0].title == "Clean" and subs[0].language == "en"
    assert (tmp_path / "french.clean.en.srt").exists()


def test_any_language_forces_audio_censoring(french_movie, tmp_path):
    assert main(["process", str(french_movie), "--no-asr", "--level", "mild", "--any-language"]) == 0
    assert len(probe(tmp_path / "french.clean.mkv").of_type("audio")) == 2


def test_batch_skips_existing_and_summarizes(movie, tmp_path, capsys):
    assert main(["process", str(tmp_path), "--no-asr", "--level", "mild"]) == 0
    assert main(["process", str(tmp_path), "--no-asr", "--level", "mild"]) == 0
    err = capsys.readouterr().err
    assert "movie.clean.mkv exists, skipping" in err


def test_two_phase_scan_then_render(movie, french_movie, tmp_path, capsys):
    # phase one: scan everything, writes span files only
    assert main(["scan", str(tmp_path), "--no-asr", "--level", "mild"]) == 0
    assert (tmp_path / "movie.censor.json").exists() and (tmp_path / "french.censor.json").exists()
    assert not list(tmp_path.glob("*.clean.*"))
    out = capsys.readouterr().out
    assert "2 file(s): 1 scanned, 1 subtitles only" in out

    assert main(["status", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "movie.mkv" in out and "scanned" in out and "2 file(s): 2 scanned" in out

    # scanning again is a no-op
    assert main(["scan", str(tmp_path), "--no-asr", "--level", "mild"]) == 0
    assert "2 already scanned" in capsys.readouterr().out

    # phase two: render from the span files, with a beep chosen now
    assert main(["render", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "2 file(s): 1 rendered, 1 subtitles only" in out
    assert (tmp_path / "movie.clean.mkv").exists() and (tmp_path / "french.en.clean.srt").exists()
    assert probe(tmp_path / "movie.clean.mkv").of_type("audio")[1].title == "Clean (ducked)"
    # default mode ducks the dialogue to 0.1 (-20 dB) instead of beeping
    assert abs(rms_db(tmp_path / "movie.clean.mkv", 1, 4.3, 5.5, "FC") - (rms_db(movie, 0, 4.3, 5.5, "FC") - 20)) < 1.5

    assert main(["status", str(tmp_path)]) == 0
    assert "2 file(s): 2 rendered" in capsys.readouterr().out

    # rendering again skips everything
    assert main(["render", str(tmp_path)]) == 0
    assert "2 skipped" in capsys.readouterr().out


def test_render_batch_skips_unscanned(movie, tmp_path, capsys):
    assert main(["render", str(tmp_path)]) == 0
    assert "1 file(s): 1 not scanned" in capsys.readouterr().out


def test_g_rated_film_is_skipped(movie, tmp_path, capsys):
    movie.with_suffix(".nfo").write_text("<movie><mpaa>Rated G</mpaa></movie>")
    assert main(["process", str(tmp_path), "--no-asr", "--level", "mild"]) == 0
    assert "1 skipped (rating)" in capsys.readouterr().out
    assert not (tmp_path / "movie.clean.mkv").exists()
    assert '"skipped": "rated G"' in (tmp_path / "movie.censor.json").read_text()
    assert main(["status", str(tmp_path)]) == 0
    assert "nothing to do  rated G" in capsys.readouterr().out
    # override
    assert main(["process", str(movie), "--no-asr", "--level", "mild", "--ignore-rating"]) == 0
    assert (tmp_path / "movie.clean.mkv").exists()


def test_batch_survives_unexpected_exception(movie, french_movie, tmp_path, capsys, monkeypatch):
    import audio_censor.cli as cli
    real = cli.scan_file

    def flaky(media, cfg, args, position=""):
        if media.name == "french.mkv":
            raise TypeError("open() got an unexpected keyword argument 'metadata_errors'")
        return real(media, cfg, args, position)

    monkeypatch.setattr(cli, "scan_file", flaky)
    assert main(["scan", str(tmp_path), "--no-asr", "--level", "mild"]) == 1      # non-zero: something failed
    out, err = capsys.readouterr()
    assert "error: french.mkv: TypeError: open() got an unexpected keyword" in err
    assert "2 file(s): 1 scanned, 1 failed" in out
    assert (tmp_path / "movie.censor.json").exists()                               # the other file was still done


def test_in_place_adds_track_to_original(movie, tmp_path, capsys):
    (tmp_path / "movie.nfo").unlink(missing_ok=True)
    size_before = movie.stat().st_size
    assert main(["process", str(movie), "--no-asr", "--level", "mild", "--in-place"]) == 0
    assert not (tmp_path / "movie.clean.mkv").exists() and not (tmp_path / "movie.clean.tmp.mkv").exists()
    info = probe(movie)
    audio = info.of_type("audio")
    assert [a.title for a in audio] == ["", "Clean (ducked)"] and audio[1].default and not audio[0].default
    assert [x.title for x in info.of_type("subtitle")] == ["Clean"]
    assert movie.stat().st_size > size_before
    # the clean subtitle sidecar must not clobber the original movie.en.srt
    assert (tmp_path / "movie.en.srt").read_text() == SRT
    assert "Oh, ****." in (tmp_path / "movie.en.clean.srt").read_text()
    meta = (tmp_path / "movie.censor.json").read_text()
    assert '"rendered_in_place": true' in meta

    # second run: skipped, no duplicate track
    assert main(["process", str(tmp_path), "--no-asr", "--level", "mild", "--in-place"]) == 0
    assert "already rendered in place, skipping" in capsys.readouterr().err
    assert len(probe(movie).of_type("audio")) == 2
    assert main(["status", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "rendered" in out and "in place" in out

    # --overwrite replaces the clean track instead of stacking another
    assert main(["process", str(movie), "--no-asr", "--level", "mild", "--in-place", "--overwrite", "--mode", "mute"]) == 0
    audio = probe(movie).of_type("audio")
    assert [a.title for a in audio] == ["", "Clean (muted)"]
    assert len(probe(movie).of_type("subtitle")) == 1


def test_in_place_mp4_becomes_mkv(movie, tmp_path):
    mp4 = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(movie), "-map", "0:v", "-map", "0:a", "-c", "copy",
                    "-metadata:s:a:0", "language=eng", str(mp4)], check=True)
    (tmp_path / "clip.en.srt").write_text(SRT)
    assert main(["process", str(mp4), "--no-asr", "--level", "mild", "--in-place", "--backup"]) == 0
    assert not mp4.exists() and (tmp_path / "clip.mkv").exists() and (tmp_path / "clip.orig.mp4").exists()
    assert len(probe(tmp_path / "clip.mkv").of_type("audio")) == 2


def test_in_place_verification_failure_keeps_original(movie, tmp_path, monkeypatch, capsys):
    import audio_censor.cli as cli
    real = cli.finalize_in_place

    def too_strict(media, tmp, expect_audio, expect_subs, backup=False):
        return real(media, tmp, expect_audio + 5, expect_subs, backup)      # impossible expectation

    monkeypatch.setattr(cli, "finalize_in_place", too_strict)
    before = movie.read_bytes()
    assert main(["scan", str(movie), "--no-asr", "--level", "mild"]) == 0
    assert main(["render", str(movie), "--in-place"]) == 1
    assert "in-place verification failed, original left untouched" in capsys.readouterr().err
    assert movie.read_bytes() == before
    assert not (tmp_path / "movie.clean.tmp.mkv").exists()
    assert '"rendered_in_place"' not in (tmp_path / "movie.censor.json").read_text()
