import shutil
import subprocess

import pytest

from audio_censor import edition, scenes
from audio_censor.cli import main
from audio_censor.media import probe
from audio_censor.scenes import Chapter


def test_plan_cut_and_time_mapping():
    cut = edition.plan_cut([(100, 160), (165, 170)], duration=300)
    assert cut.removed == [(100, 160), (165, 170)] and cut.removed_seconds == 65
    assert cut.keep == [(0, 100), (160, 165), (170, 300)]
    assert cut.map_time(50) == 50
    assert cut.map_time(120) is None and cut.map_time(167) is None
    assert cut.map_time(162) == 102 and cut.map_time(200) == 135
    assert "between(t,100.000,160.000)+between(t,165.000,170.000)" in edition.select_expr(cut)


def test_shift_subtitles_drops_and_moves_cues(tmp_path):
    src = tmp_path / "in.srt"
    src.write_text("1\n00:00:10,000 --> 00:00:12,000\nbefore\n\n"
                   "2\n00:01:45,000 --> 00:01:50,000\ninside\n\n"
                   "3\n00:03:20,000 --> 00:03:22,000\nafter\n", encoding="utf-8")
    cut = edition.plan_cut([(100, 160)], duration=300)
    assert edition.shift_subtitles(src, tmp_path / "out.srt", cut) == 2
    out = (tmp_path / "out.srt").read_text()
    assert "inside" not in out and "00:02:20,000 --> 00:02:22,00" in out


def test_shift_chapters():
    cut = edition.plan_cut([(100, 160)], duration=300)
    chapters = [Chapter(0, "A"), Chapter(120, "B"), Chapter(200, "C")]
    assert [(c.start, c.title) for c in edition.shift_chapters(chapters, cut)] == [(0, "A"), (140, "C")]


def _nvenc() -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    probe_cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=256x256:d=0.2", "-c:v", "hevc_nvenc",
                 "-pix_fmt", "p010le", "-f", "null", "-"]
    return subprocess.run(probe_cmd, capture_output=True).returncode == 0


@pytest.mark.skipif(shutil.which("mkvpropedit") is None or not _nvenc(), reason="needs NVENC and mkvtoolnix")
def test_edition_cuts_video_audio_and_subtitles(tmp_path):
    folder = tmp_path / "Film (2000)"
    folder.mkdir()
    film = folder / "Film.2000.mkv"
    subprocess.run(["ffmpeg", "-y", "-v", "error",
                    "-f", "lavfi", "-i", "testsrc2=s=320x240:r=24:d=20",
                    "-f", "lavfi", "-i", "sine=f=440:d=20:sample_rate=48000",
                    "-map", "0:v", "-map", "1:a", "-map", "1:a",
                    "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params", "log-level=error",
                    "-c:a", "ac3", "-metadata:s:a:1", "title=Clean (muted)", str(film)], check=True)
    (folder / "Film.2000.eng.clean.srt").write_text(
        "1\n00:00:02,000 --> 00:00:03,000\nhello\n\n2\n00:00:06,000 --> 00:00:07,000\ngone\n\n"
        "3\n00:00:15,000 --> 00:00:16,000\nlater\n", encoding="utf-8")
    assert main(["skip", str(film), "0:05-0:10", "--no-media-update"]) == 0
    assert main(["edition", str(film)]) == 0
    out = edition.output_path(film, "Family")
    assert out.name == "Film (2000) {edition-Family}.mkv"
    info = probe(out)
    assert abs(info.duration - 15) < 0.5
    assert info.of_type("video")[0].codec_name == "hevc"
    assert [a.title for a in info.of_type("audio")] == ["Clean (muted)"]
    subs = subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-map", "0:s:0", "-f", "srt", "-"],
                          capture_output=True, text=True).stdout
    assert "hello" in subs and "gone" not in subs and "00:00:10,000 --> 00:00:11,00" in subs
    # the original is back to its own chapters, the scene list is kept for the edition
    assert all(c.title != "Skip" for c in scenes.read_chapters(film))
    assert scenes.load(film)["scenes"] and "Family" in scenes.load(film)["edition"]


def test_folder_expansion_leaves_our_editions_alone(tmp_path):
    from audio_censor.cli import expand_inputs
    folder = tmp_path / "Film (2000)"
    folder.mkdir()
    for name, title in (("Film.2000.mkv", "Main"), ("Film (2000) {edition-Family}.mkv", "Clean (muted)"),
                        ("Film (2000) {edition-Director's Cut}.mkv", "Main")):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=d=1", "-c:a", "ac3",
                        "-metadata:s:a:0", f"title={title}", str(folder / name)], check=True)
    assert sorted(p.name for p in expand_inputs([str(folder)], False)) == [
        "Film (2000) {edition-Director's Cut}.mkv", "Film.2000.mkv"]
