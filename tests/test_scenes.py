import shutil
import subprocess

import pytest

from audio_censor import scenes
from audio_censor.cli import main
from audio_censor.scenes import Chapter


def test_parse_time_and_range():
    assert scenes.parse_time("1:02:03.5") == 3723.5
    assert scenes.parse_time("62:03") == 3723
    assert scenes.parse_time("45") == 45
    assert scenes.parse_range("1:02:03-1:02:40") == (3723, 3760)
    for bad in ("1:00-0:59", "abc-1:00", "1:00"):
        with pytest.raises(ValueError):
            scenes.parse_range(bad)


def test_merge_inserts_skip_and_resumes_the_interrupted_chapter():
    original = [Chapter(0, "Opening"), Chapter(600, "The Store"), Chapter(1200, "Home")]
    out = scenes.merge_scenes(original, [(700, 760)], duration=1800)
    assert [(c.start, c.title) for c in out] == [
        (0, "Opening"), (600, "The Store"), (700, "Skip"), (760, "The Store"), (1200, "Home")]


def test_merge_drops_chapters_inside_a_scene_and_joins_overlaps():
    original = [Chapter(0, "A"), Chapter(100, "B"), Chapter(200, "C")]
    out = scenes.merge_scenes(original, [(90, 150), (140, 210)], duration=300)
    # B (100) and C (200) start inside the joined scene 90-210; after it, C resumes
    assert [(c.start, c.title) for c in out] == [(0, "A"), (90, "Skip"), (210, "C")]


def test_merge_without_chapters_and_at_the_very_end():
    out = scenes.merge_scenes([], [(10, 20), (290, 300)], duration=300)
    assert [(c.start, c.title) for c in out] == [
        (0, "Chapter 1"), (10, "Skip"), (20, "Chapter 1"), (290, "Skip")]


def test_chapters_xml_escapes_titles():
    xml = scenes.chapters_xml([Chapter(0, "Tom & Jerry <1>"), Chapter(3723.5, "Skip")])
    assert "<ChapterString>Tom &amp; Jerry &lt;1&gt;</ChapterString>" in xml
    assert "<ChapterTimeStart>1:02:03.500</ChapterTimeStart>" in xml


@pytest.mark.skipif(shutil.which("mkvpropedit") is None or shutil.which("ffmpeg") is None,
                    reason="needs mkvtoolnix and ffmpeg")
def test_skip_command_writes_chapters_and_restores_them(tmp_path, capsys):
    film = tmp_path / "Film.mkv"
    meta = tmp_path / "chapters.txt"
    meta.write_text(";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=4000\ntitle=Intro\n"
                    "[CHAPTER]\nTIMEBASE=1/1000\nSTART=4000\nEND=10000\ntitle=Main\n")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=64x64:r=10:d=10",
                    "-i", str(meta), "-map_chapters", "1", "-c:v", "libx264", str(film)], check=True)

    assert main(["skip", str(film), "0:05-0:07", "--note", "kiss"]) == 0
    titles = [(round(c.start), c.title) for c in scenes.read_chapters(film)]
    assert titles == [(0, "Intro"), (4, "Main"), (5, "Skip"), (7, "Main")]
    assert scenes.load(film)["scenes"] == [[5.0, 7.0, "kiss"]]

    assert main(["skip", str(film), "--remove", "1"]) == 0
    assert [(round(c.start), c.title) for c in scenes.read_chapters(film)] == [(0, "Intro"), (4, "Main")]
    assert not scenes.sidecar(film).exists()
    out = capsys.readouterr().out
    assert "1 scene(s) to skip" in out and "0 scene(s) to skip" in out
