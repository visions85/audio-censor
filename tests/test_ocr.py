import shutil
import struct
import subprocess
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from audio_censor import ocr  # noqa: E402
from audio_censor.media import MediaInfo, Stream  # noqa: E402
from audio_censor.subtitles import find_subtitle_source, load_cues  # noqa: E402

# ----------------------------------------------------------------------------- a tiny PGS encoder


def _rle(indices: np.ndarray) -> bytes:
    out = bytearray()
    for row in indices:
        x = 0
        while x < len(row):
            color = int(row[x])
            run = 1
            while x + run < len(row) and row[x + run] == color and run < 16383:
                run += 1
            if color and run < 3:
                out += bytes([color]) * run
            else:
                flag = (0x40 if run > 63 else 0) | (0x80 if color else 0)
                out += bytes([0, flag | (run >> 8 if run > 63 else run)])
                if run > 63:
                    out += bytes([run & 0xFF])
                if color:
                    out += bytes([color])
            x += run
        out += b"\x00\x00"
    return bytes(out)


def _seg(kind: int, t: float, payload: bytes) -> bytes:
    pts = int(t * 90000)
    return b"PG" + struct.pack(">IIBH", pts, 0, kind, len(payload)) + payload


def _caption(t0: float, t1: float, indices: np.ndarray, x: int = 100, y: int = 900) -> bytes:
    h, w = indices.shape
    pcs = struct.pack(">HHBHBBBB", 1920, 1080, 0x10, 1, 0x80, 0, 0, 1) + struct.pack(">HBBHH", 0, 0, 0, x, y)
    # palette: 0 transparent, 1 white text, 2 black outline
    pds = bytes([0, 0]) + bytes([1, 235, 128, 128, 255]) + bytes([2, 16, 128, 128, 255])
    data = _rle(indices)
    ods = struct.pack(">HBB", 0, 0, 0xC0) + (len(data) + 4).to_bytes(3, "big") + struct.pack(">HH", w, h) + data
    clear = struct.pack(">HHBHBBBB", 1920, 1080, 0x10, 2, 0x00, 0, 0, 0)
    return (_seg(ocr.PCS, t0, pcs) + _seg(ocr.PDS, t0, pds) + _seg(ocr.ODS, t0, ods) + _seg(ocr.END, t0, b"")
            + _seg(ocr.PCS, t1, clear) + _seg(ocr.END, t1, b""))


def _box(h=40, w=120) -> np.ndarray:
    a = np.zeros((h, w), dtype=np.uint8)
    a[5:-5, 5:-5] = 2
    a[10:-10, 10:-10] = 1
    return a


# ----------------------------------------------------------------------------- decoding


def test_rle_round_trip():
    a = np.zeros((7, 300), dtype=np.uint8)
    a[1, 3:5] = 1
    a[2, 10:200] = 2          # long coloured run
    a[3, :] = 1
    a[6, 299] = 2
    assert np.array_equal(ocr._decode_rle(_rle(a), 300, 7), a)


def test_parse_sup_timing_and_ink():
    sup = _caption(1.5, 3.25, _box()) + _caption(10.0, 12.0, _box())
    caps = ocr.parse_sup(sup)
    assert [(c.start, c.end) for c in caps] == [(1.5, 3.25), (10.0, 12.0)]
    img = caps[0].image
    # text fill becomes dark, the outline and background white, with a white margin
    assert img.min() == 0 and img[0, 0] == 255
    assert (img < 128).sum() == 20 * 100 * 4   # fill area, scaled 2x each way for short captions


def test_empty_and_transparent_captions_are_dropped():
    blank = np.zeros((20, 50), dtype=np.uint8)
    assert ocr.parse_sup(_caption(0, 1, blank)) == []
    assert ocr.parse_sup(b"") == []


def test_clean_text_fixes_common_ocr_slips():
    assert ocr.clean_text("|'m here.\nl think so  \f") == "I'm here.\nI think so"
    assert ocr.clean_text("look\n\n") == "look"


def _stream(i, codec, lang="eng", title=""):
    return Stream(index=i, type_index=i, codec_type="subtitle", codec_name=codec, language=lang, title=title,
                  default=False)


def test_pick_image_stream_prefers_plain_wanted_language():
    info = MediaInfo(Path("x.mkv"), 10.0, [
        _stream(0, "hdmv_pgs_subtitle", "fre"),
        _stream(1, "hdmv_pgs_subtitle", "eng", "Forced"),
        _stream(2, "hdmv_pgs_subtitle", "eng", "SDH"),
        _stream(3, "hdmv_pgs_subtitle", "eng", "English"),
    ])
    assert ocr.pick_image_stream(info, ["eng", "en"]).type_index == 3
    info.streams = info.streams[:3]
    assert ocr.pick_image_stream(info, ["eng"]).type_index == 2
    assert ocr.pick_image_stream(MediaInfo(Path("x"), 1.0, []), ["eng"]) is None


def test_cached_ocr_becomes_a_subtitle_source(tmp_path):
    media = tmp_path / "Film.mkv"
    media.write_bytes(b"")
    stream = _stream(0, "hdmv_pgs_subtitle")
    info = MediaInfo(media, 60.0, [stream])
    ocr.save_cache(ocr.cache_path(media), stream, "eng", [(1.0, 2.5, "What the hell?\nNo!")])
    work = tmp_path / "work"
    work.mkdir()
    src = find_subtitle_source(media, info, ["eng"], work)
    assert src is not None and "cached OCR" in src.description
    cues = load_cues(src.path)
    assert [(c.start, c.end, c.text) for c in cues] == [(1.0, 2.5, "What the hell? No!")]
    # the cache is not itself a subtitle file a player would pick up
    assert not ocr.cache_path(media).name.endswith((".srt", ".ass", ".vtt"))
    assert find_subtitle_source(media, info, ["eng"], work, ocr=False) is None


def test_cache_ignored_for_another_track(tmp_path):
    media = tmp_path / "Film.mkv"
    ocr.save_cache(ocr.cache_path(media), _stream(0, "hdmv_pgs_subtitle"), "eng", [(1.0, 2.0, "hi")])
    assert ocr.load_cache(ocr.cache_path(media), _stream(2, "hdmv_pgs_subtitle")) is None


# ----------------------------------------------------------------------------- real OCR


def _text_bitmap(text: str, tmp_path: Path) -> np.ndarray:
    """Render white text with ffmpeg and map it onto the PGS palette (1 = fill)."""
    raw = tmp_path / "text.gray"
    proc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=black:s=900x90",
                           "-vf", f"drawtext=text='{text}':fontcolor=white:fontsize=56:x=20:y=15",
                           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", str(raw)],
                          capture_output=True)
    if proc.returncode or not raw.exists():
        pytest.skip("ffmpeg drawtext unavailable")
    a = np.frombuffer(raw.read_bytes(), dtype=np.uint8).reshape(90, 900)
    return np.where(a > 128, 1, 0).astype(np.uint8)


@pytest.mark.skipif(not ocr.available() or shutil.which("ffmpeg") is None, reason="needs tesseract and ffmpeg")
def test_ocr_reads_a_rendered_caption(tmp_path):
    sup = _caption(4.0, 6.0, _text_bitmap("Those are not pillows", tmp_path))
    cues = ocr.ocr_captions(ocr.parse_sup(sup), "eng", tmp_path)
    assert len(cues) == 1
    start, end, text = cues[0]
    assert (start, end) == (4.0, 6.0)
    assert "pillows" in text.lower()
