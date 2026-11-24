"""Read image subtitles (Blu-ray PGS) by OCR, so they can be scanned like text subtitles.

The track is copied out of the container as a .sup file, each caption's bitmap is
decoded here (no image library needed), turned into dark text on a white page and
handed to Tesseract. The resulting cues are cached in <file>.ocr.json, which no media
server mistakes for a subtitle file, so a re-scan does not read the film again.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .media import MediaError, MediaInfo, Stream, language_matches, run

IMAGE_SUB_CODECS = {"hdmv_pgs_subtitle"}
INSTALL_HINT = "install Tesseract to read image subtitles (e.g. pacman -S tesseract tesseract-data-eng)"
CACHE_VERSION = 1

# Tesseract wants ISO 639-2/T codes; tracks are usually tagged that way already.
_TESS_LANG = {"en": "eng", "de": "deu", "fr": "fra", "es": "spa", "it": "ita", "nl": "nld", "pt": "por",
              "sv": "swe", "da": "dan", "no": "nor", "fi": "fin", "ja": "jpn", "ger": "deu", "fre": "fra",
              "dut": "nld"}

PCS, WDS, PDS, ODS, END = 0x16, 0x17, 0x14, 0x15, 0x80


@dataclass
class Caption:
    start: float
    end: float
    image: np.ndarray     # uint8, dark text on white


# ----------------------------------------------------------------------------- availability

def available() -> bool:
    return shutil.which("tesseract") is not None


def tesseract_language(lang: str) -> str:
    """Pick a Tesseract language pack for a track's language tag, falling back to English."""
    lang = (lang or "").lower()
    code = _TESS_LANG.get(lang, lang)
    try:
        out = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True).stdout
        installed = {line.strip() for line in out.splitlines()[1:]}
    except OSError:
        installed = set()
    return code if code in installed else "eng"


def pick_image_stream(info: MediaInfo, languages: list[str]) -> Stream | None:
    """The image subtitle track to read: a full (not forced) track in a wanted language,
    preferring a plain one over SDH (fewer sound cues, same dialogue)."""
    subs = [s for s in info.of_type("subtitle") if s.codec_name in IMAGE_SUB_CODECS]
    title = lambda s: (s.title or "").lower()
    full = [s for s in subs if "forced" not in title(s)]
    wanted = [s for s in full if language_matches(s.language, languages)]
    for group in (wanted, full):
        plain = [s for s in group if "sdh" not in title(s) and "commentary" not in title(s)]
        if plain:
            return plain[0]
        if group:
            return group[0]
    return None


# ----------------------------------------------------------------------------- PGS decoding

def _decode_rle(data: bytes, width: int, height: int) -> np.ndarray:
    """Decode a PGS object's run-length encoded bitmap into palette indices."""
    out = np.zeros((height, width), dtype=np.uint8)
    x = y = i = 0
    n = len(data)
    while i < n and y < height:
        b = data[i]
        i += 1
        if b:
            if x < width:
                out[y, x] = b
            x += 1
            continue
        if i >= n:
            break
        flag = data[i]
        i += 1
        if flag == 0:                       # end of line
            x, y = 0, y + 1
            continue
        count = flag & 0x3F
        if flag & 0x40:
            count = (count << 8) | data[i]
            i += 1
        color = 0
        if flag & 0x80:
            color = data[i]
            i += 1
        if color:
            out[y, x:x + count] = color
        x += count
    return out


def _render(objects: list[tuple[int, int, np.ndarray]], palette: np.ndarray) -> np.ndarray | None:
    """Composite the caption's objects (x, y, indices) into one dark-on-white image."""
    if not objects:
        return None
    x0 = min(x for x, _, _ in objects)
    y0 = min(y for _, y, _ in objects)
    x1 = max(x + a.shape[1] for x, _, a in objects)
    y1 = max(y + a.shape[0] for _, y, a in objects)
    ink = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    for x, y, idx in objects:
        # Text is bright fill with a dark border on a transparent background: ink is
        # brightness times opacity, so the border and background both end up white.
        ink[y - y0:y - y0 + idx.shape[0], x - x0:x - x0 + idx.shape[1]] = palette[idx]
    if ink.max() < 0.2:
        return None
    img = (255 - np.clip(ink / ink.max(), 0, 1) * 255).astype(np.uint8)
    rows = np.where(img.min(axis=1) < 128)[0]
    cols = np.where(img.min(axis=0) < 128)[0]
    if not len(rows) or not len(cols):
        return None
    img = img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    if img.shape[0] < 40:                     # Tesseract likes ~30 px letters
        img = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)
    return np.pad(img, 16, constant_values=255)


def parse_sup(data: bytes) -> list[Caption]:
    """Turn a .sup stream into timed caption images."""
    captions: list[Caption] = []
    palettes: dict[int, np.ndarray] = {}
    pending: dict[int, bytearray] = {}          # object id -> RLE bytes being assembled
    sizes: dict[int, tuple[int, int]] = {}
    objects: dict[int, np.ndarray] = {}
    shown: tuple[float, list[tuple[int, int, int]], int] | None = None   # start, placements, palette
    placements: list[tuple[int, int, int]] = []
    pal_id = 0
    pts = 0.0

    def close(at: float) -> None:
        nonlocal shown
        if shown is None:
            return
        start, places, pid = shown
        shown = None
        pal = palettes.get(pid)
        if pal is None or at <= start:
            return
        objs = [(x, y, objects[o]) for o, x, y in places if o in objects]
        img = _render(objs, pal)
        if img is not None:
            captions.append(Caption(start, at, img))

    pos, n = 0, len(data)
    while pos + 13 <= n:
        if data[pos:pos + 2] != b"PG":
            pos = data.find(b"PG", pos + 1)
            if pos < 0:
                break
            continue
        pts_raw, = struct.unpack_from(">I", data, pos + 2)
        kind = data[pos + 10]
        size, = struct.unpack_from(">H", data, pos + 11)
        seg = data[pos + 13:pos + 13 + size]
        pos += 13 + size
        if kind == PCS and len(seg) >= 11:
            pts = pts_raw / 90000.0
            pal_id, count = seg[9], seg[10]
            placements = []
            off = 11
            for _ in range(count):
                if off + 8 > len(seg):
                    break
                obj, _win, crop, x, y = struct.unpack_from(">HBBHH", seg, off)
                placements.append((obj, x, y))
                off += 16 if crop & 0x40 else 8
            if not count:
                pending.clear()
        elif kind == PDS and len(seg) >= 2:
            pid = seg[0]
            pal = palettes.get(pid, np.zeros(256, dtype=np.float32)).copy()
            for off in range(2, len(seg) - 4, 5):
                idx, y_, _cr, _cb, a = seg[off:off + 5]
                pal[idx] = (y_ / 255.0) * (a / 255.0)
            palettes[pid] = pal
        elif kind == ODS and len(seg) >= 4:
            obj, _ver, seq = struct.unpack_from(">HBB", seg, 0)
            if seq & 0x80 and len(seg) >= 11:
                w, h = struct.unpack_from(">HH", seg, 7)
                sizes[obj] = (w, h)
                pending[obj] = bytearray(seg[11:])
            elif obj in pending:
                pending[obj] += seg[4:]
            if seq & 0x40 and obj in pending and obj in sizes:
                w, h = sizes[obj]
                objects[obj] = _decode_rle(bytes(pending.pop(obj)), w, h)
        elif kind == END:
            close(pts)
            if placements:
                shown = (pts, placements, pal_id)
    close(pts + 5.0)                            # a final caption with no clear: give it a few seconds
    return captions


# ----------------------------------------------------------------------------- OCR

def _write_pgm(img: np.ndarray, path: Path) -> None:
    h, w = img.shape
    path.write_bytes(b"P5\n%d %d\n255\n" % (w, h) + img.tobytes())


_FIXES = [
    (re.compile(r"\|"), "I"),                    # a lone bar is nearly always a capital I
    (re.compile(r"(?<![A-Za-z])l(?=['’][a-z])"), "I"),   # l'm, l'll
    (re.compile(r"(?<![A-Za-z])l(?= )"), "I"),   # "l think"
    (re.compile(r"[ \t]+"), " "),
]


def clean_text(text: str) -> str:
    lines = []
    for line in text.replace("\f", "").splitlines():
        for rx, rep in _FIXES:
            line = rx.sub(rep, line)
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _tesseract_batch(paths: list[Path], lang: str, listfile: Path) -> list[str]:
    """OCR many images in one Tesseract run; pages come back separated by form feeds."""
    listfile.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
    env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
    proc = subprocess.run(["tesseract", str(listfile), "stdout", "-l", lang, "--psm", "6"],
                          capture_output=True, text=True, env=env)
    pages = proc.stdout.split("\f")
    if proc.returncode == 0 and len(pages) >= len(paths) and not "".join(pages[len(paths):]).strip():
        return pages[:len(paths)]
    # Page count did not line up: fall back to one image at a time.
    out = []
    for p in paths:
        one = subprocess.run(["tesseract", str(p), "stdout", "-l", lang, "--psm", "6"],
                             capture_output=True, text=True, env=env)
        out.append(one.stdout)
    return out


def ocr_captions(captions: list[Caption], lang: str, workdir: Path,
                 jobs: int | None = None) -> list[tuple[float, float, str]]:
    if not captions:
        return []
    imgdir = workdir / "ocr"
    imgdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, cap in enumerate(captions):
        p = imgdir / f"{i:05d}.pgm"
        _write_pgm(cap.image, p)
        paths.append(p)
    jobs = max(1, min(jobs or os.cpu_count() or 1, 16, len(paths)))
    chunks = [list(range(k, len(paths), jobs)) for k in range(jobs)]
    texts: list[str] = [""] * len(paths)
    with ThreadPoolExecutor(jobs) as pool:
        futures = [pool.submit(_tesseract_batch, [paths[i] for i in ch], lang, imgdir / f"list{k}.txt")
                   for k, ch in enumerate(chunks)]
        for ch, fut in zip(chunks, futures):
            for i, text in zip(ch, fut.result()):
                texts[i] = text
    cues: list[tuple[float, float, str]] = []
    for cap, text in zip(captions, texts):
        text = clean_text(text)
        if not text:
            continue
        if cues and cues[-1][2] == text and cap.start - cues[-1][1] < 0.1:
            cues[-1] = (cues[-1][0], round(cap.end, 3), text)    # same caption redrawn (palette update)
        else:
            cues.append((round(cap.start, 3), round(cap.end, 3), text))
    return cues


# ----------------------------------------------------------------------------- cache + entry point

def cache_path(media: Path) -> Path:
    return media.with_name(media.stem + ".ocr.json")


def load_cache(path: Path, stream: Stream) -> list[tuple[float, float, str]] | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if doc.get("version") != CACHE_VERSION or doc.get("track") != stream.type_index:
        return None
    return [(float(s), float(e), str(t)) for s, e, t in doc.get("cues", [])]


def save_cache(path: Path, stream: Stream, lang: str, cues: list[tuple[float, float, str]]) -> None:
    doc = {"version": CACHE_VERSION, "track": stream.type_index, "codec": stream.codec_name,
           "title": stream.title, "ocr_language": lang, "cues": cues}
    try:
        path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass                                    # a read-only library still works, just slower next time


def read_image_subtitles(media: Path, stream: Stream, workdir: Path,
                         cache: bool = True) -> tuple[list[tuple[float, float, str]], bool]:
    """Return (cues, from_cache) for an image subtitle track."""
    cpath = cache_path(media)
    if cache:
        cues = load_cache(cpath, stream)
        if cues is not None:
            return cues, True
    if not available():
        raise MediaError(INSTALL_HINT)
    sup = workdir / f"{media.stem}.s{stream.type_index}.sup"
    run(["ffmpeg", "-y", "-v", "error", "-i", str(media), "-map", f"0:s:{stream.type_index}",
         "-c", "copy", "-f", "sup", str(sup)])
    captions = parse_sup(sup.read_bytes())
    lang = tesseract_language(stream.language)
    cues = ocr_captions(captions, lang, workdir)
    if cache:
        save_cache(cpath, stream, lang, cues)
    return cues, False
