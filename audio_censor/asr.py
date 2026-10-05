"""Speech recognition with faster-whisper (optional dependency)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from .media import eprint
from .spans import Hit
from .wordlist import Matcher, Token, tokenize

INSTALL_HINT = "speech recognition needs faster-whisper: pip install 'audio-censor[asr]'"


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float = 1.0


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def transcript_path(media: Path) -> Path:
    return media.with_name(media.stem + ".transcript.json")


def load_transcript(path: Path, expect_model: str | None = None) -> list[Word] | None:
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if expect_model and doc.get("model") not in (None, expect_model):
        return None
    return [Word(**w) for w in doc.get("words", [])]


def save_transcript(path: Path, words: list[Word], model: str, audio_track: int) -> None:
    doc = {"model": model, "audio_track": audio_track, "words": [asdict(w) for w in words]}
    path.write_text(json.dumps(doc), encoding="utf-8")


def _resolve_device(device: str, compute_type: str) -> tuple[str, str]:
    if device == "auto":
        device = "cpu"
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                device = "cuda"
        except Exception:
            pass
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


def transcribe(wav: Path, cfg: dict, duration: float = 0.0, progress: bool = True) -> list[Word]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(INSTALL_HINT) from exc

    acfg = cfg.get("asr", {})
    device, compute_type = _resolve_device(acfg.get("device", "auto"), acfg.get("compute_type", "auto"))
    model_name = acfg.get("model", "small")
    eprint(f"  loading whisper model '{model_name}' on {device} ({compute_type}) ...")
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as exc:  # download failure, bad model name, missing CUDA libs ...
        raise RuntimeError(
            f"could not load whisper model {model_name!r} on {device}: {type(exc).__name__}: "
            f"{str(exc).splitlines()[-1] if str(exc) else exc}\n"
            "  (models download from huggingface.co on first use; pass --no-asr to scan subtitles only)"
        ) from exc

    language = acfg.get("language") or None
    segments, _info = model.transcribe(
        str(wav),
        language=language,
        beam_size=int(acfg.get("beam_size", 5)),
        word_timestamps=True,
        vad_filter=bool(acfg.get("vad", True)),
        condition_on_previous_text=False,
        initial_prompt=acfg.get("initial_prompt") or None,
    )
    words: list[Word] = []
    t0 = time.time()
    last_report = t0
    for seg in segments:
        for w in seg.words or []:
            words.append(Word(float(w.start), float(w.end), w.word, float(getattr(w, "probability", 1.0))))
        now = time.time()
        if progress and duration and now - last_report > 5:
            pct = min(100.0, 100.0 * seg.end / duration)
            rate = seg.end / max(1e-6, now - t0)
            eprint(f"  transcribing {pct:5.1f}%  ({rate:.1f}x realtime)", end="\r")
            last_report = now
    if progress:
        eprint(f"  transcribed {len(words)} words in {time.time() - t0:.0f}s        ")
    return words


def scan_words(words: list[Word], matcher: Matcher, detector=None) -> list[Hit]:
    """Match the wordlist against the recognized word sequence.

    Whisper capitalizes proper nouns, so the transcript text is rebuilt and the same
    name classifier used for subtitles decides whether "Dick" is a character.
    """
    text = ""
    spans_: list[tuple[int, int, int]] = []     # (char start, char end, word index)
    for i, w in enumerate(words):
        piece = w.text if w.text.startswith((" ", "\n")) or not text else " " + w.text
        spans_.append((len(text), len(text) + len(piece), i))
        text += piece
    tokens: list[Token] = tokenize(text)
    owners: list[int] = []        # token -> index into words
    j = 0
    for t in tokens:
        while j < len(spans_) - 1 and t.start >= spans_[j][1]:
            j += 1
        owners.append(spans_[j][2])
    hits = []
    for m in matcher.find(tokens):
        first = words[owners[m.start]]
        last = words[owners[m.end - 1]]
        prob = min(words[owners[k]].probability for k in range(m.start, m.end))
        name_use = detector.is_name_use(text, tokens, m) if detector else False
        hits.append(Hit(start=first.start, end=last.end, word=m.text, pattern=m.pattern.raw,
                        tier=m.pattern.tier, source="asr", confidence=prob, name_use=name_use))
    return hits
