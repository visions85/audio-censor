"""Dub worker (F5-TTS): run with the Python of the dub model environment, not audio-censor's own.

    <dub python> dub_worker.py JOB.json

For every muted word in the job it separates the voice from the centre channel (Demucs),
then has F5-TTS regenerate just that word inside the actor's own line (speech editing:
the audio around it is kept, so the new word carries on the real voice, pace and room),
once per replacement word and seed. Each take is written as the change to make to the
centre channel over the word: new voice minus old voice, faded in and out. audio-censor
then checks the takes with Whisper and keeps the best (see dub.py).

Two lighter jobs serve the other engines (dub_pd.py, dub_auk.py), whose environments have
no Demucs or speaker model: "mode": "separate" only separates the voice of each item, and
"mode": "voice" only scores takes against the actor's own voice, so takes from every
engine are ranked the same way.

The helpers below (separate, frame, layout, finish, match_tone, naturalness) are shared by
those workers. Only third-party packages are imported here (torch, torchaudio, demucs,
f5_tts, numpy, soundfile, librosa, speechbrain): this file runs outside audio-censor's
environment.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


def setup(job: dict) -> None:
    cache = job.get("cache") or ""
    if cache:                                   # model downloads live beside the environment
        os.environ.setdefault("HF_HOME", str(Path(cache) / "hf"))
        os.environ.setdefault("CACHED_PATH_CACHE_ROOT", str(Path(cache) / "cached-path"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")      # its downloader can exhaust RAM


def separate(separator, item: dict, rate: int):
    """The voice alone in the centre (and front left/right when the item has them)."""
    import numpy as np
    import librosa
    import torch
    win = np.load(item["window"]).astype(np.float32)

    def vocals(x):                              # (2, n): Demucs works in stereo at 44.1 kHz
        _, st = separator.separate_tensor(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), rate)
        v = st["vocals"].cpu().numpy()
        if separator.samplerate != rate:
            v = np.stack([librosa.resample(c, orig_sr=separator.samplerate, target_sr=rate) for c in v])
        return np.pad(v, ((0, 0), (0, max(0, x.shape[1] - v.shape[1]))))[:, :x.shape[1]]

    voc = vocals(np.stack([win, win])).mean(0).astype(np.float32)
    sides = None
    if item.get("sides"):
        sides = vocals(np.load(item["sides"]).astype(np.float32).T).astype(np.float32)   # (2, n): FL, FR
    return voc, sides


def load_voices(item: dict):
    """The separated voice saved by a "separate" job: (centre, sides or None)."""
    import numpy as np
    voc = np.load(item["voice"]).astype(np.float32)
    sides = np.load(item["voice_sides"]).astype(np.float32) if item.get("voice_sides") else None
    return voc, sides


def layout(item: dict, voc, sides, rate: int, xfade: float):
    """(a0, a1, keep, vcrop, gains): the crop in samples, what no take touches, the side gains."""
    import numpy as np
    c0, c1 = item["crop"]
    edits = item.get("edits") or {}
    spans = [edits.get(a, item["edit"]) for a in item["texts"]]
    u0, u1 = min(e[0] for e in spans), max(e[1] for e in spans)    # every take's edit
    a0, a1 = int(c0 * rate), min(int(c1 * rate), len(voc))
    keep = np.ones(a1 - a0, bool)
    keep[max(0, int((u0 - xfade) * rate)):int((u1 + xfade) * rate)] = False
    vcrop = voc[a0:a1]
    gains = None
    if sides is not None:          # how much of the centre's voice each side plays
        vc = vcrop[keep]
        gains = [float(np.clip(np.dot(sv[a0:a1][keep], vc) / max(float(np.dot(vc, vc)), 1e-9), 0.0, 1.5))
                 for sv in sides]
    return a0, a1, keep, vcrop, gains


def frame(item: dict, alt: str, rate: int, xfade: float, n: int):
    """(e0, e1, r0, r1, fade): one replacement's edit (seconds in the crop), the window samples
    that change, and the fade over them."""
    import numpy as np
    c0 = item["crop"][0]
    e0, e1 = (item.get("edits") or {}).get(alt, item["edit"])
    r0, r1 = max(int((c0 + e0 - xfade) * rate), 0), min(int((c0 + e1 + xfade) * rate), n)
    fade = np.ones(r1 - r0, np.float32)
    k = max(1, min(int(xfade * rate), (r1 - r0) // 2))
    fade[:k] = np.linspace(0, 1, k)
    fade[-k:] = np.linspace(1, 0, k)
    return e0, e1, r0, r1, fade


def finish(wav, voc, sides, gains, keep, vcrop, a0, r0, r1, fade, rate, tone=True):
    """A model's re-voiced crop -> (change array, naturalness): only the samples of the edit
    change; front left/right get the same swap at their share of the voice."""
    import numpy as np
    wav = np.pad(wav, (0, max(0, len(vcrop) - len(wav))))[:len(vcrop)].astype(np.float32)
    if tone:                       # the model's dull 24 kHz voice -> the film's own sound
        wav = match_tone(wav, vcrop, keep, rate)
    seg = wav[r0 - a0:r1 - a0]
    seg = np.pad(seg, (0, max(0, (r1 - r0) - len(seg))))[:r1 - r0]
    delta = ((seg - voc[r0:r1]) * fade).astype(np.float32)
    nat = naturalness(voc, delta, r0, r1, rate)
    if sides is not None:
        delta = np.stack([(seg * g - sv[r0:r1]) * fade for g, sv in zip(gains, sides)] + [delta],
                         axis=1).astype(np.float32)           # (n, 3): FL, FR, FC
    return delta, {k: float(v) for k, v in nat.items()}


def speaker(job: dict, device: str):
    """(embed function or None): the voice-match model."""
    import numpy as np
    import librosa
    import torch
    cache = job.get("cache") or ""
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        save = Path(cache) / "models" / "ecapa" if cache else None
        spk = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                             savedir=str(save) if save else None, run_opts={"device": device})
    except Exception as e:                       # noqa: BLE001 - takes are still ranked by the rest
        say(f"dub: no speaker model ({e}); takes ranked without a voice match")
        return None
    rate = int(job["rate"])

    def embed(x):
        x16 = librosa.resample(x.astype(np.float32), orig_sr=rate, target_sr=16000)
        with torch.inference_mode():
            v = spk.encode_batch(torch.from_numpy(x16)[None].to(device)).squeeze().cpu().numpy()
        return v / max(float(np.linalg.norm(v)), 1e-9)
    return embed


def voice_match(embed, ref_emb, voc, delta, r0, r1, rate):
    """How much the word with this take sounds like the actor's line around it (cosine)."""
    import numpy as np
    d = delta[:, 2] if delta.ndim == 2 else delta
    w0, w1 = max(0, r0 - int(0.15 * rate)), min(len(voc), r1 + int(0.15 * rate))
    new = voc[w0:w1].copy()
    new[r0 - w0:r0 - w0 + len(d)] += d[:len(new) - (r0 - w0)]
    return round(float(embed(new) @ ref_emb), 3)


def main(job_path: str) -> int:
    job = json.loads(Path(job_path).read_text())
    setup(job)
    import numpy as np
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    rate = int(job["rate"])
    xfade = float(job.get("xfade", 0.03))
    mode = job.get("mode", "takes")

    if mode == "voice":            # score other engines' takes: [{id, voice, crop, edit, edits, texts, takes}]
        embed = speaker(job, device)
        results = []
        for item in job["items"]:
            voc = np.load(item["voice"]).astype(np.float32)
            a0, a1, keep, vcrop, _ = layout(item, voc, None, rate, xfade)
            ref = embed(vcrop[keep]) if embed and keep.sum() > rate else None
            scores = []
            for t in item["takes"]:
                d = np.load(t["delta"])
                n = d.shape[0]
                scores.append(voice_match(embed, ref, voc, d, t["start"], t["start"] + n, rate) if ref is not None else None)
            results.append({"id": item["id"], "voices": scores})
        (out / "results.json").write_text(json.dumps(results))
        return 0

    from demucs.api import Separator
    say(f"dub: loading models on {device}")
    separator = Separator(model="htdemucs", device=device)

    if mode == "separate":         # the voice only, for the engines without Demucs
        results = []
        for item in job["items"]:
            voc, sides = separate(separator, item, rate)
            np.save(out / f"v{item['id']}.npy", voc)
            r = {"id": item["id"], "voice": str(out / f"v{item['id']}.npy")}
            if sides is not None:
                np.save(out / f"vs{item['id']}.npy", sides)
                r["voice_sides"] = str(out / f"vs{item['id']}.npy")
            results.append(r)
        (out / "results.json").write_text(json.dumps(results))
        return 0

    import librosa
    from importlib.resources import files
    import f5_tts.device_utils as du
    du.get_default_device = lambda: device
    from cached_path import cached_path
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    from f5_tts.infer.utils_infer import load_checkpoint, load_vocoder
    from f5_tts.model import CFM
    from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer

    steps = int(job.get("steps", 32))
    combos = [(seed, cfg) for cfg in job.get("cfg", [2.0, 3.0]) for seed in job.get("seeds", [1, 2])]
    exp = job.get("model", "F5TTS_v1_Base")
    mcfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{exp}.yaml")))
    mel = mcfg.model.mel_spec
    sr, hop = mel.target_sample_rate, mel.hop_length
    vocab, nvocab = get_tokenizer(str(files("f5_tts").joinpath("infer/examples/vocab.txt")), "custom")
    backbone = get_class(f"f5_tts.model.{mcfg.model.backbone}")
    model = CFM(transformer=backbone(**mcfg.model.arch, text_num_embeds=nvocab, mel_dim=mel.n_mel_channels),
                mel_spec_kwargs=dict(n_fft=mel.n_fft, hop_length=hop, win_length=mel.win_length,
                                     n_mel_channels=mel.n_mel_channels, target_sample_rate=sr,
                                     mel_spec_type=mel.mel_spec_type),
                odeint_kwargs=dict(method="euler"), vocab_char_map=vocab).to(device)
    ckpt = job.get("checkpoint") or f"hf://SWivid/F5-TTS/{exp}/model_1250000.safetensors"
    model = load_checkpoint(model, str(cached_path(ckpt)), device, use_ema=True)
    vocoder = load_vocoder(vocoder_name="vocos")
    embed = speaker(job, device)

    results = []
    for n, item in enumerate(job["items"], 1):
        voc, sides = separate(separator, item, rate)
        np.save(out / f"v{item['id']}.npy", voc)                 # kept for the other engines
        if sides is not None:
            np.save(out / f"vs{item['id']}.npy", sides)
        a0, a1, keep, vcrop, gains = layout(item, voc, sides, rate, xfade)
        clip = librosa.resample(vcrop, orig_sr=rate, target_sr=sr).astype(np.float32)
        audio = torch.from_numpy(clip)[None].to(device)
        level = audio.pow(2).mean().sqrt()
        if level < 0.1:
            audio = audio * 0.1 / max(level, 1e-6)
        with torch.inference_mode():
            omel = model.mel_spec(audio).permute(0, 2, 1)
        ref_emb = embed(vcrop[keep]) if embed is not None and keep.sum() > rate else None

        takes = []
        for alt, text in item["texts"].items():
            e0, e1, r0, r1, fade = frame(item, alt, rate, xfade, len(voc))
            f0, f1 = round(e0 * sr / hop), round(e1 * sr / hop)
            cond = torch.cat((omel[:, :f0], torch.zeros(1, f1 - f0, omel.shape[2], device=device), omel[:, f1:]), 1)
            mask = torch.ones(1, cond.shape[1], dtype=torch.bool, device=device)
            mask[:, f0:f1] = False
            for seed, cfg_strength in combos:
                with torch.inference_mode():
                    gen, _ = model.sample(cond=cond, text=convert_char_to_pinyin([text]), duration=cond.shape[1],
                                          steps=steps, cfg_strength=cfg_strength, sway_sampling_coef=-1.0,
                                          seed=seed, edit_mask=mask)
                    wav = vocoder.decode(gen.to(torch.float32).permute(0, 2, 1)).cpu().numpy()[0]
                if level < 0.1:
                    wav = wav * float(level) / 0.1
                wav = librosa.resample(wav, orig_sr=sr, target_sr=rate)
                delta, nat = finish(wav, voc, sides, gains, keep, vcrop, a0, r0, r1, fade, rate, job.get("tone", True))
                if ref_emb is not None:
                    nat["voice"] = voice_match(embed, ref_emb, voc, delta, r0, r1, rate)
                name = f"d{item['id']}_{re.sub(r'[^a-z]', '', alt.lower())}_{seed}_{cfg_strength:g}.npy"
                np.save(out / name, delta)
                takes.append({"said": alt, "seed": seed, "cfg": cfg_strength, "delta": str(out / name),
                              "start": r0, **nat})
        r = {"id": item["id"], "takes": takes, "voice": str(out / f"v{item['id']}.npy")}
        if sides is not None:
            r["voice_sides"] = str(out / f"vs{item['id']}.npy")
        results.append(r)
        say(f"dub: {n}/{len(job['items'])} {item.get('label', '')}: {len(takes)} takes")

    (out / "results.json").write_text(json.dumps(results))
    return 0


def match_tone(gen, orig, keep, rate, top=11000.0):
    """Give a take the sound of the film's own voice track.

    The model speaks at 24 kHz through a vocoder: nothing above ~11 kHz, and a slightly
    different tone. The octave below the cut is copied above it (the "air" of speech is
    noise-like, so a copy passes for it), then the whole take is equalised, band by band,
    so the line around the word, which the model only re-voiced, matches the original."""
    import numpy as np
    import librosa
    n_fft, hop = 2048, 512
    G = librosa.stft(gen, n_fft=n_fft, hop_length=hop)
    freqs = librosa.fft_frequencies(sr=rate, n_fft=n_fft)
    kc = int(np.searchsorted(freqs, top))
    width = kc - int(np.searchsorted(freqs, top / 2))
    if kc < len(freqs) and width > 0:
        src = kc - width + (np.arange(kc, len(freqs)) - kc) % width
        G[kc:] = G[src] * 0.5
    O = librosa.stft(orig, n_fft=n_fft, hop_length=hop)
    frames = np.minimum(np.arange(G.shape[1]) * hop, len(keep) - 1)
    ref = keep[frames]
    if ref.sum() < 8:
        return librosa.istft(G, hop_length=hop, length=len(gen))
    po = (np.abs(O[:, ref]) ** 2).mean(1)
    pg = (np.abs(G[:, ref]) ** 2).mean(1)
    # bands a sixth of an octave wide from 60 Hz up: smooth enough not to colour the word
    edges = np.geomspace(60, rate / 2, 55)
    band = np.clip(np.searchsorted(edges, freqs) - 1, 0, len(edges) - 2)
    gain = np.ones(len(freqs))
    for b in range(len(edges) - 1):
        m = band == b
        if m.any():
            gain[m] = np.sqrt((po[m].sum() + 1e-12) / (pg[m].sum() + 1e-12))
    gain = np.clip(gain, 0.25, 4.0)            # at most 12 dB either way
    gain = np.convolve(gain, np.ones(5) / 5, mode="same")
    return librosa.istft(G * gain[:, None], hop_length=hop, length=len(gen)).astype(np.float32)


def naturalness(voc, delta, r0, r1, rate):
    """How well a take sits in the line: loudness steps at its two edges compared with the
    original's (a word that jumps out or drops away sounds pasted in), and its level
    against the word it replaces."""
    import numpy as np
    new = voc.copy()
    new[r0:r1] += delta
    w = int(0.06 * rate)

    def db(x):
        return 10 * np.log10(float(np.mean(x ** 2)) + 1e-10)

    steps = []
    for edge in (r0 + int(0.03 * rate), r1 - int(0.03 * rate)):
        a, b = max(0, edge - w), min(len(voc), edge + w)
        jump_new = db(new[edge:b]) - db(new[a:edge])
        jump_old = db(voc[edge:b]) - db(voc[a:edge])
        steps.append(abs(jump_new - jump_old))
    level = db(new[r0:r1]) - db(voc[r0:r1])
    return {"edge_db": round(max(steps), 2), "level_db": round(float(level), 2)}


def say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
