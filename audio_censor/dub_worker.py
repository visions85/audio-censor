"""Dub worker: run with the Python of the dub model environment, not audio-censor's own.

    <dub python> dub_worker.py JOB.json

For every muted word in the job it separates the voice from the centre channel (Demucs),
then has F5-TTS regenerate just that word inside the actor's own line (speech editing:
the audio around it is kept, so the new word carries on the real voice, pace and room),
once per replacement word and seed. Each take is written as the change to make to the
centre channel over the word: new voice minus old voice, faded in and out. audio-censor
then checks the takes with Whisper and keeps the best (see dub.py).

Only third-party packages are imported here (torch, torchaudio, demucs, f5_tts, numpy,
soundfile, librosa): this file runs outside audio-censor's environment.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


def main(job_path: str) -> int:
    job = json.loads(Path(job_path).read_text())
    cache = job.get("cache") or ""
    if cache:                                   # model downloads live beside the environment
        os.environ.setdefault("HF_HOME", str(Path(cache) / "hf"))
        os.environ.setdefault("CACHED_PATH_CACHE_ROOT", str(Path(cache) / "cached-path"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")      # its downloader can exhaust RAM

    import numpy as np
    import soundfile as sf
    import librosa
    import torch
    from importlib.resources import files

    import f5_tts.device_utils as du
    du.get_default_device = lambda: "cuda" if torch.cuda.is_available() else "cpu"
    from cached_path import cached_path
    from demucs.api import Separator
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    from f5_tts.infer.utils_infer import load_checkpoint, load_vocoder
    from f5_tts.model import CFM
    from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer

    device = du.get_default_device()
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    rate = int(job["rate"])
    steps = int(job.get("steps", 32))
    combos = [(seed, cfg) for cfg in job.get("cfg", [2.0, 3.0]) for seed in job.get("seeds", [1, 2])]
    xfade = float(job.get("xfade", 0.03))

    say(f"dub: loading models on {device}")
    separator = Separator(model="htdemucs", device=device)
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

    results = []
    for n, item in enumerate(job["items"], 1):
        win, wsr = np.load(item["window"]).astype(np.float32), rate    # the centre channel around the word
        # the voice alone (Demucs works in stereo at 44.1 kHz)
        x = torch.from_numpy(np.stack([win, win]))
        _, stems = separator.separate_tensor(x, wsr)
        voc = stems["vocals"].mean(0).cpu().numpy()
        voc = librosa.resample(voc, orig_sr=separator.samplerate, target_sr=rate) if separator.samplerate != rate else voc
        voc = np.pad(voc, (0, max(0, len(win) - len(voc))))[:len(win)]

        # front left/right carrying the voice too (an upmix): separate them as well, and
        # measure how much of the centre's voice each one plays (over the line, not the word)
        sides = None
        if item.get("sides"):
            lr = np.load(item["sides"]).astype(np.float32)            # (frames, 2): FL, FR
            _, st2 = separator.separate_tensor(torch.from_numpy(lr.T.copy()), wsr)
            sides = st2["vocals"].cpu().numpy()
            if separator.samplerate != rate:
                sides = np.stack([librosa.resample(c, orig_sr=separator.samplerate, target_sr=rate) for c in sides])
            sides = np.pad(sides, ((0, 0), (0, max(0, len(win) - sides.shape[1]))))[:, :len(win)]

        c0, c1 = item["crop"]                       # seconds within the window
        e0, e1 = item["edit"]                       # seconds within the crop
        clip = librosa.resample(voc[int(c0 * rate):int(c1 * rate)], orig_sr=rate, target_sr=sr).astype(np.float32)
        audio = torch.from_numpy(clip)[None].to(device)
        level = audio.pow(2).mean().sqrt()
        if level < 0.1:
            audio = audio * 0.1 / max(level, 1e-6)
        with torch.inference_mode():
            omel = model.mel_spec(audio).permute(0, 2, 1)
        f0, f1 = round(e0 * sr / hop), round(e1 * sr / hop)
        cond = torch.cat((omel[:, :f0], torch.zeros(1, f1 - f0, omel.shape[2], device=device), omel[:, f1:]), 1)
        mask = torch.ones(1, cond.shape[1], dtype=torch.bool, device=device)
        mask[:, f0:f1] = False

        # the samples of the window that change, with fades at both ends
        r0, r1 = int((c0 + e0 - xfade) * rate), int((c0 + e1 + xfade) * rate)
        r0, r1 = max(r0, 0), min(r1, len(win))
        fade = np.ones(r1 - r0, np.float32)
        k = max(1, min(int(xfade * rate), (r1 - r0) // 2))
        fade[:k] = np.linspace(0, 1, k)
        fade[-k:] = np.linspace(1, 0, k)
        gains = None
        if sides is not None:
            ref = np.ones(len(win), bool)
            ref[:int(c0 * rate)] = False
            ref[int(c1 * rate):] = False
            ref[r0:r1] = False
            vc = voc[ref]
            gains = [float(np.clip(np.dot(sv[ref], vc) / max(float(np.dot(vc, vc)), 1e-9), 0.0, 1.5)) for sv in sides]

        takes = []
        for alt, text in item["texts"].items():
            for seed, cfg_strength in combos:
                with torch.inference_mode():
                    gen, _ = model.sample(cond=cond, text=convert_char_to_pinyin([text]), duration=cond.shape[1],
                                          steps=steps, cfg_strength=cfg_strength, sway_sampling_coef=-1.0,
                                          seed=seed, edit_mask=mask)
                    wav = vocoder.decode(gen.to(torch.float32).permute(0, 2, 1)).cpu().numpy()[0]
                if level < 0.1:
                    wav = wav * float(level) / 0.1
                wav = librosa.resample(wav, orig_sr=sr, target_sr=rate)
                seg = wav[r0 - int(c0 * rate):r1 - int(c0 * rate)]
                seg = np.pad(seg, (0, max(0, (r1 - r0) - len(seg))))[:r1 - r0]
                delta = ((seg - voc[r0:r1]) * fade).astype(np.float32)
                if sides is not None:      # the same swap on the left and right, at their share of the voice
                    delta = np.stack([(seg * g - sv[r0:r1]) * fade for g, sv in zip(gains, sides)] + [delta],
                                     axis=1).astype(np.float32)           # (n, 3): FL, FR, FC
                name = f"d{item['id']}_{re.sub(r'[^a-z]', '', alt.lower())}_{seed}_{cfg_strength:g}.npy"
                np.save(out / name, delta)
                takes.append({"said": alt, "seed": seed, "cfg": cfg_strength, "delta": str(out / name), "start": r0})
        results.append({"id": item["id"], "takes": takes})
        say(f"dub: {n}/{len(job['items'])} {item.get('label', '')}: {len(takes)} takes")

    (out / "results.json").write_text(json.dumps(results))
    return 0


def say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
