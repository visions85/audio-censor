"""Dub worker (PlayDiffusion): run with the Python of a PlayDiffusion environment.

    <playdiffusion python> dub_pd.py JOB.json

The same job as dub_worker.py, done by PlayDiffusion (PlayHT's speech inpainting): the
actor's line is turned into speech tokens, the tokens of the word are masked and filled in
for the new text, keeping exactly the word's length, and the line is decoded again. Only
the samples of the edit are kept (see dub_worker.finish). Each item carries the voice
already separated by a "separate" job of dub_worker.py, since this environment has no Demucs.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import dub_worker as W

FRAME_RATE = 50          # PlayDiffusion speech tokens per second
MASK = 8857              # its mask token


def main(job_path: str) -> int:
    job = json.loads(Path(job_path).read_text())
    W.setup(job)
    import numpy as np
    import librosa
    import soundfile as sf
    import torch
    from playdiffusion import PlayDiffusion
    from playdiffusion.utils.audio_utils import get_vocoder_embedding

    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    rate = int(job["rate"])
    xfade = float(job.get("xfade", 0.03))
    seeds = list(job.get("seeds", [1, 2, 3, 4]))
    W.say("dub: loading PlayDiffusion")
    pd = PlayDiffusion()
    mm = pd.mm

    results = []
    for n, item in enumerate(job["items"], 1):
        voc, sides = W.load_voices(item)
        a0, a1, keep, vcrop, gains = W.layout(item, voc, sides, rate, xfade)
        peak = max(1e-6, float(np.abs(vcrop).max()))
        wavp = out / f"crop{item['id']}.wav"
        sf.write(wavp, vcrop / peak * 0.9, rate)              # the model works on a normalised copy
        emb = get_vocoder_embedding(str(wavp), mm).to(pd.device)
        x16 = librosa.resample(vcrop / peak * 0.9, orig_sr=rate, target_sr=16000).astype(np.float32)
        with torch.inference_mode():
            units = mm.speech_tokenizer.waveform_to_units(torch.from_numpy(x16).to(pd.device))
        if units.dim() == 1:
            units = units[None]
        takes = []
        for alt, text in item["texts"].items():
            e0, e1, r0, r1, fade = W.frame(item, alt, rate, xfade, len(voc))
            f0, f1 = int(round(e0 * FRAME_RATE)), int(round(e1 * FRAME_RATE))
            toks = mm.tokenizer.encode_normalized_to_tensor(text).to(pd.device)
            if toks.dim() == 1:
                toks = toks[None]
            for seed in seeds:
                torch.manual_seed(seed)
                code = torch.cat([units[:, :f0], torch.full((1, f1 - f0), MASK, dtype=units.dtype, device=units.device),
                                  units[:, f1:]], 1)
                with torch.inference_mode():
                    g = mm.inpainter.generate(text_tokens=toks, target_len=None, n_timesteps=int(job.get("steps", 30)),
                                              init_temp=1.0, init_diversity=1.0, guidance=0.5, rescale_cfg=0.7,
                                              topk=25, code=code, start_frame=f0, end_frame=f1)
                    g = torch.cat([units[:, :f0], g[:, f0:f1].to(units.dtype), units[:, f1:]], 1)
                    wav = mm.vocoder(g, emb).float().squeeze().cpu().numpy()
                wav = librosa.resample(wav, orig_sr=mm.vocoder.output_frequency, target_sr=rate) * peak / 0.9
                delta, nat = W.finish(wav, voc, sides, gains, keep, vcrop, a0, r0, r1, fade, rate, job.get("tone", True))
                name = f"p{item['id']}_{re.sub(r'[^a-z]', '', alt.lower())}_{seed}.npy"
                np.save(out / name, delta)
                takes.append({"said": alt, "seed": seed, "delta": str(out / name), "start": r0, **nat})
        wavp.unlink(missing_ok=True)
        results.append({"id": item["id"], "takes": takes})
        W.say(f"dub: playdiffusion {n}/{len(job['items'])} {item.get('label', '')}: {len(takes)} takes")
    (out / "results.json").write_text(json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
