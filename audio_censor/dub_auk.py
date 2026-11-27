"""Dub worker (AuK): run with the Python of an AuK environment.

    <auk python> dub_auk.py JOB.json

The same job as dub_worker.py, done by Tencent's AuK ("Replace 'X' with 'Y'."). AuK voices
the whole line again, so only its new word is kept: its output is lined up with the
original on the speech before and after the word, the word is cut out, fitted to the edit
region (so the lips still match), and finished as the other engines' takes are. It is the
slowest engine and the largest, so audio-censor runs it only for words the others could not
do ([dub] fallback). Each item carries the voice separated by dub_worker.py.

job["ckpts"] is the folder holding AuK/ (auk_base.safetensors, vae, config.yaml) and
Qwen2.5-Omni-3B/.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import dub_worker as W


def lag(a, b, rate, maxlag=0.6, hop_s=0.01):
    """(seconds b runs late against a, how well they match): log-envelope correlation."""
    import numpy as np
    hop = int(hop_s * rate)

    def env(x):
        n = len(x) // hop
        return np.log(np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-10))
    ea, eb = env(a), env(b)
    m = int(maxlag / hop_s)
    best, bl = -1e9, 0
    for L in range(-m, m + 1):
        x, y = (ea, eb[L:]) if L >= 0 else (ea[-L:], eb)
        k = min(len(x), len(y))
        if k < 20:
            continue
        x, y = x[:k] - x[:k].mean(), y[:k] - y[:k].mean()
        c = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
        if c > best:
            best, bl = c, L
    return bl * hop_s, best


def load(ckpts: Path):
    """AuK straight onto the GPU (staging both models in RAM first needs more than 16 GB)."""
    import torch
    import transformers
    import auk.infer.infer_auk as IA
    base = transformers.Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained.__func__

    def to_gpu(cls, *a, **k):
        k.setdefault("device_map", "cuda")
        k.setdefault("low_cpu_mem_usage", True)
        return base(cls, *a, **k)
    IA.Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained = classmethod(to_gpu)
    torch.set_default_device("cuda")
    try:
        eng = IA.AukInfer(str(ckpts / "AuK" / "config.yaml"), str(ckpts / "AuK" / "auk_base.safetensors"),
                          qwen_path=str(ckpts / "Qwen2.5-Omni-3B"), cpu_offload=False)
    finally:
        torch.set_default_device("cpu")
    eng.model.transformer.to(torch.bfloat16)      # it runs under bf16 autocast anyway; halves its 6 GB
    torch.cuda.empty_cache()
    return eng


def main(job_path: str) -> int:
    job = json.loads(Path(job_path).read_text())
    W.setup(job)
    import numpy as np
    import librosa
    import soundfile as sf
    import torch

    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    rate = int(job["rate"])
    xfade = float(job.get("xfade", 0.03))
    seeds = list(job.get("seeds", [1, 2]))
    W.say("dub: loading AuK")
    eng = load(Path(job["ckpts"]).expanduser())

    results = []
    for n, item in enumerate(job["items"], 1):
        voc, sides = W.load_voices(item)
        a0, a1, keep, vcrop, gains = W.layout(item, voc, sides, rate, xfade)
        peak = max(1e-6, float(np.abs(vcrop).max()))
        wavp = out / f"crop{item['id']}.wav"
        sf.write(wavp, vcrop / peak * 0.9, rate)
        old = re.sub(r"[^\w' -]", "", item["word"]).strip()
        base_len = item["edit"][1] - item["edit"][0]
        takes = []
        for alt in item["texts"]:
            e0, e1, r0, r1, fade = W.frame(item, alt, rate, xfade, len(voc))
            extra = (e1 - e0) - base_len
            s0, s1 = r0 - a0, r1 - a0                       # the edit within the crop, in samples
            for seed in seeds:
                msgs = [{"role": "user", "content": [{"type": "text", "text": f"Replace '{old}' with '{alt}'."},
                                                     {"type": "audio", "audio": str(wavp)}]}]
                with torch.inference_mode():
                    y, ysr = eng.generate(msgs, gen_seconds=len(vcrop) / rate + extra, seed=seed)
                y = y.squeeze().float().cpu().numpy()
                y = (librosa.resample(y, orig_sr=ysr, target_sr=rate) if ysr != rate else y) * peak / 0.9
                lb, _ = lag(vcrop[:max(s0, int(0.3 * rate))], y, rate)      # line up the speech before ...
                tail = vcrop[s1:]
                if len(tail) > int(0.3 * rate):                               # ... and after the word
                    start = max(0, s1 + int((extra + lb - 0.3) * rate))
                    la, _ = lag(tail, y[start:], rate)
                    end = start + int(la * rate)
                else:
                    end = s1 + int((extra + lb) * rate)
                seg = y[max(0, s0 + int(lb * rate)):max(0, end)]
                want = r1 - r0
                if not want * 0.5 <= len(seg) <= want * 2:
                    continue                                  # could not find the word in its output
                if abs(len(seg) - want) > 0.03 * want:
                    seg = librosa.effects.time_stretch(seg.astype(np.float32), rate=len(seg) / want)
                seg = np.pad(seg, (0, max(0, want - len(seg))))[:want]
                # AuK's whole line, shifted onto the original, with the word in place: for the tone match
                line = np.roll(np.pad(y, (0, max(0, len(vcrop) - len(y))))[:len(vcrop)], -int(lb * rate))
                line[s0:s1] = seg[:s1 - s0]
                delta, nat = W.finish(line, voc, sides, gains, keep, vcrop, a0, r0, r1, fade, rate, job.get("tone", True))
                name = f"a{item['id']}_{re.sub(r'[^a-z]', '', alt.lower())}_{seed}.npy"
                np.save(out / name, delta)
                takes.append({"said": alt, "seed": seed, "delta": str(out / name), "start": r0, **nat})
        wavp.unlink(missing_ok=True)
        results.append({"id": item["id"], "takes": takes})
        W.say(f"dub: auk {n}/{len(job['items'])} {item.get('label', '')}: {len(takes)} takes")
    (out / "results.json").write_text(json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
