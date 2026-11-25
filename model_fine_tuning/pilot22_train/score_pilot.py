"""Pilot r06_pilot22 val split (minus i12, minus rejected; prep.py ids "val"): per-class IoU of a part model, full frame
at 1280 (the prep cache, run05's pixel scale, as w3e_scores.py synth scores run05 val), pooled pixels over the shots,
against the prep labels (rule-A dorsal; 255 ignored).

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock env PYTHONPATH=. .venv/bin/python -m model_fine_tuning.pilot22_train.score_pilot \
        --model NAME --checkpoint CKPT      -> moonshot_cache/pilot22/d/pilot_val/<NAME>.json
Also per shot scene type (1 / 2 / 3 whales) and per category, dorsal and fluke IoU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .prep import CACHE, DS

OUT = CACHE.parent / "pilot_val"


def iou(m: np.ndarray, names: list[str]) -> dict:
    tp = np.diag(m).astype(float)
    den = m.sum(0) + m.sum(1) - np.diag(m)
    return {names[i]: (round(float(tp[i] / den[i]), 4) if den[i] else None) for i in range(len(names))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True, type=Path)
    a = ap.parse_args(argv)
    import torch
    from PIL import Image
    from ..common.data import IGNORE, pad_to_multiple, to_tensor
    from ..parts_seg.labels import CLASSES
    from ..parts_seg.model import PartsSegModel
    from .train import PilotRun
    dev = torch.device("cuda")
    model, _ = PartsSegModel.from_checkpoint(a.checkpoint, dev)
    model.eval()
    ids = json.loads((CACHE / "ids.json").read_text())["val"]
    man = {json.loads(l)["id"]: json.loads(l) for l in open(DS / "manifest.jsonl") if l.strip()}
    run = PilotRun(CACHE)
    K = len(CLASSES)
    tot = np.zeros((K, K), np.int64)
    groups: dict[str, np.ndarray] = {}
    for sid in ids:
        img, (h, w) = pad_to_multiple(run.image(sid))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(to_tensor(img)[None].to(dev)).float().argmax(1)[0, :h, :w].cpu().numpy().ravel()
        t = np.asarray(Image.open(CACHE / "labels" / f"{sid}.png")).ravel()
        ok = t != IGNORE
        c = np.bincount(t[ok].astype(np.int64) * K + p[ok], minlength=K * K).reshape(K, K)
        tot += c
        for g in (f"n_animals={man[sid].get('n_animals', 1)}", f"category={man[sid]['category']}"):
            groups[g] = groups.get(g, np.zeros((K, K), np.int64)) + c
    res = {"model": a.model, "checkpoint": str(a.checkpoint), "n_val": len(ids), "iou": iou(tot, CLASSES),
           "by_group": {g: {k: v for k, v in iou(m, CLASSES).items() if k in ("fluke", "dorsal_fin")} for g, m in sorted(groups.items())}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.model}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"model": a.model, "n": len(ids), "iou": res["iou"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
