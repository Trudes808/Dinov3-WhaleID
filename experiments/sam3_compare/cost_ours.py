"""Seconds per photo and peak GPU memory of our whale masks (live seg_r04_ft4 best.pt, candidate seg_splash_s6 last.pt +
hysteresis 0.6 / 0.1), at their production settings (long side 1280, bf16 autocast), on the same photo list SAM 3 timed.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock env PYTHONPATH=. .venv/bin/python experiments/sam3_compare/cost_ours.py --list LIST.tsv --n 100 --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from model_fine_tuning.segmentation.model import WhaleSegModel
from model_fine_tuning.segmentation.predict import segment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test_day"))
from candidate_mask import hysteresis  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ARMS = {"live": REPO / "experiments/seg_r04_ft4/checkpoints/best.pt", "s6h": REPO / "experiments/seg_splash_s6/checkpoints/last.pt"}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--list", type=Path, required=True); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
    paths = [l.split("\t")[1].strip() for l in open(a.list) if l.strip()][: a.n]
    dev = torch.device("cuda"); res = {}
    for arm, ck in ARMS.items():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time(); model, _ = WhaleSegModel.from_checkpoint(ck, dev); model.eval(); load_s = time.time() - t0
        n_params = sum(p.numel() for p in model.parameters())
        ts, tio = [], []
        with torch.no_grad():
            for i, p in enumerate(paths):
                ta = time.time(); im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB); tb = time.time()
                torch.cuda.synchronize(); tc = time.time()
                prob, m = segment(model, im, dev, 1280, 0.5)
                if arm == "s6h":
                    m = hysteresis(prob)
                torch.cuda.synchronize(); td = time.time()
                if i >= 3:   # warm-up left out
                    ts.append(td - tc); tio.append(tb - ta)
        res[arm] = {"checkpoint": str(ck.relative_to(REPO)), "params_M": round(n_params / 1e6, 1), "load_s": round(load_s, 1),
                    "s_per_photo_median": round(float(np.median(ts)), 4), "s_per_photo_mean": round(float(np.mean(ts)), 4),
                    "decode_jpeg_s_median": round(float(np.median(tio)), 4), "n_timed": len(ts),
                    "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1), "input": "long side 1280, padded to a multiple of 16, bf16 autocast; output upsampled to the photo"}
        del model
        print(arm, res[arm], flush=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
