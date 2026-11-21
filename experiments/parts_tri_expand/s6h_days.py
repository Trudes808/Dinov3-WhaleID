"""Candidate whale mask S6h (experiments/test_day/candidate_mask.py: seg_splash_s6 last.pt + hysteresis 0.6 / 0.1, long
side 1280) for the photos where W3c (either seed) paints a dorsal region and no cached S6h exists. Training and
validation days only (the shadow folders' days); never the evaluation days or the test day.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/parts_tri_expand/s6h_days.py
        -> /home/sat3737/NOAA_whales/moonshot_cache/parts_tri_expand/s6h/<day>/<stem>.png (0 / 255)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.test_day.candidate_mask import CKPT, hysteresis  # noqa: E402
from model_fine_tuning.common.data import pad_to_multiple, to_tensor  # noqa: E402
from model_fine_tuning.parts_seg.predict import EVAL_DAYS  # noqa: E402
from model_fine_tuning.segmentation.model import WhaleSegModel  # noqa: E402

SHADOW = Path("/var/tmp/sat3737-parts-cache/parts_weak/days_shadow")
CACHED = Path("/var/tmp/sat3737-parts-cache/seg_splash/pred/s6h")
OUT = Path("/home/sat3737/NOAA_whales/moonshot_cache/parts_tri_expand/s6h")
DAYS = REPO / "experiments" / "days_all"
MODELS = ("parts_seg_r05_w3c", "parts_seg_r05_w3cs1")


def mask_path(day: str, stem: str) -> Path | None:
    for p in (CACHED / day / f"{stem}.png", OUT / day / f"{stem}.png"):
        if p.exists():
            return p
    return None


def todo() -> list:
    out = []
    for day in sorted(os.listdir(SHADOW)):
        assert day not in EVAL_DAYS and day != "2026-04-07"
        stems = set()
        for m in MODELS:
            p = SHADOW / day / m / "instances.json"
            if p.exists():
                for k, v in json.loads(p.read_text())["instances"].items():
                    if "fin" in v.get("hr", {}) or "fin" in v.get("full", {}):
                        stems.add(k.rsplit("~", 1)[0])
        out += [(day, s) for s in sorted(stems) if mask_path(day, s) is None]
    return out


def main() -> None:
    jobs = todo()
    print(len(jobs), "photos", flush=True)
    dev = torch.device("cuda")
    model, _ = WhaleSegModel.from_checkpoint(CKPT, dev)
    model.eval()
    with torch.no_grad():
        for i, (day, stem) in enumerate(jobs):
            im = cv2.cvtColor(cv2.imread(str(DAYS / day / "images" / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
            H, W = im.shape[:2]
            s = 1280 / max(H, W)
            sm = cv2.resize(im, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
            padded, (h, w) = pad_to_multiple(sm)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prob = torch.sigmoid(model(to_tensor(padded)[None].to(dev)).float())[0, :h, :w].cpu().numpy()
            (OUT / day).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(OUT / day / f"{stem}.png"), hysteresis(prob).astype(np.uint8) * 255)
            if i % 500 == 0:
                print(i, day, flush=True)
    print("done")


if __name__ == "__main__":
    main()
