"""The candidate (NOT live) whale mask seg_splash_s6 + hysteresis 0.6 / 0.1 on a day: P(whale) at 1280 px, kept as a
mask share per photo and a mask png beside the live ones (OUT/<stem>_s6.png). Settings are the candidate's own, fixed
on other days (experiments/seg_splash/s6_tables.md); nothing chosen here.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/test_day/candidate_mask.py --day-dir DAY --out OUT
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from model_fine_tuning.common.data import pad_to_multiple, to_tensor
from model_fine_tuning.segmentation.model import WhaleSegModel

CKPT = Path(__file__).resolve().parents[1] / "seg_splash_s6" / "checkpoints" / "last.pt"


def hysteresis(prob: np.ndarray, hi: float = 0.6, lo: float = 0.1) -> np.ndarray:
    n, lab = cv2.connectedComponents((prob >= lo).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool); keep[np.unique(lab[prob >= hi])] = True; keep[0] = False
    return keep[lab]


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--day-dir", type=Path, required=True); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda"); model, _ = WhaleSegModel.from_checkpoint(CKPT, dev); model.eval()
    res = {}
    with torch.no_grad():
        for l in open(a.day_dir / "images" / "index.jsonl"):
            name = json.loads(l)["image"]; stem = Path(name).stem
            im = cv2.cvtColor(cv2.imread(str(a.day_dir / "images" / name)), cv2.COLOR_BGR2RGB); H, W = im.shape[:2]; s = 1280 / max(H, W)
            small = cv2.resize(im, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
            padded, (h, w) = pad_to_multiple(small); x = to_tensor(padded)[None].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prob = torch.sigmoid(model(x).float())[0, :h, :w].cpu().numpy()
            m = hysteresis(prob); cv2.imwrite(str(a.out / f"{stem}_s6.png"), m.astype(np.uint8) * 255)
            res[stem] = round(float(m.mean()), 5)
    (a.out / "s6_share.json").write_text(json.dumps(res)); print(len(res), "photos ->", a.out)


if __name__ == "__main__":
    main()
