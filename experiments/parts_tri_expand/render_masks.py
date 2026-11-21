"""Whale masks on the renders as the real photos get them: candidate S6h (seg_splash_s6 last.pt + hysteresis 0.6 / 0.1)
and live (seg_r04_ft4 best.pt, P >= 0.5), at long side 1280. Only the shots that have a collected fin.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/parts_tri_expand/render_masks.py
        -> /home/sat3737/NOAA_whales/moonshot_cache/parts_tri_expand/render_masks/<set>/<id>_{s6h,r04}.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.test_day.candidate_mask import CKPT as S6, hysteresis  # noqa: E402
from model_fine_tuning.common.data import pad_to_multiple, to_tensor  # noqa: E402
from model_fine_tuning.segmentation.model import WhaleSegModel  # noqa: E402

R04 = REPO / "experiments" / "seg_r04_ft4" / "checkpoints" / "best.pt"
CACHE = Path("/home/sat3737/NOAA_whales/moonshot_cache/parts_tri_expand")
FACTORY = REPO.parent / "Whale-Image-Data-Factory" / "data" / "renders"


def main() -> None:
    dev = torch.device("cuda")
    models = {n: WhaleSegModel.from_checkpoint(c, dev)[0].eval() for n, c in (("s6h", S6), ("r04", R04))}
    for st in sys.argv[1:] or ["r06r_check20", "r06r_check21"]:
        sids = sorted({r["sid"] for r in json.loads((CACHE / f"render_fins_{st}.json").read_text())})
        out = CACHE / "render_masks" / st
        out.mkdir(parents=True, exist_ok=True)
        for sid in sids:
            im = cv2.cvtColor(cv2.imread(str(FACTORY / st / "dataset" / "images" / f"{sid}.png")), cv2.COLOR_BGR2RGB)
            H, W = im.shape[:2]
            s = 1280 / max(H, W)
            sm = cv2.resize(im, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
            padded, (h, w) = pad_to_multiple(sm)
            x = to_tensor(padded)[None].to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for n, m in models.items():
                    prob = torch.sigmoid(m(x).float())[0, :h, :w].cpu().numpy()
                    mk = hysteresis(prob) if n == "s6h" else prob >= 0.5
                    cv2.imwrite(str(out / f"{sid}_{n}.png"), mk.astype(np.uint8) * 255)
        print(st, len(sids), "shots")


if __name__ == "__main__":
    main()
