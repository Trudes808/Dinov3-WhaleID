"""Switch prep: the ready-made ADE20K model of round 2 (readymade.py: Mask2Former Swin-L ADE semantic, published test size,
arg-max, same post-processing) on any day folder. Outputs are named by the photo's row in DAY/images/index.jsonl
(4 digits), so nothing here carries a file name; OUT/index.json maps row -> stem (keep OUT untracked).

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/test_day/readymade_day.py --day-dir DAY --out OUT
-> OUT/ade/<row>.png (uint8 ADE class id at long side 1280), OUT/index.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from readymade import MODELS, OUT_LONG, prep  # noqa: E402

EVAL = {"2024-12-18", "2024-12-26", "2025-02-19", "2025-03-21"}


def main() -> None:
    from transformers import Mask2FormerForUniversalSegmentation
    p = argparse.ArgumentParser(); p.add_argument("--day-dir", type=Path, required=True); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    if a.day_dir.resolve().name in EVAL:
        raise SystemExit("evaluation days are not read")
    stems = [Path(json.loads(l)["image"]).stem for l in open(a.day_dir / "images" / "index.jsonl")]
    od = a.out / "ade"; od.mkdir(parents=True, exist_ok=True)
    (a.out / "index.json").write_text(json.dumps({f"{i:04d}": s for i, s in enumerate(stems)}))
    hub, short, longmax = MODELS["ade"]; dev = torch.device("cuda")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(hub).to(dev).eval()
    for i, stem in enumerate(stems):
        f = od / f"{i:04d}.png"
        if f.exists():
            continue
        im = Image.open(a.day_dir / "images" / f"{stem}.jpg").convert("RGB"); W, H = im.size; so = OUT_LONG / max(W, H); oh, ow = int(round(H * so)), int(round(W * so))
        x, (h, w) = prep(im, short, longmax)
        with torch.no_grad():
            out = model(pixel_values=x.to(dev))
            cls = out.class_queries_logits[0].softmax(-1)
            mk = F.interpolate(out.masks_queries_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, :, :h, :w]
            seg = torch.einsum("qc,qhw->chw", cls[:, :-1], mk.sigmoid())
            seg = F.interpolate(seg[None], size=(oh, ow), mode="bilinear", align_corners=False)[0]
            Image.fromarray(seg.argmax(0).to(torch.uint8).cpu().numpy()).save(f)
    print("ade", len(stems), "photos ->", od, flush=True)


if __name__ == "__main__":
    main()
