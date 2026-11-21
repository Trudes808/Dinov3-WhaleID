"""Round 2: ready-made segmentation with fixed classes (public Mask2Former Swin-L weights) on a day's photos.
Rules: round2_rules.md, section 2 (written before this ran).

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/test_day/readymade.py --day-dir DAY --out OUT

Per photo (named by its 3-digit sequence number): OUT/coco/<seq>.png and OUT/ade/<seq>.png = arg-max class id (uint8) at
long side 1280; OUT/cocoinst/persons.json = person instances (score, box at 1280) and OUT/cocoinst/<seq>.png = the union of
person instances with score >= 0.5 (uint8 0 / 1).
Post-processing is done here as Mask2Former's own inference does it (semantic: sum over queries of softmax class
probability without the null class x sigmoid mask, upsampled to the padded input, padding cut, resized; instance: top 100
query-class pairs, mask = logit > 0, score = class probability x mean mask probability inside the mask), because the
transformers post-processors first squash the masks to 384 x 384 whatever the input size.
Input: shortest edge 800 (COCO, max 1333) or 640 (ADE20K, max 2560), bilinear, ImageNet mean / std, padded to 32.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MODELS = {"coco": ("facebook/mask2former-swin-large-coco-panoptic", 800, 1333), "ade": ("facebook/mask2former-swin-large-ade-semantic", 640, 2560),
          "cocoinst": ("facebook/mask2former-swin-large-coco-instance", 800, 1333)}
MEAN = np.array([0.485, 0.456, 0.406]) * 255; STD = np.array([0.229, 0.224, 0.225]) * 255
OUT_LONG = 1280


def prep(im: Image.Image, short: int, longmax: int):
    W, H = im.size; s = short / min(W, H)
    if max(W, H) * s > longmax:
        s = longmax / max(W, H)
    w, h = int(round(W * s)), int(round(H * s))
    x = (np.asarray(im.resize((w, h), Image.BILINEAR), np.float32) - MEAN) / STD
    pw, ph = (w + 31) // 32 * 32, (h + 31) // 32 * 32
    pad = np.zeros((ph, pw, 3), np.float32); pad[:h, :w] = x
    return torch.from_numpy(pad).permute(2, 0, 1)[None], (h, w)


def main() -> None:
    from transformers import Mask2FormerForUniversalSegmentation
    p = argparse.ArgumentParser(); p.add_argument("--day-dir", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--models", default="coco,ade,cocoinst"); a = p.parse_args()
    dev = torch.device("cuda")
    stems = [json.loads(l)["image"] for l in open(a.day_dir / "images" / "index.jsonl")]
    for name in a.models.split(","):
        hub, short, longmax = MODELS[name]; od = a.out / name; od.mkdir(parents=True, exist_ok=True)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(hub).to(dev).eval()
        persons = {}
        person_id = 0
        for img in stems:
            seq = Path(img).stem.split("_")[-2]
            im = Image.open(a.day_dir / "images" / img).convert("RGB"); W, H = im.size; so = OUT_LONG / max(W, H); oh, ow = int(round(H * so)), int(round(W * so))
            x, (h, w) = prep(im, short, longmax)
            with torch.no_grad():
                out = model(pixel_values=x.to(dev))
                cls = out.class_queries_logits[0].softmax(-1)            # Q x (C + 1)
                mk = F.interpolate(out.masks_queries_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, :, :h, :w]
                if name != "cocoinst":
                    seg = torch.einsum("qc,qhw->chw", cls[:, :-1], mk.sigmoid())
                    seg = F.interpolate(seg[None], size=(oh, ow), mode="bilinear", align_corners=False)[0]
                    Image.fromarray(seg.argmax(0).to(torch.uint8).cpu().numpy()).save(od / f"{seq}.png")
                else:
                    Q, C1 = cls.shape; sc = cls[:, :-1].flatten(); top, idx = sc.topk(100); q = idx // (C1 - 1); lab = idx % (C1 - 1)
                    sel = lab == person_id; rows = []; union = torch.zeros((oh, ow), dtype=torch.bool, device=dev)
                    for s_, q_ in zip(top[sel], q[sel]):
                        m = F.interpolate(mk[q_][None, None], size=(oh, ow), mode="bilinear", align_corners=False)[0, 0]
                        b = m > 0; ms = float((m.sigmoid() * b).sum() / (b.sum() + 1e-6)); score = float(s_) * ms
                        if b.sum() == 0:
                            continue
                        ys, xs = torch.nonzero(b, as_tuple=True)
                        rows.append({"score": round(score, 4), "area": int(b.sum()), "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]})
                        if score >= 0.5:
                            union |= b
                    persons[seq] = rows
                    Image.fromarray(union.to(torch.uint8).cpu().numpy()).save(od / f"{seq}.png")
        if name == "cocoinst":
            (od / "persons.json").write_text(json.dumps(persons))
        del model; torch.cuda.empty_cache(); print(name, len(stems), "photos ->", od, flush=True)


if __name__ == "__main__":
    main()
