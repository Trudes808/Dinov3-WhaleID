"""Label QA sheet for the part labels: render crop next to the label overlay, ~30 shots across pose categories.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.parts_seg.qa_labels OUT.jpg [--n 30] [--seed 0] [--ids a,b,..]

The sheet shows renders: write it OUTSIDE the repository (scratchpad).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from .labels import CLASSES, colorize, derive

RUN = Path.home() / "NOAA_whales/Whale-Image-Data-Factory/data/renders/run05/dataset"


def tile(ds: Path, sid: str, size: int = 300, label: np.ndarray | None = None) -> np.ndarray:
    img = cv2.imread(str(ds / "images" / f"{sid}.png"))
    lab = derive(ds, sid) if label is None else label
    whale = (lab > 0) & (lab != 255)                       # crop on the labelled whale, not on the under-water ignore
    ys, xs = np.nonzero(whale)
    H, W = lab.shape
    if len(xs):
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        side = max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.25 + 24
    else:
        cx, cy, side = W / 2, H / 2, H
    x0, y0 = int(cx - side / 2), int(cy - side / 2)
    M = np.array([[size / side, 0, -x0 * size / side], [0, size / side, -y0 * size / side]])
    a = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_AREA)
    col = cv2.cvtColor(colorize(lab), cv2.COLOR_RGB2BGR)
    over = np.where((lab > 0)[..., None], (0.35 * img + 0.65 * col).astype(np.uint8), img)
    b = cv2.warpAffine(over, M, (size, size), flags=cv2.INTER_NEAREST)
    t = np.concatenate([a, b], 1)
    meta = json.loads((ds / "meta" / f"{sid}.json").read_text())
    txt = f"{sid} {meta['category']}{' +calf' if meta.get('companion') else ''}"
    cv2.putText(t, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
    cv2.putText(t, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return t


def sheet(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    out = np.full((rows * (h + 4), cols * (w + 4), 3), 40, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        out[r * (h + 4): r * (h + 4) + h, c * (w + 4): c * (w + 4) + w] = t
    return out


def legend(width: int) -> np.ndarray:
    from .labels import PALETTE
    bar = np.full((22, width, 3), 255, np.uint8)
    x = 6
    for k, name in enumerate(CLASSES[1:] + ["ignore"]):
        col = PALETTE[3 * (k + 1): 3 * (k + 2)] if name != "ignore" else [255, 0, 255]
        cv2.rectangle(bar, (x, 5), (x + 14, 17), tuple(int(v) for v in col[::-1]), -1)
        cv2.putText(bar, name, (x + 18, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        x += 30 + 9 * len(name)
    return bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--cols", type=int, default=3)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(RUN / "manifest.jsonl") if l.strip()]
    if a.ids:
        ids = a.ids.split(",")
    else:
        rng = random.Random(a.seed)
        by = {}
        for r in rows:
            if not r["negative"] and r["visible_pixels"] >= 3000:
                by.setdefault(r["category"], []).append(r["id"])
        cats = sorted(by)
        ids = []
        # two per category, plus calf shots
        for c in cats:
            ids += rng.sample(by[c], min(2, len(by[c])))
        pool = [r["id"] for r in rows if not r["negative"] and r["visible_pixels"] >= 3000 and r["category"] in ("surface", "back_arch", "blow", "awash")]
        rng.shuffle(pool)
        calf = [i for i in pool[:400] if json.loads((RUN / "meta" / f"{i}.json").read_text()).get("companion")]
        ids = ids[: max(0, a.n - 4)] + calf[:4]
    tiles = [tile(RUN, i) for i in ids]
    img = sheet(tiles, a.cols)
    img = np.concatenate([legend(img.shape[1]), img], 0)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", a.out, img.shape, ids)


if __name__ == "__main__":
    main()
