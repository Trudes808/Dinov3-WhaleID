"""Contact sheets for removal rule v2 (they show crew: write them to the scratch folder or moonshot_cache only, never here).

    .venv/bin/python experiments/removal_v2/sheets.py ITEMS.json OUT.jpg [--title T] [--cols 5]
ITEMS.json: [{"image": path, "cap": [line, ...], "color": [b, g, r]}, ...]  (captions name photos by sequence / counter only)
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

W, H, CAP = 400, 267, 52


def tile(it: dict) -> np.ndarray:
    im = cv2.imread(it["image"])
    if im is None:
        im = np.zeros((H, W, 3), np.uint8)
    h, w = im.shape[:2]; s = min(W / w, H / h); im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    t = np.zeros((H + CAP, W, 3), np.uint8); t[:im.shape[0], :im.shape[1]] = im
    col = tuple(int(c) for c in it.get("color", (255, 255, 255)))
    cv2.rectangle(t, (0, 0), (W - 1, H + CAP - 1), col, 3)
    for k, line in enumerate(it.get("cap", [])[:3]):
        cv2.putText(t, str(line)[:58], (6, H + 15 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col if k == 0 else (220, 220, 220), 1)
    return t


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("items"); ap.add_argument("out"); ap.add_argument("--title", default=""); ap.add_argument("--cols", type=int, default=5)
    a = ap.parse_args()
    items = json.load(open(a.items)); tiles = [tile(it) for it in items] or [np.zeros((H + CAP, W, 3), np.uint8)]
    tiles += [np.zeros_like(tiles[0])] * ((-len(tiles)) % a.cols)
    img = np.vstack([np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)])
    if a.title:
        bar = np.zeros((34, img.shape[1], 3), np.uint8); cv2.putText(bar, a.title[:200], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        img = np.vstack([bar, img])
    cv2.imwrite(a.out, img, [cv2.IMWRITE_JPEG_QUALITY, 85]); print(a.out, len(items))


if __name__ == "__main__":
    main()
