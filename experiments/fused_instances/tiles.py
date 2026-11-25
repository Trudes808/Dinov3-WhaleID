"""Tiles and sheets for the fused-instance study (output to a scratch / untracked folder only: photos may show crew).
A tile = the photo cut around an instance: the instance outline red, the photo's other instances yellow, SAM 3 whale
instances (score >= 0.5) thin cyan / magenta, a split candidate's pieces filled blue / green, a caption.

    PYTHONPATH=. .venv/bin/python experiments/fused_instances/tiles.py --rows ROWS.jsonl --select FILE_OR_EXPR --out SHEET.jpg [--root DAYS]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fcommon as F  # noqa: E402


def inst_masks_full(day_dir: Path, stem: str, hw):
    from model_fine_tuning.cluster.instances_v2 import old_instances
    keys, masks = old_instances(day_dir, stem, hw[1], hw[0], F.table())
    return keys, masks


def outline(img, m, col, th):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, col, th, cv2.LINE_AA)


def tile(day: str, key: str, caption: str, root: Path = F.SR, size: int = 320, pieces: list | None = None, sam: bool = True, pad: float = 0.6):
    stem, k = key.rsplit("~", 1)
    D = root / day
    img = cv2.imread(str(D / "images" / f"{stem}.jpg"))
    H, W = img.shape[:2]
    keys, masks = inst_masks_full(D, stem, (H, W))
    vis = img.copy()
    if pieces:
        ov = vis.copy()
        for m, col in zip(pieces, [(255, 80, 0), (0, 200, 0), (0, 160, 255)]):
            ov[m] = col
        vis = cv2.addWeighted(ov, 0.35, vis, 0.65, 0)
    if sam:
        sm = F.sam_full(day, stem, (H, W))
        for (sc, m), col in zip(sm or [], [(255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 255, 128)]):
            outline(vis, m, col, 2)
    me = None
    for kk, m in zip(keys, masks):
        if kk == key:
            me = m
        else:
            outline(vis, m, (0, 230, 255), 3)
    if me is None or not me.any():
        ys, xs = np.array([0, H - 1]), np.array([0, W - 1])
    else:
        outline(vis, me, (0, 0, 255), 4)
        ys, xs = np.nonzero(me)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; half = max(x1 - x0, y1 - y0) * (0.5 + pad) + 40
    a, b = int(max(cx - half, 0)), int(min(cx + half, W)); c, d = int(max(cy - half, 0)), int(min(cy + half, H))
    crop = vis[c:d, a:b]
    s = size / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(int(crop.shape[1] * s), 1), max(int(crop.shape[0] * s), 1)), interpolation=cv2.INTER_AREA)
    t = np.zeros((int(size * 0.7) + 34, size, 3), np.uint8)
    crop = crop[: int(size * 0.7)]
    t[:crop.shape[0], :crop.shape[1]] = crop
    for i, line in enumerate(caption.split("\n")[:2]):
        cv2.putText(t, line, (3, int(size * 0.7) + 14 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def sheet(tiles: list, cols: int, out: Path, title: str = ""):
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    S = np.full((rows * (h + 4) + 30, cols * (w + 4), 3), 40, np.uint8)
    cv2.putText(S, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        S[30 + r * (h + 4): 30 + r * (h + 4) + h, c * (w + 4): c * (w + 4) + w] = t
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), S, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("sheet ->", out, len(tiles), "tiles")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--keys", type=Path, required=True, help="tsv: day <tab> key <tab> caption (\\n = new line)")
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--cols", type=int, default=6); ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--root", type=Path, default=F.SR); ap.add_argument("--title", default="")
    a = ap.parse_args()
    T = []
    for l in open(a.keys):
        day, key, cap = l.rstrip("\n").split("\t")
        F.check_day(day)
        T.append(tile(day, key, cap.replace("\\n", "\n"), a.root, a.size))
    sheet(T, a.cols, a.out, a.title)


if __name__ == "__main__":
    main()
