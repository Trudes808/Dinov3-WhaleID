"""Fluke tips from the saved probability maps (predict.py --save-maps), CPU only: the two tip channels are summed and the
two strongest separate peaks taken (the plain per-channel arg-max often puts both tips on the same blade: the head cannot
tell the whale's left from its right), then named either by the head ('model': the assignment with the larger channel
probabilities) or by the photo ('image': the tip on the left of the photo is the whale's left, true for 94 % of the
researcher's 83 clicked flukes, which are nearly all seen from behind).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.pose_side.decode_tips MAPS_DIR OUT_PREFIX
writes OUT_PREFIX_model.json and OUT_PREFIX_image.json ({key: {name: [x, y]}}, jpg px) and OUT_PREFIX_*_meta.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .data import KP, STATES

STRIDE = 4


def refine(p, iy, ix):
    H, W = p.shape; y0, y1, x0, x1 = max(iy - 1, 0), min(iy + 2, H), max(ix - 1, 0), min(ix + 2, W)
    w = p[y0:y1, x0:x1]; yy, xx = np.mgrid[y0:y1, x0:x1]; s = max(w.sum(), 1e-12)
    return np.array([((xx * w).sum() / s) * STRIDE + (STRIDE - 1) / 2, ((yy * w).sum() / s) * STRIDE + (STRIDE - 1) / 2])


def peak(p):
    j = int(p.argmax()); iy, ix = divmod(j, p.shape[1]); return iy, ix


def decode_one(prob):
    prob = prob.astype(np.float32); pts = {}; conf = {}
    for k, n in enumerate(KP):
        iy, ix = peak(prob[k]); pts[n] = refine(prob[k], iy, ix); conf[n] = float(prob[k].max())
    q = prob[0] + prob[1]; y1, x1 = peak(q); ny, nx = peak(prob[2])
    r = max(6.0, 0.8 * np.hypot(y1 - ny, x1 - nx))
    yy, xx = np.mgrid[0:q.shape[0], 0:q.shape[1]]; q2 = np.where((yy - y1) ** 2 + (xx - x1) ** 2 <= r * r, 0, q)
    y2, x2 = peak(q2)
    a, b = refine(q, y1, x1), refine(q2, y2, x2)
    second = float(q2.max() / max(q.max(), 1e-12))
    model = (a, b) if prob[0][y1, x1] + prob[1][y2, x2] >= prob[0][y2, x2] + prob[1][y1, x1] else (b, a)
    image = (a, b) if a[0] <= b[0] else (b, a)
    return pts, conf, model, image, second


def main():
    maps, prefix = Path(sys.argv[1]), sys.argv[2]
    outs = {"model": ({}, {}), "image": ({}, {})}
    for f in sorted(maps.glob("*.npz")):
        z = np.load(f); key = f.stem; M = z["M"]; Mi = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2]
        pts, conf, model, image, second = decode_one(z["prob"])
        for mode, (L, R) in (("model", model), ("image", image)):
            p = dict(pts); p["fluke_tip_left"], p["fluke_tip_right"] = L, R
            outs[mode][0][key] = {n: [float(v) for v in Mi[:, :2] @ p[n] + Mi[:, 2]] for n in KP}
            c = dict(conf); c["fluke_tip_left"] = c["fluke_tip_right"] = min(conf["fluke_tip_left"], conf["fluke_tip_right"]) if second < 0.05 else max(conf["fluke_tip_left"], conf["fluke_tip_right"]) * second
            outs[mode][1][key] = dict(conf=c, state={n: [float(v) for v in z["state"][k]] for k, n in enumerate(KP)}, second_peak=second)
    for mode, (P, META) in outs.items():
        Path(f"{prefix}_{mode}.json").write_text(json.dumps(P)); Path(f"{prefix}_{mode}_meta.json").write_text(json.dumps(dict(states=STATES, rows=META)))
        print(mode, len(P))


if __name__ == "__main__":
    main()
