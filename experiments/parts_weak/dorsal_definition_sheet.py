"""Preview of the new dorsal definition (fin + hump) for the user: SHEETS/dorsal_definition_new.jpg (scratchpad only).

    PYTHONPATH=. .venv/bin/python experiments/parts_weak/dorsal_definition_sheet.py

Left, 12 real fins of 2025-01-02 (the click-training day; never 2025-03-18): photo with the researcher's clicked
triangle | the old dorsal class as the baseline parts_seg_r05_a paints it | the new target region = the whole clicked
triangle on the whale (what W3 is trained on, CACHE/maps_tri). Right, 12 run05 renders: render | old training label |
new widened label (parts_weak/labels_wide.py). Each row's three panels share one scale; green = dorsal part.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fin_definition as FD  # noqa: E402
from model_fine_tuning.common import paths  # noqa: E402
from model_fine_tuning.common.data import WhaleRun  # noqa: E402
from model_fine_tuning.parts_seg.labels import PALETTE  # noqa: E402
from model_fine_tuning.parts_weak import CACHE, DAYS, SHEETS  # noqa: E402
from model_fine_tuning.parts_weak import labels_wide as LW  # noqa: E402

S = 230
GREEN = np.array([60, 200, 60], np.uint8)


def paint(rgb, mask, col=GREEN):
    o = rgb.copy()
    o[mask] = (0.4 * rgb[mask] + 0.6 * col).astype(np.uint8)
    return o


def label_overlay(rgb, lab):
    pal = np.zeros((256, 3), np.uint8)
    pal[: len(PALETTE) // 3] = np.asarray(PALETTE, np.uint8).reshape(-1, 3)
    m = (lab > 0) & (lab < 255)
    o = rgb.copy()
    o[m] = (0.45 * rgb[m] + 0.55 * pal[lab][m]).astype(np.uint8)
    return o


def pad(p):
    return np.pad(p, ((2, 2), (2, 2), (0, 0)), constant_values=255)


def real_rows(n=12):
    ck = FD.clicks("2025-01-02")
    ins = json.loads((DAYS / "2025-01-02" / "parts_seg_r05_a" / "instances.json").read_text())["instances"]
    keys = sorted(k for k in ck if k in ins and "hr_window" in ins[k])
    rng = np.random.default_rng(0)
    rows = []
    for key in rng.permutation(keys):
        stem = key.rsplit("~", 1)[0]
        mp = CACHE / "maps_tri" / "2025-01-02" / f"{stem}.png"
        if not mp.exists() or mp.is_symlink():
            continue
        bgr = cv2.imread(str(DAYS / "2025-01-02" / "images" / f"{stem}.jpg"))
        H, W = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        kp = ck[key]
        T, F, B = (np.array([kp[n]["x"] * W, kp[n]["y"] * H]) for n in ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back"))
        pts = np.stack([T, F, B])
        c = pts.mean(0)
        s = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) * 1.8 + 30
        sc = S / s
        x0, y0 = c - s / 2
        A = np.array([[sc, 0, -x0 * sc], [0, sc, -y0 * sc]])
        ph = cv2.warpAffine(rgb, A, (S, S), flags=cv2.INTER_AREA)
        hx, hy, side, osz = ins[key]["hr_window"]
        a = osz / side
        Mm = np.array([[sc / a, 0, (hx - x0) * sc], [0, sc / a, (hy - y0) * sc]])
        cm = cv2.warpAffine(np.asarray(Image.open(DAYS / "2025-01-02" / "parts_seg_r05_a" / "hr" / f"{key}.png")), Mm, (S, S), flags=cv2.INTER_NEAREST, borderValue=0)
        tri = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)[..., 0]
        tri = cv2.warpAffine(tri, A, (S, S), flags=cv2.INTER_NEAREST, borderValue=255) == 2
        q = np.round(np.c_[pts, np.ones(3)] @ A.T).astype(np.int32)
        p1 = ph.copy()
        cv2.polylines(p1, [q], True, (255, 255, 255), 1)
        for (x, y), col in zip(q, ((255, 0, 255), (0, 255, 255), (255, 255, 0))):
            cv2.circle(p1, (int(x), int(y)), 3, col, -1)
        p2 = paint(ph, cm == 2)
        p3 = paint(ph, tri)
        rows.append(np.hstack([pad(p1), pad(p2), pad(p3)]))
        if len(rows) >= n:
            break
    return rows


def render_rows(n=12):
    run = WhaleRun(LW.RUN05, paths.SPLITS_DIR / "run05.json")
    rows = []
    for sid in run.ids("val")[1::9]:
        old = np.asarray(Image.open(LW.OLD / f"{sid}.png"))
        p = LW.NEW / f"{sid}.png"
        if (old == 2).sum() < 300 or not p.exists():
            continue
        new = np.asarray(Image.open(p))
        ys, xs = np.nonzero(new == 2)
        c = np.array([xs.mean(), ys.mean()])
        s = max(np.ptp(xs), np.ptp(ys)) * 1.8 + 30
        sc = S / s
        A = np.array([[sc, 0, -(c[0] - s / 2) * sc], [0, sc, -(c[1] - s / 2) * sc]])
        img = cv2.warpAffine(run.image(sid), A, (S, S), flags=cv2.INTER_AREA)
        lo = cv2.warpAffine(old, A, (S, S), flags=cv2.INTER_NEAREST, borderValue=0)
        ln = cv2.warpAffine(new, A, (S, S), flags=cv2.INTER_NEAREST, borderValue=0)
        rows.append(np.hstack([pad(img), pad(label_overlay(img, lo)), pad(label_overlay(img, ln))]))
        if len(rows) >= n:
            break
    return rows


def main() -> int:
    L, R = real_rows(), render_rows()
    left, right = np.vstack(L), np.vstack(R)
    Hh = max(left.shape[0], right.shape[0])
    left = np.pad(left, ((0, Hh - left.shape[0]), (0, 24), (0, 0)), constant_values=255)
    right = np.pad(right, ((0, Hh - right.shape[0]), (0, 0), (0, 0)), constant_values=255)
    body = np.hstack([left, right])
    head = np.full((52, body.shape[1], 3), 255, np.uint8)
    t1 = ("REAL (2025-01-02): photo + clicked triangle (tip magenta, front base cyan, back base yellow) | old dorsal class as the "
          "baseline paints it | NEW target = whole clicked triangle")
    t2 = "RENDERS (run05): render | old label (green fin, grey body, blue head) | NEW label: fin + hump, base 2.7 x height, about 2x the area"
    cv2.putText(head, t1, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(head, t2, (6, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    out = SHEETS / "dorsal_definition_new.jpg"
    cv2.imwrite(str(out), cv2.cvtColor(np.vstack([head, body]), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(out, len(L), len(R))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
