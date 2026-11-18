"""W3c's weak maps: the clicked triangle made TWO-SIDED on 2025-01-02 (the click test day 2025-03-18 is never read).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.parts_weak.build_tri2
        -> CACHE/maps_tri2/<day>/<stem>.png: the clicked photos rebuilt from CACHE/maps_tri plus "whale, not dorsal"
           (value NOTFIN = 6); every other photo a symlink to CACHE/maps (identical to W2's).

Why: in W3 / W3b the whole clicked triangle was labelled dorsal and every other whale pixel of a real photo stayed
ignore (build.py), and no real-photo term says "not dorsal" on a dorsal view (losses.py forbids the fin only in fluke
views). On the 97 clicked photos the whale pixels down to 1.5 fin heights below the base line were 6.7 M ignore and
11 dorsal, against 1.5 M triangle pixels: the positive spread over the look-alike hump and back.

Not dorsal, per clicked fin, in its triangle frame (along: back base 0 -> front base 1; up: base line 0 -> tip 1):
  pixels of the clicked whale (its instance region from build.py AND the live segmentation's blobs) inside a window
  (along -0.6 .. 1.6, up -3 .. 1.2) that are below the base line by more than 0.2 fin heights, or beyond the two
  bases along the back by more than 0.1 of the base (behind the back base / in front of the front base); only where
  the map said ignore (never over the triangle, background or another class), and more than 4 px from the triangle.
  Whale pixels above the sloped edges between the two bases (a convex hump) stay ignore; pixels outside the whale are
  untouched.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from . import CACHE, CLICK_TRAIN_DAY, DAYS, IGNORE, INDEX, MAPS, NOTFIN
from .build import load_clicks

SRC = CACHE / "maps_tri"
OUT = CACHE / "maps_tri2"
UP_MARGIN, ALONG_MARGIN, PX_MARGIN = 0.2, 0.1, 4


def main() -> int:
    photos = json.loads(INDEX.read_text())["photos"]
    clicks = load_clicks()
    clicked = {(r["day"], r["stem"]) for r in photos if r["day"] == CLICK_TRAIN_DAY and r["points"]}
    n_px, n_fins, n_photos = 0, 0, 0
    for r in photos:
        day, stem = r["day"], r["stem"]
        dst = OUT / day / f"{stem}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if (day, stem) not in clicked:
            if not dst.exists():
                os.symlink(MAPS / day / f"{stem}.png", dst)
            continue
        m = cv2.imread(str(SRC / day / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        lab, inst = m[..., 0].copy(), m[..., 1]
        H, W = lab.shape
        blobs = cv2.imread(str(DAYS / day / "instances_v2" / "blobs" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        whale_px = blobs > 0 if blobs is not None else np.ones((H, W), bool)
        tri_all = cv2.dilate((lab == 2).astype(np.uint8), np.ones((2 * PX_MARGIN + 1, 2 * PX_MARGIN + 1), np.uint8)) > 0
        for key, kp in clicks.items():
            if key.rsplit("~", 1)[0] != stem or not all(x in kp for x in ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back")):
                continue
            k = int(key.rsplit("~", 1)[1])
            T, F, B = (np.array([kp[x]["x"] * W, kp[x]["y"] * H]) for x in ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back"))
            u = F - B
            base = float(np.hypot(*u))
            if base < 8:
                continue
            u /= base
            w = (T - B) - ((T - B) @ u) * u
            h = float(np.hypot(*w))
            if h < 4:
                continue
            nrm = w / h
            reach = 3.5 * max(base, h)
            x0, x1 = int(max(0, min(T[0], F[0], B[0]) - reach)), int(min(W, max(T[0], F[0], B[0]) + reach))
            y0, y1 = int(max(0, min(T[1], F[1], B[1]) - reach)), int(min(H, max(T[1], F[1], B[1]) + reach))
            yy, xx = np.mgrid[y0:y1, x0:x1]
            Q = np.stack([xx, yy], -1).astype(np.float64) - B
            al, up = Q @ u / base, Q @ nrm / h
            window = (al > -0.6) & (al < 1.6) & (up > -3.0) & (up < 1.2)
            outside = (up < -UP_MARGIN) | (al < -ALONG_MARGIN) | (al > 1 + ALONG_MARGIN)
            own = (inst[y0:y1, x0:x1] == k + 1) & whale_px[y0:y1, x0:x1]
            sel = window & outside & own & (lab[y0:y1, x0:x1] == IGNORE) & ~tri_all[y0:y1, x0:x1]
            sub = lab[y0:y1, x0:x1]
            sub[sel] = NOTFIN
            n_px += int(sel.sum())
            n_fins += 1
        cv2.imwrite(str(dst), np.dstack([lab, inst, m[..., 2]]))
        n_photos += 1
    s = {"rebuilt_photos": n_photos, "fins": n_fins, "not_dorsal_px": n_px,
         "rule": {"below_base_line_margin_x_fin_height": UP_MARGIN, "beyond_bases_margin_x_base": ALONG_MARGIN, "px_from_triangle": PX_MARGIN,
                  "window": "along -0.6..1.6, up -3..1.2"}}
    (CACHE / "maps_tri2_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
