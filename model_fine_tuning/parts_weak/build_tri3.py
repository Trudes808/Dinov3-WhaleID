"""W3d's weak maps: W3c's two-sided label with the EXPANDED triangle as the dorsal part (2025-01-02 only; the click
test day 2025-03-18 is never read).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.parts_weak.build_tri3
        -> CACHE/maps_tri3/<day>/<stem>.png: the clicked photos rebuilt from CACHE/maps_tri; every other photo a symlink
           to CACHE/maps (identical to W2's / W3c's).

Per clicked fin (tip / front base / back base):
  dorsal      = the expanded triangle (tri_expand.py, rule A of experiments/parts_tri_expand/rule.json) on the candidate
                whale mask S6h (seg_splash_s6 + hysteresis 0.6 / 0.1, cached at 1280 px, upsampled nearest), united with
                maps_tri's own triangle label; written only where the map said ignore or dorsal (never over certain
                background, a fluke shape or another class);
  not dorsal  = W3c's zones exactly (build_tri2.py: whale pixels of the clicked instance in the window, more than 0.2 fin
                heights under the base line or more than 0.1 base beyond either base), minus the expanded triangle grown
                by 4 px (the expanded part may reach 0.3 fin heights under the base line, where W3c said "not dorsal").
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

from . import CACHE, CLICK_TRAIN_DAY, DAYS, FIN, IGNORE, INDEX, MAPS, NOTFIN, ROOT
from . import tri_expand as TE
from .build import load_clicks

SRC = CACHE / "maps_tri"
OUT = CACHE / "maps_tri3"
S6H = Path("/var/tmp/sat3737-parts-cache/seg_splash/pred/s6h")
RULE = json.loads((ROOT / "experiments" / "parts_tri_expand" / "rule.json").read_text())["params"]
UP_MARGIN, ALONG_MARGIN, PX_MARGIN = 0.2, 0.1, 4                  # W3c's, unchanged


def main() -> int:
    photos = json.loads(INDEX.read_text())["photos"]
    clicks = load_clicks()
    clicked = {(r["day"], r["stem"]) for r in photos if r["day"] == CLICK_TRAIN_DAY and r["points"]}
    n_px, n_nf, n_fins, n_photos, tri_px = 0, 0, 0, 0, 0
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
        tri_px += int((lab == FIN).sum())
        blobs = cv2.imread(str(DAYS / day / "instances_v2" / "blobs" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        whale_px = blobs > 0 if blobs is not None else np.ones((H, W), bool)
        s6 = cv2.imread(str(S6H / day / f"{stem}.png"), cv2.IMREAD_UNCHANGED) > 0
        sx, sy = s6.shape[1] / W, s6.shape[0] / H
        exp_all = np.zeros((H, W), bool)
        fins = []
        for key, kp in clicks.items():
            if key.rsplit("~", 1)[0] != stem or not all(x in kp for x in ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back")):
                continue
            T, F, B = (np.array([kp[x]["x"] * W, kp[x]["y"] * H]) for x in ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back"))
            e_s = TE.expand(T * [sx, sy], F * [sx, sy], B * [sx, sy], s6, **RULE)
            e = cv2.resize(e_s.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            exp_all |= e
            fins.append((key, T, F, B))
        put = exp_all & ((lab == IGNORE) | (lab == FIN))
        lab[put] = FIN
        n_px += int((lab == FIN).sum())
        keep_off = cv2.dilate((lab == FIN).astype(np.uint8), np.ones((2 * PX_MARGIN + 1, 2 * PX_MARGIN + 1), np.uint8)) > 0
        for key, T, F, B in fins:
            k = int(key.rsplit("~", 1)[1])
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
            sel = window & outside & own & (lab[y0:y1, x0:x1] == IGNORE) & ~keep_off[y0:y1, x0:x1]
            lab[y0:y1, x0:x1][sel] = NOTFIN
            n_nf += int(sel.sum())
            n_fins += 1
        cv2.imwrite(str(dst), np.dstack([lab, inst, m[..., 2]]))
        n_photos += 1
    s = {"rebuilt_photos": n_photos, "fins": n_fins, "dorsal_px_after": n_px, "dorsal_px_before": tri_px, "not_dorsal_px": n_nf,
         "rule": RULE, "mask": "S6h (seg_splash_s6 + hysteresis 0.6 / 0.1) at 1280, upsampled nearest",
         "not_dorsal": "W3c's zones (0.2 fin heights under the base line, 0.1 base beyond the bases) minus the expanded part grown 4 px"}
    (CACHE / "maps_tri3_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
