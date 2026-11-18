"""Cheap 'how complex is this pose' signals per whale instance, for the simple / complex triage (CPU only).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.pose_side.splash --days train,val --out /var/tmp/sat3737-parts-cache/pose_side/splash.json

Per instance key (the part model's hr window, its class map = whale where class > 0):
  white_ring   share of white-water pixels (S < 45 and V above the open water's median by 60 % of the way to 255) in a ring
              round the whale
               (grown by 0.12 x the whale's long side, the whale itself left out)
  white_on     share of the whale's own pixels that are white (water sheeting off, foam lying on the back)
  splash       white pixels in the ring / whale pixels (the amount of white water against the size of the whale)
  whale_px, border (whale touches the window edge or the photo edge), view [P(fluke), P(dorsal)],
  n_inst (instances of the photo in the part model's json), parts (fluke / fin share of the whale's pixels)
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import DAYS, ROOT, guard


def day_list(spec):
    split = json.loads((ROOT / "model_fine_tuning" / "splits" / "days.json").read_text())
    return guard([d for x in spec.split(",") for d in (split[x] if x in split else [x])])


def one_day(day):
    guard([day])
    inst = json.loads((DAYS / day / "parts_seg_r05_a" / "instances.json").read_text())["instances"]
    vc = json.loads((DAYS / day / "safe" / "view_class.json").read_text())["views"]
    by_stem = {}
    for k in inst:
        by_stem.setdefault(k.rsplit("~", 1)[0], []).append(k)
    out = {}
    for stem, keys in by_stem.items():
        img = cv2.imread(str(DAYS / day / "images" / f"{stem}.jpg"))
        if img is None:
            continue
        H, W = img.shape[:2]
        for k in keys:
            r = inst[k]; f = DAYS / day / "parts_seg_r05_a" / "hr" / f"{k}.png"
            if not r.get("hr_window") or not f.exists():
                continue
            x0, y0, side, osz = r["hr_window"]; cls = np.array(Image.open(f)); s = cls.shape[0] / side
            M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]])
            crop = cv2.warpAffine(img, M, (cls.shape[1], cls.shape[0]), flags=cv2.INTER_AREA, borderValue=(0, 0, 0))
            valid = cv2.warpAffine(np.ones((H, W), np.uint8), M, (cls.shape[1], cls.shape[0]), flags=cv2.INTER_NEAREST) > 0
            whale = (cls > 0) & valid
            if whale.sum() < 50:
                continue
            ys, xs = np.nonzero(whale); L = max(np.ptp(xs), np.ptp(ys)) + 1; g = max(3, int(0.12 * L))
            ring = (cv2.dilate(whale.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))) > 0) & ~whale & valid
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV); far = valid & ~whale & ~ring
            wv = float(np.median(hsv[..., 2][far])) if far.sum() > 500 else float(np.median(hsv[..., 2][ring]))
            # white water = brighter than the open water round it by 60 % of the way to full white, and unsaturated
            # (an absolute V > 190 counts the pale water of overcast days as splash)
            white = (hsv[..., 2] > max(150.0, wv + 0.6 * (255.0 - wv))) & (hsv[..., 1] < 45)
            n = whale.sum()
            border = bool(xs.min() <= 1 or ys.min() <= 1 or xs.max() >= cls.shape[1] - 2 or ys.max() >= cls.shape[0] - 2 or (~valid[np.clip(ys, 0, None), xs]).any())
            touches_photo = bool((cv2.dilate(whale.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)[~valid].any())
            out[k] = dict(white_ring=round(float((white & ring).sum() / max(ring.sum(), 1)), 4), white_on=round(float((white & whale).sum() / n), 4),
                          splash=round(float((white & ring).sum() / n), 4), whale_px=int(n), border=border, photo_edge=touches_photo,
                          view=vc.get(k), n_inst=len(keys), fluke_share=round(float((cls[whale] == 1).mean()), 4), fin_share=round(float((cls[whale] == 2).mean()), 4), water_v=round(wv, 1))
    return day, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--days", default="train,val"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--procs", type=int, default=20)
    a = ap.parse_args(); days = day_list(a.days)
    with Pool(a.procs) as p:
        res = dict(p.map(one_day, days))
    a.out.write_text(json.dumps(res)); print({d: len(v) for d, v in res.items()})


if __name__ == "__main__":
    main()
