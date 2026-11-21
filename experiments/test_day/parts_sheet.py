"""Round 2 sheet: the part candidate W3c on no-whale and crew photos (scratch / untracked folders only: it shows crew).

    .venv/bin/python experiments/test_day/parts_sheet.py --root ROOT --out SHEETS/testday_parts_on_nonwhale.jpg
Panels: photo | W3c full-frame map, no gate | inside the live mask (seg_r04_ft4) | inside the candidate mask (S6-src canon).
Colours: fluke red, dorsal (fin + hump) green, other whale (pectoral, head, body) yellow; background not drawn.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

TW, TH = 300, 200
COL = np.array([[0, 0, 0], [60, 60, 230], [60, 200, 60], [40, 200, 240], [40, 200, 240], [40, 200, 240]] + [[0, 0, 0]] * 250, np.uint8)


def fit(x):
    H, W = x.shape[:2]; f = min(TW / W, TH / H); x = cv2.resize(x, (int(W * f), int(H * f)), interpolation=cv2.INTER_NEAREST)
    return cv2.copyMakeBorder(x, 0, TH - x.shape[0], 0, TW - x.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))


def lab(x, t):
    cv2.rectangle(x, (0, 0), (min(x.shape[1], 7 * len(t) + 6), 16), (0, 0, 0), -1); cv2.putText(x, t, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    R = a.root; day = R.name; R2 = R / "round2"
    T = {r["seq"]: r for r in csv.DictReader(open(R / "truth.tsv"), delimiter="\t")}
    PP = {r["seq"]: r for r in csv.DictReader(open(R2 / "parts_photos.tsv"), delimiter="\t")}
    stems = {Path(json.loads(l)["image"]).stem.split("_")[-2]: Path(json.loads(l)["image"]).stem for l in open(R / "days" / day / "images" / "index.jsonl")}
    pick = [s for s in sorted(T) if T[s]["class"] in ("marker_rail", "bow", "crew_nowhale", "marker_land", "water")]
    pick += [s for s in sorted(T) if T[s]["class"] == "whale_crew"][::3]
    tiles = []
    for s in pick:
        stem = stems[s]; im = cv2.imread(str(R / "days" / day / "images" / f"{stem}.jpg"))
        full = np.array(Image.open(R2 / "w3c_live" / day / "w3c" / "full" / f"{stem}.png")); H, W = full.shape
        sm = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        pans = [fit(sm)]
        for name, mk in (("no gate", None), ("live mask", R / "days" / day / "canon"), ("candidate mask", R2 / "days_s6" / day / "canon")):
            g = np.ones((H, W), bool) if mk is None else cv2.resize(np.array(Image.open(mk / f"{stem}_mask.png")), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            x = (0.55 * sm).astype(np.uint8); sel = g & (full > 0) & (full < 6); x[sel] = COL[full[sel]]
            if mk is not None:
                cs, _ = cv2.findContours(g.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(x, cs, -1, (255, 255, 255), 1)
            x = fit(x); r = PP[s]; key = {"no gate": "none", "live mask": "live", "candidate mask": "s6"}[name]
            lab(x, f"{name}: F {100 * float(r[key + '_fluke']):.1f} D {100 * float(r[key + '_dorsal']):.1f} O {100 * float(r[key + '_other']):.1f} %")
            pans.append(x)
        lab(pans[0], f"{s} {T[s]['class']}")
        tiles.append(np.hstack(pans + [np.full((TH, 10, 3), 255, np.uint8)]))
    while len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 255))
    cv2.imwrite(str(a.out), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 82]); print(len(pick), "photos ->", a.out)


if __name__ == "__main__":
    main()
