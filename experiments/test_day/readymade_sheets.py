"""Round 2 sheets for the ready-made segmentations (scratch / untracked folders only: they show crew).

    .venv/bin/python experiments/test_day/readymade_sheets.py --root ROOT --out SHEETS
-> SHEETS/testday_readymade_classes.jpg (marker, crew and whale + crew photos: photo | ADE groups | COCO groups)
   SHEETS/testday_readymade_on_whales.jpg (whale photos: photo | whale pixels (live AND S6h) coloured by ADE group | by COCO group)
Colours: person red, boat (rail / deck) orange, vessel magenta, land brown, water blue, sky pale cyan, other grey.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from readymade_analyze import GROUPS, lut_for  # noqa: E402

COL = {"person": (40, 40, 230), "boat": (0, 150, 255), "vessel": (220, 0, 220), "land": (30, 80, 140), "water": (200, 120, 20), "sky": (240, 230, 180), "other": (150, 150, 150)}
PAL = np.array([COL[g] for g in GROUPS], np.uint8)
TW = 320


def label(img, text, y=16):
    cv2.rectangle(img, (0, y - 14), (min(img.shape[1], 9 * len(text) + 6), y + 5), (0, 0, 0), -1)
    cv2.putText(img, text, (3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def pad(x):
    return cv2.copyMakeBorder(x, 0, 214 - x.shape[0], 0, TW - x.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))


def grid(tiles, per_row=2):
    h = max(t.shape[0] for t in tiles); tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    while len(tiles) % per_row:
        tiles.append(np.full_like(tiles[0], 255))
    return np.vstack([np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)])


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    R = a.root; D = R / "days" / R.name; RM = R / "round2" / "readymade"; a.out.mkdir(parents=True, exist_ok=True)
    T = {r["seq"]: r for r in csv.DictReader(open(R / "truth.tsv"), delimiter="\t")}
    P = {r["seq"]: r for r in csv.DictReader(open(R / "round2" / "readymade_photos.tsv"), delimiter="\t")}
    stems = {Path(json.loads(l)["image"]).stem.split("_")[-2]: Path(json.loads(l)["image"]).stem for l in open(D / "images" / "index.jsonl")}
    luts = {m: lut_for(m)[1] for m in ("ade", "coco")}

    def load(seq):
        im = cv2.imread(str(D / "images" / f"{stems[seq]}.jpg")); H, W = im.shape[:2]; f = min(TW / W, 214 / H); sm = cv2.resize(im, (int(W * f), int(H * f)), interpolation=cv2.INTER_AREA)
        g = {m: cv2.resize(luts[m][np.array(Image.open(RM / m / f"{seq}.png"))], (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_NEAREST) for m in luts}
        return sm, g

    def says(seq, m):
        r = P[seq]; f = lambda k: float(r[f"{m}_{k}"])
        s = [n for n, v in (("P", f("person")), ("B", f("boat") + f("vessel")), ("V", f("vessel"))) if v >= 0.005]
        s += ["Lsubj"] if f("land") >= 0.25 else ["Lbg"] if f("land") >= 0.005 else []
        return ",".join(s) or "-"

    def truthtags(seq):
        t = T[seq]; return ",".join(n for n, k in (("P", "person"), ("R", "rail"), ("D", "deck"), ("V", "vessel"), ("Ls", "land_subj"), ("Lb", "land_bg")) if t[k] == "1") or "-"

    tiles = []
    sel = [s for s in sorted(T) if T[s]["class"] in ("marker_rail", "bow", "marker_land", "crew_nowhale", "whale_crew")]
    for seq in sel:
        sm, g = load(seq)
        pa = (0.45 * sm + 0.55 * PAL[g["ade"]]).astype(np.uint8); pc = (0.45 * sm + 0.55 * PAL[g["coco"]]).astype(np.uint8)
        label(sm, f"{seq} {T[seq]['class']} truth {truthtags(seq)}"); label(pa, f"ADE: {says(seq, 'ade')}"); label(pc, f"COCO: {says(seq, 'coco')}")
        tiles.append(np.hstack([pad(sm), pad(pa), pad(pc)]))
    leg = np.full((214, tiles[0].shape[1], 3), 255, np.uint8); x = 4
    for gname, c in COL.items():
        cv2.rectangle(leg, (x, 5), (x + 16, 21), c, -1); cv2.putText(leg, gname, (x + 20, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1); x += 110
    cv2.putText(leg, "says: P person, B boat+vessel, V vessel >= 0.5 %; Lsubj land >= 25 %", (x, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    tiles.insert(0, leg); tiles.insert(1, np.full_like(leg, 255))
    cv2.imwrite(str(a.out / "testday_readymade_classes.jpg"), grid(tiles), [cv2.IMWRITE_JPEG_QUALITY, 82])

    M = json.loads((R / "round2" / "readymade_measures.json").read_text())["2c_whale_pixels"]
    pick = sorted(set(M["ade"]["photos_veto_ge50"]) | set(M["coco"]["photos_veto_ge50"]) | set(M["coco"]["photos_land_ge50"]))
    rest = [s for s in sorted(T) if T[s]["class"] in ("whale", "whale_crew") and s not in pick and P[s].get("whale_ade_top_group")]
    random.Random(0).shuffle(rest); pick += sorted(rest[:16])
    tiles = [leg, np.full_like(leg, 255)]
    for seq in pick:
        sm, g = load(seq); stem = stems[seq]
        live = cv2.resize(np.array(Image.open(D / "canon" / f"{stem}_mask.png")), (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        s6 = cv2.resize(np.array(Image.open(R / "cand_s6" / f"{stem}_s6.png")), (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        w = live & s6; pan = []
        for m in ("ade", "coco"):
            x = (0.6 * sm).astype(np.uint8); x[w] = PAL[g[m][w]]; pan.append(x)
            cs, _ = cv2.findContours(w.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(x, cs, -1, (255, 255, 255), 1)
        r = P[seq]
        label(sm, f"{seq} {T[seq]['class']}"); label(pan[0], f"ADE whale px: {r['whale_ade_top_class']} {float(r['whale_ade_top_share']):.2f}"); label(pan[1], f"COCO whale px: {r['whale_coco_top_class']} {float(r['whale_coco_top_share']):.2f}")
        tiles.append(np.hstack([pad(sm), *map(pad, pan)]))
    cv2.imwrite(str(a.out / "testday_readymade_on_whales.jpg"), grid(tiles), [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(len(sel), "class tiles,", len(pick), "whale tiles ->", a.out)


if __name__ == "__main__":
    main()
