"""Round 2 sheet: live chain against the candidate mask (S6-src arm) on the photos that change or stay wrong
(scratch / untracked folders only: it shows crew).

    .venv/bin/python experiments/test_day/s6_e2e_sheet.py --root ROOT --out SHEETS/testday_s6_end_to_end.jpg
Rows: the 23 whale + crew photos the live filter emptied, whale photos either arm misses, under-water and water photos
whose answer differs between the arms. Panels: live (mask red, kept instance box green, filtered box red) | candidate
(mask cyan, same box colours); label = seq, truth class, each arm's answer and the page's stack kind.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

TW = 400


def panel(D: Path, stem: str, colour, label: str) -> np.ndarray:
    im = cv2.imread(str(D / "images" / f"{stem}.jpg")); H, W = im.shape[:2]; f = min(TW / W, 267 / H)
    sm = cv2.resize(im, (int(W * f), int(H * f)), interpolation=cv2.INTER_AREA)
    m = cv2.resize(cv2.imread(str(D / "canon" / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE), (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    sm[m] = (0.5 * sm[m] + 0.5 * np.array(colour)).astype(np.uint8)
    P = json.loads((D / "safe" / "photos.json").read_text())[stem]; drop = set(json.loads((D / "safe" / "not_whale.json").read_text())["drop"])
    for it in P["instances"]:
        x0, y0, x1, y1 = (int(v * f) for v in it["bbox"])
        cv2.rectangle(sm, (x0, y0), (x1, y1), (0, 0, 255) if it["key"] in drop else (0, 220, 0), 2)
    cv2.rectangle(sm, (0, 0), (min(sm.shape[1], 7 * len(label) + 6), 18), (0, 0, 0), -1)
    cv2.putText(sm, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.copyMakeBorder(sm, 0, 267 - sm.shape[0], 0, TW - sm.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    R = a.root; day = R.name; DL = R / "days" / day; D6 = R / "round2" / "days_s6" / day
    rows = {r["seq"]: r for r in csv.DictReader(open(R / "round2" / "s6_photos.tsv"), delimiter="\t")}
    M = json.loads((R / "round2" / "s6_measures.json").read_text())
    pick = list(M["lost23"]["seqs"])
    pick += [s for s in sorted(set(M["missed_whale_photos"]["L"]) | set(M["missed_whale_photos"]["S6-src"])) if s not in pick]
    pick += [s for s, r in sorted(rows.items()) if r["truth_class"] in ("under", "water") and (int(r["L_kept"]) > 0) != (int(r["S6-src_kept"]) > 0)]
    stems = {Path(json.loads(l)["image"]).stem.split("_")[-2]: Path(json.loads(l)["image"]).stem for l in open(DL / "images" / "index.jsonl")}
    tiles = []
    ans = lambda r, arm: "whale" if int(r[f"{arm}_kept"]) else ("filtered" if int(r[f"{arm}_n"]) else "none")
    for s in pick:
        r = rows[s]
        a1 = panel(DL, stems[s], (0, 0, 255), f"{s} {r['truth_class']} | LIVE: {ans(r, 'L')}, page {r['L_page']}")
        a2 = panel(D6, stems[s], (255, 255, 0), f"CANDIDATE S6: {ans(r, 'S6-src')}, page {r['S6-src_page']}; v2 {ans(r, 'S6-v2')}")
        tiles.append(np.hstack([a1, np.full((267, 4, 3), 255, np.uint8), a2]))
    tiles = [cv2.copyMakeBorder(t, 0, 0, 0, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    while len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 255))
    sheet = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
    cv2.imwrite(str(a.out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82]); print(len(pick), "photos ->", a.out, pick)


if __name__ == "__main__":
    main()
