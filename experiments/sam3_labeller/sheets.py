"""Sheets (scratch only; they show crew).

    .venv/bin/python experiments/sam3_labeller/sheets.py pseudo  --out SHEET.jpg [--n 60]            # random training labels
    .venv/bin/python experiments/sam3_labeller/sheets.py truth   --name NAME --out SHEET.jpg          # photo | truth | ours | labeller (24 test photos)
    .venv/bin/python experiments/sam3_labeller/sheets.py markers --name NAME --out SHEET.jpg          # the 23 marker photos with our classes
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lab_common as L  # noqa: E402

# truth classes 0-9 (BGR); ours are mapped with L.TO_TRUTH first
PAL = {1: (60, 200, 60), 2: (40, 40, 230), 3: (0, 140, 255), 4: (230, 0, 230), 5: (230, 120, 30), 6: (220, 220, 0), 7: (0, 220, 255), 8: (40, 80, 140), 9: (160, 60, 160)}
NAMES = {1: "whale", 2: "person", 3: "rail", 4: "tape", 5: "deck", 6: "pontoon", 7: "vessel", 8: "land", 9: "gear", 255: "ignore"}


def overlay(img: np.ndarray, lab: np.ndarray, a: float = 0.55) -> np.ndarray:
    lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST); o = img.copy()
    for k, c in PAL.items():
        m = lab == k
        o[m] = (o[m] * (1 - a) + np.array(c) * a).astype(np.uint8)
    o[lab == 255] = (o[lab == 255] * 0.35).astype(np.uint8)
    return o


def text(im, s, y=22, scale=0.6):
    cv2.putText(im, s, (5, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3); cv2.putText(im, s, (5, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)


def legend(width: int) -> np.ndarray:
    bar = np.full((28, width, 3), 255, np.uint8); x = 6
    for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 255]:
        c = PAL.get(k, (60, 60, 60)); cv2.rectangle(bar, (x, 6), (x + 16, 22), c, -1)
        cv2.putText(bar, NAMES[k], (x + 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1); x += 30 + 9 * len(NAMES[k])
    return bar


def grid(tiles, cols):
    h, w = tiles[0].shape[:2]; rows = []
    for i in range(0, len(tiles), cols):
        r = tiles[i:i + cols] + [np.full((h, w, 3), 255, np.uint8)] * (cols - len(tiles[i:i + cols])); rows.append(np.hstack(r))
    g = np.vstack(rows); return np.vstack([legend(g.shape[1]), g])


def cmd_pseudo(a):
    idx = json.loads((L.OUT / "pseudo" / "index.json").read_text()); pick = random.Random(20261124).sample(range(len(idx)), a.n); tiles = []
    for n, i in enumerate(pick):
        r = idx[i]; img = cv2.imread(str(L.OUT / "pseudo" / r["day"] / f"{r['stem']}.jpg")); lab = cv2.imread(str(L.OUT / "pseudo" / r["day"] / f"{r['stem']}.png"), cv2.IMREAD_UNCHANGED)
        o = overlay(cv2.resize(img, (480, 320), interpolation=cv2.INTER_AREA), L.TO_TRUTH[lab]); text(o, f"{n}")
        tiles.append(np.pad(o, ((2, 2), (2, 2), (0, 0)), constant_values=255))
    (L.OUT / "sheet_pseudo_order.json").write_text(json.dumps([f"{idx[i]['day']}|{idx[i]['stem']}" for i in pick]))
    cv2.imwrite(str(a.out), grid(tiles, 6), [cv2.IMWRITE_JPEG_QUALITY, 85])


def test_photo(seq):
    stems = {L.seq_of(s): s for s in L.day_stems_any(L.TEST)}
    return stems[seq], cv2.imread(str(L.photo_path(L.TEST, stems[seq])))


def cmd_truth(a):
    lab_dir = L.TD / "pixel_truth" / "labels"; seqs = sorted(p.stem for p in lab_dir.glob("*.png")); pres = {}
    for s in seqs:
        t = cv2.imread(str(lab_dir / f"{s}.png"), cv2.IMREAD_UNCHANGED); pres[s] = {k for k in range(10) if (t == k).mean() >= 0.002}
    rng = random.Random(20261125); pick = []
    for k in [4, 3, 5, 8, 2, 1]:                    # 4 photos per class: tape, rail, deck, land, person, whale (fixed order, seed)
        pool = [s for s in seqs if k in pres[s] and s not in pick]; pick += rng.sample(pool, min(4, len(pool)))
    tiles = []
    for s in sorted(pick):
        stem, img = test_photo(s); small = cv2.resize(img, (400, 267), interpolation=cv2.INTER_AREA)
        t = cv2.imread(str(lab_dir / f"{s}.png"), cv2.IMREAD_UNCHANGED)
        ours = L.TO_TRUTH[cv2.imread(str(L.OUT / "pred" / a.name / L.TEST / f"{stem}.png"), cv2.IMREAD_UNCHANGED)]
        lb = L.TO_TRUTH[cv2.imread(str(L.OUT / "labeller_test" / L.TEST / f"{stem}.png"), cv2.IMREAD_UNCHANGED)]
        ims = [small.copy(), overlay(small, t), overlay(small, ours), overlay(small, lb)]
        for im, nm in zip(ims, [s, "truth", "ours", "SAM 3 labeller"]):
            text(im, nm)
        tiles.append(np.pad(np.hstack(ims), ((3, 3), (3, 3), (0, 0)), constant_values=255))
    cv2.imwrite(str(a.out), grid(tiles, 1) if False else grid([np.vstack(tiles[i:i + 12]) for i in range(0, len(tiles), 12)], 2), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(sorted(pick))


def cmd_markers(a):
    mk = [l.rstrip("\n").split("\t") for l in open(L.TD / "pixel_truth" / "markers.tsv")][1:]
    res = json.loads((L.OUT / f"markers_{a.name}.json").read_text())["rows"]; tiles = []
    for r in mk:
        seq, kind = r[0], r[3]
        if kind == "none":
            continue
        stem, img = test_photo(seq); small = cv2.resize(img, (420, 280), interpolation=cv2.INTER_AREA)
        ours = L.TO_TRUTH[cv2.imread(str(L.OUT / "pred" / a.name / L.TEST / f"{stem}.png"), cv2.IMREAD_UNCHANGED)]
        o = overlay(small, ours); x = res[seq]
        text(small, f"{seq} truth {kind} {r[4]}"); text(o, f"call {x['call']} / tape {x['colour']}"); text(o, f"rail {100*x['rail_tape']:.1f}% tube {100*x['tube']:.1f}% tape {100*x['tape_big']:.2f}%", 44, 0.5)
        tiles.append(np.pad(np.hstack([small, o]), ((3, 3), (3, 3), (0, 0)), constant_values=255))
    cv2.imwrite(str(a.out), grid(tiles, 3), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["pseudo", "truth", "markers"]); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=60); ap.add_argument("--name", default=None); a = ap.parse_args()
    {"pseudo": cmd_pseudo, "truth": cmd_truth, "markers": cmd_markers}[a.cmd](a)


if __name__ == "__main__":
    main()
