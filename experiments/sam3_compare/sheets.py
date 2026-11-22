"""Contact sheets for the SAM 3 comparison (they show crew: written to scratch / moonshot_cache only, never to the repo).

    PYTHONPATH=. .venv/bin/python experiments/sam3_compare/sheets.py KIND --settings SETTINGS.json --out SHEET.jpg [--seqs 001,002 ...]
KIND: differs | extra | markers | hard | val_only (val-day SAM 3 whale paint outside our masks)
Each tile: photo with outlines; whale sheets: S6h green, SAM 3 magenta, pixel truth whale (where it exists) yellow.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import s3common as C  # noqa: E402

TW = 560
COL = {"s6": (0, 220, 0), "sam3": (255, 0, 255), "truth": (0, 230, 255), "live": (255, 160, 0),
       "rail": (0, 0, 255), "tape": (0, 255, 255), "deck": (255, 128, 0), "land": (0, 200, 0), "person": (255, 0, 255),
       "pontoon": (128, 0, 255), "other_vessel": (0, 128, 255), "gear": (255, 255, 0)}


def outline(img, m, col, th=2):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, col, th, cv2.LINE_AA)


def fill(img, m, col, a=0.35):
    o = img.copy(); o[m] = col; return cv2.addWeighted(o, a, img, 1 - a, 0)


def tile(seq, layers, title, filled=False):
    st = C.test_stems()[seq]
    im = cv2.imread(str(C.TD / "days" / "2026-04-07" / "images" / f"{st}.jpg"))
    h = int(round(im.shape[0] * TW / im.shape[1])); im = cv2.resize(im, (TW, h), interpolation=cv2.INTER_AREA)
    for m, col in layers:
        if m.shape != im.shape[:2]:
            m = cv2.resize(m.astype(np.uint8), (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        if filled:
            im = fill(im, m, col)
        outline(im, m, col)
    cv2.rectangle(im, (0, 0), (TW, 22), (0, 0, 0), -1)
    cv2.putText(im, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def grid(tiles, ncol, legend):
    h = max(t.shape[0] for t in tiles); rows = []
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % ncol:
        tiles.append(np.zeros_like(tiles[0]))
    for i in range(0, len(tiles), ncol):
        rows.append(np.hstack([cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(40, 40, 40)) for t in tiles[i:i + ncol]]))
    sheet = np.vstack(rows)
    bar = np.zeros((30, sheet.shape[1], 3), np.uint8)
    cv2.putText(bar, legend, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, sheet])


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("kind"); ap.add_argument("--settings", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--seqs", default=""); ap.add_argument("--notes", type=Path)
    ap.add_argument("--ncol", type=int, default=4)
    a = ap.parse_args(); S = json.loads(a.settings.read_text()); idx = C.prompt_index("val" if a.kind == "val_only" else "test")
    seqs = [s for s in a.seqs.split(",") if s]
    notes = json.loads(a.notes.read_text()) if a.notes else {}
    wj = [idx[p] for p in S["class_prompts"]["whale"]]; wthr = S["class_thr"]["whale"]
    truth = C.truth_tsv(); tiles = []
    if a.kind == "val_only":   # --seqs holds DAY/STEM items (val days only)
        for it in seqs:
            day, stem = it.split("/", 1); assert day in C.VAL
            rec = C.load("val", stem); g = (int(round(1709 * TW / 2560)), TW)
            im = cv2.resize(cv2.imread(str(C.LIVE / day / "images" / f"{stem}.jpg")), (g[1], g[0]), interpolation=cv2.INTER_AREA)
            u = np.zeros(g, bool)
            for j in wj:
                u |= C.union(rec, j, wthr, g)
            for m, col in ((C.s6_mask_val(day, stem, g), COL["s6"]), (C.live_mask(day, stem, g), COL["live"]), (u, COL["sam3"])):
                outline(im, m, col)
            cv2.rectangle(im, (0, 0), (TW, 22), (0, 0, 0), -1)
            cv2.putText(im, f"{day} {stem.split('-')[-1]} " + notes.get(it, ""), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(im)
        cv2.imwrite(str(a.out), grid(tiles, a.ncol, "val days: green S6h, light blue live, magenta SAM 3 whale"), [cv2.IMWRITE_JPEG_QUALITY, 85]); print(a.out, len(tiles)); return
    for seq in seqs:
        rec = C.load("test", seq)
        W = TW; H = int(round(1709 * TW / 2560)); g = (H, W)
        if a.kind in ("differs", "hard"):
            u = np.zeros(g, bool)
            for j in wj:
                u |= C.union(rec, j, wthr, g)
            s6 = C.test_s6(seq, g); layers = [(s6, COL["s6"]), (u, COL["sam3"])]
            pt = C.TD / "pixel_truth" / "labels" / f"{seq}.png"
            if pt.exists():
                t = cv2.resize(np.array(Image.open(pt)), (W, H), interpolation=cv2.INTER_NEAREST) == 1
                layers.append((t, COL["truth"]))
            title = f"{seq} {truth[seq]['class']} " + notes.get(seq, "")
            tiles.append(tile(seq, layers, title[:70]))
        else:
            layers = []
            classes = ["rail", "tape", "deck", "land"] if a.kind == "markers" else ["person", "rail", "tape", "deck", "pontoon", "other_vessel", "land", "gear"]
            for c in classes:
                layers.append((C.union(rec, [idx[p] for p in S["class_prompts"][c]], S["class_thr"][c], g), COL[c]))
            if a.kind == "markers":
                layers.append((C.union(rec, [idx[p] for p in S["marker_rule"]["bollard_prompts"]], S["marker_rule"]["bollard_score"], g), (255, 255, 255)))
            title = f"{seq} {truth[seq]['class']} {truth[seq]['marker']} " + notes.get(seq, "")
            tiles.append(tile(seq, layers, title[:70], filled=True))
    if a.kind in ("differs", "hard"):
        legend = "green = S6h (ours), magenta = SAM 3 'whale', yellow = pixel truth whale (where drawn)"
    elif a.kind == "markers":
        legend = "fill: red rail, yellow tape, blue deck, green land, white = 'cleat' instances; title = truth class, marker"
    else:
        legend = "fill: magenta person, red rail, yellow tape, blue deck, purple pontoon, orange other vessel, green land, cyan gear"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), grid(tiles, a.ncol, legend), [cv2.IMWRITE_JPEG_QUALITY, 85]); print(a.out, len(tiles))


if __name__ == "__main__":
    main()
