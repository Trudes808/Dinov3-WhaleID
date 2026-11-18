"""Stratified sample of real whale instances for by-eye labelling, and BLIND sheets (photo crop + the instance's outline,
never the model's output).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.part_gaps.sample [--n-train 400 --n-val 200]

Strata: the view classifier's confident three-way answer (fluke / dorsal / other, else "uncertain") x instance size
(long side of the box in the 2560 frame: <200, 200-400, 400-800, >=800). Allocation proportional to sqrt(stratum
size), at least 8 per stratum; every sampled instance carries its weight = stratum size / sampled in stratum, so shares
and costs are population estimates. Not-whale instances (DAY/safe/not_whale.json drop) are left out: the pipeline
never gives them to the part model's consumers. Writes experiments/part_gaps/sample.json (keys + weights only) and
sheets SCRATCH/label_sheets/<split>_<nn>.jpg + .json (tile -> key).
"""

from __future__ import annotations

import argparse
import json
import random

import cv2
import numpy as np

from ..common import geometry, paths
from ..parts_seg.predict import instances_of
from . import common as C

TILE = 300
COLS, ROWS = 5, 4


def population(name="parts_seg_r05_a"):
    rows = C.arm_rows(name, C.TRAIN_DAYS + C.VAL_DAYS)
    pop = []
    for r in rows:
        if r["not_whale"] or not r["bbox"]:
            continue
        L = C.size_px(r["bbox"])
        pop.append({"key": r["key"], "day": r["day"], "stem": r["stem"], "split": r["split"], "bbox": r["bbox"],
                    "L": L, "size": C.size_bin(L), "vc": C.vc_stratum(r), "n_inst": r["n_inst"]})
    return pop


def allocate(pop, n, rng):
    strata = {}
    for p in pop:
        strata.setdefault((p["vc"], p["size"]), []).append(p)
    w = {k: np.sqrt(len(v)) for k, v in strata.items()}
    tot = sum(w.values())
    alloc = {k: min(len(v), max(8, int(round(n * w[k] / tot)))) for k, v in strata.items()}
    out = []
    for k, v in sorted(strata.items()):
        pick = rng.sample(v, alloc[k])
        for p in pick:
            out.append(dict(p, weight=len(v) / alloc[k], stratum=f"{k[0]}|{k[1]}"))
    return out, {f"{k[0]}|{k[1]}": [len(v), alloc[k]] for k, v in sorted(strata.items())}


_TABLE = None


def table():
    global _TABLE
    if _TABLE is None:
        t = geometry.load_folded_table(paths.FOLDED_TABLE)
        _TABLE = {"x_min_m": float(t["x_min_m"]), "length_m": float(t["length_m"])}
    return _TABLE


def instance_mask_2560(day, stem, k, W, H):
    masks, (x0, y0, s) = instances_of(day, stem, table())
    if k >= len(masks):
        return None
    M = np.array([[1 / s, 0, x0], [0, 1 / s, y0]], dtype=np.float64)
    return cv2.warpAffine(masks[k].astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0


def blind_tile(p, size=TILE, outline=True, margin=1.4):
    day, stem = p["day"], p["stem"]
    C.check_day(day)
    bgr = cv2.imread(str(C.DAYS_ALL / day / "images" / f"{stem}.jpg"))
    H, W = bgr.shape[:2]
    sc = W / 2560.0
    x1, y1, x2, y2 = [v * sc for v in p["bbox"]]
    side = max(x2 - x1, y2 - y1) * margin + 30 * sc
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    gx0, gy0 = cx - side / 2, cy - side / 2
    M = np.array([[size / side, 0, -gx0 * size / side], [0, size / side, -gy0 * size / side]])
    img = cv2.warpAffine(bgr, M, (size, size), flags=cv2.INTER_AREA if side > size else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 0, 40))
    if outline:
        k = int(p["key"].rsplit("~", 1)[1])
        m = instance_mask_2560(day, stem, k, int(round(2560)), int(round(H / sc)))
        if m is not None:
            M2 = np.array([[size / side * sc, 0, -gx0 * size / side], [0, size / side * sc, -gy0 * size / side]])
            mm = cv2.warpAffine(m.astype(np.uint8), M2, (size, size), flags=cv2.INTER_NEAREST)
            cnts, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (255, 255, 0), 1)
    return img


def sheet(tiles, title, path):
    h, w = tiles[0].shape[:2]
    nrows = (len(tiles) + COLS - 1) // COLS
    out = np.full((26 + nrows * (h + 4), COLS * (w + 4), 3), 255, np.uint8)
    cv2.putText(out, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    for i, t in enumerate(tiles):
        r, c = divmod(i, COLS)
        out[26 + r * (h + 4): 26 + r * (h + 4) + h, c * (w + 4): c * (w + 4) + w] = t
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-sheets", action="store_true")
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    pop = population()
    res = {"note": "stratified sample of real whale instances (view classifier three-way x size); weight = stratum size / "
                   "sampled; labels in labels_ai.json are AI labels made blind to the model output", "splits": {}}
    sample = []
    for split, n in (("train", a.n_train), ("val", a.n_val)):
        sp = [p for p in pop if p["split"] == split]
        s, alloc = allocate(sp, n, rng)
        rng.shuffle(s)
        for i, p in enumerate(s):
            p["sheet"], p["tile"] = f"{split}_{i // (COLS * ROWS):02d}", i % (COLS * ROWS) + 1
        res["splits"][split] = {"population": len(sp), "sampled": len(s), "strata_pop_sampled": alloc}
        sample += s
    res["sample"] = [{k: p[k] for k in ("key", "day", "split", "bbox", "L", "size", "vc", "n_inst", "weight", "stratum", "sheet", "tile")}
                     for p in sample]
    C.OUT.mkdir(parents=True, exist_ok=True)
    (C.OUT / "sample.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["splits"], indent=1))
    if a.no_sheets:
        return 0
    sd = C.SCRATCH / "label_sheets"
    sd.mkdir(parents=True, exist_ok=True)
    by = {}
    for p in sample:
        by.setdefault(p["sheet"], []).append(p)
    for sh, ps in sorted(by.items()):
        ps.sort(key=lambda p: p["tile"])
        tiles = []
        for p in ps:
            t = blind_tile(p)
            cv2.putText(t, str(p["tile"]), (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
            cv2.putText(t, str(p["tile"]), (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
            tiles.append(t)
        sheet(tiles, f"{sh}: blind label sheet (photo + instance outline only)", sd / f"{sh}.jpg")
        (sd / f"{sh}.json").write_text(json.dumps({str(p["tile"]): p["key"] for p in ps}))
    print("sheets", len(by), "in", sd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
