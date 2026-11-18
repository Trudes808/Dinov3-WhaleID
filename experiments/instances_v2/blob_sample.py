"""Scoring b, step 1: a random sample of the blobs the 25 % clean-up (canon/predict.py keep_main_components) deletes,
stratified by blob size, drawn as numbered tiles for by-eye labelling (BLIND: the tiles do not show the filter score).
Tile: left the blob zoomed (magenta line drawn just outside it), right the whole photo (yellow = the zoom window,
red = the photo's largest blob).

    PYTHONPATH=. .venv/bin/python experiments/instances_v2/blob_sample.py [--per-bin 40] [--seed 0]            the first sample (blob_sample.json, sheets lab_NN)
    PYTHONPATH=. .venv/bin/python experiments/instances_v2/blob_sample.py --seed 1 --min-area 1000 --exclude blob_sample.json --out blob_sample_test.json --tag labtest
                                                                    the held-out sample for the blob gate (>= 1000 px)

-> experiments/instances_v2/blob_sample.json (the sample: day, stem, blob id, area, box) and scratchpad sheets lab_NN.jpg
"""
import argparse, json, random, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common as C

BINS = [100, 300, 1000, 3000, 10000, 10 ** 9]
ap = argparse.ArgumentParser(); ap.add_argument("--per-bin", type=int, default=40); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--min-area", type=int, default=100); ap.add_argument("--tag", default="lab"); ap.add_argument("--out", default="blob_sample.json")
ap.add_argument("--exclude", default=None, help="an earlier sample json whose blobs are not drawn again"); a = ap.parse_args()
EXCL = {(s["day"], s["stem"], s["b"]) for s in json.loads((Path(__file__).parent / a.exclude).read_text())} if a.exclude else set()
pool = {k: [] for k in range(len(BINS) - 1)}
for day in C.TRAINVAL:
    for stem, r in C.blob_rows(day).items():
        if not r["blobs"]:
            continue
        big = r["blobs"][0]["area"]
        for b in r["blobs"][1:]:
            if b["area"] < 0.25 * big and b["area"] >= a.min_area and (day, stem, b["b"]) not in EXCL:
                k = int(np.searchsorted(BINS, b["area"], side="right") - 1); pool[k].append((day, stem, b["b"], b["area"], b["bbox"]))
rng = random.Random(a.seed); sample = []
for k, v in pool.items():
    rng.shuffle(v); sample += [{"day": d, "stem": s, "b": b, "area": ar, "bbox": bb, "bin": f"{BINS[k]}-{BINS[k + 1] if BINS[k + 1] < 10 ** 9 else 'inf'}"} for d, s, b, ar, bb in v[:a.per_bin]]
    print(f"bin {BINS[k]}-{BINS[k + 1]}: {len(v)} deleted blobs on train + val days, {min(a.per_bin, len(v))} sampled")
rng.shuffle(sample)
for i, s in enumerate(sample):
    s["tile"] = i + 1
print("days in the sample:", len({s["day"] for s in sample}))
(Path(__file__).parent / a.out).write_text(json.dumps(sample, indent=0))


if True:
    tiles = [C.blob_tile(s) for s in sample]
    print(C.write_sheets(tiles, [{k: s[k] for k in ("tile", "day", "stem", "b", "area")} for s in sample], a.tag, cols=3, per=18))
