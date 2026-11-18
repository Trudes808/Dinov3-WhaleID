"""Why S5 drops whale pixels (LOG 2026-11-17, user question: mirror reflections in the renders?). CPU, training days.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.why drop  [--names s5,s1,r05,s4]
        sure whale blobs of the TRAINING days (blob_labels.json) at 1280 px: per pixel the local thickness (2 x the
        largest inscribed radius within 7 px) and the position in its column of the blob (0 = top, 1 = bottom);
        per model the share of pixels with P < 0.05 ("confidently dropped") by thickness bin and by height band; and
        the S5 training label (CACHE/real_s5) of the same pixels: positive / ignore, and ring negatives near the blob.
        -> experiments/seg_splash/why_drop.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from multiprocessing import Pool

import cv2
import numpy as np

from . import common as C
from .data import blob_1280
from .predict import load_prob

TBINS = [0, 6, 10, 20, 40, 10 ** 6]
TNAMES = ["<6 px", "6-10", "10-20", "20-40", ">=40"]
HB = ["top 20 %", "20-40", "40-60", "60-80", "bottom 20 %"]


def geometry(m: np.ndarray):
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    th = 2 * cv2.dilate(dt, np.ones((7, 7), np.uint8))
    ys, xs = np.nonzero(m)
    top = np.full(m.shape[1], 10 ** 6); bot = np.full(m.shape[1], -1)
    np.minimum.at(top, xs, ys); np.maximum.at(bot, xs, ys)
    rel = (ys - top[xs]) / np.maximum(bot[xs] - top[xs], 1)
    return ys, xs, th[ys, xs], rel


def _one(args):
    b, names = args
    p = {n: load_prob(n, b["day"], b["stem"]) for n in names}; H, W = p[names[0]].shape
    m = blob_1280(C.blob_image(b["day"], b["stem"]), b["b"], (H, W))
    if m.sum() < 30:
        return None
    ys, xs, th, rel = geometry(m)
    tb = np.digitize(th, TBINS) - 1; hb = np.minimum((rel * 5).astype(int), 4)
    out = {"n": len(ys), "tb": np.bincount(tb, minlength=5).tolist(), "hb": np.bincount(hb, minlength=5).tolist()}
    for n in names:
        d = p[n][ys, xs] < 0.05
        out[n] = {"tb": np.bincount(tb[d], minlength=5).tolist(), "hb": np.bincount(hb[d], minlength=5).tolist(), "all": int(d.sum())}
    lab = cv2.imread(str(C.CACHE / "real_s5" / b["day"] / f"{b['stem']}.png"), cv2.IMREAD_UNCHANGED)
    if lab is not None:
        l = lab[ys, xs]
        out["label"] = {"pos_tb": np.bincount(tb[l == 1], minlength=5).tolist(), "ign_tb": np.bincount(tb[l == 255], minlength=5).tolist()}
        ring = (cv2.dilate(m.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0) & ~m & (lab == 0)     # negatives within 8 px of the blob
        below = ring.copy(); below[:max(0, ys.min()), :] = False
        # negatives that lie BELOW the blob's lower edge (the waterline side) vs above it
        bot = np.full(W, -1); np.maximum.at(bot, xs, ys); ry, rx = np.nonzero(ring)
        lower = (bot[rx] >= 0) & (ry > bot[rx])
        out["label"]["ring_neg"] = int(ring.sum()); out["label"]["ring_neg_below"] = int(lower.sum()); out["label"]["pos"] = int((l == 1).sum())
        out["mean_th"] = float(np.median(th))
    return out


def cmd_drop(a) -> None:
    names = a.names.split(",")
    B = [b for b in C.load_labels() if b["split"] == a.split and b["label"] == "whale" and b["sure"]]
    with Pool(8) as P:
        R = [r for r in P.imap_unordered(_one, [(b, names) for b in B], chunksize=8) if r]
    agg = {"blobs": len(R), "pixels_by_thickness": np.sum([r["tb"] for r in R], 0).tolist(), "pixels_by_height": np.sum([r["hb"] for r in R], 0).tolist()}
    for n in names:
        dt = np.sum([r[n]["tb"] for r in R], 0); dh = np.sum([r[n]["hb"] for r in R], 0)
        agg[n] = {"dropped_share": round(float(sum(r[n]["all"] for r in R) / sum(r["n"] for r in R)), 4),
                  "dropped_rate_by_thickness": dict(zip(TNAMES, np.round(dt / np.maximum(agg["pixels_by_thickness"], 1), 4).tolist())),
                  "dropped_rate_by_height": dict(zip(HB, np.round(dh / np.maximum(agg["pixels_by_height"], 1), 4).tolist())),
                  "dropped_pixels_by_thickness_share": dict(zip(TNAMES, np.round(dt / max(dt.sum(), 1), 4).tolist())),
                  "dropped_pixels_by_height_share": dict(zip(HB, np.round(dh / max(dh.sum(), 1), 4).tolist()))}
    L = [r for r in R if "label" in r]
    if L:
        pos = np.sum([r["label"]["pos_tb"] for r in L], 0); ign = np.sum([r["label"]["ign_tb"] for r in L], 0); tot = np.sum([r["tb"] for r in L], 0)
        agg["s5_label_on_these_pixels"] = {"positive_share_by_thickness": dict(zip(TNAMES, np.round(pos / np.maximum(tot, 1), 4).tolist())),
                                           "ignore_share_by_thickness": dict(zip(TNAMES, np.round(ign / np.maximum(tot, 1), 4).tolist()))}
        thin = [r for r in L if r["mean_th"] < 10]; thick = [r for r in L if r["mean_th"] >= 20]
        f = lambda S, k: round(float(sum(r["label"][k] for r in S) / max(sum(r["label"]["pos"] for r in S), 1)), 3)
        agg["ring_negatives_within_8px_per_positive"] = {"thin blobs (median thickness < 10 px)": f(thin, "ring_neg"), "thick blobs (>= 20 px)": f(thick, "ring_neg"),
                                                         "thin, below the lower edge only": f(thin, "ring_neg_below"), "thick, below the lower edge only": f(thick, "ring_neg_below"),
                                                         "n_thin": len(thin), "n_thick": len(thick)}
    (C.EXP / f"why_drop_{a.split}.json").write_text(json.dumps(agg, indent=1)); print(json.dumps(agg, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("drop"); p.add_argument("--names", default="s5,s1,r05,s4,s0"); p.add_argument("--split", default="train"); p.set_defaults(fn=cmd_drop)
    a = ap.parse_args(argv); a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
