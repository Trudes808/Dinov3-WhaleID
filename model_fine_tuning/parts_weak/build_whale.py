"""Certain-whale maps for W2 (CPU): pixels where background is certainly wrong.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.parts_weak.build_whale [--workers 3]
        -> CACHE/whale/<day>/<stem>.png   uint8 at the 2560-px frame, 1 = certainly whale, 0 = no claim
        -> CACHE/whale_summary.json

Per photo of index.json (training days only, not 2025-03-18), per old instance that is not skipped (not dropped as a
non-whale, not code X, not over a labelled not-a-whale blob):
  whale = old instance mask (DAY/cse, as predict.py) AND the live segmentation's blobs (DAY/instances_v2/blobs:
          seg_r04_ft4 P >= 0.5) AND, where seg_r04_ft4's probability is cached (seg_splash/pred/r04), P >= 0.9;
  then eroded: only pixels deeper than max(8 px, 5 % of the instance's long side) inside it are kept, and kept pieces
  under 200 px dropped. A thin awash back (thinner than ~10 % of the whale's length) or a wet flank edge thus makes
  no claim at all; only the solid interior of the whale does.
  Blobs labelled by eye as whale (blob_labels.json, role whale, sure) are added with the same erosion.
  Labelled not-a-whale / ignore blobs (grown 8 px) are removed.
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import get_context

import cv2
import numpy as np

from . import CACHE, DAYS, INDEX, check_train_day
from .build import BLOB_LABELS, R04

OUT = CACHE / "whale"
ERODE_MIN, ERODE_FRAC, MIN_PIECE, P_MIN = 8.0, 0.05, 200, 0.9


def deep(mask: np.ndarray, L: float) -> np.ndarray:
    if not mask.any():
        return mask
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    m = d > max(ERODE_MIN, ERODE_FRAC * L)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= MIN_PIECE
    return keep[lab]


def work(job):
    from ..cluster.instances_v2 import old_instances
    from ..common import geometry, paths
    cv2.setNumThreads(1)
    rec, wblobs = job
    day, stem, W, H = rec["day"], rec["stem"], rec["W"], rec["H"]
    try:
        check_train_day(day)
        d = DAYS / day
        table = geometry.load_folded_table(paths.FOLDED_TABLE)
        keys, masks = old_instances(d, stem, W, H, table)
        blobs = cv2.imread(str(d / "instances_v2" / "blobs" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        if blobs is None:
            blobs = np.zeros((H, W), np.uint8)
        conf = blobs > 0
        pr = R04 / day / f"{stem}.png"
        if pr.exists():
            p = cv2.imread(str(pr), cv2.IMREAD_UNCHANGED)
            conf &= cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR) >= round(P_MIN * 255)
        out = np.zeros((H, W), bool)
        ok = {i["key"]: i for i in rec["inst"] if not i.get("skipped")}
        for k, m in enumerate(masks):
            if keys[k] not in ok:
                continue
            out |= deep(m & conf, ok[keys[k]]["L"])
        for b in wblobs:
            bm = blobs == b["b"]
            ys, xs = np.nonzero(bm)
            if len(xs):
                out |= deep(bm, float(max(np.ptp(xs), np.ptp(ys)) + 1))
        neg = np.zeros((H, W), bool)
        for b in rec["nw"]:
            neg |= blobs == b["b"]
        if neg.any():
            out &= ~(cv2.dilate(neg.astype(np.uint8), np.ones((17, 17), np.uint8)) > 0)
        (OUT / day).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUT / day / f"{stem}.png"), out.astype(np.uint8))
        area = sum(int(m.sum()) for k, m in enumerate(masks) if keys[k] in ok)
        return {"day": day, "stem": stem, "px": int(out.sum()), "inst_px": area}, None
    except Exception as e:  # noqa: BLE001
        return {"day": day, "stem": stem}, f"{type(e).__name__}: {e}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    photos = json.loads(INDEX.read_text())["photos"]
    wl = [b for b in json.loads(BLOB_LABELS.read_text())["blobs"] if b["split"] == "train" and b["role"] == "whale" and b.get("sure")]
    by = {}
    for b in wl:
        by.setdefault((b["day"], b["stem"]), []).append(b)
    jobs = [(r, by.get((r["day"], r["stem"]), [])) for r in photos]
    t0 = time.time()
    rows, errs = [], 0
    with get_context("fork").Pool(a.workers) as pool:
        for i, (r, err) in enumerate(pool.imap_unordered(work, jobs, chunksize=4)):
            if err:
                errs += 1
                print("error", r, err, flush=True)
            else:
                rows.append(r)
            if i % 1000 == 999:
                print(f"  {i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    px, ipx = sum(r["px"] for r in rows), sum(r["inst_px"] for r in rows)
    summ = {"photos": len(rows), "errors": errs, "certain_whale_px": px, "instance_px": ipx, "share_of_instance_px": round(px / max(ipx, 1), 3),
            "photos_with_claim": sum(r["px"] > 0 for r in rows), "by_eye_whale_blobs": len(wl),
            "rule": {"erode_px": f"max({ERODE_MIN}, {ERODE_FRAC} x long side)", "min_piece_px": MIN_PIECE, "p_min_where_cached": P_MIN},
            "seconds": round(time.time() - t0)}
    (CACHE / "whale_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
