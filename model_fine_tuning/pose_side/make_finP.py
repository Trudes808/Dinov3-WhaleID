"""Point-based dorsal cut-outs ('finP'): the whole fin-and-hump triangle (the researcher's front base, back base, tip; the
user's decision of 2026-11-17: "we want that whole region") with a margin, from the head's predicted points; turn and
mirror kept from parts_up exactly as finB does, so only the centre and the scale differ from finB. Written next to the
cut-outs v3 sets so that model_fine_tuning/cutouts_v3/embed.py can read them once SET_PART has 'finP': 'fin'.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.pose_side.make_finP PRED.json --days 2025-01-15,2025-02-27,2025-03-14 [--set finP]

Geometry (2560 px): base = |front - back|, h = the tip's height over the base line. The front base is the head's
weakest point (cross-day 64 % within a quarter base), so the base is clamped to [1.8, 4.0] x h along the predicted
direction (real clicks: base / h quartiles 1.9 / 2.7 / 3.5). Centre = the middle of the triangle's box; side =
MARGIN x the triangle's longer extent. Workers are spawned (rawpy's OpenMP deadlocks under fork, as in make.py). Uses the cut-out code's own readers (read-only) and the full-resolution source.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import get_context

import cv2
import numpy as np

from . import guard
from ..cutouts_v3 import CACHE, DAYS
from ..cutouts_v3 import make as MK

MARGIN = 1.6
PRED = {}


def _init(pred_path, days, set_name, margin=None):
    global SET, MARGIN
    SET = set_name
    if margin:
        MARGIN = margin
    PRED.update(json.loads(open(pred_path).read()))
    MK._init(days)


def triangle(p):
    F, B, T = (np.array(p[n], float) for n in ("dorsal_fin_front", "dorsal_fin_back", "dorsal_fin_tip"))
    d = F - B; base = np.linalg.norm(d) + 1e-6; u = d / base; n = np.array([-u[1], u[0]]); h = abs((T - B) @ n) + 1e-6
    base_c = float(np.clip(base, 1.8 * h, 4.0 * h)); F = B + u * base_c
    P = np.array([F, B, T]); lo, hi = P.min(0), P.max(0)
    return (lo + hi) / 2, MARGIN * float(max(hi - lo)), dict(base=round(base, 1), base_used=round(base_c, 1), h=round(h, 1))


def work(job):
    day, stem, source, _fk, gk = job
    from PIL import Image
    from ..cluster.chart import decode_full
    from ..cluster.instances_v2 import old_instances
    from ..common import geometry, paths
    cv2.setNumThreads(1); d = DAYS / day; rows = []
    try:
        keys = [k for k in gk if k in PRED and (d / "parts_up" / f"{k}__fin.npz").exists()]
        if not keys:
            return stem, rows, None
        W, H = Image.open(d / "images" / f"{stem}.jpg").size
        okeys, omasks = old_instances(d, stem, W, H, geometry.load_folded_table(paths.FOLDED_TABLE)); old = dict(zip(okeys, omasks))
        full = decode_full(source); scale_full = full.shape[1] / W
        for k in keys:
            if k not in old:
                continue
            z = np.load(d / "parts_up" / f"{k}__fin.npz"); M_up = z["M"].astype(float); mirrored = bool(z["mirrored"])
            R = M_up[:, :2] / float(np.hypot(M_up[0, 0], M_up[0, 1]))
            c, side, info = triangle(PRED[k])
            M = MK.to_raw_M(R, c, side, scale_full); chart = MK.warp(full, M); whale = MK.warp_mask2560(old[k], M, scale_full)
            if mirrored:
                chart, whale = chart[:, ::-1].copy(), whale[:, ::-1].copy()
            MK.save(CACHE / day / SET, k, chart, dict(M=M, mirrored=mirrored, whale=whale, size=MK.SIZE))
            rows.append(dict(key=k, variant=SET, **info))
        return stem, rows, None
    except Exception as e:  # noqa: BLE001
        return stem, rows, f"{type(e).__name__}: {e}"


SET = "finP"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pred"); ap.add_argument("--days", required=True); ap.add_argument("--set", default="finP"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--margin", type=float, help="side / the triangle's extent (default MARGIN); finP4 4.1 = B's median zoom, finP6 6.0 = A's")
    a = ap.parse_args(); days = guard(a.days.split(","))
    for day in days:
        jobs = [j for j in MK.jobs_of(day) if j[4]]; rows = []; err = 0
        with get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.pred, [day], a.set, a.margin)) as pool:
            for stem, rr, e in pool.imap_unordered(work, jobs, chunksize=1):
                rows += rr; err += bool(e)
                if e:
                    print("  error", stem[-40:], e, flush=True)
        (CACHE / day / a.set).mkdir(parents=True, exist_ok=True)
        (CACHE / day / a.set / "meta.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{day}: {len(rows)} finP cut-outs, {err} errors", flush=True)


if __name__ == "__main__":
    main()
