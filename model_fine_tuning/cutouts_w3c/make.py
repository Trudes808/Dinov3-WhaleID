"""W3c-centred fin cut-outs from the full-resolution RAW, offline (nothing of a day folder is written).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_w3c.make train,val --seed s0 [--variants wide,tight] [--workers 10]

The fins of cut-outs v3 (cutouts_v3.make.jobs_of: instances the view classifier calls dorsal with a parts_up fin
cut-out), same keys. Turn and mirror of the parts_up fin cut-out kept (its npz M and "mirrored"); centre = centroid
of the W3c dorsal region (regions.py, largest connected region), side = k x L with L the region's extent along the
axis and k from RESULTS/rule.json (fixed before any identity score). No region: nothing is written and the fin keeps
today's cut-out (A = parts_up). -> CACHE/<day>/<seed>_<variant>/<key>.jpg (768 px) + .npz (M, mirrored, whale, size),
CACHE/<day>/meta_<seed>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import get_context

import numpy as np

from . import CACHE, DAYS, day_list
from .choose import load_rule
from ..cutouts_v3 import make as v3


def work(job):
    day, stem, source, keys, seed, variants, rule = job
    import cv2
    from PIL import Image
    from ..cluster.chart import decode_full
    from ..cluster.instances_v2 import old_instances
    from ..common import geometry, paths
    cv2.setNumThreads(1)
    d = DAYS / day; rows = []
    try:
        W, H = Image.open(d / "images" / f"{stem}.jpg").size
        okeys, omasks = old_instances(d, stem, W, H, geometry.load_folded_table(paths.FOLDED_TABLE)); old = dict(zip(okeys, omasks))
        regs = REG[day]; full = None
        for key in keys:
            r = regs.get(key)
            if key not in old or r is None or not r["up_fin"]:
                continue
            z = np.load(d / "parts_up" / f"{key}__fin.npz")
            M_up = z["M"].astype(float); mirrored = bool(z["mirrored"]); sf_up = float(z["scale_full"])
            R = M_up[:, :2] / float(np.hypot(M_up[0, 0], M_up[0, 1]))
            g = r["reg"]
            row = dict(key=key, src="A" if g is None else "w3c")
            if g is not None:
                L = float(g[f"L_{rule['axis']}"]); c = np.array(g["c"])
                cA = np.array(r["A"]["c"])
                row.update(L=round(L, 1), H=round(float(g[f"H_{rule['axis']}"]), 1), px2560=round(g["area2560"], 1), shift_L=round(float(np.linalg.norm(c - cA) / L), 3),
                           sideA_L=round(r["A"]["side"] / L, 3))
                if full is None:
                    full = decode_full(source)
                sf = full.shape[1] / W
                assert abs(sf - sf_up) < 1e-3 * sf_up, (sf, sf_up)
                for v in variants:
                    M = v3.to_raw_M(R, c, rule[v] * L, sf)
                    chart = v3.warp(full, M); whale = v3.warp_mask2560(old[key], M, sf)
                    if mirrored:
                        chart, whale = chart[:, ::-1].copy(), whale[:, ::-1].copy()
                    v3.save(CACHE / day / f"{seed}_{v}", key, chart, dict(M=M, mirrored=mirrored, whale=whale, size=v3.SIZE))
            rows.append(row)
        return stem, rows, None
    except Exception as e:  # noqa: BLE001
        import traceback
        return stem, rows, f"{type(e).__name__}: {e} {traceback.format_exc()[-400:]}"


REG: dict = {}


def _init(day, seed):
    REG[day] = json.loads((CACHE / "regions" / seed / f"{day}.json").read_text())


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("days"); ap.add_argument("--seed", default="s0"); ap.add_argument("--variants", default="wide,tight")
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv); rule = load_rule(); variants = a.variants.split(",")
    for day in day_list(a.days):
        t0 = time.time()
        jobs = [(day, s, src, gk, a.seed, variants, rule) for day_, s, src, fk, gk in v3.jobs_of(day, a.limit) if gk]
        rows, nerr = [], 0
        with get_context("spawn").Pool(min(a.workers, 10), initializer=_init, initargs=(day, a.seed)) as pool:
            for i, (stem, rr, err) in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
                rows += rr
                if err:
                    nerr += 1; print("  error", stem[-40:], err, flush=True)
        (CACHE / day).mkdir(parents=True, exist_ok=True)
        meta = CACHE / day / (f"meta_{a.seed}.jsonl" if not a.limit else f"meta_{a.seed}.limit.jsonl")
        meta.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{a.seed} {day}: {len(jobs)} photos, fins {len(rows)} (w3c {sum(r['src'] == 'w3c' for r in rows)}, A {sum(r['src'] == 'A' for r in rows)}), {nerr} errors, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
