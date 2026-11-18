"""Seed-paired summary of the fin retrain test (score.py --retrain output folder): per part, first match per arm from
queries_<part>.json, the change for each seed pairing (RP6s0 - R0, RP6s1 - R0s1), the two R0 seeds against each
other, and the mean over seeds (RP6s0 + RP6s1 - R0 - R0s1) / 2, each with a 90 % interval from resampling whole stacks.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.pose_side.retrain_summary SCORE_DIR [--out experiments/pose_side/retrain_finP6.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir", type=Path); ap.add_argument("--out", type=Path); ap.add_argument("--reps", type=int, default=4000)
    a = ap.parse_args(); res = {}; rng = np.random.default_rng(0)
    for part in ("fin", "body", "fluke"):
        f = a.dir / f"queries_{part}.json"
        if not f.exists():
            continue
        Q = json.loads(f.read_text()); arms = ["R0", "R0s1", "RP6s0", "RP6s1"]
        Q = [q for q in Q if all(x in q["hit"] for x in arms)]
        H = {x: np.array([q["hit"][x] for q in Q], float) for x in arms + (["G"] if all("G" in q["hit"] for q in Q) else [])}
        st = np.array([f"{q['day']}|{q['stack']}" for q in Q]); ust, inv = np.unique(st, return_inverse=True)
        contrasts = {"RP6s0 - R0 (seed 0)": H["RP6s0"] - H["R0"], "RP6s1 - R0s1 (seed 1)": H["RP6s1"] - H["R0s1"],
                     "R0s1 - R0 (seed noise)": H["R0s1"] - H["R0"], "RP6s1 - RP6s0 (seed noise)": H["RP6s1"] - H["RP6s0"],
                     "mean over seeds RP6 - R0": (H["RP6s0"] + H["RP6s1"] - H["R0"] - H["R0s1"]) / 2}
        out = dict(n=len(Q), stacks=len(ust), first={k: round(float(v.mean()), 3) for k, v in H.items()})
        sums = {k: np.bincount(inv, weights=v, minlength=len(ust)) for k, v in contrasts.items()}; cnt = np.bincount(inv, minlength=len(ust))
        for k, v in contrasts.items():
            boots = []
            for _ in range(a.reps):
                s = rng.integers(0, len(ust), len(ust)); boots.append(sums[k][s].sum() / cnt[s].sum())
            lo, hi = np.percentile(boots, [5, 95])
            out[k] = dict(change=round(float(v.mean()), 3), ci90=[round(float(lo), 3), round(float(hi), 3)],
                          found=int((v > 0).sum()), lost=int((v < 0).sum()))
        res[part] = out
        print(f"\n{part}: {len(Q)} queries in {len(ust)} stacks; first match " + ", ".join(f"{k} {v}" for k, v in out["first"].items()))
        for k in contrasts:
            r = out[k]; print(f"   {k:30s} {r['change']:+.3f}  [{r['ci90'][0]:+.3f}, {r['ci90'][1]:+.3f}]  found {r['found']} / lost {r['lost']}")
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
