"""Fusion of label-free pair scores with BASE (z-score mean), on the bench's standard splits.

For each repetition the BASE head is fitted on the first 16 pool days (bench.splits), BASE scores and the candidate's
scores are z-scored over the test pairs (no labels), averaged with weights, and bench.compare scores the fused score
against BASE on the same pairs.  Pairs the candidate cannot score keep BASE's z alone ('fill'), so coverage = BASE's.

    fuse(name, part, {comp: S}, weights) -> same summary as bench.evaluate_pairs (saved to results/)
"""
import numpy as np

import common  # noqa: F401
import bench


def _z(S, days):
    v = np.array([s for d in days for s in S.get(d, {}).values()], float)
    m, sd = (v.mean(), v.std() + 1e-9) if len(v) else (0.0, 1.0)
    return {d: {p: (s - m) / sd for p, s in S.get(d, {}).items()} for d in days}


def fused(parts, weights, days, B):
    Z = [_z(S, days) for S in parts] + [_z(B, days)]
    w = list(weights) + [1.0]
    out = {}
    for d in days:
        out[d] = {}
        for p in B.get(d, {}):
            num = den = 0.0
            for z, wi in zip(Z, w):
                if p in z[d]:
                    num += wi * z[d][p]; den += wi
            out[d][p] = num / den
    return out


def fuse(name, part, comps, weights, reps=bench.REPS):
    cand, base = [], []
    for hold, pool in bench.splits(reps):
        test = bench.VAL + hold
        B = bench.base_scores(part, test, pool[:bench.SIZES[-1]])
        F = fused([comps[k] for k in comps], weights, test, B)
        r = bench.compare(name, part, F, test, pool[:bench.SIZES[-1]], quiet=True, save=False)
        cand.append(r['cand']); base.append(r['base'])
    B = bench.base_scores(part, bench.VAL, bench.TRAIN)
    F = fused([comps[k] for k in comps], weights, bench.VAL, B)
    v = bench.compare(name, part, F, bench.VAL, bench.TRAIN, quiet=True, save=False)
    print(f'== {name} | {part} | fusion with BASE ({", ".join(f"{k} x{w}" for k, w in zip(comps, weights))}), BASE alone where a component is missing')
    print(bench._line('fused, val + set-aside days', cand))
    print(bench._line('BASE (16 days), same pairs', base))
    print(bench._delta('fused - BASE', cand, base))
    print(f'   val days: fused {bench._row(v["cand"])}\n             BASE (21)  {bench._row(v["base"])}')
    for r in sorted(v['flank']['cand']):
        print(f'      val, {r:<16s} fused {bench._row(v["flank"]["cand"][r])} | BASE AUC {v["flank"]["base"][r]["auc"]:.3f}')
    out = dict(name=name, part=part, kind='fusion', comps=list(comps), weights=list(weights), held=dict(cand=cand, base=base), val=v)
    bench._save(name, part, out)
    return out
