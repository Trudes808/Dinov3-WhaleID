"""Bench runs for the public re-identification embeddings (extract.py). CPU only.

    OMP_NUM_THREADS=8 .venv/bin/python experiments/moonshot_sizing/public_reid/run_eval.py feat MODEL VARIANT PART
        -> bench.evaluate('public_reid_<MODEL>_<VARIANT>', PART, X)
    OMP_NUM_THREADS=8 .venv/bin/python experiments/moonshot_sizing/public_reid/run_eval.py fuse MODEL VARIANT PART raw|head [w]
        -> candidate (raw cosine, or the standard head fitted on the same 16 / 21 days as BASE) fused with BASE: weighted
           mean of z-scores of the cluster-pair scores (z over all test pairs pooled, so a monotone rescaling per
           component; w = candidate weight, default 0.5); pairs the candidate cannot score keep BASE's z-score alone.
           Scored with bench.compare on every standard split, against BASE on the same pairs.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bench  # noqa: E402

FEAT = Path('/home/sat3737/NOAA_whales/moonshot_cache/public_reid/feat')


def load(model, variant, days=None):
    X = {}
    for d in days or bench.TRAIN + bench.VAL:
        p = FEAT / model / variant / f'{d}.npz'
        if p.exists():
            z = np.load(p)
            X[d] = {str(k): v.astype(np.float32) for k, v in zip(z['keys'], z['X'])}
    return X


def _z(S, days):
    v = np.array([s for d in days for s in S.get(d, {}).values()], float)
    mu, sd = v.mean(), v.std() + 1e-9
    return {d: {p: (s - mu) / sd for p, s in S.get(d, {}).items()} for d in days}


def fused(part, X, test, fit, mode, w):
    C = bench.embed_scores(part, X, test, fit if mode == 'head' else None)
    B = bench.base_scores(part, test, fit)
    zc, zb = _z(C, test), _z(B, test)
    return {d: {p: (w * zc[d][p] + (1 - w) * s if p in zc[d] else s) for p, s in zb[d].items()} for d in test}


def fused_multi(part, cands, test, fit):
    """cands = [(X, mode)]: BASE and every candidate z-scored, equal weights, mean over the components a pair has."""
    zs = [_z(bench.base_scores(part, test, fit), test)]
    for X, mode in cands:
        zs.append(_z(bench.embed_scores(part, X, test, fit if mode == 'head' else None), test))
    out = {}
    for d in test:
        out[d] = {}
        for p in zs[0][d]:
            v = [z[d][p] for z in zs if p in z[d]]
            out[d][p] = float(np.mean(v))
    return out


def fuse_multi(name, part, cands):
    cand, base = [], []
    for hold, pool in bench.splits():
        test, fit = bench.VAL + hold, pool[:bench.SIZES[-1]]
        r = bench.compare(name, part, fused_multi(part, cands, test, fit), test, fit, quiet=True, save=False)
        cand.append(r['cand'])
        base.append(r['base'])
    v = bench.compare(name, part, fused_multi(part, cands, bench.VAL, bench.TRAIN), bench.VAL, bench.TRAIN, quiet=True, save=False)
    print(f'== {name} | {part} | equal-weight z fusion of BASE + {len(cands)} candidates, head on 16 days')
    print(bench._line('fused, val + set-aside days', cand))
    print(bench._line('BASE (16 days), same pairs', base))
    print(bench._delta('fused - BASE', cand, base))
    print(f'   val days (21): fused {bench._row(v["cand"])}\n                  BASE  {bench._row(v["base"])}')
    bench._save(name, part, dict(name=name, part=part, kind='fusion', mode='multi', held=dict(cand=cand, base=base), val=v))


def fuse_curve(name, part, cands):
    """The fusion with the head(s) fitted on 5 / 10 / 16 days: does the fused score keep gaining with more days?"""
    res = {}
    for size in bench.SIZES:
        cand, base = [], []
        for hold, pool in bench.splits():
            test, fit = bench.VAL + hold, pool[:size]
            r = bench.compare(name, part, fused_multi(part, cands, test, fit), test, fit, quiet=True, save=False)
            cand.append(r['cand'])
            base.append(r['base'])
        res[size] = (cand, base)
        print(bench._line(f'fused, head on {size} days', cand))
        print(bench._line(f'BASE,  head on {size} days', base))
        print(bench._delta('   fused - BASE', cand, base))
    for a, b in zip(bench.SIZES[:-1], bench.SIZES[1:]):
        print(bench._delta(f'   fused, {a} -> {b} days', res[b][0], res[a][0]))
        print(bench._delta(f'   BASE,  {a} -> {b} days', res[b][1], res[a][1]))
    bench._save(name, part, dict(name=name, part=part, kind='fusion_curve', held={str(k): dict(cand=v[0], base=v[1]) for k, v in res.items()}))


def fuse(name, part, X, mode, w):
    cand, base = [], []
    for hold, pool in bench.splits():
        test, fit = bench.VAL + hold, pool[:bench.SIZES[-1]]
        r = bench.compare(name, part, fused(part, X, test, fit, mode, w), test, fit, quiet=True, save=False)
        cand.append(r['cand'])
        base.append(r['base'])
    v = bench.compare(name, part, fused(part, X, bench.VAL, bench.TRAIN, mode, w), bench.VAL, bench.TRAIN, quiet=True, save=False)
    print(f'== {name} | {part} | fusion ({mode}, w {w}) with BASE, head on 16 days')
    print(bench._line('fused, val + set-aside days', cand))
    print(bench._line('BASE (16 days), same pairs', base))
    print(bench._delta('fused - BASE', cand, base))
    print(f'   val days (21): fused {bench._row(v["cand"])}\n                  BASE  {bench._row(v["base"])}')
    bench._save(name, part, dict(name=name, part=part, kind='fusion', mode=mode, w=w, held=dict(cand=cand, base=base), val=v))


if __name__ == '__main__':
    if sys.argv[1] in ('multi', 'curve'):          # multi|curve PART NAME model:variant:mode ...
        part, nm = sys.argv[2], sys.argv[3]
        cands = [(load(*c.split(':')[:2]), c.split(':')[2]) for c in sys.argv[4:]]
        (fuse_multi if sys.argv[1] == 'multi' else fuse_curve)(f'public_reid_{"fuse" if sys.argv[1] == "multi" else "curve"}_{nm}', part, cands)
        sys.exit()
    cmd, model, variant, part = sys.argv[1:5]
    X = load(model, variant)
    if cmd == 'feat':
        bench.evaluate(f'public_reid_{model}_{variant}', part, X)
    else:
        mode = sys.argv[5]
        w = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
        fuse(f'public_reid_fuse_{model}_{variant}_{mode}_w{w:g}', part, X, mode, w)
