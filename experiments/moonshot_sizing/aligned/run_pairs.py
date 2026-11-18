"""Instance-pair score matrices per day and variant (cached), cluster-pair scores, bench.evaluate_pairs.

    python run_pairs.py PART VARIANT [VARIANT ...] [--qual none|default] [--agg max|top3] [--bench]
"""
import argparse
import time

import numpy as np

import score
from common import CACHE, bench

VARIANTS = {
    'pos': lambda p, d: score.pos_scores(p, d, 'tok11', center=False),
    'posc': lambda p, d: score.pos_scores(p, d, 'tok11', center=True),
    'posc_nb': lambda p, d: score.pos_scores(p, d, 'tok11', center=True, nb=True),
    'posc_top': lambda p, d: score.pos_scores(p, d, 'tok11', center=True, mode='top'),
    'pos8c': lambda p, d: score.pos_scores(p, d, 'tok8', center=True),
    'outline': lambda p, d: score.outline_scores(d),
    'outline_raw': lambda p, d: score.outline_scores(d, refine=False),
    'edge': lambda p, d: score.profile_scores(score.fluke_profiles(d), shift_cols=2, vshift=0.1),
}


def matrix(part, day, var):
    f = CACHE / 'mats' / part / var / f'{day}.npz'
    if f.exists():
        z = np.load(f)
        return z['S'], z['den']
    S, den = VARIANTS[var](part, day)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, S=S.astype(np.float32), den=den.astype(np.float32))
    return S, den


def pair_scores(part, var, qual='default', agg='max', days=None):
    out = {}
    for d in days or bench.TRAIN + bench.VAL:
        S, _ = matrix(part, d, var)
        keep = score.quality(part, d, None if qual == 'none' else score.QUAL)
        out[d] = score.cluster_pairs(part, d, S, keep, agg)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('part'); ap.add_argument('variants', nargs='+')
    ap.add_argument('--qual', default='default'); ap.add_argument('--agg', default='max'); ap.add_argument('--bench', action='store_true')
    a = ap.parse_args()
    for v in a.variants:
        t = time.time()
        S = pair_scores(a.part, v, a.qual, a.agg)
        print(f'{v}: matrices + pairs {time.time() - t:.0f} s', flush=True)
        if a.bench:
            name = f'aligned_{v}' + ('' if a.qual == 'default' else f'_q{a.qual}') + ('' if a.agg == 'max' else f'_{a.agg}')
            bench.evaluate_pairs(name, a.part, S)
