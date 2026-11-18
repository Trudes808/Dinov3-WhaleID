"""A small head fitted on train days only: per-cell cosines of the aligned tiles -> logistic regression.

Cluster pair -> the instance pair with the best posc score (the best-aligned look at both fins) -> its per-cell cosine
of day-centred block-11 tokens on a coarse grid (tile pooled POOL x POOL token cells), 0 where either cell has no whale,
plus the overlap indicators and posc itself.  Fitted per repetition on the 16 pool days' pairs (catalogue labels), scored
on val + the 5 set-aside days, compared with BASE fitted on the same 16 days (bench.compare); one fit on all 21 train
days for the val days.  Also fused with BASE (z-mean).

    python learned.py PART [POOL] [--fuse]
"""
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

import common  # noqa: F401
import bench
import fuse
import grid
import run_pairs
import score


def cell_tokens(part, day, pool, wmin=0.3):
    T, W = grid.pooled(part, day, 'tok11', pool)              # N h w D (whale-weighted means), N h w
    ok = W > wmin
    mu = (T * ok[..., None]).sum(0) / np.maximum(ok.sum(0), 1)[..., None]
    T = T - mu[None]
    T /= np.maximum(np.linalg.norm(T, axis=-1, keepdims=True), 1e-6)
    N, h, w, D = T.shape
    return T.reshape(N, h * w, D), ok.reshape(N, h * w)


def pair_features(part, day, pool, qual=True):
    S, _ = run_pairs.matrix(part, day, 'posc')
    L = score.load(part, day)
    idx = {k: i for i, k in enumerate(L['keys'])}
    keep = score.quality(part, day, score.QUAL if qual else None)
    T, ok = cell_tokens(part, day, pool)
    cl = {c: [idx[k] for k in ks if k in idx and keep[idx[k]]] for c, ks in bench.clusters(day, part).items()}
    F, P, Y = [], [], []
    for a, b, same in bench.pairs(day, part):
        ia, ib = cl.get(a, []), cl.get(b, [])
        if not ia or not ib:
            continue
        sub = S[np.ix_(ia, ib)]
        if np.all(np.isnan(sub)):
            continue
        r, c = np.unravel_index(np.nanargmax(sub), sub.shape)
        i, j = ia[r], ib[c]
        both = ok[i] & ok[j]
        cos = np.where(both, (T[i] * T[j]).sum(-1), 0.0)
        F.append(np.concatenate([cos, both.astype(np.float32), [np.nanmax(sub)]])); P.append((a, b)); Y.append(same)
    return np.array(F, np.float32), P, np.array(Y, bool)


def run(part, pool=3, do_fuse=False):
    days = bench.TRAIN + bench.VAL
    feats = {d: pair_features(part, d, pool) for d in days}

    def fit_predict(fit_days, test_days):
        X = np.concatenate([feats[d][0] for d in fit_days if len(feats[d][0])]); y = np.concatenate([feats[d][2] for d in fit_days if len(feats[d][0])])
        m = LogisticRegression(C=0.05, max_iter=2000, class_weight='balanced').fit(X, y)
        return {d: dict(zip(feats[d][1], m.decision_function(feats[d][0]))) if len(feats[d][0]) else {} for d in test_days}

    name = f'aligned_cells_p{pool}'
    cand, base, fc, fb = [], [], [], []
    for size in bench.SIZES:
        rows_c, rows_b = [], []
        for hold, pool_days in bench.splits():
            test = bench.VAL + hold
            S = fit_predict(pool_days[:size], test)
            r = bench.compare(name, part, S, test, pool_days[:size], quiet=True, save=False)
            rows_c.append(r['cand']); rows_b.append(r['base'])
            if size == bench.SIZES[-1] and do_fuse:
                B = bench.base_scores(part, test, pool_days[:size])
                Fz = fuse.fused([S], [1.0], test, B)
                rf = bench.compare(name + '_fuse', part, Fz, test, pool_days[:size], quiet=True, save=False)
                fc.append(rf['cand']); fb.append(rf['base'])
        print(bench._line(f'cells head, {size} days', rows_c)); print(bench._line(f'BASE, {size} days, same pairs', rows_b)); print(bench._delta('   cells - BASE', rows_c, rows_b))
        cand.append(rows_c); base.append(rows_b)
    for k in range(len(bench.SIZES) - 1):
        print(bench._delta(f'   cells, {bench.SIZES[k]} -> {bench.SIZES[k + 1]} days', cand[k + 1], cand[k]))
        print(bench._delta(f'   BASE,  {bench.SIZES[k]} -> {bench.SIZES[k + 1]} days', base[k + 1], base[k]))
    S = fit_predict(bench.TRAIN, bench.VAL)
    v = bench.compare(name, part, S, bench.VAL, bench.TRAIN, quiet=False, save=True)
    out = dict(name=name, part=part, kind='fitted_pairs', held={str(s): dict(cand=c, base=b) for s, c, b in zip(bench.SIZES, cand, base)}, val=v)
    if do_fuse:
        print(bench._line('cells head + BASE fused, 16 days', fc)); print(bench._line('BASE, 16 days', fb)); print(bench._delta('   fused - BASE', fc, fb))
        B = bench.base_scores(part, bench.VAL, bench.TRAIN)
        vf = bench.compare(name + '_fuse', part, fuse.fused([S], [1.0], bench.VAL, B), bench.VAL, bench.TRAIN, quiet=False, save=True)
        out['fuse'] = dict(held=dict(cand=fc, base=fb), val=vf)
    bench._save(name, part, out)


if __name__ == '__main__':
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3, '--fuse' in sys.argv)
