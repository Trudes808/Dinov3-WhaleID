"""CPU: score fine-tuned HW-b6 embeddings on the 8 test days against raw HW-b6 on the SAME cluster pairs.

    OMP_NUM_THREADS=8 .venv/bin/python experiments/moonshot_sizing/fin_tune/score.py NAME [NAME ...]
NAME = a folder under CACHE/emb (s5_0, ...; 'raw' = raw HW-b6 through this folder's own embedding code) or 'cache' (raw
HW-b6 from public_reid's cached features). Cluster-pair score = best cosine over the two clusters' dorsal instances
(check_hw.pair_scores: unit vectors, no centring, nothing fitted). Training-day count is read from the name (s<N>_...).
Writes results/fin_tune_<NAME>__dorsal.json through bench.compare (candidate vs BASE fitted on pool[:N]) and
fin_tune/results/<NAME>.json with the comparison to raw HW-b6 (paired, same pairs; bootstrap over test days).
"""
import json
import sys
from pathlib import Path

import numpy as np

from common import CACHE, HERE, bench, hw_raw, split, unit
from size_levers import at_precision, auc

OUT = HERE / 'results'


def feats(name, day):
    if name == 'cache':
        F = hw_raw(day)
        keys = list(F)
        X = np.stack([F[k] for k in keys])
    else:
        z = np.load(CACHE / 'emb' / name / f'{day}.npz')
        keys, X = [str(k) for k in z['keys']], z['X']
    return dict(zip(keys, unit(X)))


def pair_scores(day, F):
    cl = {c: [k for k in ks if k in F] for c, ks in bench.clusters(day, 'dorsal').items()}
    M = {c: np.stack([F[k] for k in ks]) for c, ks in cl.items() if ks}
    return {(a, b): float((M[a] @ M[b].T).max()) for a, b, same in bench.pairs(day, 'dorsal') if a in M and b in M}


def scores(name, days):
    return {d: pair_scores(d, feats(name, d)) for d in days}


def subsets(days):
    """{tag: {day: set of pairs}}: flank relation (as bench), gap > 1 h, val days, set-aside days."""
    out = {}
    for d in days:
        D = bench.D(d)
        for a, b, _ in bench.pairs(d, 'dorsal'):
            fa, fb = bench.flank(d, a), bench.flank(d, b)
            r = 'flank unknown' if not fa or not fb else 'same flank' if fa == fb and len(fa) == 1 else 'opposite flanks' if not (fa & fb) else 'mixed'
            tags = [r] + (['gap > 1 h'] if D.gap(a, b) > 3600 else []) + (['val days'] if d in bench.VAL else ['set-aside days'])
            for t in tags:
                out.setdefault(t, {x: set() for x in days})[d].add((a, b))
    return out


def arrays(S, R, days, only=None):
    """per day: (cand pos, cand neg, raw pos, raw neg) on the pairs both score."""
    out = {}
    for d in days:
        cp, cn, rp, rn = [], [], [], []
        for a, b, same in bench.pairs(d, 'dorsal'):
            k = (a, b)
            if k in S[d] and k in R[d] and (only is None or k in only[d]):
                (cp if same else cn).append(S[d][k])
                (rp if same else rn).append(R[d][k])
        out[d] = tuple(np.array(v, float) for v in (cp, cn, rp, rn))
    return out


def m(pos, neg):
    n = max(1, len(pos))
    return dict(same=len(pos), diff=len(neg), auc=auc(pos, neg), ap=bench._ap(pos, neg), r95=at_precision(pos, neg, .95) / n, r90=at_precision(pos, neg, .90) / n)


def pooled(A, days):
    cat = lambda j: np.concatenate([A[d][j] for d in days])  # noqa: E731
    return m(cat(0), cat(1)), m(cat(2), cat(3))


def boot(A, days, n=1000, seed=0):
    r = np.random.default_rng(seed)
    days = [d for d in days if len(A[d][0]) and len(A[d][1])]
    res = {k: [] for k in ('auc', 'ap', 'r95', 'r90')}
    for _ in range(n):
        s = [days[i] for i in r.integers(len(days), size=len(days))]
        c, b = pooled(A, s)
        for k in res:
            res[k].append(c[k] - b[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in res.items()}


def report(name, S, R, test, ref='raw HW-b6 (cache)'):
    A = arrays(S, R, test)
    c, b = pooled(A, test)
    rows = dict(all=dict(cand=c, raw=b, delta={k: c[k] - b[k] for k in ('auc', 'ap', 'r95', 'r90')}, boot95=boot(A, test)))
    days_up = sum(1 for d in test if len(A[d][0]) and len(A[d][1]) and auc(A[d][0], A[d][1]) > auc(A[d][2], A[d][3]))
    rows['all']['days_auc_up'] = f'{days_up} of {sum(1 for d in test if len(A[d][0]) and len(A[d][1]))}'
    rows['per_day_auc'] = {d: [auc(A[d][0], A[d][1]), auc(A[d][2], A[d][3])] for d in test if len(A[d][0]) and len(A[d][1])}
    for t, only in subsets(test).items():
        As = arrays(S, R, test, only)
        cc, bb = pooled(As, test)
        rows[t] = dict(cand=cc, raw=bb, delta={k: cc[k] - bb[k] for k in ('auc', 'ap', 'r95', 'r90')})
    print(f'== fin_tune {name} vs {ref} | {len(test)} test days | same pairs: {c["same"]} same / {c["diff"]} different')
    for t in ['all', 'same flank', 'opposite flanks', 'flank unknown', 'mixed', 'gap > 1 h', 'val days', 'set-aside days']:
        if t not in rows:
            continue
        cc, bb, dd = rows[t]['cand'], rows[t]['raw'], rows[t]['delta']
        print(f'   {t:<16s} cand AUC {cc["auc"]:.3f} AP {cc["ap"]:.3f} P95 {100 * cc["r95"]:5.1f} % P90 {100 * cc["r90"]:5.1f} % | raw AUC {bb["auc"]:.3f} AP {bb["ap"]:.3f} '
              f'P95 {100 * bb["r95"]:5.1f} % P90 {100 * bb["r90"]:5.1f} % | diff AUC {dd["auc"]:+.3f} AP {dd["ap"]:+.3f} P95 {100 * dd["r95"]:+5.1f} P90 {100 * dd["r90"]:+5.1f}  ({cc["same"]}/{cc["diff"]})')
    bt = rows['all']['boot95']
    print(f'   bootstrap over test days, 95 % interval of the difference: AUC [{bt["auc"][0]:+.3f}, {bt["auc"][1]:+.3f}]  AP [{bt["ap"][0]:+.3f}, {bt["ap"][1]:+.3f}]  '
          f'P95 [{100 * bt["r95"][0]:+.1f}, {100 * bt["r95"][1]:+.1f}]  P90 [{100 * bt["r90"][0]:+.1f}, {100 * bt["r90"][1]:+.1f}]; per-day AUC up on {rows["all"]["days_auc_up"]}')
    return rows


def _steps(name):
    f = CACHE / 'runs' / name / 'steps.txt'
    return int(f.read_text()) if f.exists() else None


def main():
    hold, pool, test = split()
    R = scores('cache', test)
    OUT.mkdir(exist_ok=True)
    for name in sys.argv[1:]:
        S = scores(name, test)
        n = int(name[1:].split('_')[0]) if name[0] in 'se' and name[1:].split('_')[0].isdigit() else 16
        cmp = bench.compare(f'fin_tune_{name}', 'dorsal', S, test, pool[:n], quiet=True)
        rows = report(name, S, R, test)
        cov = (sum(len(S[d]) for d in test), sum(len(bench.pairs(d, 'dorsal')) for d in test))
        print(f'   bench.compare BASE (live features, head on pool[:{n}]) on its common pairs: AUC {cmp["base"]["auc"]:.3f}, candidate {cmp["cand"]["auc"]:.3f}; pairs scored {cov[0]} of {cov[1]}')
        json.dump(dict(name=name, test_days=test, train_days=pool[:n] if name not in ('raw', 'cache') else [], steps=_steps(name), pairs_scored=cov, vs_raw=rows,
                       bench_compare=dict(cand=cmp['cand'], base=cmp['base'])), open(OUT / f'{name}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
