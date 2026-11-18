"""Fusion of the local-match score with BASE (the live matcher through the standard head).

    python lm_fuse.py VARIANT PART SCORE [MIRROR]
1. z-mean, no fitting: per day, z-score of BASE and of log1p(local score) over the day's scored pairs, mean of the two;
   pairs without a local score get the local z of 0 ("filled", full BASE coverage). Standard protocol: in each of the
   20 repetitions BASE is fitted on the 16 pool days and the fusion read on val + the 5 set-aside days, paired against
   BASE on the same pairs; plus val with BASE fitted on all 21 train days.
2. logistic, fitted on train days only (BASE on each train day from a head fitted on the other 20 train days),
   features [BASE, log1p(local), local missing]; read on val with bench.compare (BASE fitted on all 21).
Saved as results/local_match_<...>__PART.json.
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm_score as SC  # noqa: E402
from lm_lib import bench  # noqa: E402


def zday(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-9)


def zmean(B, L, days, w=0.5):
    out = {}
    for d in days:
        ps = list(B.get(d, {}))
        if not ps:
            out[d] = {}
            continue
        zb = zday([B[d][p] for p in ps])
        have = [p in L.get(d, {}) for p in ps]
        lv = np.array([np.log1p(L[d][p]) for p, h in zip(ps, have) if h])
        zl = np.zeros(len(ps))
        if len(lv) > 1:
            zl[np.array(have)] = zday(lv)
        out[d] = {p: float((1 - w) * zb[i] + w * zl[i]) for i, p in enumerate(ps)}
    return out


def main():
    variant, part, score = sys.argv[1:4]
    mirror = sys.argv[4] if len(sys.argv) > 4 else 'max'
    L = SC.cluster_scores(variant, part, bench.TRAIN + bench.VAL, mirror, no_fallback=True)[score]
    tag = f'local_match_{variant}_{score}' + (f'_{mirror}' if part == 'dorsal' else '')
    res = {}
    for w in (0.5,):
        cand, base = [], []
        for hold, pool in bench.splits():
            test = bench.VAL + hold
            B = bench.base_scores(part, test, pool[:16])
            F = zmean(B, L, test, w)
            r = bench.compare('x', part, F, test, pool[:16], quiet=True, save=False)
            cand.append(r['cand']); base.append(r['base'])
        Bv = bench.base_scores(part, bench.VAL, bench.TRAIN)
        v = bench.compare(f'{tag}_zmean{w}', part, zmean(Bv, L, bench.VAL, w), bench.VAL, bench.TRAIN, quiet=True, save=False)
        print(f'== fusion z-mean (local weight {w}) {tag} | {part}: all BASE pairs (local missing = 0)')
        print(bench._line('fused, val + set-aside', cand)); print(bench._line('BASE', base)); print(bench._delta('fused - BASE', cand, base))
        print(f'   val (BASE on 21): fused {bench._row(v["cand"])}\n                     BASE  {bench._row(v["base"])}')
        for rr in sorted(v['flank']['cand']):
            print(f'      val, {rr:<16s} fused AUC {v["flank"]["cand"][rr]["auc"]:.3f} P90 {100 * v["flank"]["cand"][rr]["r90"]:.1f} % | BASE AUC {v["flank"]["base"][rr]["auc"]:.3f} P90 {100 * v["flank"]["base"][rr]["r90"]:.1f} %')
        res[f'zmean{w}'] = dict(held=dict(cand=cand, base=base), val=v)
    # logistic on train days
    X, y = [], []
    for d in bench.TRAIN:
        others = [o for o in bench.TRAIN if o != d]
        B = bench.base_scores(part, [d], others)[d]
        same = {(a, b): s for a, b, s in bench.pairs(d, part)}
        for p, bv in B.items():
            lv = L.get(d, {}).get(p)
            X.append([bv, np.log1p(lv) if lv is not None else 0.0, lv is None]); y.append(same[p])
    X, y = np.array(X, float), np.array(y, int)
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
    Bv = bench.base_scores(part, bench.VAL, bench.TRAIN)
    Fv = {d: {p: float(clf.decision_function([[bv, np.log1p(L[d][p]) if p in L.get(d, {}) else 0.0, p not in L.get(d, {})]])[0]) for p, bv in Bv[d].items()} for d in bench.VAL}
    print(f'== fusion logistic (fitted on {len(y)} train-day pairs, coef {np.round(clf.coef_[0], 3).tolist()}) read on val:')
    v = bench.compare(f'{tag}_logit', part, Fv, bench.VAL, bench.TRAIN)
    res['logit'] = dict(coef=clf.coef_[0].tolist(), val=v)
    json.dump(res, open(bench.RESULTS / f'{tag}_fusion__{part}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
