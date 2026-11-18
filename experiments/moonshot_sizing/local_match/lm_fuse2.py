"""Calibrated fusion (WildFusion-style), fitted on TRAIN days only: BASE -> log-odds by Platt scaling (a logistic on
the score; isotonic steps would tie pairs and cost AUC by themselves), local score -> P(same) by isotonic regression,
and the local evidence is added only when it is POSITIVE (the val screen showed that a low inlier count is no evidence of
"different": letting it pull pairs down lowered AUC):
    fused = logit_Platt(BASE) + w * max(0, logit(P_local) - logit(prior))
Pairs without a local score keep the BASE term, so the coverage is BASE's.

    python lm_fuse2.py VARIANT PART SCORE [MIRROR] [GEO]
Protocol: (1) the 20 bench repetitions: the isotonic maps are fitted on the 16 pool days (BASE on a pool day = head
fitted on the other 20 train days, computed once), the fusion is read on val + the 5 set-aside days with BASE fitted on
the 16 pool days, paired against BASE on the same pairs; (2) val with everything fitted on the 21 train days.
"""
import functools
import json
import os
import sys

import numpy as np
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm_score as SC  # noqa: E402
from lm_lib import bench  # noqa: E402

EPS = 1e-3


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


@functools.lru_cache(None)
def base_loo(part):
    """BASE scores on each train day from a head fitted on the other 20 train days."""
    return {d: bench.base_scores(part, [d], [o for o in bench.TRAIN if o != d])[d] for d in bench.TRAIN}


def fit_iso(S, days, part):
    x, y = [], []
    for d in days:
        same = {(a, b): s for a, b, s in bench.pairs(d, part)}
        for p, v in S.get(d, {}).items():
            x.append(v); y.append(same[p])
    x, y = np.array(x, float), np.array(y, float)
    if len(x) < 2:                                     # no local scores on the fit days: no local evidence
        x, y = np.array([0.0, 1.0]), np.array([0.5, 0.5])
    return IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(x, y), float(y.mean())


def fit_platt(S, days, part):
    from sklearn.linear_model import LogisticRegression
    x, y = [], []
    for d in days:
        same = {(a, b): s for a, b, s in bench.pairs(d, part)}
        for p, v in S.get(d, {}).items():
            x.append([v]); y.append(same[p])
    return LogisticRegression().fit(np.array(x), np.array(y))


GATE = 'gate' in sys.argv


def same_flank(d, a, b):
    fa, fb = bench.flank(d, a), bench.flank(d, b)
    return bool(fa) and fa == fb and len(fa) == 1


def fuse(B, L, isoB, isoL, prior_l, days, w, part='fluke'):
    out = {}
    for d in days:
        o = {}
        for p, bv in B.get(d, {}).items():
            f = float(isoB.decision_function([[bv]])[0])
            if p in L.get(d, {}) and not (GATE and part == 'dorsal' and not same_flank(d, *p)):
                f += w * max(0.0, logit(isoL.predict([L[d][p]])[0]) - logit(prior_l))
            o[p] = float(f)
        out[d] = o
    return out


def main():
    variant, part, score = sys.argv[1:4]
    mirror = sys.argv[4] if len(sys.argv) > 4 else 'max'
    geo = len(sys.argv) > 5 and sys.argv[5] == 'geo'
    L = SC.cluster_scores(variant, part, bench.TRAIN + bench.VAL, mirror, geo=geo, no_fallback=True)[score]
    tag = f'local_match_{variant}_{score}' + (f'_{mirror}' if part == 'dorsal' else '') + ('_isofuse_gate' if GATE else '_isofuse')
    BL = base_loo(part)
    res = {}
    W = (1.0, 0.5)
    cand, base = {w: [] for w in W}, []
    for hold, pool in bench.splits():
        test, fit = bench.VAL + hold, pool[:16]
        isoB = fit_platt(BL, fit, part)
        isoL, pl = fit_iso(L, fit, part)
        B = bench.base_scores(part, test, fit)            # BASE fitted once per repetition, as in bench.compare
        only = {d: set(B.get(d, {})) for d in test}
        base.append(bench.measure(part, B, test, only))
        for w in W:
            cand[w].append(bench.measure(part, fuse(B, L, isoB, isoL, pl, test, w, part), test, only))
    isoB = fit_platt(BL, bench.TRAIN, part)
    isoL, pl = fit_iso(L, bench.TRAIN, part)
    Bv = bench.base_scores(part, bench.VAL, bench.TRAIN)
    nl = sum(len(L.get(d, {})) for d in bench.TRAIN + bench.VAL)
    nt = sum(len(bench.pairs(d, part)) for d in bench.TRAIN + bench.VAL)
    print(f'local scores on {nl} of {nt} pairs of the 24 days; isotonic fitted on {sum(len(L.get(d, {})) for d in bench.TRAIN)} train-day pairs')
    for w in W:
        v = bench.compare(f'{tag}{w}', part, fuse(Bv, L, isoB, isoL, pl, bench.VAL, w, part), bench.VAL, bench.TRAIN, quiet=True, save=False)
        print(f'== Platt(BASE) + positive isotonic local evidence, local weight {w}: {tag} | {part} (coverage = BASE pairs)')
        print(bench._line('fused, val + set-aside', cand[w])); print(bench._line('BASE (16 days)', base)); print(bench._delta('fused - BASE', cand[w], base))
        print(f'   val (fit on 21): fused {bench._row(v["cand"])}\n                    BASE  {bench._row(v["base"])}')
        for rr in sorted(v['flank']['cand']):
            print(f'      val, {rr:<16s} fused AUC {v["flank"]["cand"][rr]["auc"]:.3f} P90 {100 * v["flank"]["cand"][rr]["r90"]:.1f} % | BASE AUC {v["flank"]["base"][rr]["auc"]:.3f} P90 {100 * v["flank"]["base"][rr]["r90"]:.1f} %')
        res[str(w)] = dict(held=dict(cand=cand[w], base=base), val=v, local_pairs=nl, total_pairs=nt)
    json.dump(res, open(bench.RESULTS / f'{tag}__{part}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
