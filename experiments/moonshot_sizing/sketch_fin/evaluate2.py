"""sketch_fin step 6: where does the fitted gain come from, viewing quality or drawing content? (CPU, train + val only)

Logistic regression, fitted leave one day out (never on the day it is read on), three feature sets:
  A  HW-b6 cosine
  B  A + viewing quality only (both fins side views and >= 40 px, frames per cluster, fin height, base ratio, same
     side by keypoints, HW x well-seen) -- nothing from the drawing's content
  C  B + drawing content (contour distance, missing marks, nick disagreement, shape class differs, mark correlation)
Read as: exclusion (rank by P(different), ruled out at precision 0.95 / 0.90, ties against) both in-sample ranking
(one pooled threshold, as evaluate.py) and with an honest threshold (fitted on the other 23 days, applied to the held-out
day: realised precision and share ruled out); positive side (found at P 0.95 / 0.90 pooled, AUC).

    python evaluate2.py > evaluate2.log
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.argv = sys.argv[:1]
sys.path.insert(0, str(HERE.parent))
import bench  # noqa: E402
from check_hw import at_prec, hw_feats, pair_scores  # noqa: E402
from size_levers import auc  # noqa: E402
from evaluate import ruled_out, rel, DD  # noqa: E402

CACHE = Path('/home/sat3737/NOAA_whales/moonshot_cache/sketch_fin')
ALL = bench.TRAIN + bench.VAL
F = {d: pickle.load(open(CACHE / 'pairs' / f'{d}.pkl', 'rb')) for d in ALL}
CLU = {d: pickle.load(open(CACHE / 'clusters' / f'{d}.pkl', 'rb')) for d in ALL}
HW = {d: pair_scores(d, hw_feats(d)) for d in ALL}


def x_of(d, p, f, hw, kind):
    a, b = (CLU[d][p[0]], CLU[d][p[1]])
    well = float(f['well'])
    x = [hw]
    if kind in ('B', 'C'):
        x += [well, hw * well, min(a['n'], b['n']), np.log(min(a['H'], b['H']) + 1), min(a['base'], b['base']), float(f['sameside'])]
    if kind == 'C':
        dc = f.get('dc')
        x += [dc if dc is not None else 0.0, float(f.get('miss', 0) or 0) if f.get('sameside') else 0.0, float(f.get('nick_dis') or 0),
              float(f['cls_a'] is not None and f['cls_b'] is not None and f['cls_a'] != f['cls_b']),
              f.get('mark_ncc') if f.get('mark_ncc') is not None else 0.0, float(f.get('mark_ncc') is None)]
    return x


def fit_scores(kind):
    out = {}
    data = {d: [(p, x_of(d, p, f, HW[d][p], kind), f['same']) for p, f in F[d].items() if p in HW[d]] for d in ALL}
    for d in ALL:
        X = np.array([r[1] for o in ALL if o != d for r in data[o]])
        y = np.array([not r[2] for o in ALL if o != d for r in data[o]])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        lr = LogisticRegression(max_iter=3000, C=1.0).fit((X - mu) / sd, y)
        if data[d]:
            Xd = (np.array([r[1] for r in data[d]]) - mu) / sd
            out[d] = dict(zip([r[0] for r in data[d]], lr.decision_function(Xd).astype(float)))   # higher = different
        else:
            out[d] = {}
    return out


def pooled(S, days, sel=None):
    pos, neg = [], []                                          # pos = different
    for d in days:
        for p, v in S[d].items():
            if sel and not sel(d, p):
                continue
            (neg if F[d][p]['same'] else pos).append(v)
    return np.array(pos), np.array(neg)


def honest(S, prec, days):
    """threshold from the other days at ruled-out precision prec, applied to each held-out day."""
    tp = fp = ndiff = 0
    for d in days:
        others = [o for o in ALL if o != d]
        pos, neg = pooled(S, others)
        thr = ruled_out(pos, neg, prec)[1]
        for p, v in S[d].items():
            s = F[d][p]['same']
            ndiff += not s
            if v >= thr:
                tp += not s
                fp += s
    return tp, fp, ndiff


if __name__ == '__main__':
    OUT = {}
    S = {'A  HW only': {d: {p: -v for p, v in HW[d].items() if p in F[d]} for d in ALL}}
    for k, name in (('A', 'A  LR HW'), ('B', 'B  LR HW + viewing quality'), ('C', 'C  LR HW + quality + drawing')):
        S[name] = fit_scores(k)
    n_all = sum(1 for d in ALL for a, b, s in bench.pairs(d, 'dorsal') if not s)
    print(f'pairs with HW and a drawing: {sum(len(v) for v in S["A  LR HW"].values())} of {sum(len(bench.pairs(d, "dorsal")) for d in ALL)}; different on the bench {n_all}')
    for tag, days in (('all 24 days', ALL), ('val 3 days', bench.VAL)):
        print(f'\n-- {tag}: exclusion (rank by P(different), ties against) and the positive side')
        for name, Sc in S.items():
            pos, neg = pooled(Sc, days)
            o95, o90 = ruled_out(pos, neg, .95)[0], ruled_out(pos, neg, .90)[0]
            h95, h90 = honest(Sc, .95, days), honest(Sc, .90, days)
            f95, f90 = at_prec(-neg, -pos, .95)[1], at_prec(-neg, -pos, .90)[1]
            r = dict(n_same=len(neg), n_diff=len(pos), auc=auc(pos, neg), out95=o95, out90=o90, honest95=h95, honest90=h90, found95=f95, found90=f90)
            OUT[f'{tag}|{name}'] = r
            print(f'   {name:<32s} {len(neg):5d} same / {len(pos):5d} diff  AUC {r["auc"]:.3f}  ruled out P0.95 {100 * o95 / len(pos):5.1f} %  P0.90 {100 * o90 / len(pos):5.1f} %'
                  f' | honest thr P0.95: {h95[0]} out, {h95[1]} same wrongly ({100 * h95[0] / max(1, h95[0] + h95[1]):.1f} % precision); P0.90: {h90[0]} out, {h90[1]} wrong ({100 * h90[0] / max(1, h90[0] + h90[1]):.1f} %)'
                  f' | same found P0.95 {100 * f95 / len(neg):4.1f} %  P0.90 {100 * f90 / len(neg):4.1f} %')
        if tag == 'all 24 days':
            for rr in ('same flank', 'opposite flanks', 'flank unknown', 'gap > 1 h'):
                sel = (lambda d, p: DD[d].gap(*p) > 3600) if rr == 'gap > 1 h' else (lambda d, p, rr=rr: rel(d, p) == rr)
                row = []
                for name in ('A  HW only', 'B  LR HW + viewing quality', 'C  LR HW + quality + drawing'):
                    pos, neg = pooled(S[name], days, sel)
                    o95, o90 = ruled_out(pos, neg, .95)[0], ruled_out(pos, neg, .90)[0]
                    OUT[f'{tag}|{name}|{rr}'] = dict(n_same=len(neg), n_diff=len(pos), auc=auc(pos, neg), out95=o95, out90=o90)
                    row.append(f'{name[:1]}: AUC {auc(pos, neg):.3f} out P0.95 {100 * o95 / len(pos):4.1f} % P0.90 {100 * o90 / len(pos):4.1f} %')
                print(f'      {rr:<16s} {len(neg):5d} same / {len(pos):5d} diff   ' + ' | '.join(row))
    # well-seen pairs only: HW low cosine alone
    for tag, days in (('all 24 days', ALL), ('val 3 days', bench.VAL)):
        pos, neg = pooled(S['A  HW only'], days, lambda d, p: F[d][p]['well'])
        print(f'\n   HW-b6 low cosine, both fins seen well (side view, >= 40 px) [{tag}]: {len(neg)} same / {len(pos)} diff, ruled out P0.95 '
              f'{100 * ruled_out(pos, neg, .95)[0] / len(pos):.1f} %  P0.90 {100 * ruled_out(pos, neg, .90)[0] / len(pos):.1f} %')
        pos, neg = pooled(S['A  HW only'], days, lambda d, p: not F[d][p]['well'])
        print(f'   HW-b6 low cosine, at least one fin not seen well [{tag}]: {len(neg)} same / {len(pos)} diff, ruled out P0.95 '
              f'{100 * ruled_out(pos, neg, .95)[0] / len(pos):.1f} %  P0.90 {100 * ruled_out(pos, neg, .90)[0] / len(pos):.1f} %')
    pickle.dump(S, open(CACHE / 'evaluate2_scores.pkl', 'wb'))
    json.dump(OUT, open(HERE / 'evaluate2_results.json', 'w'), indent=1, default=float)
