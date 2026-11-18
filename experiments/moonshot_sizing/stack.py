"""The best evidence of the 2026-11-18 agent round stacked, at pair level and at day level (mine, CPU, read-only, no
eval day). Fins: HW-b6 raw cosine (public_reid cache) + the local_match agent's ALIKED / LightGlue homography inliers
(hom12, plain) as positive-only evidence. Flukes: BASE + the same local evidence (the agent's fusion). The fusion is
the local_match agent's form: Platt log-odds of the main score + max(0, isotonic log-odds of the local score - prior).

1  fins, the bench's 20 splits: BASE (16 days) | HW-b6 raw | HW-b6 + local, fitted on the 16 pool days, read on val +
   the 5 set-aside days, same pairs.
2  day level (size_levers score, start = perfect linking within 60 s): joins inside catalogue groups above the
   threshold that gives pair precision 0.95 / 0.90 on the OTHER 23 days; fin joins, fluke joins, both.

    python stack.py
"""
import functools
import itertools
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'local_match'))
import bench  # noqa: E402
import lm_score as SC  # noqa: E402
from check_hw import ALL, at_prec, hw_feats, pair_scores  # noqa: E402
from size_levers import Day, components, line, score, total  # noqa: E402

EPS = 1e-3


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


@functools.lru_cache(None)
def same_of(day, part):
    return {(a, b): s for a, b, s in bench.pairs(day, part)}


def xy(S, days, part):
    x, y = [], []
    for d in days:
        sm = same_of(d, part)
        for p, v in S.get(d, {}).items():
            x.append(v)
            y.append(sm[p])
    return np.array(x, float), np.array(y, float)


def fit(M, L, days, part):
    """Platt on the main score, isotonic on the local score, both on `days`."""
    x, y = xy(M, days, part)
    platt = LogisticRegression().fit(x[:, None], y)
    xl, yl = xy(L, days, part)
    if len(xl) < 2:
        return platt, None, 0.5
    return platt, IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(xl, yl), float(yl.mean())


def fuse(M, L, model, days, w=1.0):
    platt, iso, prior = model
    out = {}
    for d in days:
        ps = list(M.get(d, {}))
        if not ps:
            out[d] = {}
            continue
        f = platt.decision_function(np.array([M[d][p] for p in ps])[:, None])
        if iso is not None and L.get(d):
            has = np.array([p in L[d] for p in ps])
            if has.any():
                lv = iso.predict(np.array([L[d][p] for p, h in zip(ps, has) if h]))
                f[has] += w * np.maximum(0.0, logit(lv) - logit(prior))
        out[d] = dict(zip(ps, f.astype(float)))
    return out


def mean_rows(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in ('auc', 'ap', 'r95', 'r90')}


if __name__ == '__main__':
    HW = {d: pair_scores(d, hw_feats(d)) for d in ALL}
    Lf = SC.cluster_scores('aliked1024', 'dorsal', ALL, 'plain', no_fallback=True)['hom12']
    Lk = SC.cluster_scores('aliked1024', 'fluke', ALL, 'max', no_fallback=True)['hom12']
    print(f'local fin scores on {sum(len(v) for v in Lf.values())} pairs of {len(Lf)} days; fluke on {sum(len(v) for v in Lk.values())} pairs of {len(Lk)} days')

    print('\n== 1  fins, 20 splits, val + set-aside days, same pairs ==')
    rows = {'BASE (16 days)': [], 'HW-b6 raw': [], 'HW-b6 + local w 1.0': [], 'HW-b6 + local w 0.5': []}
    for hold, pool in bench.splits():
        test, fitd = bench.VAL + hold, pool[:16]
        B = bench.base_scores('dorsal', test, fitd)
        only = {d: set(B.get(d, {})) & set(HW[d]) for d in test}
        m = fit(HW, Lf, fitd, 'dorsal')
        rows['BASE (16 days)'].append(bench.measure('dorsal', B, test, only))
        rows['HW-b6 raw'].append(bench.measure('dorsal', HW, test, only))
        rows['HW-b6 + local w 1.0'].append(bench.measure('dorsal', fuse(HW, Lf, m, test, 1.0), test, only))
        rows['HW-b6 + local w 0.5'].append(bench.measure('dorsal', fuse(HW, Lf, m, test, 0.5), test, only))
    ref = rows['HW-b6 raw']
    for name, rs in rows.items():
        m = mean_rows(rs)
        da = np.array([a['auc'] - b['auc'] for a, b in zip(rs, ref)])
        d95 = np.array([a['r95'] - b['r95'] for a, b in zip(rs, ref)]) * 100
        print(f'   {name:<22s} AUC {m["auc"]:.3f}  AP {m["ap"]:.3f}  found at P 0.95 {100 * m["r95"]:5.1f} %  at 0.90 {100 * m["r90"]:5.1f} %'
              f'   against HW raw: AUC {da.mean():+.3f} (up {int((da > 0).sum())}/20), P 0.95 {d95.mean():+.1f} points (up {int((d95 > 0).sum())}/20)')

    print('\n== 2  day level, start = perfect linking within 60 s; thresholds from the other 23 days ==')
    Ds = {d: Day(d) for d in ALL}
    BL = {d: bench.base_scores('fluke', [d], [o for o in bench.TRAIN if o != d])[d] for d in bench.TRAIN}
    BL.update(bench.base_scores('fluke', bench.VAL, bench.TRAIN))
    fused = {'fin HW': {}, 'fin HW+local': {}, 'fluke BASE': {}, 'fluke BASE+local': {}}
    thr = {k: {} for k in fused}
    for d in ALL:
        others = [x for x in ALL if x != d]
        mf, mk = fit(HW, Lf, others, 'dorsal'), fit(BL, Lk, others, 'fluke')
        for name, part, S in (('fin HW', 'dorsal', fuse(HW, {}, (mf[0], None, .5), ALL)), ('fin HW+local', 'dorsal', fuse(HW, Lf, mf, ALL)),
                              ('fluke BASE', 'fluke', fuse(BL, {}, (mk[0], None, .5), ALL)), ('fluke BASE+local', 'fluke', fuse(BL, Lk, mk, ALL))):
            fused[name][d] = S[d]
            x, y = xy(S, others, part)
            thr[name][d] = {p: at_prec(x[y > 0], x[y == 0], p)[0] for p in (0.95, 0.90)}

    def run(names, prec, days):
        rows, nj, nw = [], 0, 0
        for d in days:
            D = Ds[d]
            pr = [(a, b) for a, b in itertools.combinations(D.ids, 2) if D.maj[a] == D.maj[b] and D.gap(a, b) <= 60]
            for name in names:
                part = 'dorsal' if name.startswith('fin') else 'fluke'
                j = [p for p, v in fused[name][d].items() if v >= thr[name][d][prec]]
                nj += len(j)
                nw += sum(not same_of(d, part)[p] for p in j)
                pr += j
            rows.append(score(D, components(D.ids, pr)))
        return total(rows), nj, nw

    for tag, days in (('24 days', ALL), ('val', bench.VAL)):
        print(line(f'{tag}: no joins beyond 60 s', run([], 0.95, days)[0]))
        for names in (['fin HW'], ['fin HW+local'], ['fluke BASE'], ['fluke BASE+local'], ['fin HW+local', 'fluke BASE+local']):
            for prec in (0.95, 0.90):
                t, nj, nw = run(names, prec, days)
                print(line(f'   {" + ".join(names)} at P {prec:.2f} ({nj} joins, {nw} wrong)', t))
