"""sketch_fluke step 4: bench numbers (CPU, no eval day).

  1  drawing alone through bench.evaluate_pairs (no fitting): sided (ventral with ventral, dorsal with dorsal; max),
     ventral only, any side (side ignored)
  2  exclusion: rank the pairs by "differentness" (light-share difference on the region both drawings saw, ventral
     with ventral; or the 5-step class difference), share of the DIFFERENT-animal pairs ruled out before precision
     (share of different pairs among those ruled out) falls below 0.95 / 0.90; ties count against us (same pairs first).
     No fitting. Pooled over the 24 days and per split (val + 5 set-aside days, the bench's 20 splits).
  3  class repeatability: 5-step light class of the ventral drawing, same-animal vs different-animal cluster pairs
  4  fusion, the bench's 20 splits: Platt(BASE) + positive isotonic keypoint evidence (stack.py / lm_fuse2 form) and
     + the drawing (isotonic log-odds of the sided similarity, both signs) + exclusion veto; all fitted on the 16 pool
     days of the split, read on val + the 5 set-aside days, same pairs. BASE = the live features with the head fitted
     leave-one-day-out (train days: the other 20 train days; val: the 21 train days), as in stack.py section 2.
  5  veto: joins = BASE + keypoints above the threshold that gives pair precision 0.90 / 0.95 on the pool days; a join
     is vetoed when its differentness passes the threshold that gives 'different' precision 0.95 on the pool days.
    python run.py [1 2 3 4 5]
"""
import collections
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

import draw  # noqa: F401
import bench
from prep import CACHE
from size_levers import at_precision, auc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'local_match'))
ALL = bench.TRAIN + bench.VAL
P = pickle.load(open(CACHE / 'pair_scores.pkl', 'rb'))
S_all = P['S']
SAME = {d: {(a, b): s for a, b, s in bench.pairs(d, 'fluke')} for d in ALL}


LIGHT = 0.15     # a cluster drawing with >= 15 % light blade is taken as an underside view (the upper surface is dark)


def lightc(d, c):
    return P['cls'].get((d, c), {}).get('any', (0, 0.0))[1]


def sim_of(r, mode, d=None, p=None):
    if mode == 'light':
        return r['any'][0] if 'any' in r and min(lightc(d, p[0]), lightc(d, p[1])) >= LIGHT else None
    if mode == 'sided':
        v = [r[s][0] for s in ('ventral', 'dorsal') if s in r]
    elif mode == 'ventral':
        v = [r['ventral'][0]] if 'ventral' in r else []
    else:
        v = [r['any'][0]] if 'any' in r else []
    return max(v) if v else None


def scores(mode):
    out = {}
    for d in ALL:
        out[d] = {}
        for p, r in S_all[d].items():
            v = sim_of(r, mode, d, p)
            if v is not None:
                out[d][p] = v
    return out


def differentness(kind='light'):
    """{day: {pair: value}}, higher = more different. kind: light / class / contra on pairs of two light (underside)
    drawings; light_any / class_any = the same on every drawn pair (side unknown for dark drawings)."""
    out = {}
    for d in ALL:
        out[d] = {}
        for (a, b), r in S_all[d].items():
            if 'any' not in r:
                continue
            both = min(lightc(d, a), lightc(d, b)) >= LIGHT
            if not kind.endswith('_any') and not both:
                continue
            k = kind.replace('_any', '')
            if k == 'light':
                out[d][(a, b)] = r['any'][1]
            elif k == 'contra':
                out[d][(a, b)] = r['any'][3]
            else:
                out[d][(a, b)] = abs(P['cls'][(d, a)]['any'][0] - P['cls'][(d, b)]['any'][0])
    return out


def rule_out(Dv, days):
    """(ruled out at 0.95, at 0.90 as shares of ALL different pairs of the days; same at 0.95/0.90 of the covered
    different pairs; covered different, all different, covered same, base rate)."""
    pos, neg = [], []                      # pos = different pairs (what we want to rule out)
    ndiff = 0
    for d in days:
        for (a, b), same in SAME[d].items():
            ndiff += not same
            if (a, b) in Dv.get(d, {}):
                (neg if same else pos).append(Dv[d][(a, b)])
    pos, neg = np.array(pos, float), np.array(neg, float)
    r95, r90 = at_precision(pos, neg, 0.95), at_precision(pos, neg, 0.90)
    return dict(r95_all=r95 / max(ndiff, 1), r90_all=r90 / max(ndiff, 1), r95_cov=r95 / max(len(pos), 1), r90_cov=r90 / max(len(pos), 1),
                n95=r95, n90=r90, cov_diff=len(pos), cov_same=len(neg), all_diff=ndiff, auc=auc(pos, neg),
                base_diff=len(pos) / max(len(pos) + len(neg), 1))


def part1():
    for mode in ('any', 'light'):
        bench.evaluate_pairs(f'sketch_fluke_{mode}', 'fluke', scores(mode))


def part2():
    res = {}
    for kind in ('light_any', 'class_any', 'light', 'class', 'contra'):
        Dv = differentness(kind)
        pooled = rule_out(Dv, ALL)
        sp = [rule_out(Dv, bench.VAL + hold) for hold, _ in bench.splits()]
        m = {k: float(np.mean([s[k] for s in sp])) for k in sp[0]}
        res[kind] = dict(pooled=pooled, splits_mean=m)
        print(f'== exclusion by {kind} ({"every drawn pair" if kind.endswith("_any") else "two light underside drawings"})')
        for tag, r in (('24 days pooled', pooled), ('val + set-aside, mean of 20', m)):
            print(f"   {tag:<28s} covered {r['cov_diff']:.0f} different / {r['cov_same']:.0f} same (base rate different {r['base_diff']:.3f}), AUC (different vs same) {r['auc']:.3f}; "
                  f"ruled out at P 0.95: {100 * r['r95_all']:.2f} % of all different pairs ({100 * r['r95_cov']:.1f} % of covered), at 0.90: {100 * r['r90_all']:.2f} % ({100 * r['r90_cov']:.1f} %)")
    json.dump(res, open(HERE / 'exclusion.json', 'w'), indent=1)


def part3():
    rows = collections.defaultdict(list)
    for d in ALL:
        for (a, b), same in SAME[d].items():
            ca, cb = P['cls'].get((d, a), {}).get('any'), P['cls'].get((d, b), {}).get('any')
            if ca and cb:
                rows[same].append((ca[0], cb[0], ca[1], cb[1]))
    out = {}
    for same in (True, False):
        r = np.array(rows[same], float)
        dc = np.abs(r[:, 0] - r[:, 1])
        out['same' if same else 'different'] = dict(n=len(r), exact=float((dc == 0).mean()), within1=float((dc <= 1).mean()), ge2=float((dc >= 2).mean()),
                                                    light_corr=float(np.corrcoef(r[:, 2], r[:, 3])[0, 1]), mean_abs_light=float(np.abs(r[:, 2] - r[:, 3]).mean()))
        print(f"   {'same' if same else 'different'} animal pairs, both drawn: {len(r)}; class equal {100 * out['same' if same else 'different']['exact']:.1f} %, within one step "
              f"{100 * out['same' if same else 'different']['within1']:.1f} %, two or more steps apart {100 * out['same' if same else 'different']['ge2']:.1f} %; mean |light share difference| {out['same' if same else 'different']['mean_abs_light']:.3f}")
    cl = collections.Counter(v['any'][0] for v in P['cls'].values() if 'any' in v)
    print('   class histogram of the cluster drawings (1 dark .. 5 white):', dict(sorted(cl.items())))
    out['hist'] = dict(cl)
    json.dump(out, open(HERE / 'classes.json', 'w'), indent=1)


# ---- fusion ----------------------------------------------------------------------------------------------------------
def base_lodo():
    fp = CACHE / 'base_lodo.pkl'
    if fp.exists():
        return pickle.load(open(fp, 'rb'))
    BL = {d: bench.base_scores('fluke', [d], [o for o in bench.TRAIN if o != d])[d] for d in bench.TRAIN}
    BL.update(bench.base_scores('fluke', bench.VAL, bench.TRAIN))
    pickle.dump(BL, open(fp, 'wb'))
    return BL


def local():
    import lm_score
    return lm_score.cluster_scores('aliked1024', 'fluke', ALL, 'max', no_fallback=True)['hom12']


EPS = 1e-3


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def xy(S, days):
    x, y = [], []
    for d in days:
        for p, v in S.get(d, {}).items():
            x.append(v); y.append(SAME[d][p])
    return np.array(x, float), np.array(y, float)


def fit_iso(S, days):
    x, y = xy(S, days)
    return IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(x, y), float(y.mean())


def fused(BL, Lk, Dg, fit_days, test_days, use_local=True, use_draw=False, wd=1.0):
    from sklearn.linear_model import LogisticRegression
    x, y = xy(BL, fit_days)
    platt = LogisticRegression().fit(x[:, None], y)
    isoL, priL = fit_iso(Lk, fit_days) if use_local else (None, None)
    isoD, priD = fit_iso(Dg, fit_days) if use_draw else (None, None)
    out = {}
    for d in test_days:
        ps = list(BL.get(d, {}))
        if not ps:
            out[d] = {}; continue
        f = platt.decision_function(np.array([BL[d][p] for p in ps])[:, None])
        for k, p in enumerate(ps):
            if use_local and p in Lk.get(d, {}):
                f[k] += max(0.0, logit(isoL.predict([Lk[d][p]])[0]) - logit(priL))
            if use_draw and p in Dg.get(d, {}):
                f[k] += wd * (logit(isoD.predict([Dg[d][p]])[0]) - logit(priD))
        out[d] = dict(zip(ps, f.astype(float)))
    return out


def thr_at(S, days, prec):
    x, y = xy(S, days)
    o = np.lexsort((y, -x)); tp = np.cumsum(y[o]); pr = tp / np.arange(1, len(o) + 1)
    ok = np.where(pr >= prec)[0]
    return x[o][ok[-1]] if len(ok) else np.inf


def diff_thr(Dv, days, prec):
    """smallest differentness at which the 'different' precision on `days` is still >= prec (ties against us)."""
    x, y = xy(Dv, days)                    # y = same
    yd = 1 - y
    o = np.lexsort((yd, -x)); tp = np.cumsum(yd[o]); pr = tp / np.arange(1, len(o) + 1)
    ok = np.where(pr >= prec)[0]
    return x[o][ok[-1]] if len(ok) else np.inf


def part45():
    BL, Lk, Dg, Dv = base_lodo(), local(), scores('any'), differentness('contra')
    tags = ('BASE', 'BASE+kp', 'BASE+kp+drawing', 'BASE+kp+drawing w0.5')
    rows = {t: collections.defaultdict(list) for t in tags}
    veto = collections.defaultdict(lambda: np.zeros(4))
    for hold, pool in bench.splits():
        test, fitd = bench.VAL + hold, pool[:16]
        Sx = {'BASE': {d: BL[d] for d in test}, 'BASE+kp': fused(BL, Lk, Dg, fitd, test),
              'BASE+kp+drawing': fused(BL, Lk, Dg, fitd, test, use_draw=True), 'BASE+kp+drawing w0.5': fused(BL, Lk, Dg, fitd, test, use_draw=True, wd=0.5)}
        allp = {d: set(BL[d]) for d in test}
        cov = {d: set(BL[d]) & set(Dg[d]) for d in test}
        hi = {d: {p for p in BL[d] if bench.D(d).gap(*p) > 3600} for d in test}
        cov_hi = {d: cov[d] & hi[d] for d in test}
        side = {'both light': {d: {p for p in cov[d] if min(lightc(d, p[0]), lightc(d, p[1])) >= LIGHT} for d in test},
                'not both light': {d: {p for p in cov[d] if min(lightc(d, p[0]), lightc(d, p[1])) < LIGHT} for d in test}}
        for t in tags:
            rows[t]['all'].append(bench.measure('fluke', Sx[t], test, allp))
            rows[t]['drawn'].append(bench.measure('fluke', Sx[t], test, cov))
            rows[t]['drawn, gap > 1 h'].append(bench.measure('fluke', Sx[t], test, cov_hi))
            for k in side:
                rows[t][k].append(bench.measure('fluke', Sx[t], test, side[k]))
        rows['drawing alone'] = rows.get('drawing alone', collections.defaultdict(list))
        rows['drawing alone']['drawn'].append(bench.measure('fluke', Dg, test, cov))
        rows['drawing alone']['drawn, gap > 1 h'].append(bench.measure('fluke', Dg, test, cov_hi))
        for k in side:
            rows['drawing alone'][k].append(bench.measure('fluke', Dg, test, side[k]))
        # veto
        F = fused(BL, Lk, Dg, fitd, fitd)
        for prec in (0.90, 0.95):
            th = thr_at(F, fitd, prec)
            tv = diff_thr(Dv, fitd, 0.95)
            tv90 = diff_thr(Dv, fitd, 0.90)
            veto[f'thr {prec}'] = veto.get(f'thr {prec}', []) + [(float(th), float(tv), float(tv90))]
            for d in test:
                for p, v in Sx['BASE+kp'][d].items():
                    if v >= th:
                        same = SAME[d][p]
                        vet = p in Dv[d] and Dv[d][p] >= tv
                        vet90 = p in Dv[d] and Dv[d][p] >= tv90
                        veto[prec] += np.array([same, not same, vet and same, vet and not same])
                        veto[(prec, 0.9)] += np.array([same, not same, vet90 and same, vet90 and not same])
    out = {t: {k: {m: float(np.nanmean([r[m] for r in v])) for m in ('auc', 'ap', 'r95', 'r90', 'same', 'diff')} for k, v in rows[t].items()} for t in rows}
    print('== fusion, 20 splits, val + set-aside days (mean); same pairs within each subset')
    for sub in ('all', 'drawn', 'drawn, gap > 1 h', 'both light', 'not both light'):
        for t in ('drawing alone',) + tags:
            if sub in out[t]:
                m = out[t][sub]
                print(f"   {sub:<18s} {t:<22s} AUC {m['auc']:.3f}  AP {m['ap']:.3f}  found at P 0.95 {100 * m['r95']:5.1f} %  at 0.90 {100 * m['r90']:5.1f} %  ({m['same']:.0f} same / {m['diff']:.0f} different per split)")
    for prec, v in veto.items():
        if isinstance(prec, str):
            continue
        print(f'== veto ({prec}: join precision[, veto threshold at different-precision 0.90; else 0.95]) on BASE+kp joins, key {prec} (summed over the 20 splits): right joins {v[0]:.0f}, wrong joins {v[1]:.0f}; vetoed right {v[2]:.0f}, vetoed wrong {v[3]:.0f}')
    out['veto'] = {str(k): (v.tolist() if hasattr(v, 'tolist') else v) for k, v in veto.items()}
    json.dump(out, open(HERE / 'fusion.json', 'w'), indent=1)


if __name__ == '__main__':
    parts = sys.argv[1:] or ['1', '2', '3', '4']
    if '1' in parts:
        part1()
    if '2' in parts:
        part2()
    if '3' in parts:
        part3()
    if '4' in parts or '5' in parts:
        part45()
