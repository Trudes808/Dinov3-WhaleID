"""sketch_fin step 5: numbers on the shared bench (CPU, train + val days only; the bench refuses eval days).

1  label-free drawing scorers through bench.evaluate_pairs (nothing fitted on the days they are read on; the shape
   classes are k-means without labels on the train days)
2  exclusion: rank pairs by "differentness", share of the different-animal pairs ruled out at precision 0.95 / 0.90
   (ties count against: same-animal pairs first inside a tie), all 24 days pooled and val; HW-b6 (low cosine) on the
   same pairs as reference; a fitted exclusion (logistic regression, leave one day out) with and without the drawing
3  shape-class repeatability
4  HW-b6 + drawing as positive evidence (stack.py's fusion: Platt on HW, isotonic on the drawing, positive part only;
   and a logistic fusion that may also push down), bench's 20 splits, fitted on the 16 pool days
5  veto: HW-b6 joins at the leave-one-day-out threshold for pair precision 0.95 / 0.90, removed where the exclusion
   score passes its own leave-one-day-out threshold for ruled-out precision 0.95; pair and day level

    python evaluate.py > evaluate.log
"""
import collections
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.argv = sys.argv[:1]                                  # check_hw reads sys.argv[1] at import
sys.path.insert(0, str(HERE.parent))
import bench  # noqa: E402
import stack as ST  # noqa: E402
from check_hw import at_prec, hw_feats, pair_scores  # noqa: E402
from size_levers import Day, auc, components, line, score, total  # noqa: E402

CACHE = Path('/home/sat3737/NOAA_whales/moonshot_cache/sketch_fin')
ALL = bench.TRAIN + bench.VAL
F = {d: pickle.load(open(CACHE / 'pairs' / f'{d}.pkl', 'rb')) for d in ALL}
HW = {d: pair_scores(d, hw_feats(d)) for d in ALL}
DD = {d: Day(d) for d in ALL}
OUT = {}


def rel(d, k):
    fa, fb = bench.flank(d, k[0]), bench.flank(d, k[1])
    fa, fb = set(fa) if fa else set(), set(fb) if fb else set()
    if not fa or not fb:
        return 'flank unknown'
    if len(fa) > 1 or len(fb) > 1:
        return 'mixed'
    return 'same flank' if fa == fb else 'opposite flanks'


def ranks(S):
    """label-free percentile ranks of a pooled {day: {pair: v}}"""
    items = [(d, p, v) for d in S for p, v in S[d].items()]
    v = np.array([x[2] for x in items], float)
    r = np.argsort(np.argsort(v)) / max(1, len(v) - 1)
    for u in np.unique(v):
        m = v == u
        if m.sum() > 1:
            r[m] = r[m].mean()
    out = collections.defaultdict(dict)
    for (d, p, _), x in zip(items, r):
        out[d][p] = float(x)
    return dict(out)


def S_of(fn):
    out = {}
    for d in ALL:
        out[d] = {}
        for p, f in F[d].items():
            v = fn(f)
            if v is not None:
                out[d][p] = float(v)
    return out


# ---- the drawing's evidences, label-free --------------------------------------------------------------------------
S_contour = S_of(lambda f: -f['dc'] if f.get('dc') is not None else None)
S_marks = S_of(lambda f: f.get('mark_ncc'))
S_class = S_of(lambda f: float(f['cls_a'] == f['cls_b']) if f['cls_a'] is not None and f['cls_b'] is not None else None)
Rc, Rm = ranks(S_contour), ranks(S_marks)
S_draw = {d: {p: float(np.mean([R[d][p] for R in (Rc, Rm) if p in R.get(d, {})])) for p in set(Rc.get(d, {})) | set(Rm.get(d, {}))} for d in ALL}
# differentness (higher = more likely different)
D_contour = S_of(lambda f: f['dc'] if f.get('dc') is not None else None)
D_miss = S_of(lambda f: f['miss'] if f.get('sameside') and f.get('coseen', 0) > 0 else None)
D_nick = S_of(lambda f: f['nick_dis'] if f.get('nick_dis') is not None else None)
D_class = S_of(lambda f: float(f['cls_a'] != f['cls_b']) if f['cls_a'] is not None and f['cls_b'] is not None else None)
D_marklow = S_of(lambda f: -f['mark_ncc'] if f.get('mark_ncc') is not None else None)
Rdc, Rmi, Rni = ranks(D_contour), ranks(D_miss), ranks(D_nick)
D_draw = {d: {p: float(max(R[d][p] for R in (Rdc, Rmi, Rni) if p in R.get(d, {}))) for p in set(Rdc.get(d, {})) | set(Rmi.get(d, {})) | set(Rni.get(d, {}))} for d in ALL}
D_hw = {d: {p: -v for p, v in HW[d].items()} for d in ALL}


def ruled_out(pos_d, neg_d, prec):
    """pos_d = differentness of different-animal pairs (the target), neg_d = of same-animal pairs. Count of different
    pairs ruled out before the precision of 'different' falls below prec; ties: same-animal pairs first."""
    v = np.concatenate([pos_d, neg_d])
    y = np.concatenate([np.ones(len(pos_d)), np.zeros(len(neg_d))])
    o = np.lexsort((y, -v))
    tp = np.cumsum(y[o])
    p = tp / np.arange(1, len(o) + 1)
    ok = np.nonzero(p >= prec)[0]
    if not len(ok):
        return 0, np.inf
    return int(tp[ok[-1]]), float(v[o][ok[-1]])


def excl_row(Dsc, days, only=None, ref=D_hw):
    """ruled-out measure for Dsc and for HW on the same pairs."""
    diff, same, rdiff, rsame, ndiff_all = [], [], [], [], 0
    for d in days:
        for a, b, s in bench.pairs(d, 'dorsal'):
            ndiff_all += not s
            p = (a, b)
            if only is not None and not only(d, p):
                continue
            if p in Dsc.get(d, {}) and p in ref.get(d, {}):
                (same if s else diff).append(Dsc[d][p])
                (rsame if s else rdiff).append(ref[d][p])
    diff, same, rdiff, rsame = (np.array(x, float) for x in (diff, same, rdiff, rsame))
    r = dict(n_same=len(same), n_diff=len(diff), n_diff_all=ndiff_all)
    for tag, (x, y) in (('cand', (diff, same)), ('hw', (rdiff, rsame))):
        r[tag] = dict(auc_diff=auc(x, y) if len(x) and len(y) else float('nan'),
                      out95=ruled_out(x, y, 0.95)[0], out90=ruled_out(x, y, 0.90)[0])
    return r


def fmt_excl(tag, r):
    nd = max(1, r['n_diff'])
    c, h = r['cand'], r['hw']
    return (f"   {tag:<34s} pairs {r['n_same']:5d} same / {r['n_diff']:5d} diff (of {r['n_diff_all']} diff on the bench)  "
            f"drawing: AUC(diff) {c['auc_diff']:.3f}, ruled out at P0.95 {c['out95']:4d} ({100 * c['out95'] / nd:4.1f} %), at 0.90 {c['out90']:4d} ({100 * c['out90'] / nd:4.1f} %)"
            f" | HW-b6 low cosine same pairs: AUC {h['auc_diff']:.3f}, P0.95 {h['out95']} ({100 * h['out95'] / nd:.1f} %), 0.90 {h['out90']} ({100 * h['out90'] / nd:.1f} %)")


if __name__ == '__main__':
    print('== 1  label-free drawing scorers, bench.evaluate_pairs ==')
    for name, S in (('sketch_fin_contour', S_contour), ('sketch_fin_marks', S_marks), ('sketch_fin_class', S_class), ('sketch_fin_draw', S_draw)):
        bench.evaluate_pairs(name, 'dorsal', S)
    print('\n   HW-b6 on the same pairs (all 24 days pooled), by relation:')
    for name, S in (('contour', S_contour), ('marks', S_marks), ('draw', S_draw)):
        for r in ('same flank', 'opposite flanks', 'flank unknown', 'mixed', 'gap > 1 h', 'all'):
            pos, neg, hp, hn = [], [], [], []
            for d in ALL:
                for (a, b), v in S[d].items():
                    if (a, b) not in HW[d]:
                        continue
                    if r == 'gap > 1 h':
                        if DD[d].gap(a, b) <= 3600:
                            continue
                    elif r != 'all' and rel(d, (a, b)) != r:
                        continue
                    s = F[d][(a, b)]['same']
                    (pos if s else neg).append(v)
                    (hp if s else hn).append(HW[d][(a, b)])
            if pos and neg:
                print(f'      {name:<8s} {r:<16s} {len(pos):5d} same / {len(neg):5d} diff  drawing AUC {auc(np.array(pos), np.array(neg)):.3f} '
                      f'P0.95 {100 * at_prec(np.array(pos), np.array(neg), .95)[1] / len(pos):4.1f} % P0.90 {100 * at_prec(np.array(pos), np.array(neg), .90)[1] / len(pos):4.1f} %'
                      f' | HW-b6 AUC {auc(np.array(hp), np.array(hn)):.3f} P0.95 {100 * at_prec(np.array(hp), np.array(hn), .95)[1] / len(hp):4.1f} % P0.90 {100 * at_prec(np.array(hp), np.array(hn), .90)[1] / len(hp):4.1f} %')
                OUT.setdefault('relations', {}).setdefault(name, {})[r] = dict(same=len(pos), diff=len(neg), auc=auc(np.array(pos), np.array(neg)), hw_auc=auc(np.array(hp), np.array(hn)))

    print('\n== 2  exclusion: share of different-animal pairs ruled out (ties against), with HW-b6 low cosine on the same pairs ==')
    gap1h = lambda d, p: DD[d].gap(*p) > 3600
    for tag, days in (('all 24 days', ALL), ('val 3 days', bench.VAL)):
        print(f' -- {tag}')
        for name, Dsc in (('contour distance', D_contour), ('missing marks (same side)', D_miss), ('nick disagreement', D_nick),
                          ('shape class differs', D_class), ('low mark correlation', D_marklow), ('drawing max-rank', D_draw)):
            r = excl_row(Dsc, days)
            OUT.setdefault('exclusion', {}).setdefault(tag, {})[name] = r
            print(fmt_excl(name, r))
            if name in ('drawing max-rank', 'missing marks (same side)', 'contour distance') and tag == 'all 24 days':
                for rr in ('same flank', 'opposite flanks', 'flank unknown'):
                    r2 = excl_row(Dsc, days, lambda d, p, rr=rr: rel(d, p) == rr)
                    OUT['exclusion'][tag][f'{name} | {rr}'] = r2
                    print(fmt_excl(f'  {rr}', r2))
                r2 = excl_row(Dsc, days, gap1h)
                OUT['exclusion'][tag][f'{name} | gap > 1 h'] = r2
                print(fmt_excl('  gap > 1 h', r2))
        # miss counts: how often a marked absence happens at all
    cnt = collections.Counter()
    for d in ALL:
        for p, f in F[d].items():
            if f.get('sameside') and f.get('coseen', 0) > 0:
                cnt[(f['same'], min(f['miss'], 3))] += 1
    print('   missing-mark counts (same, min(miss,3)) ->', dict(sorted(cnt.items())))
    OUT['miss_counts'] = {f'{k[0]}|{k[1]}': v for k, v in cnt.items()}

    # fitted exclusion: logistic regression on HW (+ drawing), leave one day out, P(different)
    def feats(f, hw, use_draw):
        x = [hw]
        if use_draw:
            dc = f.get('dc')
            x += [dc if dc is not None else 0.0, float(dc is None),
                  float(f.get('miss', 0) or 0) if f.get('sameside') else 0.0,
                  float(f.get('nick_dis') or 0), float(f['cls_a'] is not None and f['cls_b'] is not None and f['cls_a'] != f['cls_b']),
                  f.get('mark_ncc') if f.get('mark_ncc') is not None else 0.0, float(f.get('mark_ncc') is None)]
        return x
    Dfit = {}
    for use in (False, True):
        tag = 'LR HW + drawing' if use else 'LR HW only'
        Dfit[tag] = {}
        for d in ALL:
            X, y = [], []
            for o in ALL:
                if o == d:
                    continue
                for p, f in F[o].items():
                    if p in HW[o]:
                        X.append(feats(f, HW[o][p], use)); y.append(not f['same'])
            lr = LogisticRegression(max_iter=2000).fit(np.array(X), np.array(y))
            ps = [p for p in F[d] if p in HW[d]]
            if ps:
                Dfit[tag][d] = dict(zip(ps, lr.decision_function(np.array([feats(F[d][p], HW[d][p], use) for p in ps])).astype(float)))
    print(' -- fitted differentness, logistic regression, leave one day out (pairs with HW and a drawing)')
    for tag in Dfit:
        for dtag, days in (('all 24 days', ALL), ('val 3 days', bench.VAL)):
            r = excl_row(Dfit[tag], days)
            OUT['exclusion'].setdefault(dtag, {})[tag] = r
            print(fmt_excl(f'{tag} [{dtag}]', r))

    print('\n== 3  shape-class repeatability (well-seen side views, both classed) ==')
    for tag, days in (('train 21', bench.TRAIN), ('val 3', bench.VAL)):
        for rr in ('all', 'same flank', 'opposite flanks'):
            s, n = [0, 0], [0, 0]
            for d in days:
                for p, f in F[d].items():
                    if f['cls_a'] is None or f['cls_b'] is None or (rr != 'all' and rel(d, p) != rr):
                        continue
                    i = int(f['same'])
                    n[i] += 1
                    s[i] += f['cls_a'] == f['cls_b']
            print(f'   {tag:<9s} {rr:<16s} same animal: same class {s[1]}/{n[1]} = {100 * s[1] / max(1, n[1]):.1f} %   '
                  f'different animals of one group: same class {s[0]}/{n[0]} = {100 * s[0] / max(1, n[0]):.1f} %')
            OUT.setdefault('class', {})[f'{tag}|{rr}'] = dict(same_eq=s[1], same_n=n[1], diff_eq=s[0], diff_n=n[0])

    print('\n== 4  HW-b6 + drawing (positive evidence), 20 splits, val + set-aside days, same pairs as BASE and HW ==')
    rows = collections.defaultdict(list)
    for hold, pool in bench.splits():
        test, fitd = bench.VAL + hold, pool[:16]
        B = bench.base_scores('dorsal', test, fitd)
        only = {d: set(B.get(d, {})) & set(HW[d]) for d in test}
        rows['HW-b6 raw'].append(bench.measure('dorsal', HW, test, only))
        for nm, Sd in (('draw', S_draw), ('marks', S_marks), ('contour', S_contour)):
            m = ST.fit(HW, Sd, fitd, 'dorsal')
            rows[f'HW-b6 + {nm} (positive only)'].append(bench.measure('dorsal', ST.fuse(HW, Sd, m, test), test, only))
        # logistic fusion, may push down
        X, y = [], []
        for o in fitd:
            for p, f in F[o].items():
                if p in HW[o]:
                    X.append(feats(f, HW[o][p], True)); y.append(f['same'])
        lr = LogisticRegression(max_iter=2000).fit(np.array(X), np.array(y))
        Sl = {d: {p: float(lr.decision_function(np.array([feats(F[d][p], HW[d][p], True)]))[0]) for p in F[d] if p in HW[d]} for d in test}
        rows['HW-b6 + drawing (logistic, both ways)'].append(bench.measure('dorsal', Sl, test, {d: only[d] & set(Sl[d]) for d in test}))
        rows['HW-b6 raw, LR pairs'].append(bench.measure('dorsal', HW, test, {d: only[d] & set(Sl[d]) for d in test}))
    ref = rows['HW-b6 raw']
    for name, rs in rows.items():
        m = ST.mean_rows(rs)
        da = np.array([a['auc'] - b['auc'] for a, b in zip(rs, ref)])
        d95 = np.array([a['r95'] - b['r95'] for a, b in zip(rs, ref)]) * 100
        print(f'   {name:<40s} AUC {m["auc"]:.3f}  AP {m["ap"]:.3f}  found at P 0.95 {100 * m["r95"]:5.1f} %  at 0.90 {100 * m["r90"]:5.1f} %'
              f'   against HW raw: AUC {da.mean():+.3f} (up {int((da > 0).sum())}/20), P 0.95 {d95.mean():+.1f} points (up {int((d95 > 0).sum())}/20)  '
              f'pairs {rs[0]["same"]}+{rs[0]["diff"]} (split 1)')
        OUT.setdefault('fusion', {})[name] = dict(m, d_auc=float(da.mean()), d_r95=float(d95.mean()), up_auc=int((da > 0).sum()))

    print('\n== 5  veto of HW-b6 joins by the exclusion score (thresholds from the other 23 days) ==')
    same_of = {d: {(a, b): s for a, b, s in bench.pairs(d, 'dorsal')} for d in ALL}
    for vname, Dsc in (('drawing max-rank', D_draw), ('missing marks', D_miss), ('contour distance', D_contour), ('LR HW + drawing', Dfit['LR HW + drawing'])):
        for prec in (0.95, 0.90):
            tot = collections.Counter()
            rows_day, rows_day_v, rows_day_0 = [], [], []
            for d in ALL:
                others = [o for o in ALL if o != d]
                hp = [HW[o][p] for o in others for p in HW[o] if same_of[o][p]]
                hn = [HW[o][p] for o in others for p in HW[o] if not same_of[o][p]]
                thr = at_prec(np.array(hp), np.array(hn), prec)[0]
                dp = [Dsc[o][p] for o in others for p in Dsc.get(o, {}) if not same_of[o][p]]
                dn = [Dsc[o][p] for o in others for p in Dsc.get(o, {}) if same_of[o][p]]
                vt = ruled_out(np.array(dp), np.array(dn), 0.95)[1]
                joins = [p for p, v in HW[d].items() if v >= thr]
                veto = [p for p in joins if p in Dsc.get(d, {}) and Dsc[d][p] >= vt]
                tot['joins'] += len(joins)
                tot['wrong'] += sum(not same_of[d][p] for p in joins)
                tot['vetoed'] += len(veto)
                tot['vetoed_wrong'] += sum(not same_of[d][p] for p in veto)
                Dd = DD[d]
                base = [(a, b) for a, b in itertools.combinations(Dd.ids, 2) if Dd.maj[a] == Dd.maj[b] and Dd.gap(a, b) <= 60]
                rows_day_0.append(score(Dd, components(Dd.ids, base)))
                rows_day.append(score(Dd, components(Dd.ids, base + joins)))
                rows_day_v.append(score(Dd, components(Dd.ids, base + [p for p in joins if p not in set(veto)])))
            print(f'   {vname:<20s} HW joins at P {prec:.2f}: {tot["joins"]} joins, {tot["wrong"]} wrong; vetoed {tot["vetoed"]} '
                  f'({tot["vetoed_wrong"]} wrong, {tot["vetoed"] - tot["vetoed_wrong"]} right)')
            print('      ' + line('no joins beyond 60 s (perfect 60-s linking)', total(rows_day_0)))
            print('      ' + line(f'HW joins at P {prec:.2f}', total(rows_day)))
            print('      ' + line(f'HW joins at P {prec:.2f}, vetoed by {vname}', total(rows_day_v)))
            OUT.setdefault('veto', {})[f'{vname}|{prec}'] = dict(tot, day=total(rows_day), day_veto=total(rows_day_v))
    json.dump(OUT, open(HERE / 'evaluate_results.json', 'w'), indent=1, default=float)
    print('saved', HERE / 'evaluate_results.json')
