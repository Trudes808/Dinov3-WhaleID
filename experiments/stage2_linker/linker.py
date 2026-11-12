"""Short-gap continuity linker (step 3): P(two safe clusters are one whale) for pairs 1.5 s < gap <= 60 s, trained on the
catalog labels of the train days, compared on the val days with the untrained scores. Catalog truth = the majority
animal of each cluster. Eval days are not read here.
usage: linker.py [--train DAYS|all] [--save]    (tables/<day>.pkl from tables.py)"""
import sys, os, pickle, json, argparse
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '/home/sat3737/NOAA_whales/Dinov3-WhaleID')
from model_fine_tuning.cluster.link_scores import CONT_KEYS, nrank
EVAL = ["2024-12-18", "2024-12-26", "2025-02-19", "2025-03-21"]; VAL = ["2025-01-15", "2025-02-27", "2025-03-14"]; REF = '2025-01-02'
LO, HI = 1.5, 60.0
TIME = ["gap", "log_gap", "same_cam", "n_a", "n_b", "n_min", "span_a", "span_b", "overlap"]; SEQ = TIME + list(CONT_KEYS) + ["lab_mean_d"]
APP = [f"app_{p}_{s}" for p in ("body", "fin", "fluke") for s in ("max", "mean", "top3", "med", "n", "rowmax")] + ["app_best", "app_n"]
WIN = ["win_all_rm", "win_all_max", "win_sal_rm", "win_sal_max"]
ap = argparse.ArgumentParser(); ap.add_argument('--train', default='all'); ap.add_argument('--save', default=None); ap.add_argument('--quiet', action='store_true')
ap.add_argument('--loo', action='store_true', help='the saved variant refitted with each train day held out -> ../safe_clusters/link_gbt_loo/<day>.pkl, and its score on the held-out day'); a = ap.parse_args()
have = sorted(f[:-4] for f in os.listdir('tables')); TRAIN = [d for d in have if d not in EVAL and d not in VAL] if a.train == 'all' else a.train.split(',')
T = {d: pickle.load(open(f'tables/{d}.pkl', 'rb')) for d in TRAIN + VAL}
def col(rows, nm): return np.array([r.get(nm, np.nan) if r.get(nm) is not None else np.nan for r in rows], float)
def X(rows, names): return np.stack([col(rows, nm) for nm in names], 1)
RK = {}
def day_ranks(day):               # within-day ranks of the appearance and window columns, over the day's pairs beyond a burst
    if day not in RK:
        rows = [r for r in T[day] if r['gap'] > LO]; RK[day] = {}
        for nm in APP + WIN:
            if nm.endswith('_n'): continue
            v = nrank(col(rows, nm)); RK[day][nm] = {(r['a'], r['b']): x for r, x in zip(rows, v)}
    return RK[day]
def Xr(rows, names, rk, day=None):   # rk: the appearance and window columns as within-day ranks
    M = X(rows, names)
    if rk:
        R = day_ranks(day)
        for j, nm in enumerate(names):
            if nm in R: M[:, j] = [R[nm].get((r['a'], r['b']), np.nan) for r in rows]
    return M
def gbt(Xm, y, seed, leaf=40, it=300):
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=it, l2_regularization=1.0, min_samples_leaf=leaf, early_stopping=False, random_state=seed).fit(Xm, y)
def band(rows, lo=LO, hi=HI): return [r for r in rows if lo < r['gap'] <= hi]
def known(rows): return [r for r in rows if r['cat'] != '?']
def fit(names, days, rk=False, lo=LO, hi=HI, leaf=40):
    Xs, ys = [], []
    for d in days:
        rows = known(band(T[d], lo, hi))
        if rows: Xs.append(Xr(rows, names, rk, d)); ys.append(np.array([r['cat'] == 'same' for r in rows], int))
    Xs = np.vstack(Xs); ys = np.concatenate(ys)
    return dict(names=names, rk=rk, models=[gbt(Xs, ys, s, leaf) for s in range(3)], n=len(ys), pos=int(ys.sum()), days=len(days))
def predict(m, rows, day=None): return np.mean([c.predict_proba(Xr(rows, m['names'], m['rk'], day))[:, 1] for c in m['models']], 0)
def combo(cols):
    M = np.stack(cols, 1); n = np.isfinite(M).sum(1); s = np.nansum(M, 1) / np.maximum(n, 1); s[n == 0] = 0.0; return s
def stats(y, s):
    s = np.where(np.isfinite(s), s, -1e9); o = np.argsort(-s, kind='stable'); yy = y[o]; cum = np.cumsum(yy); prec = cum / np.arange(1, len(yy) + 1)
    def at(p):
        ok = np.where(prec >= p)[0]; ok = ok[ok >= 9]; return int(cum[ok[-1]]) if len(ok) else 0
    return dict(p50=float(prec[min(49, len(prec) - 1)]), r95=at(.95), r90=at(.90), r80=at(.80), auc=float(roc_auc_score(y, s)))
def untrained(rows):
    g = col(rows, 'gap'); rk = {nm: nrank(col(rows, nm)) for nm in ('app_fin_max', 'app_body_max', 'app_best', 'win_sal_max')}; rk['gap'] = nrank(-g)
    return {'time gap alone (shorter first)': -g, 'student best part': col(rows, 'app_best'), 'window, where-to-look': col(rows, 'win_sal_max'),
            'rank mean: student fin + body': combo([rk['app_fin_max'], rk['app_body_max']]), 'rank mean: student fin + body + window': combo([rk['app_fin_max'], rk['app_body_max'], rk['win_sal_max']]),
            'rank mean: student fin + body + window + time gap': combo([rk['app_fin_max'], rk['app_body_max'], rk['win_sal_max'], rk['gap']])}
def find(par, x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
def merge(day, score_fn, hi=HI):
    """Merging in score order over the pairs with gap <= hi (within-burst pairs included, they get the same score function):
    correct merges before the 1st / 3rd / 6th merge of two piles of different catalog animals; a pile keeps its root's animal."""
    z = json.load(open(f'../days_all/{day}/safe/clusters.json')); maj = {}
    for c in z['clusters']:
        e = sorted((c.get('catalog') or {}).items(), key=lambda kv: (-kv[1], kv[0]))
        if e: maj[c['id']] = e[0][0]
    rows = [r for r in known(T[day]) if r['gap'] <= hi]; v = score_fn(rows); v = np.where(np.isfinite(v), v, -1e9); par = {k: k for k in maj}
    orc = {k: k for k in maj}
    for r in rows:
        if r['cat'] == 'same': orc[find(orc, r['a'])] = find(orc, r['b'])
    need = len(maj) - len(set(find(orc, k) for k in maj)); good = wrong = 0; marks = {}
    for q in np.argsort(-v, kind='stable'):
        fa, fb = find(par, rows[q]['a']), find(par, rows[q]['b'])
        if fa == fb: continue
        if maj[fa] == maj[fb]: good += 1
        else:
            wrong += 1
            if wrong in (1, 3, 6): marks[wrong] = good
            if wrong >= 6: break
        par[fa] = fb
    for w in (1, 3, 6): marks.setdefault(w, good)
    return marks, need, len(maj), len(set(maj.values()))
def held_out():
    """Every train day scored by a linker fitted on the other train days: the honest score for a train day (the page uses
    these files there) and a second, wider check of the linker than the three val days."""
    names = SEQ + APP + WIN; os.makedirs('../safe_clusters/link_gbt_loo', exist_ok=True); res = {}
    for d in TRAIN:
        m = fit(names, [x for x in TRAIN if x != d], False, LO, HI)
        pickle.dump(dict(name='link_gbt_v1', kind='all train days but this one, band pairs: time + continuity + student + window', names=m['names'], rk=False, models=m['models'], band=(LO, HI),
                         held_out=d, pairs=m['n'], same=m['pos'], fitted='2026-11-11'), open(f'../safe_clusters/link_gbt_loo/{d}.pkl', 'wb'))
        rows = known(band(T[d])); y = np.array([r['cat'] == 'same' for r in rows], int)
        if len(y) < 30 or y.min() == y.max(): print(f'   {d}: {len(y)} band pairs, not scored'); continue
        S = untrained(rows); S['linker, this day held out'] = predict(m, rows, d); res[d] = ({k: stats(y, v) for k, v in S.items()}, int(y.sum()), int((1 - y).sum()))
        x = res[d][0]; print(f"   {d}: same {res[d][1]:4d} diff {res[d][2]:4d} | AUC linker {x['linker, this day held out']['auc']:.3f}, student best {x['student best part']['auc']:.3f}, rank mean + gap {x['rank mean: student fin + body + window + time gap']['auc']:.3f}"
                          f" | same at precision 0.90: linker {x['linker, this day held out']['r90']}, student {x['student best part']['r90']}, rank mean {x['rank mean: student fin + body + window']['r90']}", flush=True)
    print(f"== {len(res)} train days, each held out, {LO} s < gap <= {HI} s: catalog-same pairs {sum(v[1] for v in res.values())}, different {sum(v[2] for v in res.values())} ==")
    for nm in next(iter(res.values()))[0]:
        x = [v[0][nm] for v in res.values()]; best = sum(1 for v in res.values() if v[0]['linker, this day held out']['auc'] > v[0][nm]['auc'])
        print(f"   {nm:52s} mean AUC {np.mean([v['auc'] for v in x]):.3f} | same at precision 0.95 / 0.90 / 0.80 (sum over days): {sum(v['r95'] for v in x):5d} {sum(v['r90'] for v in x):5d} {sum(v['r80'] for v in x):5d}" + ('' if nm.startswith('linker') else f" | the linker's AUC is higher on {best} of {len(res)} days"))


if __name__ == '__main__':
    if a.loo:
        held_out(); raise SystemExit(0)
    print(f'train days {len(TRAIN)}: {TRAIN[0]} .. {TRAIN[-1]} | band {LO} s < gap <= {HI} s')
    nb = [(d, len(known(band(T[d]))), sum(r["cat"] == "same" for r in known(band(T[d])))) for d in TRAIN]; print(f'  catalog-labelled band pairs on the train days: {sum(x[1] for x in nb)} ({sum(x[2] for x in nb)} same)')
    ALLF = SEQ + APP + WIN        # hi: the widest gap of the training pairs (the band only, or every pair within 15 min; the val check is always the band)
    CONF = [('ref day, band pairs: time + continuity + student + window', ALLF, [REF], False, HI),
            ('ref day, pairs to 15 min: time + continuity + student + window', ALLF, [REF], False, 900.0),
            ('all train days, band pairs: time + continuity + student + window', ALLF, TRAIN, False, HI),
            ('all train days, pairs to 15 min: time + continuity + student + window', ALLF, TRAIN, False, 900.0),
            ('all train days, pairs to 15 min: the same, appearance as day ranks', ALLF, TRAIN, True, 900.0),
            ('all train days, band pairs: the same, appearance as day ranks', ALLF, TRAIN, True, HI),
            ('all train days, pairs to 15 min: time + continuity + window', SEQ + WIN, TRAIN, False, 900.0),
            ('all train days, pairs to 15 min: time + continuity + student', SEQ + APP, TRAIN, False, 900.0),
            ('all train days, band pairs: time + continuity only', SEQ, TRAIN, False, HI),
            ('all train days, pairs to 15 min: time + continuity only', SEQ, TRAIN, False, 900.0)]
    M = {}
    for nm, names, days, rk, hi in CONF:
        if nm.startswith('all train') and len(TRAIN) < 2: continue
        M[nm] = fit(names, days, rk, LO, hi); print(f"  fitted: {nm}: {M[nm]['n']} pairs ({M[nm]['pos']} same) of {M[nm]['days']} day(s)", flush=True)
    res = {}
    for d in VAL:
        rows = known(band(T[d])); y = np.array([r['cat'] == 'same' for r in rows], int); S = untrained(rows)
        for nm, m in M.items(): S[nm] = predict(m, rows, d)
        res[d] = ({nm: stats(y, v) for nm, v in S.items()}, int(y.sum()), int((1 - y).sum()))
    print(f"\n== val days, {LO} s < gap <= {HI} s: catalog-same pairs {sum(res[d][1] for d in VAL)}, different {sum(res[d][2] for d in VAL)} (per day {[(res[d][1], res[d][2]) for d in VAL]}) ==")
    for nm in res[VAL[0]][0]:
        x = [res[d][0][nm] for d in VAL]
        print(f"   {nm:66s} top-50 prec {np.mean([v['p50'] for v in x]):.2f} | same at precision 0.95 / 0.90 / 0.80: {sum(v['r95'] for v in x):4d} {sum(v['r90'] for v in x):4d} {sum(v['r80'] for v in x):4d} | at 0.90 per day {[v['r90'] for v in x]} | AUC {[round(v['auc'], 3) for v in x]} mean {np.mean([v['auc'] for v in x]):.3f}")
    print(f"\n== val days, merging in score order over the pairs with gap <= {HI} s ==")
    fns = {nm: (lambda rows, nm=nm: untrained(rows)[nm]) for nm in untrained(known(band(T[VAL[0]])))}
    for nm, m in M.items(): fns[nm] = m
    for nm, fn in fns.items():
        k = [merge(d, (lambda rows, fn=fn, d=d: predict(fn, rows, d)) if isinstance(fn, dict) else fn) for d in VAL]
        print(f"   {nm:66s} correct merges before the 1st / 3rd / 6th wrong: {sum(x[0][1] for x in k):4d} {sum(x[0][3] for x in k):4d} {sum(x[0][6] for x in k):4d} of {sum(x[1] for x in k)} possible | per day before the 1st {[x[0][1] for x in k]}")
    if a.save:
        m = M[a.save]; out = '../safe_clusters/link_gbt.pkl'
        pickle.dump(dict(name='link_gbt_v1', kind=a.save, names=m['names'], rk=m['rk'], models=m['models'], band=(LO, HI), train_days=TRAIN, pairs=m['n'], same=m['pos'], fitted='2026-11-11'), open(out, 'wb')); print('saved', a.save, '->', out)
