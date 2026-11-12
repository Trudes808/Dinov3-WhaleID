"""The frozen scores against the catalog on a split's days (val: where they were chosen; eval: read once, for the report):
the rank mean over all cluster pairs beyond a burst within 15 min, and the short-gap linker in its band.
usage: final_eval.py val|eval"""
import sys, json, pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '/home/sat3737/NOAA_whales/Dinov3-WhaleID')
from model_fine_tuning.cluster.link_scores import nrank, linker_p, LINKER
SPLIT = {'val': ["2025-01-15", "2025-02-27", "2025-03-14"], 'eval': ["2024-12-18", "2024-12-26", "2025-02-19", "2025-03-21"]}[sys.argv[1]]
T = {d: pickle.load(open(f'tables/{d}.pkl', 'rb')) for d in SPLIT}; M = pickle.load(open(LINKER, 'rb')) if LINKER.exists() else None; LO, HI = (M['band'] if M else (1.5, 60.0))
col = lambda rows, nm: np.array([r.get(nm, np.nan) if r.get(nm) is not None else np.nan for r in rows], float)
def combo(cols):
    X = np.stack(cols, 1); n = np.isfinite(X).sum(1); s = np.nansum(X, 1) / np.maximum(n, 1); s[n == 0] = 0.0; return s
def stats(y, s):
    s = np.where(np.isfinite(s), s, -1e9); o = np.argsort(-s, kind='stable'); yy = y[o]; cum = np.cumsum(yy); prec = cum / np.arange(1, len(yy) + 1)
    def at(p):
        ok = np.where(prec >= p)[0]; ok = ok[ok >= 9]; return int(cum[ok[-1]]) if len(ok) else 0
    return dict(p100=float(prec[min(99, len(prec) - 1)]), r95=at(.95), r90=at(.90), r80=at(.80), auc=float(roc_auc_score(y, s)) if 0 < y.mean() < 1 else float('nan'))
def scores(day, rows_all, sel):
    """Every score for the rows picked by sel; ranks are taken over the day's pairs beyond a burst, as the page does."""
    far = np.array([r['gap'] > 1.5 for r in rows_all]); rk = {}
    for nm in ('app_fin_max', 'app_body_max', 'app_best', 'win_sal_max'):
        v = np.full(len(rows_all), np.nan); v[far] = nrank(col([r for r, f in zip(rows_all, far) if f], nm)); rk[nm] = v
    g = col(rows_all, 'gap'); out = {'time gap alone (shorter first)': -g, 'student best part': col(rows_all, 'app_best'), 'window, where-to-look': col(rows_all, 'win_sal_max'),
           'rank mean: student fin + body': combo([rk['app_fin_max'], rk['app_body_max']]), 'rank mean: student fin + body + window (the page)': combo([rk['app_fin_max'], rk['app_body_max'], rk['win_sal_max']])}
    if M is not None: out['short-gap linker (the page, in its band)'] = linker_p(M, rows_all)
    return {k: v[sel] for k, v in out.items()}
def find(par, x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
def merge(day, name, hi):
    z = json.load(open(f'../days_all/{day}/safe/clusters.json')); maj = {}
    for c in z['clusters']:
        e = sorted((c.get('catalog') or {}).items(), key=lambda kv: (-kv[1], kv[0]))
        if e: maj[c['id']] = e[0][0]
    rows_all = T[day]; sel = np.array([r['cat'] != '?' and r['gap'] <= hi for r in rows_all]); rows = [r for r, s in zip(rows_all, sel) if s]
    if name == 'the page: within a burst by day order, then the linker, then the rank mean':
        S = scores(day, rows_all, sel); g = col(rows, 'gap'); lp = S.get('short-gap linker (the page, in its band)', np.full(len(rows), np.nan)); rm = S['rank mean: student fin + body + window (the page)']
        v = np.where(g <= 1.5, 3.0, np.where(g <= HI, 1.0 + np.nan_to_num(lp, nan=0.0), np.nan_to_num(rm, nan=0.0)))
    else:
        v = scores(day, rows_all, sel)[name]
    v = np.where(np.isfinite(v), v, -1e9); par = {k: k for k in maj}; orc = {k: k for k in maj}
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
print(f'{sys.argv[1]} days {SPLIT}; linker {"none" if M is None else M["kind"]}')
for label, lo, hi in (('all pairs beyond a burst within 15 min', 1.5, 900.0), (f'the short band {LO} s < gap <= {HI} s', LO, HI), ('beyond the band, to 15 min', HI, 900.0)):
    res = {}
    for d in SPLIT:
        sel = np.array([r['cat'] != '?' and lo < r['gap'] <= hi for r in T[d]]); y = np.array([r['cat'] == 'same' for r, s in zip(T[d], sel) if s], int)
        res[d] = ({k: stats(y, v) for k, v in scores(d, T[d], sel).items()}, int(y.sum()), int((1 - y).sum()))
    print(f"\n== {label}: catalog-same pairs {sum(res[d][1] for d in SPLIT)}, different {sum(res[d][2] for d in SPLIT)} (per day {[(res[d][1], res[d][2]) for d in SPLIT]}) ==")
    for nm in res[SPLIT[0]][0]:
        if 'linker' in nm and lo != LO: continue
        x = [res[d][0][nm] for d in SPLIT]
        print(f"   {nm:52s} top-100 prec {np.mean([v['p100'] for v in x]):.2f} | same at precision 0.95 / 0.90 / 0.80: {sum(v['r95'] for v in x):4d} {sum(v['r90'] for v in x):4d} {sum(v['r80'] for v in x):4d} | at 0.90 per day {[v['r90'] for v in x]} | AUC {[round(v['auc'], 3) for v in x]} mean {np.nanmean([v['auc'] for v in x]):.3f}")
names = ['time gap alone (shorter first)', 'student best part', 'rank mean: student fin + body + window (the page)'] + (['short-gap linker (the page, in its band)'] if M else [])
for hi in (HI, 900.0):
    print(f"\n== merging in score order over the pairs with gap <= {hi:.0f} s ==")
    for nm in names + (['the page: within a burst by day order, then the linker, then the rank mean'] if hi > HI else []):
        if 'linker' in nm and 'page:' not in nm and hi > HI: continue
        k = [merge(d, nm, hi) for d in SPLIT]
        print(f"   {nm:76s} correct merges before the 1st / 3rd / 6th wrong: {sum(x[0][1] for x in k):4d} {sum(x[0][3] for x in k):4d} {sum(x[0][6] for x in k):4d} of {sum(x[1] for x in k)} needed | per day before the 1st {[x[0][1] for x in k]} | clusters {sum(x[2] for x in k)}, animals {sum(x[3] for x in k)}")
