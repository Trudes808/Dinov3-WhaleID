"""The short-gap band on the val days (non-whales left out): cluster pairs 1.5 s < gap <= 60 s against the catalog.
On the labelled day most of the merges still missing after the researcher's links are between pieces within 60 s of
each other, so: can any score order the catalog-same pairs in that band, and how far does merging in score order over
the pairs within 60 s get? Scores: the time gap alone, the student, the where-to-look window score, untrained rank
means, and linkers trained on 2025-01-02 (catalog labels / researcher verdicts; non-whale pairs dropped from training).
Also the catalog oracle by joining window on the val days. Read-only."""
import json, collections
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import transfer_eval as T
from nowhale import NOT_WHALE, DAYS
REF, VAL = T.REF, T.VAL
Rr = T.Rr; NWR = NOT_WHALE[REF]
LO, HI = 1.5, 60.0
def gbt(X, y, leaf, seed):
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0, min_samples_leaf=leaf, early_stopping=False, random_state=seed).fit(X, y)
def fit(names, mode, band=False):
    X = T.table(Rr, names)
    if mode == 'researcher':
        idx, y = [], []
        for r in T.RECS:
            t = {'same': 1, 'different': 0, 'no_similarity': 0}.get(r['verdict'])
            if t is None or r['a'] in NWR or r['b'] in NWR: continue
            idx.append(T.pidx[(min(r['a'], r['b']), max(r['a'], r['b']))]); y.append(t)
        idx = np.array(idx); y = np.array(y); leaf = 10
    else:
        keep = np.array([a not in NWR and b not in NWR for a, b in Rr['pairs']])
        m = (Rr['cat'] != '?') & (Rr['gap'] > LO) & keep
        if band: m &= Rr['gap'] <= HI
        idx = np.where(m)[0]; y = (Rr['cat'][idx] == 'same').astype(int); leaf = 40
    return len(idx), names, [gbt(X[idx], y, leaf, s) for s in range(3)]
GEO = [k for k in T.SEQ]
CONF = (('linker on catalog labels: time and geometry only', GEO, 'catalog', False), ('linker on catalog labels: all + window', T.SEQ + T.APP + T.WIN, 'catalog', False),
        ('linker on catalog labels of the band only: all + window', T.SEQ + T.APP + T.WIN, 'catalog', True), ('linker on researcher verdicts: all + window', T.SEQ + T.APP + T.WIN, 'researcher', False))
MODELS = {}
for nm, names, mode, band in CONF:
    n, names, ms = fit(names, mode, band); MODELS[nm] = (names, ms); print(f'{nm}: {n} training pairs', flush=True)
TAB = {}
def table(day, names):
    k = (day, tuple(names))
    if k not in TAB: TAB[k] = T.table(T.data[day], names)
    return TAB[k]
def scores(day, m, models=True):
    R = T.data[day]; S = {nm: R['S'][nm][m] for nm in ('app_fin_max', 'app_body_max', 'app_best', 'win_sal_max')}; g = R['gap'][m]
    rk = {nm: T.nrank(S[nm]) for nm in S}; rk['gap'] = T.nrank(-g)
    out = {'time gap alone (shorter first)': -g, 'student best part': S['app_best'], 'window, where-to-look': S['win_sal_max'],
           'rank mean: student fin + body': T.combo([rk['app_fin_max'], rk['app_body_max']]),
           'rank mean: student fin + body + window': T.combo([rk['app_fin_max'], rk['app_body_max'], rk['win_sal_max']]),
           'rank mean: student fin + body + window + time gap': T.combo([rk['app_fin_max'], rk['app_body_max'], rk['win_sal_max'], rk['gap']])}
    if models:
        for nm, (names, ms) in MODELS.items():
            X = table(day, names)[m]; out[nm] = np.mean([c.predict_proba(X)[:, 1] for c in ms], 0)
    return out
def stats(y, s):
    s = np.where(np.isfinite(s), s, -1e9); o = np.argsort(-s, kind='stable'); yy = y[o]; cum = np.cumsum(yy); prec = cum / np.arange(1, len(yy) + 1)
    def at(p):
        ok = np.where(prec >= p)[0]; ok = ok[ok >= 9]; return int(cum[ok[-1]]) if len(ok) else 0
    return dict(p50=float(prec[min(49, len(prec) - 1)]), r95=at(.95), r90=at(.90), r80=at(.80), auc=float(roc_auc_score(y, s)))
def keepmask(day):
    R = T.data[day]; ex = NOT_WHALE[day]; return np.array([a not in ex and b not in ex for a, b in R['pairs']]) & (R['cat'] != '?')
def band_eval(day, lo, hi, models=True):
    R = T.data[day]; m = keepmask(day) & (R['gap'] > lo) & (R['gap'] <= hi); y = (R['cat'][m] == 'same').astype(int)
    return {nm: stats(y, v) for nm, v in scores(day, m, models).items()}, int(y.sum()), int((1 - y).sum())
def major(c):
    e = sorted((c.get('catalog') or {}).items(), key=lambda kv: -kv[1]); return e[0][0] if e else None
def find(par, x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
def merge(day, hi, models=True):
    R = T.data[day]; ex = NOT_WHALE[day]; z = json.load(open(f'{DAYS}/{day}/safe/clusters.json')); C = {c['id']: c for c in z['clusters'] if c['id'] not in ex}
    maj = {k: major(c) for k, c in C.items()}; ids = [k for k in maj if maj[k]]; animals = sorted(set(maj[k] for k in ids)); km = keepmask(day)
    orc = {}
    for w in (1.5, 60.0, 300.0, 900.0):
        par = {k: k for k in ids}
        for (a, b), k, g in zip(R['pairs'], km, R['gap']):
            if k and g <= w and maj[a] == maj[b]: par[find(par, a)] = find(par, b)
        orc[w] = len(set(find(par, k) for k in ids))
    m = km & (R['gap'] <= hi); pairs = [p for p, k in zip(R['pairs'], m) if k]; kr = {}
    for nm, v in scores(day, m, models).items():
        v = np.where(np.isfinite(v), v, -1e9); par = {k: k for k in ids}; good = wrong = 0; marks = {}
        for q in np.argsort(-v, kind='stable'):
            a, b = pairs[q]; fa, fb = find(par, a), find(par, b)
            if fa == fb: continue
            # a pile keeps the animal of its root; a wrong merge is one between piles of different animals
            if maj[fa] == maj[fb]: good += 1
            else:
                wrong += 1
                if wrong in (1, 3, 6): marks[wrong] = good
                if wrong >= 6: break
            par[fa] = fb
        for w in (1, 3, 6): marks.setdefault(w, good)
        kr[nm] = marks
    return dict(clusters=len(ids), animals=len(animals), orc=orc, need=len(ids) - orc[hi], kr=kr)
if __name__ == '__main__':
    for lo, hi in ((LO, HI), (HI, 300.0), (300.0, 900.0)):
        res = {d: band_eval(d, lo, hi) for d in VAL}
        print(f"\n== val days, non-whales left out, {lo} s < gap <= {hi} s: catalog-same pairs {sum(res[d][1] for d in VAL)}, different {sum(res[d][2] for d in VAL)} (per day same/different: {[(res[d][1], res[d][2]) for d in VAL]}) ==")
        for nm in res[VAL[0]][0]:
            a = [res[d][0][nm] for d in VAL]
            print(f"   {nm:58s} top-50 precision (mean) {np.mean([x['p50'] for x in a]):.2f} | same pairs at precision 0.95 / 0.90 / 0.80 (sum): {sum(x['r95'] for x in a):4d} {sum(x['r90'] for x in a):4d} {sum(x['r80'] for x in a):4d}"
                  f" | per day at 0.90: {[x['r90'] for x in a]} | AUC per day {[round(x['auc'], 3) for x in a]} mean {np.mean([x['auc'] for x in a]):.3f}")
    r, npos, nneg = band_eval(REF, LO, HI, models=False)
    print(f"\n== labelled day {REF} (student in-sample), {LO} s < gap <= {HI} s, non-whales left out: catalog-same {npos}, different {nneg} ==")
    for nm, x in r.items(): print(f"   {nm:58s} top-50 precision {x['p50']:.2f} | same pairs at precision 0.95 / 0.90 / 0.80: {x['r95']:4d} {x['r90']:4d} {x['r80']:4d} | AUC {x['auc']:.3f}")
    for hi in (HI, 300.0):
        mg = {d: merge(d, hi) for d in VAL}; ncl = sum(mg[d]['clusters'] for d in VAL); nan = sum(mg[d]['animals'] for d in VAL); need = sum(mg[d]['need'] for d in VAL)
        print(f"\n== val days, merging in score order over the pairs with gap <= {hi} s ==")
        print('   pieces per catalog animal, pooled: start %.2f | catalog oracle joining every catalog-same pair within 1.5 s %.2f, 60 s %.2f, 300 s %.2f, 15 min %.2f' % ((ncl / nan,) + tuple(sum(mg[d]['orc'][w] for d in VAL) / nan for w in (1.5, 60.0, 300.0, 900.0))))
        print(f"   per day (clusters, animals, oracle components at 1.5 s / 60 s / 300 s / 15 min): {[(mg[d]['clusters'], mg[d]['animals'], mg[d]['orc'][1.5], mg[d]['orc'][60.0], mg[d]['orc'][300.0], mg[d]['orc'][900.0]) for d in VAL]}")
        for nm in mg[VAL[0]]['kr']:
            k = [mg[d]['kr'][nm] for d in VAL]; g1 = sum(x[1] for x in k)
            print(f"   {nm:58s} correct merges before the 1st / 3rd / 6th wrong merge: {g1:4d} {sum(x[3] for x in k):4d} {sum(x[6] for x in k):4d} of {need} possible | per day before the 1st: {[x[1] for x in k]} | pieces per animal at the 1st wrong merge {(ncl - g1) / nan:.2f}")
