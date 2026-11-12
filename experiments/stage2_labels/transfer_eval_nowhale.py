"""Val-day check of the linkers trained on 2025-01-02 with the tracked non-whales left out on both sides:
training pairs that touch a non-whale cluster are dropped, and so are the val-day pairs that touch one.
Same measures as transfer_eval.py / nowhale.py (catalog truth, cluster pairs within 15 min, gap > 1.5 s)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import transfer_eval as T
from nowhale import NOT_WHALE
REF, VAL = T.REF, T.VAL
Rr = T.Rr; NWR = NOT_WHALE[REF]

def fit(names, mode, seed=0):
    X = T.table(Rr, names)
    if mode == 'researcher':
        idx = []; y = []
        for r in T.RECS:
            t = {'same': 1, 'different': 0, 'no_similarity': 0}.get(r['verdict'])
            if t is None or r['a'] in NWR or r['b'] in NWR: continue
            idx.append(T.pidx[(min(r['a'], r['b']), max(r['a'], r['b']))]); y.append(t)
        idx = np.array(idx); y = np.array(y)
    else:
        keep = np.array([a not in NWR and b not in NWR for a, b in Rr['pairs']])
        m = (Rr['cat'] != '?') & (Rr['gap'] > 1.5) & keep; idx = np.where(m)[0]; y = (Rr['cat'][idx] == 'same').astype(int)
    leaf = 10 if mode == 'researcher' else 40
    return len(idx), HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0, min_samples_leaf=leaf, early_stopping=False, random_state=seed).fit(X[idx], y)

CONF = (('linker on researcher verdicts: student', T.APP, 'researcher'), ('linker on researcher verdicts: student + window', T.APP + T.WIN, 'researcher'),
        ('linker on researcher verdicts: all + window', T.SEQ + T.APP + T.WIN, 'researcher'),
        ('linker on catalog labels: student', T.APP, 'catalog'), ('linker on catalog labels: student + window', T.APP + T.WIN, 'catalog'),
        ('linker on catalog labels: all + window', T.SEQ + T.APP + T.WIN, 'catalog'))
MODELS = {}
for nm, names, mode in CONF:
    fits = [fit(names, mode, s) for s in range(3)]; MODELS[nm] = (names, [f[1] for f in fits]); print(f'{nm}: {fits[0][0]} training pairs', flush=True)

def evaluate(day, g):
    R = T.data[day]; keep = np.array([a not in NOT_WHALE[day] and b not in NOT_WHALE[day] for a, b in R['pairs']])
    m = (R['gap'] > g) & (R['cat'] != '?') & keep; y = (R['cat'][m] == 'same').astype(int); out = {}
    S = {nm: (R['S'][nm][m] if nm in R['S'] else np.array([r.get(nm, np.nan) if r.get(nm) is not None else np.nan for r in R['rows']], float)[m]) for nm in T.SINGLE}
    rk = {nm: T.nrank(S[nm]) for nm in T.SINGLE}
    out['student best part'] = T.top_stats(y, S['app_best'])
    out['rank mean: student best + window where-to-look'] = T.top_stats(y, T.combo([rk['app_best'], rk['win_sal_max']]))
    out['rank mean: student fin + body + window where-to-look'] = T.top_stats(y, T.combo([rk['app_fin_max'], rk['app_body_max'], rk['win_sal_max']]))
    for nm, (names, ms) in MODELS.items():
        X = T.table(R, names)[m]; out[nm] = T.top_stats(y, np.mean([c.predict_proba(X)[:, 1] for c in ms], 0))
    return out, int(y.sum()), int((1 - y).sum())

for g in (1.5, 60.0):
    res = {d: evaluate(d, g) for d in VAL}
    print(f'\n== val days together, non-whales left out, gap > {g} s: same pairs {sum(res[d][1] for d in VAL)}, different {sum(res[d][2] for d in VAL)} ==')
    for nm in res[VAL[0]][0]:
        a = [res[d][0][nm] for d in VAL]
        print(f"   {nm:56s} top-100 precision (mean) {np.mean([x['p100'] for x in a]):.2f} | same pairs at precision 0.95 / 0.90 / 0.80 (sum): {sum(x['r95'] for x in a):4d} {sum(x['r90'] for x in a):4d} {sum(x['r80'] for x in a):4d}"
              f" | per day at 0.95: {[x['r95'] for x in a]} | mean AUC {np.mean([x['auc'] for x in a]):.3f} AP {np.mean([x['ap'] for x in a]):.3f}")
