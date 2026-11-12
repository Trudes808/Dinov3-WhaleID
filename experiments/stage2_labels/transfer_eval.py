"""Analysis of transfer_raw_<day>.pkl: catalog truth over all cluster pairs within 15 min (gap > 1.5 s).
Single scores, untrained rank combinations, and linkers trained on 2025-01-02 (researcher labels / catalog labels)."""
import pickle, numpy as np, sys, collections
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import rankdata
REF = '2025-01-02'; VAL = ['2025-01-15', '2025-02-27', '2025-03-14']
DS = pickle.load(open('ds.pkl', 'rb')); RECS = DS['recs']
def loadday(day):
    R = pickle.load(open(f'transfer_raw_{day}.pkl', 'rb')); rows = pickle.load(open(f'feats_{day}_all.pkl', 'rb'))
    assert [(r['a'], r['b']) for r in rows] == R['pairs']
    R['rows'] = rows; return R
APPN = None
def table(R, names):
    X = np.full((len(R['rows']), len(names)), np.nan)
    for i, r in enumerate(R['rows']):
        for j, nm in enumerate(names):
            v = R['S'][nm][i] if nm in R['S'] else r.get(nm)
            if v is not None and np.isfinite(v): X[i, j] = v
    return X
def nrank(v):
    out = np.full(len(v), np.nan); f = np.isfinite(v)
    if f.any(): out[f] = rankdata(v[f]) / f.sum()
    return out
def combo(cols):
    M = np.stack(cols, 1); n = np.isfinite(M).sum(1); s = np.nansum(M, 1) / np.maximum(n, 1); s[n == 0] = 0.0; return s
def top_stats(y, s):
    s = np.where(np.isfinite(s), s, -1e9); order = np.argsort(-s, kind='stable'); yy = y[order]; cum = np.cumsum(yy); k = np.arange(1, len(yy) + 1); prec = cum / k
    def at(p):
        ok = np.where(prec >= p)[0]; ok = ok[ok >= 9]       # at least 10 pairs
        return int(cum[ok[-1]]) if len(ok) else 0
    wrong = np.cumsum(1 - yy)
    return dict(p50=float(prec[49]), p100=float(prec[99]), p200=float(prec[min(199, len(prec) - 1)]), r95=at(0.95), r90=at(0.90), r80=at(0.80), w0=int(yy[wrong <= 0].sum()), w5=int(yy[wrong <= 5].sum()),
                auc=float(roc_auc_score(y, s)), ap=float(average_precision_score(y, s)))

SINGLE = ['app_fin_max', 'app_body_max', 'app_best', 'win_all_rm', 'win_sal_rm', 'win_sal_max']
LABEL = {'app_fin_max': 'student fin', 'app_body_max': 'student body', 'app_best': 'student best part', 'win_all_rm': 'window, all windows', 'win_sal_rm': 'window, where-to-look (row-max)', 'win_sal_max': 'window, where-to-look (max)'}
data = {d: loadday(d) for d in [REF] + VAL}
APP = sorted(k for k in set().union(*[set(r) for r in data[REF]['rows'][:3000]]) if k.startswith('app_'))
WIN = ['win_all_rm', 'win_all_max', 'win_sal_rm', 'win_sal_max']
SEQ = sorted(k for k in set().union(*[set(r) for r in data[REF]['rows'][:3000]]) if not k.startswith('app_') and k not in ('a', 'b', 'cat', 'cat_a', 'cat_b', 'suggested'))

# ---- linkers trained on the reference day
Rr = data[REF]; pidx = {p: i for i, p in enumerate(Rr['pairs'])}
def fit(names, mode, seed=0):
    X = table(Rr, names)
    if mode == 'researcher':
        idx = []; y = []
        for r in RECS:
            t = {'same': 1, 'different': 0, 'no_similarity': 0}.get(r['verdict'])
            if t is None: continue
            idx.append(pidx[(min(r['a'], r['b']), max(r['a'], r['b']))]); y.append(t)
        idx = np.array(idx); y = np.array(y)
    else:
        m = (Rr['cat'] != '?') & (Rr['gap'] > 1.5); idx = np.where(m)[0]; y = (Rr['cat'][idx] == 'same').astype(int)
    leaf = 10 if mode == 'researcher' else 40
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0, min_samples_leaf=leaf, early_stopping=False, random_state=seed).fit(X[idx], y)
MODELS = {}
for nm, names, mode in (('linker trained on researcher labels: student', APP, 'researcher'), ('linker trained on researcher labels: student + window', APP + WIN, 'researcher'),
                        ('linker trained on researcher labels: window only', WIN, 'researcher'),
                        ('linker trained on catalog labels: student', APP, 'catalog'), ('linker trained on catalog labels: student + window', APP + WIN, 'catalog'),
                        ('linker trained on catalog labels: window only', WIN, 'catalog'),
                        ('linker trained on catalog labels: all + window', SEQ + APP + WIN, 'catalog'), ('linker trained on researcher labels: all + window', SEQ + APP + WIN, 'researcher')):
    MODELS[nm] = (names, [fit(names, mode, s) for s in range(3)])

def evaluate(day, g=1.5, verbose=True):
    R = data[day]; m = (R['gap'] > g) & (R['cat'] != '?'); y = (R['cat'][m] == 'same').astype(int)
    out = {}
    S = {nm: (R['S'][nm][m] if nm in R['S'] else np.array([r.get(nm, np.nan) if r.get(nm) is not None else np.nan for r in R['rows']], float)[m]) for nm in SINGLE}
    cov = {nm: (float(np.isfinite(S[nm][y == 1]).mean()), float(np.isfinite(S[nm][y == 0]).mean())) for nm in SINGLE}
    for nm in SINGLE:
        st = top_stats(y, S[nm]); f = np.isfinite(S[nm]); st['auc_cov'] = float(roc_auc_score(y[f], S[nm][f])) if f.sum() > 10 and 0 < y[f].mean() < 1 else float('nan'); st['cov'] = cov[nm]; out[LABEL[nm]] = st
    rk = {nm: nrank(S[nm]) for nm in SINGLE}
    out['rank mean: student best + window where-to-look'] = top_stats(y, combo([rk['app_best'], rk['win_sal_max']]))
    out['rank mean: student fin + body + window where-to-look'] = top_stats(y, combo([rk['app_fin_max'], rk['app_body_max'], rk['win_sal_max']]))
    out['rank mean: student fin + body'] = top_stats(y, combo([rk['app_fin_max'], rk['app_body_max']]))
    if day != REF:
        for nm, (names, ms) in MODELS.items():
            X = table(R, names)[m]; out[nm] = top_stats(y, np.mean([c.predict_proba(X)[:, 1] for c in ms], 0))
    if verbose:
        print(f'\n{day}  gap > {g} s: catalog-same pairs {int(y.sum())}, catalog-different {int((1 - y).sum())}')
        for nm, st in out.items():
            extra = f"  covered-pairs AUC {st['auc_cov']:.3f} (score exists for {st['cov'][0]:.2f} of same, {st['cov'][1]:.2f} of different)" if 'cov' in st else ''
            print(f"   {nm:58s} precision of the top 50 / 100 / 200: {st['p50']:.2f} {st['p100']:.2f} {st['p200']:.2f} | same pairs found at precision 0.95 / 0.90 / 0.80: {st['r95']:4d} {st['r90']:4d} {st['r80']:4d} | AUC {st['auc']:.3f} AP {st['ap']:.3f}{extra}")
    return out, int(y.sum()), int((1 - y).sum())

if __name__ == '__main__':
    res = {}
    for g in (1.5, 60.0):
        for d in [REF] + VAL: res[(d, g)] = evaluate(d, g, verbose=(g == 1.5))
        print(f'\n== val days together, gap > {g} s: same pairs {sum(res[(d, g)][1] for d in VAL)}, different {sum(res[(d, g)][2] for d in VAL)} ==')
        for nm in res[(VAL[0], g)][0]:
            a = [res[(d, g)][0][nm] for d in VAL]
            print(f"   {nm:58s} top-100 precision (mean) {np.mean([x['p100'] for x in a]):.2f} | same pairs found at precision 0.95 / 0.90 / 0.80 (sum): {sum(x['r95'] for x in a):4d} {sum(x['r90'] for x in a):4d} {sum(x['r80'] for x in a):4d} | mean AUC {np.mean([x['auc'] for x in a]):.3f} AP {np.mean([x['ap'] for x in a]):.3f}")
    pickle.dump(res, open('transfer_eval.pkl', 'wb'))
