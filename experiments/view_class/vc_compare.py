"""Compare feature sets / heads for the view-class probes under the same day-grouped cross-validation (train + val days):
    python vc_compare.py
Reports, per variant: fluke / dorsal AUC on all label sources, three-class accuracy on all sources (hard cases
over-represented) and on the random sample r0 (population rates), and the "other" row of both."""
import json, pickle, sys, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from vc_common import *
warnings.filterwarnings('ignore')

T = all_targets(); days = {d: load_day(d) for d in DAYS if split_of(d) != 'eval'}
keys = [k for d in days for k in days[d]['keys']]; kid = {k: i for i, k in enumerate(keys)}
day_of = np.array([d for d in days for _ in days[d]['keys']])
base = np.vstack([days[d]['X'] for d in days])
flag = np.array([[('fluke' in days[d]['meta'][k]['parts']), ('fin' in days[d]['meta'][k]['parts'])] for d in days for k in days[d]['keys']], np.float32)
geo = np.array([[np.log(days[d]['meta'][k]['w'] / max(days[d]['meta'][k]['h'], 1)), np.log(days[d]['meta'][k]['w'] * days[d]['meta'][k]['h'] + 1) / 14] for d in days for k in days[d]['keys']], np.float32)
F = {}
for d in days:
    z = np.load(f'{HERE}/feats/{d}.npz'); zi = {str(k): i for i, k in enumerate(z['keys'])}; idx = [zi[k] for k in days[d]['keys']]
    for n in z.files:
        if n != 'keys': F.setdefault(n, []).append(unit(z[n][idx]))
F = {n: np.vstack(v) for n, v in F.items()}
Y = np.full((len(keys), 2), -9, np.int8); rnd = np.zeros(len(keys), bool)
for k, v in T.items():
    if k in kid: Y[kid[k]] = v['y']; rnd[kid[k]] = v['src'].get('round') == 'r0'
lab = (Y != -9).any(1); fit_days = np.array(sorted(set(day_of))); np.random.default_rng(0).shuffle(fit_days); folds = np.array_split(fit_days, 8)


def cv(X, head='lr', C=0.5, cw='balanced'):
    S = np.full((len(keys), 2), np.nan, np.float32)
    for f in folds:
        te = np.isin(day_of, f) & lab
        for j in range(2):
            m = ~np.isin(day_of, f) & (Y[:, j] >= 0); sc = StandardScaler().fit(X[m]); A = sc.transform(X[m]); B = sc.transform(X[te])
            if head == 'lr':
                S[te, j] = LogisticRegression(C=C / X.shape[1] * 100, class_weight=cw, max_iter=3000).fit(A, Y[m, j]).decision_function(B)
            elif head == 'svm':
                p = PCA(128, random_state=0).fit(A); S[te, j] = SVC(C=C, kernel='rbf', gamma='scale', class_weight=cw).fit(p.transform(A), Y[m, j]).decision_function(p.transform(B))
    return S


def report(name, S, thr=(0.0, 0.0)):
    out = [f'{name:34s}']
    for j in range(2):
        m = lab & (Y[:, j] >= 0); out.append(f'{roc_auc_score(Y[m, j], S[m, j]):.4f}')
    for sub in (lab, lab & rnd):
        m = sub & (Y >= 0).all(1); true = np.where(Y[m, 0] == 1, 0, np.where(Y[m, 1] == 1, 1, 2)); pred = np.where(S[m, 0] > thr[0], 0, np.where(S[m, 1] > thr[1], 1, 2))
        cm = np.zeros((3, 3), int)
        for t, p in zip(true, pred): cm[t, p] += 1
        out.append(f'{np.trace(cm) / cm.sum():.3f} F{cm[0].tolist()} D{cm[1].tolist()} O{cm[2].tolist()}')
    print(' | '.join(out), flush=True)


print('variant | AUC fluke | AUC dorsal | all sources: acc, rows true F / D / O -> predicted [F, D, O] | random sample r0: same')
which = sys.argv[1:] or ['all']
V = {
    'base224 cls+mean (v1)': lambda: base,
    'base224 + chart flags': lambda: np.hstack([base, flag * 3, geo]),
    'sq448 cls+mean': lambda: np.hstack([F['sq_cls'], F['sq_mean']]),
    'sq448 cls+mean+max': lambda: np.hstack([F['sq_cls'], F['sq_mean'], F['sq_max']]),
    'sq448 cls+mean+mid': lambda: np.hstack([F['sq_cls'], F['sq_mean'], F['sq_mid']]),
    'ar cls+mean': lambda: np.hstack([F['ar_cls'], F['ar_mean']]),
    'ar cls+mean+max': lambda: np.hstack([F['ar_cls'], F['ar_mean'], F['ar_max']]),
    'base224 + sq448 + ar (cls+mean)': lambda: np.hstack([base, F['sq_cls'], F['sq_mean'], F['ar_cls'], F['ar_mean']]),
    'sq448 + ar (cls+mean)': lambda: np.hstack([F['sq_cls'], F['sq_mean'], F['ar_cls'], F['ar_mean']]),
}
res = {}
for name, fn in V.items():
    X = fn(); res[name] = cv(X); report(name, res[name])
for name in ('base224 cls+mean (v1)', 'sq448 + ar (cls+mean)'):
    X = V[name](); S = cv(X, head='svm', C=4.0); res[name + ' svm'] = S; report(name + ' | rbf svm', S)
pickle.dump({'keys': keys, 'res': res, 'Y': Y, 'rnd': rnd, 'lab': lab}, open(f'{HERE}/compare_scores.pkl', 'wb'))
