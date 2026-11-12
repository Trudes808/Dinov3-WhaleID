"""Fit and check the view-class probes: vc_fit.py [--C 0.5] [--save NAME] [--feat inst|...]
Two logistic regressions (fluke, dorsal fin) on frozen DINOv3 crop features of an instance (cls + mean patch token,
DAY/safe/inst_feat.npz), trained on every label source of the train + val days (vc_common.all_targets; marginal codes
carry no loss). Prints: the chart flags the page uses today against the by-eye labels (a random sample, so its rates
are the population's); day-grouped cross-validation of the probes; the probes on the eval days against labels that
were never fitted on. Writes oof_scores.pkl (every instance scored by probes that did not see its day).
--save NAME writes experiments/safe_clusters/NAME.pkl fitted on all train + val labels."""
import argparse, json, pickle, collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from vc_common import *

ap = argparse.ArgumentParser(); ap.add_argument('--C', type=float, default=0.5); ap.add_argument('--save'); ap.add_argument('--folds', type=int, default=8)
ap.add_argument('--eval', action='store_true', help='also print the eval-day checks (read sparingly)')
a = ap.parse_args()
T = all_targets(); days = {d: load_day(d) for d in DAYS}
keys = [k for d in DAYS for k in days[d]['keys']]; kid = {k: i for i, k in enumerate(keys)}
X = np.vstack([days[d]['X'] for d in DAYS]); day_of = np.array([d for d in DAYS for _ in days[d]['keys']]); split = np.array([split_of(d) for d in day_of])
flag = np.array([[('fluke' in days[d]['meta'][k]['parts']), ('fin' in days[d]['meta'][k]['parts'])] for d in DAYS for k in days[d]['keys']])
Y = np.full((len(keys), 2), -9, np.int8); eye = np.zeros(len(keys), bool); rnd = np.zeros(len(keys), bool); miss = 0
for k, v in T.items():
    if k in kid: Y[kid[k]] = v['y']; eye[kid[k]] = 'eye' in v['src']; rnd[kid[k]] = v['src'].get('round') in ('r0', 'ev')
    else: miss += 1
fitp = split != 'eval'                                        # the pool the probes may learn from
print(f'{int(((Y != -9).any(1) & fitp).sum())} labelled instances on train + val days, {int(((Y != -9).any(1) & ~fitp).sum())} on eval days ({miss} labelled keys are not in a safe cluster)')
for j, n in enumerate(('fluke', 'dorsal')):
    print(f'   {n}: yes {int(((Y[:, j] == 1) & fitp).sum())}, no {int(((Y[:, j] == 0) & fitp).sum())}, marginal {int(((Y[:, j] == -1) & fitp).sum())}')


def pr(y, s):
    y = np.asarray(y); s = np.asarray(s); tp = (y == 1) & s; return f'precision {tp.sum() / max(s.sum(), 1):.3f} ({tp.sum()}/{s.sum()}) recall {tp.sum() / max((y == 1).sum(), 1):.3f} ({tp.sum()}/{(y == 1).sum()})'


def fit(tr, j):
    m = tr & (Y[:, j] >= 0); sc = StandardScaler().fit(X[m]); clf = LogisticRegression(C=a.C / X.shape[1] * 100, class_weight='balanced', max_iter=3000).fit(sc.transform(X[m]), Y[m, j])
    return sc, clf


def score(model, Xs):
    sc, clf = model; return clf.decision_function(sc.transform(Xs))


S = np.full((len(keys), 2), np.nan, np.float32)
fit_days = np.array(sorted(set(day_of[fitp]))); rng = np.random.default_rng(0); rng.shuffle(fit_days)
for f in np.array_split(fit_days, a.folds):
    te = np.isin(day_of, f)
    for j in range(2): S[te, j] = score(fit(fitp & ~te, j), X[te])
full = [fit(fitp, j) for j in range(2)]
for j in range(2): S[~fitp, j] = score(full[j], X[~fitp])


def three(m, pred_f, pred_d):
    true = np.where(Y[m, 0] == 1, 0, np.where(Y[m, 1] == 1, 1, 2)); pred = np.where(pred_f[m], 0, np.where(pred_d[m], 1, 2)); cm = np.zeros((3, 3), int)
    for t, p in zip(true, pred): cm[t, p] += 1
    return f'{cm.tolist()} accuracy {np.trace(cm) / max(cm.sum(), 1):.3f}'


def block(title, m0):
    print(f'\n-- {title}')
    for j, n in enumerate(('fluke', 'dorsal')):
        m = m0 & (Y[:, j] >= 0)
        if (Y[m, j] == 1).sum() == 0 or (Y[m, j] == 0).sum() == 0: print(f'   {n}: one class only ({int(m.sum())})'); continue
        print(f'   {n:6s} chart flag: {pr(Y[m, j], flag[m, j])}')
        print(f'   {n:6s} probe AUC {roc_auc_score(Y[m, j], S[m, j]):.4f} | at 0: {pr(Y[m, j], S[m, j] > 0)} | at 1.5: {pr(Y[m, j], S[m, j] > 1.5)}')
    m = m0 & (Y >= 0).all(1)
    print(f'   three classes, rows true fluke / dorsal / other, columns predicted ({int(m.sum())} clear cases)\n      probe      {three(m, S[:, 0] > 0, S[:, 1] > 0)}\n      chart flag {three(m, flag[:, 0], flag[:, 1])}')


block('RANDOM by-eye sample (round r0), train + val days, day-grouped cross-validation: the population rates', fitp & rnd)
block('all label sources, train + val days, day-grouped cross-validation', fitp)
if a.eval:
    block('EVAL days, RANDOM by-eye sample labelled blind (round ev; probes fitted on train + val only)', ~fitp & rnd)
    block('EVAL days, the other sources: researcher clicks of 2024-12-18 + catalogue flukes (few "other" in these)', ~fitp & ~rnd)
    for d in sorted(set(day_of[~fitp])):
        m = (day_of == d) & (Y[:, 0] == 1); print(f'      {d}: fluke labels {int(m.sum())}, probe > 0 on {int((S[m, 0] > 0).sum())}, flag on {int(flag[m, 0].sum())}')
print(f'\n-- share of all {len(keys)} instances: probe fluke {np.mean(S[:, 0] > 0):.3f}, dorsal only {np.mean((S[:, 1] > 0) & ~(S[:, 0] > 0)):.3f}, other {np.mean((S[:, 1] <= 0) & (S[:, 0] <= 0)):.3f} | flags fluke {flag[:, 0].mean():.3f}, fin only {np.mean(flag[:, 1] & ~flag[:, 0]):.3f}')
pickle.dump({'keys': keys, 'S': S, 'day': day_of, 'Y': Y}, open(f'{HERE}/oof_scores.pkl', 'wb'))
if a.save:
    out = {'name': a.save, 'C': a.C, 'labels': int(((Y != -9).any(1) & fitp).sum()), 'days': sorted(set(day_of[fitp & (Y != -9).any(1)]))}
    for j, n in enumerate(('fluke', 'dorsal')):
        sc, clf = full[j]; out[n] = {'mean': sc.mean_.astype(np.float32), 'scale': sc.scale_.astype(np.float32), 'coef': clf.coef_[0].astype(np.float32), 'intercept': float(clf.intercept_[0])}
    pickle.dump(out, open(f'{R}/experiments/safe_clusters/{a.save}.pkl', 'wb')); print('saved', a.save)
