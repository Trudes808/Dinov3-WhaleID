"""Fit the view-specific match projections and their calibration -> experiments/safe_clusters/view_match.pkl
    python vm_fit.py
Per part (fluke, dorsal fin), on the instances the view classifier says show it: centre per day, whitened PCA 128,
within-animal whitening (shrink 0.3), then the 32 directions that separate the animals of one day best (LDA with the
between-animal scatter taken inside each day). Classes = the catalogue animal of the instance's stack.
  all   projection from the 21 train days: used for the val, eval and any new day
  loo   one projection per train day, fitted without it: a train day is never scored by a projection that saw it
Calibration: on the out-of-sample similarities of the 24 train + val days, at the stack-animal level inside catalogue
encounters (the best cosine of a stack to any stack of an animal, as the page's rank() takes it): logistic P(same)."""
import sys, pickle, warnings, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from vm_common import *
warnings.filterwarnings('ignore')
DIM, OUT, SHRINK, THR = 128, 32, 0.3, 0.5; PARTS = ('fluke', 'dorsal')


def gather(day, part):
    d = load_day(day); F = view_feats(day); pv = d.pf if part == 'fluke' else d.pd; rows = []; per = {}; lab = []
    for s in range(d.n):
        if d.kind[s] == 'extra': continue
        idx = [j for j, p in enumerate(pv[s]) if p >= THR]
        if not idx: continue
        per[s] = (len(rows), len(idx)); rows += [F[d.keys[s][j]] for j in idx]; lab += [f'{day}|{d.cat[s]}' if d.cat[s] else None] * len(idx)
    X = np.array(rows, np.float32) if rows else np.zeros((0, 3072), np.float32)
    return dict(day=day, d=d, X=X - (X.mean(0) if len(X) else 0), lab=np.array(lab, object), per=per)


def fit_projection(G):
    X = np.vstack([g['X'] for g in G]); y = np.concatenate([g['lab'] for g in G]); keep = np.array([l is not None for l in y]); X, y = X[keep], y[keep]
    U, S, Vt = np.linalg.svd(X, full_matrices=False); Pm = Vt[:DIM].T / (S[:DIM] / np.sqrt(len(X))); Z = X @ Pm; cls = {}
    for i, l in enumerate(y): cls.setdefault(l, []).append(i)
    W = np.zeros((DIM, DIM)); n = 0; means = {}
    for l, idx in cls.items():
        if len(idx) < 2: continue
        z = Z[idx]; m = z.mean(0); means[l] = (m, len(idx)); W += (z - m).T @ (z - m); n += len(idx)
    W = W / n; W = (1 - SHRINK) * W + SHRINK * np.eye(DIM) * np.trace(W) / DIM; ev, E = np.linalg.eigh(W); T = E @ np.diag(ev ** -0.5) @ E.T; Bm = np.zeros((DIM, DIM))
    for day in {l.split('|')[0] for l in means}:
        ms = [(m @ T, k) for l, (m, k) in means.items() if l.startswith(day)]
        if len(ms) < 2: continue
        c = np.average([m for m, _ in ms], axis=0, weights=[k for _, k in ms])
        for m, k in ms: Bm += k * np.outer(m - c, m - c)
    ev2, E2 = np.linalg.eigh(Bm)
    return (Pm @ T @ E2[:, ::-1][:, :OUT]).astype(np.float32), len(X), len(means)


def stack_sim(g, Pm):
    """(n stacks x n stacks) best cosine between the part instances of two stacks; NaN where a stack does not show the part."""
    d = g['d']; Z = g['X'] @ Pm; Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9); S = np.full((d.n, d.n), np.nan, np.float32); ids = sorted(g['per'])
    if not ids: return S
    starts = np.array([g['per'][s][0] for s in ids]); R = np.maximum.reduceat(np.maximum.reduceat(Z @ Z.T, starts, axis=0), starts, axis=1); S[np.ix_(ids, ids)] = R
    return S


def animal_rows(d, S, student):
    """Stack-animal rows inside catalogue encounters: (same animal, best new cosine, best student code, both exist)."""
    out = []
    for g, cs in d.scope.items():
        for c in cs:
            for s in d.animals[c]:
                for c2 in cs:
                    T = [t for t in d.animals[c2] if t != s and (s, t) not in d.conf]
                    if not T: continue
                    v = S[s, T]; v = v[np.isfinite(v)]; st = student[s, T]
                    if len(v): out.append((c == c2, float(v.max()), float(st.max())))
    return out


if __name__ == '__main__':
    t0 = time.time(); model = {'dim': DIM, 'out': OUT, 'shrink': SHRINK, 'thr': THR, 'fitted': time.strftime('%Y-%m-%d'), 'train_days': TRAIN, 'parts': {}}
    for part in PARTS:
        G = {day: gather(day, part) for day in TRAIN + VAL}
        P_all, n, k = fit_projection([G[d] for d in TRAIN]); loo = {}
        for day in TRAIN: loo[day] = fit_projection([G[d] for d in TRAIN if d != day])[0]
        rows = {}
        for day in TRAIN + VAL:
            S = stack_sim(G[day], loo.get(day, P_all)); rows[day] = animal_rows(G[day]['d'], S, G[day]['d'].F if part == 'fluke' else G[day]['d'].Dn)
        A = np.array([r for day in TRAIN + VAL for r in rows[day]], float); isval = np.concatenate([[day in VAL] * len(rows[day]) for day in TRAIN + VAL]).astype(bool)
        lr = LogisticRegression(C=100, max_iter=500).fit(A[:, 1:2], A[:, 0]); cal = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))
        lrv = LogisticRegression(C=100, max_iter=500).fit(A[~isval, 1:2], A[~isval, 0])                       # the same from the train days only, to check it on val
        pv = lrv.predict_proba(A[isval, 1:2])[:, 1]
        m = A[:, 2] > 0
        print(f'{part}: projection from {n} instances of {k} animals on {len(TRAIN)} train days ({time.time() - t0:.0f} s); {len(A)} stack-animal rows out of sample ({int(isval.sum())} on val), same animal {A[:, 0].mean():.3f}')
        for tag, mm in (('train days (student in-sample, new out of sample)', ~isval), ('val days (both out of sample)', isval)):
            h = mm & m; print(f'   {tag:52s} AUC new {roc_auc_score(A[mm, 0], A[mm, 1]):.3f} ({int(mm.sum())} rows) | where the student has a cosine ({int(h.sum())}): new {roc_auc_score(A[h, 0], A[h, 1]):.3f} student {roc_auc_score(A[h, 0], A[h, 2]):.3f}')
        print(f'   calibration P(same) = sigmoid({cal[0]:.3f} x cosine {cal[1]:+.3f}); cosine 0.2 -> {1 / (1 + np.exp(-(cal[0] * 0.2 + cal[1]))):.2f}, 0.5 -> {1 / (1 + np.exp(-(cal[0] * 0.5 + cal[1]))):.2f}, 0.8 -> {1 / (1 + np.exp(-(cal[0] * 0.8 + cal[1]))):.2f}; fitted on train only and read on val: mean P {pv.mean():.3f} against {A[isval, 0].mean():.3f} same')
        model['parts'][part] = {'all': P_all, 'loo': loo, 'cal': cal, 'instances': n, 'animals': k}
    pickle.dump(model, open(f'{R}/experiments/safe_clusters/view_match.pkl', 'wb')); print('saved experiments/safe_clusters/view_match.pkl')
