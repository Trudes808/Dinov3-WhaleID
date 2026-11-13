"""Exp 3b: is the view-specific similarity identity or just "same moment"? Out-of-fold over the train + val days
(the metric never sees the day it scores), AUC by the time gap between the two stacks, next to the student cosine
(in-sample on the train days, out-of-sample on val) and to the gap itself.
    python vm_exp3b.py fluke|dorsal [dim] [mode]  -> oof_<part>.pkl"""
import sys, warnings, pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from vm_common import *
warnings.filterwarnings('ignore')
part = sys.argv[1]; DIM = int(sys.argv[2]) if len(sys.argv) > 2 else 128; MODE = sys.argv[3] if len(sys.argv) > 3 else 'lda'; THR = 0.5


def gather(day):
    d = load_day(day); F = view_feats(day); pv = d.pf if part == 'fluke' else d.pd; rows = []; per = {}; lab = []
    for s in range(d.n):
        if d.kind[s] == 'extra': continue
        idx = [j for j, p in enumerate(pv[s]) if p >= THR]
        if not idx: continue
        per[s] = (len(rows), len(idx)); rows += [F[d.keys[s][j]] for j in idx]; lab += [f'{day}|{d.cat[s]}' if d.cat[s] else None] * len(idx)
    X = np.array(rows, np.float32) if rows else np.zeros((0, 3072), np.float32)
    return dict(day=day, d=d, X=X - (X.mean(0) if len(X) else 0), lab=np.array(lab, object), per=per)


def fit(G, dim=DIM, mode=MODE, shrink=0.3):
    X = np.vstack([g['X'] for g in G]); y = np.concatenate([g['lab'] for g in G]); keep = np.array([l is not None for l in y]); X, y = X[keep], y[keep]
    U, S, Vt = np.linalg.svd(X, full_matrices=False); Pm = Vt[:dim].T / (S[:dim] / np.sqrt(len(X))); Z = X @ Pm
    cls = {}
    for i, l in enumerate(y): cls.setdefault(l, []).append(i)
    W = np.zeros((dim, dim)); n = 0; means = {}
    for l, idx in cls.items():
        if len(idx) < 2: continue
        z = Z[idx]; m = z.mean(0); means[l] = (m, len(idx)); W += (z - m).T @ (z - m); n += len(idx)
    W = W / n; W = (1 - shrink) * W + shrink * np.eye(dim) * np.trace(W) / dim; ev, E = np.linalg.eigh(W); T = E @ np.diag(ev ** -0.5) @ E.T
    if mode == 'lda':
        Bm = np.zeros((dim, dim))
        for day in {l.split('|')[0] for l in means}:
            ms = [(m @ T, k) for l, (m, k) in means.items() if l.startswith(day)]
            if len(ms) < 2: continue
            c = np.average([m for m, _ in ms], axis=0, weights=[k for _, k in ms])
            for m, k in ms: Bm += k * np.outer(m - c, m - c)
        ev2, E2 = np.linalg.eigh(Bm); T = T @ E2[:, ::-1][:, :max(8, dim // 4)]
    elif mode == 'pca': T = np.eye(dim)
    return Pm @ T


def score_day(g, Pm):
    Z = g['X'] @ Pm; Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9); d = g['d']; out = []
    for a, b, same in scope_pairs(d):
        if a not in g['per'] or b not in g['per']: continue
        (oa, na), (ob, nb) = g['per'][a], g['per'][b]; gap = max(0.0, max(d.t0[a], d.t0[b]) - min(d.t1[a], d.t1[b]))
        out.append((g['day'], a, b, same, float((Z[oa:oa + na] @ Z[ob:ob + nb].T).max()), float((d.F if part == 'fluke' else d.Dn)[a, b]), gap, float(d.B[a, b]), float(d.P[a, b])))
    return out


if __name__ == '__main__':
    days = TRAIN + VAL; G = {day: gather(day) for day in days}; order = np.array(sorted(days)); np.random.default_rng(1).shuffle(order); rows = []
    for f in np.array_split(order, 6):
        Pm = fit([G[d] for d in days if d not in f])
        for d in f: rows += score_day(G[d], Pm)
    pickle.dump(rows, open(f'{HERE}/oof_{part}_{MODE}{DIM}.pkl', 'wb'))
    A = np.array([r[3:] for r in rows], float); day = np.array([r[0] for r in rows]); y, new, old, gap = A[:, 0], A[:, 1], A[:, 2], A[:, 3]; isval = np.isin(day, VAL)
    print(f'{part}, {MODE} {DIM}: {len(A)} stack pairs inside encounters where both stacks show the part (train + val days, out of fold); same animal {y.mean():.3f}')
    def line(name, m):
        if m.sum() < 40 or y[m].min() == y[m].max(): return
        h = m & (old > 0); r = np.argsort(np.argsort(new[h])) + np.argsort(np.argsort(old[h]))
        print(f'   {name:34s} n {int(m.sum()):6d} same {y[m].mean():.2f} | new, all pairs {roc_auc_score(y[m], new[m]):.3f} | on pairs with a student cosine ({int(h.sum())}): new {roc_auc_score(y[h], new[h]):.3f}  student {roc_auc_score(y[h], old[h]):.3f}  both (rank mean) {roc_auc_score(y[h], r):.3f} | minus gap alone {roc_auc_score(y[m], -gap[m]):.3f}')
    for tag, m0 in (('TRAIN days (student in-sample)', ~isval), ('VAL days', isval)):
        print(' ' + tag); line('all gaps', m0); line('gap <= 60 s', m0 & (gap <= 60)); line('gap 60 s - 10 min', m0 & (gap > 60) & (gap <= 600)); line('gap > 10 min', m0 & (gap > 600))
    # what the new score adds to the gap and the student cosine (logistic, fitted on train-day pairs, scored on val)
    def feats(use): return np.column_stack([np.log1p(gap)] + [c for c, u in zip((old * (old > 0) / 255, (old > 0).astype(float), new), use) for c in ([c] if u else [])])
    for name, use in (('gap + student', (1, 1, 0)), ('gap + new', (0, 0, 1)), ('gap + student + new', (1, 1, 1))):
        Xf = feats(use); clf = LogisticRegression(C=10, max_iter=500).fit(Xf[~isval], y[~isval]); print(f'   logistic on {name:22s}: val AUC {roc_auc_score(y[isval], clf.decision_function(Xf[isval])):.3f}')
