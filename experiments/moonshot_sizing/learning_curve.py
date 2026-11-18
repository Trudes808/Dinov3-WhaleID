"""Learning curve of the frozen-feature fin / fluke match (2026-11-17, user: "yes please try that as an experiment").

Question: would more labelled days lift the matcher? The view-match projection (experiments/view_match/vm_fit.py:
day-centred frozen DINOv3 view features -> whitened PCA 128 -> within-animal whitening -> the 32 directions that
separate the animals of a day) is refitted on 2, 5, 10 and 16 train days (nested subsets) and on all 21, and scored on
days it did not see. Nothing is fine-tuned, no GPU, read-only; no eval day is read.

Per repetition: 5 random train days are set aside as extra test days, the other 16 are shuffled and the first 2 / 5 /
10 / 16 are used. Test sets: the 3 val days (the same for every fit) and val + the 5 set-aside days (the same for the
four sizes of one repetition). Score = safe-cluster pairs of one catalogue group more than 60 s apart that both show
the part, best cosine over their instances: AUC, average precision, and the share of same-animal pairs found before
precision falls below 0.95 / 0.90 (one threshold over all test days).

    python learning_curve.py [repetitions, default 10]
"""
import collections
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from size_levers import DAYS, SPLIT, Day, at_precision, auc, group_of  # noqa: E402

THR, DIM, OUT, SHRINK = 0.5, 128, 32, 0.3          # as vm_fit.py
PARTS = ('fluke', 'dorsal')
SIZES = (2, 5, 10, 16)
HOLD = 5


def unit_blocks(X):
    X = X.astype(np.float32).reshape(len(X), 4, 768)
    return (X / np.maximum(np.linalg.norm(X, axis=2, keepdims=True), 1e-6)).reshape(len(X), -1)


class Feat:
    """One day: per part, the day-centred features of the instances that show it, their safe cluster and class label."""

    def __init__(s, day):
        s.day = day
        s.D = D = Day(day)
        z = np.load(DAYS / day / 'safe' / 'view_feat.npz')
        keys = [str(k) for k in z['keys']]
        X = unit_blocks(z['X'])
        V = json.load(open(DAYS / day / 'safe' / 'view_class.json'))['views']
        key2c = {key: k for k, c in D.C.items() for key in c['keys']}
        main = {k: any(key.endswith('~0') for key in c['keys']) for k, c in D.C.items()}   # vm_fit leaves out stacks without a main whale
        s.part = {}
        for j, part in enumerate(PARTS):
            idx = [i for i, k in enumerate(keys) if k in V and V[k][j] >= THR]
            Xp = X[idx] - X[idx].mean(0) if idx else np.zeros((0, X.shape[1]), np.float32)
            cl = [key2c.get(keys[i]) for i in idx]
            lab = [f'{day}|{D.maj[c]}' if c is not None and D.maj.get(c) and main[c] else None for c in cl]
            s.part[part] = dict(X=Xp, cl=cl, lab=np.array(lab, object))


def top_svd(X, k):
    """The k largest singular values and right vectors of X, exact, through the smaller Gram matrix (much faster than a
    full SVD of n x 3072)."""
    X = X.astype(np.float64)
    n, d = X.shape
    if n < d:
        w, U = eigh(X @ X.T, subset_by_index=[n - k, n - 1])
        S = np.sqrt(np.maximum(w[::-1], 1e-300))
        return S, (U[:, ::-1].T @ X) / S[:, None]
    w, V = eigh(X.T @ X, subset_by_index=[d - k, d - 1])
    return np.sqrt(np.maximum(w[::-1], 1e-300)), V[:, ::-1].T


def fit_projection(parts, dim=DIM, out=OUT, labels=True):
    """vm_fit.fit_projection, with the PCA and LDA sizes capped by what the data can support."""
    X = np.vstack([p['X'] for p in parts])
    y = np.concatenate([p['lab'] for p in parts])
    keep = np.array([l is not None for l in y])
    X, y = X[keep], y[keep]
    S, Vt = top_svd(X, int(min(dim, len(X) - 1)))
    dim = int(min(dim, (S > 1e-6 * S[0]).sum(), len(X) - 1))
    Pm = Vt[:dim].T / (S[:dim] / np.sqrt(len(X)))
    if not labels:
        return Pm.astype(np.float32), len(X), 0
    Z = X @ Pm
    cls = collections.defaultdict(list)
    for i, l in enumerate(y):
        cls[l].append(i)
    W = np.zeros((dim, dim))
    n = 0
    means = {}
    for l, idx in cls.items():
        if len(idx) < 2:
            continue
        z = Z[idx]
        m = z.mean(0)
        means[l] = (m, len(idx))
        W += (z - m).T @ (z - m)
        n += len(idx)
    W = W / max(n, 1)
    W = (1 - SHRINK) * W + SHRINK * np.eye(dim) * np.trace(W) / dim
    ev, E = np.linalg.eigh(W)
    T = E @ np.diag(ev ** -0.5) @ E.T
    Bm = np.zeros((dim, dim))
    rank = 0
    for day in {l.split('|')[0] for l in means}:
        ms = [(m @ T, k) for l, (m, k) in means.items() if l.startswith(day)]
        if len(ms) < 2:
            continue
        rank += len(ms) - 1
        c = np.average([m for m, _ in ms], axis=0, weights=[k for _, k in ms])
        for m, k in ms:
            Bm += k * np.outer(m - c, m - c)
    ev2, E2 = np.linalg.eigh(Bm)
    out = int(max(1, min(out, rank, dim)))
    return (Pm @ T @ E2[:, ::-1][:, :out]).astype(np.float32), len(X), len(means)


def pairs_of(F, part, P):
    """(same-animal scores, different-animal scores) of the day's cluster pairs: one catalogue group, > 60 s apart."""
    d, D = F.part[part], F.D
    Z = d['X'] @ P if P is not None else d['X'].copy()
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    by = collections.defaultdict(list)
    for i, c in enumerate(d['cl']):
        if c is not None and D.maj.get(c):
            by[c].append(i)
    pos, neg = [], []
    for a, b in itertools.combinations(sorted(by), 2):
        if group_of(D.maj[a]) != group_of(D.maj[b]) or D.gap(a, b) <= 60:
            continue
        (pos if D.maj[a] == D.maj[b] else neg).append(float((Z[by[a]] @ Z[by[b]].T).max()))
    return pos, neg


def avg_precision(pos, neg):
    v = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(-v, kind='stable')
    tp = np.cumsum(y[o])
    return float((tp / np.arange(1, len(o) + 1))[y[o] > 0].mean()) if len(pos) else float('nan')


def measure(feats, days, part, P):
    pos, neg = [], []
    for d in days:
        p, n = pairs_of(feats[d], part, P)
        pos += p
        neg += n
    pos, neg = np.array(pos), np.array(neg)
    return dict(same=len(pos), diff=len(neg), auc=auc(pos, neg), ap=avg_precision(pos, neg),
                r95=at_precision(pos, neg, 0.95) / max(1, len(pos)), r90=at_precision(pos, neg, 0.90) / max(1, len(pos)),
                n95=at_precision(pos, neg, 0.95), n90=at_precision(pos, neg, 0.90))


def fmt(rows, key, scale=1.0, nd=3):
    v = np.array([r[key] for r in rows], float) * scale
    return f'{v.mean():.{nd}f} +- {v.std():.{nd}f}' if len(v) > 1 else f'{v.mean():.{nd}f}'


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    t0 = time.time()
    TRAIN, VAL = SPLIT['train'], SPLIT['val']
    feats = {d: Feat(d) for d in TRAIN + VAL}
    print(f'loaded {len(feats)} days in {time.time() - t0:.0f} s; {reps} repetitions; sizes {SIZES} + all {len(TRAIN)} train days', flush=True)
    for part in PARTS:
        res = collections.defaultdict(list)                # (size, test set) -> rows
        info = collections.defaultdict(list)               # size -> (instances, animals)
        P_all, n, k = fit_projection([feats[d].part[part] for d in TRAIN])
        res[(len(TRAIN), 'val')].append(measure(feats, VAL, part, P_all))
        info[len(TRAIN)].append((n, k))
        res[('raw', 'val')].append(measure(feats, VAL, part, None))
        for r in range(reps):
            rng = np.random.default_rng(1000 + r)
            order = [TRAIN[i] for i in rng.permutation(len(TRAIN))]
            hold, pool = order[:HOLD], order[HOLD:]
            P0, _, _ = fit_projection([feats[d].part[part] for d in pool], labels=False)
            for tag, test in (('val', VAL), ('val+held', VAL + hold)):
                res[(0, tag)].append(measure(feats, test, part, P0))
                if tag == 'val+held':
                    res[('raw', tag)].append(measure(feats, test, part, None))
            for size in SIZES:
                P, n, k = fit_projection([feats[d].part[part] for d in pool[:size]])
                info[size].append((n, k))
                for tag, test in (('val', VAL), ('val+held', VAL + hold)):
                    res[(size, tag)].append(measure(feats, test, part, P))
            print(f'   {part}: repetition {r + 1} of {reps} done ({time.time() - t0:.0f} s)', flush=True)
        print(f'\n== {part} match: learning curve over labelled train days (mean +- sd over {reps} repetitions) ==')
        for tag, what in (('val', 'the 3 val days'), ('val+held', 'val + 5 set-aside train days')):
            ex = res[(SIZES[0], tag)]
            print(f'-- scored on {what}: {np.mean([r["same"] for r in ex]):.0f} same-animal pairs, {np.mean([r["diff"] for r in ex]):.0f} different-animal pairs')
            print(f'{"train days":>28s} {"animals":>9s} {"instances":>10s} {"AUC":>16s} {"avg precision":>16s} {"found at P 0.95 (%)":>22s} {"found at P 0.90 (%)":>22s}')
            for size in ('raw', 0) + SIZES + (len(TRAIN),):
                rows = res.get((size, tag))
                if not rows:
                    continue
                name = 'no fit (raw cosine)' if size == 'raw' else 'no labels (PCA only)' if size == 0 else str(size)
                an = f'{np.mean([x[1] for x in info[size]]):.0f}' if size in info else '-'
                ins = f'{np.mean([x[0] for x in info[size]]):.0f}' if size in info else '-'
                print(f'{name:>28s} {an:>9s} {ins:>10s} {fmt(rows, "auc"):>16s} {fmt(rows, "ap"):>16s} {fmt(rows, "r95", 100, 1):>22s} {fmt(rows, "r90", 100, 1):>22s}')
        # paired differences between consecutive sizes on the larger test set (same repetition, same test days)
        seq = SIZES
        print('-- paired gain from one size to the next, val + set-aside days (mean +- sd over repetitions; repetitions with a gain):')
        for a, b in zip(seq[:-1], seq[1:]):
            for key, sc, nm in (('auc', 1, 'AUC'), ('ap', 1, 'avg precision'), ('r95', 100, 'found at P 0.95 (points)'), ('r90', 100, 'found at P 0.90 (points)')):
                dlt = np.array([y[key] - x[key] for x, y in zip(res[(a, 'val+held')], res[(b, 'val+held')])]) * sc
                print(f'   {a:>2d} -> {b:>2d} days  {nm:<26s} {dlt.mean():+.3f} +- {dlt.std():.3f}   up in {int((dlt > 0).sum())} of {len(dlt)}')
    print(f'\ndone in {time.time() - t0:.0f} s')
