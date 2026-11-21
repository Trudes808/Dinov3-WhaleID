"""Pairwise evidence between safe clusters of a day (2026-11-19). CPU. Same fold rule as readers.py (train day: fitted
on the other train days; val day: fitted on the train days).

  must    within a burst (<= 1.5 s, link_scores tier) or the real short-gap linker P >= 0.9 (<= 60 s)
  cannot  the two clusters hold different instances of one photo (two whales in one frame)
  link    log-likelihood ratio of "same" from the linker's P (1.5-60 s pairs), logistic recalibration
  fin     HW-b6 fin embedding, best instance-pair cosine (no day-centring) + how well the fins were seen (log of the smaller
          fin box) + flank relation, logistic -> LLR of same; pairs more than 60 s apart, both showing a fin
  fluke   frozen view features through the bench's standard head (fitted on the fold's train days) -> best instance-pair
          cosine, logistic -> LLR; pairs more than 60 s apart, both showing a fluke
Every LLR = fitted log-odds minus the log-odds of the fit's base rate (so the group prior is added in the inference).
Calibration pairs: both clusters in one catalogue group (passers-by left out), on the fold's days.

    python pairs.py   -> moonshot_cache/notes_auto/pairs/<day>.pkl and aggregate AUCs
"""
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data  # noqa: E402
import readers  # noqa: E402

bench = data.bench
OUT = data.CACHE / 'pairs'


def gap(c, d):
    return max(0.0, max(c['t0'], d['t0']) - min(c['t1'], d['t1']))


def raw(rec, Zfl):
    """Raw pair features for all cluster pairs of the day."""
    C = rec['clusters']
    ks = sorted(C)
    out = dict(must=set(), cannot=set(), link={}, burst={}, fin={}, fluke={})
    for a, b, g, tier, lp in rec['links']:
        p = (min(a, b), max(a, b))
        if lp is not None and lp >= 0.9 and g <= 60:
            out['must'].add(p)
        if lp is not None:
            out['link'][p] = (lp, g)
        elif tier == 'within a burst':
            out['burst'][p] = (1.0,)
    for k, c in C.items():
        for o in c['co']:
            out['cannot'].add((min(k, o), max(k, o)))
    fins = [k for k in ks if C[k]['hw_inst'] is not None]
    for a, b in itertools.combinations(fins, 2):
        A, B = C[a], C[b]
        if gap(A, B) <= 60:
            continue
        cos = float((A['hw_inst'] @ B['hw_inst'].T).max())
        q = np.log(max(1.0, min(A['fin_px'], B['fin_px'])))
        fa = (A['flankL'], A['flankR'])
        fb = (B['flankL'], B['flankR'])
        known = any(fa) and any(fb)
        same = known and ((fa[0] and fb[0]) or (fa[1] and fb[1]))
        out['fin'][(a, b)] = [cos, q, float(known and same), float(known and not same)]
    if out['fin']:                                        # label-free per-day standardisation of the cosine
        v = np.array([x[0] for x in out['fin'].values()])
        mu, sd = v.mean(), v.std() + 1e-6
        for x in out['fin'].values():
            x.append((x[0] - mu) / sd)
    fl = [k for k in ks if k in Zfl]
    for a, b in itertools.combinations(fl, 2):
        if gap(C[a], C[b]) <= 60:
            continue
        out['fluke'][(a, b)] = (float((Zfl[a] @ Zfl[b].T).max()),)
    return out


def fluke_Z(day, fit):
    """{cluster: unit embeddings of its fluke instances} through the bench head fitted on `fit` days."""
    st = {d: bench._struct(d, 'fluke', bench._base(d)) for d in set(fit) | {day}}
    P = bench._fit([st[d] for d in fit])
    s = st[day]
    if not len(s['keys']):
        return {}
    Z = s['X'] @ P
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    by = {}
    for i, c in enumerate(s['cl']):
        if c is not None:
            by.setdefault(c, []).append(i)
    return {c: Z[ix] for c, ix in by.items()}


def feats(kind, v):
    if kind == 'fin':
        cos, q, sm, op, z = v
        return [z, z * q, q, sm, op, z * sm, z * op, cos]
    if kind in ('fluke', 'burst'):
        return [v[0]]
    lp, g = v
    lp = min(max(lp, 1e-4), 1 - 1e-4)
    return [np.log(lp / (1 - lp)), np.log1p(g)]


def same_group_pair(A, B):
    return A['truth'] and B['truth'] and A['truth'][0] == 'G' and B['truth'][0] == 'G' and A['truth'][1] == B['truth'][1]


def calib(kind, raws, recs):
    X, y = [], []
    for d, R in raws.items():
        C = recs[d]['clusters']
        for (a, b), v in R[kind].items():
            if same_group_pair(C[a], C[b]):
                X.append(feats(kind, v))
                y.append(C[a]['truth'] == C[b]['truth'])
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(np.array(X), np.array(y))
    prior = float(np.mean(y))
    return lr, np.log(prior / (1 - prior))


def build():
    days = data.days()
    recs = {d: data.load(d) for d in days}
    allfit = [d for d in bench.TRAIN]                     # the head may use train days without a log (catalogue labels only)
    raws = {}
    for d in days:
        fit = [x for x in allfit if x != d]
        raws[d] = raw(recs[d], fluke_Z(d, fit))
        print(d, {k: len(v) for k, v in raws[d].items()}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(raws, open(data.CACHE / 'pairs_raw.pkl', 'wb'))
    return raws, recs


def llrs(raws, recs):
    out = {}
    for d in raws:
        fd = readers.fit_days(d, list(raws))
        R = dict(raws[d])
        for kind in ('link', 'burst', 'fin', 'fluke'):
            if kind == 'burst':                            # a constant: the share of same animal among burst pairs of the fold
                y = [recs[x]['clusters'][a]['truth'] == recs[x]['clusters'][b]['truth'] for x in fd for a, b in raws[x]['burst']
                     if same_group_pair(recs[x]['clusters'][a], recs[x]['clusters'][b])]
                yb = [recs[x]['clusters'][a]['truth'] == recs[x]['clusters'][b]['truth'] for x in fd for a, b in raws[x]['fin']
                      if same_group_pair(recs[x]['clusters'][a], recs[x]['clusters'][b])]
                r, rb = (sum(y) + 1) / (len(y) + 2), np.mean(yb)
                R[kind] = {p: float(np.log(r / (1 - r)) - np.log(rb / (1 - rb))) for p in raws[d][kind]}
                continue
            lr, pri = calib(kind, {x: raws[x] for x in fd}, recs)
            items = list(raws[d][kind].items())
            if items:
                f = lr.decision_function(np.array([feats(kind, v) for _, v in items])) - pri
                R[kind] = {p: float(x) for (p, _), x in zip(items, f)}
            else:
                R[kind] = {}
        out[d] = R
    pickle.dump(out, open(data.CACHE / 'pairs_llr.pkl', 'wb'))
    return out


def report(L, raws, recs, days, tag):
    print(f'\n== pair evidence, {tag} ==')
    for kind in ('link', 'fin', 'fluke'):
        y, s, cos = [], [], []
        for d in days:
            C = recs[d]['clusters']
            for p, v in L[d][kind].items():
                if same_group_pair(C[p[0]], C[p[1]]):
                    y.append(C[p[0]]['truth'] == C[p[1]]['truth'])
                    s.append(v)
                    cos.append(raws[d][kind][p][0])
        y, s = np.array(y), np.array(s)
        o = np.argsort(-s)
        prec = np.cumsum(y[o]) / np.arange(1, len(o) + 1)
        ok = np.nonzero(prec >= 0.95)[0]
        o2 = np.argsort(s)
        npv = np.cumsum(~y[o2]) / np.arange(1, len(o2) + 1)
        ok2 = np.nonzero(npv >= 0.95)[0]
        print(f'   {kind:<6s} same-group pairs {len(y):6d} (same {int(y.sum())})  AUC fitted {roc_auc_score(y, s):.3f} (raw score {roc_auc_score(y, cos):.3f})  '
              f'same found at precision 0.95: {int(np.cumsum(y[o])[ok[-1]]) if len(ok) else 0}; different ruled out at 0.95: {int(np.cumsum(~y[o2])[ok2[-1]]) if len(ok2) else 0} of {int((~y).sum())}')
    n = dict(right=0, wrong=0, cross=0)
    m = dict(same=0, diff=0)
    for d in days:
        C = recs[d]['clusters']
        for a, b in L[d]['must']:
            if C[a]['truth'] and C[b]['truth']:
                n['right' if C[a]['truth'] == C[b]['truth'] else 'wrong'] += 1
        for a, b in L[d]['cannot']:
            if C[a]['truth'] and C[b]['truth']:
                m['same' if C[a]['truth'] == C[b]['truth'] else 'diff'] += 1
    print(f'   must-links (linker >= 0.9) between labelled clusters: {n["right"]} right, {n["wrong"]} wrong | '
          f'cannot-links (two whales in one frame): {m["diff"]} different animals, {m["same"]} same animal (catalogue)')


if __name__ == '__main__':
    raws, recs = build()
    L = llrs(raws, recs)
    report(L, raws, recs, [d for d in data.days() if d in bench.TRAIN], 'train days (calibration on the other train days; linker and view head are in-sample there)')
    report(L, raws, recs, [d for d in data.days() if d in bench.VAL], 'val days (calibration on the train days)')
