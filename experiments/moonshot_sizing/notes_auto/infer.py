"""Layers 2-4: assign every safe cluster of a noted group to one of the group's noted whales, with a confidence.

Units = pieces: the clusters a group holds after placement (place.py, aliases on), joined by the real short-gap linker
(P >= 0.9 within 60 s). Evidence (all real, fitted with the scored day held out: readers.py, pairs.py):
  unary  per piece and noted whale: noted views (coverage), height / form reader against the noted dorsal shape, fluke type
         reader, role reader (calf / mother / other), white pectoral and sex readers, frame anchors from the comments.
         A piece's term = the sum over its clusters / sqrt(number of clusters that carry the term).
  pair   LLR of same animal between clusters (soft linker, within-burst rate, HW-b6 fins, fluke head), summed between two
         pieces / sqrt(number of scored pairs); two whales in one frame = cannot.
  count  the group has exactly the noted whales: each noted whale left without a piece costs `empty` (log scale).
Methods
  simple  average-linkage merge of the pieces on the pair LLRs down to the noted count (cannot-links respected), then
          the merged pieces matched to whales by the unary sums (Hungarian); confidence of a piece = softmax over the
          best total assignment with that piece forced to each whale.
  joint   Gibbs sampling of piece -> whale with unary + pair + count terms, plus whole-label swap moves; confidence =
          posterior marginal of the chosen whale. Both multiplied by P(group right | how it was placed) (rates on the
          other days).
"""
import collections
import itertools
import math
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data  # noqa: E402
import place  # noqa: E402
import readers  # noqa: E402

UNARY = ('views', 'height', 'form', 'fluke', 'role', 'wp', 'sex', 'anchor')
PAIR = ('link', 'burst', 'fin', 'fluke')


class Ctx:
    """Everything cached for the days: records, placements, readers, pair LLRs."""

    def __init__(s, alias=True, cues=False):
        s.days = data.days()
        s.recs = {d: data.load(d) for d in s.days}
        s.R = pickle.load(open(data.CACHE / 'readers.pkl', 'rb'))
        s.L = pickle.load(open(data.CACHE / 'pairs_llr.pkl', 'rb'))
        recs = [s.recs[d] for d in s.days]
        s.place = place.run(recs, cues=cues, alias=alias)
        s.pgroup = {}
        for d in s.days:                                   # P(group right | placement kind) from the other days
            n = collections.Counter()
            for o in s.days:
                if o == d or o not in data.bench.TRAIN:
                    continue
                P = s.place[o][1]
                for k, c in s.recs[o]['clusters'].items():
                    if c['truth'] is None or P[k]['g'] is None:
                        continue
                    b = pkind(P[k])
                    n[(b, 'n')] += 1
                    n[(b, 'ok')] += place.right_group(c, P[k]['g'])
            s.pgroup[d] = {b: (n[(b, 'ok')] + 1) / (n[(b, 'n')] + 2) for b in {x[0] for x in n}}


def pkind(p):
    if p['inside']:
        return 'inside' if p['n_in'] == 1 else 'inside several'
    return 'out <= 2 min' if p['dist'] <= 2 else 'out <= 5 min' if p['dist'] <= 5 else 'out > 5 min'


# ---- unary terms ----------------------------------------------------------------------------------------------------
def _lse(v):
    m = max(v)
    return m + math.log(sum(math.exp(x - m) for x in v))


def cluster_unary(ctx, d, k, whales, wn, cfg):
    """{term: np.array over whales} for one cluster (only the terms that read something)."""
    rec = ctx.recs[d]
    c = rec['clusters'][k]
    out = {}
    oracle = cfg.get('oracle_readers')
    tw = readers.whale_of(rec, c) if oracle else None
    if cfg.get('views'):
        q = ctx.R['views'][d]
        v = np.zeros(len(wn))
        any_ = False
        for i, w in enumerate(wn):
            cov = whales[w]['cov']
            if cov is None:
                continue
            for view in readers.VIEWS:
                q1, q0 = q[view]
                p = q1 if view in cov else q0
                v[i] += math.log(p) if readers.shows(c, view) else math.log(1 - p)
            any_ = True
        if any_:
            known = [i for i, w in enumerate(wn) if whales[w]['cov'] is not None]
            v[[i for i in range(len(wn)) if i not in known]] = np.mean(v[known])
            out['views'] = v
    for attr in ('height', 'form', 'fluke', 'role', 'wp', 'sex'):
        if not cfg.get(attr):
            continue
        if oracle:
            if tw is None or k not in ctx.R[attr][d]:
                continue
            tv = readers.label(attr, tw)
            if tv is None:
                continue
            L = {cl: (0.0 if cl == tv else math.log(0.05)) for cl in readers.CLASSES[attr]}
        else:
            L = ctx.R[attr][d].get(k)
            if L is None:
                continue
        un = _lse(list(L.values())) - math.log(len(L))
        vals = [readers.label(attr, whales[w]) for w in wn]
        if all(v is None for v in vals):
            continue
        out[attr] = np.array([L[v] if v is not None and v in L else un for v in vals])
    if cfg.get('anchor'):
        fr = set(c['frames'])
        for (g, w, cam, f) in rec['log']['anchors']:
            if (cam, f) in fr and w in wn:
                v = np.zeros(len(wn))
                v[wn.index(w)] = 10.0
                out['anchor'] = v
    return out


def piece_unary(ctx, d, pieces, whales, wn, cfg):
    U = np.zeros((len(pieces), len(wn)))
    for i, pc in enumerate(pieces):
        acc = collections.defaultdict(list)
        for k in pc:
            for t, v in cluster_unary(ctx, d, k, whales, wn, cfg).items():
                acc[t].append(v)
        for t, vs in acc.items():
            U[i] += cfg.get('tu', 1.0) * cfg[t] * np.sum(vs, 0) / math.sqrt(len(vs))
    return U


# ---- pair terms -----------------------------------------------------------------------------------------------------
def piece_pairs(ctx, d, pieces, cfg):
    L = ctx.L[d]
    C = ctx.recs[d]['clusters']
    where = {k: i for i, pc in enumerate(pieces) for k in pc}
    acc = collections.defaultdict(list)
    for kind in PAIR:
        if not cfg.get('p_' + kind):
            continue
        for (a, b), v in L[kind].items():
            if a in where and b in where and where[a] != where[b]:
                i, j = sorted((where[a], where[b]))
                acc[(i, j)].append(cfg['p_' + kind] * v)
    if cfg.get('oracle_link'):                             # ceiling: perfect short-gap linking as evidence (handled in pieces)
        pass
    n = len(pieces)
    W = np.zeros((n, n))
    for (i, j), vs in acc.items():
        W[i, j] = W[j, i] = cfg.get('tp', 1.0) * np.sum(vs) / math.sqrt(len(vs))
    bad = np.zeros((n, n), bool)
    if cfg.get('cannot', True):
        for a, b in L['cannot']:
            if a in where and b in where and where[a] != where[b]:
                bad[where[a], where[b]] = bad[where[b], where[a]] = True
    W[bad] = -30.0
    return W, bad


def make_pieces(ctx, d, ks, cfg):
    C = ctx.recs[d]['clusters']
    ks = sorted(ks)
    S = set(ks)
    if cfg.get('oracle_link'):
        pr = [(a, b) for a, b in itertools.combinations(ks, 2) if C[a]['truth'] and C[a]['truth'] == C[b]['truth']
              and max(0.0, max(C[a]['t0'], C[b]['t0']) - min(C[a]['t1'], C[b]['t1'])) <= 60]
    elif cfg.get('must', True):
        pr = [(a, b) for a, b in ctx.L[d]['must'] if a in S and b in S]
    else:
        pr = []
    from size_levers import components
    return [sorted(c) for c in components(ks, pr)]


# ---- inference ------------------------------------------------------------------------------------------------------
def gibbs(U, W, empty, rng, chains=4, sweeps=160, burn=40):
    P, n = U.shape
    if n == 1:
        return np.ones((P, 1))
    marg = np.zeros((P, n))
    for ch in range(chains):
        z = rng.integers(0, n, P)
        cnt = np.bincount(z, minlength=n)
        M = np.zeros((P, n))
        for w in range(n):
            M[:, w] = W[:, z == w].sum(1)
        for sw in range(sweeps):
            for p in rng.permutation(P):
                a = z[p]
                M[:, a] -= W[:, p]
                cnt[a] -= 1
                lg = U[p] + M[p] - empty * (cnt == 0)
                lg = lg - lg.max()
                pr = np.exp(lg)
                pr /= pr.sum()
                b = rng.choice(n, p=pr)
                z[p] = b
                cnt[b] += 1
                M[:, b] += W[:, p]
            for _ in range(n):                            # whole-label swaps (pair and count terms unchanged)
                a, b = rng.choice(n, 2, replace=False)
                ia, ib = z == a, z == b
                dE = (U[ia, b] - U[ia, a]).sum() + (U[ib, a] - U[ib, b]).sum()
                if np.log(rng.random()) < dE:
                    z[ia], z[ib] = b, a
                    cnt[a], cnt[b] = cnt[b], cnt[a]
                    M[:, [a, b]] = M[:, [b, a]]
            if sw >= burn:
                marg[np.arange(P), z] += 1
    return marg / marg.sum(1, keepdims=True)


def agglo(W, bad, K):
    cl = [[i] for i in range(len(W))]
    while len(cl) > K:
        best, bi = -np.inf, None
        for i, j in itertools.combinations(range(len(cl)), 2):
            if bad[np.ix_(cl[i], cl[j])].any():
                continue
            m = W[np.ix_(cl[i], cl[j])].mean()
            if m > best:
                best, bi = m, (i, j)
        if bi is None:
            break
        i, j = bi
        cl[i] += cl[j]
        del cl[j]
    return cl


def simple(U, W, bad, n):
    cl = agglo(W, bad, n)
    V = np.array([U[c].sum(0) for c in cl])                 # merged piece x whale
    m = len(cl)
    marg = np.zeros((len(U), n))
    if m <= n:
        tot = np.full((m, n), -np.inf)
        for i in range(m):
            for w in range(n):
                cost = -V.copy()
                cost[i, :] = 1e6
                cost[i, w] = -V[i, w]
                r, c = linear_sum_assignment(cost)
                tot[i, w] = -cost[r, c].sum()
        for i in range(m):
            p = np.exp(tot[i] - tot[i].max())
            marg[cl[i]] = p / p.sum()
    else:                                               # more pieces than whales left (cannot-links): each piece on its own
        for i in range(m):
            p = np.exp(V[i] - V[i].max())
            marg[cl[i]] = p / p.sum()
    return marg


def run_day(args):
    d, cfg = args
    ctx = CTX
    rec = ctx.recs[d]
    C = rec['clusters']
    P = ctx.place[d][1]
    rng = np.random.default_rng(cfg.get('seed', 0))
    by = collections.defaultdict(list)
    for k in C:
        if P[k]['g'] is not None:
            by[P[k]['g']].append(k)
    out = {}
    for k in C:
        out[k] = (None, None, 0.0, None)
    for g, ks in by.items():
        whales = rec['log']['whales'].get(g, {})
        wn = sorted(whales)
        if not wn:
            continue
        pieces = make_pieces(ctx, d, ks, cfg)
        U = piece_unary(ctx, d, pieces, whales, wn, cfg)
        W, bad = piece_pairs(ctx, d, pieces, cfg)
        if cfg.get('method') == 'simple':
            M = simple(U, W, bad, len(wn) if cfg.get('count', True) else len(pieces))
        elif cfg.get('method') == 'first':
            M = np.zeros((len(pieces), len(wn)))
            M[:, 0] = 1.0
        else:
            M = gibbs(U, W, cfg.get('empty', 3.0) if cfg.get('count', True) else 0.0, rng, chains=cfg.get('chains', 4), sweeps=cfg.get('sweeps', 160))
        for i, pc in enumerate(pieces):
            j = int(np.argmax(M[i]))
            if M[i].max() - 1 / len(wn) < 1e-9 and len(wn) > 1 and not cfg.get('method') == 'first':
                j = int(rng.integers(len(wn)))            # no evidence: an arbitrary whale
            srt = np.sort(M[i])[::-1]
            us = np.sort(U[i])[::-1]
            sup = W[i] @ M                                  # expected pair support of the piece for each whale
            oth = [x for x in range(len(wn)) if x != j]
            own_u = (U[i, j] - max(U[i, oth])) if oth else 0.0
            own_s = (sup[j] - max(sup[oth])) if oth else 0.0
            for k in pc:
                pg = ctx.pgroup[d].get(pkind(P[k]), 0.5)
                c = C[k]
                f = [_logit(M[i, j]), _logit(srt[0]) - _logit(srt[1]) if len(wn) > 1 else 10.0, math.log(len(wn)), float(len(wn) == 1),
                     _logit(pg), math.log(len(pc)), math.log(c['size']), float(c['fin']), float(c['fluke']),
                     (us[0] - us[1]) if len(wn) > 1 else 0.0, float(len(wn) == 2), float(len(wn) == 3),
                     float(np.clip(own_u, -10, 10)), float(np.clip(own_s, -10, 10)), float(len(pieces) / len(wn))]
                out[k] = (g, wn[j], float(M[i, j]) * pg, f)
    return d, out


def _logit(p):
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def calibrate(ctx, res):
    """Replace the confidence of every assignment by P(right) from a logistic fit on the train days other than the
    scored day (features: chosen marginal, margin to the second whale, group size, placement kind, piece size, views,
    unary margin). Val days are read with a fit on the train days."""
    from sklearn.linear_model import LogisticRegression
    F = {}
    for d in ctx.days:
        C = ctx.recs[d]['clusters']
        X, y, w = [], [], []
        for k, c in C.items():
            r = res[d][k]
            if c['truth'] is None or r[1] is None:
                continue
            t = c['truth']
            X.append(r[3])
            y.append(t[0] == 'G' and r[0] == t[1] and r[1] == t[2])
            w.append(c['size'])
        F[d] = (np.array(X), np.array(y), np.array(w, float))
    new = {}
    for d in ctx.days:
        fit = [o for o in ctx.days if o != d and o in data.bench.TRAIN]
        X = np.vstack([F[o][0] for o in fit if len(F[o][0])])
        y = np.concatenate([F[o][1] for o in fit])
        w = np.concatenate([F[o][2] for o in fit])
        lr = LogisticRegression(C=1.0, max_iter=3000).fit(X, y, sample_weight=w)
        new[d] = {}
        for k, r in res[d].items():
            if r[1] is None:
                new[d][k] = r
            else:
                new[d][k] = (r[0], r[1], float(lr.predict_proba(np.array([r[3]]))[0, 1]), r[3])
    return new


CTX = None


def run(ctx, cfg, days=None, procs=8):
    global CTX
    CTX = ctx
    days = days or ctx.days
    import multiprocessing as mp
    with mp.get_context('fork').Pool(min(procs, len(days))) as pool:
        res = dict(pool.map(run_day, [(d, cfg) for d in days]))
    return res


# ---- scoring --------------------------------------------------------------------------------------------------------
def rows(ctx, res, days):
    """[(day, cluster, photos, right, assigned, conf, truth kind, group type, group size)] for labelled clusters."""
    out = []
    for d in days:
        rec = ctx.recs[d]
        for k, c in rec['clusters'].items():
            if c['truth'] is None:
                continue
            g, w, conf = res[d][k][:3]
            t = c['truth']
            right = t[0] == 'G' and g == t[1] and w == t[2]
            G = rec['log']['groups'].get(t[1]) if t[0] == 'G' else None
            nw = len(rec['log']['whales'].get(t[1], {})) if t[0] == 'G' else 0
            out.append((d, k, c['size'], right, g is not None and w is not None, conf, t[0], G['type'] if G else None, nw))
    return out


def whole_clean(ctx, res, days):
    from size_levers import Day, score, total
    rs = []
    for d in days:
        D = Day(d)
        lab = collections.defaultdict(list)
        for k in D.ids:
            g, w = res[d][k][:2]
            lab[(g, w) if w is not None else ('single', k)].append(k)
        rs.append(score(D, list(lab.values())))
    return total(rs)


def pr_curve(R, fracs=(0.25, 0.5, 0.75, 1.0), weight='photos'):
    """Precision when only the most confident share of the labelled group clusters' photos (or clusters) is kept."""
    A = [r for r in R if r[4]]
    A.sort(key=lambda r: -r[5])
    wt = np.array([r[2] if weight == 'photos' else 1 for r in A], float)
    ok = np.array([r[3] for r in A], float)
    tot = sum(r[2] if weight == 'photos' else 1 for r in R if r[6] == 'G')
    cw = np.cumsum(wt)
    out = {}
    for f in fracs:
        n = np.searchsorted(cw, f * tot, side='right')
        n = max(1, min(n, len(A)))
        out[f] = (float((ok[:n] * wt[:n]).sum() / wt[:n].sum()), float(cw[n - 1] / tot))
    return out


def at_precision(ctx, res, days, target):
    """Threshold on the confidence chosen on the OTHER days (all 23 days with a log; for a val day the train days only)
    so that pooled photo precision there is >= target; applied to each day. -> (photos right, photos assigned, labelled group photos)."""
    allR = {d: rows(ctx, res, [d]) for d in ctx.days}
    right = assigned = tot = 0
    for d in days:
        others = [o for o in ctx.days if o != d and (o in data.bench.TRAIN)]
        A = sorted([r for o in others for r in allR[o] if r[4]], key=lambda r: -r[5])
        wt = np.array([r[2] for r in A], float)
        ok = np.array([r[3] for r in A], float)
        prec = np.cumsum(ok * wt) / np.cumsum(wt)
        good = np.nonzero(prec >= target)[0]
        thr = A[good[-1]][5] if len(good) else np.inf
        for r in allR[d]:
            tot += r[2] if r[6] == 'G' else 0
            if r[4] and r[5] >= thr:
                assigned += r[2]
                right += r[2] * r[3]
    return right, assigned, tot


def summary(ctx, res, days):
    R = rows(ctx, res, days)
    G = [r for r in R if r[6] == 'G']
    ph = sum(r[2] for r in G)
    cl = len(G)
    acc_p = sum(r[2] * r[3] for r in G) / ph
    acc_c = sum(r[3] for r in G) / cl
    wc = whole_clean(ctx, res, days)
    return dict(acc_photos=acc_p, acc_clusters=acc_c, whole_clean=wc['clean'] / wc['animals'], animals=wc['animals'], pieces=wc['pieces'],
                mixed=wc['mixed'], curve=pr_curve(R), curve_c=pr_curve(R, weight='clusters'),
                p95=at_precision(ctx, res, days, 0.95), p98=at_precision(ctx, res, days, 0.98))


def fmt(name, s):
    c = s['curve']
    p95, p98 = s['p95'], s['p98']
    return (f"{name:<44s} whale right: photos {100 * s['acc_photos']:5.1f} %  clusters {100 * s['acc_clusters']:5.1f} %  | whole+clean {100 * s['whole_clean']:4.0f} % "
            f"(pieces/animal {s['pieces'] / s['animals']:.2f}, mixed {s['mixed']}) | precision at top 25/50/75/100 % of photos "
            + '/'.join(f"{100 * c[f][0]:.0f}" for f in (0.25, 0.5, 0.75, 1.0))
            + f" | at P0.95: {100 * p95[0] / p95[2]:4.1f} % of photos sorted right (realised precision {100 * p95[0] / max(1, p95[1]):.1f}), "
            f"at P0.98: {100 * p98[0] / p98[2]:4.1f} % ({100 * p98[0] / max(1, p98[1]):.1f})")
