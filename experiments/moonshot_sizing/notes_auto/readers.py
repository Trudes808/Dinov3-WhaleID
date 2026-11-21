"""Per-cluster readers of the noted whale attributes (2026-11-19). CPU.

Fold rule: a train day is read by a reader fitted on the OTHER train days that have a log; a val day by a reader fitted
on all train days with a log. Val days never enter a fit. Labels for fitting: the catalogue animal of the cluster ->
its noted whale row (log whale # = catalogue A number).

Readers (class-balanced logistic regression, so the output is proportional to a likelihood P(x | class)):
  height   low / moderate / high       HW-b6 fin embedding (cluster mean, day-centred), PCA 64
  form     triangle / hook / round / square / point / pyramid      same input
  fluke    fluke type 1-5              frozen view features of the fluke instances (PCA 64) + lightness of the fluke
  role     calf / mother / other       size in the frame against the other whale in the same photo, apparent size,
                                       share of two-whale photos, views
  wp       white pectoral Y / N        frozen view features of all instances, PCA 64
  sex      male / female               same input
  views    coverage rates: P(cluster shows fluke / left fin / right fin | the view is noted for its whale or not)

    python readers.py   -> moonshot_cache/notes_auto/readers.pkl, aggregate quality of each reader to stdout
"""
import collections
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data  # noqa: E402

CLASSES = dict(height=list(data.HEIGHTS), form=list(data.FORMS), fluke=[1, 2, 3, 4, 5], role=['calf', 'mother', 'other'], wp=[0, 1], sex=['Male', 'Female'])
VIEWS = ('fluke', 'L', 'R')


def fit_days(day, logged):
    tr = [d for d in data.bench.TRAIN if d in logged]
    return [d for d in tr if d != day]


def whale_of(rec, c):
    t = c['truth']
    if not t or t[0] != 'G':
        return None
    return rec['log']['whales'].get(t[1], {}).get(t[2])


def label(attr, w):
    if w is None:
        return None
    if attr == 'role':
        r = w['role']
        return None if r is None else r if r in ('calf', 'mother') else 'other'
    return w[attr]


def shows(c, v):
    return c['fluke'] if v == 'fluke' else c['flankL'] if v == 'L' else c['flankR']


# ---- inputs per cluster ---------------------------------------------------------------------------------------------
def x_of(rec, attr):
    """{cluster: feature vector} for the clusters the reader can read."""
    C = rec['clusters']
    if attr in ('height', 'form'):
        hw = [c['hw'] for c in C.values() if c['hw'] is not None]
        mu = np.mean(hw, 0) if hw else 0
        out = {}
        for k, c in C.items():
            if c['hw'] is not None:
                x = c['hw'] - mu
                out[k] = x / max(np.linalg.norm(x), 1e-9)
        return out
    if attr == 'fluke':
        return {k: np.concatenate([c['base_fluke'], 0.02 * c['light']]) if c['light'] is not None else np.concatenate([c['base_fluke'], np.zeros(4)])
                for k, c in C.items() if c['base_fluke'] is not None}
    if attr in ('wp', 'sex'):
        return {k: c['base_all'] for k, c in C.items() if c['base_all'] is not None}
    if attr == 'role':
        out = {}
        for k, c in C.items():
            r = c['rel'] or (0.0, 0.0, 0.0)
            out[k] = np.array([c['frac_multi'], *r, float(c['rel'] is not None), c['app'] if c['app'] is not None else 0.0,
                               float(c['app'] is not None), np.log(c['size']), c['pfin'], c['pfluke']], float)
        return out
    raise KeyError(attr)


class Reader:
    def __init__(s, attr, pca=64):
        s.attr, s.pca = attr, pca

    def fit(s, recs):
        X, y = [], []
        for rec in recs:
            xs = x_of(rec, s.attr)
            for k, c in rec['clusters'].items():
                lab = label(s.attr, whale_of(rec, c))
                if k in xs and lab is not None and lab in CLASSES[s.attr]:
                    X.append(xs[k])
                    y.append(lab)
        X = np.array(X, np.float64)
        s.classes = [c for c in CLASSES[s.attr] if c in set(y)]
        s.n_train = len(y)
        if s.attr == 'role':
            s.pre = StandardScaler().fit(X)
        else:
            s.pre = PCA(min(s.pca, len(X) - 1), random_state=0).fit(X)
        s.lr = LogisticRegression(C=0.5, max_iter=3000, class_weight='balanced').fit(s.pre.transform(X), y)
        return s

    def read(s, rec):
        """{cluster: {class: log-likelihood up to a constant}} (classes not seen in training get the mean)."""
        xs = x_of(rec, s.attr)
        if not xs:
            return {}
        ks = list(xs)
        P = s.lr.predict_proba(s.pre.transform(np.array([xs[k] for k in ks], np.float64)))
        L = np.log(np.clip(P, 1e-4, 1))
        out = {}
        for i, k in enumerate(ks):
            d = dict(zip(s.lr.classes_, L[i]))
            m = float(np.mean(L[i]))
            out[k] = {c: float(d.get(c, m)) for c in CLASSES[s.attr]}
        return out


def view_rates(recs):
    """q[v] = (P(cluster shows v | v noted for its whale), P(shows v | v not noted)), coverage known."""
    n = collections.Counter()
    for rec in recs:
        for c in rec['clusters'].values():
            w = whale_of(rec, c)
            if w is None or w['cov'] is None:
                continue
            for v in VIEWS:
                noted = v in w['cov']
                n[(v, noted)] += 1
                n[(v, noted, 'shows')] += shows(c, v)
    return {v: ((n[(v, True, 'shows')] + 1) / (n[(v, True)] + 2), (n[(v, False, 'shows')] + 1) / (n[(v, False)] + 2)) for v in VIEWS}


def build():
    days = data.days()
    recs = {d: data.load(d) for d in days}
    out = collections.defaultdict(dict)
    for d in days:
        fd = fit_days(d, days)
        tr = [recs[x] for x in fd]
        for attr in CLASSES:
            out[attr][d] = Reader(attr).fit(tr).read(recs[d])
        out['views'][d] = view_rates(tr)
        print(d, 'fitted on', len(fd), 'days', flush=True)
    pickle.dump(dict(out), open(data.CACHE / 'readers.pkl', 'wb'))
    return dict(out), recs


def quality(R, recs, days, tag):
    """What each reader achieves on its own: cluster accuracy (balanced), and inside groups: is the cluster's own whale's
    noted value ranked first among the values noted in its group (when they differ)."""
    print(f'\n== readers on their own, {tag} ({len(days)} days) ==')
    for attr in CLASSES:
        y, p, hit, tot, ch = [], [], 0, 0, 0.0
        for d in days:
            rec = recs[d]
            for k, c in rec['clusters'].items():
                w = whale_of(rec, c)
                lab = label(attr, w)
                if k not in R[attr][d] or lab is None:
                    continue
                L = R[attr][d][k]
                y.append(lab)
                p.append(max(L, key=L.get))
                vals = {label(attr, ww) for ww in rec['log']['whales'][c['truth'][1]].values()} - {None}
                if len(vals) > 1:
                    tot += 1
                    hit += max(vals, key=lambda v: L[v]) == lab
                    ch += 1 / len(vals)
        y, p = np.array(y, object), np.array(p, object)
        cls = sorted(set(y), key=str)
        bal = np.mean([np.mean(p[y == c] == c) for c in cls])
        maj = collections.Counter(y).most_common(1)[0]
        extra = ''
        if attr in ('wp', 'sex'):
            s = np.array([R[attr][d][k][CLASSES[attr][1]] - R[attr][d][k][CLASSES[attr][0]] for d in days for k, c in recs[d]['clusters'].items()
                          if k in R[attr][d] and label(attr, whale_of(recs[d], c)) is not None])
            extra = f', AUC {roc_auc_score(y == CLASSES[attr][1], s):.3f}'
        if attr == 'role':
            s = np.array([R[attr][d][k]['calf'] - max(R[attr][d][k]['mother'], R[attr][d][k]['other']) for d in days for k, c in recs[d]['clusters'].items()
                          if k in R[attr][d] and label(attr, whale_of(recs[d], c)) is not None])
            extra = f', calf AUC {roc_auc_score(y == "calf", s):.3f}'
        print(f'   {attr:<7s} clusters {len(y):5d}  accuracy {np.mean(p == y):.3f} (majority class {maj[1] / len(y):.3f}), balanced {bal:.3f} (chance {1 / len(cls):.3f}){extra} | '
              f'own whale\'s value first among its group\'s {hit}/{tot} = {hit / max(1, tot):.3f} (chance {ch / max(1, tot):.3f})')
    n = collections.Counter()
    for d in days:
        rec = recs[d]
        for c in rec['clusters'].values():
            w = whale_of(rec, c)
            if w is None or w['cov'] is None:
                continue
            for v in VIEWS:
                if shows(c, v):
                    n[(v, 'shows')] += 1
                    n[(v, 'noted')] += v in w['cov']
    print('   views: of the clusters that show a view, share whose own whale has it noted: ' +
          ', '.join(f'{v} {n[(v, "noted")]}/{n[(v, "shows")]} = {n[(v, "noted")] / max(1, n[(v, "shows")]):.3f}' for v in VIEWS))


if __name__ == '__main__':
    R, recs = build()
    quality(R, recs, [d for d in data.days() if d in data.bench.TRAIN], 'train days, each read by a fit on the other train days')
    quality(R, recs, [d for d in data.days() if d in data.bench.VAL], 'val days, fit on the train days')
