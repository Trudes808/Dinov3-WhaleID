"""Shared bench for fin / fluke matcher candidates (2026-11-18). CPU only, read-only on the day folders.

WHAT IT SCORES  pairs of safe clusters of ONE catalogue group, more than 60 s apart, that both show the part (view
classifier P >= 0.5): same animal against different animals. A cluster pair's score = whatever the candidate gives
(for embeddings: the best cosine over the instance pairs). Numbers: AUC, average precision (AP) and the share of the
same-animal pairs found before precision falls below 0.95 / 0.90 (one threshold over all test days).

STANDARD PROTOCOL (the same splits for every candidate)  20 repetitions; in each, 5 random train days are set aside,
the other 16 are shuffled, a head is fitted on the first 5 / 10 / 16 of them and scored on val + the 5 set-aside days;
plus one fit on all 21 train days scored on the 3 val days. Every candidate is compared with BASE = the live matcher's
features (DAY/safe/view_feat.npz through the same head) on the SAME pairs, repetition by repetition.

    import bench
    bench.evaluate(name, part, X)         X = {day: {instance key: 1-d vector}}  features WITHOUT labels in them: they go
                                          through the standard head (day-centring, whitened PCA 128, within-animal
                                          whitening, 32 LDA directions); also scored raw (cosine, no fit)
    bench.evaluate_pairs(name, part, S)   S = {day: {(cluster a, cluster b): score}}, a < b, higher = more alike: for
                                          scorers that need NO fitting on these days (local matching, public models)
    bench.compare(name, part, S, test_days, base_fit_days)
                                          one fixed split: for anything TRAINED on labelled days (score only days the
                                          model did not see; BASE is fitted on base_fit_days for an equal-data comparison)
    bench.embed_scores(part, X, days, fit_days=None)   cluster-pair scores from embeddings (raw cosine, or the head
                                          fitted on fit_days), to feed compare / to fuse with other scores
    bench.base_scores(part, days, fit_days)            the live matcher's scores, for fusion
Helpers: instances(day, part) -> [(key, cluster, animal)], clusters(day, part), pairs(day, part) -> [(a, b, same)],
box(day, key) (x0, y0, x1, y1 in the 2560-px image), image_path(day, key), flank(day, cluster), TRAIN, VAL.
Results are printed and saved to results/<name>__<part>.json. Eval days are refused.
"""
import collections
import functools
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import learning_curve as L  # noqa: E402
from size_levers import DAYS, SPLIT, Day, at_precision, auc, group_of  # noqa: E402

TRAIN, VAL, EVAL = list(SPLIT['train']), list(SPLIT['val']), set(SPLIT['eval'])
PARTS = ('fluke', 'dorsal')
THR, REPS, SIZES, HOLD = 0.5, 20, (5, 10, 16), 5
RESULTS = HERE / 'results'


def _ok(day):
    if day in EVAL:
        raise ValueError(f'{day} is an eval day: not read here')
    return day


@functools.lru_cache(None)
def D(day):
    return Day(_ok(day))


@functools.lru_cache(None)
def _views(day):
    return json.load(open(DAYS / _ok(day) / 'safe' / 'view_class.json'))


@functools.lru_cache(None)
def _base(day):
    z = np.load(DAYS / _ok(day) / 'safe' / 'view_feat.npz')
    X = L.unit_blocks(z['X'])
    return {str(k): X[i] for i, k in enumerate(z['keys'])}


@functools.lru_cache(None)
def _key2c(day):
    return {key: k for k, c in D(day).C.items() for key in c['keys']}


@functools.lru_cache(None)
def instances(day, part):
    """[(instance key, safe cluster id or None, catalogue animal or None)]: every instance that shows the part."""
    j = PARTS.index(part)
    V, k2c, Dd = _views(day)['views'], _key2c(day), D(day)
    out = []
    for key in _base(day):
        v = V.get(key)
        if v and v[j] >= THR:
            c = k2c.get(key)
            out.append((key, c, Dd.maj.get(c) if c is not None else None))
    return out


def clusters(day, part):
    """{safe cluster id: [instance keys that show the part]} for the clusters with a catalogue animal."""
    out = collections.defaultdict(list)
    for key, c, a in instances(day, part):
        if c is not None and a:
            out[c].append(key)
    return dict(out)


@functools.lru_cache(None)
def pairs(day, part):
    """[(cluster a, cluster b, same animal)], a < b: the pairs the bench scores on this day."""
    Dd = D(day)
    out = []
    for a, b in itertools.combinations(sorted(clusters(day, part)), 2):
        if group_of(Dd.maj[a]) == group_of(Dd.maj[b]) and Dd.gap(a, b) > 60:
            out.append((a, b, Dd.maj[a] == Dd.maj[b]))
    return out


def box(day, key):
    """(x0, y0, x1, y1) of the instance in the day's 2560-px image, as the view features were cut."""
    return tuple(float(x) for x in _views(day)['box'][key])


def image_path(day, key):
    return DAYS / _ok(day) / 'images' / f"{key.rsplit('~', 1)[0]}.jpg"


def flank(day, cluster):
    """Set of head sides (+1 / -1) of the cluster's confident dorsal views; empty = unknown."""
    return D(day).flank[cluster]


def splits(reps=REPS):
    """[(set-aside train days, the other 16 shuffled)]: the same for every candidate."""
    out = []
    for r in range(reps):
        rng = np.random.default_rng(1000 + r)
        order = [TRAIN[i] for i in rng.permutation(len(TRAIN))]
        out.append((order[:HOLD], order[HOLD:]))
    return out


# ---- embeddings -> cluster-pair scores -------------------------------------------------------------------------------
def _struct(day, part, X_by_key, center=True):
    Dd, k2c = D(day), _key2c(day)
    keys = [k for k, c, a in instances(day, part) if k in X_by_key]
    X = np.array([np.asarray(X_by_key[k], np.float32).ravel() for k in keys], np.float32) if keys else np.zeros((0, 1), np.float32)
    if center and len(keys):
        X = X - X.mean(0)
    cl = [k2c.get(k) for k in keys]
    main = {c: any(key.endswith('~0') for key in Dd.C[c]['keys']) for c in set(cl) if c is not None}
    lab = [f'{day}|{Dd.maj[c]}' if c is not None and Dd.maj.get(c) and main[c] else None for c in cl]
    return dict(X=X, cl=cl, lab=np.array(lab, object), keys=keys)


def _pair_scores(day, part, st, P):
    Dd = D(day)
    if not len(st['keys']):
        return {}
    Z = st['X'] @ P if P is not None else st['X'].astype(np.float32)
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    by = collections.defaultdict(list)
    for i, c in enumerate(st['cl']):
        if c is not None and Dd.maj.get(c):
            by[c].append(i)
    return {(a, b): float((Z[by[a]] @ Z[by[b]].T).max()) for a, b, _ in pairs(day, part) if a in by and b in by}


def _fit(structs, labels=True):
    d = max(s['X'].shape[1] for s in structs)
    return L.fit_projection(structs, dim=min(L.DIM, d), labels=labels)[0]


def embed_scores(part, X, days, fit_days=None, center=True, _structs=None):
    """{day: {(a, b): score}} from embeddings X = {day: {key: vector}}: raw cosine (fit_days None) or the standard head
    fitted on fit_days (which must be in X)."""
    st = _structs or {d: _struct(d, part, X[d], center) for d in set(days) | set(fit_days or [])}
    P = _fit([st[d] for d in fit_days]) if fit_days else None
    return {d: _pair_scores(d, part, st[d], P) for d in days}


def base_scores(part, days, fit_days, keys=None):
    """The live matcher's features through the standard head. keys = {day: set of instance keys} restricts them."""
    X = {d: ({k: v for k, v in _base(d).items() if k in keys[d]} if keys else _base(d)) for d in set(days) | set(fit_days)}
    return embed_scores(part, X, days, fit_days)


# ---- measures --------------------------------------------------------------------------------------------------------
def _ap(pos, neg):
    v = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.lexsort((y, -v))                                       # ties: different-animal pairs first (2026-11-18)
    tp = np.cumsum(y[o])
    return float((tp / np.arange(1, len(o) + 1))[y[o] > 0].mean()) if len(pos) else float('nan')


def measure(part, S, days, only=None):
    """S = {day: {(a, b): score}}; only = {day: set of pairs} restricts to those pairs."""
    pos, neg = [], []
    for d in days:
        for a, b, same in pairs(d, part):
            if (a, b) in S.get(d, {}) and (only is None or (a, b) in only[d]):
                (pos if same else neg).append(S[d][(a, b)])
    pos, neg = np.array(pos, float), np.array(neg, float)
    n = max(1, len(pos))
    return dict(same=len(pos), diff=len(neg), auc=auc(pos, neg), ap=_ap(pos, neg), r95=at_precision(pos, neg, 0.95) / n, r90=at_precision(pos, neg, 0.90) / n)


def _common(part, A, B, days):
    return {d: {p for p in A.get(d, {}) if p in B.get(d, {})} for d in days}


def _row(m):
    return f"AUC {m['auc']:.3f}  AP {m['ap']:.3f}  found at P 0.95 {100 * m['r95']:5.1f} %  at 0.90 {100 * m['r90']:5.1f} %  ({m['same']} same / {m['diff']} different)"


def _mean(rows, key):
    v = np.array([r[key] for r in rows], float)
    return float(v.mean()), float(v.std())


def _line(tag, rows):
    return (f"{tag:<34s} AUC {_mean(rows, 'auc')[0]:.3f} +- {_mean(rows, 'auc')[1]:.3f}  AP {_mean(rows, 'ap')[0]:.3f} +- {_mean(rows, 'ap')[1]:.3f}  "
            f"found at P 0.95 {100 * _mean(rows, 'r95')[0]:4.1f} +- {100 * _mean(rows, 'r95')[1]:3.1f} %  at 0.90 {100 * _mean(rows, 'r90')[0]:4.1f} +- {100 * _mean(rows, 'r90')[1]:3.1f} %")


def _delta(tag, cand, base):
    out = []
    for key, sc, nm in (('auc', 1, 'AUC'), ('ap', 1, 'AP'), ('r95', 100, 'found at P 0.95, points'), ('r90', 100, 'found at P 0.90, points')):
        d = np.array([c[key] - b[key] for c, b in zip(cand, base)]) * sc
        out.append(f'{nm} {d.mean():+.3f} +- {d.std():.3f} (up in {int((d > 0).sum())} of {len(d)})')
    return f'{tag:<34s} ' + ' | '.join(out)


def _coverage(part, S, days):
    tot = sum(len(pairs(d, part)) for d in days)
    return sum(len(S.get(d, {})) for d in days), tot


def _flank_rows(part, S, days, only):
    if part != 'dorsal':
        return {}
    rel = collections.defaultdict(lambda: {d: set() for d in days})
    for d in days:
        for a, b, _ in pairs(d, part):
            fa, fb = flank(d, a), flank(d, b)
            r = 'flank unknown' if not fa or not fb else 'same flank' if fa == fb and len(fa) == 1 else 'opposite flanks' if not (fa & fb) else 'mixed'
            if (a, b) in only[d]:
                rel[r][d].add((a, b))
    return {r: measure(part, S, days, sel) for r, sel in rel.items()}


def _save(name, part, out):
    RESULTS.mkdir(exist_ok=True)
    json.dump(out, open(RESULTS / f'{name}__{part}.json', 'w'), indent=1)


def compare(name, part, S, test_days, base_fit_days, quiet=False, save=True):
    """One fixed split. S = the candidate's cluster-pair scores for test_days; BASE = the live features with the head
    fitted on base_fit_days, on the same pairs."""
    B = base_scores(part, test_days, base_fit_days)
    only = _common(part, S, B, test_days)
    c, b = measure(part, S, test_days, only), measure(part, B, test_days, only)
    got, tot = _coverage(part, {d: {p: 0 for p in only[d]} for d in test_days}, test_days)
    out = dict(name=name, part=part, test_days=list(test_days), base_fit_days=list(base_fit_days), pairs_scored=got, pairs_total=tot, cand=c, base=b,
               flank=dict(cand=_flank_rows(part, S, test_days, only), base=_flank_rows(part, B, test_days, only)))
    if not quiet:
        print(f'== {name} | {part} | {len(test_days)} test days, BASE fitted on {len(base_fit_days)} days | pairs scored {got} of {tot}')
        print(f'   candidate  {_row(c)}\n   BASE       {_row(b)}')
        for r in sorted(out['flank']['cand']):
            print(f'      {r:<16s} candidate {_row(out["flank"]["cand"][r])}\n      {"":<16s} BASE      {_row(out["flank"]["base"][r])}')
    if save:
        _save(name, part, out)
    return out


def evaluate_pairs(name, part, S, reps=REPS):
    """Standard protocol for a scorer that was NOT fitted on these days: S = {day: {(a, b): score}} for train + val days."""
    cand, base = [], []
    for hold, pool in splits(reps):
        r = compare(name, part, S, VAL + hold, pool[:SIZES[-1]], quiet=True, save=False)
        cand.append(r['cand'])
        base.append(r['base'])
    v = compare(name, part, S, VAL, TRAIN, quiet=True, save=False)
    got, tot = _coverage(part, S, TRAIN + VAL)
    print(f'== {name} | {part} | pair scorer, no fitting | pairs scored {got} of {tot} on the 24 days')
    print(_line('candidate, val + set-aside days', cand))
    print(_line(f'BASE ({SIZES[-1]} days), same pairs', base))
    print(_delta('candidate - BASE', cand, base))
    print(f'   val days: candidate {_row(v["cand"])}\n             BASE (21)  {_row(v["base"])}')
    for r in sorted(v['flank']['cand']):
        print(f'      val, {r:<16s} candidate {_row(v["flank"]["cand"][r])} | BASE AUC {v["flank"]["base"][r]["auc"]:.3f}')
    out = dict(name=name, part=part, kind='pairs', pairs_scored=got, pairs_total=tot, held=dict(cand=cand, base=base), val=v)
    _save(name, part, out)
    return out


def evaluate(name, part, X, reps=REPS, sizes=SIZES, center=True):
    """Standard protocol for label-free features X = {day: {instance key: vector}} (train + val days)."""
    days = TRAIN + VAL
    st = {d: _struct(d, part, X.get(d, {}), center) for d in days}
    keys = {d: set(st[d]['keys']) for d in days}
    bs = {d: _struct(d, part, {k: v for k, v in _base(d).items() if k in keys[d]}) for d in days}
    n_inst, n_all = sum(len(k) for k in keys.values()), sum(len(instances(d, part)) for d in days)
    res = collections.defaultdict(list)
    for hold, pool in splits(reps):
        test = VAL + hold
        res['raw'].append(measure(part, {d: _pair_scores(d, part, st[d], None) for d in test}, test))
        for size in sizes:
            Pc, Pb = _fit([st[d] for d in pool[:size]]), _fit([bs[d] for d in pool[:size]])
            res[('cand', size)].append(measure(part, {d: _pair_scores(d, part, st[d], Pc) for d in test}, test))
            res[('base', size)].append(measure(part, {d: _pair_scores(d, part, bs[d], Pb) for d in test}, test))
    Pc, Pb = _fit([st[d] for d in TRAIN]), _fit([bs[d] for d in TRAIN])
    Sv, Bv = {d: _pair_scores(d, part, st[d], Pc) for d in VAL}, {d: _pair_scores(d, part, bs[d], Pb) for d in VAL}
    only = _common(part, Sv, Bv, VAL)
    vc, vb = measure(part, Sv, VAL), measure(part, Bv, VAL)
    fl = dict(cand=_flank_rows(part, Sv, VAL, only), base=_flank_rows(part, Bv, VAL, only))
    print(f'== {name} | {part} | features of {n_inst} of {n_all} instances that show the part | dim {max(s["X"].shape[1] for s in st.values())}')
    print(_line('raw cosine, no fit', res['raw']))
    for size in sizes:
        print(_line(f'candidate, head on {size} days', res[('cand', size)]))
        print(_line(f'BASE,      head on {size} days', res[('base', size)]))
        print(_delta('   candidate - BASE', res[('cand', size)], res[('base', size)]))
    for a, b in zip(sizes[:-1], sizes[1:]):
        print(_delta(f'   candidate, {a} -> {b} days', res[('cand', b)], res[('cand', a)]))
    print(f'   val days, head on all 21: candidate {_row(vc)}\n                             BASE      {_row(vb)}')
    for r in sorted(fl['cand']):
        print(f'      val, {r:<16s} candidate AUC {fl["cand"][r]["auc"]:.3f}  found at P 0.90 {100 * fl["cand"][r]["r90"]:4.1f} % | BASE AUC {fl["base"][r]["auc"]:.3f}  ({fl["cand"][r]["same"]} same / {fl["cand"][r]["diff"]} different)')
    out = dict(name=name, part=part, kind='features', instances=n_inst, instances_total=n_all, held={str(k): v for k, v in res.items()}, val=dict(cand=vc, base=vb, flank=fl))
    _save(name, part, out)
    return out


if __name__ == '__main__':                       # self-check: the live features against themselves, and without labels
    for part in PARTS:
        evaluate('base_selfcheck', part, {d: _base(d) for d in TRAIN + VAL}, reps=int(sys.argv[1]) if len(sys.argv) > 1 else REPS)
