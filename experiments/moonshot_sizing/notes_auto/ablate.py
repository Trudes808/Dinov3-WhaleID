"""Ablation of the trip-log assignment (2026-11-19). CPU, no eval day.

Every row: placement by time windows (+ aliases), then the joint inference (infer.py) with the named evidence, then the
confidence model (fitted on train days other than the scored day). Temperatures tu = 1, tp = 0.5 were chosen on the train
days only (grid tu 0.5-1.5, tp 0.15-1: whale-right photos 53-58 % everywhere, so the choice hardly matters).
Aggregates go to stdout (-> ablate.log in this folder); per-day rows and the results to moonshot_cache/notes_auto/.

    python ablate.py
"""
import collections
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import data  # noqa: E402
import infer  # noqa: E402

BASE = dict(tu=1.0, tp=0.5, empty=3.0, count=True)
PAIRS = dict(p_link=1, p_burst=1, p_fin=1, p_fluke=1)
L_VIEWS = dict(views=1)
L_ROLE = dict(role=1)
L_SHAPE = dict(height=1, form=1)
L_FLUKE = dict(fluke=1)
L_PEC_SEX = dict(wp=1, sex=1)
EVERY = dict(**L_VIEWS, **L_ROLE, **L_SHAPE, **L_FLUKE, **L_PEC_SEX, **PAIRS, anchor=1)


def cum(*parts):
    out = dict(BASE)
    for p in parts:
        out.update(p)
    return out


ROWS = [
    ('A0 time windows only, first noted whale for all', dict(method='first')),
    ('A1 + count (no per-whale evidence)', cum()),
    ('A2 + noted views', cum(L_VIEWS)),
    ('A3 + roles (role reader)', cum(L_VIEWS, L_ROLE)),
    ('A4 + shape reader (height, form)', cum(L_VIEWS, L_ROLE, L_SHAPE)),
    ('A5 + fluke type reader', cum(L_VIEWS, L_ROLE, L_SHAPE, L_FLUKE)),
    ('A6 + white pectoral and sex readers', cum(L_VIEWS, L_ROLE, L_SHAPE, L_FLUKE, L_PEC_SEX)),
    ('A7 + fin, fluke matchers and soft linker', cum(L_VIEWS, L_ROLE, L_SHAPE, L_FLUKE, L_PEC_SEX, PAIRS)),
    ('A8 everything (+ frame anchors)', cum(EVERY)),
    ('B  everything, simple method (merge to count, match)', cum(EVERY, dict(method='simple'))),
    ('C1 views alone', cum(L_VIEWS)),
    ('C2 role reader alone', cum(L_ROLE)),
    ('C3 shape reader alone', cum(L_SHAPE)),
    ('C4 fluke type reader alone', cum(L_FLUKE)),
    ('C5 white pectoral reader alone', cum(dict(wp=1))),
    ('C6 sex reader alone', cum(dict(sex=1))),
    ('C7 matchers alone (pairs + count; labels arbitrary)', cum(PAIRS)),
    ('C8 everything but the readers (views + pairs)', cum(L_VIEWS, PAIRS, dict(anchor=1))),
    ('C9 everything without the count term', cum(EVERY, dict(count=False))),
    ('X1 ceiling: perfect short-gap linking + everything', cum(EVERY, dict(oracle_link=1))),
    ('X2 ceiling: perfect readers + views, no pair terms', cum(L_VIEWS, L_ROLE, L_SHAPE, L_FLUKE, L_PEC_SEX, dict(oracle_readers=1))),
    ('X3 ceiling: perfect readers + everything', cum(EVERY, dict(oracle_readers=1))),
    ('X4 ceiling: perfect readers + perfect linking + everything', cum(EVERY, dict(oracle_readers=1, oracle_link=1))),
]


def chance(ctx, res, days):
    """Expected photo share right if each placed cluster got a random noted whale of its group (1 / n)."""
    num = den = 0.0
    for d in days:
        rec = ctx.recs[d]
        for k, c in rec['clusters'].items():
            if c['truth'] is None or c['truth'][0] != 'G':
                continue
            den += c['size']
            g = res[d][k][0]
            if g == c['truth'][1]:
                num += c['size'] / max(1, len(rec['log']['whales'].get(g, {})))
    return num / den


def acc_days(ctx, res, days):
    """per day (right photos, group photos)"""
    out = {}
    for d in days:
        R = [r for r in infer.rows(ctx, res, [d]) if r[6] == 'G']
        out[d] = (sum(r[2] * r[3] for r in R), sum(r[2] for r in R))
    return out


def boot(a, b, days, n=2000, seed=0):
    """Bootstrap over days of the pooled photo-accuracy difference b - a: (mean, 2.5 %, 97.5 %)."""
    rng = np.random.default_rng(seed)
    A = np.array([a[d] for d in days], float)
    B = np.array([b[d] for d in days], float)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(days), len(days))
        v.append(B[i, 0].sum() / B[i, 1].sum() - A[i, 0].sum() / A[i, 1].sum())
    v = np.array(v)
    return float(B[:, 0].sum() / B[:, 1].sum() - A[:, 0].sum() / A[:, 1].sum()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def at_prec_clusters(ctx, res, days, target):
    allR = {d: infer.rows(ctx, res, [d]) for d in ctx.days}
    right = assigned = tot = 0
    for d in days:
        others = [o for o in ctx.days if o != d and o in data.bench.TRAIN]
        A = sorted([r for o in others for r in allR[o] if r[4]], key=lambda r: -r[5])
        ok = np.array([r[3] for r in A], float)
        prec = np.cumsum(ok) / np.arange(1, len(ok) + 1)
        good = np.nonzero(prec >= target)[0]
        thr = A[good[-1]][5] if len(good) else np.inf
        for r in allR[d]:
            tot += r[6] == 'G'
            if r[4] and r[5] >= thr:
                assigned += 1
                right += r[3]
    return right, assigned, tot


def main():
    ctx = infer.Ctx()
    TR = [d for d in ctx.days if d in data.bench.TRAIN]
    VAL = [d for d in ctx.days if d in data.bench.VAL]
    priv = open(data.CACHE / 'ablate_days.log', 'w')
    store, acc = {}, {}
    print(f'days with a log: {len(ctx.days)} ({len(TR)} train, {len(VAL)} val)')
    import place
    recs = [ctx.recs[d] for d in ctx.days]
    for nm, kw in (('time windows only', dict(cues=False, alias=False)), ('+ aliases (used below)', dict(cues=False, alias=True)), ('+ aliases + frame cues', dict(cues=True, alias=True))):
        pr = place.run(recs, **kw)
        print(f'placement, {nm}: M {sorted(collections.Counter(pr[d][0] for d in pr).items())}\n   all: {place.summary(recs, pr, ctx.days)}\n   val: {place.summary(recs, pr, VAL)}')
    for tag, days in (('all', ctx.days), ('val', VAL)):
        n = collections.Counter()
        for d in days:
            P = ctx.place[d][1]
            for k, c in ctx.recs[d]['clusters'].items():
                if c['truth'] is None or P[k]['g'] is None:
                    continue
                b = infer.pkind(P[k])
                n[(b, 'n')] += c['size']
                n[(b, 'ok')] += place.right_group(c, P[k]['g']) * c['size']
        print(f'   {tag}: group right by how the cluster was placed (photos): ' + ', '.join(f"{b} {n[(b, 'ok')]}/{n[(b, 'n')]} = {100 * n[(b, 'ok')] / n[(b, 'n')]:.1f} %" for b in sorted({x[0] for x in n})))
    for name, cfg in ROWS:
        res = infer.calibrate(ctx, infer.run(ctx, cfg))
        store[name] = res
        acc[name] = acc_days(ctx, res, ctx.days)
        for tag, days in (('all', ctx.days), ('val', VAL)):
            s = infer.summary(ctx, res, days)
            print(infer.fmt(f'{name[:40]} [{tag}]', s), flush=True)
        for d in ctx.days:
            print(infer.fmt(f'{name[:40]} [{d}]', infer.summary(ctx, res, [d])), file=priv, flush=True)
    print(f'\nchance (a random noted whale of the placed group): all {100 * chance(ctx, store[ROWS[1][0]], ctx.days):.1f} %, val {100 * chance(ctx, store[ROWS[1][0]], VAL):.1f} % of group photos')

    print('\n== layer differences in whale-right photos, bootstrap over the 23 days (mean, 95 % interval, points) ==')
    seq = [r[0] for r in ROWS[:9]] + [ROWS[9][0]]
    for a, b in list(zip(seq[1:9], seq[2:9])) + [(seq[1], seq[8]), (seq[8], seq[9]), (ROWS[17][0], seq[8]), (ROWS[18][0], seq[8]),
                                                 (seq[8], ROWS[19][0]), (seq[8], ROWS[21][0]), (ROWS[21][0], ROWS[20][0])]:
        m, lo, hi = boot(acc[a], acc[b], ctx.days)
        print(f'   {b[:44]:<44s} - {a[:30]:<30s} {100 * m:+5.1f} ({100 * lo:+5.1f} .. {100 * hi:+5.1f})')
    for name in [r[0] for r in ROWS[10:17]]:
        m, lo, hi = boot(acc[ROWS[1][0]], acc[name], ctx.days)
        print(f'   {name[:44]:<44s} - A1 (count only)               {100 * m:+5.1f} ({100 * lo:+5.1f} .. {100 * hi:+5.1f})')

    best = ROWS[8][0]
    res = store[best]
    print(f'\n== precision / coverage, {best} ==')
    for tag, days in (('all days', ctx.days), ('val days', VAL), ('train days', TR)):
        R = infer.rows(ctx, res, days)
        cp, cc = infer.pr_curve(R), infer.pr_curve(R, weight='clusters')
        p95, p98 = infer.at_precision(ctx, res, days, 0.95), infer.at_precision(ctx, res, days, 0.98)
        c95, c98 = at_prec_clusters(ctx, res, days, 0.95), at_prec_clusters(ctx, res, days, 0.98)
        print(f'   {tag}: precision of the most confident 25 / 50 / 75 / 100 % of group photos: ' + ' / '.join(f'{100 * cp[f][0]:.1f}' for f in cp)
              + ' | of group clusters: ' + ' / '.join(f'{100 * cc[f][0]:.1f}' for f in cc))
        for t, p, c in ((0.95, p95, c95), (0.98, p98, c98)):
            print(f'      threshold for precision {t} chosen on the other (train) days: photos assigned {p[1]} of {p[2]} group photos, right {p[0]} '
                  f'({100 * p[0] / p[2]:.1f} % of group photos sorted, realised precision {100 * p[0] / max(1, p[1]):.1f} %) | clusters assigned {c[1]}, right {c[0]} of {c[2]} '
                  f'({100 * c[0] / c[2]:.1f} %, precision {100 * c[0] / max(1, c[1]):.1f} %)')

    print(f'\n== by noted group type and size, {best}, all days (whale-right photos) ==')
    R = [r for r in infer.rows(ctx, res, ctx.days) if r[6] == 'G']
    for key, f in (('type', lambda r: short(r[7])), ('noted whales', lambda r: '1' if r[8] == 1 else '2' if r[8] == 2 else '3' if r[8] == 3 else '4-6' if r[8] <= 6 else '7+')):
        by = collections.defaultdict(lambda: [0, 0, 0])
        for r in R:
            b = by[f(r)]
            b[0] += r[2] * r[3]
            b[1] += r[2]
            b[2] += r[2] / max(1, r[8])
        print(f'   by {key}: ' + ', '.join(f'{k} {v[0]}/{v[1]} = {100 * v[0] / v[1]:.0f} % (chance {100 * v[2] / v[1]:.0f})' for k, v in sorted(by.items(), key=str)))
    pickle.dump(dict(store=store, acc=acc), open(data.CACHE / 'ablate_results.pkl', 'wb'))


def short(t):
    if not t:
        return 'unknown'
    t = t.lower()
    for k, v in (('competitive', 'CG'), ('mce', 'MCE'), ('mye', 'MCE'), ('mc ', 'MC'), ('mc -', 'MC'), ('pair', 'Pair'), ('single', 'Single'), ('singer', 'Singer'), ('trio', 'Trio'), ('pod', 'Pod')):
        if k in t:
            return v
    return 'other'


if __name__ == '__main__':
    main()
