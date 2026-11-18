"""Independent check of the public_reid agent's main claim (2026-11-18): the Happywhale-recipe EfficientNet-B6 dorsal
embedding (cached by public_reid/extract.py, label-free, plain cosine, nothing fitted on our days) against the live
fin matcher. CPU only, read-only, train + val days, no eval day.

1  per day: AUC of same-group cluster pairs > 60 s apart (the bench's pairs), HW raw cosine against the live view-match
   embedding (fitted on the 21 train days, so in-sample on those) and against raw frozen view features
2  by flank relation, over all 24 days
3  by time gap between the two clusters
4  the day-level score of size_levers: animals whole + clean when cluster pairs of one catalogue group are joined
   automatically above a threshold. The threshold for a day is the one that gives the wanted pair precision on the
   OTHER days (leave one day out), so nothing is tuned on the day it is applied to.

    python check_hw.py [variant, default box]
"""
import collections
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench  # noqa: E402
from size_levers import DAYS, Day, auc, components, group_of, line, score, total  # noqa: E402

CACHE = Path('/home/sat3737/NOAA_whales/moonshot_cache/public_reid/feat')
VAR = sys.argv[1] if len(sys.argv) > 1 else 'box'
ALL = bench.TRAIN + bench.VAL


def unit(X):
    X = np.asarray(X, np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def hw_feats(day):
    z = np.load(CACHE / 'hw_b6' / VAR / f'{day}.npz')
    return dict(zip([str(k) for k in z['keys']], unit(z['X'])))


def live_feats(day):
    z = np.load(DAYS / day / 'safe' / 'view_match.npz')
    return dict(zip([str(k) for k in z['dorsal|keys']], unit(z['dorsal|E'])))


def raw_feats(day):
    z = np.load(DAYS / day / 'safe' / 'view_feat.npz')
    X = z['X'].astype(np.float32).reshape(len(z['X']), 4, 768)
    X = (X / np.maximum(np.linalg.norm(X, axis=2, keepdims=True), 1e-6)).reshape(len(X), -1)
    keys = [str(k) for k in z['keys']]
    use = {k for k, c, a in bench.instances(day, 'dorsal')}
    idx = [i for i, k in enumerate(keys) if k in use]
    Xc = X[idx] - X[idx].mean(0)
    return dict(zip([keys[i] for i in idx], unit(Xc)))


def pair_scores(day, F):
    """{(a, b): best cosine over the two clusters' dorsal instances} for the bench's pairs that F covers."""
    cl = {c: [k for k in ks if k in F] for c, ks in bench.clusters(day, 'dorsal').items()}
    M = {c: np.stack([F[k] for k in ks]) for c, ks in cl.items() if ks}
    return {(a, b): float((M[a] @ M[b].T).max()) for a, b, same in bench.pairs(day, 'dorsal') if a in M and b in M}


def at_prec(pos, neg, prec):
    """(threshold, same pairs found) at the lowest threshold whose precision is still >= prec."""
    v = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(-v, kind='stable')
    tp = np.cumsum(y[o])
    p = tp / np.arange(1, len(o) + 1)
    ok = np.nonzero(p >= prec)[0]
    if not len(ok):
        return np.inf, 0
    return float(v[o][ok[-1]]), int(tp[ok[-1]])


if __name__ == '__main__':
    Ds = {d: Day(d) for d in ALL}
    S = {'HW-b6 raw': {}, 'frozen raw': {}}
    S['live fin match'] = {}
    for d in ALL:
        S['HW-b6 raw'][d] = pair_scores(d, hw_feats(d))
        S['frozen raw'][d] = pair_scores(d, raw_feats(d))
        if 'live fin match' in S:
            S['live fin match'][d] = pair_scores(d, live_feats(d))
    same = {d: {(a, b): s for a, b, s in bench.pairs(d, 'dorsal')} for d in ALL}

    def split(name, days, only=None):
        pos, neg = [], []
        for d in days:
            for k, v in S[name][d].items():
                if only is None or only(d, k):
                    (pos if same[d][k] else neg).append(v)
        return np.array(pos), np.array(neg)

    print(f'== 1  per day, dorsal pairs of one group > 60 s apart; HW variant "{VAR}" ==')
    print(f'{"day":<12s} {"split":<6s} {"same":>6s} {"diff":>6s} ' + ' '.join(f'{n:>16s}' for n in S))
    wins = collections.Counter()
    for d in ALL:
        row, n = [], None
        for name in S:
            pos, neg = split(name, [d])
            n = (len(pos), len(neg))
            row.append(auc(pos, neg) if len(pos) and len(neg) else float('nan'))
        if n[0] and n[1]:
            for i, name in enumerate(list(S)[1:], 1):
                wins[name] += row[0] > row[i]
            wins['days'] += 1
        print(f'{d:<12s} {"val" if d in bench.VAL else "train":<6s} {n[0]:6d} {n[1]:6d} ' + ' '.join(f'{v:16.3f}' for v in row))
    print('HW ahead of', {k: f'{v} of {wins["days"]} days' for k, v in wins.items() if k != 'days'})
    for tag, days in (('all 24 days pooled', ALL), ('train 21 pooled', bench.TRAIN), ('val 3 pooled', bench.VAL)):
        out = []
        for name in S:
            pos, neg = split(name, days)
            out.append(f'{name} AUC {auc(pos, neg):.3f}, found at P 0.95 {at_prec(pos, neg, .95)[1]} / at 0.90 {at_prec(pos, neg, .90)[1]} of {len(pos)}')
        print(f'   {tag}: ' + ' | '.join(out))

    print('\n== 2  by flank relation, all 24 days pooled ==')

    def rel(d, k):
        fa, fb = bench.flank(d, k[0]), bench.flank(d, k[1])
        fa, fb = set(fa) if fa else set(), set(fb) if fb else set()
        if not fa or not fb:
            return 'flank unknown'
        if len(fa) > 1 or len(fb) > 1:
            return 'mixed'
        return 'same flank' if fa == fb else 'opposite flanks'
    for r in ('same flank', 'opposite flanks', 'flank unknown', 'mixed'):
        out = []
        for name in S:
            pos, neg = split(name, ALL, lambda d, k: rel(d, k) == r)
            out.append(f'{name} {auc(pos, neg):.3f}')
        print(f'   {r:<16s} {len(pos):5d} same / {len(neg):5d} diff   ' + ' | '.join(out))

    print('\n== 3  by time gap between the two clusters, all 24 days pooled ==')
    for lo, hi in ((60, 300), (300, 900), (900, 3600), (3600, 1e9)):
        out = []
        for name in S:
            pos, neg = split(name, ALL, lambda d, k: lo < Ds[d].gap(*k) <= hi)
            out.append(f'{name} {auc(pos, neg):.3f}')
        print(f'   {lo:>5.0f} - {hi:<8.0f} s {len(pos):5d} same / {len(neg):5d} diff   ' + ' | '.join(out))

    print('\n== 4  day-level list score: join same-group fin pairs above a leave-one-day-out threshold ==')
    print('   start = perfect linking within 60 s (oracle) or the safe clusters as they are; joins are automatic, wrong ones included')
    for start, w in (('safe clusters', None), ('perfect 60-s linking', 60)):
        for days, tag in ((ALL, '24 days'), (bench.VAL, 'val')):
            base_rows = []
            for d in days:
                D = Ds[d]
                pr = [(a, b) for i, a in enumerate(D.ids) for b in D.ids[i + 1:] if w and D.maj[a] == D.maj[b] and D.gap(a, b) <= w]
                base_rows.append(score(D, components(D.ids, pr)))
            print(line(f'{start}, {tag}: no fin joins', total(base_rows)))
            for name in S:
                for prec in (0.99, 0.95, 0.90):
                    rows, nj, nw = [], 0, 0
                    for d in days:
                        D = Ds[d]
                        pos, neg = split(name, [x for x in ALL if x != d])
                        t, _ = at_prec(pos, neg, prec)
                        pr = [(a, b) for i, a in enumerate(D.ids) for b in D.ids[i + 1:] if w and D.maj[a] == D.maj[b] and D.gap(a, b) <= w]
                        j = [k for k, v in S[name][d].items() if v >= t]
                        nj += len(j)
                        nw += sum(not same[d][k] for k in j)
                        rows.append(score(D, components(D.ids, pr + j)))
                    print(line(f'   + {name} joins at P {prec:.2f} ({nj} joins, {nw} wrong)', total(rows)))
