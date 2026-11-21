"""Day-level test of TWO-SIDED fin evidence (2026-11-18, mine, CPU, read-only, no eval day).

The sketch_fin agent's fit B gives every bench fin pair (same catalogue group, more than 60 s apart) a log-odds of
"different" from the HW-b6 cosine plus how well each fin was seen, fitted with the pair's day held out
(moonshot_cache/sketch_fin/evaluate2_scores.pkl). Unlike a threshold on the cosine it can say "different".
Here: start from the safe clusters joined by perfect linking within 60 s (or as they are), then merge pieces inside a
catalogue group bottom-up on the log-odds of "same" between them, while the best pair of pieces is still positive.
Between two pieces the cluster-pair log-odds are combined by sum, by mean, or by sum / sqrt(n) (frames of one piece
are not independent evidence). "count" shifts the log-odds by the prior of a group with n animals (1 / n against the
pooled share of same pairs), n = the catalogue's count for the group, i.e. what a field-log count would give.
Compared with the positive-only rule (join every pair above the cosine threshold with pair precision 0.95 on the
other days).

    python signed_day.py
"""
import collections
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench  # noqa: E402
from size_levers import Day, components, group_of, line, score, total  # noqa: E402

ALL = bench.TRAIN + bench.VAL
S = pickle.load(open('/home/sat3737/NOAA_whales/moonshot_cache/sketch_fin/evaluate2_scores.pkl', 'rb'))
COMB = {'sum': lambda v: float(np.sum(v)), 'mean': lambda v: float(np.mean(v)), 'sum/sqrt(n)': lambda v: float(np.sum(v) / np.sqrt(len(v)))}


def start_pieces(D, w):
    pr = [(a, b) for a, b in itertools.combinations(D.ids, 2) if w and D.maj[a] == D.maj[b] and D.gap(a, b) <= w]
    return components(D.ids, pr)


def merge(D, pieces, W, comb, shift=None, stop=0.0):
    """W: {(a, b): log-odds of same}. Greedy: merge the two pieces of a group with the highest combined log-odds > stop."""
    by = collections.defaultdict(list)
    for p in pieces:
        by[group_of(D.maj[p[0]])].append(list(p))
    out = []
    for g, ps in by.items():
        off = shift(g) if shift else 0.0
        while len(ps) > 1:
            best, bi = stop, None
            for i, j in itertools.combinations(range(len(ps)), 2):
                v = [W[(a, b) if a < b else (b, a)] + off for a in ps[i] for b in ps[j] if ((a, b) if a < b else (b, a)) in W]
                if not v:
                    continue
                c = COMB[comb](v)
                if c > best:
                    best, bi = c, (i, j)
            if bi is None:
                break
            i, j = bi
            ps[i] += ps[j]
            del ps[j]
        out += ps
    return out


if __name__ == '__main__':
    Ds = {d: Day(d) for d in ALL}
    same_share = np.mean([s for d in ALL for a, b, s in bench.pairs(d, 'dorsal')])
    for name in ('A  LR HW', 'B  LR HW + viewing quality'):
        W = {d: {p: -v for p, v in S[name][d].items()} for d in ALL}
        print(f'\n#### evidence: {name} (log-odds of same = minus the fitted log-odds of different; day held out)')
        for start, w in (('perfect 60-s linking', 60), ('safe clusters as they are', None)):
            P0 = {d: start_pieces(Ds[d], w) for d in ALL}
            for tag, days in (('24 days', ALL), ('val', bench.VAL)):
                print(f'== start: {start}; {tag}')
                print(line('no fin merging', total([score(Ds[d], P0[d]) for d in days])))
                for comb in COMB:
                    rows = [score(Ds[d], merge(Ds[d], P0[d], W[d], comb)) for d in days]
                    print(line(f'   merge while log-odds > 0, {comb}', total(rows)))
                    rows = []
                    for d in days:
                        D = Ds[d]
                        n = collections.Counter(group_of(a) for a in D.animals)
                        sh = lambda g, n=n: float(np.log((1 / max(n[g], 1.0001)) / max(1e-6, 1 - 1 / max(n[g], 1.0001))) - np.log(same_share / (1 - same_share))) if n[g] > 1 else 6.0
                        rows.append(score(D, merge(D, P0[d], W[d], comb, shift=sh)))
                    print(line(f'   the same with the group count as prior, {comb}', total(rows)))
