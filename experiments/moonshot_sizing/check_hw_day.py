"""What the Happywhale-recipe fin embedding buys at the level of a day (2026-11-18, mine, CPU, read-only, no eval day).

Start: the safe clusters joined by perfect linking within 60 s (catalogue oracle), or as they are. Inside each
catalogue group the pieces that show a fin are merged bottom-up on the fin score (cluster pair = best cosine over
instances; piece pair = single, average or complete linkage over their cluster pairs). Two stopping rules:
  count      until the group's fin-bearing pieces are as many as its animals that show a fin (the best a count from
             the field log could do; pieces without a fin stay as they are)
  threshold  until the best piece pair is below t; t for a day is the value that gives the most whole + clean animals
             on the OTHER days (leave one day out)
Score as size_levers: animals whole + clean. Embeddings: HW-b6 (moonshot_cache/public_reid) and the live fin match.

    python check_hw_day.py
"""
import collections
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench  # noqa: E402
from check_hw import ALL, hw_feats, live_feats  # noqa: E402
from size_levers import Day, components, group_of, line, score, total  # noqa: E402

LINK = {'single': np.max, 'average': np.mean, 'complete': np.min}
GRID = np.round(np.arange(0.20, 0.96, 0.025), 3)


def start_pieces(D, w):
    pr = [(a, b) for a, b in itertools.combinations(D.ids, 2) if w and D.maj[a] == D.maj[b] and D.gap(a, b) <= w]
    return components(D.ids, pr)


def cluster_scores(day, F):
    cl = {c: [k for k in ks if k in F] for c, ks in bench.clusters(day, 'dorsal').items()}
    M = {c: np.stack([F[k] for k in ks]) for c, ks in cl.items() if ks}
    return M, {}


def merge(D, pieces, M, how, stop, cache):
    """pieces: lists of cluster ids (labelled). Returns the merged pieces. stop = ('count', None) or ('thr', t)."""
    by_group = collections.defaultdict(list)
    out = []
    for p in pieces:
        fin = [c for c in p if c in M]
        if fin:
            by_group[group_of(D.maj[p[0]])].append(p)      # a start piece is pure in both starts, so p[0] names its group
        else:
            out.append(p)

    def cs(a, b):
        k = (a, b) if a < b else (b, a)
        if k not in cache:
            cache[k] = float((M[a] @ M[b].T).max())
        return cache[k]

    for g, ps in by_group.items():
        ps = [list(p) for p in ps]
        target = len({D.maj[c] for p in ps for c in p if c in M}) if stop[0] == 'count' else 1
        while len(ps) > target:
            best, bi = -9, None
            for i, j in itertools.combinations(range(len(ps)), 2):
                v = LINK[how]([cs(a, b) for a in ps[i] if a in M for b in ps[j] if b in M])
                if v > best:
                    best, bi = v, (i, j)
            if stop[0] == 'thr' and best < stop[1]:
                break
            i, j = bi
            ps[i] = ps[i] + ps[j]
            del ps[j]
        out += ps
    return out


if __name__ == '__main__':
    Ds = {d: Day(d) for d in ALL}
    emb = {'HW-b6': {d: hw_feats(d) for d in ALL}, 'live fin match': {d: live_feats(d) for d in ALL}}
    for start, w in (('perfect 60-s linking', 60), ('safe clusters as they are', None)):
        P0 = {d: start_pieces(Ds[d], w) for d in ALL}
        for tag, days in (('24 days', ALL), ('val', bench.VAL)):
            print(f'\n== start: {start}; {tag} ==')
            print(line('no fin merging', total([score(Ds[d], P0[d]) for d in days])))
            # the oracle for this start: every same-animal pair of fin-bearing pieces joined
            rows = []
            for d in days:
                D = Ds[d]
                M, _ = cluster_scores(d, emb['HW-b6'][d])
                pr = [(a, b) for a, b in itertools.combinations([c for c in D.ids if c in M], 2) if D.maj[a] == D.maj[b]]
                rows.append(score(D, components(D.ids, pr + [(p[0], c) for p in P0[d] for c in p[1:]])))
            print(line('perfect fin matching (any flank) on top', total(rows)))
            for name in emb:
                Ms = {d: cluster_scores(d, emb[name][d])[0] for d in ALL}
                caches = {d: {} for d in ALL}
                for how in LINK:
                    rows = [score(Ds[d], merge(Ds[d], P0[d], Ms[d], how, ('count', None), caches[d])) for d in days]
                    print(line(f'   {name}, {how} linkage, count given', total(rows)))
                for how in LINK:
                    table = {t: {d: score(Ds[d], merge(Ds[d], P0[d], Ms[d], how, ('thr', t), caches[d])) for d in ALL} for t in GRID}
                    rows, ts = [], []
                    for d in days:
                        t = max(GRID, key=lambda t: sum(table[t][x]['clean'] for x in ALL if x != d))
                        ts.append(t)
                        rows.append(table[t][d])
                    print(line(f'   {name}, {how} linkage, threshold {np.median(ts):.3f} (other days)', total(rows)))
