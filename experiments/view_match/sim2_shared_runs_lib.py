import collections
import sim2
from sim import BIT
def sort_day(day, rng):
    out = {}
    for g, cs in day.scope.items():
        stacks = [i for c in cs for i in day.animals[c]]; ph = lambda i: 0 if day.bits[i] & BIT['fluke'] else 1 if day.bits[i] & BIT['fin'] else 2
        stacks.sort(key=lambda i: (ph(i), day.t0[i], i)); pieces = collections.defaultdict(list)
        for s in stacks:
            c = day.cat[s]; mine = [k for k in pieces if k[0] == c]; best, bk, run = -1, None, None
            for k in mine:
                if max(day.P[s, t] for t in pieces[k]) >= 230: run = k
                b = day.scores([s], pieces[k])['blend']
                if b > best: best, bk = b, k
            if run is not None: bk = run
            elif bk is None or rng.random() < sim2.P0 * min(max((sim2.HI - best) / (sim2.HI - sim2.LO), 0.0), 1.0): bk = (c, len(mine))
            pieces[bk].append(s)
        out[g] = dict(pieces)
    return out
