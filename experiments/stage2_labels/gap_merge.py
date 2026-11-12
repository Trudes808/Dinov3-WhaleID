"""Val days, non-whales left out: merging cluster pairs in order of the time gap alone (shortest first), with the tie
order randomised (the result of band.py depends on the order of equal gaps), and a plain threshold on the gap:
join every pair up to T seconds apart. Catalog truth by cluster majority. Read-only."""
import pickle, json, collections
import numpy as np
from nowhale import NOT_WHALE, DAYS, REF, VAL
def major(c):
    e = sorted((c.get('catalog') or {}).items(), key=lambda kv: -kv[1]); return e[0][0] if e else None
def find(par, x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
def load(day):
    R = pickle.load(open(f'transfer_raw_{day}.pkl', 'rb')); ex = NOT_WHALE[day]; z = json.load(open(f'{DAYS}/{day}/safe/clusters.json'))
    C = {c['id']: c for c in z['clusters'] if c['id'] not in ex}; maj = {k: major(c) for k, c in C.items()}; ids = [k for k in maj if maj[k]]
    g = np.asarray(R['gap'], float); c = np.asarray(R['cat']); keep = np.array([a not in ex and b not in ex for a, b in R['pairs']]) & (c != '?')
    return [p for p, k in zip(R['pairs'], keep) if k], g[keep], maj, ids
def ordered(pairs, g, maj, ids, order, stop=6):
    par = {k: k for k in ids}; good = wrong = 0; marks = {}
    for q in order:
        a, b = pairs[q]; fa, fb = find(par, a), find(par, b)
        if fa == fb: continue
        if maj[fa] == maj[fb]: good += 1
        else:
            wrong += 1
            if wrong in (1, 3, 6): marks[wrong] = (good, float(g[q]))
            if wrong >= stop: break
        par[fa] = fb
    for w in (1, 3, 6): marks.setdefault(w, (good, float('nan')))
    return marks
def threshold(pairs, g, maj, ids, t):
    par = {k: k for k in ids}; good = wrong = 0
    for q in np.argsort(g, kind='stable'):
        if g[q] > t: break
        a, b = pairs[q]; fa, fb = find(par, a), find(par, b)
        if fa == fb: continue
        if maj[fa] == maj[fb]: good += 1
        else: wrong += 1
        par[fa] = fb
    comps = collections.defaultdict(collections.Counter)
    for k in ids: comps[find(par, k)][maj[k]] += 1
    per = collections.defaultdict(int)
    for cnt in comps.values():
        for a in cnt: per[a] += 1
    return good, wrong, len(comps), sum(len(c) > 1 for c in comps.values()), sum(per.values()), len(per)
rng = np.random.default_rng(0)
for days, name in ((VAL, 'val days together'), ([REF], 'labelled day 2025-01-02')):
    D = {d: load(d) for d in days}; res = collections.defaultdict(list)
    for s in range(200):
        tot = {1: 0, 3: 0, 6: 0}
        for d in days:
            pairs, g, maj, ids = D[d]; m = ordered(pairs, g, maj, ids, np.lexsort((rng.random(len(g)), g)))
            for w in tot: tot[w] += m[w][0]
        for w in tot: res[w].append(tot[w])
    print(f"{name}: time gap alone, 200 random tie orders: correct merges before the 1st wrong merge median {int(np.median(res[1]))} (range {min(res[1])}-{max(res[1])}), before the 3rd {int(np.median(res[3]))} ({min(res[3])}-{max(res[3])}), before the 6th {int(np.median(res[6]))} ({min(res[6])}-{max(res[6])})")
    for d in days:
        pairs, g, maj, ids = D[d]; m = ordered(pairs, g, maj, ids, np.argsort(g, kind='stable'))
        print(f"   {d}: gap of the 1st / 3rd / 6th wrong merge {m[1][1]:.2f} / {m[3][1]:.2f} / {m[6][1]:.2f} s")
    for t in (0.5, 1.0, 1.5, 3.0, 5.0, 15.0, 60.0):
        r = [threshold(*D[d], t) for d in days]; n = sum(len(D[d][3]) for d in days); an = sum(x[5] for x in r)
        print(f"   join every pair up to {t:4.1f} s apart: correct merges {sum(x[0] for x in r):3d}, wrong merges {sum(x[1] for x in r):3d}, piles {sum(x[2] for x in r)} of {n} clusters, mixed piles {sum(x[3] for x in r)}, pieces per animal {sum(x[4] for x in r) / an:.2f}")
