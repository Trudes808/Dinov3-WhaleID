"""Transitive joining of the researcher's links on 2025-01-02 against the catalog.
Links: cluster-level "same" (bursts page) and, optionally, the "same" verdicts of the sequential photo pairs (pairs page)
that fall between two different safe clusters, and "likely_same". Reports pieces per catalog animal, mixed piles, the
share of catalog photos outside their pile's majority animal, and the effect of dropping the links that the catalog
contradicts. Read-only."""
import json, collections, sys
import numpy as np
DAY = sys.argv[1] if len(sys.argv) > 1 else '2025-01-02'
z = json.load(open(f'/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all/{DAY}/safe/clusters.json')); C = {c['id']: c for c in z['clusters']}
L = json.load(open(f'/home/sat3737/NOAA_whales/Whale-Review-Web/data/{DAY}/pair_marks.json'))['labels']
def major(c):
    cat = c.get('catalog')
    return max(cat.items(), key=lambda kv: kv[1])[0] if isinstance(cat, dict) and cat else None
maj = {k: major(c) for k, c in C.items()}; animals = sorted(set(v for v in maj.values() if v))
key2c = {}
for k, c in C.items():
    for key in c['keys']: key2c[key] = k
clu = [r for r in L if r.get('level') == 'cluster']; seq = [r for r in L if r.get('level') != 'cluster']
print(f'{DAY}: records {len(L)} (cluster level {len(clu)}, sequential photo pairs {len(seq)}); clusters {len(C)}, with a catalog animal {sum(v is not None for v in maj.values())}, animals {len(animals)}')
links = {'same': [], 'likely': [], 'seq': []}
for r in clu:
    if r['verdict'] == 'same': links['same'].append((r['set_a'], r['set_b'], r))
    if r['verdict'] == 'likely_same': links['likely'].append((r['set_a'], r['set_b'], r))
found = collections.Counter()
for r in seq:
    a, b = key2c.get(r['key_a']), key2c.get(r['key_b'])
    kind = 'instance not in a cluster' if a is None or b is None else ('inside one cluster' if a == b else 'between two clusters')
    found[(r['verdict'], kind)] += 1
    if r['verdict'] == 'same' and a is not None and b is not None and a != b: links['seq'].append((a, b, r))
print('sequential photo pairs by verdict and place:', dict(found))
have = {(min(a, b), max(a, b)) for a, b, _ in links['same']}
print(f"sequential same links between two clusters: {len(links['seq'])}, of which already a cluster-level same link: {sum((min(a, b), max(a, b)) in have for a, b, _ in links['seq'])}")
def conflict(a, b): return maj[a] is not None and maj[b] is not None and maj[a] != maj[b]
for nm in links:
    cf = [(a, b) for a, b, _ in links[nm] if conflict(a, b)]
    print(f"   {nm:7s} links {len(links[nm]):4d}, between clusters of different catalog animals {len(cf)}" + (f": {cf}" if nm == 'seq' else ''))

def comps(use, drop_conflicts=False):
    par = {k: k for k in C}
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for nm in use:
        for a, b, _ in links[nm]:
            if drop_conflicts and conflict(a, b): continue
            par[find(a)] = find(b)
    out = collections.defaultdict(list)
    for k in C: out[find(k)].append(k)
    return list(out.values())
def report(name, cs, show=6):
    per = collections.defaultdict(set); mixed = []; tot = 0; off = 0; inmixed = 0; cl_mixed = 0
    for i, cm in enumerate(cs):
        cnt = collections.Counter(maj[k] for k in cm if maj[k])          # clusters per majority animal
        ph = collections.Counter()
        for k in cm:
            for a, n in (C[k].get('catalog') or {}).items(): ph[a] += n   # catalog photos per animal
        for a in cnt: per[a].add(i)
        tot += sum(ph.values())
        if ph: off += sum(ph.values()) - max(ph.values())
        if len(cnt) > 1: mixed.append((len(cm), sum(len(C[k]['photos']) for k in cm), dict(cnt.most_common()), dict(ph.most_common()))); inmixed += sum(ph.values()); cl_mixed += len(cm)
    multi = [cm for cm in cs if len(cm) > 1]
    print(f"\n{name}: piles {len(cs)} ({len(multi)} of more than one cluster, largest {max(len(cm) for cm in cs)} clusters); pieces per catalog animal {np.mean([len(per[a]) for a in animals]):.2f}"
          f" (most pieces {sorted((len(per[a]) for a in animals), reverse=True)[:5]}); piles mixing catalog animals {len(mixed)}, holding {cl_mixed} clusters and {inmixed / tot:.2f} of the catalog photos;"
          f" catalog photos outside their pile's majority animal {off} of {tot} ({off / tot:.3f})")
    for n, p, cnt, ph in sorted(mixed, key=lambda t: -t[0])[:show]:
        print(f"      {n:3d} clusters, {p:3d} photos, {len(cnt)} animals | clusters per animal {({a.split('_', 1)[1]: v for a, v in cnt.items()})} | catalog photos {({a.split('_', 1)[1]: v for a, v in ph.items()})}")
    return per
report('safe clusters', comps([]))
report('cluster-level same links', comps(['same']))
report('cluster-level same + sequential same links', comps(['same', 'seq']))
report('cluster-level same links without the catalog conflicts', comps(['same'], True))
report('cluster-level same + sequential same without the catalog conflicts', comps(['same', 'seq'], True))
report('same + sequential + likely_same', comps(['same', 'seq', 'likely']))
report('same + sequential + likely_same without the catalog conflicts', comps(['same', 'seq', 'likely'], True))
