"""State of 2025-01-02 with ALL of the researcher's same links: the cluster-level links of the bursts page plus the
"same" verdicts of the sequential photo pairs (pairs page) that fall between two safe clusters. state.py / state2.py
used the cluster-level links only. Same definitions as state2.py (pieces per catalog animal = components holding at
least one cluster of the animal by cluster majority). Read-only."""
import json, collections, itertools
import numpy as np
day = '2025-01-02'
z = json.load(open(f'/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all/{day}/safe/clusters.json')); C = {c['id']: c for c in z['clusters']}
L = json.load(open(f'/home/sat3737/NOAA_whales/Whale-Review-Web/data/{day}/pair_marks.json'))['labels']
def major(c):
    e = sorted((c.get('catalog') or {}).items(), key=lambda kv: -kv[1]); return e[0][0] if e else None
maj = {k: major(c) for k, c in C.items()}; animals = sorted(set(v for v in maj.values() if v)); cat_ids = [k for k in C if maj[k]]
key2c = {key: k for k, c in C.items() for key in c['keys']}
def gap(a, b):
    A, B = C[a], C[b]; return max(0.0, max(A['start'], B['start']) - min(A['end'], B['end']))
def norm(a, b): return (min(a, b), max(a, b))
lab = {}; same_c = []; same_s = []
for r in L:
    if r.get('level') == 'cluster':
        lab[norm(r['set_a'], r['set_b'])] = r['verdict']
        if r['verdict'] == 'same': same_c.append(norm(r['set_a'], r['set_b']))
for r in L:
    if r.get('level') != 'cluster':
        a, b = key2c.get(r['key_a']), key2c.get(r['key_b'])
        if a is None or b is None or a == b: continue
        lab.setdefault(norm(a, b), r['verdict'])
        if r['verdict'] == 'same': same_s.append(norm(a, b))
def conflict(p): return maj[p[0]] is not None and maj[p[1]] is not None and maj[p[0]] != maj[p[1]]
def uf(pairs):
    par = {k: k for k in C}
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in pairs: par[find(a)] = find(b)
    comp = collections.defaultdict(list)
    for k in C: comp[find(k)].append(k)
    return list(comp.values())
def per_animal(comps):
    per = collections.defaultdict(set)
    for ci, cm in enumerate(comps):
        for k in cm:
            if maj[k]: per[maj[k]].add(ci)
    return float(np.mean([len(v) for v in per.values()])), per
print(f'same links: cluster level {len(same_c)}, sequential photo pairs between two clusters {len(same_s)}; links the catalog contradicts {sum(map(conflict, same_c))} + {sum(map(conflict, same_s))}')
print(f'sequential same links by gap: up to 1.5 s {sum(gap(*p) <= 1.5 for p in same_s)}, 1.5-60 s {sum(1.5 < gap(*p) <= 60 for p in same_s)}, beyond 60 s {sum(gap(*p) > 60 for p in same_s)}')
allp = [p for p in itertools.combinations(sorted(C), 2) if gap(*p) <= 900]
catsame = [p for p in allp if maj[p[0]] and maj[p[0]] == maj[p[1]]]
sug = {norm(s['a'], s['b']) for s in z['suggestions']}
for name, links in (('cluster-level same links', same_c), ('all same links', same_c + same_s), ('all same links without the ones the catalog contradicts', [p for p in same_c + same_s if not conflict(p)])):
    compS = uf(links); m, per = per_animal(compS); cid = {k: i for i, cm in enumerate(compS) for k in cm}
    print(f'\n== {name}: {len(links)} links, components {len(compS)} ({sum(len(c) > 1 for c in compS)} of more than one cluster), pieces per catalog animal {m:.2f} ==')
    rows = collections.Counter(); gaps = []
    for a, cs in per.items():
        cs = sorted(cs)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                A = [k for k in compS[cs[i]] if maj[k] == a]; B = [k for k in compS[cs[j]] if maj[k] == a]
                vs = [lab[norm(x, y)] for x in A for y in B if norm(x, y) in lab]
                rows['never shown' if not vs else 'judged different' if 'different' in vs else 'likely same' if 'likely_same' in vs else 'unsure' if 'unsure' in vs else 'no similarity'] += 1
                gaps.append(min(gap(x, y) for x in A for y in B))
    gaps = np.array(gaps)
    print(f'   catalog-same component pairs still apart {len(gaps)}: {dict(rows)}; closest approach up to 60 s {(gaps <= 60).sum()}, 60 s to 15 min {((gaps > 60) & (gaps <= 900)).sum()}, beyond 15 min {(gaps > 900).sum()}')
    print(f'   catalog-same cluster pairs within 15 min already inside one component: {sum(cid[a] == cid[b] for a, b in catsame)} of {len(catsame)}')
    rest = [p for p in sug if p not in lab]; rest_x = [p for p in rest if cid[p[0]] != cid[p[1]]]
    rs = [p for p in rest_x if maj[p[0]] and maj[p[0]] == maj[p[1]]]
    print(f'   open suggestions {len(rest)}, between different components {len(rest_x)}: catalog-same {len(rs)}, catalog-different {sum(map(conflict, rest_x))}')
    print(f'   links + every catalog-same open suggestion: pieces per animal {per_animal(uf(links + rs))[0]:.2f}')
    comp_pairs = collections.defaultdict(list)
    for a, b in allp:
        if cid[a] != cid[b]: comp_pairs[norm(cid[a], cid[b])].append((a, b))
    def cmaj(ci):
        c = collections.Counter(maj[k] for k in compS[ci] if maj[k]); return c.most_common(1)[0][0] if c else None
    same_cp = [p for p in comp_pairs if cmaj(p[0]) and cmaj(p[0]) == cmaj(p[1])]
    print(f'   component pairs within 15 min {len(comp_pairs)}, catalog-same (component majorities) {len(same_cp)}: with an open suggestion {sum(any(q in sug and q not in lab for q in comp_pairs[p]) for p in same_cp)},'
          f' with a labelled pair and no link {sum(any(q in lab for q in comp_pairs[p]) for p in same_cp)}, with neither {sum(not any(q in sug or q in lab for q in comp_pairs[p]) for p in same_cp)}')
    for w in (1.5, 60, 300, 900):
        orc = [p for p in itertools.combinations(cat_ids, 2) if maj[p[0]] == maj[p[1]] and gap(*p) <= w]
        print(f'   links + every catalog-same pair within {w:5.1f} s: pieces per animal {per_animal(uf(links + orc))[0]:.2f}')
    for lo, hi in ((-1, 1.5), (1.5, 60), (60, 900)):
        band = [p for p in rest_x if lo < gap(*p) <= hi]; bs = [p for p in band if maj[p[0]] and maj[p[0]] == maj[p[1]]]
        print(f'   open suggestions between different components with gap in ({lo}, {hi}] s: {len(band)} (catalog-same {len(bs)}, catalog-different {sum(map(conflict, band))}, unknown {len(band) - len(bs) - sum(map(conflict, band))});'
              f' links + the catalog-same ones of them: pieces per animal {per_animal(uf(links + bs))[0]:.2f}')
