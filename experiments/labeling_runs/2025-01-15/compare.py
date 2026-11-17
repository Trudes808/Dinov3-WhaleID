"""Read-only: the researcher's Labeling Assistant result for 2025-01-15 (first pass, 2026-11-13 / 14) against the 2025 catalogue.

    python compare.py > compare.log

state_snapshot.json = data/2025-01-15/labeling.json as it stood when the day was finished (rev 222, with its log).
day_min.json = what the comparison needs of the day: per photo its catalogue id, per stack its photos, the export's
animal per photo, the model's group boundaries, and what Review showed at the end (hints, similarities of all pairs of
animals, the "could not tell" pairs; read from the page on a copy of the state). No photos here."""
import collections, itertools, json, time
D = json.load(open('day_min.json')); S = json.load(open('state_snapshot.json')); L = S['log']; P = D['photos']; ST = D['stacks']; rows = D['export']; H = D['review']
sh = lambda x: (x or '-').split('_')[-1]; cat = [sh(p['cat']) for p in P]
mine = [(sh(r['id']) if r['id'] else ('x' if r['rejected'] else 'open')) for r in rows]
nameof = {r['aid']: sh(r['id']) for r in rows if r['aid'] and r['aid'] != 'x' and r['id']}
into = {e['from']: e['animal'] for e in L if e.get('action') == 'merge'}
def fin(a):
    while a in into: a = into[a]
    return a
nm = lambda a: nameof.get(fin(a), f'({a})') if a else '-'
tab = collections.Counter(zip(mine, cat)); M = sorted(set(mine)); C = sorted(set(cat), key=lambda c: (c == '-', c))
print('PHOTOS: my animal (rows) x catalogue animal (columns)'); print(f"{'':8s}" + ''.join(f'{c:>7s}' for c in C) + '  total')
for m in M: print(f'{m:8s}' + ''.join(f"{tab[(m, c)] or '':>7}" for c in C) + f'  {sum(tab[(m, c)] for c in C):5d}')
maj = {m: max((c for c in C if c != '-'), key=lambda c: tab[(m, c)]) for m in M if m not in ('x', 'open')}
pure = sum(tab[(m, maj[m])] for m in maj); n = sum(tab[(m, c)] for m in maj for c in C if c != '-'); print(f'\nphotos in an animal whose majority catalogue animal is theirs: {pure} of {n} = {pure / n:.3f}')
idx = [i for i in range(len(P)) if cat[i] != '-' and mine[i] not in ('x', 'open')]; tp = fp = fn = 0
for i, j in itertools.combinations(idx, 2):
    a, b = mine[i] == mine[j], cat[i] == cat[j]; tp += a and b; fp += a and not b; fn += b and not a
print(f'photo pairs: together and one whale {tp}, together but two whales {fp}, apart but one whale {fn}: precision {tp / (tp + fp):.3f}, recall {tp / (tp + fn):.3f}')
print('\nCATALOGUE ANIMAL -> MY ANIMALS')
for c in C:
    if c != '-': print(f'  {c:7s} ' + ', '.join(f'{m} {x}' for x, m in sorted(((tab[(m, c)], m) for m in M if tab[(m, c)]), reverse=True)))
print('\nTHE DAY BY CATALOGUE UNIT, and the model\'s P(new group) there (proposed from', D['boundary']['propose'], ', "possible" from', D['boundary']['maybe'], ')')
B = {b['at']: b['p'] for b in D['boundaries']}; runs = []
for i in range(len(P)):
    u = cat[i] if cat[i][0] in 'O-' else cat[i][:3]
    if runs and runs[-1][0] == u: runs[-1][2] = i
    else: runs.append([u, i, i])
for u, a, b in runs: print(f"  {u:4s} {P[a]['clock']}-{P[b]['clock']} {b - a + 1:3d} photos, gap before {(P[a]['t'] - P[a - 1]['t']) / 60 if a else 0:5.1f} min, P(new group) {B.get(a, 0):.2f}")
smaj = {}
for s in ST:
    c = collections.Counter(cat[p] for p, k in s['items'] if k <= 0 and cat[p] != '-'); smaj[s['id']] = c.most_common(1)[0][0] if c else '-'
print('\nstacks of the stack builder holding two catalogue animals:', [s['id'] for s in ST if len({cat[p] for p, k in s['items'] if k <= 0 and cat[p] != '-'}) > 1])
clock = lambda s: P[ST[s]['items'][0][0]]['clock']; acts = collections.Counter(e.get('action') for e in L); print('\nLOG', len(L), dict(acts))
no = [e for e in L if e.get('action') == 'no' and not e.get('compare')]
print(f"no {len(no)} (of them can't tell {sum(1 for e in no if e.get('unsure'))}) on {len({e['head'] for e in no})} stacks; new animals by running out of candidates {sum(1 for e in no if e.get('animal'))}, by A {acts['new']}; decisions with a hand join {sum(1 for e in L if e.get('joined'))}")
print('"no" to the animal the catalogue says it is:')
for n_, e in enumerate(L):
    if e.get('action') == 'no' and not e.get('compare') and e['head'] < len(ST) and maj.get(nm(e['refused'])) == smaj[e['head']]: print(f"  #{n_} stack {e['head']} {clock(e['head'])} ({smaj[e['head']]}) refused {nm(e['refused'])} sim {e.get('sim')}{' (cant tell)' if e.get('unsure') else ''}")
print('"yes" to an animal the catalogue says it is not:')
for n_, e in enumerate(L):
    if e.get('action') == 'assign' and e.get('head') is not None and e['head'] < len(ST):
        tc = maj.get(nm(e['animal'])); bad = [x for x in e.get('stacks', []) if x < len(ST) and smaj[x] not in ('-', tc)]
        if bad and tc: print(f"  #{n_} head {e['head']} {clock(e['head'])} -> {nm(e['animal'])} ({tc}); of other whales {[(x, smaj[x]) for x in bad]}; rank {e.get('rank')} sim {e.get('sim')} run {e.get('run')} joined {len(e.get('joined') or [])} via {e.get('via')}")
catof = {p['stem']: sh(p['cat']) for p in P}; ut = collections.Counter()
for e in L:
    for pid in e.get('unsure_add') or []:
        ka, kb = pid.split('|'); ut['one whale' if catof.get(ka.rsplit('~', 1)[0]) == catof.get(kb.rsplit('~', 1)[0]) else 'two whales'] += 1
print("can't tell answers against the catalogue:", dict(ut))
true = lambda a, b: maj.get(sh(a)) == maj.get(sh(b))
print('\nREVIEW AT THE END: hints'); 
for x in H['hints']: print(f"  {sh(x['a'])} {sh(x['b'])} {x['level']:8s} {x['score']:.2f} -> {'ONE WHALE' if true(x['a'], x['b']) else 'two whales'}")
SIM = sorted(H['sims'], key=lambda r: -r['sim']); print('pairs of my animals that are one whale in the catalogue, the model\'s similarity and its rank among', len(SIM), 'pairs:')
for i, r in enumerate(SIM):
    if true(r['a'], r['b']): print(f"  {sh(r['a'])} {sh(r['b'])} ({maj[sh(r['a'])]}) {r['sim']:.2f} rank {i + 1} {r['parts']}")
print('  the most similar pairs that are two whales:', [(sh(r['a']), sh(r['b']), r['sim']) for r in SIM if not true(r['a'], r['b'])][:5])
t = [e['t'] for e in L]; print('\nactive minutes (gaps capped at 2 / 5 min):', round(sum(min(b - a, 120) for a, b in zip(t, t[1:])) / 60), '/', round(sum(min(b - a, 300) for a, b in zip(t, t[1:])) / 60))
