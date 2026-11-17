"""Read-only: a day sorted on the Labeling Assistant against the catalogue. python analyze.py DAY [--fetch]
Fetches the day, its log and the export from the running page (GET only) into ./DAY/ and prints the comparison."""
import collections, itertools, json, re, sys, time, urllib.request
from pathlib import Path
day = sys.argv[1]; out = Path(day); out.mkdir(exist_ok=True)
if '--fetch' in sys.argv or not (out / 'day.json').exists():
    for name, url in (('day.json', ''), ('log.json', '/log'), ('export.json', '/export')):
        (out / name).write_bytes(urllib.request.urlopen(f'http://localhost:8765/api/labeling/{day}{url}', timeout=300).read())
D = json.load(open(out / 'day.json')); L = json.load(open(out / 'log.json'))['log']; E = json.load(open(out / 'export.json')); S = D['state']; P = D['photos']; ST = D['stacks']; rows = E['rows']
sh = lambda x: (x or '-').split('_')[-1]; cat = [sh(p.get('cat')) for p in P]
unit = lambda c: 'O' if c[0] == 'O' else c[:3] if c[0] == 'G' else None
mine = [(sh(r['id']) if r['id'] else ('x' if r['rejected'] else 'open')) for r in rows]
nameof = {r['aid']: sh(r['id']) for r in rows if r['aid'] and r['aid'] != 'x' and r['id']}
into = {e['from']: e['animal'] for e in L if e.get('action') == 'merge'}
def fin(a):
    seen = set()
    while a in into and a not in seen: seen.add(a); a = into[a]
    return a
nm = lambda a: nameof.get(fin(a), f'({a})') if a else '-'
tab = collections.Counter(zip(mine, cat)); M = sorted(set(mine)); C = sorted(set(cat), key=lambda c: (c == '-', c))
R = {'day': day, 'photos': len(P), 'stacks': sum(1 for s in ST if s['kind'] != 'extra'), 'cat_animals': len([c for c in C if c != '-']), 'my_animals': len([m for m in M if m not in ('x', 'open')]), 'x': mine.count('x'), 'open': mine.count('open')}
print(f"=== {day}: {len(P)} photos, {R['stacks']} stacks | catalogue {R['cat_animals']} animals {dict(D.get('cat_groups') or {})} | mine {R['my_animals']} animals, {R['x']} no whale, {R['open']} open")
maj = {m: max((c for c in C if c != '-'), key=lambda c: tab[(m, c)]) for m in M if m not in ('x', 'open')}
pure = sum(tab[(m, maj[m])] for m in maj); n = sum(tab[(m, c)] for m in maj for c in C if c != '-')
idx = [i for i in range(len(P)) if cat[i] != '-' and mine[i] not in ('x', 'open')]; tp = fp = fn = 0
for i, j in itertools.combinations(idx, 2):
    a, b = mine[i] == mine[j], cat[i] == cat[j]; tp += a and b; fp += a and not b; fn += b and not a
R.update(pure=pure, of=n, prec=tp / max(1, tp + fp), rec=tp / max(1, tp + fn))
print(f'photos in an animal whose majority catalogue animal is theirs: {pure}/{n} = {pure / n:.3f} | photo pairs precision {R["prec"]:.3f} recall {R["rec"]:.3f}')
print('\nCATALOGUE ANIMAL -> MY ANIMALS (pieces)')
pieces = {}
for c in C:
    if c == '-': continue
    parts = sorted(((tab[(m, c)], m) for m in M if tab[(m, c)]), reverse=True); pieces[c] = parts
    own = [m for x, m in parts if m not in ('x', 'open') and maj.get(m) == c]
    print(f"  {c:7s} {sum(x for x, _ in parts):3d} ph -> " + ', '.join(f'{m} {x}' for x, m in parts) + (f'   [{len(own)} animals of mine]' if len(own) != 1 else ''))
R['split_whales'] = sum(1 for c in pieces if sum(1 for x, m in pieces[c] if m not in ('x', 'open') and maj.get(m) == c) > 1); R['extra_animals'] = sum(max(0, sum(1 for x, m in pieces[c] if m not in ('x', 'open') and maj.get(m) == c) - 1) for c in pieces)
R['not_found'] = [c for c in pieces if not any(maj.get(m) == c for x, m in pieces[c])]
mixed = {m: sorted(((tab[(m, c)], c) for c in C if tab[(m, c)] and c != '-'), reverse=True) for m in maj}; mixed = {m: v for m, v in mixed.items() if len(v) > 1}
print('mixed animals of mine:', {m: [f'{c} {x}' for x, c in v] for m, v in mixed.items()}); R['wrong_photos'] = n - pure
print('catalogue animals with no animal of mine:', R['not_found'], '| split whales:', R['split_whales'], '| extra animals from splits:', R['extra_animals'])
# encounters: my segments against the catalogue's units
B = {b['at']: b for b in D['boundaries']}; m_ = D['model']['boundary']; pidx = {p['stem']: i for i, p in enumerate(P)}
starts = sorted({0} | ({pidx[s] for s in S['cuts'] if s in pidx} if S.get('cuts') is not None else {b['at'] for b in D['boundaries'] if b['p'] >= m_['propose']}))
seg = [0] * len(P)
for k, a in enumerate(starts):
    for i in range(a, starts[k + 1] if k + 1 < len(starts) else len(P)): seg[i] = k
segmeta = S.get('segments') or {}
print(f"\nENCOUNTERS: mine {len(starts)} (cuts {'set by me' if S.get('cuts') is not None else 'the model proposal, untouched'}; types {[segmeta.get(P[a]['stem'], {}).get('type', '') for a in starts]})")
cu = [unit(c) if c != '-' else None for c in cat]
for k, a in enumerate(starts):
    b = (starts[k + 1] if k + 1 < len(starts) else len(P)) - 1; us = collections.Counter(c_ if not c_.startswith('O') else c_ for c_ in (cat[i] if cat[i][0] == 'O' else cat[i][:3] for i in range(a, b + 1) if cat[i] != '-'))
    print(f"  seg {k + 1:2d} {P[a]['clock'][:5]}-{P[b]['clock'][:5]} {b - a + 1:3d} ph: catalogue {dict(us)}")
gsegs = collections.defaultdict(set)
for i in range(len(P)):
    if cu[i] and cu[i] != 'O': gsegs[cu[i]].add(seg[i])
R['groups_cut_in_two'] = {g: len(v) for g, v in gsegs.items() if len(v) > 1}; multi = collections.defaultdict(set)
for i in range(len(P)):
    if cu[i] and cu[i] != 'O': multi[seg[i]].add(cu[i])
R['segs_with_two_groups'] = sum(1 for v in multi.values() if len(v) > 1)
print('catalogue groups lying in more than one of my encounters:', R['groups_cut_in_two'], '| my encounters holding two catalogue groups:', R['segs_with_two_groups'])
def change(at):
    l = next((cu[i] for i in range(at - 1, -1, -1) if cu[i]), None); r = next((cu[i] for i in range(at, len(P)) if cu[i]), None); return bool(l and r and l != r)
rv = [e for e in L if e.get('action') == 'cuts_review']
if rv:
    e = rv[-1]; print(f"gap question: {e.get('possible')} asked, {e.get('cut')} cut; cut and truly a change: {[ (P[pidx[s]]['clock'][:5], change(pidx[s])) for s in e.get('at', [])]}; kept and truly a change: {[(P[pidx[s]]['clock'][:5], change(pidx[s])) for s in e.get('same', [])]}")
print('cuts I have:', [(P[a]['clock'][:5], 'change' if change(a) else 'NOT a change', f"P {B[a]['p']:.2f}" if a in B else 'inside a sighting') for a in starts[1:]])
print('catalogue changes I did not cut:', [(P[b['at']]['clock'][:5], f"P {b['p']:.2f}") for b in D['boundaries'] if change(b['at']) and b['at'] not in starts])
# decisions
smaj = {}
for s in ST:
    c = collections.Counter(cat[p] for p, k in s['items'] if k <= 0 and cat[p] != '-'); smaj[s['id']] = c.most_common(1)[0][0] if c else '-'
clock = lambda s: P[ST[s]['items'][0][0]]['clock'] if s < len(ST) else '?'
acts = collections.Counter(e.get('action') for e in L); no = [e for e in L if e.get('action') == 'no' and not e.get('compare')]
t = [e['t'] for e in L]; act2 = sum(min(b - a, 120) for a, b in zip(t, t[1:])) / 60; act5 = sum(min(b - a, 300) for a, b in zip(t, t[1:])) / 60
R.update(log=len(L), no=len(no), cant=sum(1 for e in no if e.get('unsure')), yes=acts['assign'], newA=acts['new'], made=sum(1 for e in no if e.get('animal')), joins=sum(1 for e in L if e.get('joined')), take_out=acts['take_out'], merges=acts['merge'], undo=acts['undo'], photo_out=acts['photo_out'], minutes=round(act2))
print(f"\nLOG {len(L)}: {dict(acts)}")
print(f"no {len(no)} (can't tell {R['cant']}) on {len({e['head'] for e in no})} stacks; yes {acts['assign']}; new by A {acts['new']} (auto first {sum(1 for e in L if e.get('action') == 'new' and e.get('auto'))}); new by running out of candidates {R['made']}; joins {R['joins']}; active min {act2:.0f} (2-min cap) / {act5:.0f} (5-min cap); sittings from {time.strftime('%m-%d %H:%M', time.localtime(t[0]))} to {time.strftime('%m-%d %H:%M', time.localtime(t[-1]))}")
ranks = collections.Counter(min(e.get('rank', 0), 4) for e in L if e.get('action') == 'assign'); print('yes by rank of the animal chosen (4 = 4th or lower):', dict(sorted(ranks.items())))
wn = [(n_, e) for n_, e in enumerate(L) if e.get('action') == 'no' and not e.get('compare') and e['head'] < len(ST) and maj.get(nm(e['refused'])) == smaj[e['head']] and smaj[e['head']] != '-']
print(f'"no" to the animal the catalogue says it is: {len(wn)} of {len(no)}'); R['wrong_no'] = len(wn)
for n_, e in wn: print(f"  #{n_} stack {e['head']} {clock(e['head'])} ({len(ST[e['head']]['items'])} ph, {smaj[e['head']]}) refused {nm(e['refused'])} sim {e.get('sim', e.get('score'))}{' (cant tell)' if e.get('unsure') else ''} mode {e.get('mode')}")
wy = []
for n_, e in enumerate(L):
    if e.get('action') == 'assign' and e.get('head') is not None and e['head'] < len(ST):
        tc = maj.get(nm(e['animal'])); bad = [x for x in e.get('stacks', []) if x < len(ST) and smaj[x] not in ('-', tc)]
        if bad and tc: wy.append((n_, e, bad, tc))
print(f'"yes" to an animal the catalogue says it is not: {len(wy)} of {acts["assign"]}'); R['wrong_yes'] = len(wy)
for n_, e, bad, tc in wy: print(f"  #{n_} head {e['head']} {clock(e['head'])} -> {nm(e['animal'])} ({tc}); other whales' stacks {[(x, smaj[x], clock(x)) for x in bad]} rank {e.get('rank')} sim {e.get('sim')} run {e.get('run')} joined {len(e.get('joined') or [])} via {e.get('via')}")
# new animals: was the whale already there?
first_seen = {}; founded = []
for n_, e in enumerate(L):
    a = e.get('action')
    if a in ('new', 'assign') or (a == 'no' and e.get('animal')):
        if e.get('head') is None or e['head'] >= len(ST): continue
        c = smaj[e['head']]; aid = e.get('animal')
        if (a == 'new' or (a == 'no' and e.get('animal'))) and c != '-':
            founded.append((n_, e, c, c in first_seen))
        if c != '-' and c not in first_seen: first_seen[c] = n_
miss = [f for f in founded if f[3]]; R['new_total'] = len(founded); R['new_already_there'] = len(miss)
print(f'\nnew animals made: {len(founded)}; of them for a whale that already had an animal at that moment: {len(miss)}')
for n_, e, c, _ in miss:
    how = 'A pressed' if e['action'] == 'new' else 'ran out of candidates'
    print(f"  #{n_} stack {e['head']} {clock(e['head'])} ({len(ST[e['head']]['items'])} ph, {c}) -> {nm(e['animal'])}: {how}; cands {e.get('cands')} top {nm(e.get('top'))} {e.get('top_score', '')} mode {e.get('mode')} auto {e.get('auto', False)}")
catof = {p['stem']: sh(p.get('cat')) for p in P}; ut = collections.Counter()
for e in L:
    for pid in e.get('unsure_add') or []:
        ka, kb = pid.split('|'); ut['one whale' if catof.get(ka.rsplit('~', 1)[0]) == catof.get(kb.rsplit('~', 1)[0]) else 'two whales'] += 1
print("can't tell answers against the catalogue:", dict(ut)); R['cant_truth'] = dict(ut)
print('stacks of the stack builder holding two catalogue animals:', [(s['id'], clock(s['id'])) for s in ST if len({cat[p] for p, k in s['items'] if k <= 0 and cat[p] != '-'}) > 1])
print('photos with 2+ whales:', sum(1 for p in P if sum(1 for b in p['inst'] if b) >= 2), '| animals with a role:', sum(1 for a in S['animals'].values() if a.get('role')), '| passers-by (O) of mine:', sum(1 for a in S['animals'].values() if a.get('other')))
json.dump({'R': R, 'mine': mine, 'cat': cat, 'maj': maj}, open(out / 'result.json', 'w'))
