"""Read-only: the researcher's Photo Labeling result for 2024-12-23 against the catalog."""
import json, csv, collections, time
D = json.load(open('day.json')); S = json.load(open('state_snapshot.json')); P = D['photos']; assign = S['assign']
rows = list(csv.DictReader(open('export.csv'))); exp = {r['file'].rsplit('.', 1)[0]: r for r in rows}
sh = lambda x: (x or '?').split('_')[-1]
name = lambda p: p['stem'].split('__')[-1]
# stack of every instance; the photo's own animal = the assignment of its main instance
stack_of = {}
for s in D['stacks']:
    for pi, k in s['items']: stack_of[(pi, k)] = s
def key(pi, k): return f"{P[pi]['stem']}~{k if k >= 0 else 'x'}"
print('stack kinds:', collections.Counter(s['kind'] for s in D['stacks']), '| items with k<0:', sum(1 for s in D['stacks'] for pi, k in s['items'] if k < 0))
print('assign keys not ~0:', [k[-12:] + '=' + v for k, v in assign.items() if not k.endswith('~0')])
mine = {}
for pi, p in enumerate(P):
    cand = [k for (q, k) in stack_of if q == pi and k <= 0]
    kk = key(pi, cand[0]) if cand else None
    mine[pi] = assign.get(kk)
    e = exp.get(name(p)); p['_exp'] = e['title'] if e else None; p['_status'] = e['status'] if e else None
A = S['animals']; aname = {}
for pi, p in enumerate(P):
    if mine[pi] and mine[pi] != 'x' and p['_exp']: aname[mine[pi]] = p['_exp']
print('researcher animals:', {a: (sh(aname.get(a)), collections.Counter(mine.values())[a]) for a in sorted(A, key=lambda x: int(x[1:]))})
print('export statuses:', collections.Counter(r['status'] for r in rows), '| also_in_frame:', collections.Counter(bool(r['also_in_frame']) for r in rows))
tab = collections.Counter((mine[pi], p['cat']) for pi, p in enumerate(P)); cats = sorted({p['cat'] for p in P}, key=str); ans = sorted({m for m in mine.values()}, key=lambda a: (a is None, a == 'x', int(a[1:]) if a and a != 'x' else 0))
print('\nphotos: researcher animal (rows) x catalog animal (columns)')
print(f"{'':16s}" + ''.join(f'{sh(c):>8s}' for c in cats) + '   total')
for a in ans:
    print(f"{str(a) + ' ' + sh(aname.get(a, '')):16s}" + ''.join(f'{tab[(a, c)] or "":>8}' for c in cats) + f'   {sum(tab[(a, c)] for c in cats)}')
print(f"{'total':16s}" + ''.join(f'{sum(tab[(a, c)] for a in ans):>8}' for c in cats))
# timeline
print('\nday in capture order: photo number, clock, catalog, researcher, stack')
prev = None
for pi, p in enumerate(P):
    st = [s for (q, k), s in stack_of.items() if q == pi and k <= 0]; sid = st[0]['id'] if st else None
    gap = '' if pi == 0 else f"+{p['t'] - P[pi - 1]['t']:.0f}s"
    print(f"{pi:3d} {p['clock']} {gap:>7s} {p['cam'].split('|')[0][-6:]:6s} fl {p['fl']:>4} miss {str(p['miss']):>4s} {name(p)[-12:]:12s} cat {sh(p['cat']):6s} me {str(mine[pi]):4s} {sh(aname.get(mine[pi], '')):6s} stack {sid} {st[0]['kind'] if st else ''} n_inst {len(p['inst'])}")
json.dump({'mine': mine, 'aname': aname}, open('mine.json', 'w'))
