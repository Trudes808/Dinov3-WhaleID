"""The 2024-12-23 test run (state as saved at 14:48, before the researcher's own merge): the candidate scores for
every pair of the researcher's animals inside an encounter and across encounters."""
import json, itertools, collections, numpy as np
from sim import Day
day = Day('2024-12-23'); D = day.D; P = D['photos']; S = json.load(open('../2024-12-23/state_snapshot.json')); sh = lambda x: (x or '?').split('_')[-1]
key = lambda p, k: f"{P[p]['stem']}~{k if k >= 0 else 'x'}"
byA = collections.defaultdict(list)
for s in D['stacks']:
    a = S['assign'].get(key(*s['items'][0]))
    if a and a != 'x' and s['kind'] != 'extra': byA[a].append(s['id'])
cat = {a: collections.Counter(day.cat[i] for i in ss if day.cat[i]).most_common(1)[0][0] for a, ss in byA.items()}; seg = {a: S['animals'][a]['seg'] for a in byA}
rows = []
for a, b in itertools.combinations(sorted(byA, key=lambda x: int(x[1:])), 2):
    r = day.scores(byA[a], byA[b]); r.update(a=a, b=b, same=cat[a] == cat[b], enc=seg[a] == seg[b]); rows.append(r)
for title, sel in (('inside an encounter', [r for r in rows if r['enc']]), ('across encounters', [r for r in rows if not r['enc']])):
    print(f'\n{title}: {len(sel)} pairs, {sum(r["same"] for r in sel)} one whale by the catalog')
    for r in sorted(sel, key=lambda r: -r['stack_mean'])[:9]:
        f = lambda v: ' -  ' if v is None else f'{v:.2f}'
        print(f"   {r['a']:>3s} ({len(byA[r['a']])} stacks) + {r['b']:>3s} ({len(byA[r['b']])})  {'ONE WHALE' if r['same'] else 'different'}  stack_mean {r['stack_mean']:.2f}  blend {r['blend']:.2f}  link {r['link']:.2f} | fluke {f(r['fluke'])} dorsal {f(r['dorsal'])} other {f(r['other'])}")
