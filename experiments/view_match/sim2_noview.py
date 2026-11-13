"""A second class of merge hints from the view labels: two pieces with NO fluke or dorsal fin in common (they could
not be compared), each the other's closest such piece by the page's stack-average similarity (what animalSim gives).
Counts on top of the page's rule; runs approximated as in sim2_shared_runs.py.  usage: sim2_noview.py val|eval|train"""
import sys, os, collections
import numpy as np
which = sys.argv[1]; sys.argv = [sys.argv[0], which, '0.2']
import sim2
from sim import SPLITS
from sim2_shared_runs_lib import sort_day
alld = sorted(f[:-5] for f in os.listdir('cache') if f.endswith('.json'))
days = {'val': SPLITS['val'], 'eval': SPLITS['eval'], 'train': [d for d in alld if d not in SPLITS['val'] + SPLITS['eval']]}[which]
rule = lambda r: r['stack_mean'] >= 0.5 or (r['mut_stack_mean'] and r['stack_mean'] >= 0.25)
T = collections.Counter(); X = {t: collections.Counter() for t in (0.10, 0.15, 0.20)}
for seed in range(sim2.SEEDS):
    rng = np.random.default_rng(seed)
    for d in days:
        day = sim2.Day(d)
        for g, pieces in sort_day(day, rng).items():
            an = collections.Counter(k[0] for k in pieces); T['split'] += sum(1 for v in an.values() if v > 1); T['enc'] += len(pieces) >= 2; hs = sim2.hints(day, pieces)
            shown = [r for r in hs if rule(r)]; T['shown'] += len(shown); T['wrong'] += sum(not r['same'] for r in shown); base = {r['a'][0] for r in shown if r['same']}; T['reached'] += len(base)
            nv = [r for r in hs if not r['shared']]; best = {}
            for r in nv:
                for x in (r['a'], r['b']):
                    if x not in best or r['stack_mean'] > best[x]['stack_mean']: best[x] = r
            for t in X:
                extra = [r for r in nv if not rule(r) and r['stack_mean'] >= t and best[r['a']] is r and best[r['b']] is r]
                X[t]['hints'] += len(extra); X[t]['right'] += sum(r['same'] for r in extra); X[t]['new whales'] += len({r['a'][0] for r in extra if r['same']} - base)
n = sim2.SEEDS
print(f"{which}: per run {T['split'] / n:.1f} whales left in pieces; the page's rule: true hint for {T['reached'] / n:.1f}, {T['wrong'] / n:.1f} wrong hints ({T['wrong'] / max(T['enc'], 1):.2f} per encounter)")
for t, v in X.items(): print(f"   + no view in common, each the other's closest, similarity >= {t:.2f}: {v['hints'] / n:5.1f} more hints, {v['right'] / n:5.1f} right ({v['right'] / max(v['hints'], 1):.2f}), split whales reached that had no hint: {v['new whales'] / n:5.1f}")
