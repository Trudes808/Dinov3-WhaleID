"""Among the pairs of pieces the merge hints choose from (sim2's researcher): what the two pieces have in common by
the view classifier's labels, how often such a pair is one whale, and how many of those the page's rule shows.
usage (cwd with cache/): sim2_shared.py val|eval|train"""
import sys, os, collections
import numpy as np
which = sys.argv[1]; sys.argv = [sys.argv[0], which, '0.2']
import sim2
from sim import SPLITS
alld = sorted(f[:-5] for f in os.listdir('cache') if f.endswith('.json'))
days = {'val': SPLITS['val'], 'eval': SPLITS['eval'], 'train': [d for d in alld if d not in SPLITS['val'] + SPLITS['eval']]}[which]
rule = lambda r: r['stack_mean'] >= 0.5 or (r['mut_stack_mean'] and r['stack_mean'] >= 0.25)
T = collections.defaultdict(collections.Counter)
for seed in range(sim2.SEEDS):
    rng = np.random.default_rng(seed)
    for d in days:
        day = sim2.Day(d)
        for g, pieces in sim2.sort_day(day, rng).items():
            for r in sim2.hints(day, pieces):
                k = {'': 'no fluke or fin in common', 'fluke': 'a fluke only', 'dorsal': 'a dorsal fin only', 'flukedorsal': 'both'}[r['shared']]
                T[k]['pairs'] += 1; T[k]['same'] += r['same']; T[k]['shown'] += rule(r); T[k]['same shown'] += r['same'] and rule(r)
n = sim2.SEEDS; tot = sum(v['same'] for v in T.values())
print(f'{which}: candidate pairs of pieces inside an encounter, per run')
for k in ('both', 'a fluke only', 'a dorsal fin only', 'no fluke or fin in common'):
    v = T[k]
    if v['pairs']: print(f"   in common: {k:28s} {v['pairs'] / n:7.1f} pairs, one whale {v['same'] / n:6.1f} ({v['same'] / v['pairs']:.2f}) = {v['same'] / max(tot, 1):.2f} of all true pairs; the rule shows {v['shown'] / n:5.1f}, of them right {v['same shown'] / n:5.1f}")
