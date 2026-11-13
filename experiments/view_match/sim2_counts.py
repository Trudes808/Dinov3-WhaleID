"""sim2's researcher and the page's merge-hint rule, in absolute numbers: split whales, how many get a true hint, wrong hints.
usage (cwd = a folder with cache/): sim2_counts.py val|eval|train"""
import sys, collections
import numpy as np
sys.argv = [sys.argv[0], sys.argv[1], '0.2']
import sim2
from sim import SPLITS
import os
which = sys.argv[1]; alld = sorted(f[:-5] for f in os.listdir('cache') if f.endswith('.json'))
days = {'val': SPLITS['val'], 'eval': SPLITS['eval'], 'train': [d for d in alld if d not in SPLITS['val'] + SPLITS['eval']]}[which]
rules = {'0.50 / 0.25 mutual (the page)': lambda r: r['stack_mean'] >= 0.5 or (r['mut_stack_mean'] and r['stack_mean'] >= 0.25),
         '0.45 / 0.25 mutual': lambda r: r['stack_mean'] >= 0.45 or (r['mut_stack_mean'] and r['stack_mean'] >= 0.25),
         '0.45 / 0.20 mutual': lambda r: r['stack_mean'] >= 0.45 or (r['mut_stack_mean'] and r['stack_mean'] >= 0.20),
         '0.40 / 0.20 mutual': lambda r: r['stack_mean'] >= 0.40 or (r['mut_stack_mean'] and r['stack_mean'] >= 0.20)}
T = collections.Counter(); R = {k: collections.Counter() for k in rules}
for seed in range(sim2.SEEDS):
    rng = np.random.default_rng(seed)
    for d in days:
        day = sim2.Day(d)
        for g, pieces in sim2.sort_day(day, rng).items():
            rows = sim2.hints(day, pieces); animals = collections.Counter(k[0] for k in pieces); T['animals'] += len(animals); T['pieces'] += len(pieces); T['split whales'] += sum(1 for v in animals.values() if v > 1); T['encounters'] += len(pieces) >= 2
            for name, rule in rules.items():
                sh = [r for r in rows if rule(r)]; hit = {r['a'][0] for r in sh if r['same']}; R[name]['reached'] += len(hit); R[name]['wrong'] += sum(not r['same'] for r in sh)
n = sim2.SEEDS
print(f"{which}: per run: {T['animals'] / n:.0f} animals -> {T['pieces'] / n:.1f} pieces, {T['split whales'] / n:.1f} whales left in pieces")
for name in rules: print(f"   {name:30s} true hint for {R[name]['reached'] / n:5.1f} of them, {(T['split whales'] - R[name]['reached']) / n:5.1f} split whales without a hint, {R[name]['wrong'] / n:5.1f} wrong hints ({R[name]['wrong'] / max(T['encounters'], 1):.2f} per encounter)")
