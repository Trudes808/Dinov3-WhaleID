"""Whose similarity should the merge hints use? The simulated researcher splits by similarity A (starts a new animal
more often the lower A is for the true animal); the hints are then computed with similarity B. A = B is the
self-referential case of sim2. usage: sim2_cross.py val|eval DIR_STUDENT DIR_VIEWMATCH"""
import sys, os, collections
import numpy as np
split, dirs = sys.argv[1], {'student (classifier-gated)': sys.argv[2], 'view match blend': sys.argv[3]}
sys.argv = [sys.argv[0], split, '0.2']
import sim2
from sim import SPLITS
days = SPLITS[split]
def load(d, day):
    cwd = os.getcwd(); os.chdir(d)
    try: return sim2.Day(day)
    finally: os.chdir(cwd)
D = {n: {day: load(p, day) for day in days} for n, p in dirs.items()}
rule = lambda r, lo=0.25: r['stack_mean'] >= 0.5 or (r['mut_stack_mean'] and r['stack_mean'] >= lo)
print(f'{split}: per run; the page rule (0.50, or 0.25 when each is the other\'s closest)')
for A in dirs:
    for B in list(dirs) + ['the higher of the two']:
        T = collections.Counter()
        for seed in range(sim2.SEEDS):
            rng = np.random.default_rng(seed)
            for day in days:
                for g, pieces in sim2.sort_day(D[A][day], rng).items():
                    animals = collections.Counter(k[0] for k in pieces); T['split'] += sum(1 for v in animals.values() if v > 1); T['enc'] += len(pieces) >= 2
                    if B in dirs: sh = [r for r in sim2.hints(D[B][day], pieces) if rule(r)]
                    else:
                        shown = {}
                        for n in dirs:
                            for r in sim2.hints(D[n][day], pieces):
                                if rule(r): shown[(r['a'], r['b'])] = r
                        sh = list(shown.values())
                    T['reached'] += len({r['a'][0] for r in sh if r['same']}); T['wrong'] += sum(not r['same'] for r in sh)
        n = sim2.SEEDS
        print(f"   researcher splits by: {A:28s} hints by: {B:28s} {T['split'] / n:5.1f} split whales, true hint for {T['reached'] / n:5.1f}, without a hint {(T['split'] - T['reached']) / n:5.1f}, wrong hints {T['wrong'] / n:5.1f} ({T['wrong'] / max(T['enc'], 1):.2f} per encounter)")
