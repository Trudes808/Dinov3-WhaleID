"""As sim2_shared.py, with the page's runs approximated: a stack whose link to a stack of its own whale is >= 0.9
goes along with it without a question (the run), so only stacks that had to be asked can be split off. Then, for the
pairs of pieces with no fluke or fin in common: what tells the true pairs apart (AUC), and what a few rules would show.
usage (cwd with cache/): sim2_shared_runs.py val|eval|train"""
import sys, os, collections
import numpy as np
from sklearn.metrics import roc_auc_score
which = sys.argv[1]; sys.argv = [sys.argv[0], which, '0.2']
import sim2
from sim import SPLITS, BIT
alld = sorted(f[:-5] for f in os.listdir('cache') if f.endswith('.json'))
days = {'val': SPLITS['val'], 'eval': SPLITS['eval'], 'train': [d for d in alld if d not in SPLITS['val'] + SPLITS['eval']]}[which]
rule = lambda r: r['stack_mean'] >= 0.5 or (r['mut_stack_mean'] and r['stack_mean'] >= 0.25)
def sort_day(day, rng):
    out = {}
    for g, cs in day.scope.items():
        stacks = [i for c in cs for i in day.animals[c]]; ph = lambda i: 0 if day.bits[i] & BIT['fluke'] else 1 if day.bits[i] & BIT['fin'] else 2
        stacks.sort(key=lambda i: (ph(i), day.t0[i], i)); pieces = collections.defaultdict(list)
        for s in stacks:
            c = day.cat[s]; mine = [k for k in pieces if k[0] == c]; best, bk, run = -1, None, None
            for k in mine:
                if max(day.P[s, t] for t in pieces[k]) >= 230: run = k
                b = day.scores([s], pieces[k])['blend']
                if b > best: best, bk = b, k
            if run is not None: bk = run
            elif bk is None or rng.random() < sim2.P0 * min(max((sim2.HI - best) / (sim2.HI - sim2.LO), 0.0), 1.0): bk = (c, len(mine))
            pieces[bk].append(s)
        out[g] = dict(pieces)
    return out
T = collections.defaultdict(collections.Counter); rows = []; N = collections.Counter()
for seed in range(sim2.SEEDS):
    rng = np.random.default_rng(seed)
    for d in days:
        day = sim2.Day(d)
        for g, pieces in sort_day(day, rng).items():
            an = collections.Counter(k[0] for k in pieces); N['animals'] += len(an); N['pieces'] += len(pieces); N['split'] += sum(1 for v in an.values() if v > 1); hs = sim2.hints(day, pieces)
            best_nc = {}
            for r in hs:
                r['gap'] = day.gap(np.asarray(pieces[r['a']]), np.asarray(pieces[r['b']])); r['n'] = min(len(pieces[r['a']]), len(pieces[r['b']]))
                k = {'': 'no fluke or fin in common', 'fluke': 'a fluke only', 'dorsal': 'a dorsal fin only', 'flukedorsal': 'both'}[r['shared']]; r['k'] = k
                T[k]['pairs'] += 1; T[k]['same'] += r['same']; T[k]['shown'] += rule(r); T[k]['same shown'] += r['same'] and rule(r)
                if not r['shared']:
                    for x in (r['a'], r['b']):
                        if x not in best_nc or r['other'] > best_nc[x]['other']: best_nc[x] = r
            for r in hs:
                if not r['shared']: r['mut'] = best_nc[r['a']] is r and best_nc[r['b']] is r; rows.append(r)
n = sim2.SEEDS; tot = sum(v['same'] for v in T.values())
print(f"{which}, with runs: per run {N['animals'] / n:.0f} animals -> {N['pieces'] / n:.1f} pieces, {N['split'] / n:.1f} whales left in pieces")
for k in ('both', 'a fluke only', 'a dorsal fin only', 'no fluke or fin in common'):
    v = T[k]
    if v['pairs']: print(f"   in common: {k:28s} {v['pairs'] / n:7.1f} pairs, one whale {v['same'] / n:6.1f} ({v['same'] / v['pairs']:.2f}) = {v['same'] / max(tot, 1):.2f} of all true pairs; the rule shows {v['shown'] / n:5.1f}, of them right {v['same shown'] / n:5.1f}")
y = np.array([r['same'] for r in rows])
if len(rows) > 50 and 0 < y.mean() < 1:
    for nm, v in (('other part of the blend', [r['other'] for r in rows]), ('link score', [r['link'] for r in rows]), ('stack-average similarity', [r['stack_mean'] for r in rows]), ('shorter time gap', [-r['gap'] for r in rows])):
        print(f"   no view in common, {nm:26s} AUC {roc_auc_score(y, v):.3f}")
    for nm, f in (('other >= 0.5', lambda r: r['other'] >= 0.5), ('other >= 0.35', lambda r: r['other'] >= 0.35), ('each the other\'s closest such pair and other >= 0.25', lambda r: r['mut'] and r['other'] >= 0.25), ('each the other\'s closest such pair and other >= 0.15', lambda r: r['mut'] and r['other'] >= 0.15),
                  ('gap <= 60 s', lambda r: r['gap'] <= 60), ('gap <= 300 s', lambda r: r['gap'] <= 300)):
        sh = [r for r in rows if f(r) and not rule(r)]; print(f"   extra hints '{nm}': {len(sh) / n:5.1f} per run, right {sum(r['same'] for r in sh) / n:5.1f} ({sum(r['same'] for r in sh) / max(len(sh), 1):.2f})")
