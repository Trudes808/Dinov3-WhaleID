"""Merge hints, second simulation: a researcher who never joins two whales but sometimes starts a new animal for a
stack of a whale already there, more often the less the page's similarity says they match (the 2024-12-23 test run:
all 5 wrong answers were "new animal", none was a wrong "same"). Stacks come in the page's order (flukes, dorsal fins,
the rest). P(new | true animal offered, similarity b) = P0 * clip((HI - b) / (HI - LO), 0, 1).
Then: which rule on which animal-to-animal score shows the pairs of pieces of one whale, at what price in wrong hints?
usage: sim2.py val|train|eval [P0]"""
import sys, os, collections, itertools, json
import numpy as np
from sim import Day, SPLITS, BIT
P0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6; LO, HI = 0.25, 0.75; SEEDS = 5
def sort_day(day, rng):
    """-> {scope: {piece id: (catalogue animal, [stacks])}}"""
    out = {}
    for g, cs in day.scope.items():
        stacks = [i for c in cs for i in day.animals[c]]; ph = lambda i: 0 if day.bits[i] & BIT['fluke'] else 1 if day.bits[i] & BIT['fin'] else 2
        stacks.sort(key=lambda i: (ph(i), day.t0[i], i)); pieces = collections.defaultdict(list); owner = {}
        for s in stacks:
            c = day.cat[s]; mine = [k for k in pieces if k[0] == c]; best, bk = -1, None
            for k in mine:                                   # the researcher is offered the pieces of the true animal; a wrong animal is always refused
                b = day.scores([s], pieces[k])['blend']
                if b > best: best, bk = b, k
            if bk is None or rng.random() < P0 * min(max((HI - best) / (HI - LO), 0.0), 1.0): bk = (c, len(mine))
            pieces[bk].append(s)
        out[g] = dict(pieces)
    return out
def hints(day, pieces):
    ks = list(pieces); rows = []
    for a, b in itertools.combinations(ks, 2):
        if day.conflict(pieces[a], pieces[b]): continue
        r = day.scores(pieces[a], pieces[b]); r.update(a=a, b=b, same=a[0] == b[0]); rows.append(r)
    for nm in ('stack_mean', 'blend', 'link'):               # mutual best match: each is the other's most similar animal of the encounter
        best = {}
        for r in rows:
            for x in (r['a'], r['b']):
                if x not in best or r[nm] > best[x][0]: best[x] = (r[nm], id(r))
        for r in rows: r['mut_' + nm] = best[r['a']][1] == id(r) and best[r['b']][1] == id(r)
    return rows
RULES = [('link >= 0.85 (the page until now)', lambda r: r['link'] >= 0.85)]
for t in (0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2): RULES.append((f'similarity >= {t:.2f}', lambda r, t=t: r['stack_mean'] >= t))
for t in (0.6, 0.5, 0.4, 0.3): RULES.append((f'best-per-part blend >= {t:.2f}', lambda r, t=t: r['blend'] >= t))
for hi, lo in ((0.45, 0.25), (0.4, 0.25), (0.4, 0.2), (0.35, 0.2), (0.4, 0.15), (0.35, 0.15)):
    RULES.append((f'similarity >= {hi:.2f}, or >= {lo:.2f} when each is the other\'s closest', lambda r, hi=hi, lo=lo: r['stack_mean'] >= hi or (r['mut_stack_mean'] and r['stack_mean'] >= lo)))
new = lambda r, hi=0.45, lo=0.25: r['stack_mean'] >= hi or (r['mut_stack_mean'] and r['stack_mean'] >= lo)
RULES.append(('the rule above with 0.45 / 0.25, or link >= 0.85', lambda r: new(r) or r['link'] >= 0.85))
RULES.append(('similarity >= 0.50, or >= 0.25 when each is the other\'s closest', lambda r: new(r, 0.5, 0.25)))
RULES.append(('similarity >= 0.45, or >= 0.30 when each is the other\'s closest', lambda r: new(r, 0.45, 0.3)))
for hi, lo in ((0.6, 0.3), (0.6, 0.35), (0.65, 0.35), (0.7, 0.35), (0.65, 0.4)):
    RULES.append((f'blend >= {hi:.2f}, or >= {lo:.2f} when each is the other\'s closest (by blend)', lambda r, hi=hi, lo=lo: r['blend'] >= hi or (r['mut_blend'] and r['blend'] >= lo)))
for lo in (0.3, 0.25, 0.2, 0.15): RULES.append((f'each the other\'s closest and similarity >= {lo:.2f}', lambda r, lo=lo: r['mut_stack_mean'] and r['stack_mean'] >= lo))
def run(tag, days):
    encs = []; frag = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        for d in days:
            day = Day(d)
            for g, pieces in sort_day(day, rng).items():
                rows = hints(day, pieces); na = len({k[0] for k in pieces}); encs.append(dict(day=d, scope=g, animals=na, pieces=len(pieces), rows=rows)); frag.append((na, len(pieces)))
    na = sum(a for a, _ in frag); npc = sum(p for _, p in frag); pos = sum(r['same'] for e in encs for r in e['rows']); neg = sum(not r['same'] for e in encs for r in e['rows'])
    split_animals = sum(1 for e in encs for c, n in collections.Counter(k[0] for r in e['rows'] for k in ()).items())
    print(f"\n===== {tag}: {len(days)} days x {SEEDS} runs, P0 {P0}: {na} animals sorted into {npc} ({npc / na:.2f} per animal; the test run: 10 for 6 = 1.67); pairs of pieces of one whale {pos}, of different whales {neg} =====")
    print(f"   {'rule':66s} {'shown of the':>12s} {'split whales':>12s} {'wrong hints':>12s} {'right of':>9s} {'encounters with':>16s}")
    print(f"   {'':66s} {'true pairs':>12s} {'reached':>12s} {'per encounter':>12s} {'shown':>9s} {'over 3 wrong':>16s}")
    for name, rule in RULES:
        tp = fp = 0; reached = total = 0; over = 0
        for e in encs:
            sh = [r for r in e['rows'] if rule(r)]; t = sum(r['same'] for r in sh); f = len(sh) - t; tp += t; fp += f; over += f > 3
            split = collections.defaultdict(set)
            for r in e['rows']:
                if r['same']: split[r['a'][0]].add(False)
            for r in sh:
                if r['same']: split[r['a'][0]].add(True)
            total += len(split); reached += sum(True in v for v in split.values())
        multi = sum(1 for e in encs if e['pieces'] >= 2)
        print(f"   {name:66s} {tp / max(pos, 1):12.2f} {reached / max(total, 1):12.2f} {fp / multi:12.2f} {tp / max(tp + fp, 1):9.2f} {over / multi:16.2f}")
    # how the true pairs score
    v = np.array([r['stack_mean'] for e in encs for r in e['rows'] if r['same']]); w = np.array([r['stack_mean'] for e in encs for r in e['rows'] if not r['same']])
    print(f"   similarity of the true pairs: 10th / 25th / 50th percentile {np.percentile(v, [10, 25, 50]).round(2)} | of pairs of different whales: 50th / 75th / 90th {np.percentile(w, [50, 75, 90]).round(2)}")
if __name__ == '__main__':
    alld = sorted(f[:-5] for f in os.listdir('cache') if f.endswith('.json')); which = sys.argv[1]
    run(which, {'val': SPLITS['val'], 'eval': SPLITS['eval'], 'train': [d for d in alld if d not in SPLITS['val'] + SPLITS['eval']]}[which])
