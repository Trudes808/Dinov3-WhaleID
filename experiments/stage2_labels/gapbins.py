"""Catalog same-rate of cluster pairs by time gap (pairs within 15 min, both clusters with a catalog animal, non-whales
left out), per day and for the val days together; and on the labelled day the researcher's verdicts in the same bins.
Read-only."""
import pickle, json, collections
import numpy as np
from nowhale import NOT_WHALE, REF, VAL
BINS = [(-1, 0), (0, 1.5), (1.5, 5), (5, 15), (15, 60), (60, 300), (300, 900)]
def load(day):
    R = pickle.load(open(f'transfer_raw_{day}.pkl', 'rb')); ex = NOT_WHALE[day]
    keep = np.array([a not in ex and b not in ex for a, b in R['pairs']]); return R, keep
tot = collections.defaultdict(lambda: np.zeros(3, int))
for day in [REF] + VAL:
    R, keep = load(day); g = np.asarray(R['gap'], float); c = np.asarray(R['cat']); line = []
    for lo, hi in BINS:
        m = keep & (g > lo) & (g <= hi); v = np.array([(c[m] == 'same').sum(), (c[m] == 'diff').sum(), (c[m] == '?').sum()])
        if day in VAL: tot[(lo, hi)] += v
        line.append(f"({lo}, {hi}] s: same {v[0]}, different {v[1]}, unknown {v[2]}, same rate {v[0] / max(v[0] + v[1], 1):.2f}")
    print(day + ' | ' + ' | '.join(line))
print('val days together | ' + ' | '.join(f"({lo}, {hi}] s: same {v[0]}, different {v[1]}, unknown {v[2]}, same rate {v[0] / max(v[0] + v[1], 1):.2f}" for (lo, hi), v in tot.items()))
# researcher's verdicts on the labelled day in the same bins (cluster-level records)
z = json.load(open(f'/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all/{REF}/safe/clusters.json')); C = {c['id']: c for c in z['clusters']}
L = [r for r in json.load(open(f'/home/sat3737/NOAA_whales/Whale-Review-Web/data/{REF}/pair_marks.json'))['labels'] if r.get('level') == 'cluster']
def gap(a, b):
    A, B = C[a], C[b]; return max(0.0, max(A['start'], B['start']) - min(A['end'], B['end']))
for lo, hi in BINS:
    v = collections.Counter(r['verdict'] for r in L if lo < gap(r['set_a'], r['set_b']) <= hi); n = sum(v.values())
    print(f"researcher verdicts on {REF}, ({lo}, {hi}] s: {n} pairs | " + ', '.join(f"{k} {x} ({x / max(n, 1):.2f})" for k, x in v.most_common()))
sug = collections.Counter(); lab = {(min(r['set_a'], r['set_b']), max(r['set_a'], r['set_b'])) for r in L}
for s in z['suggestions']:
    gp = gap(s['a'], s['b']); b = next((f'({lo}, {hi}]' for lo, hi in BINS if lo < gp <= hi), '> 900'); sug[(b, (min(s['a'], s['b']), max(s['a'], s['b'])) in lab)] += 1
print('suggestions by gap bin (labelled on the bursts page / open):', {b: (sug[(b, True)], sug[(b, False)]) for b in dict.fromkeys(k[0] for k in sorted(sug, key=lambda k: [f'({lo}, {hi}]' for lo, hi in BINS].index(k[0]) if k[0] != '> 900' else 99))})
