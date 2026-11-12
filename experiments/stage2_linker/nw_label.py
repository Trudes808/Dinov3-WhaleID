"""Add by-eye labels from a round's sheets: nw_label.py ROUND POS NEG SKIP, each a list like "0-39,41,44" of tile numbers
(not a whale / whale / unclear or a mask that merges a person with a whale)."""
import sys, json
from nw_common import load_labels, save_labels
def tiles(s):
    out = []
    for part in s.split(','):
        part = part.strip()
        if not part: continue
        a, _, b = part.partition('-'); out.extend(range(int(a), int(b or a) + 1))
    return out
rnd = sys.argv[1]; idx = json.load(open(f'sheets/nw_r{rnd}.json')); L = load_labels(); n = {}
for name, spec in zip(('pos', 'neg', 'skip'), sys.argv[2:5]):
    for t in tiles(spec):
        d, k = idx[str(t)]; e = L.setdefault(d, {'pos': [], 'neg': [], 'skip': []})
        for other in ('pos', 'neg', 'skip'):
            if k in e[other]: e[other].remove(k)
        e[name].append(k); n[name] = n.get(name, 0) + 1
seen = sum(len(tiles(s)) for s in sys.argv[2:5]); assert seen == len(set(t for s in sys.argv[2:5] for t in tiles(s))), 'a tile is listed twice'
save_labels(L); print('added', n, '| totals', {k: sum(len(v[k]) for v in L.values()) for k in ('pos', 'neg', 'skip')}, '| tiles on the sheets', len(idx), 'labelled now', seen)
