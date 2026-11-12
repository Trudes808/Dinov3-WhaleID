"""Start nw_labels.json from the clusters found by eye on 2026-11-11 (stage2_labels/nowhale.py): every instance of
those clusters. The two splash fragments of 2025-01-02 (clusters 147, 148) are left unlabelled ('skip'): a splash or a
blow belongs to a whale and is not what this filter is for."""
from nw_common import *
NOT_WHALE = {'2025-03-14': {45, 47, 48, 49, 50, 51, 53, 54, 55, 56, 62, 63, 64, 65, 66, 67}, '2025-01-02': {344, 349, 363, 364, 367, 396, 399, 401, 402}}
SKIP = {'2025-01-02': {147, 148}}
L = {}
for day in NOT_WHALE:
    C = json.load(open(f'{D}/{day}/safe/clusters.json'))['clusters']
    L[day] = {'pos': sorted(k for c in C if c['id'] in NOT_WHALE[day] for k in c['keys']), 'neg': [], 'skip': sorted(k for c in C if c['id'] in SKIP.get(day, ()) for k in c['keys'])}
    print(day, 'not-a-whale instances', len(L[day]['pos']), 'skip', len(L[day]['skip']))
save_labels(L)
