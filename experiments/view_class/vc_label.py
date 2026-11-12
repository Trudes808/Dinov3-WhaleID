"""Record by-eye labels of one sheet: vc_label.py NAME SHEET "code code ..." (one code per tile, in tile order; see vc_common)."""
import sys, json, re
from vc_common import *
name, sheet, codes = sys.argv[1], sys.argv[2], sys.argv[3].split()
idx = json.load(open(f'{HERE}/sheets/{name}.json'))[sheet]
assert len(codes) == len(idx), f'{len(codes)} codes for {len(idx)} tiles'
L = load_labels()
for i, c in enumerate(codes):
    assert re.fullmatch(r'X|O|[DdFf]{1,2}', c) and len(set(c.lower())) == len(c), f'bad code {c!r} at tile {i}'
    day, key = idx[str(i)]; L[key] = {'day': day, 'code': c, 'round': name}
save_labels(L)
from collections import Counter
print(sheet, 'ok;', len(L), 'labels;', dict(Counter(v['code'] for v in L.values())))
