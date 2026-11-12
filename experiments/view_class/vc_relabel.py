"""Change labels after a look at the burst context: vc_relabel.py NAME SHEET "row:code row:code ..." (rows not listed keep their code;
every row of the sheet is marked ctx = seen in context)."""
import sys, json, re
from vc_common import *
name, sheet, spec = sys.argv[1], sys.argv[2], dict(s.split(':') for s in sys.argv[3].split())
idx = json.load(open(f'{HERE}/sheets/{name}.json'))[sheet]; L = load_labels(); n = 0
for r, (day, key) in idx.items():
    L.setdefault(key, {'day': day, 'code': None, 'round': name})['rev'] = name
    if r in spec:
        assert re.fullmatch(r'X|O|[DdFf]{1,2}', spec[r]); L[key]['was'] = L[key]['code']; L[key]['code'] = spec[r]; n += 1
save_labels(L)
from collections import Counter
print(sheet, n, 'changed;', dict(Counter(v['code'] for v in L.values())))
