"""sketch_template: how much of a drawing's mark area the pad drawing loses because it falls outside the template's skin
(kept transfer), per drawing with a noted shape, 21 train + 3 val days. python clip_loss.py -> clip_loss.json"""
import json
import numpy as np
import common as C
FS = C.FS
import sys
METHOD = sys.argv[1] if len(sys.argv) > 1 else 'stretch'
T = FS.pad_templates(C.CACHE / 'templates.npz')
W = {s: FS.pad_template_view(t['cur'])['whale'] for s, t in T.items()}
tot = kept = 0; per = []
for d in C.DAYS:
    N = C.noted(d)
    for (c, s), v in C.drawings(d).items():
        sh = N.get(v['meta']['animal'])
        if not sh:
            continue
        r, _ = FS.pad_transfer(v, T[sh]['cur'], method=METHOD)
        if r is None:
            continue
        m = r['marks'] & r['seen']; a = int(m.sum())
        if a:
            k = int((m & W[sh]).sum()); tot += a; kept += k; per.append(k / a)
out = dict(drawings=len(per), mark_area_kept=round(kept / tot, 3), drawings_losing_over_20pct=round(float(np.mean(np.array(per) < 0.8)), 3))
out['method'] = METHOD; print(out); json.dump(out, open(C.HERE / f'clip_loss_{METHOD}.json', 'w'), indent=1)
