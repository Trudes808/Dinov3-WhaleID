"""Cache every instance's canonical drawing (seen mask, normalised brightness, canonical photo) + a fluke shape template.
    python build.py      -> CACHE/inst/<day>.npz + <day>.json, CACHE/template.npy
"""
import json
import numpy as np
import draw  # noqa  (puts the bench folder on the path)
import bench
from prep import CACHE

out = CACHE / 'inst'
out.mkdir(exist_ok=True)
tmpl, nt = np.zeros((draw.H, draw.W)), 0
for d in bench.TRAIN + bench.VAL:
    rs = [r for r in draw.day_instances(d) if r.get('seen') is not None]
    np.savez_compressed(out / f'{d}.npz', seen=np.array([r['seen'] for r in rs], bool), n=(np.array([r['n'] for r in rs]) * 200).clip(0, 255).astype(np.uint8),
                        im=np.array([r['im'] for r in rs], np.uint8))
    meta = [{k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items() if k not in ('seen', 'n', 'im', 'M')} for r in rs]
    json.dump(meta, open(out / f'{d}.json', 'w'))
    for r in rs:
        if r['ok'] and r.get('frame') == 'points' and (r['n'][r['seen']] > 0.5).mean() > 0.3 and r['notch'] > 0.08:
            tmpl += r['seen']; nt += 1
    print(d, len(rs), nt, flush=True)
np.save(CACHE / 'template.npy', tmpl / max(nt, 1))
open(out / 'DONE', 'w').write('ok')
