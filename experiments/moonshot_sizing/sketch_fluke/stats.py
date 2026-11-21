"""Counts: drawable flukes, side agreement between the notch geometry and the CSE flag."""
import collections, sys
import numpy as np
import draw, bench
rows = []
for d in bench.TRAIN + bench.VAL:
    for r in draw.day_instances(d):
        rows.append(dict(day=d, ok=r['ok'], why=r['why'], notch=r.get('notch', np.nan), under=r['underside'], area=r.get('area', np.nan),
                         span=r.get('span_raw', np.nan), side=draw.side_of(r), p5=r.get('p5', np.nan), lite=float((r['n'][r['seen']] > 0.5).mean()) if r.get('seen') is not None and r['seen'].sum() else np.nan))
tot_inst = sum(len(bench.instances(d, 'fluke')) for d in bench.TRAIN + bench.VAL)
print('fluke instances', tot_inst, 'with cut-out', len(rows))
print(collections.Counter(r['why'] for r in rows))
ok = [r for r in rows if r['ok']]
print('drawable', len(ok), 'sides', collections.Counter(r['side'] for r in ok))
g = np.array([r['notch'] for r in rows]); u = np.array([r['under'] for r in rows])
f = np.isfinite(g)
print('notch>0 vs CSE underside crosstab', collections.Counter(zip(g[f] > 0.05, u[f])))
for nm, sel in (('notch below', g > 0.08), ('notch above', g < -0.02)):
    li = np.array([r['lite'] for r in rows])[sel & np.isfinite(np.array([r['lite'] for r in rows]))]
    print(nm, len(li), 'light share mean', li.mean().round(3), 'share with light > 0.2', (li > 0.2).mean().round(3))
for nm, sel in (('CSE under', u), ('CSE top', ~u)):
    li = np.array([r['lite'] for r in rows])[sel & np.isfinite(np.array([r['lite'] for r in rows]))]
    print(nm, len(li), 'light share mean', li.mean().round(3), 'share with light > 0.2', (li > 0.2).mean().round(3))
print('area quantiles (ok-ish)', np.nanpercentile([r['area'] for r in rows], [10, 25, 50, 75, 90]).round(3))
print('span quantiles', np.nanpercentile([r['span'] for r in rows], [10, 25, 50, 75, 90]).round(0))
import json; json.dump([{k: (None if isinstance(v, float) and not np.isfinite(v) else (bool(v) if isinstance(v, np.bool_) else v)) for k, v in r.items()} for r in rows], open('/home/sat3737/NOAA_whales/moonshot_cache/sketch_fluke/inst_stats.json', 'w'))
