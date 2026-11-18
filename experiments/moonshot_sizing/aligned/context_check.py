"""Bench check (not a matcher): how far do photo-condition cues alone go?  Pair score = minus the Lab distance of the
whale colour and of the water colour (photos.json, per instance), best over instance pairs.  Label-free."""
import numpy as np

import common  # noqa: F401
import bench
from common import photos

for part in ('dorsal', 'fluke'):
    for tag, fields in (('whale+water Lab', ('lab_whale', 'lab_water')), ('water Lab only', ('lab_water',))):
        S = {}
        for d in bench.TRAIN + bench.VAL:
            ph = photos(d); V = {}
            for st, p in ph.items():
                for ins in p.get('instances', []):
                    if all(ins.get(f) for f in fields):
                        V[ins['key']] = np.concatenate([np.array(ins[f], float) for f in fields])
            cl = {c: [V[k] for k in ks if k in V] for c, ks in bench.clusters(d, part).items()}
            S[d] = {}
            for a, b, _ in bench.pairs(d, part):
                if cl.get(a) and cl.get(b):
                    A, B = np.array(cl[a]), np.array(cl[b])
                    S[d][(a, b)] = float(-np.linalg.norm(A[:, None] - B[None], axis=-1).min())
        bench.evaluate_pairs(f'aligned_ctx_{"ww" if len(fields) == 2 else "water"}', part, S)
