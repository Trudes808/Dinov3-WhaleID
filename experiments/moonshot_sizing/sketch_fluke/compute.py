"""Counts of drawable flukes per side and the drawing comparisons of every bench pair -> CACHE/pair_scores.pkl"""
import collections, pickle, time
import numpy as np
import score as SC
import bench
t0 = time.time()
cnt = collections.Counter(); ncl = collections.Counter(); allS = {}; cls = {}
for d in bench.TRAIN + bench.VAL:
    meta = SC.inst(d)[0]
    cnt['instances with cut-out'] += len(meta)
    for r in meta:
        cnt[f"drawable side={r['side']}" if r['drawable'] else f"not drawable ({r['why'] or 'shape: prec/rec'})"] += 1
    D = SC.cluster_drawings(d)
    for c, v in D.items():
        for s in v:
            ncl[s] += 1
        cls[(d, c)] = {s: (x['cls'], x['light'], x['frames'], x['rec']) for s, x in v.items()}
    allS[d] = SC.day_scores(d)
    print(d, len(allS[d]), 'of', len(bench.pairs(d, 'fluke')), round(time.time() - t0), flush=True)
print(dict(cnt)); print('clusters with a drawing per side', dict(ncl), 'of', sum(len(bench.clusters(d, 'fluke')) for d in bench.TRAIN + bench.VAL))
pickle.dump(dict(S=allS, cls=cls, cnt=dict(cnt), ncl=dict(ncl)), open(SC.CACHE / 'pair_scores.pkl', 'wb'))
