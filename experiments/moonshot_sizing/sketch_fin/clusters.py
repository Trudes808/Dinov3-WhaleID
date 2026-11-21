"""sketch_fin step 3: one consensus drawing per safe cluster (dorsal instances of the cluster, majority side, frames
registered on the best-seen frame), for every bench cluster of the 21 train + 3 val days.

    OMP_NUM_THREADS=1 python clusters.py [n_proc]
Writes CACHE/clusters/<day>.pkl: {cluster: dict(top, trail, marks (packed), seen_n, whale (packed), n, n_all, side,
nick_top, nick_trail, H (median px in the source), ncc (median), ref key, base)}.
"""
import pickle
import sys
from multiprocessing import Pool

import numpy as np

import extract  # noqa: F401
import bench
import draw as DR
from extract import CACHE


def run_day(day):
    out = CACHE / 'clusters' / f'{day}.pkl'
    if out.exists():
        return day
    keys, z, info = DR.load_day(day)
    idx = {k: i for i, k in enumerate(keys)}
    rgb, mask, ins = z['rgb'], z['mask'], z['inside']
    res = {}
    for c, ks in bench.clusters(day, 'dorsal').items():
        ks = [k for k in ks if k in idx]
        if not ks:
            continue
        ii = [idx[k] for k in ks]
        used, fr, cons, side, ref = DR.cluster_drawing((rgb[ii], mask[ii], ins[ii]), info, ks)
        res[c] = dict(top=cons['top'], trail=cons['trail'], marks=np.packbits(cons['marks']), seen_n=cons['seen_n'].astype(np.uint8),
                      raw=cons['raw'].astype(np.uint8), whale=np.packbits(cons['whale']), n=cons['n'], n_all=len(ks), side=bool(side),
                      nick_top=list(cons['nick_top']), nick_trail=list(cons['nick_trail']),
                      H=float(np.median([info[k]['H'] for k in used])), base=float(np.median([info[k]['base'] for k in used])),
                      ncc=float(np.median([f.get('ncc', 1.0) for f in fr])), ref=used[ref], keys=used)
    out.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(out, 'wb'))
    return day


def load(day):
    r = pickle.load(open(CACHE / 'clusters' / f'{day}.pkl', 'rb'))
    shp = (extract.TH, extract.TW)
    for v in r.values():
        if isinstance(v['marks'], np.ndarray) and v['marks'].ndim == 1:
            v['marks'] = np.unpackbits(v['marks'])[:shp[0] * shp[1]].reshape(shp).astype(bool)
            v['whale'] = np.unpackbits(v['whale'])[:shp[0] * shp[1]].reshape(shp).astype(bool)
    return r


if __name__ == '__main__':
    import cv2
    cv2.setNumThreads(1)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    days = sorted(bench.TRAIN + bench.VAL, key=lambda d: -len(bench.instances(d, 'dorsal')))
    with Pool(n) as pool:
        for d in pool.imap_unordered(run_day, days):
            print('done', d, flush=True)
    print('ALL DONE', flush=True)
