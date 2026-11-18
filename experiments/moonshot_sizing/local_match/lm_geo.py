"""Offline geometric scores from the stored LightGlue matches (CPU): what the first pilot showed is that two different
backs give many geometrically consistent matches ALONG THE OUTLINE (back line against the water; a smooth curve fits
any homography), so here the matches are split by their depth inside the whale mask and the spread of the inliers
across the outline is measured.

    python lm_geo.py VARIANT PART DAY [DAY ...]     -> CACHE/VARIANT/DAY__PART__geo.npz (jobs, stats, names)
Stats per instance pair (RANSAC thresholds 1.2 % of the chart):
  aff_all       affine inliers, all matches
  aff_int       affine inliers among the interior matches (both ends >= DEPTH px inside the mask)
  hom_int       homography inliers among the interior matches
  n_int         interior matches (no geometry)
  minor_all     sqrt of the smaller eigenvalue of the affine inliers' covariance (px; ~0 = inliers on a line / curve)
  aff_all_2d    aff_all * min(1, minor_all / 40)
  conf_int      summed LightGlue confidence of the interior affine inliers
"""
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm_lib as LM  # noqa: E402

DEPTH = 15
NAMES = ('aff_all', 'aff_int', 'hom_int', 'n_int', 'minor_all', 'aff_all_2d', 'conf_int')
_D = {}


def _init(dt):
    _D.update(dt)


def _one(args):
    pa, pb, sc, ka, kb, mi, size = args
    thr = 0.012 * size
    da, db = _D[ka], _D[kb]
    if mi:
        db = db[:, ::-1]
    out = dict.fromkeys(NAMES, 0.0)
    if len(pa) >= 3:
        Ma, ia = cv2.estimateAffine2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=thr, maxIters=2000, confidence=0.995)
        if Ma is not None:
            inl = ia.ravel().astype(bool)
            out['aff_all'] = int(inl.sum())
            if inl.sum() >= 3:
                ev = np.linalg.eigvalsh(np.cov(pa[inl].T))
                out['minor_all'] = float(np.sqrt(max(ev[0], 0)))
            out['aff_all_2d'] = out['aff_all'] * min(1.0, out['minor_all'] / 40.0)
    ia_ = np.clip(pa.round().astype(int), 0, size - 1)
    ib_ = np.clip(pb.round().astype(int), 0, size - 1)
    deep = (da[ia_[:, 1], ia_[:, 0]] >= DEPTH) & (db[ib_[:, 1], ib_[:, 0]] >= DEPTH) if len(pa) else np.zeros(0, bool)
    out['n_int'] = int(deep.sum())
    if deep.sum() >= 3:
        qa, qb, qs = pa[deep], pb[deep], sc[deep]
        Ma, ia = cv2.estimateAffine2D(qa, qb, method=cv2.RANSAC, ransacReprojThreshold=thr, maxIters=2000, confidence=0.995)
        if Ma is not None:
            inl = ia.ravel().astype(bool)
            out['aff_int'] = int(inl.sum())
            out['conf_int'] = float(qs[inl].sum())
        if deep.sum() >= 5:
            H, ih = cv2.findHomography(qa, qb, cv2.USAC_MAGSAC, thr, maxIters=2000, confidence=0.995)
            if H is not None:
                out['hom_int'] = int(ih.sum())
    return [out[n] for n in NAMES]


def main():
    variant, part, days = sys.argv[1], sys.argv[2], sys.argv[3:]
    for day in days:
        fp = LM.CACHE / variant / f'{day}__{part}__geo.npz'
        mp = LM.CACHE / variant / f'{day}__{part}__matches.npz'
        if fp.exists() or not mp.exists():
            print(day, 'cached' if fp.exists() else 'no matches yet'); continue
        z = np.load(LM.CACHE / variant / f'{day}__{part}__pairs.npz')
        m = np.load(mp)
        J, keys = z['jobs'], [str(k) for k in z['keys']]
        size = torch.load(LM.CACHE / variant / f'{day}__{part}.pt', weights_only=False)['size']
        dt = {}
        for k in keys:
            _, wh, _ = LM.load_chart(day, k, part, size)
            dt[k] = cv2.distanceTransform(wh.astype(np.uint8), cv2.DIST_L2, 5)
        off, pa, pb, sc = m['off'], m['pa'], m['pb'], m['sc']
        tasks = [(pa[off[i]:off[i + 1]], pb[off[i]:off[i + 1]], sc[off[i]:off[i + 1]], keys[J[i, 2]], keys[J[i, 3]], int(J[i, 4]), size) for i in range(len(J))]
        with Pool(8, initializer=_init, initargs=(dt,)) as pool:
            rows = pool.map(_one, tasks, chunksize=128)
        np.savez_compressed(fp, jobs=J, stats=np.array(rows, np.float32).reshape(len(J), -1), names=np.array(NAMES), keys=np.array(keys))
        print(day, part, len(J), 'instance pairs', flush=True)


if __name__ == '__main__':
    main()
