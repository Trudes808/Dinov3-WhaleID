"""Sanity check: do the verified matches land where the researcher looked? (2025-01-02, a train day; CPU, read-only)

For every record of Whale-Review-Web/data/2025-01-02/pair_marks.json with marked region pairs whose two photos both
have a fin (or both a fluke) cut-out: ALIKED + LightGlue + RANSAC between the two charts (fins: best of plain / mirrored),
then the share of the inliers inside the marked boxes (mapped into the chart) against the share of the whale pixels the
boxes cover (= what random points on the whale would give).
    python lm_marks.py [VARIANT]
"""
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm_lib as LM  # noqa: E402
import lm_feats as FE  # noqa: E402
from lm_lib import bench  # noqa: E402
from lm_match import lightglue  # noqa: E402

DAY = '2025-01-02'
MARKS = '/home/sat3737/NOAA_whales/Whale-Review-Web/data/2025-01-02/pair_marks.json'


def box_poly(day, key, part, r, size):
    z = np.load(LM.chart_paths(day, key, part)[1], allow_pickle=True)
    M, sf = z['M'], float(z['scale_full'])
    h, w = cv2.imread(str(bench.image_path(day, key))).shape[:2]
    W, H = w * sf, h * sf
    pts = np.array([[r['x'], r['y']], [r['x'] + r['w'], r['y']], [r['x'] + r['w'], r['y'] + r['h']], [r['x'], r['y'] + r['h']]]) * [W, H]
    c = (M @ np.c_[pts, np.ones(4)].T).T                                  # 1536-px chart
    if bool(z['mirrored']):
        c[:, 0] = 1535 - c[:, 0]
    if 'flipped_v' in z.files and bool(z['flipped_v']):
        c[:, 1] = 1535 - c[:, 1]
    return (c * size / 1536.0).astype(np.float32)


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else 'aliked1024'
    size = 1024
    dev = torch.device('cpu')
    ext = FE.extractor(variant, dev)
    lg = lightglue(variant, dev)
    L = json.load(open(MARKS))['labels']
    rows = []
    for r in L:
        if not r.get('regions') or not r.get('key_a') or not r.get('key_b'):
            continue
        ka, kb = r['key_a'], r['key_b']
        part = next((p for p in ('dorsal', 'fluke') if LM.meta(DAY, ka, p) and LM.meta(DAY, kb, p)), None)
        if part is None:
            continue
        A, wa, _ = LM.load_chart(DAY, ka, part, size)
        B, wb, _ = LM.load_chart(DAY, kb, part, size)
        fa = FE.run(ext, variant, A, wa, dev)
        best = None
        for mi in ((0, 1) if part == 'dorsal' else (0,)):
            Bm, wbm = (np.ascontiguousarray(B[:, ::-1]), np.ascontiguousarray(wb[:, ::-1])) if mi else (B, wb)
            fb = FE.run(ext, variant, Bm, wbm, dev)
            v0, v1 = torch.ones(1, len(fa['kp']), dtype=torch.bool), torch.ones(1, len(fb['kp']), dtype=torch.bool)
            m, sc = LM.lg_batch(lg, fa['kp'].float()[None], fa['desc'].float()[None], v0, fb['kp'].float()[None], fb['desc'].float()[None], v1, size)[0]
            pa, pb = fa['kp'].float().numpy()[m[:, 0]], fb['kp'].float().numpy()[m[:, 1]]
            st, inl = LM.verify(pa, pb, sc, float(wa.sum()), 0.012 * size)
            if inl is None and len(pa) >= 3:
                Ma, ia = cv2.estimateAffine2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=0.012 * size)
                inl = ia.ravel().astype(bool) if Ma is not None else None
            if best is None or (inl is not None and inl.sum() > best[3].sum()):
                best = (pa, pb, mi, inl if inl is not None else np.zeros(len(pa), bool), st)
        pa, pb, mi, inl, st = best
        polys_a = [box_poly(DAY, ka, part, g['a'], size) for g in r['regions']]
        polys_b = []
        for g in r['regions']:
            q = box_poly(DAY, kb, part, g['b'], size)
            if mi:
                q[:, 0] = size - 1 - q[:, 0]
            polys_b.append(q)
        ma = np.zeros((size, size), np.uint8); mb = np.zeros((size, size), np.uint8)
        for q in polys_a:
            cv2.fillPoly(ma, [q.astype(np.int32)], 1)
        for q in polys_b:
            cv2.fillPoly(mb, [q.astype(np.int32)], 1)
        wbm = wb[:, ::-1] if mi else wb
        dil = lambda m: cv2.dilate(m, np.ones((15, 15), np.uint8)) > 0
        ma_d, mb_d = dil(ma), dil(mb)
        chance_a = (ma_d & wa).sum() / max(wa.sum(), 1)
        chance_b = (mb_d & wbm).sum() / max(wbm.sum(), 1)
        ins = lambda p, m: m[np.clip(p[:, 1].astype(int), 0, size - 1), np.clip(p[:, 0].astype(int), 0, size - 1)]
        n_in = int(inl.sum())
        in_a = float(ins(pa[inl], ma_d).mean()) if n_in else np.nan
        in_b = float(ins(pb[inl], mb_d).mean()) if n_in else np.nan
        all_a = float(ins(pa, ma_d).mean()) if len(pa) else np.nan
        rows.append(dict(part=part, verdict=r.get('verdict'), catalog=r.get('catalog'), n=len(pa), inl=n_in, in_a=in_a, in_b=in_b, all_a=all_a,
                         chance_a=float(chance_a), chance_b=float(chance_b), cue=r.get('cue')))
        print(f"{part:6s} {r.get('verdict')!s:10s} matches {len(pa):4d} inliers {n_in:3d} in boxes A {in_a:.2f} B {in_b:.2f} (all matches A {all_a:.2f}) chance A {chance_a:.2f} B {chance_b:.2f} cue {r.get('cue')}", flush=True)
    for part in ('dorsal', 'fluke'):
        R = [x for x in rows if x['part'] == part]
        if not R:
            continue
        g = lambda k: np.nanmean([x[k] for x in R])
        has = [x for x in R if x['inl'] > 0]
        print(f'== {part}: {len(R)} marked photo pairs, {len(has)} with inliers; mean inliers {g("inl"):.1f}; share of inliers in the marked boxes '
              f'A {np.nanmean([x["in_a"] for x in has]):.2f} B {np.nanmean([x["in_b"] for x in has]):.2f}; all matches A {g("all_a"):.2f}; '
              f'chance (box share of the whale) A {g("chance_a"):.2f} B {g("chance_b"):.2f}')
    json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'marks_{variant}.json'), 'w'), indent=0)


if __name__ == '__main__':
    main()
