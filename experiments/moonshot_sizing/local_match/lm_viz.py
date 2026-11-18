"""Look at what the matcher matches (CPU, a few pairs). Images go to the scratchpad only (crew can be visible).

    python lm_viz.py VARIANT PART SCORE N          top-N same-animal and top-N different-animal cluster pairs on val days
    python lm_viz.py marks                          researcher's marked regions (2025-01-02) against the inliers
"""
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm_lib as LM  # noqa: E402
import lm_score as SC  # noqa: E402
from lm_lib import bench  # noqa: E402
from lm_match import lightglue  # noqa: E402

OUT = '/tmp/claude-1072903897/-home-sat3737-NOAA-whales/06180765-15d8-433b-8ac1-13d344adb66e/scratchpad/local_match'
_F = {}
_LG = {}


def feats(variant, day, part):
    if (variant, day, part) not in _F:
        _F[(variant, day, part)] = torch.load(LM.CACHE / variant / f'{day}__{part}.pt', weights_only=False)
    return _F[(variant, day, part)]


def match_pair(variant, day, part, ka, kb, mirror, thr=0.012):
    F = feats(variant, day, part)
    if variant not in _LG:
        _LG[variant] = lightglue(variant, torch.device('cpu'))
    fa, fb = F['feats'][ka]['plain'], F['feats'][kb]['mirror' if mirror else 'plain']
    size = F['size']
    kp0, d0, kp1, d1 = fa['kp'].float()[None], fa['desc'].float()[None], fb['kp'].float()[None], fb['desc'].float()[None]
    v0, v1 = torch.ones(kp0.shape[:2], dtype=torch.bool), torch.ones(kp1.shape[:2], dtype=torch.bool)
    m, sc = LM.lg_batch(_LG[variant], kp0, d0, v0, kp1, d1, v1, size)[0]
    pa, pb = fa['kp'].float().numpy()[m[:, 0]], fb['kp'].float().numpy()[m[:, 1]]
    st, inl = LM.verify(pa, pb, sc, F['feats'][ka]['area'], thr * size)
    return pa, pb, sc, inl, st, size


def draw(day, part, ka, kb, mirror, pa, pb, inl, size, title):
    A, wa, _ = LM.load_chart(day, ka, part, size)
    B, wb, _ = LM.load_chart(day, kb, part, size)
    if mirror:
        B = np.ascontiguousarray(B[:, ::-1])
    can = np.hstack([A, B]).copy()
    can = cv2.cvtColor(can, cv2.COLOR_RGB2BGR)
    for i, (p, q) in enumerate(zip(pa, pb)):
        good = inl is not None and inl[i]
        col = (0, 255, 0) if good else (0, 0, 255)
        if good or len(pa) < 60:
            cv2.line(can, (int(p[0]), int(p[1])), (int(q[0]) + size, int(q[1])), col, 2 if good else 1)
    cv2.putText(can, title, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2)
    return cv2.resize(can, (size, size // 2))


def top_pairs(variant, part, score, n):
    R = SC.cluster_scores(variant, part, bench.VAL, 'max')[score]
    rows = []
    for d in bench.VAL:
        same = {(a, b): s for a, b, s in bench.pairs(d, part)}
        for p, v in R.get(d, {}).items():
            rows.append((v, d, p, same[p]))
    rows.sort(key=lambda r: -r[0])
    sheets = {True: [], False: []}
    for v, d, (a, b), s in rows:
        if len(sheets[s]) >= n:
            if all(len(x) >= n for x in sheets.values()):
                break
            continue
        # the instance pair that gave the cluster score
        z = np.load(LM.CACHE / variant / f'{d}__{part}__pairs.npz')
        J, S, names = z['jobs'], z['stats'], [str(x) for x in z['names']]
        keys = [str(k) for k in z['keys']]
        sel = (J[:, 0] == a) & (J[:, 1] == b)
        nk = np.array([len(feats(variant, d, part)['feats'][k]['plain']['kp']) for k in keys], float)
        vals = SC.inst_scores(J[sel], S[sel], names, nk)[score]
        j = J[sel][int(np.argmax(vals))]
        ka, kb, mi = keys[j[2]], keys[j[3]], int(j[4])
        pa, pb, sc, inl, st, size = match_pair(variant, d, part, ka, kb, mi)
        title = f"{'SAME' if s else 'DIFF'} {d} c{a}-c{b} {score}={v:.1f} n={st['n']} sim={st['sim']} hom={st['hom']}{' mirror' if mi else ''}"
        sheets[s].append(draw(d, part, ka, kb, mi, pa, pb, inl, size, title))
        print(title, flush=True)
    for s, ims in sheets.items():
        if ims:
            cv2.imwrite(f"{OUT}/top_{variant}_{part}_{score}_{'same' if s else 'diff'}.jpg", np.vstack(ims), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == '__main__':
    if sys.argv[1] == 'marks':
        pass
    else:
        top_pairs(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
