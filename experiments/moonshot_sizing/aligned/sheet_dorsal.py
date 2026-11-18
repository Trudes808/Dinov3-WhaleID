"""Contact sheet of canonical dorsal tiles (scratchpad only): left = photo crop with the three points, right = tile with
mask outline, back base / tip markers and the canonical front base guide.  python sheet_dorsal.py [n] [seed] [name]"""
import sys

import cv2
import numpy as np

from common import SHEETS, TH, TW, PXH, X0, Y0, TIP_C, bench, dorsal_frame, keypoints, warp


def main(n=40, seed=0, name='dorsal_sheet', keys=None):
    rng = np.random.default_rng(seed)
    if keys is None:
        allk = [(d, k) for d in bench.TRAIN + bench.VAL for k, c, a in bench.instances(d, 'dorsal') if c is not None]
        keys = [allk[i] for i in rng.choice(len(allk), n, replace=False)]
    P, _ = keypoints()
    rows = []
    for d, k in keys:
        fr = dorsal_frame(k)
        if fr is None:
            continue
        M, info = fr
        t, m, inside = warp(d, k, M)
        t = t.copy()
        cs, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(t, cs, -1, (255, 255, 0), 1)
        for q, col in (((0, 0), (255, 0, 0)), (tuple(TIP_C), (0, 255, 0))):
            cv2.circle(t, (int((q[0] - X0) * PXH), int((q[1] - Y0) * PXH)), 4, col, 1)
        fx, fy = info['Fc']
        cv2.circle(t, (int((fx - X0) * PXH), int((fy - Y0) * PXH)), 4, (255, 0, 255), 1)
        # photo crop
        p = P[k]; pts = np.array([p['dorsal_fin_back'], p['dorsal_fin_tip'], p['dorsal_fin_front']])
        c = pts.mean(0); s = max(info['H'] * 3.0, 40)
        stem = k.rsplit('~', 1)[0]
        from common import _img
        im = _img(d, stem)
        x0, y0 = int(max(0, c[0] - s)), int(max(0, c[1] - s * 0.6)); x1, y1 = int(min(im.shape[1], c[0] + s)), int(min(im.shape[0], c[1] + s * 0.6))
        cr = im[y0:y1, x0:x1].copy()
        for q, col in zip(pts, ((255, 0, 0), (0, 255, 0), (255, 0, 255))):
            cv2.circle(cr, (int(q[0] - x0), int(q[1] - y0)), max(2, int(s / 60)), col, -1)
        cr = cv2.resize(cr, (int(TH * cr.shape[1] / max(1, cr.shape[0])), TH))[:, :TW]
        row = np.zeros((TH + 18, TW * 2 + 8, 3), np.uint8)
        row[:TH, :cr.shape[1]] = cr; row[:TH, TW + 8:] = t
        cv2.putText(row, f"{d} H{info['H']:.0f}px base{info['base']:.2f} mir{int(info['mirrored'])} hs{bench.D(d) and ''}", (2, TH + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        rows.append(row)
    cols = 2
    while len(rows) % cols:
        rows.append(np.zeros_like(rows[0]))
    grid = np.vstack([np.hstack(rows[i:i + cols]) for i in range(0, len(rows), cols)])
    SHEETS.mkdir(parents=True, exist_ok=True)
    out = SHEETS / f'{name}.jpg'
    cv2.imwrite(str(out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out, grid.shape)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40, int(sys.argv[2]) if len(sys.argv) > 2 else 0, sys.argv[3] if len(sys.argv) > 3 else 'dorsal_sheet')
