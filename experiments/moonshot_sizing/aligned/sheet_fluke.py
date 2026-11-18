"""Contact sheet of canonical fluke tiles (scratchpad only): photo crop with L (red) / R (green) / notch (magenta) and
the tile.  python sheet_fluke.py n seed name"""
import sys

import cv2
import numpy as np

import fluke
from common import SHEETS, _img, bench, warp

n, seed, name = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
P = fluke.points()
allk = [(d, k) for d in bench.TRAIN + bench.VAL for k, c, a in bench.instances(d, 'fluke') if c is not None and k in P]
rng = np.random.default_rng(seed)
rows = []
for i in rng.choice(len(allk), min(n, len(allk)), replace=False):
    d, k = allk[i]
    fr = fluke.fluke_frame(k)
    if fr is None:
        continue
    M, info = fr
    t, m, _ = warp(d, k, M, (fluke.TW, fluke.TH)); t = t.copy()
    cs, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(t, cs, -1, (255, 255, 0), 1)
    p = P[k]; pts = np.array([p['L'], p['R'], p['N']]); c = pts.mean(0); s = max(info['S'] * 1.6, 40)
    im = _img(d, k.rsplit('~', 1)[0])
    x0, y0 = int(max(0, c[0] - s)), int(max(0, c[1] - s * 0.8)); x1, y1 = int(min(im.shape[1], c[0] + s)), int(min(im.shape[0], c[1] + s * 0.8))
    cr = im[y0:y1, x0:x1].copy()
    for q, col in zip(pts, ((255, 0, 0), (0, 255, 0), (255, 0, 255))):
        cv2.circle(cr, (int(q[0] - x0), int(q[1] - y0)), max(2, int(s / 50)), col, -1)
    cr = cv2.resize(cr, (int(fluke.TH * cr.shape[1] / max(1, cr.shape[0])), fluke.TH))[:, :fluke.TW]
    row = np.zeros((fluke.TH + 18, fluke.TW * 2 + 8, 3), np.uint8)
    row[:fluke.TH, :cr.shape[1]] = cr; row[:fluke.TH, fluke.TW + 8:] = t
    cv2.putText(row, f"{d} S{info['S']:.0f} turned{int(info['turned'])} N{info['Nc'][0]:.2f},{info['Nc'][1]:.2f} 2nd{info['second']:.2f}", (2, fluke.TH + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    rows.append(row)
while len(rows) % 2:
    rows.append(np.zeros_like(rows[0]))
grid = np.vstack([np.hstack(rows[i:i + 2]) for i in range(0, len(rows), 2)])
out = SHEETS / f'{name}.jpg'; cv2.imwrite(str(out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80]); print(out)
