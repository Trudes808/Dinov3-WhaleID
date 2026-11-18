"""Sheet: canonical dorsal grey tiles with the mask outline (cyan) and the refined top-edge profile (red). Scratchpad."""
import sys

import cv2
import numpy as np

import score
from common import PXH, SHEETS, X0, Y0

day, n, off = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0
L = score.load('dorsal', day); Pf = score.profiles(day)
rows = []
for i in range(off, min(off + n, len(L['keys']))):
    g = cv2.cvtColor(L['z']['grey'][i], cv2.COLOR_GRAY2RGB)
    cs, _ = cv2.findContours((L['z']['mask'][i] > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(g, cs, -1, (0, 255, 255), 1)
    for x, y in zip(score.XS, Pf[i]):
        if np.isfinite(y):
            cv2.circle(g, (int((x - X0) * PXH), int((y - Y0) * PXH)), 2, (255, 0, 0), -1)
    info = L['info'][L['keys'][i]]
    cv2.putText(g, f"{i} H{info['H']:.0f} b{info['base']:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    rows.append(g)
while len(rows) % 3:
    rows.append(np.zeros_like(rows[0]))
grid = np.vstack([np.hstack(rows[i:i + 3]) for i in range(0, len(rows), 3)])
out = SHEETS / f'profile_{day}_{off}.jpg'; cv2.imwrite(str(out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80]); print(out)
