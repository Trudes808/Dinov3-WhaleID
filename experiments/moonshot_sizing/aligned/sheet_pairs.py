"""Sheet: best-aligned instance pair (posc) of same-animal cluster pairs > 60 s apart, tiles side by side. Scratchpad."""
import sys

import cv2
import numpy as np

import run_pairs
import score
from common import SHEETS, bench

day, n, want = sys.argv[1], int(sys.argv[2]), sys.argv[3] == 'same'
S, _ = run_pairs.matrix('dorsal', day, 'posc')
L = score.load('dorsal', day); idx = {k: i for i, k in enumerate(L['keys'])}; keep = score.quality('dorsal', day)
cl = {c: [idx[k] for k in ks if k in idx and keep[idx[k]]] for c, ks in bench.clusters(day, 'dorsal').items()}
rng = np.random.default_rng(0); prs = [p for p in bench.pairs(day, 'dorsal') if p[2] == want and cl.get(p[0]) and cl.get(p[1])]
rows = []
for a, b, _ in [prs[i] for i in rng.choice(len(prs), min(n, len(prs)), replace=False)]:
    sub = S[np.ix_(cl[a], cl[b])]
    if np.all(np.isnan(sub)):
        continue
    r, c = np.unravel_index(np.nanargmax(sub), sub.shape); i, j = cl[a][r], cl[b][c]
    t = [cv2.cvtColor(L['z']['grey'][x], cv2.COLOR_GRAY2RGB) for x in (i, j)]
    row = np.hstack([t[0], np.zeros((t[0].shape[0], 6, 3), np.uint8), t[1]])
    cv2.putText(row, f"{a}|{b} posc {np.nanmax(sub):.2f} flank {sorted(bench.flank(day, a))}/{sorted(bench.flank(day, b))}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
    rows.append(row)
grid = np.vstack(rows)
out = SHEETS / f'pairs_{day}_{sys.argv[3]}.jpg'; cv2.imwrite(str(out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75]); print(out)
