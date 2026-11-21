"""Quick look at the prepared cut-outs (blades up) with the mask outline, tip / notch points and the parts.py underside
flag. Sheet to the scratchpad only.   python look.py DAY [N]"""
import json
import sys

import cv2
import numpy as np

from prep import CACHE, SHEETS

day = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 48
z = np.load(CACHE / 'prep' / f'{day}.npz')
meta = json.load(open(CACHE / 'prep' / f'{day}.json'))
rng = np.random.default_rng(0)
idx = sorted(rng.choice(len(meta), min(N, len(meta)), replace=False))
tiles = []
for i in idx:
    im = z['im'][i].copy()
    w = (z['whale'][i] > 127).astype(np.uint8)
    f = (z['fluke'][i] > 127).astype(np.uint8)
    cs, _ = cv2.findContours(w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im, cs, -1, (0, 255, 0), 1)
    cs, _ = cv2.findContours(f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im, cs, -1, (255, 0, 255), 1)
    m = meta[i]
    p = np.array(m['pts'])
    for j, col in enumerate(((0, 0, 255), (255, 0, 0), (0, 255, 255))):
        if np.isfinite(p[j]).all():
            cv2.circle(im, tuple(int(v) for v in p[j]), 5, col, 2)
    txt = f"c{m['cluster']} {'UNDER' if m['underside'] else 'top'} fv{int(m['flipped_v'])} conf {m['conf'] if m['conf'] is None else round(m['conf'], 2)}"
    cv2.putText(im, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
    cv2.putText(im, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    tiles.append(cv2.resize(im, (256, 256)))
while len(tiles) % 8:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[k:k + 8]) for k in range(0, len(tiles), 8)]
out = SHEETS / f'look_{day}.jpg'
cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(out)
