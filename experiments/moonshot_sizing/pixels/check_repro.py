"""pixels: per-block cosine between my sq / ar rgb layer-11 cls + mean and the live view_feat.npz (BASE)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bench  # noqa: E402
import run  # noqa: E402

for day in sys.argv[1:] or ['2025-01-15']:
    z = np.load(bench.DAYS / day / 'safe' / 'view_feat.npz')
    base = {str(k): x for k, x in zip(z['keys'], z['X'].astype(np.float32).reshape(-1, 4, 768))}
    for i, (v, pool) in enumerate([('sq', 'cls'), ('sq', 'mean'), ('ar', 'cls'), ('ar', 'mean')]):
        mine = run.block('base', v, 'rgb', 11, pool, 'dorsal', day)
        c = [float(mine[k] @ base[k][i] / np.linalg.norm(base[k][i])) for k in mine if k in base]
        print(f'{day} {v} {pool}: n {len(c)}  cosine median {np.median(c):.4f}  min {np.min(c):.4f}  p5 {np.percentile(c, 5):.4f}')
