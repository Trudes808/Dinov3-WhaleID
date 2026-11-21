"""sketch_v2 step 0 (2026-11-19): the viewing-angle cues of every dorsal frame of the 21 train + 3 val days (no eval day).
Per instance of a safe cluster with a catalogue animal: fin height H (back base -> tip, source px), base = |front - back|
/ H, perpendicular height hp (tip to the base line), base_p = |front - back| / hp, tip offset along the base line (tx, in
hp, + = toward the tail), the angle between the base line and the cut-out's horizontal, and the parts_up fields profile,
head_angle, view_side, pose. Writes CACHE/geometry.json.     python geometry.py"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'moonshot_sizing'))
sys.path.insert(0, str(HERE.parents[1]))
import bench  # noqa: E402
from model_fine_tuning.cluster import fin_sketch as FS  # noqa: E402

CACHE = Path('/home/sat3737/NOAA_whales/moonshot_cache/sketch_aside/v2')


def rows(day):
    d = FS.day_dir(day)
    P = json.loads((d / 'sketch' / 'points.json').read_text())['points']
    Dd = bench.D(day)
    out = []
    for c, ks in bench.clusters(day, 'dorsal').items():
        for k in ks:
            if k not in P:
                continue
            p = P[k]
            zp = d / 'parts_up' / f'{k}__fin.npz'
            meta = {}
            if zp.exists():
                z = np.load(zp, allow_pickle=True)
                meta = dict(profile=float(z['profile']), head_angle=float(z['head_angle']) if 'head_angle' in z.files else None,
                            view_side=int(z['view_side']), pose=str(z['pose']) if 'pose' in z.files else None, mirrored=bool(z['mirrored']))
            s = FS.source(d, k)
            if s is None:
                continue
            img, m, f, src = s
            B, T, F = f(p['back']), f(p['tip']), f(p['front'])
            H = float(np.linalg.norm(T - B))
            bl = float(np.linalg.norm(F - B))
            u = (B - F) / max(bl, 1e-6)
            nrm = np.array([-u[1], u[0]])
            hp = abs(float((T - B) @ nrm))
            tx = float((T - B) @ u) / max(hp, 1e-6)
            ang = float(np.degrees(np.arctan2(u[1], abs(u[0]))))
            out.append(dict(day=day, key=k, cl=int(c), animal=Dd.maj[c], side=FS.side_of(p), src=src, H=H, base=bl / max(H, 1e-6),
                            hp=hp, base_p=bl / max(hp, 1e-6), tx=tx, axis_deg=ang, vis=p.get('vis'), **meta))
    return out


if __name__ == '__main__':
    from multiprocessing import Pool
    with Pool(8) as pool:
        R = [r for rs in pool.map(rows, bench.TRAIN + bench.VAL) for r in rs]
    CACHE.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(CACHE / 'geometry.json', 'w'))
    print(len(R), 'frames')
