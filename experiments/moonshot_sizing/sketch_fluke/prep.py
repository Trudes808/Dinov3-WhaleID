"""sketch_fluke step 1 (2026-11-18): gather every fluke instance of the 21 train + 3 val days that has a parts_up cut-out,
at 384 px, blades UP (the chart is flipped vertically: peduncle down, whale's right tip on the image right as in
parts.py), with the whale / fluke masks, the parts.py view flags and the aligned agent's tip / notch points mapped into
the same 384-px frame. CPU, read-only on the day folders; arrays go to moonshot_cache/sketch_fluke/prep/<day>.npz.

    python prep.py            (all days; skips days already written)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bench  # noqa: E402

CACHE = Path('/home/sat3737/NOAA_whales/moonshot_cache/sketch_fluke')
SHEETS = Path('/tmp/claude-1072903897/-home-sat3737-NOAA-whales/06180765-15d8-433b-8ac1-13d344adb66e/scratchpad/sketch_fluke')
PTS = Path('/home/sat3737/NOAA_whales/moonshot_cache/aligned/fluke_points.json')
S = 384
DAYS = bench.TRAIN + bench.VAL


def chart_files(day, key):
    pu = bench.DAYS / day / 'parts_up'
    return pu / 'charts' / f'{key}__fluke.jpg', pu / f'{key}__fluke.npz'


def prep_day(day, P):
    keys, ims, wh, fl, meta = [], [], [], [], []
    k2c = {k: (c, a) for k, c, a in bench.instances(day, 'fluke')}
    for key, (c, a) in k2c.items():
        fj, fz = chart_files(day, key)
        if not fz.exists() or not fj.exists():
            continue
        z = np.load(fz, allow_pickle=True)
        im = cv2.imread(str(fj), cv2.IMREAD_REDUCED_COLOR_4)
        if im is None or im.shape[:2] != (S, S):
            im = cv2.resize(cv2.imread(str(fj)), (S, S), interpolation=cv2.INTER_AREA)
        w = cv2.resize(z['whale'].astype(np.uint8) * 255, (S, S), interpolation=cv2.INTER_AREA)
        f = cv2.resize(z['fluke'].astype(np.uint8) * 255, (S, S), interpolation=cv2.INTER_AREA)
        # blades up: undo the chart's peduncle-up convention (vertical flip only)
        im, w, f = im[::-1], w[::-1], f[::-1]
        M, sf, flipped = z['M'], float(z['scale_full']), bool(z['flipped_v'])
        pts = np.full((3, 2), np.nan)
        r = P.get(key)
        if r is not None:
            q = np.array([r['L'], r['R'], r['N']], float) * sf                  # 2560 px -> full RAW
            c1536 = (M @ np.c_[q, np.ones(3)].T).T
            if flipped:
                c1536[:, 1] = 1535 - c1536[:, 1]
            c1536[:, 1] = 1535 - c1536[:, 1]                                   # our blades-up flip
            pts = c1536 * S / 1536.0
        # box in the 2560 image and the chart scale (for "how big was the fluke in the photo")
        keys.append(key); ims.append(im); wh.append(w); fl.append(f)
        meta.append(dict(key=key, cluster=c, animal=a, underside=bool(z['underside']), flipped_v=flipped, pixels=int(z['pixels']),
                         scale=float(np.hypot(M[0, 0], M[0, 1])), pts=pts.tolist(), conf=(r or {}).get('conf'), second=(r or {}).get('second')))
    np.savez_compressed(CACHE / 'prep' / f'{day}.npz', im=np.array(ims, np.uint8).reshape(-1, S, S, 3), whale=np.array(wh, np.uint8).reshape(-1, S, S),
                        fluke=np.array(fl, np.uint8).reshape(-1, S, S))
    json.dump(meta, open(CACHE / 'prep' / f'{day}.json', 'w'))
    return len(keys)


if __name__ == '__main__':
    (CACHE / 'prep').mkdir(parents=True, exist_ok=True)
    P = json.load(open(PTS))
    for d in DAYS:
        if (CACHE / 'prep' / f'{d}.json').exists():
            continue
        print(d, prep_day(d, P), flush=True)
    open(CACHE / 'prep' / 'DONE', 'w').write('ok')
