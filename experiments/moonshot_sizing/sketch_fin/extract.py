"""sketch_fin step 1 (2026-11-18): every dorsal instance of the 21 train + 3 val days warped into one canonical fin frame.

Source: the RAW fin cut-out (parts_up/charts/<key>__fin.jpg, 1536 px, its own `whale` mask) when it exists, else the
2560-px photo + canon mask. Frame: the pose-side keypoints (back base, tip, front base; aligned/common.py), mapped into
the cut-out, similarity back base -> (0, 0), tip -> (-0.25, -1) in fin heights, mirrored so the head is always LEFT.
Tile x in [X0, X1], y in [Y0, Y1] fin heights at PXH px per fin height. No eval day is read (bench refuses them).

    OMP_NUM_THREADS=1 python extract.py [n_proc]
Writes CACHE/tiles/<day>.npz (keys, rgb, mask, inside) and <day>_info.json.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.append(str(HERE.parent / 'aligned'))
import bench  # noqa: E402
import common as AC  # noqa: E402

CACHE = Path('/home/sat3737/NOAA_whales/moonshot_cache/sketch_fin')
SHEETS = Path('/tmp/claude-1072903897/-home-sat3737-NOAA-whales/06180765-15d8-433b-8ac1-13d344adb66e/scratchpad/sketch_fin')
TIP_C = np.array([-0.25, -1.0])
X0, X1, Y0, Y1 = -3.2, 1.0, -1.3, 0.7
PXH = 96
TW, TH = int(round((X1 - X0) * PXH)), int(round((Y1 - Y0) * PXH))


def to_tile(q):
    return np.array([(q[0] - X0) * PXH, (q[1] - Y0) * PXH])


def frame(B, T, F):
    """similarity (2x3, maybe with reflection) source px -> tile px; mirrored flag; F in canonical units."""
    M = AC._sim([B, T], [to_tile([0, 0]), to_tile(TIP_C)])
    Fc = (M[:, :2] @ F + M[:, 2]) / PXH + np.array([X0, Y0])
    mirrored = bool(Fc[0] > 0)
    if mirrored:
        R = np.array([[-1.0, 0, 0], [0, 1.0, 0]])
        Ms = AC._sim([R[:, :2] @ B, R[:, :2] @ T], [to_tile([0, 0]), to_tile(TIP_C)])
        M = Ms @ np.vstack([R, [0, 0, 1]])
        Fc = (M[:, :2] @ F + M[:, 2]) / PXH + np.array([X0, Y0])
    return M, mirrored, Fc


def source(day, key):
    """(rgb, mask bool, point map 2560-px -> source px, source mirrored, name)."""
    zp = bench.DAYS / day / 'parts_up' / f'{key}__fin.npz'
    jp = bench.DAYS / day / 'parts_up' / 'charts' / f'{key}__fin.jpg'
    if zp.exists() and jp.exists():
        z = np.load(zp, allow_pickle=True)
        img = cv2.cvtColor(cv2.imread(str(jp)), cv2.COLOR_BGR2RGB)
        Mc, sf, mir = z['M'], float(z['scale_full']), bool(z['mirrored'])
        fv = bool(z['flipped_v']) if 'flipped_v' in z.files else False
        n = img.shape[0] - 1

        def f(p):
            c = Mc[:, :2] @ (np.asarray(p, float) * sf) + Mc[:, 2]
            c = c * img.shape[0] / 1536.0
            if mir:
                c[0] = n - c[0]
            if fv:
                c[1] = n - c[1]
            return c
        m = z['whale']
        if m.shape != img.shape[:2]:
            m = cv2.resize(m.astype(np.uint8), img.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
        return img, m, f, mir != fv, 'chart'      # an upside-down flip also swaps the apparent side
    stem = key.rsplit('~', 1)[0]
    img = cv2.cvtColor(cv2.imread(str(bench.image_path(day, key))), cv2.COLOR_BGR2RGB)
    m = cv2.imread(str(bench.DAYS / day / 'canon' / f'{stem}_mask.png'), cv2.IMREAD_GRAYSCALE)
    m = (m > 0) if m is not None else np.zeros(img.shape[:2], bool)
    return img, m, (lambda p: np.asarray(p, float)), False, 'image'


def one(day, key):
    P, META = AC.keypoints()
    p = P.get(key)
    if p is None:
        return None
    img, m, f, smir, src = source(day, key)
    B, T, F = (f(p[n]) for n in ('dorsal_fin_back', 'dorsal_fin_tip', 'dorsal_fin_front'))
    H = float(np.linalg.norm(T - B))
    if H < 4:
        return None
    M, mir, Fc = frame(B, T, F)
    down = abs(np.linalg.det(M[:, :2])) < 1
    t = cv2.warpAffine(img, M, (TW, TH), flags=cv2.INTER_AREA if down else cv2.INTER_CUBIC, borderValue=(0, 0, 0))
    mt = cv2.warpAffine(m.astype(np.uint8) * 255, M, (TW, TH), flags=cv2.INTER_LINEAR, borderValue=0)
    ins = cv2.warpAffine(np.full(img.shape[:2], 255, np.uint8), M, (TW, TH), flags=cv2.INTER_NEAREST, borderValue=0)
    H2560 = float(np.linalg.norm(np.array(p['dorsal_fin_tip']) - np.array(p['dorsal_fin_back'])))
    conf = META[key]['conf']
    info = dict(H=H, H2560=H2560, src=src, mirrored=bool(mir != smir), base=float(np.linalg.norm(F - B) / H), Fc=Fc.tolist(),
                conf=[conf['dorsal_fin_back'], conf['dorsal_fin_tip'], conf['dorsal_fin_front']])
    return key, t, mt, ins, info


def run_day(day):
    out = CACHE / 'tiles' / f'{day}.npz'
    if out.exists():
        return day
    K, R, Mk, I, INFO = [], [], [], [], {}
    for key, c, a in bench.instances(day, 'dorsal'):
        r = one(day, key)
        if r is None:
            continue
        K.append(r[0]); R.append(r[1]); Mk.append(r[2]); I.append(r[3]); INFO[r[0]] = r[4]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, keys=np.array(K), rgb=np.array(R, np.uint8), mask=np.array(Mk, np.uint8), inside=np.array(I, np.uint8))
    json.dump(INFO, open(CACHE / 'tiles' / f'{day}_info.json', 'w'))
    return day


if __name__ == '__main__':
    cv2.setNumThreads(1)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    days = sorted(bench.TRAIN + bench.VAL, key=lambda d: -len(bench.instances(d, 'dorsal')))
    with Pool(n) as pool:
        for d in pool.imap_unordered(run_day, days):
            print('done', d, flush=True)
    print('ALL DONE', flush=True)
