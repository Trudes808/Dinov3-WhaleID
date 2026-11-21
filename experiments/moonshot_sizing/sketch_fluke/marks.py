"""Do the drawn marks fall where the researcher marked her evidence? (2025-01-02, train day; CPU, read-only)

For every pair_marks.json record with marked region pairs whose photos have a drawable-or-not fluke drawing (any frame
that maps to the canvas): the boxes are mapped photo -> 1536 chart -> blades-up 384 -> canonical canvas. "Drawn mark"
pixels = seen pixels at a light / dark boundary of the drawing (local range of the blurred brightness > 0.35 in a 5 x 5
window) or light pixels (n > 0.5). Reported: share of the boxes' seen area that is drawn mark, against the share over the
whole seen fluke (chance).   python marks.py
"""
import json

import cv2
import numpy as np

import draw
import bench
from prep import CACHE

DAY = '2025-01-02'
MARKS = '/home/sat3737/NOAA_whales/Whale-Review-Web/data/2025-01-02/pair_marks.json'


def mark_map(seen, n):
    nb = cv2.GaussianBlur(n, (0, 0), 1.5)
    k = np.ones((5, 5), np.uint8)
    rng = cv2.dilate(nb, k) - cv2.erode(nb, k)
    return seen & ((rng > 0.35) | (nb > 0.5))


def main():
    arr, meta = draw.load(DAY)
    idx = {m['key']: i for i, m in enumerate(meta)}
    L = json.load(open(MARKS))['labels']
    rows = []
    cache = {}
    for r in L:
        if not r.get('regions') or r.get('key_a') not in idx or r.get('key_b') not in idx:
            continue                                     # both photos need a fluke cut-out
        for side in ('a', 'b'):
            key = r[f'key_{side}']
            if key not in idx:
                continue
            if key not in cache:
                ins = draw.instance(arr, meta[idx[key]], idx[key])
                cache[key] = ins if ins.get('M') is not None else None
            ins = cache[key]
            if ins is None:
                continue
            z = np.load(bench.DAYS / DAY / 'parts_up' / f'{key}__fluke.npz', allow_pickle=True)
            h, w = cv2.imread(str(bench.image_path(DAY, key))).shape[:2]
            sf = float(z['scale_full'])
            mk = mark_map(ins['seen'], ins['n'])
            box = np.zeros_like(ins['seen'])
            for g in r['regions']:
                q = g[side]
                pts = np.array([[q['x'], q['y']], [q['x'] + q['w'], q['y']], [q['x'] + q['w'], q['y'] + q['h']], [q['x'], q['y'] + q['h']]]) * [w * sf, h * sf]
                c = (z['M'] @ np.c_[pts, np.ones(4)].T).T
                if bool(z['flipped_v']):
                    c[:, 1] = 1535 - c[:, 1]
                c[:, 1] = 1535 - c[:, 1]
                c = c * 384 / 1536.0
                cc = (ins['M'][:, :2] @ c.T).T + ins['M'][:, 2]
                cv2.fillPoly(box.view(np.uint8), [cc.astype(np.int32)], 1)
            box = cv2.dilate(box.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            bs = box & ins['seen']
            if bs.sum() < 5:
                continue
            rows.append(dict(key=key, cue=r.get('cue'), drawable=bool(ins['ok']), frame=ins.get('frame'), in_box=float(mk[bs].mean()), chance=float(mk[ins['seen']].mean()),
                             box_share=float(bs.sum() / ins['seen'].sum())))
    for tag, R in (('all mapped photos', rows), ('frame from the tip points', [x for x in rows if x['frame'] == 'points'])):
        if not R:
            continue
        a = np.array([x['in_box'] for x in R]); c = np.array([x['chance'] for x in R])
        print(f'{tag}: {len(R)} marked photos (of {len({k for k in cache})} photos looked up); share of drawn-mark pixels inside the marked boxes {a.mean():.3f} '
              f'vs over the whole seen fluke {c.mean():.3f}; box > chance in {int((a > c).sum())} of {len(R)}; mean box share of the seen fluke {np.mean([x["box_share"] for x in R]):.3f}')
    json.dump(rows, open(draw.HERE / 'marks.json', 'w'), indent=0)


if __name__ == '__main__':
    main()
