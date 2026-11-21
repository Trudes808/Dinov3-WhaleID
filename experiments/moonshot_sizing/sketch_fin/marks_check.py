"""Do the drawn marks fall where the researcher marked her evidence? (2025-01-02, a train day; CPU, read-only)

Every region box of Whale-Review-Web/data/2025-01-02/pair_marks.json on a photo that has a dorsal tile here is mapped
into that tile (2560-px photo -> cut-out -> keypoint frame). Measured per photo: share of the frame's drawn mark pixels
inside the (dilated) boxes against the share of the seen skin the boxes cover (= chance); per box: does it hold a drawn
mark, against the same box shifted to random places on the seen skin of that tile.

    python marks_check.py
"""
import json
from pathlib import Path

import cv2
import numpy as np

import extract as E
import bench
import draw as DR

DAY = '2025-01-02'
MARKS = '/home/sat3737/NOAA_whales/Whale-Review-Web/data/2025-01-02/pair_marks.json'


def tile_map(day, key):
    P, _ = E.AC.keypoints()
    p = P[key]
    img, m, f, smir, src = E.source(day, key)
    B, T, F = (f(p[n]) for n in ('dorsal_fin_back', 'dorsal_fin_tip', 'dorsal_fin_front'))
    M, mir, Fc = E.frame(B, T, F)
    return lambda q: M[:, :2] @ f(q) + M[:, 2]


def main(sheet=0):
    import sheet as SH
    tiles = []
    keys, z, info = DR.load_day(DAY)
    idx = {k: i for i, k in enumerate(keys)}
    L = json.load(open(MARKS))['labels']
    boxes = {}
    for r in L:
        if not r.get('regions'):
            continue
        for side in ('a', 'b'):
            k = r.get(f'key_{side}')
            if k in idx:
                boxes.setdefault(k, []).extend(g[side] for g in r['regions'] if side in g)
    rng = np.random.default_rng(0)
    pix_in, pix_ch, box_hit, box_ch, nbox, box_frac = [], [], [], [], 0, []
    for k, bx in boxes.items():
        i = idx[k]
        fr = DR.frame_drawing(z['rgb'][i], z['mask'][i], z['inside'][i])
        h, w = cv2.imread(str(bench.image_path(DAY, k))).shape[:2]
        mp = tile_map(DAY, k)
        bm = np.zeros((E.TH, E.TW), np.uint8)
        polys = []
        for b in bx:
            pts = np.array([[b['x'], b['y']], [b['x'] + b['w'], b['y']], [b['x'] + b['w'], b['y'] + b['h']], [b['x'], b['y'] + b['h']]]) * [w, h]
            q = np.array([mp(p) for p in pts])
            c = q.mean(0).astype(int)
            if not (0 <= c[0] < E.TW and 0 <= c[1] < E.TH) or not fr['whale'][c[1], c[0]]:
                continue
            polys.append(q)
            cv2.fillPoly(bm, [q.astype(np.int32)], 1)
        if not polys:
            continue
        bd = cv2.dilate(bm, np.ones((9, 9), np.uint8)) > 0
        if sheet:
            ph, dr = fr['rgb'].copy(), DR.render(dict(fr, nick_top=[], nick_trail=[]))
            for q in polys:
                for im in (ph, dr):
                    cv2.polylines(im, [q.astype(np.int32)], True, (255, 200, 0), 2)
            tiles.append(np.hstack([SH.label(ph, k[-12:]), dr]))
            if len(tiles) >= sheet:
                break
        seen, mk = fr['seen'], fr['marks']
        if mk.sum():
            pix_in.append((mk & bd).sum() / mk.sum())
            pix_ch.append((bd & seen).sum() / max(seen.sum(), 1))
        ys, xs = np.nonzero(seen)
        for q in polys:
            nbox += 1
            one = np.zeros_like(bm)
            cv2.fillPoly(one, [q.astype(np.int32)], 1)
            one = cv2.dilate(one, np.ones((9, 9), np.uint8)) > 0
            box_hit.append(bool((one & mk).any()))
            box_frac.append((one & seen).sum() / max(seen.sum(), 1))
            cy, cx = np.nonzero(one)
            hits = []
            for _ in range(50):
                j = rng.integers(len(ys))
                dy, dx = ys[j] - int(cy.mean()), xs[j] - int(cx.mean())
                yy, xx = np.clip(cy + dy, 0, E.TH - 1), np.clip(cx + dx, 0, E.TW - 1)
                hits.append(bool(mk[yy, xx].any()))
            box_ch.append(np.mean(hits))
    if sheet:
        out = E.SHEETS / f'researcher_boxes_{DAY}.jpg'
        img = np.vstack(tiles)
        cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(out)
        return
    print(f'{len(pix_in)} photos with drawn marks and researcher boxes on the fin tile; {nbox} boxes')
    print(f'share of drawn mark pixels inside her boxes: {np.mean(pix_in):.3f}  (chance = box share of the seen skin {np.mean(pix_ch):.3f}; ratio {np.mean(pix_in) / np.mean(pix_ch):.2f})')
    print(f'boxes holding a drawn mark: {np.mean(box_hit):.3f}  (same box at random places on the seen skin {np.mean(box_ch):.3f})')
    bf, bh, bc = np.array(box_frac), np.array(box_hit, float), np.array(box_ch)
    sm = bf < 0.1
    print(f'small boxes (< 10 % of the seen skin): {int(sm.sum())}; holding a drawn mark {bh[sm].mean():.3f} (random placement {bc[sm].mean():.3f})')
    json.dump(dict(small_boxes=int(sm.sum()), small_hit=float(bh[sm].mean()), small_chance=float(bc[sm].mean()), photos=len(pix_in), boxes=nbox, pix_in=float(np.mean(pix_in)), pix_chance=float(np.mean(pix_ch)),
                   box_hit=float(np.mean(box_hit)), box_chance=float(np.mean(box_ch))), open(Path(__file__).with_name('marks_check.json'), 'w'), indent=1)


if __name__ == '__main__':
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
