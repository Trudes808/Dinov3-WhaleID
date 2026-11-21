"""Sheets "photo | drawing" (scratchpad only).
    python sheet.py light            instances with the most light area (side-cue check)
    python sheet.py clusters DAY     cluster consensus drawings of one day: frames | consensus drawing
"""
import sys

import cv2
import numpy as np

import draw
from prep import SHEETS


def render(seen, n, stable=None, scale=2):
    """Pen-and-paper style: paper white, unseen fluke area not drawn, dark skin black hatching-free fill (grey), light
    pigment left white, outline of the seen fluke in black. Unstable pixels (frames disagree) faded."""
    H, W = seen.shape
    img = np.full((H, W, 3), 255, np.uint8)
    nb = cv2.GaussianBlur(n, (0, 0), 1.5)
    dark = seen & (nb < 0.35)
    mid = seen & (nb >= 0.35) & (nb <= 0.5)
    img[dark] = (60, 60, 60)
    img[mid] = (170, 170, 170)
    if stable is not None:
        weak = seen & (stable < 0.67)
        img[weak] = (img[weak] * 0.4 + np.array([200, 220, 255]) * 0.6).astype(np.uint8)   # pale orange = unsure
    img = cv2.resize(img, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours(cv2.resize(seen.astype(np.uint8), (W * scale, H * scale), interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, (0, 0, 0), 2)
    for X in (-1, 1):
        x, y = draw.to_px(X, 0)
        cv2.circle(img, (int(x * scale), int(y * scale)), 4, (0, 0, 255), 1)
    return img


def label(img, txt):
    cv2.putText(img, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 3)
    cv2.putText(img, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    return img


def grid(tiles, cols):
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    return np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)])


if __name__ == '__main__':
    import bench
    if sys.argv[1] == 'light':
        out = []
        for d in bench.VAL + bench.TRAIN[:6]:
            for r in draw.day_instances(d):
                if r.get('seen') is not None and r['seen'].sum() > 500:
                    out.append((float((r['n'][r['seen']] > 0.5).mean()), d, r))
        out.sort(key=lambda t: -t[0])
        tiles = []
        for li, d, r in out[:40]:
            ph = cv2.resize(r['im'], (512, 256))
            txt = f"{d[5:]} c{r['cluster']} light {li:.2f} notch {r['notch']:+.2f} CSE {'under' if r['underside'] else 'top'} {'OK' if r['ok'] else r['why']}"
            tiles.append(np.hstack([label(ph, txt), render(r['seen'], r['n'])]))
        cv2.imwrite(str(SHEETS / 'light_check.jpg'), grid(tiles, 2), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(SHEETS / 'light_check.jpg')


def tmpl_overlay(img, scale=2):
    import score as SC
    cs, _ = cv2.findContours(cv2.resize(SC.T.astype(np.uint8), (draw.W * scale, draw.H * scale), interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, (0, 160, 0), 1)
    return img


def sheet_random(days, name, sel=lambda r: r['ok'], k=40, seed=0):
    import score as SC
    tiles = []
    pool = []
    for d in days:
        meta, seen, n, im = SC.inst(d)
        pool += [(d, i) for i, r in enumerate(meta) if sel(r)]
    rng = np.random.default_rng(seed)
    for j in rng.choice(len(pool), min(k, len(pool)), replace=False):
        d, i = pool[j]
        meta, seen, n, im = SC.inst(d)
        r = meta[i]
        txt = f"{d[5:]} c{r['cluster']} {r['side']} p{r['prec']:.2f} r{r['rec']:.2f} {r.get('frame')} {'DRAW' if r['drawable'] else r['why'] or '-'}"
        tiles.append(np.hstack([label(cv2.resize(im[i], (512, 256)), txt), tmpl_overlay(render(seen[i], n[i]))]))
    cv2.imwrite(str(SHEETS / f'{name}.jpg'), grid(tiles, 2), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(SHEETS / f'{name}.jpg')


def cluster_tile(d, c, side, w=512):
    """best frame photo (canonical) | consensus drawing, for one cluster."""
    import score as SC
    meta, seen, n, im = SC.inst(d)
    cons = SC.cluster_drawings(d)[c][side]
    k = {r['key']: i for i, r in enumerate(meta)}
    i = max((k[x] for x in cons['keys']), key=lambda i: seen[i].sum())
    ph = cv2.resize(im[i], (w, w // 2))
    dr = render(cons['seen'], cons['n'], cons['stable'], scale=w // draw.W)
    return ph, dr, cons


def sheet_pairs(sel, name, k=16, seed=0):
    """sel: list of (day, a, b, same, side, extra text)"""
    rng = np.random.default_rng(seed)
    pick = sel if len(sel) <= k else [sel[j] for j in sorted(rng.choice(len(sel), k, replace=False))]
    tiles = []
    for d, a, b, same, side, txt in pick:
        pa, da, ca = cluster_tile(d, a, side, 512)
        pb, db, cb = cluster_tile(d, b, side, 512)
        label(pa, f"{d[5:]} c{a} {'SAME' if same else 'DIFF'} {txt}")
        label(da, f"class {ca['cls']} light {ca['light']:.2f} {ca['frames']} fr")
        label(db, f"class {cb['cls']} light {cb['light']:.2f} {cb['frames']} fr")
        label(pb, f"c{b}")
        tiles.append(np.hstack([pa, da, np.full((256, 8, 3), 255, np.uint8), pb, db]))
    out = np.vstack([np.vstack([t, np.full((6, t.shape[1], 3), 128, np.uint8)]) for t in tiles])
    cv2.imwrite(str(SHEETS / f'{name}.jpg'), out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(SHEETS / f'{name}.jpg')


def sheet_clusters(days, name, k=16):
    """researcher sheet: per cluster, the best frame (canonical) | the consensus drawing; most-seen clusters first."""
    import score as SC
    items = []
    for d in days:
        for c, v in SC.cluster_drawings(d).items():
            if 'any' in v:
                items.append((v['any']['rec'], d, c, v['any']))
    items.sort(key=lambda t: -t[0])
    tiles = []
    for rec, d, c, cons in items[:k]:
        ph, dr, _ = cluster_tile(d, c, 'any', 512)
        label(ph, f"{d} cluster {c}")
        label(dr, f"{cons['frames']} frame(s); class {cons['cls']} (1 dark .. 5 white)")
        tiles.append(np.hstack([ph, dr]))
    cv2.imwrite(str(SHEETS / f'{name}.jpg'), grid(tiles, 2), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(SHEETS / f'{name}.jpg')
