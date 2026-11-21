"""Sheets "photo | drawing" (images go to the scratchpad only, never the repo).

    python sheet.py frames <day> <cluster>          every frame of one safe cluster: photo | frame drawing, + consensus
    python sheet.py animal <day> <n_animals> [seed] per animal of the day: its clusters, photo | consensus drawing
"""
import collections
import sys

import cv2
import numpy as np

import extract  # noqa: F401  (puts the bench on sys.path)
import bench
import draw as DR
from extract import SHEETS, TH, TW


def label(img, text, scale=0.5):
    img = img.copy()
    cv2.putText(img, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 0), 1, cv2.LINE_AA)
    return img


def day_frames(day):
    keys, z, info = DR.load_day(day)
    idx = {k: i for i, k in enumerate(keys)}
    return keys, z, info, idx


def cluster_drawing(day, c, z, info, idx, keys_of):
    ks = [k for k in keys_of if k in idx]
    if not ks:
        return None
    ii = [idx[k] for k in ks]
    ks, fr, cons, side, ref = DR.cluster_drawing((z['rgb'][ii], z['mask'][ii], z['inside'][ii]), info, ks)
    return ks, fr, cons, ref, side


def render_clean(cons, H=100, base=2.0):
    """render with the clean-up the comparison uses (largest mark components only, trailing-edge nicks)."""
    import score as SC
    c = SC.clean(dict(cons, H=H, base=base, raw=cons['raw'], seen_n=cons['seen_n']))
    return DR.render(dict(cons, marks=c['mk'], nick_top=[], nick_trail=c['nk']))


def pairs_sheet(days, n_same=5, n_diff=5, seed=0):
    """same-animal pairs > 1 h apart and different-animal pairs of one group, both fins seen well: photo | drawing."""
    import pickle
    import score as SC
    rng = np.random.default_rng(seed)
    cand = {True: [], False: []}
    for day in days:
        P = pickle.load(open(SC.CACHE / 'pairs' / f'{day}.pkl', 'rb'))
        Dd = bench.D(day)
        for p, f in P.items():
            if f['well'] and f['sameside'] and f['na'] >= 2 and f['nb'] >= 2 and (not f['same'] or Dd.gap(*p) > 3600):
                cand[f['same']].append((day, p, f))
    blocks = []
    for same, n in ((True, n_same), (False, n_diff)):
        pick = [cand[same][i] for i in rng.permutation(len(cand[same]))[:n]]
        for day, (a, b), f in pick:
            keys, z, info, idx = day_frames(day)
            cl = bench.clusters(day, 'dorsal')
            Dd = bench.D(day)
            row = []
            for c in (a, b):
                ks, fr, cons, best, side = cluster_drawing(day, c, z, info, idx, cl[c])
                row += [label(fr[best]['rgb'], f'{day} c{c}'), render_clean(cons)]
            head = np.full((22, row[0].shape[1] * 4, 3), 255, np.uint8)
            txt = (f'{"SAME animal" if same else "DIFFERENT animals"} {Dd.maj[a] if same else Dd.maj[a] + " / " + Dd.maj[b]}  gap {Dd.gap(a, b) / 60:.0f} min  '
                   f'contour dist {f["dc"]:.3f} H  mark corr {f.get("mark_ncc") or 0:.2f}  missing marks {f.get("miss")}')
            cv2.putText(head, txt, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            blocks += [head, np.hstack(row), np.zeros((4, row[0].shape[1] * 4, 3), np.uint8)]
    img = np.vstack(blocks)
    out = SHEETS / f'pairs_{"_".join(days)}_s{seed}.jpg'
    img = cv2.resize(img, (img.shape[1] * 3 // 4, img.shape[0] * 3 // 4), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out)


def frames_sheet(day, c):
    keys, z, info, idx = day_frames(day)
    ks, fr, cons, best, side = cluster_drawing(day, c, z, info, idx, bench.clusters(day, 'dorsal')[c])
    rows = []
    for k, f in zip(ks, fr):
        f2 = dict(f, nick_top=[], nick_trail=DR.nicks(f['trail'], -1)[0])
        rows.append(np.hstack([label(f['rgb'], f"H {info[k]['H']:.0f}px ncc {f.get('ncc', 1):.2f}"), DR.render(f2)]))
    rows.append(np.hstack([label(DR.render(cons, faint=cons['raw'] >= 1), 'all marks (grey = seen in one frame)'), label(render_clean(cons), f'consensus drawing of {cons["n"]} frames')]))
    out = SHEETS / f'frames_{day}_c{c}.jpg'
    cv2.imwrite(str(out), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(out)


def animal_sheet(day, n_an, seed=0, max_cl=6):
    keys, z, info, idx = day_frames(day)
    Dd = bench.D(day)
    cl = bench.clusters(day, 'dorsal')
    by = collections.defaultdict(list)
    for c in cl:
        by[Dd.maj[c]].append(c)
    rng = np.random.default_rng(seed)
    an = [a for a in by if len(by[a]) >= 3]
    rng.shuffle(an)
    blocks = []
    for a in an[:n_an]:
        cs = sorted(by[a], key=lambda c: -len(cl[c]))[:max_cl]
        row = []
        for c in cs:
            r = cluster_drawing(day, c, z, info, idx, cl[c])
            if r is None:
                continue
            ks, fr, cons, best, side = r
            ph = label(fr[best]['rgb'], f'c{c} side {"B" if side else "A"} {Dd.C[c]["t0"] if "t0" in Dd.C[c] else ""}')
            row.append(np.vstack([ph, label(render_clean(cons), f'{cons["n"]} frames')]))
        if row:
            blocks.append(np.hstack(row + [np.full_like(row[0], 255)] * (max_cl - len(row))))
            blocks.append(np.zeros((6, blocks[-1].shape[1], 3), np.uint8))
    out = SHEETS / f'animals_{day}_s{seed}.jpg'
    img = np.vstack(blocks)
    img = cv2.resize(img, (img.shape[1] * 2 // 3, img.shape[0] * 2 // 3), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out)


if __name__ == '__main__':
    if sys.argv[1] == 'frames':
        frames_sheet(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == 'pairs':
        pairs_sheet(sys.argv[2].split(','), seed=int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    else:
        animal_sheet(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 0)
