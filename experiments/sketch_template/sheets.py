"""sketch_template: sheets for the lead (21 train + 3 val days only; fin cut-outs and drawings, no people). Out of the
repository: moonshot_cache/sheets/sketch_template/.
  animals   a dozen animals with a noted shape: the fin cut-out of its best stack | the as-seen drawings of up to three
            stacks | the pad drawing (the noted template, the marks of all its stacks of that side carried onto it)
  pairs     two stacks on one shared template: true pairs seen at clearly different angles and different-animal pairs of
            one group: cut-out A | cut-out B | A as seen | B as seen | A on B's template | B on B's template | overlay
            (A's marks red, B's blue, both purple)
    python sheets.py animals|pairs [--n 12] [--seed 0]"""
import argparse
import collections
import json

import cv2
import numpy as np

import common as C

FS = C.FS
OUT = C.CACHE.parent / 'sheets' / 'sketch_template'


def text(img, s, y=20, sc=0.55, col=(0, 0, 0)):
    cv2.putText(img, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)
    return img


def cutout(day, keys, side):
    d = FS.day_dir(day)
    P = json.loads((d / 'sketch' / 'points.json').read_text())['points']
    best = None
    for k in keys:
        if k not in P:
            continue
        r = FS.cutout_view(d, k, P[k])
        if r is not None and (best is None or r[1]['H'] > best[1]['H']):
            best = r
    if best is None:
        return np.full((FS.DH, FS.DW, 3), 235, np.uint8)
    t = best[0]
    return np.ascontiguousarray(t[:, ::-1]) if side == 'R' else t


def framed(img, w=2, col=(160, 160, 160)):
    return cv2.copyMakeBorder(img, w, w, w, w, cv2.BORDER_CONSTANT, value=col)


def seen_drawing(v, side, extra=''):
    lab = f"as seen - {v['meta']['n']} fr - {FS.VIEW_TEXT.get(v['meta'].get('label', ''), '') or 'angle ?'}" + extra
    return FS.render_view(v, side, single=v['meta']['n'] < 2, label=lab)


def animals(n=12, seed=0):
    rng = np.random.default_rng(seed)
    cand = []
    for d in C.DAYS:
        N = C.noted(d)
        by = collections.defaultdict(list)
        for (c, s), v in C.drawings(d).items():
            a = v['meta']['animal']
            if a in N:
                by[(a, s)].append(v)
        for (a, s), vs in by.items():
            if len(vs) >= 3:
                cand.append((d, a, s, vs, N[a]))
    pick = [cand[i] for i in rng.permutation(len(cand))]
    seen_an, rows = set(), []
    for d, a, s, vs, shape in pick:
        if a in seen_an:
            continue
        seen_an.add(a)
        vs = sorted(vs, key=lambda v: -v['meta']['score'])
        pad, info = FS.pad_compose(vs, s, shape)
        cut = text(cutout(d, vs[0]['meta']['keys'], s).copy(), 'fin cut-out (best stack)', col=(255, 255, 255))
        tiles = [cut] + [seen_drawing(v, s) for v in vs[:3]]
        while len(tiles) < 4:
            tiles.append(np.full((FS.DH, FS.DW, 3), 255, np.uint8))
        ref = [x['why'] for x in info['stacks'] if not x['used']]
        tiles.append(text(pad, f"{info['used']} of {len(vs)} stacks carried over" + (f" ({', '.join(sorted(set(ref)))})" if ref else '') +
                          f"; {info['marks_solid']} solid, {info['marks_light']} light", y=40, sc=0.45, col=(40, 75, 150)))
        row = np.hstack([framed(t) for t in tiles])
        head = text(np.full((26, row.shape[1], 3), 255, np.uint8), f"{d} animal {len(rows) // 2 + 1} - {'left' if s == 'L' else 'right'} side - noted shape: {shape} - {len(vs)} stacks with a drawing of this side", y=18)
        rows += [head, row]
        if len(rows) // 2 >= n:
            break
    sheet = np.vstack(rows)
    title = np.full((50, sheet.shape[1], 3), 255, np.uint8)
    text(title, 'Pad drawings: fin cut-out | as-seen drawings of up to three stacks (the outline photographed) | the pad drawing: the NOTED shape\'s template (blue) with the marks of all the stacks of that side carried onto it', 20, 0.6)
    text(title, '(solid = found in two or more stacks, light = in one; hatched = skin no stack saw). Train + val days only; animals are not named here.', 42, 0.6)
    out = OUT / 'pad_animals.jpg'; OUT.mkdir(parents=True, exist_ok=True)
    img = np.vstack([title, sheet])
    cv2.imwrite(str(out), cv2.cvtColor(cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(out)


def overlay(tv, a, b, side):
    img = FS.render_view(dict(tv, seen=tv['whale']), 'L', contour_ink=FS.PAD_INK)
    k = np.ones((2 * FS.V_TOL + 1, 2 * FS.V_TOL + 1), np.uint8)
    am, bm = a & tv['whale'], b & tv['whale']
    img[am] = (215, 40, 40); img[bm] = (40, 90, 220); img[am & bm] = (150, 40, 170)
    # B within V_TOL of A's marks (what the hit test counts) as a thin ring
    ring = (cv2.dilate(bm.astype(np.uint8), k) > 0) & ~bm
    img[ring & ~am] = (195, 205, 240)
    if side == 'R':
        img = np.ascontiguousarray(img[:, ::-1])
    return text(img, 'A red, B blue, both purple (light blue = within the hit tolerance of B)', sc=0.45)


def pairs(n_same=5, n_diff=5, seed=1):
    rng = np.random.default_rng(seed)
    T = FS.pad_templates(C.CACHE / 'templates.npz')
    same, diff = [], []
    for d in C.DAYS:
        V = C.drawings(d); N = C.noted(d); Dd = C.bench.D(d)
        for a, b, sm in C.bench.pairs(d, 'dorsal'):
            for s in ('L', 'R'):
                if (a, s) not in V or (b, s) not in V:
                    continue
                shape = N.get(Dd.maj[b])
                if not shape:
                    continue
                va, vb = V[(a, s)], V[(b, s)]
                if va['meta']['n'] < 2 or vb['meta']['n'] < 2:
                    continue
                ang = abs(np.log(va['meta']['bperp'] / vb['meta']['bperp']))
                if sm and ang >= 0.4:
                    same.append((d, va, vb, s, shape, ang))
                elif not sm:
                    diff.append((d, va, vb, s, shape, ang))
    pick = [('same animal, different angles', x) for x in [same[i] for i in rng.permutation(len(same))][:n_same * 3]] + \
           [('different animals, one group', x) for x in [diff[i] for i in rng.permutation(len(diff))][:n_diff * 3]]
    rows, got = [], collections.Counter()
    for kind, (d, va, vb, s, shape, ang) in pick:
        if got[kind] >= (n_same if kind.startswith('same') else n_diff):
            continue
        tc = T[shape]['cur']; et = FS.pad_edges(tc); tv = FS.pad_template_view(tc)
        ra, ia = FS.pad_transfer(va, tc, et); rb, ib = FS.pad_transfer(vb, tc, et)
        if ra is None or rb is None:
            continue
        got[kind] += 1
        pa, _ = FS.pad_compose([va], s, shape, label=f"A on B's pad shape ({shape})")
        pb, _ = FS.pad_compose([vb], s, shape, label=f"B on its pad shape ({shape})")
        tiles = [text(cutout(d, va['meta']['keys'], s).copy(), 'A: fin cut-out', col=(255, 255, 255)), text(cutout(d, vb['meta']['keys'], s).copy(), 'B: fin cut-out', col=(255, 255, 255)),
                 seen_drawing(va, s, ' (A)'), seen_drawing(vb, s, ' (B)'), pa, pb, overlay(tv, ra['marks'] & ra['seen'], rb['marks'] & rb['seen'], s)]
        row = np.hstack([framed(t) for t in tiles])
        head = text(np.full((26, row.shape[1], 3), 255, np.uint8), f"{d} - {kind} - {'left' if s == 'L' else 'right'} side - base-ratio difference {ang:.2f} (log) - template: B's noted shape", y=18)
        rows += [head, row]
    sheet = np.vstack(rows)
    title = text(np.full((30, sheet.shape[1], 3), 255, np.uint8), "Two stacks on one shared template (B's noted shape): can the marks be compared place by place? Train + val days only.", 20, 0.6)
    out = OUT / 'pad_pairs.jpg'; OUT.mkdir(parents=True, exist_ok=True)
    img = np.vstack([title, sheet])
    cv2.imwrite(str(out), cv2.cvtColor(cv2.resize(img, None, fx=0.45, fy=0.45, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(out, dict(got))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['animals', 'pairs']); ap.add_argument('--n', type=int, default=12); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    cv2.setNumThreads(2)
    animals(a.n, a.seed) if a.what == 'animals' else pairs(seed=a.seed + 1)
