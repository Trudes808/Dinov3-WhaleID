"""sketch_template B (2026-11-19): does carrying the marks onto the NOTED shape's template make two sightings' marks
comparable place by place? The measure of sketch_v2/mark_size.py: for two drawings of the same side of two stacks of one
catalogue group more than 60 s apart (bench.pairs), every DRAWN mark of the first (>= MARK_D fin heights) that falls
where the second saw the skin is checked: does the second have a mark (drawn or small) within V_TOL px? Same animal
against different animals; the lift (hit_same - hit_diff) is the "extra hit rate". 21 train + 3 val days only.

The template is the one of the SECOND drawing's animal (as in the page: a stack not yet assigned is carried onto the
candidate whale's template); a direction whose second animal has no noted shape is skipped for every frame alike.

frames (each drawing mapped on its own, then compared place by place, unless "+reg"):
  fixed            the fin_sketch picture frame as it is (tip and back base at the same place; no transfer)
  fixed+reg        the second drawing lined up onto the first (fin_sketch.register_view: stretch along the body, scale,
                   shift; what the animal-level drawing does today, mark_size.py)
  outline          fin_sketch.pad_maps onto the noted template: a point's place relative to the observed outline (height
                   kept; between the leading and trailing edge at that height; ahead / behind scaled by the width ratio)
  outline_generic  the same onto the template of ALL animals (is it the noted shape, or the outline-relative mapping?)
  outline+reg      outline, then the second lined up onto the first in the template frame
  stretch          one stretch along the body about the tip, the fin's width ratio (template / observed)
  similarity       a similarity fit of the observed outline to the template outline (iterated closest points)
  affine           an affine fit, likewise
A drawing whose transfer is refused (fin_sketch.pad_edges / the width ratio) stays as it is (the fixed frame) in every
template frame; the rows say whether either drawing was refused.

    python transfer.py           -> transfer.json (aggregates), CACHE/transfer_rows.npz (rows)
"""
import collections
import functools
import json
import sys
from multiprocessing import Pool

import cv2
import numpy as np
from scipy.spatial import cKDTree

import common as C

FS = C.FS
import os
METHODS = tuple(os.environ['TRANSFER_METHODS'].split(',')) if os.environ.get('TRANSFER_METHODS') else \
    ('fixed', 'fixed+reg', 'outline', 'outline_generic', 'outline+reg', 'stretch', 'stretch_half', 'kept', 'kept_v', 'similarity', 'affine')
TAG = os.environ.get('TRANSFER_TAG', '')
TEMPLATES = None


def templates():
    global TEMPLATES
    if TEMPLATES is None:
        TEMPLATES = FS.pad_templates(C.CACHE / 'templates.npz')
    return TEMPLATES


@functools.lru_cache(None)
def t_edges(shape):
    return FS.pad_edges(templates()[shape]['cur'])


@functools.lru_cache(None)
def t_points(shape):
    """the template outline as points (picture px): the top (u >= -2.6) and the trailing edge."""
    cur = templates()[shape]['cur']
    P = []
    ok = ~np.isnan(cur['top']) & (FS.PAD_UG >= -2.6)
    P.append(np.stack([FS.PAD_UG[ok], cur['top'][ok]], 1))
    ok = ~np.isnan(cur['trail'])
    P.append(np.stack([cur['trail'][ok], FS.PAD_VG[ok]], 1))
    p = np.concatenate(P)
    return np.stack([(p[:, 0] - FS.DU0) * FS.DPX, (p[:, 1] - FS.DV0) * FS.DPX], 1)


def o_points(cur):
    P = []
    ok = ~np.isnan(cur['top']) & (FS.PAD_UG >= -2.6)
    P.append(np.stack([FS.PAD_UG[ok], cur['top'][ok]], 1))
    ok = ~np.isnan(cur['trail'])
    P.append(np.stack([cur['trail'][ok], FS.PAD_VG[ok]], 1))
    p = np.concatenate(P)
    return np.stack([(p[:, 0] - FS.DU0) * FS.DPX, (p[:, 1] - FS.DV0) * FS.DPX], 1)


def icp(src, dst, kind, it=25, trim=0.8):
    """2x3 (src px -> dst px): iterated closest points, similarity (Umeyama) or affine (least squares), the worst 20 %
    of the matches left out each round."""
    tree = cKDTree(dst)
    M = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    for _ in range(it):
        s = src @ M[:, :2].T + M[:, 2]
        d, j = tree.query(s)
        keep = d <= np.quantile(d, trim)
        a, b = src[keep], dst[j[keep]]
        if kind == 'affine':
            X = np.hstack([a, np.ones((len(a), 1))])
            sol, *_ = np.linalg.lstsq(X, b, rcond=None)
            M = sol.T
        else:
            ma, mb = a.mean(0), b.mean(0); A, B = a - ma, b - mb
            U, S, Vt = np.linalg.svd(B.T @ A / len(a))
            D = np.diag([1, np.sign(np.linalg.det(U @ Vt))])
            R = U @ D @ Vt; sc = np.trace(np.diag(S) @ D) / max((A ** 2).sum() / len(a), 1e-9)
            M = np.hstack([sc * R, (mb - sc * R @ ma)[:, None]])
    return M


def warp(a, M):
    return cv2.warpAffine(a.astype(np.uint8), M.astype(np.float32), (FS.DW, FS.DH), flags=cv2.INTER_NEAREST) > 0


def warp_lab(lab, M=None, maps=None):
    if maps is not None:
        return cv2.remap(lab.astype(np.float32), maps[0], maps[1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.int32)
    return cv2.warpAffine(lab.astype(np.float32), M.astype(np.float32), (FS.DW, FS.DH), flags=cv2.INTER_NEAREST).astype(np.int32)


class Drawing:
    def __init__(self, v):
        self.v = v
        self.cur = FS.pad_curves(v)
        self.e = FS.pad_edges(self.cur)
        am = v['marks'] & v['seen']
        n, self.lab, st, _ = cv2.connectedComponentsWithStats(am.astype(np.uint8), connectivity=8)
        self.n = n
        self.dia = np.sqrt(4 * st[:, cv2.CC_STAT_AREA] / FS.DPX ** 2 / np.pi)
        dep = cv2.distanceTransform(v['whale'].astype(np.uint8), cv2.DIST_L2, 3) / FS.DPX
        idx = self.lab.ravel(); ok = idx > 0
        cnt = np.maximum(np.bincount(idx[ok], minlength=n), 1)
        self.depth = np.bincount(idx[ok], weights=dep.ravel()[ok], minlength=n) / cnt          # mean depth inside the outline (H)
        yy, xx = np.mgrid[:FS.DH, :FS.DW]
        self.cu = np.bincount(idx[ok], weights=xx.ravel()[ok], minlength=n) / cnt / FS.DPX + FS.DU0
        self.cv = np.bincount(idx[ok], weights=yy.ravel()[ok], minlength=n) / cnt / FS.DPX + FS.DV0
        self.tgt = v['marks'] | v['small']
        self._m = {}

    def refused(self, shape):
        if not self.e['ok']:
            return self.e['why']
        k = FS.pad_ratio(self.e, t_edges(shape))
        if not FS.PAD_K_RANGE[0] <= k <= FS.PAD_K_RANGE[1]:
            return 'width ratio out of range'
        return ''

    def mapped(self, method, shape):
        """(labels, target, seen, view-like dict for registration) in the method's frame for this template."""
        key = (method, shape)
        if key in self._m:
            return self._m[key]
        v = self.v
        base = method.replace('+reg', '')
        tshape = 'all' if base == 'outline_generic' else shape
        why = self.refused(tshape) if base != 'fixed' else ''
        if base == 'fixed' or why:
            out = (self.lab, self.tgt, v['seen'], v)
        elif base in ('kept', 'kept_v'):            # what the page draws: fin_sketch.pad_transfer, inside the template's skin
            r, _ = FS.pad_transfer(v, templates()[tshape]['cur'], t_edges(tshape), method='stretch_v' if base == 'kept_v' else 'stretch')
            k = FS.pad_ratio(self.e, t_edges(tshape))
            wh = FS.pad_template_view(templates()[tshape]['cur'])['whale']
            lab = warp_lab(self.lab, maps=FS.pad_stretch_maps(k, self.e, t_edges(tshape))) if base == 'kept_v' else warp_lab(self.lab, M=FS.pad_stretch_m(k))
            out = (lab, (r['marks'] | r['small']) & wh, r['seen'] & wh, None)
        elif base in ('outline', 'outline_generic', 'outline_s', 'outline_nov'):
            eo, et = self.e, t_edges(tshape)
            if base == 'outline_s':                  # edges smoothed along the height (0.06 H), no vertical correction
                from scipy.ndimage import gaussian_filter1d
                sm = lambda e: dict(e, lead=gaussian_filter1d(e['lead'], 6, mode='nearest'), trail=gaussian_filter1d(e['trail'], 6, mode='nearest'))
                eo, et = sm(eo), sm(et)
            maps = FS.pad_maps(eo, et, vcorr=base not in ('outline_s', 'outline_nov'))
            pull = lambda a: FS.pad_pull(a, maps)
            tv = FS.pad_template_view(templates()[tshape]['cur'])
            mk = pull(v['marks'])
            out = (warp_lab(self.lab, maps=maps), pull(self.tgt), pull(v['seen']), dict(tv, marks=mk, seen=pull(v['seen'])))
        else:
            if base in ('stretch', 'stretch_half'):
                k = FS.pad_ratio(self.e, t_edges(tshape)); tx = FS.TIP_PX[0]
                k = k ** 0.5 if base == 'stretch_half' else k
                M = np.array([[k, 0, tx - k * tx], [0, 1.0, 0]])
            else:
                M = icp(o_points(self.cur), t_points(tshape), base)
            out = (warp_lab(self.lab, M=M), warp(self.tgt, M), warp(v['seen'], M), None)
        self._m[key] = out
        return out


def compare(x, y, method, shape):
    """rows (diameter, hit) for x's drawn marks against y, in the method's frame."""
    lx, _, _, vx = x.mapped(method, shape)
    _, ty, sy, vy = y.mapped(method, shape)
    if method.endswith('+reg'):
        M, sc = FS.register_view(vx if vx is not None else x.v, vy if vy is not None else y.v)
        if sc < FS.V_MIN_SCORE:
            return None
        ty, sy = warp(ty, M), warp(sy, M)
    k = np.ones((2 * FS.V_TOL + 1, 2 * FS.V_TOL + 1), np.uint8)
    near = cv2.dilate(ty.astype(np.uint8), k) > 0
    out = {}
    if x.n <= 1:
        return out
    idx = lx.ravel(); ok = idx > 0
    cnt = np.bincount(idx[ok], minlength=x.n)
    seen = np.bincount(idx[ok], weights=sy.ravel()[ok].astype(float), minlength=x.n)
    hit = np.bincount(idx[ok], weights=near.ravel()[ok].astype(float), minlength=x.n)
    for i in range(1, x.n):
        if x.dia[i] < FS.MARK_D * 0.999:
            continue
        if cnt[i] == 0 or seen[i] / cnt[i] < 0.8:
            out[i] = None
        else:
            out[i] = bool(hit[i] > 0)
    return out


def run(day):
    cv2.setNumThreads(1)
    V = C.drawings(day); N = C.noted(day); Dd = C.bench.D(day)
    DR = {k: Drawing(v) for k, v in V.items()}
    rows = []
    for a, b, same in C.bench.pairs(day, 'dorsal'):
        for s in ('L', 'R'):
            if (a, s) not in DR or (b, s) not in DR:
                continue
            for p, q in ((a, b), (b, a)):
                shape = N.get(Dd.maj[q])
                if not shape:
                    continue
                x, y = DR[(p, s)], DR[(q, s)]
                res = {m: compare(x, y, m, shape) for m in METHODS}
                ref_x, ref_y = x.refused(shape), y.refused(shape)
                ang = abs(np.log(max(x.v['meta']['bperp'], 0.1) / max(y.v['meta']['bperp'], 0.1)))
                labs = {x.v['meta'].get('label', ''), y.v['meta'].get('label', '')}
                for i in range(1, x.n):
                    if x.dia[i] < FS.MARK_D * 0.999:
                        continue
                    r = [day in C.bench.VAL, bool(same), float(x.dia[i]), float(ang), labs == {'side', 'angled'}, bool(ref_x), bool(ref_y),
                         float(x.v['meta']['n'] >= 2 and y.v['meta']['n'] >= 2), float(x.depth[i]), float(x.cu[i]), float(x.cv[i])]
                    for m in METHODS:
                        h = None if res[m] is None else res[m].get(i)
                        r.append(-1 if h is None else int(h))
                    rows.append(r)
    return day, rows


COLS = ['val', 'same', 'dia', 'angle', 'side_vs_angled', 'refused_x', 'refused_y', 'multi', 'depth', 'u', 'v'] + list(METHODS)


def summarise(R, sel, common=True):
    out = {}
    ok_all = np.all(R[:, len(COLS) - len(METHODS):] >= 0, 1) if common else None
    for j, m in enumerate(METHODS):
        h = R[:, len(COLS) - len(METHODS) + j]
        ok = sel & (ok_all if common else (h >= 0))
        s = ok & (R[:, 1] > 0); d = ok & (R[:, 1] == 0)
        if s.sum() < 20 or d.sum() < 20:
            continue
        hs, hd = float(h[s].mean()), float(h[d].mean())
        out[m] = dict(n_same=int(s.sum()), n_diff=int(d.sum()), hit_same=round(hs, 3), hit_diff=round(hd, 3), lift=round(hs - hd, 3),
                      ratio=round(hs / max(hd, 1e-9), 3))
    return out


def main():
    with Pool(8) as p:
        res = p.map(run, C.DAYS)
    R = np.array([r for _, rs in res for r in rs], float)
    C.CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(C.CACHE / f'transfer_rows{TAG}.npz', R=R, cols=np.array(COLS))
    allr = np.ones(len(R), bool)
    refused = (R[:, 5] > 0) | (R[:, 6] > 0)
    out = {'rows': int(len(R)), 'marks_same': int((R[:, 1] > 0).sum()), 'marks_diff': int((R[:, 1] == 0).sum()),
           'share_rows_with_a_refused_drawing': round(float(refused.mean()), 3)}
    subsets = {'all': allr, 'both_transferred': ~refused, 'different_angles_base': R[:, 3] >= 0.4, 'different_angles_base_transferred': (R[:, 3] >= 0.4) & ~refused,
               'side_vs_angled': R[:, 4] > 0, 'similar_angles': R[:, 3] < 0.15, 'val_days': R[:, 0] > 0, 'both_multi_frame': R[:, 7] > 0,
               'large_marks_0.10': R[:, 2] >= 0.10,
               'interior_depth_0.1': R[:, 8] >= 0.1, 'edge_depth_lt_0.1': R[:, 8] < 0.1,
               'interior_different_angles': (R[:, 8] >= 0.1) & (R[:, 3] >= 0.4), 'interior_similar_angles': (R[:, 8] >= 0.1) & (R[:, 3] < 0.15),
               'on_fin_v_lt_-0.3': R[:, 10] < -0.3, 'hump_v_ge_-0.3': R[:, 10] >= -0.3,
               'on_fin_different_angles': (R[:, 10] < -0.3) & (R[:, 3] >= 0.4)}
    # refusals per drawing (every drawing of a catalogue animal against its own noted template, or the generic one)
    why = collections.Counter()
    for d in C.DAYS:
        N = C.noted(d)
        for (c, s), v in C.drawings(d).items():
            shape = N.get(v['meta']['animal']) or 'all'
            why[Drawing(v).refused(shape) or 'transferred'] += 1
    out['drawings_by_outcome'] = dict(why)
    print(dict(why))
    for k, sel in subsets.items():
        out[k] = summarise(R, sel)
        print(k, json.dumps(out[k]))
    out['own_valid_sets'] = summarise(R, allr, common=False)
    print('own', json.dumps(out['own_valid_sets']))
    json.dump(out, open(C.HERE / f'transfer{TAG}.json', 'w'), indent=1)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        d, rows = run(sys.argv[1]); print(d, len(rows))
    else:
        main()
