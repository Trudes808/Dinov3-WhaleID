"""sketch_template C (2026-11-19): within-burst repeatability. A stack side with at least 4 drawable frames is drawn twice
from two halves of its frames (alternating, so both halves span the burst), with fin_sketch's own steps (one ->
side_drawing -> to_view, as draw_day). The same moment, the same light: a mark of half A should be found in half B. Is it
found as often in the template frame (each half carried onto the animal's noted template on its own) as in the fixed
frame? And how often does one half refuse the transfer when the other does not (the refusal's own repeatability)?
A's drawn marks (>= MARK_D), B's marks (drawn or small) within V_TOL px, where B saw the skin. 21 train + 3 val days.
    python splithalf.py    -> splithalf.json"""
import collections
import json
from multiprocessing import Pool

import cv2
import numpy as np

import common as C
from transfer import Drawing, compare

FS = C.FS
METHODS = ('fixed', 'outline', 'stretch', 'stretch_half', 'kept_v', 'affine')
MIN_FRAMES = 4


def half_view(d, keys, P):
    got = []
    for k in keys:
        r, _ = FS.one(d, k, P[k])
        if r is not None:
            got.append(r)
    if len(got) < 2:
        return None
    dr, used, ncc, Fr = FS.side_drawing([r[:3] for r in got], [r[3] for r in got])
    if not used or len(used) < 2:
        return None
    Fm = np.median(np.stack(Fr), 0)
    M, phi, bperp = FS.view_frame(Fm)
    v = FS.to_view(dr, M)
    Hm = float(np.median([got[j][3]['H'] for j in used]))
    v['meta'] = dict(n=len(used), bperp=bperp, H=Hm, label='')
    return v


def run(day):
    cv2.setNumThreads(1)
    d = FS.day_dir(day)
    P = json.loads((d / 'sketch' / 'points.json').read_text())['points']
    idx = json.loads((d / C.SUB / 'index.json').read_text())['clusters']
    N = C.noted(day); Dd = C.bench.D(day)
    rows, refuse = [], collections.Counter()
    for c in C.bench.clusters(day, 'dorsal'):
        shape = N.get(Dd.maj[c])
        if not shape:
            continue
        for s, x in (idx.get(str(c)) or {}).items():
            keys = [k for k in x['keys'] if k in P]
            if len(keys) < MIN_FRAMES:
                continue
            A, B = half_view(d, keys[0::2], P), half_view(d, keys[1::2], P)
            if A is None or B is None:
                continue
            DA, DB = Drawing(A), Drawing(B)
            ra, rb = bool(DA.refused(shape)), bool(DB.refused(shape))
            refuse[(ra, rb)] += 1
            for x_, y_ in ((DA, DB), (DB, DA)):
                res = {m: compare(x_, y_, m, shape) for m in METHODS}
                for i in range(1, x_.n):
                    if x_.dia[i] < FS.MARK_D * 0.999:
                        continue
                    r = [ra or rb]
                    for m in METHODS:
                        h = res[m].get(i) if res[m] else None
                        r.append(-1 if h is None else int(h))
                    rows.append(r)
    return day, rows, refuse


def main():
    with Pool(8) as p:
        res = p.map(run, C.DAYS)
    R = np.array([r for _, rs, _ in res for r in rs], float)
    ref = collections.Counter()
    for _, _, c in res:
        ref.update(c)
    ok = np.all(R[:, 1:] >= 0, 1)
    out = {'marks': int(len(R)), 'marks_valid_all_frames': int(ok.sum()),
           'halves': {'both_ok': ref[(False, False)], 'both_refused': ref[(True, True)], 'one_refused': ref[(True, False)] + ref[(False, True)]}}
    for tag, sel in (('all', ok), ('both_transferred', ok & (R[:, 0] == 0))):
        out[tag] = {m: dict(n=int(sel.sum()), hit=round(float(R[sel, 1 + j].mean()), 3)) for j, m in enumerate(METHODS)}
    print(json.dumps(out, indent=1))
    json.dump(out, open(C.HERE / 'splithalf.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
