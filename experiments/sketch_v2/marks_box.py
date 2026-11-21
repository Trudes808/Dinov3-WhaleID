"""sketch_v2 C + D: the researcher's evidence boxes of 2025-01-02 (a train day) against the marks of each photo's fin
drawing (the prototype's marks_check.py on fin_sketch v2). Every region box of Whale-Review-Web/data/2025-01-02/
pair_marks.json on a photo with a drawable fin is mapped into that photo's fin tile (2560-px photo -> cut-out ->
keypoint frame); per box: does the frame's drawing hold a mark in it (box dilated by 4 px), against the same box moved
to 50 random places of the seen skin. Marks: all the frame finds (>= MIN_AREA tile px) and only those of the size
drawn (fin_sketch.mark_min_area). SMALL boxes (< 10 % of the seen skin) apart. Called by marks_light.py boxes TONE."""
import json
from pathlib import Path

import cv2
import numpy as np

import bench
from model_fine_tuning.cluster import fin_sketch as FS

DAY = '2025-01-02'
MARKS = Path('/home/sat3737/NOAA_whales/Whale-Review-Web/data/2025-01-02/pair_marks.json')


def run(tone, patch_source):
    FS.source = patch_source(tone)
    d = FS.day_dir(DAY)
    P = json.loads((d / 'sketch' / 'points.json').read_text())['points']
    boxes = {}
    for r in json.loads(MARKS.read_text())['labels']:
        for g in r.get('regions') or []:
            for side in ('a', 'b'):
                k = r.get(f'key_{side}')
                if k in P and side in g:
                    boxes.setdefault(k, []).append(g[side])
    rng = np.random.default_rng(0)
    rows = []
    for k, bx in boxes.items():
        r, _ = FS.one(d, k, P[k])
        if r is None:
            continue
        t, mt, ins, info = r
        fr = FS.frame_drawing(t, mt, ins)
        whale, seen = fr['whale'], fr['seen']
        m_all = fr['marks'] & seen
        m_drawn = FS.clean_marks(m_all, FS.mark_min_area(info['H']))[0]
        s = FS.source(d, k)
        img, m, f, name = s
        Mt = FS.frame(f(P[k]['back']), f(P[k]['tip']), f(P[k]['front']))[0]
        h, w = cv2.imread(str(d / 'images' / f"{k.rsplit('~', 1)[0]}.jpg")).shape[:2]
        ys, xs = np.nonzero(seen)
        if len(ys) < 200:
            continue
        for b in bx:
            pts = np.array([[b['x'], b['y']], [b['x'] + b['w'], b['y']], [b['x'] + b['w'], b['y'] + b['h']], [b['x'], b['y'] + b['h']]]) * [w, h]
            q = np.array([Mt[:, :2] @ f(p) + Mt[:, 2] for p in pts])
            c = q.mean(0).astype(int)
            if not (0 <= c[0] < FS.TW and 0 <= c[1] < FS.TH) or not whale[c[1], c[0]]:
                continue
            one = np.zeros((FS.TH, FS.TW), np.uint8); cv2.fillPoly(one, [q.astype(np.int32)], 1)
            one = cv2.dilate(one, np.ones((9, 9), np.uint8)) > 0
            frac = (one & seen).sum() / max(seen.sum(), 1)
            cy, cx = np.nonzero(one)
            ch_all, ch_drawn = [], []
            for _ in range(50):
                j = rng.integers(len(ys))
                yy, xx = np.clip(cy + ys[j] - int(cy.mean()), 0, FS.TH - 1), np.clip(cx + xs[j] - int(cx.mean()), 0, FS.TW - 1)
                ch_all.append(bool(m_all[yy, xx].any())); ch_drawn.append(bool(m_drawn[yy, xx].any()))
            rows.append(dict(frac=float(frac), hit_all=bool((one & m_all).any()), hit_drawn=bool((one & m_drawn).any()),
                             ch_all=float(np.mean(ch_all)), ch_drawn=float(np.mean(ch_drawn)), med=float(np.median(cv2.cvtColor(t, cv2.COLOR_RGB2GRAY)[whale]))))
    out = {'tone': tone, 'boxes': len(rows)}
    for tag, sel in (('all', [True] * len(rows)), ('small', [r['frac'] < 0.1 for r in rows]), ('small_dark', [r['frac'] < 0.1 and r['med'] < 70 for r in rows])):
        R = [r for r, s_ in zip(rows, sel) if s_]
        if not R:
            continue
        out[tag] = dict(n=len(R), hit_all=round(float(np.mean([r['hit_all'] for r in R])), 3), chance_all=round(float(np.mean([r['ch_all'] for r in R])), 3),
                        hit_drawn=round(float(np.mean([r['hit_drawn'] for r in R])), 3), chance_drawn=round(float(np.mean([r['ch_drawn'] for r in R])), 3))
    return out
