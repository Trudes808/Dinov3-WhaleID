"""sketch_v2 sheets for the lead (fin cut-outs and drawings only; written OUTSIDE the repo, to
moonshot_cache/sheets/sketch_v2/). Train / val days only for anything tuned.
    python sheets.py before_after DAY [N]     cut-out | v1 drawing | v2 drawing (same stacks)
    python sheets.py small DAY [N]            cut-out | v2 drawing with the marks left out as too small in orange
    python sheets.py animal DAY [N]           an animal's stacks (one side) | the animal-level drawing
    python sheets.py views DAY [N]            side-on (est.) | angled drawings with their cut-outs"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'moonshot_sizing'))
sys.path.insert(0, str(HERE.parents[1]))
from model_fine_tuning.cluster import fin_sketch as FS  # noqa: E402

OUT = Path('/home/sat3737/NOAA_whales/moonshot_cache/sheets/sketch_v2')
SUB = 'sketch/v2'


def label(img, text, h=22):
    head = np.full((h, img.shape[1], 3), 255, np.uint8)
    cv2.putText(head, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([head, img])


def to_h(img, h):
    return cv2.resize(img, (int(round(img.shape[1] * h / img.shape[0])), h), interpolation=cv2.INTER_AREA)


def rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def cut(d, e, s, P):
    c = [(FS.cutout_view(d, k, P[k]), k) for k in e['keys']]
    c = [x for x, k in c if x is not None]
    if not c:
        return None
    t = max(c, key=lambda x: x[1]['H'])[0]
    return t[:, ::-1] if s == 'R' else t


def gap(h, w=8):
    return np.full((h, w, 3), 255, np.uint8)


def write(rows, name):
    W = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], W - r.shape[1], 3), 255, np.uint8)]) for r in rows]
    img = np.vstack(sum([[r, np.full((8, W, 3), 255, np.uint8)] for r in rows], []))
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 86])
    print(OUT / name)


def items(day, n, seed=0, pred=None):
    d = FS.day_dir(day); idx = json.loads((d / SUB / 'index.json').read_text())['clusters']
    it = [(c, s, e) for c, v in idx.items() for s, e in v.items() if pred is None or pred(e)]
    rng = np.random.default_rng(seed)
    return [it[i] for i in rng.permutation(len(it))[:n]]


def before_after(day, n=10):
    d = FS.day_dir(day); P = json.loads((d / 'sketch' / 'points.json').read_text())['points']
    v1 = json.loads((d / 'sketch' / 'index.json').read_text())['clusters']
    rows = []
    for c, s, e in items(day, 4 * n, pred=lambda e: e['n'] >= 2):
        o = (v1.get(c) or {}).get(s)
        if not o or (d / 'sketch' / o['png']).stat().st_size == 0:
            continue
        t = cut(d, e, s, P)
        if t is None:
            continue
        H = 200
        r = np.hstack([to_h(t, H), gap(H), to_h(rgb(d / 'sketch' / o['png']), H), gap(H), to_h(rgb(d / SUB / e['png']), H)])
        rows.append(label(r, f"{day} cluster {c} {'left' if s == 'L' else 'right'} side - cut-out | before (v1, {o['n']} frames) | after (v2, {e['n']} frames, view {e['view'] or '?'}, {e['marks']} marks, {e['small_left_out']} small left out)"))
        if len(rows) >= n:
            break
    write(rows, f'before_after_{day}.jpg')


def small(day, n=10):
    d = FS.day_dir(day); P = json.loads((d / 'sketch' / 'points.json').read_text())['points']
    rows = []
    for c, s, e in items(day, 5 * n, seed=1, pred=lambda e: e['small_left_out'] >= 2):
        t = cut(d, e, s, P)
        if t is None:
            continue
        v = FS.load_view(d / SUB / e['npz'])
        a = FS.render_view(v, s, single=e['single'], show_small=True)
        H = 200
        rows.append(label(np.hstack([to_h(t, H), gap(H), to_h(a, H)]), f"{day} cluster {c} {s} - cut-out | drawing: black/grey = drawn ({e['marks']}), ORANGE = left out as too small ({e['small_left_out']})"))
        if len(rows) >= n:
            break
    write(rows, f'small_marks_{day}.jpg')


def animal(day, n=6):
    """animals of the catalogue (a train / val day) with >= 3 stacks drawn on one side: their stacks | the composition."""
    import bench
    d = FS.day_dir(day); idx = json.loads((d / SUB / 'index.json').read_text())['clusters']
    Dd = bench.D(day)
    by = {}
    for c in idx:
        a = Dd.maj.get(int(c))
        if a:
            for s in idx[c]:
                by.setdefault((a, s), []).append(c)
    rows = []
    for (a, s), cs in sorted(by.items(), key=lambda x: -len(x[1])):
        if len(cs) < 3:
            continue
        cs = cs[:6]
        views = [FS.load_view(d / SUB / idx[c][s]['npz']) for c in cs]
        img, info = FS.compose(views, s)
        H = 110
        st = [to_h(rgb(d / SUB / idx[c][s]['png']), H) for c in cs]
        top = np.hstack(sum([[x, gap(H, 4)] for x in st], []))
        bot = to_h(img, 2 * H)
        rows.append(label(top, f"{day} one catalogue animal, {s} side: its {len(cs)} stacks' drawings (outline taken from stack {info['ref'] + 1}, the most side-on)"))
        rows.append(label(bot, f"animal-level: {info['used']} of {len(cs)} lined up; {info['marks_solid']} marks in 2+ stacks (solid), {info['marks_light']} in one (light)"))
        if len(rows) >= 2 * n:
            break
    write(rows, f'animal_level_{day}.jpg')


def views(day, n=8):
    d = FS.day_dir(day); P = json.loads((d / 'sketch' / 'points.json').read_text())['points']
    rows = []
    for want in ('side', 'angled', ''):
        k = 0
        for c, s, e in items(day, 6 * n, seed=2, pred=lambda e: e['view'] == want and e['n'] >= 2):
            t = cut(d, e, s, P)
            if t is None:
                continue
            H = 160
            rows.append(label(np.hstack([to_h(t, H), gap(H), to_h(rgb(d / SUB / e['png']), H)]),
                              f"{day} cluster {c} {s} - label {FS.VIEW_TEXT[want] or 'angle ?'} (pose |nose right| {e['profile']}, base ratio {e['bperp']})"))
            k += 1
            if k >= n // 2:
                break
    write(rows, f'view_labels_{day}.jpg')




def drawing_with(day, e, s, tone):
    """the stack's drawing made again with the fin cut-outs toned ('none' | 'curve' | 'raw'), in the picture frame."""
    import marks_light as ML
    orig = FS.source
    FS.source = ML.patch_source(tone)
    try:
        d = FS.day_dir(day); P = json.loads((d / 'sketch' / 'points.json').read_text())['points']
        rs = [FS.one(d, k, P[k])[0] for k in e['keys']]
        rs = [r for r in rs if r is not None]
        dr, used, ncc, Fr = FS.side_drawing([r[:3] for r in rs], [r[3] for r in rs])
        if dr is None:
            return None, None
        v = FS.to_view(dr, FS.view_frame()[0])
        t = cv2.warpAffine(rs[0][0], FS.view_frame()[0], (FS.DW, FS.DH), flags=cv2.INTER_CUBIC)
        img = FS.render_view(v, s, single=len(used) < 2)
        if s == 'R':
            t = t[:, ::-1]
        return img, t
    finally:
        FS.source = orig


def dark(days, n=10, tones=('none', 'curve')):
    """dark, underexposed fins (whale median brightness < marks_light.DARK in the cut-out): cut-out as used | drawing
    without | with each exposure adjustment (the toned cut-out shown above its drawing)."""
    import marks_light as ML
    rows = []
    for day in days.split(','):
        d = FS.day_dir(day); idx = json.loads((d / SUB / 'index.json').read_text())['clusters']
        P = json.loads((d / 'sketch' / 'points.json').read_text())['points']
        for c, v in idx.items():
            for s, e in v.items():
                if e['n'] < 2 or e['H'] < 60:
                    continue
                if (day, int(c)) not in ML.DARK_SET:        # dark (whale median < DARK in the cut-outs), RAW cut-outs decoded
                    continue
                H = 150
                out = []
                for t in tones:
                    img, cutv = drawing_with(day, e, s, t)
                    if img is None:
                        break
                    out.append(np.vstack([to_h(cutv, H), to_h(img, H)]))
                if len(out) < len(tones):
                    continue
                r = np.hstack(sum([[x, gap(2 * H)] for x in out], []))
                rows.append(label(r, f"{day} cluster {c} {s} ({e['n']} frames) - " + ' | '.join(f'{t}: cut-out above, drawing below' for t in tones)))
                if len(rows) >= n:
                    break
            if len(rows) >= n:
                break
    write(rows, f"dark_fins_{'_'.join(tones)}.jpg")


if __name__ == '__main__':
    if sys.argv[1] == 'dark':
        dark(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10, tuple(sys.argv[4].split(',')) if len(sys.argv) > 4 else ('none', 'curve'))
    else:
        f = {'before_after': before_after, 'small': small, 'animal': animal, 'views': views}[sys.argv[1]]
        f(sys.argv[2], *(int(x) for x in sys.argv[3:]))
