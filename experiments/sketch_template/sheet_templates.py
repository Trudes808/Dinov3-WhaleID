"""sketch_template: the template sheet (for the researchers to compare with their chart of standard shapes). Every noted
shape's template (black), head left as on the pad, over the side-on outlines of the animals noted with it (light),
with how it was made (direct from n animals, or from its parts: height + form). Drawings only, no photos.
    python sheet_templates.py OUT.jpg"""
import json
import sys

import cv2
import numpy as np

import common as C
import outline as O

FS = C.FS


def poly(cur):
    P = lambda u, v: np.stack([(np.asarray(u) - FS.DU0) * FS.DPX, (np.asarray(v) - FS.DV0) * FS.DPX], 1)
    t, r, b = cur['top'], cur['trail'], cur['back']
    return [P(O.UG, t), P(r, O.VG), P(O.BG, b)]


def draw_line(img, pts, col, w):
    ok = ~np.isnan(pts).any(1)
    for i in range(len(pts) - 1):
        if ok[i] and ok[i + 1]:
            cv2.line(img, tuple(int(round(c * 4)) for c in pts[i]), tuple(int(round(c * 4)) for c in pts[i + 1]), col, w, cv2.LINE_AA, shift=2)


def tile(shape, T, members, title, sub):
    tv = FS.pad_template_view(T[shape]['cur'])
    img = FS.render_view(tv, 'L')
    for m in members:
        for p in poly(m):
            draw_line(img, p, (170, 200, 235), 1)
    for p in (tv['top'], tv['trail']):
        draw_line(img, p, (0, 0, 0), 3)
    cv2.putText(img, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, sub, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
    return cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(200, 200, 200))


def main(out):
    T = FS.pad_templates(C.CACHE / 'templates.npz')
    z = np.load(C.CACHE / 'animal_curves.npz'); meta = json.loads(str(z['meta']))
    mem = {}
    for a, m in meta.items():
        mem.setdefault(m['shape'], []).append({p: z[f'{a}|{p}'] for p in ('top', 'trail', 'back')})
    rows = []
    for h in C.HEIGHTS:
        row = []
        for f in C.FORMS:
            s = f'{h} {f}'; n = len(mem.get(s, []))
            how = T[s]['how']
            sub = f"from {n} animals (side-on drawings)" if how == 'direct' else f"from its parts: {h} + {f} ({n} animal{'s' if n != 1 else ''} of its own)"
            row.append(tile(s, T, mem.get(s, []), s, sub))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    head = np.full((60, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(head, 'Dorsal fin templates per noted shape (trip log "height form"), head left, side profile. Black = template; light blue = the side-on outlines of the animals noted with that shape (21 train + 3 val days).',
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(head, 'Same frame on every tile: tip on the blue cross, back base on the lower dotted line, fin height = the distance between the dotted lines. "From its parts" = height template + form template - average (too few animals of its own).',
                (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    sheet = np.vstack([head, sheet])
    cv2.imwrite(out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(out, sheet.shape)


if __name__ == '__main__':
    main(sys.argv[1])
