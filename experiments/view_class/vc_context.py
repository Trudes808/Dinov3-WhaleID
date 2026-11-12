"""Context strips for unclear labels: vc_context.py NAME CODES [--rows 6] -> sheets/NAME_<i>.jpg + NAME.json
One row per labelled instance whose code is in CODES (comma list): the frame itself in the middle (yellow ticks) and
the two frames before / after it on the same camera, all cut at the same place (the box x 2.4)."""
import sys, json
from PIL import Image, ImageDraw
from vc_common import *

name, codes = sys.argv[1], set(sys.argv[2].split(',')); ROWS = 6; CW, CH = 316, 250
L = load_labels(); todo = sorted((v['day'], k) for k, v in L.items() if v['code'] in codes and not v.get('ctx') and not v.get('rev') and (len(sys.argv) < 4 or v['round'] == sys.argv[3]))
index = {}; cache = {}
for s in range(0, len(todo), ROWS):
    sheet = Image.new('RGB', (5 * (CW + 4), ROWS * (CH + 22)), (0, 0, 0)); idx = {}
    for r, (day, key) in enumerate(todo[s:s + ROWS]):
        if day not in cache:
            P = json.load(open(f'{D}/{day}/safe/photos.json')); order = {}
            for stem, p in P.items(): order.setdefault(p.get('camera'), []).append((p.get('time') or 0, stem))
            for c in order.values(): c.sort()
            cache[day] = (P, order, load_day(day, feats=False)['meta'])
        P, order, meta = cache[day]; m = meta[key]; seq = order[m['cam']]; i = [st for _, st in seq].index(m['stem'])
        x0, y0, x1, y1 = m['bbox']; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; w = max((x1 - x0) * 2.4, 300); h = max((y1 - y0) * 2.4, w * CH / CW); w = max(w, h * CW / CH)
        for c, j in enumerate(range(i - 2, i + 3)):
            if not 0 <= j < len(seq): continue
            t, stem = seq[j]; im = Image.open(f'{D}/{day}/images/{stem}.jpg').convert('RGB')
            if j == i:
                d = ImageDraw.Draw(im); lw = max(3, int(w / 150)); tk = max(12, int(0.06 * max(x1 - x0, y1 - y0)))
                for (px, py, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
                    d.line((px, py, px + sx * tk, py), fill=(255, 230, 0), width=lw); d.line((px, py, px, py + sy * tk), fill=(255, 230, 0), width=lw)
            tile = im.crop((int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))).resize((CW, CH), Image.LANCZOS)
            sheet.paste(tile, (c * (CW + 4) + 2, r * (CH + 22) + 20))
            ImageDraw.Draw(sheet).text((c * (CW + 4) + 6, r * (CH + 22) + 4), f"{r}: {t - seq[i][0]:+.1f}s" if j != i else f"{r}: THIS ({L[key]['code']})", fill=(255, 255, 255))
        idx[r] = (day, key)
    sheet.save(f'{HERE}/sheets/{name}_{s // ROWS:03d}.jpg', quality=88); index[f'{s // ROWS:03d}'] = idx
json.dump(index, open(f'{HERE}/sheets/{name}.json', 'w')); print(len(todo), 'instances,', len(index), 'sheets')
