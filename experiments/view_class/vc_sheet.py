"""Contact sheets of instances for labelling by eye: vc_sheet.py NAME KEYS.json [--per 18]
KEYS.json = [[day, key], ...]; writes sheets/NAME_<i>.jpg and sheets/NAME.json (sheet -> tile -> day, key).
A tile shows the instance box with a margin (aspect between 1:1 and 3:1) and corner ticks on the box; no model output."""
import sys, json
from PIL import Image, ImageDraw
from vc_common import *

CW, CH, COLS, ROWS = 520, 260, 3, 6


def tile(day, key, bbox, label):
    im = Image.open(f"{D}/{day}/images/{key.split('~')[0]}.jpg").convert('RGB'); W, H = im.size
    x0, y0, x1, y1 = bbox; w, h = x1 - x0, y1 - y0; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cw, ch = max(w * 1.3, 220), max(h * 1.3, 110)
    if cw / ch > 3.0: ch = cw / 3.0
    if cw / ch < 1.0: cw = ch
    if cw / ch < CW / CH and cw / ch >= 1.0: pass
    sc = min(CW / cw, CH / ch); box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
    d = ImageDraw.Draw(im); t = max(8, int(0.04 * max(w, h))); lw = max(2, int(1.5 / sc))
    for (px, py, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        px -= sx * 2 * lw; py -= sy * 2 * lw
        d.line((px, py, px + sx * t, py), fill=(255, 230, 0), width=lw); d.line((px, py, px, py + sy * t), fill=(255, 230, 0), width=lw)
    c = im.crop(tuple(int(round(v)) for v in box)).resize((max(1, int(cw * sc)), max(1, int(ch * sc))), Image.LANCZOS)
    out = Image.new('RGB', (CW, CH + 18), (25, 25, 25)); out.paste(c, ((CW - c.size[0]) // 2, 18 + (CH - c.size[1]) // 2))
    ImageDraw.Draw(out).text((4, 3), label, fill=(255, 255, 255))
    return out


if __name__ == '__main__':
    name, keys = sys.argv[1], json.load(open(sys.argv[2])); per = COLS * ROWS; metas = {}
    index = {}
    for s in range(0, len(keys), per):
        sheet = Image.new('RGB', (COLS * (CW + 4), ROWS * (CH + 22)), (0, 0, 0)); idx = {}
        for i, (day, key, *cap) in enumerate(keys[s:s + per]):
            if day not in metas: metas[day] = load_day(day, feats=False)['meta']
            m = metas[day][key]
            sheet.paste(tile(day, key, m['bbox'], f"{i}  {cap[0]}" if cap else f"{i}"), ((i % COLS) * (CW + 4) + 2, (i // COLS) * (CH + 22) + 2)); idx[i] = (day, key)
        sheet.save(f'{HERE}/sheets/{name}_{s // per:03d}.jpg', quality=88); index[f'{s // per:03d}'] = idx
    json.dump(index, open(f'{HERE}/sheets/{name}.json', 'w')); print('saved', len(index), 'sheets for', len(keys), 'instances')
