"""Zoom on the dorsal fin (the top of the whale mask) in the page's cached full-resolution views. Local files only."""
import json, math, numpy as np
from PIL import Image, ImageDraw, ImageFont
D = json.load(open('day.json')); P = D['photos']; M = json.load(open('mine.json')); mine = M['mine']; aname = M['aname']
R = '/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all/2024-12-23'; H = '/home/sat3737/NOAA_whales/Whale-Review-Web/data/2024-12-23/label_hires'
sh = lambda x: (x or '?').split('_')[-1]
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15); fb = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 17)
def region(bb, W, Hh):
    x0, y0, x1, y1 = bb; w, h = x1 - x0, y1 - y0; rw, rh = w + 2 * (max(w, 80.0) * 0.6 + 40), h + 2 * (max(h, 80.0) * 0.6 + 40); rw, rh = max(rw, 0.75 * rh), max(rh, 0.75 * rw); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return [int(max(0, math.floor(cx - rw / 2))), int(max(0, math.floor(cy - rh / 2))), int(min(W, math.ceil(cx + rw / 2))), int(min(Hh, math.ceil(cy + rh / 2)))]
def fin(pi, frac=0.42):
    p = P[pi]; bb = p['inst'][0]; small = Image.open(f"{R}/images/{p['stem']}.jpg"); W, Hh = small.size; reg = region(bb, W, Hh)
    m = np.asarray(Image.open(f"{R}/canon/{p['stem']}_mask.png")) > 0; x0, y0, x1, y1 = [int(v) for v in bb]; sub = m[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1]
    cols = np.where(sub.any(0))[0]; top = np.array([np.argmax(sub[:, c]) for c in cols]); k = int(np.argmin(top)); ax, ay = x0 + cols[k], y0 + top[k]
    side = max(frac * (bb[2] - bb[0]), 110); c = (ax - side / 2, ay - side * 0.38, ax + side / 2, ay + side * 0.62)
    try:
        hi = Image.open(f"{H}/{p['stem']}~0.jpg").convert('RGB'); sx = hi.width / (reg[2] - reg[0]); sy = hi.height / (reg[3] - reg[1])
        out = hi.crop((int((c[0] - reg[0]) * sx), int((c[1] - reg[1]) * sy), int((c[2] - reg[0]) * sx), int((c[3] - reg[1]) * sy))); px = side * sx
    except FileNotFoundError:
        out = small.convert('RGB').crop(tuple(int(v) for v in c)); px = side
    return out, px
def sheet(name, rows, T=440):
    cols = max(len(r[1]) for r in rows); W = cols * (T + 8) + 8; Hh = sum(34 + T + 44 for _ in rows) + 8; im = Image.new('RGB', (W, Hh), (24, 24, 24)); d = ImageDraw.Draw(im); y = 4
    for title, pis, colour in rows:
        d.rectangle((0, y, W, y + 30), fill=colour); d.text((8, y + 6), title, fill=(255, 255, 255), font=fb); y += 34
        for j, pi in enumerate(pis):
            v, px = fin(pi); v = v.resize((T, T), Image.LANCZOS if px > T else Image.BICUBIC); x = 8 + j * (T + 8); im.paste(v, (x, y)); p = P[pi]
            d.text((x + 2, y + T + 2), f"#{pi}  {p['clock']}  catalog {sh(p['cat'])}", fill=(255, 255, 255), font=font); d.text((x + 2, y + T + 21), f"you: {sh(aname.get(mine[str(pi)], '?'))}   crop {px:.0f} camera px wide", fill=(190, 190, 190), font=font)
        y += T + 44
    im.save(f'{name}.jpg', quality=90); print('saved', name, im.size)
RED, GRN = (120, 40, 40), (36, 92, 46)
sheet('fins_G01', [("Dorsal fin, the two photos you split off (your G01A02)", [14, 15], RED), ("Dorsal fin, your G01A01 (same whale by the catalog)", [0, 18, 19, 17], GRN)])
sheet('fins_G02', [("Dorsal fin, your G02A01 (3 photos)", [30, 31, 35], RED), ("Dorsal fin, your G02A02 (same whale by the catalog)", [34, 38, 40, 42], GRN), ("continued", [39, 43, 44, 46], GRN)])
sheet('fins_G03', [("Dorsal fin, your G03A01 (4 photos)", [54, 59, 83, 84], RED), ("Dorsal fin, your G03A03 (same whale by the catalog)", [61, 64, 67, 80], GRN), ("continued, and your G04A01 (photo 85)", [79, 58, 68, 85], GRN)])
