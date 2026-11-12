"""Contact sheets of the whale views (the page's own full-resolution view) for the splits of 2024-12-23. Local files only."""
import json, io, urllib.request, urllib.parse, collections, sys
from PIL import Image, ImageDraw, ImageFont
D = json.load(open('day.json')); P = D['photos']; M = json.load(open('mine.json')); mine = M['mine']; aname = M['aname']
R = '/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all/2024-12-23'
F = json.load(open(f'{R}/safe/photos.json')); sh = lambda x: (x or '?').split('_')[-1]
parts = {}
for line in open(f'{R}/parts_hr/index.jsonl'):
    r = json.loads(line)
    for i in r.get('instances') or []: parts[i['key']] = i['parts']
try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15); fb = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 17)
except Exception: font = fb = ImageFont.load_default()
def view(pi):
    p = P[pi]; key = f"{p['stem']}~0"
    try:
        raw = urllib.request.urlopen('http://127.0.0.1:8765/img/labeling/2024-12-23/hires/' + urllib.parse.quote(key) + '.jpg', timeout=120).read(); return Image.open(io.BytesIO(raw)).convert('RGB')
    except Exception as e:
        im = Image.open(f"{R}/images/{p['stem']}.jpg").convert('RGB')
        if p['inst']:
            x0, y0, x1, y1 = p['inst'][0]; w, h = x1 - x0, y1 - y0; m = 0.6 * max(w, h); im = im.crop((int(max(x0 - m, 0)), int(max(y0 - m, 0)), int(min(x1 + m, im.width)), int(min(y1 + m, im.height))))
        return im
def info(pi):
    p = P[pi]; f = F.get(p['stem'], {}); key = f"{p['stem']}~0"; pr = '+'.join(x for x in (parts.get(key) or []) if x != 'body') or ('body' if parts.get(key) else 'no whale found')
    side = {1: 'head right', -1: 'head left'}.get(f.get('head_side'), '') if (f.get('cue_confidence') or 0) >= 0.6 else ''
    w = (p['inst'][0][2] - p['inst'][0][0]) if p['inst'] else 0
    return f"#{pi}  {p['clock']}  {pr}  {side}", f"catalog {sh(p['cat'])}   you: {sh(aname.get(mine[str(pi)], 'rejected' if mine[str(pi)] == 'x' else '?'))}   {p['fl']} mm, whale {w:.0f} px"
def sheet(name, rows, TW=470, TH=330):
    cols = max(len(r[1]) for r in rows); H = sum(34 + TH + 44 for _ in rows) + 8; W = cols * (TW + 8) + 8
    im = Image.new('RGB', (W, H), (24, 24, 24)); d = ImageDraw.Draw(im); y = 4
    for title, pis, colour in rows:
        d.rectangle((0, y, W, y + 30), fill=colour); d.text((8, y + 6), title, fill=(255, 255, 255), font=fb); y += 34
        for j, pi in enumerate(pis):
            v = view(pi); s = min(TW / v.width, TH / v.height); v = v.resize((max(int(v.width * s), 1), max(int(v.height * s), 1)), Image.LANCZOS); x = 8 + j * (TW + 8)
            im.paste(v, (x + (TW - v.width) // 2, y + (TH - v.height) // 2)); a, b = info(pi); d.text((x + 2, y + TH + 2), a, fill=(255, 255, 255), font=font); d.text((x + 2, y + TH + 21), b, fill=(190, 190, 190), font=font)
        y += TH + 44
    im.save(f'{name}.jpg', quality=88); print('saved', name, im.size)
RED, GRN, BLU, GRY = (120, 40, 40), (36, 92, 46), (36, 70, 110), (70, 70, 70)
which = sys.argv[1:] or ['g01', 'g02', 'g03a', 'g03b']
if 'g01' in which: sheet('split_G01_singer', [("Split off as a second animal (your G01A02). The catalog: the same whale, G01A01", [14, 15, 16], RED),
    ("Your G01A01, the shots just before (other camera, 09:29:56 to 09:30:01)", [0, 7, 13], GRN), ("Your G01A01, the shots just after (same camera, from 09:33:29)", [17, 18, 19, 20], GRN), ("Your G01A01, later (09:33:43 to 09:51:59)", [24, 27, 28, 29], GRN)])
if 'g02' in which: sheet('split_G02_pair', [("Your G02A01 (3 photos). The catalog: the same whale as your G02A02 below, G02A01", [30, 31, 35], RED),
    ("Your G02A02 (17 photos). The catalog: G02A01", [34, 38, 39, 40], GRN), ("Your G02A02, continued", [42, 43, 48, 53], GRN), ("Your G02A03 (4 photos). The catalog: the second whale of the pair, G02A02. Correct", [32, 33, 36, 37], BLU)])
if 'g03a' in which: sheet('split_G03_mother_a', [("Your G03A01 (4 photos). The catalog: the same whale as your G03A03 below, the mother G03A01", [54, 59, 83, 84], RED),
    ("Your G03A03 (24 photos). The catalog: the mother G03A01", [55, 57, 58, 60], GRN), ("Your G03A03, continued", [61, 64, 67, 72], GRN), ("Your G03A03, continued", [73, 74, 79, 80], GRN)])
if 'g03b' in which: sheet('split_G03_mother_b', [("Your G04A01 (1 photo, 37 min later, after a proposed group cut) and the photo you rejected. The catalog: the mother G03A01", [85, 86], RED),
    ("Your G03A02 (3 photos). The catalog: the calf. Correct", [78, 81, 82], BLU), ("No whale found by the model; you put them with G03A03. The catalog: G03A01. Correct", [62, 63, 66], GRY)])
