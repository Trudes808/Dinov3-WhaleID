"""Contact sheet of chosen instances: nw_sheet.py OUT ROUND SPLIT MIN_SCORE [unlabelled] -> sheets/OUT.jpg + .json (tile -> day, key, score)."""
import sys, json, pickle
from PIL import Image, ImageDraw
from nw_common import *
out, rnd, split, lo = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]); unl = len(sys.argv) > 5
S = pickle.load(open(f'nw_scores_{rnd}.pkl', 'rb'))['scores']; L = load_labels(); days = {'val': VAL, 'train': TRAIN}[split]; cand = []
for d in days:
    P = json.load(open(f'{D}/{d}/safe/photos.json')); lab = L.get(d, {}); seen = set(lab.get('pos', [])) | set(lab.get('neg', [])) | set(lab.get('skip', []))
    for stem, p in P.items():
        for it in p['instances']:
            s = S[d].get(it['key'])
            if s is not None and s >= lo and not (unl and it['key'] in seen): cand.append((s, d, it['key'], it['bbox'], p.get('focal_mm') or 0))
cand.sort(key=lambda t: -t[0]); T = 250; cols = 8; rows = (len(cand) + cols - 1) // cols; sheet = Image.new('RGB', (cols * (T + 4), max(rows, 1) * (T + 20)), (20, 20, 20)); index = {}
for i, (sco, d, k, b, foc) in enumerate(cand[:96]):
    index[i] = (d, k, sco); im = Image.open(f"{D}/{d}/images/{k.split('~')[0]}.jpg").convert('RGB'); x0, y0, x1, y1 = b; side = max(max(x1 - x0, y1 - y0) * 2.2, 420); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ImageDraw.Draw(im).rectangle((x0, y0, x1, y1), outline=(255, 220, 0), width=max(3, int(side / 120)))
    c = im.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))).resize((T, T), Image.LANCZOS)
    t = Image.new('RGB', (T, T + 16), (20, 20, 20)); t.paste(c, (0, 16)); ImageDraw.Draw(t).text((2, 2), f"{i}  s{sco:.1f}  {foc:.0f}mm  {d[5:]}", fill=(255, 255, 255)); sheet.paste(t, ((i % cols) * (T + 4) + 2, (i // cols) * (T + 20) + 2))
sheet.save(f'sheets/{out}.jpg', quality=82); json.dump(index, open(f'sheets/{out}.json', 'w')); print('saved', f'sheets/{out}.jpg', len(cand), 'instances', sheet.size)
