"""Eval days, frozen filter, for the report only: one sheet with every flagged instance and the highest-scoring
unflagged ones (to see what the filter misses). Nothing is tuned after this look."""
import json
from PIL import Image, ImageDraw
from nw_common import D, EVAL
cand = []
for d in EVAL:
    nw = json.load(open(f'{D}/{d}/safe/not_whale.json')); P = json.load(open(f'{D}/{d}/safe/photos.json'))
    for stem, p in P.items():
        for it in p['instances']: cand.append((nw['scores'][it['key']], d, it['key'], it['bbox'], p.get('focal_mm') or 0))
cand.sort(key=lambda t: -t[0]); nf = sum(1 for c in cand if c[0] > 2.0); cand = cand[:48]; T = 250; cols = 8
sheet = Image.new('RGB', (cols * (T + 4), 6 * (T + 20)), (20, 20, 20)); index = {}
for i, (sco, d, k, b, foc) in enumerate(cand):
    index[i] = (d, k, sco); im = Image.open(f"{D}/{d}/images/{k.split('~')[0]}.jpg").convert('RGB'); x0, y0, x1, y1 = b; side = max(max(x1 - x0, y1 - y0) * 2.2, 420); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ImageDraw.Draw(im).rectangle((x0, y0, x1, y1), outline=(255, 220, 0), width=max(3, int(side / 120)))
    c = im.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))).resize((T, T), Image.LANCZOS)
    t = Image.new('RGB', (T, T + 16), (20, 20, 20)); t.paste(c, (0, 16)); ImageDraw.Draw(t).text((2, 2), f"{i}  s{sco:.1f}  {foc:.0f}mm  {d[5:]}" + ("  FLAG" if sco > 2 else ""), fill=(255, 255, 255) if sco > 2 else (170, 170, 170)); sheet.paste(t, ((i % cols) * (T + 4) + 2, (i // cols) * (T + 20) + 2))
sheet.save('sheets/nw_eval.jpg', quality=82); json.dump(index, open('sheets/nw_eval.json', 'w')); print('flagged on the eval days:', nf, '| sheet: the flagged ones first, then the next', 48 - nf, 'by score; lowest score shown %.2f' % cand[-1][0])
