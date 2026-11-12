"""Contact sheet of the wrong links (P >= threshold, different catalog animals) of a day's safe clustering: both photos
with the main instance box, the catalog ids, dt and the pair features. python wrong_links_sheet.py DAY OUT.jpg [max]"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
day, out = sys.argv[1], sys.argv[2]; mx = int(sys.argv[3]) if len(sys.argv) > 3 else 24
D = Path("experiments/days_all") / day
rep = json.loads((D / "safe" / "clusters.json").read_text())["report"]
feats = json.loads((D / "safe" / "photos.json").read_text())
labels = json.load(open("model_fine_tuning/splits/labels_all_days.json"))["days"][day]["photos"]
tiles = []
for a, b, p, f in rep["wrong_links"][:mx]:
    row = []
    for s in (a, b):
        im = Image.open(D / "images" / f"{s}.jpg").convert("RGB"); W, H = im.size
        bb = (feats[s].get("instances") or [{}])[0].get("bbox")
        if bb:
            cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2; half = max(bb[2] - bb[0], bb[3] - bb[1]) * 1.6 + 120
            box = (int(max(cx - half, 0)), int(max(cy - half * 0.6, 0)), int(min(cx + half, W)), int(min(cy + half * 0.6, H)))
        else:
            box = (0, 0, W, H)
        crop = im.crop(box); s_ = 560 / crop.width; crop = crop.resize((560, int(crop.height * s_)))
        d = ImageDraw.Draw(crop)
        if bb:
            d.rectangle([(bb[0] - box[0]) * s_, (bb[1] - box[1]) * s_, (bb[2] - box[0]) * s_, (bb[3] - box[1]) * s_], outline=(255, 220, 0), width=2)
        canvas = Image.new("RGB", (560, 380), (0, 0, 0)); canvas.paste(crop.crop((0, 0, 560, min(crop.height, 350))), (0, 0))
        ImageDraw.Draw(canvas).text((4, 352), f"{s.split('-')[-1][:12]}  {labels.get(s.split('__')[-1])}  n_inst {len(feats[s].get('instances') or [])}  {(feats[s].get('instances') or [{}])[0].get('parts')}", fill=(255, 255, 255))
        row.append(canvas)
    pair = Image.new("RGB", (1130, 400), (30, 30, 30)); pair.paste(row[0], (0, 0)); pair.paste(row[1], (570, 0))
    ImageDraw.Draw(pair).text((4, 384), f"P {p}  dt {f.get('dt')}  skip {f.get('skip')}  d_pos {f.get('d_pos')}  d_size {f.get('d_size')}  d_rot {f.get('d_rot')}  d_range {f.get('d_range')}  app {f.get('app_best')}", fill=(255, 255, 0))
    tiles.append(pair)
if not tiles:
    print("no wrong links"); sys.exit(0)
cols = 2; rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (1130 * cols, 400 * rows), (0, 0, 0))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * 1130, (i // cols) * 400))
sheet.save(out, quality=88); print(out, len(tiles), "pairs")
