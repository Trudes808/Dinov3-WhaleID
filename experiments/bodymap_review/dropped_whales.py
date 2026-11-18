"""Read-only: what the mask clean-up before the body map removes (canon/predict.py keep_main_components: only the largest
blob and blobs >= 25 % of it survive). The pixel counts before and after are in DAY/canon/points.jsonl; for a sample the
segmentation is run again (inference only, nothing is written to a day) to see WHAT was removed.

    PYTHONPATH=. .venv/bin/python experiments/bodymap_review/dropped_whales.py SHEET_DIR [--days train,val] [--n 24] [--keys FILE]
"""
import argparse, json, random, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); DAYS = ROOT / "experiments" / "days_all"
ap = argparse.ArgumentParser(); ap.add_argument("sheet", type=Path); ap.add_argument("--days", default="train,val"); ap.add_argument("--n", type=int, default=24); ap.add_argument("--min-share", type=float, default=0.05)
ap.add_argument("--keys", type=Path, default=None, help="json list of [day, stem] to draw instead of a random sample"); ap.add_argument("--tag", default="removed")
a = ap.parse_args(); a.sheet.mkdir(parents=True, exist_ok=True)
split = json.loads((ROOT / "model_fine_tuning/splits/days.json").read_text()); days = [d for x in a.days.split(",") for d in (split[x] if x in split else [x])]
rows = []
for day in days:
    f = DAYS / day / "canon" / "points.jsonl"
    if f.exists():
        for l in open(f):
            r = json.loads(l)
            if r.get("whale_pixels", 0) > 0 and "whale_pixels_before_cleanup" in r:
                rows.append((day, Path(r["image"]).stem, r["whale_pixels_before_cleanup"], r["whale_pixels"]))
sh = np.array([(b - k) / k for _, _, b, k in rows]); print(len(rows), "photos with a whale mask")
for t in (0.02, 0.05, 0.10, 0.25, 0.5):
    print(f"   clean-up removed more than {t:.0%} of what it kept: {np.mean(sh > t):.1%} of photos")
if a.keys:
    sel = [tuple(x) for x in json.loads(a.keys.read_text())]
else:
    sel = [(d, s) for (d, s, b, k), x in zip(rows, sh) if x > a.min_share]; random.Random(2).shuffle(sel); sel = sel[:a.n]
import torch
from model_fine_tuning.segmentation.model import WhaleSegModel
from model_fine_tuning.segmentation.predict import segment
dev = torch.device("cuda"); model, _ = WhaleSegModel.from_checkpoint(ROOT / "experiments/seg_r04_ft4/checkpoints/best.pt", dev); tiles = []; P = 440; info = []
for i, (day, stem) in enumerate(sel):
    img = cv2.imread(str(DAYS / day / "images" / f"{stem}.jpg")); prob, mask = segment(model, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dev, 1280, 0.5)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8); areas = st[1:, cv2.CC_STAT_AREA]
    if n <= 1: continue
    keep = areas >= 0.25 * areas.max(); t = img.copy(); rem = []
    for j in range(1, n):
        cs, _ = cv2.findContours((lab == j).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if keep[j - 1]: cv2.drawContours(t, cs, -1, (0, 0, 255), 4)
        elif areas[j - 1] >= 0.02 * areas.max():
            cv2.drawContours(t, cs, -1, (0, 255, 255), 4); rem.append(round(float(areas[j - 1] / areas.max()), 2))
    t = cv2.resize(t, (P, int(P * t.shape[0] / t.shape[1])), interpolation=cv2.INTER_AREA); cv2.putText(t, f"{i + 1}  removed {rem}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    tiles.append(t); info.append({"i": i + 1, "day": day, "stem": stem, "removed": rem})
h = max(t.shape[0] for t in tiles); tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
while len(tiles) % 3: tiles.append(np.zeros_like(tiles[0]))
for s in range(0, len(tiles), 12):
    cv2.imwrite(str(a.sheet / f"{a.tag}_{s // 12 + 1:02d}.jpg"), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(s, min(s + 12, len(tiles)), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
(a.sheet / f"{a.tag}_legend.json").write_text(json.dumps(info)); print(len(tiles), "tiles ->", a.sheet, "(red outline = kept, yellow = removed blobs of 2 % of the largest or more)")
