"""Sheet: live mask (magenta) beside the candidate S6 mask (cyan) on whale photos with crew and other hard photos.
    .venv/bin/python experiments/test_day/s6_sheet.py ROOT OUT.jpg SEQ [SEQ ...]"""
import sys, csv, numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
from sheets import tile, grid
R = Path(sys.argv[1]); D = R / "days" / R.name; out = sys.argv[2]; want = sys.argv[3:]
stems = {Path(l.split('"image": "')[1].split('"')[0]).stem.split("_")[-2]: Path(l.split('"image": "')[1].split('"')[0]).stem for l in open(D / "images/index.jsonl")}
T = {r["seq"]: r for r in csv.DictReader(open(R / "truth.tsv"), delimiter="\t")}
tiles = []
for s in want:
    st = stems[s]; im = np.array(Image.open(D / "images" / f"{st}.jpg").convert("RGB")).astype(np.float32)
    for name, m, c in (("live", np.array(Image.open(D / "canon" / f"{st}_mask.png")) > 0, (255, 0, 255)),
                       ("S6 cand.", np.array(Image.open(R / "cand_s6" / f"{st}_s6.png").resize((im.shape[1], im.shape[0]), Image.NEAREST)) > 0, (0, 255, 255))):
        a = im.copy(); a[m] = a[m] * 0.45 + np.array(c) * 0.55
        tiles.append(tile(Image.fromarray(a.astype(np.uint8)), 330, 220, f"{s} {name}", text2=f"{T[s]['class']}  {100 * m.mean():.1f} %"))
grid(tiles, 6, "live mask seg_r04_ft4 (magenta) vs candidate seg_splash_s6 + hysteresis (cyan, NOT live)").save(out, quality=85); print(out)
