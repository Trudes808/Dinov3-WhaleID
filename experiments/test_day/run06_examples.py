"""Real example frames for the run06 gap list (files + one sheet; untracked output only, they show crew).
    .venv/bin/python experiments/test_day/run06_examples.py DAY_DIR OUT_DIR"""
import sys
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
from sheets import tile, grid, photos
D, O = Path(sys.argv[1]), Path(sys.argv[2]); O.mkdir(parents=True, exist_ok=True)
ITEMS = [("1 crew in foreground, whale behind", ["324", "353", "386"]), ("2 bow rail + coloured tape", ["005", "072", "078", "226"]),
         ("3 bow deck / cleat close-up", ["145", "135", "389"]), ("4 gear: drone, pole camera, crossbow, rod", ["034", "265", "336"]),
         ("5 other vessel near whale", ["053", "056"]), ("6 land as subject / background", ["050", "228", "000"]),
         ("7 splash / footprint, no whale", ["247", "322", "346"])]
P = {p["seq"]: p for p in photos(D)}; tiles = []
for name, seqs in ITEMS:
    for s in seqs:
        im = Image.open(P[s]["path"]).convert("RGB"); im.thumbnail((1600, 1600)); f = O / f"item{name[0]}_{s}.jpg"; im.save(f, quality=90)
        tiles.append(tile(im, 360, 240, f"{s}", text2=name))
grid(tiles, 6, "run06 gap list: real examples per item (by-eye truth day 2026-04-07)").save(O.parent / "sheets" / "testday_run06_examples.jpg", quality=85)
print(len(tiles), "->", O)
