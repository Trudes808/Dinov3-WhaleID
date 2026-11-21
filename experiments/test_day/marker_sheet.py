"""Every marker photo of the test day (rail tape, land, bow cleat) on one sheet, with the time and the gaps around it.
    .venv/bin/python experiments/test_day/marker_sheet.py DAY_DIR ROOT OUT.jpg"""
import sys
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
from sheets import tile, grid, photos, truth, COLOURS, SHORT
D, R, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
P = photos(D); T = truth(R); seqs = [p["seq"] for p in P]; t = {s: float(T[s]["t"]) for s in seqs}
tiles = []
for i, p in enumerate(P):
    r = T[p["seq"]]
    if r["class"] not in ("marker_rail", "marker_land", "bow"):
        continue
    mk = r["marker"].replace("rail_", "").rstrip("?")
    before = t[p["seq"]] - t[seqs[i - 1]] if i else 0; after = t[seqs[i + 1]] - t[p["seq"]] if i + 1 < len(seqs) else 0
    lab = f"{p['seq']} {SHORT[r['class']]}" + (f" {mk}" if r["class"] == "marker_rail" else "")
    tiles.append(tile(Image.open(p["path"]), 480, 320, lab, border=COLOURS[r["class"]], bw=7,
                      text2=f"{r['time'][:8]}  gap before {before:.0f} s, after {after:.0f} s"))
grid(tiles, 4, f"test day 2026-04-07: {len(tiles)} marker photos (rail tape / land / bow cleat), frame order").save(out, quality=88)
print(len(tiles), "markers ->", out)
