"""Result sheets of the test day (photos show crew: write only to an untracked / scratch folder).

    .venv/bin/python experiments/test_day/result_sheets.py --root ROOT --keys INST_KEYS.json --out SHEET_DIR

testday_nowhale_errors.jpg  photos the pipeline answers wrongly (no whale said whale; whale said no whale), mask in
                            magenta, kept instance boxes green, filtered boxes red
testday_false_instances.jpg every instance that is not a plain whale by eye (crew, rail, gear, splash, fused), its box on
                            the photo, the filter's answer and score, the view class
testday_seg_on_nonwhale.jpg the live whale mask on photos without a whale (by class) and on whale photos with crew
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from sheets import SHORT, grid, tile  # noqa: E402

EYE = {"W": "whale", "P": "person", "R": "rail/boat", "G": "gear", "S": "splash", "F": "whale+crew fused"}


def overlay(D: Path, stem: str, boxes=()) -> Image.Image:
    im = np.array(Image.open(D / "images" / f"{stem}.jpg").convert("RGB")).astype(np.float32)
    m = np.array(Image.open(D / "canon" / f"{stem}_mask.png")) > 0
    im[m] = im[m] * 0.45 + np.array([255, 0, 255]) * 0.55
    out = Image.fromarray(im.astype(np.uint8)); d = ImageDraw.Draw(out)
    for b, c in boxes:
        d.rectangle(list(b), outline=c, width=10)
    return out


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); p.add_argument("--keys", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    R = a.root; D = R / "days" / R.name
    PH = {r["seq"]: r for r in csv.DictReader(open(R / "photos.tsv"), delimiter="\t")}
    IN = list(csv.DictReader(open(R / "instances.tsv"), delimiter="\t"))
    stems = {k.split("~")[0].split("_")[-2]: k.split("~")[0] for k in json.loads(a.keys.read_text())}
    P = json.loads((D / "safe" / "photos.json").read_text()); drop = set(json.loads((D / "safe" / "not_whale.json").read_text())["drop"])
    for s in PH:
        stems.setdefault(s, next(k for k in P if k.split("_")[-2] == s))

    def boxes(s):
        return [(it["bbox"], (230, 25, 75) if it["key"] in drop else (60, 220, 75)) for it in P[stems[s]]["instances"]]

    # errors
    tiles = []
    for s, r in sorted(PH.items()):
        yes = r["has_whale"].rstrip("?") == "yes"; said = int(r["n_kept"]) > 0
        if r["has_whale"] == "no" and said:
            kind = "no whale -> said whale"
        elif yes and not said:
            kind = "whale -> said none" + (" (filtered)" if int(r["n_inst"]) else " (no instance)")
        else:
            continue
        tiles.append(tile(overlay(D, stems[s], boxes(s)), 420, 280, f"{s} {SHORT[r['truth_class']]}", text2=kind))
    grid(tiles, 5, f"test day: {len(tiles)} photo-level errors of 'whale in photo' (magenta = live mask, green box kept, red box filtered)").save(a.out / "testday_nowhale_errors.jpg", quality=85)
    # false instances
    tiles = []
    for r in IN:
        if r["eye"] == "W":
            continue
        s = r["seq"]; b = [float(x) for x in r["bbox"].split(",")]
        im = overlay(D, stems[s], [(b, (230, 25, 75) if r["flagged"] == "1" else (60, 220, 75))])
        tiles.append(tile(im, 300, 200, f"{s} #{r['inst']} {EYE[r['eye']]}", text2=f"{'filtered' if r['flagged'] == '1' else 'KEPT'} {float(r['nw_score']):+.1f} view {r['view']}"))
    grid(tiles, 8, f"test day: {len(tiles)} instances that are not a plain whale by eye (box red = filtered, green = kept; score > 2.0 filters)").save(a.out / "testday_false_instances.jpg", quality=82)
    # segmentation on non-whale photos
    tiles = []
    for cls in ("marker_rail", "bow", "crew_nowhale", "marker_land", "water", "under", "whale_crew"):
        sel = [s for s, r in sorted(PH.items()) if r["truth_class"] == cls]
        if cls == "whale_crew":
            sel = sel[::3]
        for s in sel[:14]:
            r = PH[s]; tiles.append(tile(overlay(D, stems[s]), 300, 200, f"{s} {SHORT[cls]}", text2=f"mask {100 * float(r['mask_share']):.1f} % of photo"))
    grid(tiles, 8, "test day: the live whale mask (magenta, seg_r04_ft4) on photos without a whale and on whale photos with crew").save(a.out / "testday_seg_on_nonwhale.jpg", quality=82)
    print("sheets ->", a.out)


if __name__ == "__main__":
    main()
