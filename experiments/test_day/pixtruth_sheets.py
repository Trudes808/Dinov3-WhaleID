"""Sheets for the pixel-truth scoring: photo | truth | every model's map in the same false colours.

    .venv/bin/python experiments/test_day/pixtruth_sheets.py --root $R --half SCRATCH/pixtruth/half --out SCRATCH/pixtruth/sheets

pixtruth_examples.jpg: 24 photos chosen from the truth alone (seed 20261122, >= 2 photos per class with >= 2000 truth px,
rest random). pixtruth_worst.jpg: per model and scored class, the photo with the lowest IoU among photos whose truth
class has >= 500 px (from scores/per_photo.tsv). Images show crew: scratch folder only.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from pixtruth_draw import PAL, font  # noqa: E402
from pixtruth_score import GROUPS, H, W, m2f_luts, rs  # noqa: E402

GCOL = {"person": PAL[2], "boat": PAL[5], "vessel": PAL[7], "land": PAL[8], "water": PAL[0], "sky": (120, 170, 230), "other": (40, 40, 40)}
TW = 330


def colour_truth(t):
    o = np.zeros((*t.shape, 3), np.uint8)
    for c, col in PAL.items():
        o[t == c] = col
    return o


def colour_groups(g):
    o = np.zeros((*g.shape, 3), np.uint8)
    for i, n in enumerate(GROUPS):
        o[g == i] = GCOL[n]
    return o


def colour_bin(b, on=PAL[1]):
    o = np.zeros((*b.shape, 3), np.uint8); o[:] = PAL[0]; o[b > 0] = on
    return o


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, required=True); ap.add_argument("--half", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
    R = a.root; P = R / "pixel_truth"; D = R / "days" / "2026-04-07"; a.out.mkdir(parents=True, exist_ok=True)
    m = json.loads((P / "manifest.json").read_text()); luts = m2f_luts()
    stems = {p.name.split("_")[-3]: p.name[: -len("_mask.png")] for p in (D / "canon").glob("*_mask.png")}
    small = lambda arr: np.asarray(Image.fromarray(arr).resize((TW, int(TW * H / W)), Image.NEAREST if arr.ndim == 3 and arr.dtype == np.uint8 else Image.BILINEAR))

    def panels(seq):
        st = stems[seq]; t = np.array(Image.open(P / "labels" / f"{seq}.png"))
        f = R / "round2" / "w3c_live" / "2026-04-07" / "w3c" / "full" / f"{st}.png"; w3 = np.array(Image.open(f))
        return {"photo": np.asarray(Image.open(a.half / f"{seq}.jpg").convert("RGB").resize((TW, int(TW * H / W)), Image.BILINEAR)),
                "truth": colour_truth(t),
                "M2F ADE": colour_groups(rs(luts["ade"][np.array(Image.open(R / "round2" / "readymade" / "ade" / f"{seq}.png"))])),
                "M2F COCO": colour_groups(rs(luts["coco"][np.array(Image.open(R / "round2" / "readymade" / "coco" / f"{seq}.png"))])),
                "live mask": colour_bin(rs(np.array(Image.open(D / "canon" / f"{st}_mask.png")))),
                "S6h": colour_bin(rs(np.array(Image.open(R / "cand_s6" / f"{st}_s6.png")))),
                "W3c any part": colour_bin(rs(((w3 >= 1) & (w3 <= 5)).astype(np.uint8)))}

    def sheet(items, path, title):
        keys = list(panels(items[0][0]).keys()); th = int(TW * H / W)
        out = Image.new("RGB", (TW * len(keys), 60 + len(items) * (th + 22) + 60), (255, 255, 255)); d = ImageDraw.Draw(out)
        d.text((6, 6), title, fill=(0, 0, 0), font=font(18))
        for j, k in enumerate(keys):
            d.text((j * TW + 6, 34), k, fill=(0, 0, 0), font=font(16))
        for i, (seq, lab) in enumerate(items):
            pp = panels(seq); y = 60 + i * (th + 22)
            d.text((6, y), f"{seq} {lab}", fill=(0, 0, 0), font=font(15))
            for j, k in enumerate(keys):
                out.paste(Image.fromarray(small(pp[k])), (j * TW, y + 20))
        y = 60 + len(items) * (th + 22) + 8; x = 6
        for name, col in [("whale", PAL[1]), ("person", PAL[2]), ("rail", PAL[3]), ("tape", PAL[4]), ("deck / M2F boat", PAL[5]), ("pontoon", PAL[6]),
                          ("other vessel / M2F vessel", PAL[7]), ("land", PAL[8]), ("gear", PAL[9]), ("water", PAL[0]), ("M2F sky", GCOL["sky"]), ("M2F other", GCOL["other"]), ("unsure", PAL[255])]:
            d.rectangle([x, y, x + 18, y + 18], fill=col, outline=(0, 0, 0)); d.text((x + 22, y), name, fill=(0, 0, 0), font=font(15)); x += 30 + int(d.textlength(name, font=font(15)))
        out.save(path, quality=88)

    notes = {r["seq"]: r for r in csv.DictReader(open(P / "notes.tsv"), delimiter="\t")}
    cls = {s: {int(k): int(v) for k, v in zip(*np.unique(np.array(Image.open(P / "labels" / f"{s}.png")), return_counts=True))} for s in m["sample"]}
    rng = random.Random(20261122); chosen = []
    for c in [2, 3, 4, 5, 6, 7, 8, 9, 1]:
        pool = sorted(s for s in m["sample"] if cls[s].get(c, 0) >= 2000 and s not in chosen); rng.shuffle(pool)
        while sum(cls[s].get(c, 0) >= 2000 for s in chosen) < 2 and pool:
            chosen.append(pool.pop())
    rest = sorted(s for s in m["sample"] if s not in chosen); rng.shuffle(rest); chosen = sorted(chosen + rest[: 24 - len(chosen)])
    sheet([(s, notes[s]["photo_class"]) for s in chosen], a.out / "pixtruth_examples.jpg", "Pixel truth (my eye + SAM 2.1) vs models, test day 2026-04-07; photos chosen from the truth alone")
    worst = []
    rows = list(csv.DictReader(open(P / "scores" / "per_photo.tsv"), delimiter="\t"))
    for mdl, pair in [("ade", "person"), ("ade", "any_boat"), ("ade", "land"), ("coco", "person"), ("coco", "any_boat"), ("coco", "land"),
                      ("live", "whale"), ("s6h", "whale"), ("w3c_anypart", "whale")]:
        sel = [r for r in rows if r["model"] == mdl and r["pair"] == pair and int(r["truth_px"]) >= 500 and r["iou"] != ""]
        r = min(sel, key=lambda r: float(r["iou"])); worst.append((r["seq"], f"worst {mdl} {pair}: IoU {float(r['iou']):.3f}"))
    sheet(worst, a.out / "pixtruth_worst.jpg", "Worst photo per model and class (lowest per-photo IoU, truth class >= 500 px)")
    (a.out / "examples_list.json").write_text(json.dumps({"examples": chosen, "worst": worst}))
    print("examples", chosen); print("worst", worst)


if __name__ == "__main__":
    main()
