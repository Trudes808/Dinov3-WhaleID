"""Verification set for the pixel truth (the user checks a sample by eye). The list is fixed in the manifest ("verify").

    .venv/bin/python experiments/test_day/pixtruth_verify.py --root $R --half SCRATCH/pixtruth/half --out SCRATCH/pixtruth/verify

Writes V01.jpg ... V30.jpg (photo left, truth right: outlines + light fill, "unsure" hatched grey, legend, V-number and
sequence number; rail markers get a zoomed row on the tape), verify_overview_1.jpg / _2.jpg and verify_list.tsv.
Images show crew: scratch folder only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pixtruth_draw import NAMES, PAL, font

PLAIN = {0: "water / sky", 1: "whale", 2: "person", 3: "rail tube", 4: "tape on rail", 5: "deck / bow / cleat", 6: "pontoon / hull",
         7: "other vessel", 8: "land", 9: "gear (drone, pole, rod, crossbow)", 255: "unsure, not scored (hatched)"}


def draw(img: np.ndarray, lab: np.ndarray, alpha=0.28, lw=4) -> np.ndarray:
    o = img.astype(np.float32).copy()
    for c, col in PAL.items():
        if c in (0, 255):
            continue
        m = lab == c
        if m.any():
            o[m] = o[m] * (1 - alpha) + np.array(col) * alpha
    u = lab == 255
    if u.any():
        yy, xx = np.mgrid[: lab.shape[0], : lab.shape[1]]
        hatch = ((xx + yy) // 6) % 4 == 0
        o[u] = o[u] * 0.6 + np.array([150, 150, 150]) * 0.4
        o[u & hatch] = (60, 60, 60)
    o = o.clip(0, 255).astype(np.uint8)
    for c, col in PAL.items():
        if c == 0:
            continue
        cs, _ = cv2.findContours((lab == c).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(o, cs, -1, col if c != 255 else (90, 90, 90), lw)
    return o


def legend(width: int, present: list[int]) -> Image.Image:
    im = Image.new("RGB", (width, 70), (255, 255, 255)); d = ImageDraw.Draw(im); f = font(20); x, y = 10, 8
    for c in [1, 2, 3, 4, 5, 6, 7, 8, 9, 255]:
        txt = PLAIN[c] + ("" if c in present else " (none here)")
        w = int(d.textlength(txt, font=f)) + 50
        if x + w > width:
            x, y = 10, y + 30
        d.rectangle([x, y + 2, x + 28, y + 22], fill=PAL[c] if c != 255 else (150, 150, 150), outline=(0, 0, 0))
        d.text((x + 36, y), txt, fill=(0, 0, 0) if c in present else (150, 150, 150), font=f); x += w
    return im


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, required=True); ap.add_argument("--half", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
    P = a.root / "pixel_truth"; m = json.loads((P / "manifest.json").read_text()); a.out.mkdir(parents=True, exist_ok=True)
    notes = {r["seq"]: r for r in csv.DictReader(open(P / "notes.tsv"), delimiter="\t")}
    PW = 1200; small = []
    rows = []
    for v, s in m["verify"]["list"].items():
        img = np.asarray(Image.open(a.half / f"{s}.jpg").convert("RGB")); lab = np.asarray(Image.open(P / "labels" / f"{s}.png"))
        ov = draw(img, lab)
        H, W = lab.shape; ph = int(PW * H / W)
        left = Image.fromarray(img).resize((PW, ph), Image.LANCZOS); right = Image.fromarray(ov).resize((PW, ph), Image.LANCZOS)
        panels = [(left, right)]
        if (lab == 4).any() and notes[s]["photo_class"] == "marker_rail":
            ys, xs = np.nonzero(lab == 4); cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
            hw = max(220, int((xs.max() - xs.min()) * 1.2)); hh = int(hw * ph / PW)
            x0, y0 = max(0, cx - hw), max(0, cy - hh); x1, y1 = min(W, x0 + 2 * hw), min(H, y0 + 2 * hh)
            panels.append((Image.fromarray(img[y0:y1, x0:x1]).resize((PW, ph), Image.LANCZOS), Image.fromarray(draw(img[y0:y1, x0:x1], lab[y0:y1, x0:x1], lw=3)).resize((PW, ph), Image.LANCZOS)))
        present = [int(k) for k in np.unique(lab) if k != 0]
        lg = legend(2 * PW, present)
        out = Image.new("RGB", (2 * PW, 44 + len(panels) * ph + lg.size[1]), (255, 255, 255)); d = ImageDraw.Draw(out)
        d.text((10, 8), f"{v}  photo {s}   left: photo   right: truth (my eye + SAM 2.1)" + ("   lower row: zoom on the tape" if len(panels) > 1 else ""), fill=(0, 0, 0), font=font(24))
        for i, (l, r) in enumerate(panels):
            out.paste(l, (0, 44 + i * ph)); out.paste(r, (PW, 44 + i * ph))
        out.paste(lg, (0, 44 + len(panels) * ph)); out.save(a.out / f"{v}.jpg", quality=90)
        t = Image.fromarray(ov).resize((600, int(600 * H / W)), Image.LANCZOS); ImageDraw.Draw(t).rectangle([0, 0, 150, 30], fill=(0, 0, 0))
        ImageDraw.Draw(t).text((6, 4), f"{v} {s}", fill=(255, 255, 255), font=font(20)); small.append(t)
        rows.append((v, s, ",".join(NAMES[c] for c in present), notes[s]["sureness"], notes[s]["note"], ""))
    for k in range(2):
        part = small[k * 15:(k + 1) * 15]; th = part[0].size[1]
        sheet = Image.new("RGB", (3 * 600, 5 * th + 70), (255, 255, 255))
        for i, t in enumerate(part):
            sheet.paste(t, ((i % 3) * 600, (i // 3) * th))
        sheet.paste(legend(1800, list(PLAIN)), (0, 5 * th)); sheet.save(a.out / f"verify_overview_{k + 1}.jpg", quality=88)
    with open(a.out / "verify_list.tsv", "w") as f:
        f.write("V\tseq\tclasses_present\tmy_sureness\tmy_note\tcorrect? / what is wrong\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    print("wrote", len(rows), "to", a.out)


if __name__ == "__main__":
    main()
