"""Round 2: the tape-colour rule registered in round2_rules.md, section 4 (first principles; not a clean test, see there),
run once on every photo of the day. Rail region from the ready-made label maps (readymade.py).

    .venv/bin/python experiments/test_day/tape_rule.py --root ROOT --rm ROOT/round2/readymade --out ROOT/round2/tape_rule.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from transformers import AutoConfig

ADE_RAIL = ["railing", "fence", "bannister", "pole"]; COCO_RAIL = ["fence-merged"]
BINS = {"red": lambda h: (h < 10) | (h >= 160), "yellow": lambda h: (h >= 20) & (h < 35), "green": lambda h: (h >= 35) & (h < 85), "blue": lambda h: (h >= 85) & (h < 130)}


def ids(hub: str, names: list[str]) -> list[int]:
    c = AutoConfig.from_pretrained(hub).id2label
    return [i for i, n in c.items() if n.strip() in names]


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); p.add_argument("--rm", type=Path, required=True); p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    R = a.root; D = R / "days" / R.name
    T = {r["seq"]: r for r in csv.DictReader(open(R / "truth.tsv"), delimiter="\t")}
    ai = ids("facebook/mask2former-swin-large-ade-semantic", ADE_RAIL); ci = ids("facebook/mask2former-swin-large-coco-panoptic", COCO_RAIL)
    rows = []; k = np.ones((31, 31), np.uint8)                      # 15 px reach
    for l in open(D / "images" / "index.jsonl"):
        stem = Path(json.loads(l)["image"]).stem; seq = stem.split("_")[-2]
        im = cv2.imread(str(D / "images" / f"{stem}.jpg")); H, W = im.shape[:2]
        ade = cv2.resize(cv2.imread(str(a.rm / "ade" / f"{seq}.png"), cv2.IMREAD_UNCHANGED), (W, H), interpolation=cv2.INTER_NEAREST)
        coco = cv2.resize(cv2.imread(str(a.rm / "coco" / f"{seq}.png"), cv2.IMREAD_UNCHANGED), (W, H), interpolation=cv2.INTER_NEAREST)
        rail = np.isin(ade, ai) | np.isin(coco, ci)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); h = hsv[..., 0]; s = hsv[..., 1] / 255.0; v = hsv[..., 2] / 255.0
        white = rail & (s < 0.2) & (v >= 0.55); col = rail & (s >= 0.25) & (v >= 0.15)
        near = cv2.dilate(white.astype(np.uint8), k) > 0; tape = col & near
        cnt = {b: int((tape & f(h)).sum()) for b, f in BINS.items()}; cnt["other"] = int(tape.sum()) - sum(cnt.values())
        o = sorted(cnt.items(), key=lambda x: -x[1]); call = o[0][0] if o[0][1] >= 300 and o[0][1] >= 2 * o[1][1] else "none"
        t = T[seq]; truth = t["marker"].replace("rail_", "").rstrip("?") if t["class"] == "marker_rail" else "-"
        rows.append({"seq": seq, "truth_class": t["class"], "truth_colour": truth, "call": call, "rail_px": int(rail.sum()), "white_px": int(white.sum()), **{f"tape_{b}": n for b, n in cnt.items()}})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t"); w.writeheader(); w.writerows(rows)
    from collections import Counter
    print("markers:", [(r["seq"], r["truth_colour"], r["call"]) for r in rows if r["truth_class"] == "marker_rail"])
    print("calls on other photos:", Counter((r["truth_class"], r["call"]) for r in rows if r["truth_class"] != "marker_rail" and r["call"] != "none"))


if __name__ == "__main__":
    main()
