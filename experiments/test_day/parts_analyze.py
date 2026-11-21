"""Round 2: what the part candidate W3c paints on the test day (rules: round2_rules.md, section 3 + addendum).

    .venv/bin/python experiments/test_day/parts_analyze.py --root ROOT
Inputs: ROOT/round2/w3c_live and w3c_s6 (parts_w3c.py), the ready-made maps (readymade.py), the live and S6-src canon masks,
round 1's inst_eye.txt / inst_keys.json and round 2's s6_eye labels (s6_labels.json) when present.
-> ROOT/round2/parts_measures.json, ROOT/round2/parts_instances.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from readymade_analyze import GROUPS, lut_for  # noqa: E402

PART = np.array([0, 1, 2, 3, 3, 3, 0] + [0] * 249, np.uint8)       # bg, fluke, dorsal, other(pec, head, body); 255 -> bg
PART[255] = 0
PN = ["background", "fluke", "dorsal", "other"]


def call(p: dict, tf: float, rp: float, td: float) -> str:
    px = p["pixels_near"]; tot = max(p["covered"], 1); f, d, pec = px["fluke"] / tot, px["dorsal_fin"] / tot, px["pectoral_fin"] / tot
    return "fluke" if (f >= tf and f >= rp * pec) else "dorsal" if d >= td else "other"


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); a = p.parse_args()
    R = a.root; R2 = R / "round2"; day = R.name
    T = {r["seq"]: r for r in csv.DictReader(open(R / "truth.tsv"), delimiter="\t")}
    stems = {Path(json.loads(l)["image"]).stem.split("_")[-2]: Path(json.loads(l)["image"]).stem for l in open(R / "days" / day / "images" / "index.jsonl")}
    luts = {m: lut_for(m)[1] for m in ("ade", "coco")}
    canon = {"live": R / "days" / day / "canon", "s6": R2 / "days_s6" / day / "canon"}
    # ---- pixel level: full-frame map, no gate / inside the arm's mask
    byclass = {g: defaultdict(lambda: np.zeros(4, np.int64)) for g in ("none", "live", "s6")}
    bygroup = {m: {g: defaultdict(lambda: np.zeros(4, np.int64)) for g in ("none", "live", "s6")} for m in luts}
    bygroup_nw = {m: {g: defaultdict(lambda: np.zeros(4, np.int64)) for g in ("none", "live", "s6")} for m in luts}
    per_photo = []
    for seq, stem in sorted(stems.items()):
        t = T[seq]; full = PART[np.array(Image.open(R2 / "w3c_live" / day / "w3c" / "full" / f"{stem}.png"))]
        H, W = full.shape
        gates = {"none": np.ones((H, W), bool)}
        for arm, d in canon.items():
            gates[arm] = cv2.resize(np.array(Image.open(d / f"{stem}_mask.png")), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        rm = {m: cv2.resize(luts[m][np.array(Image.open(R2 / "readymade" / m / f"{seq}.png"))], (W, H), interpolation=cv2.INTER_NEAREST) for m in luts}
        row = {"seq": seq, "truth_class": t["class"]}
        for g, gm in gates.items():
            c = np.bincount(full[gm], minlength=4); byclass[g][t["class"]] += c
            for k in (1, 2, 3):
                row[f"{g}_{PN[k]}"] = round(float(c[k] / full.size), 5)
            for m in luts:
                for gi, gn in enumerate(GROUPS):
                    sel = gm & (rm[m] == gi)
                    if sel.any():
                        cc = np.bincount(full[sel], minlength=4); bygroup[m][g][gn] += cc
                        if t["has_whale"] == "no":
                            bygroup_nw[m][g][gn] += cc
        per_photo.append(row)
    sh = lambda c: {PN[k]: round(float(c[k] / max(c.sum(), 1)), 4) for k in range(4)} | {"px": int(c.sum())}
    out = {"pixels_by_truth_class": {g: {c: {**sh(v), "photos": sum(1 for r in per_photo if r["truth_class"] == c)} for c, v in d.items()} for g, d in byclass.items()},
           "frame_share_by_truth_class": {g: {c: {k: round(float(np.mean([r[f"{g}_{k}"] for r in per_photo if r["truth_class"] == c])), 5) for k in ("fluke", "dorsal", "other")}
                                              for c in sorted({r["truth_class"] for r in per_photo})} for g in ("none", "live", "s6")},
           "pixels_by_readymade_group_all_photos": {m: {g: {gn: sh(v) for gn, v in d.items()} for g, d in x.items()} for m, x in bygroup.items()},
           "pixels_by_readymade_group_nowhale_photos": {m: {g: {gn: sh(v) for gn, v in d.items()} for g, d in x.items()} for m, x in bygroup_nw.items()}}
    # ---- instance level (production): hr window counts inside the instance mask
    keys1 = json.loads((R / "inst_keys.json").read_text()); eye1 = {}
    for l in open(R / "inst_eye.txt"):
        if l.startswith("#") or not l.strip():
            continue
        r_, c = l.split(); x, _, y = r_.partition("-")
        for i in range(int(x), int(y or x) + 1):
            eye1[keys1[i]] = c
    lab6 = json.loads((R2 / "s6_labels.json").read_text()) if (R2 / "s6_labels.json").exists() else {}
    irows = []; inst = {}
    for arm, eye in (("live", eye1), ("s6", lab6.get("src", {}))):
        I = json.loads((R2 / f"w3c_{arm}" / day / "w3c" / "instances.json").read_text())["instances"]
        tab = defaultdict(lambda: np.zeros(4, np.int64)); calls = defaultdict(Counter)
        for k, v in I.items():
            if "hr" not in v:
                continue
            e = eye.get(k, "?"); px = v["hr"]["pixels_near"]
            c = np.array([px["background"], px["fluke"], px["dorsal_fin"], px["pectoral_fin"] + px["head"] + px["body"]]); tab[e] += c
            cw = call(v["hr"], 0.005, 0.0, 0.02); ca = call(v["hr"], 0.03, 1.0, 0.0005); calls[e][cw] += 1
            irows.append({"arm": arm, "key_suffix": k.split("~")[1], "seq": k.split("~")[0].split("_")[-2], "eye": e, "covered": v["hr"]["covered"], **{PN[j]: int(c[j]) for j in range(4)}, "call_w3c_rule": cw, "call_a_rule": ca})
        inst[arm] = {"pixels_by_eye": {e: sh(c) for e, c in tab.items()}, "calls_w3c_rule_by_eye": {e: dict(c) for e, c in calls.items()}, "instances": sum(sum(c.values()) for c in calls.values())}
    out["instances_hr"] = inst
    with open(R2 / "parts_instances.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(irows[0]), delimiter="\t"); w.writeheader(); w.writerows(irows)
    with open(R2 / "parts_photos.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_photo[0]), delimiter="\t"); w.writeheader(); w.writerows(per_photo)
    (R2 / "parts_measures.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out["pixels_by_truth_class"], indent=0)[:3000])


if __name__ == "__main__":
    main()
