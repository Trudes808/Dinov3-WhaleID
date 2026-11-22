"""C3: markers from a class map (ours, or the labeller's map), rule fixed in rules_labeller.md part C before any run.

    .venv/bin/python experiments/sam3_labeller/markers.py --name NAME [--src pred|labeller] --out OUT/markers_NAME.json
Test day scored against pixel_truth/markers.tsv and tape_colours.tsv; val days: false calls only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lab_common as L  # noqa: E402

PT = L.TD / "pixel_truth"


def hue_class(h: float) -> str:
    if h < 20 or h >= 320:
        return "red"
    return "yellow" if h < 70 else "green" if h < 170 else "blue" if h < 260 else "other"


def facts(lab: np.ndarray) -> dict:
    H, W = lab.shape; A = H * W
    rt = ((lab == 3) | (lab == 4)).astype(np.uint8)
    tube = 2 * float(cv2.distanceTransform(rt, cv2.DIST_L2, 5).max()) / W if rt.any() else 0.0
    tape = (lab == 4).astype(np.uint8); n, cc, st, _ = cv2.connectedComponentsWithStats(tape, connectivity=8)
    big = int(st[1:, cv2.CC_STAT_AREA].argmax()) + 1 if n > 1 else 0
    return {"whale_px": int((lab == 1).sum()), "rail_tape": float(rt.mean()), "tube": tube, "tape_big": float(st[big, cv2.CC_STAT_AREA] / A) if big else 0.0,
            "land": float((lab == 6).mean()), "deck": float((lab == 5).mean()), "person": float((lab == 2).mean()), "_big": (cc == big) if big else None}


def call(f: dict) -> tuple[str, bool]:
    rail = f["rail_tape"] >= 0.04 and f["tube"] >= 0.055 and f["tape_big"] >= 0.0012
    if f["whale_px"] >= 50:
        return "none", rail
    if rail:
        return "rail", rail
    if f["land"] >= 0.25:
        return "land", rail
    if f["deck"] >= 0.10 and f["person"] <= 0.10:
        return "bollard", rail
    return "none", rail


def tape_colour(big: np.ndarray | None, photo_bgr: np.ndarray) -> str:
    if big is None:
        return "none"
    h, w = photo_bgr.shape[:2]
    m = cv2.resize(big.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    k = max(1, int(round(0.002 * w))); m = cv2.erode(m, np.ones((k, k), np.uint8)).astype(bool)
    hsv = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2HSV)
    Hh = hsv[..., 0][m].astype(np.float64) * 2; S = hsv[..., 1][m] / 255.0; V = hsv[..., 2][m] / 255.0
    col = (S >= 0.25) & (V >= 0.20)
    if col.sum() < 200:
        return "none"
    ang = np.deg2rad(Hh[col]); return hue_class((np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) + 360) % 360)


def load_map(src: str, name: str, day: str, stem: str) -> np.ndarray:
    if src == "pred":
        return cv2.imread(str(L.OUT / "pred" / name / day / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    return cv2.imread(str(L.OUT / ("labeller_test" if day == L.TEST else "pseudo_val") / day / f"{stem}.png"), cv2.IMREAD_UNCHANGED)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--src", default="pred", choices=["pred", "labeller"])
    ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
    mk = {r["seq"]: r for r in (dict(zip(h, l.rstrip("\n").split("\t"))) for h in [open(PT / "markers.tsv").readline().rstrip("\n").split("\t")]
                                for l in list(open(PT / "markers.tsv"))[1:])}
    tc = {l.split("\t")[0]: l.split("\t")[1].strip() for l in open(PT / "tape_colours.tsv") if not l.startswith("seq")}
    rows = {}
    for stem in L.day_stems_any(L.TEST):
        seq = L.seq_of(stem); lab = load_map(a.src, a.name, L.TEST, stem); lab = np.where(lab == 255, 0, lab).astype(np.uint8)
        f = facts(lab); c, railpass = call(f)
        col = tape_colour(f["_big"], cv2.imread(str(L.photo_path(L.TEST, stem)))) if (c == "rail" or seq in tc or f["tape_big"] > 0) else "none"
        rows[seq] = {"truth": mk[seq]["marker"], "call": c, "rail_rule_with_whale": bool(railpass and c != "rail"), "colour": col,
                     **{k: v for k, v in f.items() if not k.startswith("_")}}
    kinds = ["rail", "bollard", "land"]
    summ = {k: {"n": sum(r["truth"] == k for r in rows.values()), "found": sum(r["truth"] == k and r["call"] != "none" for r in rows.values()),
                "right_kind": sum(r["truth"] == k and r["call"] == k for r in rows.values())} for k in kinds}
    others = [s for s, r in rows.items() if r["truth"] == "none"]
    summ["false_markers"] = {"n_other": len(others), "n": sum(rows[s]["call"] != "none" for s in others), "seqs": [s for s in others if rows[s]["call"] != "none"],
                             "calls": {k: sum(rows[s]["call"] == k for s in others) for k in kinds}}
    summ["rail_rule_passed_with_whale"] = [s for s, r in rows.items() if r["rail_rule_with_whale"]]
    summ["tape_colour"] = {"right": sum(tc[s].split()[0] == rows[s]["colour"] for s in tc), "n": len(tc),
                           "per_marker": {s: [tc[s].split()[0], rows[s]["colour"], rows[s]["call"]] for s in sorted(tc)},
                           "false_colour_calls_other_photos": sum(1 for s, r in rows.items() if s not in tc and r["colour"] != "none")}
    val = {"n": 0, "calls": {k: 0 for k in kinds}, "stems": []}
    for day in L.VAL:
        for stem in L.day_stems(day):
            lab = load_map(a.src, a.name, day, stem)
            if lab is None:
                continue
            lab = np.where(lab == 255, 0, lab).astype(np.uint8); c, _ = call(facts(lab)); val["n"] += 1
            if c != "none":
                val["calls"][c] += 1; val["stems"].append(f"{day}/{stem}")
    summ["val_days"] = val
    print(json.dumps({k: v for k, v in summ.items()}, indent=1)[:3000])
    a.out.write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
