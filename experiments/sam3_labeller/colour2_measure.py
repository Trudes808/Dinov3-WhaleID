"""Tape colour, round 2 (colour2): measure every tape band the scene model paints on the real rail markers of the two
unculled days (2026-04-07 dev, 2026-03-30 the spent held-out day) and on the boat marker renders. Text only in the repo;
per-photo tables and sheets go to the scratchpad / review_sheets (photos show the boat and crew).

    .venv/bin/python experiments/sam3_labeller/colour2_measure.py real    --out SCRATCH/colour2/bands_real.json
    .venv/bin/python experiments/sam3_labeller/colour2_measure.py renders --out SCRATCH/colour2/bands_renders.json

Per photo: every 8-connected tape component of the scene map (class 4 at 1280) with >= 0.02 % of the frame, measured on
the photo the reader reads (2560 px day JPEG / 1280 render JPEG): area, centre distance, sharpness, the reader's HSV
statistics, CIE Lab, and the white tube beside the band (rail pixels in a ring round the band) as a local white.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

NW = Path("/home/sat3737/NOAA_whales")
RUNS = NW / "moonshot_cache" / "sam3_labeller" / "v3" / "day_runs"
TD = NW / "moonshot_cache" / "test_days"
V3 = NW / "moonshot_cache" / "sam3_labeller" / "v3"
FACTORY = NW / "Whale-Image-Data-Factory" / "data" / "renders"
CREW = {"2026-03-30": {"137": "red", "140": "green", "141": "red"}}     # truth_final.txt notes: band near centre, not markers


def truth_colours() -> dict:
    out = {}
    for day, f in (("2026-04-07", TD / "2026-04-07" / "pixel_truth" / "tape_colours.tsv"),
                   ("2026-03-30", TD / "2026-03-30" / "truth" / "tape_colours.tsv")):
        for r in csv.DictReader(open(f), delimiter="\t"):
            out[(day, r["seq"])] = {"truth": r["tape_colour_by_eye"].split()[0], "note": r["tape_colour_by_eye"], "role": "marker"}
        for s, c in CREW.get(day, {}).items():
            out[(day, s)] = {"truth": c, "note": "crew photo, not a marker", "role": "crew"}
    return out


def circ_mean_deg(h: np.ndarray) -> float:
    a = np.deg2rad(h)
    return float((np.rad2deg(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) + 360) % 360)


def band_stats(comp: np.ndarray, lab: np.ndarray, photo: np.ndarray) -> dict:
    """comp: bool component at the scene grid; lab: scene map; photo: BGR at its own size."""
    H, W = photo.shape[:2]; h0, w0 = lab.shape
    m = cv2.resize(comp.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    k = max(1, int(round(0.002 * W))); me = cv2.erode(m, np.ones((k, k), np.uint8)).astype(bool)
    hsv = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)
    Hh = hsv[..., 0][me].astype(np.float64) * 2; S = hsv[..., 1][me] / 255.0; V = hsv[..., 2][me] / 255.0
    col = (S >= 0.25) & (V >= 0.20)
    rgb = photo[me][:, ::-1].astype(np.float64)                     # R, G, B
    labc = cv2.cvtColor(photo, cv2.COLOR_BGR2LAB)[me].astype(np.float64)
    # white tube beside the band: rail pixels in a ring 2-30 px (scene grid) round the band
    ring = (cv2.dilate(comp.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0) & ~(cv2.dilate(comp.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & (lab == 3)
    rm = cv2.resize(ring.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    wrgb = photo[rm][:, ::-1].astype(np.float64) if rm.any() else np.zeros((0, 3))
    if len(wrgb) >= 50:
        lum = wrgb.mean(1); sel = wrgb[lum >= np.percentile(lum, 50)]  # brighter half: the lit side of the white tube
        white = np.median(sel, 0)
    else:
        white = None
    ys, xs = np.nonzero(comp)
    cy, cx = ys.mean() / h0, xs.mean() / w0
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    g = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    sx, sy = W / w0, H / h0
    crop = g[int(y0 * sy):int((y1 + 1) * sy), int(x0 * sx):int((x1 + 1) * sx)]
    sharp = float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size > 25 else 0.0
    d = {"area": float(comp.mean()), "cx": round(cx, 4), "cy": round(cy, 4),
         "dist": round(float(np.hypot(cx - 0.5, cy - 0.5) / np.hypot(0.5, 0.5)), 4), "bbox": [int(x0), int(y0), int(x1), int(y1)],
         "sharp": round(sharp, 2), "n_px": int(me.sum()), "n_col": int(col.sum()),
         "hue_reader": round(circ_mean_deg(Hh[col]), 2) if col.sum() else None,
         "hue_all": round(circ_mean_deg(Hh[S >= 0.08]), 2) if (S >= 0.08).sum() else None,
         "S_mean": round(float(S.mean()), 4) if len(S) else None, "V_mean": round(float(V.mean()), 4) if len(V) else None,
         "S_col": round(float(S[col].mean()), 4) if col.sum() else None, "V_col": round(float(V[col].mean()), 4) if col.sum() else None,
         "rgb_med": [round(float(x), 1) for x in np.median(rgb, 0)] if len(rgb) else None,
         "rgb_mean": [round(float(x), 1) for x in rgb.mean(0)] if len(rgb) else None,
         "lab_mean": [round(float(x), 2) for x in labc.mean(0)] if len(labc) else None,     # OpenCV 8-bit Lab (L*2.55, a+128, b+128)
         "white_rgb": [round(float(x), 1) for x in white] if white is not None else None, "white_n": int(len(wrgb))}
    return d


def photo_bands(lab: np.ndarray, photo: np.ndarray, min_frac: float = 0.0002) -> list[dict]:
    lab = np.where(lab == 255, 0, lab).astype(np.uint8)
    tape = (lab == 4).astype(np.uint8); n, cc, st, _ = cv2.connectedComponentsWithStats(tape, connectivity=8)
    A = lab.size; out = []
    big = int(st[1:, cv2.CC_STAT_AREA].argmax()) + 1 if n > 1 else 0
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] / A < min_frac and i != big:
            continue
        d = band_stats(cc == i, lab, photo); d["largest"] = i == big; out.append(d)
    return out


def seq_of(stem: str) -> str:
    return stem.split("_")[-2]


def cmd_real(a) -> None:
    T = truth_colours(); res = []
    for day in ("2026-04-07", "2026-03-30"):
        recs = {seq_of(json.loads(l)["photo"]): json.loads(l) for l in open(RUNS / f"{day}_markers.jsonl")}
        stems = {seq_of(p.stem): p.stem for p in (RUNS / "days" / day / "images").glob("*.jpg")}
        for (d, seq), t in sorted(T.items()):
            if d != day:
                continue
            stem = stems[seq]; r = recs[seq]
            lab = cv2.imread(str(RUNS / "days" / day / "scene3" / "scene" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
            photo = cv2.imread(str(RUNS / "days" / day / "images" / f"{stem}.jpg"))
            res.append({"day": day, "seq": seq, **t, "old_call": r["tape_colour_any"], "old_kind": r["marker"], "bands": photo_bands(lab, photo)})
            print(day, seq, t["truth"], r["tape_colour_any"], len(res[-1]["bands"]), file=sys.stderr)
    Path(a.out).write_text(json.dumps(res, indent=1))


def cmd_renders(a) -> None:
    """every rail marker render of r06r_boat22_markers: exact tape label (rendered scene map) and, for the held-out 20 %,
    the frozen scene model's map too."""
    hold = {x["stem"] for x in json.loads((V3 / "renders" / "hold" / "index.json").read_text()) if x["day"] == "r06r_boat22_markers"}
    res = []
    for part in ("train", "hold"):
        for x in json.loads((V3 / "renders" / part / "index.json").read_text()):
            if x["day"] != "r06r_boat22_markers" or x["shot_type"] != "marker_rail":
                continue
            d = V3 / "renders" / part / x["day"]
            photo = cv2.imread(str(d / f"{x['stem']}.jpg")); exact = cv2.imread(str(d / f"{x['stem']}.png"), cv2.IMREAD_UNCHANGED)
            exact = np.where(exact > 6, 0, exact)
            row = {"run": x["day"], "id": x["stem"], "part": part, "truth": (x.get("marker") or {}).get("colour"), "marker": x.get("marker"),
                   "bands_exact": photo_bands(exact, photo)}
            pf = V3 / "pred" / "seg_scene_v3b_s0" / x["day"] / f"{x['stem']}.png"
            if x["stem"] in hold and pf.exists():
                row["bands_pred"] = photo_bands(cv2.imread(str(pf), cv2.IMREAD_UNCHANGED), photo)
            res.append(row)
    print(len(res), "rail renders", sum("bands_pred" in r for r in res), "with the frozen model's map", file=sys.stderr)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["real", "renders"]); ap.add_argument("--out", required=True)
    a = ap.parse_args(); {"real": cmd_real, "renders": cmd_renders}[a.cmd](a)
