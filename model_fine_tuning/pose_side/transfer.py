"""'The same place on two photos': carry the researcher's region centre from photo a to photo b through a canonical
frame and measure how far it lands from the researcher's region centre in b, in units of b's CLICKED fin base (front
to back) or fluke span (tip to tip).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.pose_side.transfer OUT.json [--pred PRED.json ...]

Frames (all from 2D points of one photo; the region is put in the frame of a and taken out in the frame of b):
  cutout   today's cut-outs v3 (finB: similarity centred on the part map's fin region; fluke: levelled by the principal
           axis, stock up, never mirrored); the baseline
  sim2     similarity from two points (fin: front base, back base; fluke: left tip, right tip)
  aff3     affine from three points (fin: front base, back base, tip; fluke: left tip, right tip, notch)
  hom4     homography from four points (fluke: tips, notch, tail stock 'spine_end' when clicked; fin: + 'spine_end' no)
Points come from the clicks ('click', the ceiling of the idea) or from a prediction json {key: {name: [x, y]}} (jpg px).
A region is used for the fin frame if both photos have the three fin points (from the click set, so all methods are
scored on the same regions), for the fluke frame if both have the three fluke points; if both, the nearer frame.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import CLICK_DAYS
from .realdata import apply, clicks, cutout_M, inv, pairs

FIN = ("dorsal_fin_front", "dorsal_fin_back", "dorsal_fin_tip")
FLUKE = ("fluke_tip_left", "fluke_tip_right", "fluke_notch")


def sim_from(p, q):
    """similarity taking the two points p (2x2) to q (2x2)"""
    p, q = np.asarray(p, float), np.asarray(q, float)
    a = complex(*(p[1] - p[0])); b = complex(*(q[1] - q[0])); s = b / a
    R = np.array([[s.real, -s.imag], [s.imag, s.real]]); t = q[0] - R @ p[0]
    return np.hstack([R, t[:, None]])


def frame_M(kp, part, how):
    """2x3 / 3x3 matrix from photo pixels to a canonical frame defined by the points; None if points missing."""
    names = FIN if part == "fin" else FLUKE
    if not all(n in kp for n in names):
        return None
    P = np.array([kp[n] for n in names], float)
    if part == "fin":
        C = np.array([[0.0, 0.0], [1.0, 0.0], [0.6, -1.0]])        # front, back, tip
    else:
        C = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 0.25]])        # left tip, right tip, notch (stock down)
    if how == "sim2":
        return sim_from(P[:2], C[:2])
    if how == "tb2":                                  # fin only: similarity from the back base and the tip (the two points
        if part != "fin":                             # the head finds best); the front base is not used
            return None
        return sim_from(P[1:3], C[1:3])
    if how == "aff3":
        return cv2.getAffineTransform(P.astype(np.float32), C.astype(np.float32)).astype(float)
    if how == "robust":
        # fluke: the tips and the notch are nearly in a line seen from behind, so a free affine is ill-conditioned:
        #   similarity from the tips + a shear along the span axis that puts the notch on the centre line (no stretch)
        # fin: three-point affine unless the base is foreshortened (fin seen end-on, base < 0.35 x the fin's height):
        #   then a similarity from the base's midpoint to the tip
        if part == "fluke":
            S = sim_from(P[:2], C[:2]); n = S[:, :2] @ P[2] + S[:, 2]
            sh = -n[0] / n[1] if abs(n[1]) > 0.02 else 0.0
            Sh = np.array([[1.0, sh], [0.0, 1.0]]); return np.hstack([Sh @ S[:, :2], (Sh @ S[:, 2])[:, None]])
        base = np.linalg.norm(P[1] - P[0]); mid = (P[0] + P[1]) / 2; h = np.linalg.norm(P[2] - mid)
        if base >= 0.35 * h:
            return cv2.getAffineTransform(P.astype(np.float32), C.astype(np.float32)).astype(float)
        return sim_from(np.array([mid, P[2]]), np.array([(C[0] + C[1]) / 2, C[2]]))
    if how == "hom4":
        extra = "spine_end"
        if part != "fluke" or extra not in kp:
            return None
        P4 = np.vstack([P, kp[extra]]); C4 = np.vstack([C, [0.0, 0.9]])
        H, _ = cv2.findHomography(P4.astype(np.float32), C4.astype(np.float32), 0)
        return H
    raise ValueError(how)


def unit(kp, part):
    a, b = (FIN[:2] if part == "fin" else FLUKE[:2])
    return float(np.hypot(*(np.asarray(kp[a]) - np.asarray(kp[b]))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out", type=Path); ap.add_argument("--pred", nargs="*", default=[])
    a = ap.parse_args()
    preds = {Path(p).stem: json.loads(Path(p).read_text()) for p in a.pred}
    rows = []
    for day in CLICK_DAYS:
        ck = clicks(day)
        for pr in pairs(day):
            ka, kb = pr["key_a"], pr["key_b"]
            if ka not in ck or kb not in ck:
                continue
            parts = [p for p, names in (("fin", FIN), ("fluke", FLUKE)) if all(n in ck[ka] and n in ck[kb] for n in names)]
            if not parts:
                continue
            for ca, cb, sa, sb in pr["regions"]:
                def dist(p):
                    names = FIN if p == "fin" else FLUKE
                    return np.linalg.norm(np.mean([ck[ka][n] for n in names], 0) - ca) / max(unit(ck[ka], p), 1)
                part = min(parts, key=dist)
                u = unit(ck[kb], part)
                if u < 8:
                    continue
                r = dict(day=day, key_a=ka, key_b=kb, part=part, same_burst=pr["same_burst"], unit_b=u, near=float(dist(part)),
                         region_half=float(np.hypot(*sb) / 2 / u), err={})
                Ma, Mb = cutout_M(day, ka, "finB" if part == "fin" else "fluke"), cutout_M(day, kb, "finB" if part == "fin" else "fluke")
                if Ma is not None and Mb is not None:
                    r["err"]["cutout"] = float(np.linalg.norm(apply(inv(Mb), apply(Ma, ca)) - cb) / u)
                r["err"]["none"] = float(np.linalg.norm(ca - cb) / u)   # same pixel position (only meaningful in bursts)
                srcs = {"click": ck}; srcs.update(preds)
                for sname, kps in srcs.items():
                    pa, pb = kps.get(ka), kps.get(kb)
                    if not pa or not pb:
                        continue
                    for how in ("sim2", "tb2", "aff3", "robust", "hom4"):
                        A, B = frame_M(pa, part, how), frame_M(pb, part, how)
                        if A is None or B is None:
                            continue
                        r["err"][f"{sname}_{how}"] = float(np.linalg.norm(apply(inv(B), apply(A, ca)) - cb) / u)
                rows.append(r)
    a.out.write_text(json.dumps(rows))
    summarise(rows)


def summarise(rows, methods=None):
    import collections
    methods = methods or sorted({m for r in rows for m in r["err"]})
    for part in ("fin", "fluke"):
        for sb in (True, False, None):
            R = [r for r in rows if r["part"] == part and (sb is None or r["same_burst"] == sb)]
            if not R:
                continue
            common = [r for r in R if "cutout" in r["err"]]
            print(f"\n{part} regions, {'same burst' if sb else 'not same burst' if sb is False else 'all'}: n {len(R)} "
                  f"(region half-size median {np.median([r['region_half'] for r in R]):.2f} units, region from its frame's centre median "
                  f"{np.median([r['near'] for r in R]):.2f})")
            for m in methods:
                e = np.array([r["err"][m] for r in R if m in r["err"]])
                if len(e) == 0:
                    continue
                print(f"   {m:22s} n {len(e):4d}  median {np.median(e):.3f}  p75 {np.percentile(e, 75):.3f}  "
                      f"<0.10 {np.mean(e < 0.10):.0%}  <0.25 {np.mean(e < 0.25):.0%}  <0.5 {np.mean(e < 0.5):.0%}")


if __name__ == "__main__":
    main()
