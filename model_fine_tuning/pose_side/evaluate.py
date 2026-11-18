"""Score the predicted points against the researcher's clicks (2025-01-02, 2025-03-18) and in the region transfer.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.pose_side.evaluate --out experiments/pose_side/results.json \
        A=/var/tmp/.../pred_A.json Bx=/var/tmp/.../pred_B_0102.json+/var/tmp/.../pred_B_0318.json

Distances in units of the CLICKED fin base (front to back) for fin points and the CLICKED tip-to-tip span for fluke
points (as experiments/bodymap_review/click_probe.py). Bx = each click day predicted by the head fine-tuned on the
other click day's clicks only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import CACHE, CLICK_DAYS
from .data import KP
from .realdata import clicks
from . import transfer as T


def q(v):
    v = np.asarray(v, float)
    return dict(n=int(len(v)), median=round(float(np.median(v)), 3), p75=round(float(np.percentile(v, 75)), 3),
                within_010=round(float(np.mean(v < 0.10)), 3), within_025=round(float(np.mean(v < 0.25)), 3)) if len(v) else dict(n=0)


def score(P, meta=None):
    res = {n: [] for n in KP}; pair, swap, fin_c, fin_size, fl3, rows = [], [], [], [], [], []
    for day in CLICK_DAYS:
        for key, kp in clicks(day).items():
            if key not in P:
                continue
            p = {n: np.array(v) for n, v in P[key].items()}; c = {n: np.array(v) for n, v in kp.items()}
            fb = np.linalg.norm(c["dorsal_fin_front"] - c["dorsal_fin_back"]) if all(n in c for n in T.FIN[:2]) else None
            fs = np.linalg.norm(c["fluke_tip_left"] - c["fluke_tip_right"]) if all(n in c for n in T.FLUKE[:2]) else None
            for n in KP:
                u = fb if n.startswith("dorsal") else fs
                if n in c and u and u >= 8:
                    res[n].append(float(np.linalg.norm(p[n] - c[n]) / u))
            if fs and fs >= 8:
                L, R = c["fluke_tip_left"], c["fluke_tip_right"]; Lp, Rp = p["fluke_tip_left"], p["fluke_tip_right"]
                st = np.linalg.norm(Lp - L) + np.linalg.norm(Rp - R); cr = np.linalg.norm(Lp - R) + np.linalg.norm(Rp - L)
                pair.append(min(st, cr) / 2 / fs); swap.append(bool(cr < st))
                if "fluke_notch" in c:
                    e = max(np.linalg.norm(Lp - L), np.linalg.norm(Rp - R), np.linalg.norm(p["fluke_notch"] - c["fluke_notch"])) / fs
                    fl3.append(float(e)); rows.append(dict(key=key, day=day, part="fluke", err=float(e), conf=min(meta[key]["conf"][n] for n in T.FLUKE) if meta else None))
            if fb and fb >= 8 and all(n in c for n in T.FIN):
                Cc = np.mean([c[n] for n in T.FIN], 0); Cp = np.mean([p[n] for n in T.FIN], 0)
                fin_c.append(float(np.linalg.norm(Cc - Cp) / fb))
                e = max(np.linalg.norm(p[n] - c[n]) for n in T.FIN) / fb
                rows.append(dict(key=key, day=day, part="fin", err=float(e), conf=min(meta[key]["conf"][n] for n in T.FIN) if meta else None))
    out = {n: q(v) for n, v in res.items()}
    out["fluke_tips_as_pair"] = q(pair); out["fluke_left_right_swapped"] = round(float(np.mean(swap)), 3) if swap else None
    out["fluke_all_three_worst"] = q(fl3)
    fc = np.array(fin_c); out["fin_centre_off"] = dict(q(fc), over_one_base=round(float(np.mean(fc > 1)), 3)) if len(fc) else {}
    if meta:                       # coverage: keep the most confident share of instances, worst point error of the three
        cov = {}
        for part in ("fin", "fluke"):
            R = sorted([r for r in rows if r["part"] == part], key=lambda r: -r["conf"])
            for share in (0.5, 0.7, 0.9, 1.0):
                S = R[:max(1, int(round(share * len(R))))]; e = np.array([r["err"] for r in S])
                cov[f"{part}_{int(share * 100)}"] = dict(n=len(S), worst_point_median=round(float(np.median(e)), 3), worst_within_025=round(float(np.mean(e < 0.25)), 3))
        out["coverage_by_confidence"] = cov
    return out, rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("preds", nargs="+")
    a = ap.parse_args(); allres = {}; files = []
    for spec in a.preds:
        name, paths = spec.split("=", 1); P, META = {}, {}
        for p in paths.split("+"):
            P.update(json.loads(Path(p).read_text())); mp = Path(p).with_name(Path(p).stem + "_meta.json")
            if mp.exists():
                META.update(json.loads(mp.read_text())["rows"])
        merged = CACHE / f"merged_{name}.json"; merged.write_text(json.dumps(P)); files.append(merged)
        allres[name], _ = score(P, META or None)
    print(json.dumps(allres, indent=1))
    rows = []
    import sys
    sys.argv = ["transfer", str(CACHE / "transfer_rows.json"), "--pred", *map(str, files)]
    T.main()
    rows = json.loads((CACHE / "transfer_rows.json").read_text())
    tr = {}
    for part in ("fin", "fluke"):
        for sb, lab in ((True, "same_burst"), (False, "other"), (None, "all")):
            R = [r for r in rows if r["part"] == part and (sb is None or r["same_burst"] == sb)]
            for m in sorted({m for r in R for m in r["err"]}):
                e = [r["err"][m] for r in R if m in r["err"]]
                if e:
                    tr[f"{part}|{lab}|{m}"] = q(e)
    allres["region_transfer"] = tr
    a.out.write_text(json.dumps(allres, indent=1))


if __name__ == "__main__":
    main()
