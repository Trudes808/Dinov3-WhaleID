"""What the part map paints inside the whale, per view (validation days), for the cut-out blockers.

    PYTHONPATH=. .venv/bin/python experiments/parts_weak/shares.py --models parts_seg_r05_a parts_seg_r05_s1 ... \
        --out experiments/parts_weak/shares_w2.json

Per model, variant (hr / full) and view (by eye, else the confident view classifier): over the instances, the median
share of the instance (pixels inside the instance mask) painted background, the share of instances more than half
background, and in fluke views the median body and head share (is the tail stock still painted fluke?) and fluke share.
Paired bootstrap over photos (2000, 95 %) of "more than half background" for each model against each of the first two
(the seed-0 and seed-1 baselines).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import compare as CP  # noqa: E402
from model_fine_tuning.common import paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    val = json.loads((paths.SPLITS_DIR / "days.json").read_text())["val"]
    rows = {m: CP.rows_of(m, val) for m in a.models}
    res = {}
    for var in ("hr", "full"):
        res[var] = {}
        half = {}
        for m in a.models:
            res[var][m] = {}
            half[m] = {}
            for v in ("fluke", "dorsal", "other"):
                sel = [r for r in rows[m] if var in r["pred"] and not r["not_whale"] and CP.view_of(r) == v]
                px = [r["pred"][var]["pixels"] for r in sel]
                tot = [max(sum(p.values()), 1) for p in px]
                bg = np.array([p["background"] / t for p, t in zip(px, tot)])
                o = {"n": len(sel), "bg_median": round(float(np.median(bg)), 3), "over_half_bg": round(float(np.mean(bg > 0.5)), 3)}
                for c in ("fluke", "dorsal_fin", "body", "head", "pectoral_fin"):
                    o[f"{c}_median"] = round(float(np.median([p[c] / t for p, t in zip(px, tot)])), 3)
                res[var][m][v] = o
                half[m][v] = {r["key"]: (r["stem"], int(b > 0.5)) for r, b in zip(sel, bg)}
        res[var]["paired_over_half_bg"] = {f"{m}-{b}": {v: CP.paired(half[b][v], half[m][v]) for v in ("fluke", "dorsal", "other")}
                                           for b in a.models[:2] for m in a.models if m != b}
    a.out.write_text(json.dumps(res, indent=1))
    for var in ("hr", "full"):
        for m in a.models:
            print(var, f"{m:26s}", {v: (x["bg_median"], x["over_half_bg"], x["body_median"], x["fluke_median"]) for v, x in res[var][m].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
