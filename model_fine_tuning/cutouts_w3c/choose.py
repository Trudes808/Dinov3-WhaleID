"""The two cut-out sizes, chosen on the researcher's clicked fins of the training days 2025-01-02 and 2025-03-18 by
geometry alone (no identity score is read), and the geometry table of the clicked fins for A (parts_up) and the W3c
variants.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_w3c.choose explore         # distributions, nothing written
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_w3c.choose fix --axis A --wide K --tight K   # writes RESULTS/rule.json
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_w3c.choose check [--seed s0]  # geometry table -> RESULTS/clickcheck_<seed>.json

Cut-out of a fin = a square in the parts_up fin cut-out's turn (chart axes ex, ey), centred on the W3c region's
centroid, side = k x L, L = the region's extent along the back (axis chosen in "explore": the one whose L / clicked
base varies least). Triangle = front base, back base, tip as clicked. Margin m: every click at least m x side from
every edge.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from PIL import Image

from . import CACHE, DAYS, RESULTS
from ..cutouts_v3.clickcheck import CLICK_DAYS, clicks

FIN_KP = ("dorsal_fin_front", "dorsal_fin_back", "dorsal_fin_tip")


def clicked(seed: str = "s0", days=CLICK_DAYS):
    """[(day, key, triangle (3x2, 2560 px), region record, A record)] for clicked fins with a parts_up fin cut-out."""
    out = []
    for day in days:
        R = json.loads((CACHE / "regions" / seed / f"{day}.json").read_text())
        for key, kp in clicks(day).items():
            if not all(k in kp for k in FIN_KP) or key not in R or not R[key]["up_fin"]:
                continue
            W, H = Image.open(DAYS / day / "images" / f"{key.rsplit('~', 1)[0]}.jpg").size
            tri = np.array([[kp[k][0] * W, kp[k][1] * H] for k in FIN_KP])
            out.append((day, key, tri, R[key]["reg"], R[key]["A"], R[key]["view"]))
    return out


def frame(A: dict):
    ex = np.array(A["ex"]); ey = np.array([-ex[1], ex[0]])
    return np.stack([ex, ey])


def need_k(tri, c, L, Rm, m):
    """smallest k (side = k L) that holds the whole triangle with margin m"""
    d = np.abs((tri - c) @ Rm.T)                     # chart-axis distances of the three clicks from the centre
    return float(d.max() / (L * (0.5 - m)))


def geom(tri, c, side, Rm):
    base = float(np.linalg.norm(tri[0] - tri[1])); ctr = tri.mean(0)
    d = (tri - c) @ Rm.T / side + 0.5                 # in units of the side, 0..1 inside
    inside = (d >= 0) & (d <= 1)
    return dict(off=float(np.linalg.norm(c - ctr) / base), share=base / side, tip_in=bool(inside[2].all()), tri_in=bool(inside.all()),
                tri_in_m10=bool(((d >= 0.1) & (d <= 0.9)).all()))


def summ(rows):
    off = np.array([r["off"] for r in rows]); sh = np.array([r["share"] for r in rows])
    return dict(n=len(rows), centre_off_median=round(float(np.median(off)), 3), off_over_1=round(float(np.mean(off > 1)), 3),
                base_share_median=round(float(np.median(sh)), 3), base_share_q=[round(float(x), 3) for x in np.percentile(sh, [25, 75])],
                tip_inside=round(float(np.mean([r["tip_in"] for r in rows])), 3), triangle_inside=round(float(np.mean([r["tri_in"] for r in rows])), 3),
                triangle_inside_margin10=round(float(np.mean([r["tri_in_m10"] for r in rows])), 3))


def cmd_explore(a):
    rows = clicked(a.seed); have = [r for r in rows if r[3]]
    print(f"clicked fins with a parts_up fin cut-out: {len(rows)} (dorsal view {sum(r[5] == 'dorsal' for r in rows)}), with a W3c region {len(have)}")
    for ax in ("chart", "pca", "inst"):
        q = np.array([r[3][f"L_{ax}"] / np.linalg.norm(r[2][0] - r[2][1]) for r in have])
        print(f"  L_{ax} / clicked base: median {np.median(q):.3f}, quartiles {np.percentile(q, 25):.3f} / {np.percentile(q, 75):.3f}, "
              f"IQR of log {np.subtract(*np.percentile(np.log(q), [75, 25])):.3f}, 5-95 % {np.percentile(q, 5):.2f}-{np.percentile(q, 95):.2f}")
    for ax in ("chart", "pca", "inst"):
        for m in (0.0, 0.05, 0.1):
            k = np.array([need_k(r[2], np.array(r[3]["c"]), r[3][f"L_{ax}"], frame(r[4]), m) for r in have])
            print(f"  axis {ax:5s} margin {m:.2f}: k needed, percentiles 50 / 75 / 80 / 90 / 95: " + " / ".join(f"{np.percentile(k, p):.2f}" for p in (50, 75, 80, 90, 95)))
    for day in CLICK_DAYS:
        hd = [r for r in have if r[0] == day]
        k = np.array([need_k(r[2], np.array(r[3]["c"]), r[3]["L_chart"], frame(r[4]), 0.1) for r in hd])
        print(f"  {day} (n {len(hd)}) chart axis margin 0.1: 50 / 95 % {np.percentile(k, 50):.2f} / {np.percentile(k, 95):.2f}")
    # A for reference
    A = [geom(r[2], np.array(r[4]["c"]), r[4]["side"], frame(r[4])) for r in rows]
    print("  A (parts_up):", summ(A))
    zoomA = np.array([r[4]["side"] / r[3]["L_chart"] for r in have])
    print(f"  A's side in region lengths: median {np.median(zoomA):.2f}, quartiles {np.percentile(zoomA, 25):.2f} / {np.percentile(zoomA, 75):.2f}")


def cmd_fix(a):
    rule = dict(axis=a.axis, wide=a.wide, tight=a.tight, note=a.note, chosen_on="clicked fins of 2025-01-02 and 2025-03-18 (training days), geometry only, W3c seed 0")
    RESULTS.mkdir(parents=True, exist_ok=True); (RESULTS / "rule.json").write_text(json.dumps(rule, indent=1)); print(rule)


def load_rule():
    return json.loads((RESULTS / "rule.json").read_text())


def cmd_check(a):
    rule = load_rule(); out = {}
    for days, tag in ((CLICK_DAYS, "both"), ((CLICK_DAYS[0],), CLICK_DAYS[0]), ((CLICK_DAYS[1],), CLICK_DAYS[1])):
        rows = clicked(a.seed, days); res = {"A": [], "wide": [], "tight": []}; nfb = 0
        for day, key, tri, reg, A, view in rows:
            Rm = frame(A)
            res["A"].append(geom(tri, np.array(A["c"]), A["side"], Rm))
            for v in ("wide", "tight"):
                if reg is None:
                    res[v].append(geom(tri, np.array(A["c"]), A["side"], Rm))
                else:
                    res[v].append(geom(tri, np.array(reg["c"]), rule[v] * reg[f"L_{rule['axis']}"], Rm))
            nfb += reg is None
        out[tag] = {v: summ(rr) for v, rr in res.items()}; out[tag]["fallback"] = nfb
    v3 = json.loads((RESULTS.parent / "cutouts_v3" / "clickcheck.json").read_text())
    out["v3_from_clickcheck_json"] = {k: v for k, v in v3.items() if k.startswith("fin_")}
    print(json.dumps(out, indent=1)); (RESULTS / f"clickcheck_{a.seed}.json").write_text(json.dumps(out, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["explore", "fix", "check"]); ap.add_argument("--seed", default="s0")
    ap.add_argument("--axis", default="chart"); ap.add_argument("--wide", type=float); ap.add_argument("--tight", type=float); ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    {"explore": cmd_explore, "fix": cmd_fix, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
