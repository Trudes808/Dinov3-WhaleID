"""Tape colour, round 2: candidate rules on the band measurements of colour2_measure.py; fit on one day, score on the other
(both directions), and score the boat marker renders. Text only.

    .venv/bin/python experiments/sam3_labeller/colour2_rules.py --real SCRATCH/colour2/bands_real.json \
        --renders SCRATCH/colour2/bands_renders.json --out SCRATCH/colour2/rules_scores.json

Band choice (which tape component of the photo is read):
  largest  the reader's choice (largest 8-connected tape component)
  centre   among components with >= 25 % of the largest's area, the one whose centroid is nearest the frame centre
Colour (of the chosen band):
  hsv_old  the reader's rule (HSV, S >= 0.25 and V >= 0.20, circular mean hue, fixed ranges, "other" 260-320)
  hsv_nn   circular mean HSV hue of the band's pixels with S >= 0.08, nearest class prototype hue
  lab_nn   hue angle of the band's mean CIE a*, b*, nearest class prototype angle
  labwb_nn as lab_nn after dividing the band's RGB by the white tube's RGB beside it (white balance by the rail)
  ab_nc    nearest class centroid in (a*, b*) (a two-feature nearest-centroid classifier)
Prototypes / centroids are fitted on the fit day's labelled bands: every rail marker's centre band (its truth colour) and
the other bands the truth notes name (off-centre yellow / blue / green bands, crew photos 137, 140, 141). The scored item
is the photo (rail markers; the three crew photos reported apart).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COL = ["red", "yellow", "green", "blue"]
# off-centre bands the truth notes name (2026-03-30 truth README / truth_final.txt): photo seq -> colour of the OTHER band
OFF = {("2026-03-30", s): c for s, c in [("137", "yellow"), ("141", "yellow"), ("165", "yellow"), ("174", "blue"), ("193", "green"),
                                        ("201", "yellow"), ("203", "green"), ("206", "yellow"), ("216", "blue"), ("238", "blue")]}


def circ_mean(x) -> float:
    a = np.deg2rad(np.asarray(x, float))
    return float((np.rad2deg(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) + 360) % 360)


def cdist(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


def old_hue_class(h) -> str:
    if h is None:
        return "none"
    if h < 20 or h >= 320:
        return "red"
    return "yellow" if h < 70 else "green" if h < 170 else "blue" if h < 260 else "other"


def lab_ab(b: dict, wb: bool = False) -> tuple[float, float, float]:
    """mean CIE L*, a*, b* of the band (from OpenCV 8-bit Lab means; wb: from the white-balanced median RGB)."""
    if not wb:
        L, a, bb = b["lab_mean"]
        return L / 2.55, a - 128, bb - 128
    rgb = np.array(b["rgb_mean"]) / 255.0; w = np.array(b["white_rgb"] or [1, 1, 1]) / 255.0
    rgb = np.clip(rgb / np.maximum(w, 1e-3) * w.mean(), 0, 1)      # grey-world on the tube: keep its brightness
    return srgb_to_lab(rgb)


def srgb_to_lab(rgb) -> tuple[float, float, float]:
    c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = M @ c / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return 116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])


def feat(b: dict, space: str):
    if space == "hsv_nn":
        return b["hue_all"]
    if space in ("lab_nn", "labwb_nn"):
        _, a, bb = lab_ab(b, space == "labwb_nn")
        return float(np.rad2deg(np.arctan2(bb, a)) % 360)
    if space == "ab_nc":
        _, a, bb = lab_ab(b)
        return (a, bb)
    raise KeyError(space)


def fit(space: str, bands: list[tuple[dict, str]]) -> dict:
    if space == "hsv_old":
        return {}
    by = {c: [feat(b, space) for b, t in bands if t == c] for c in COL}
    if space == "ab_nc":
        return {c: tuple(np.mean(v, 0)) for c, v in by.items() if v}
    return {c: circ_mean(v) for c, v in by.items() if v}


def classify(b: dict | None, space: str, proto: dict) -> str:
    if b is None or b["n_px"] < 200:
        return "none"
    if space == "hsv_old":
        return old_hue_class(b["hue_reader"]) if b["n_col"] >= 200 else "none"
    x = feat(b, space)
    if x is None:
        return "none"
    if space == "ab_nc":
        return min(proto, key=lambda c: np.hypot(x[0] - proto[c][0], x[1] - proto[c][1]))
    return min(proto, key=lambda c: cdist(x, proto[c]))


def pick(bands: list[dict], how: str) -> dict | None:
    if not bands:
        return None
    big = max(bands, key=lambda b: b["area"])
    if how == "largest":
        return big
    cand = [b for b in bands if b["area"] >= 0.25 * big["area"]]
    return min(cand, key=lambda b: b["dist"])


def labelled_bands(rows: list[dict]) -> list[tuple[dict, str]]:
    """truth band = the component nearest the centre (the truth's definition); the other noted band gets its note colour."""
    out = []
    for r in rows:
        if not r["bands"]:
            continue
        c = pick(r["bands"], "centre"); out.append((c, r["truth"]))
        o = [b for b in r["bands"] if b is not c and b["area"] >= 0.25 * max(x["area"] for x in r["bands"])]
        if (r["day"], r["seq"]) in OFF and o:
            out.append((max(o, key=lambda b: b["area"]), OFF[(r["day"], r["seq"])]))
    return out


def score(rows, how, space, proto):
    per = {}
    for r in rows:
        per[r["seq"]] = (r["truth"], classify(pick(r["bands"], how), space, proto))
    return per


def margins(rows, space, proto, how="centre"):
    """per item: distance to its truth prototype minus distance to the nearest other prototype (negative = right)."""
    out = {}
    for r in rows:
        b = pick(r["bands"], how)
        if b is None or space in ("hsv_old",):
            continue
        x = feat(b, space)
        d = {c: (np.hypot(x[0] - p[0], x[1] - p[1]) if space == "ab_nc" else cdist(x, p)) for c, p in proto.items()}
        if r["truth"] in d:
            out[r["seq"]] = round(float(d[r["truth"]] - min(v for c, v in d.items() if c != r["truth"])), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--real", required=True); ap.add_argument("--renders", required=True)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    R = json.loads(Path(a.real).read_text()); RR = json.loads(Path(a.renders).read_text())
    days = {"2026-04-07": [r for r in R if r["day"] == "2026-04-07"], "2026-03-30": [r for r in R if r["day"] == "2026-03-30"]}
    spaces = ["hsv_old", "hsv_nn", "lab_nn", "labwb_nn", "ab_nc"]; hows = ["largest", "centre"]
    out = {"cross": {}, "notes": "fit on one day's labelled bands, score the other day's rail markers (crew photos apart)"}
    lines = []
    for fit_day, test_day in (("2026-04-07", "2026-03-30"), ("2026-03-30", "2026-04-07")):
        lb = labelled_bands(days[fit_day])
        for how in hows:
            for sp in spaces:
                proto = fit(sp, lb)
                mk_fit = [r for r in days[fit_day] if r["role"] == "marker"]; mk_test = [r for r in days[test_day] if r["role"] == "marker"]
                cr_test = [r for r in days[test_day] if r["role"] == "crew"]
                pf = score(mk_fit, how, sp, proto); pt = score(mk_test, how, sp, proto); pc = score(cr_test, how, sp, proto)
                k = f"{fit_day}->{test_day}|{how}|{sp}"
                out["cross"][k] = {"proto": {c: (list(map(float, v)) if isinstance(v, tuple) else round(v, 1)) for c, v in proto.items()},
                                   "fit_right": [sum(t == c for t, c in pf.values()), len(pf)],
                                   "test_right": [sum(t == c for t, c in pt.values()), len(pt)], "test_per": pt,
                                   "crew_right": [sum(t == c for t, c in pc.values()), len(pc)], "crew_per": pc,
                                   "test_margins": margins(mk_test, sp, proto, how) if sp != "hsv_old" else {}}
                wrong = {s: v for s, v in pt.items() if v[0] != v[1]}
                lines.append(f"{k:45s} fit {out['cross'][k]['fit_right'][0]:2d}/{len(pf)}  test {out['cross'][k]['test_right'][0]:2d}/{len(pt)}"
                             f"  crew {out['cross'][k]['crew_right'][0]}/{len(pc)}  wrong {wrong}")
    # renders: prototypes from each real day (and both); the render's band = exact tape label, and the frozen model's map on the held-out 25
    out["renders"] = {}
    for src in ("2026-04-07", "2026-03-30", "both"):
        lb = labelled_bands(days[src] if src != "both" else R)
        for sp in spaces:
            proto = fit(sp, lb)
            for which in ("bands_exact", "bands_pred"):
                rows = [{"seq": r["id"], "truth": r["truth"], "bands": r[which]} for r in RR if which in r]
                for how in hows:
                    p = score(rows, how, sp, proto)
                    conf = {}
                    for t, c in p.values():
                        conf[f"{t}->{c}"] = conf.get(f"{t}->{c}", 0) + 1
                    k = f"protos {src}|{which}|{how}|{sp}"
                    out["renders"][k] = {"right": [sum(t == c for t, c in p.values()), len(p)], "confusion": conf}
                    if how == "centre" or sp == "hsv_old":
                        lines.append(f"renders {k:45s} {out['renders'][k]['right']}  " + " ".join(f"{x}:{n}" for x, n in sorted(conf.items()) if x.split('->')[0] != x.split('->')[1]))
    Path(a.out).write_text(json.dumps(out, indent=1)); print("\n".join(lines))


if __name__ == "__main__":
    main()
