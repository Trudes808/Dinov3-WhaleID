"""W3e (rule A on real AND synthetic dorsal labels) against W3c and W3d, two seeds each, W3c's measures with the same
scripts and days, plus the render-side measures of w3e_scores.py -> two_seeds_w3e.md.

    PYTHONPATH=. .venv/bin/python experiments/parts_tri_expand/two_seeds_w3e.py

Separability rule (W3c's own, parts_weak/two_seeds.py, unchanged): an arm is SEPARABLE from a reference when both of
its seeds lie on the same side of both reference seeds and the gap between the nearest seeds is larger than the larger
of the two seed distances. Validation days for the standing measures; held-out clicks of 2025-03-18 for the shape;
r06r_check21 renders for the render IoU; run05 val renders for the synthetic IoU.
"""
from __future__ import annotations

import json
from pathlib import Path

from two_seeds import get, verdict

E = Path(__file__).resolve().parent
SV = Path("/home/sat3737/NOAA_whales/moonshot_cache/parts_tri_expand/synth_val")
ARMS = {"W3c": ("parts_seg_r05_w3c", "parts_seg_r05_w3cs1"), "W3d": ("parts_seg_r05_w3d", "parts_seg_r05_w3ds1"),
        "W3e": ("parts_seg_r05_w3e", "parts_seg_r05_w3es1")}


def synth(m, cache, cls):
    p = SV / f"{m}.json"
    return json.loads(p.read_text())["iou"][cache][cls] if p.exists() else None


def main() -> int:
    c = json.loads((E / "compare_w3e.json").read_text())
    f = json.loads((E / "finless_w3e.json").read_text())
    fd = json.loads((E / "fin_definition_w3e.json").read_text())
    sh = json.loads((E / "shares_w3e.json").read_text())
    nm = json.loads((E / "new_measures_w3e.json").read_text())
    rr = json.loads((E / "w3e_renders.json").read_text())["models"]
    rows = {
        "NEW: IoU with the expanded-triangle truth, held-out clicks 03-18": lambda m: get(nm, "clicks", "2025-03-18", m, "iou_expanded", "mean"),
        "NEW: lowest painted point, fin heights (clicks 03-18, median; truth -0.26)": lambda m: get(fd, "real", "2025-03-18", m, "down", "median"),
        "NEW: connected region over triangle area (clicks 03-18, median; truth 1.93)": lambda m: get(fd, "real", "2025-03-18", m, "comp_area_over_triangle", "median"),
        "NEW: renders check21, IoU with the NEW (rule A) label, full frame 1280": lambda m: get(rr, m, "full", "iou_new", "mean"),
        "NEW: renders check21, IoU with the NEW label, per-whale hr window": lambda m: get(rr, m, "hr", "iou_new", "mean"),
        "NEW: renders check21, IoU with the NEW label, fin zoom window": lambda m: get(rr, m, "zoom", "iou_new", "mean"),
        "renders check21, IoU with the OLD label, full frame (scored at 2560)": lambda m: get(rr, m, "full", "iou_old", "mean"),
        "renders check21, IoU with the OLD label, per-whale hr window": lambda m: get(rr, m, "hr", "iou_old", "mean"),
        "renders check21, IoU with the OLD label, helper's protocol (1280)": lambda m: get(nm, "renders", m, "fin_hump", "mean"),
        "synthetic dorsal IoU, run05 val, OLD label": lambda m: synth(m, "old", "dorsal_fin"),
        "synthetic dorsal IoU, run05 val, rule-A label": lambda m: synth(m, "triA", "dorsal_fin"),
        "synthetic pectoral IoU (run05 val)": lambda m: synth(m, "old", "pectoral_fin"),
        "synthetic fluke IoU (run05 val)": lambda m: synth(m, "old", "fluke"),
        "fin region found, dorsal views (cut-outs, val)": lambda m: get(c, "cutout_needs", "val", m, "share_fin_region"),
        "fin tip inside the largest fin region (hr, clicks 03-18)": lambda m: get(c, "clicks_2025_03_18", m, "hr", "fin_tip_in_largest_fin_region", "within"),
        "clicked triangle covered (median)": lambda m: get(fd, "real", "2025-03-18", m, "triangle_covered", "median"),
        "dorsal share of the whale in dorsal views (hr, val, median)": lambda m: get(sh, "hr", m, "dorsal", "dorsal_fin_median"),
        "finless backs called dorsal, by eye (n 33)": lambda m: get(f, "hr", m, "t_eye", "called_dorsal"),
        "finless backs called dorsal, classifier (n 151)": lambda m: get(f, "hr", m, "t_vc", "called_dorsal"),
        "three-way call by eye, crops (n 79)": lambda m: get(c, "three_way", "two_part", "hr", "val_eye", m, "acc"),
        "three-way call by eye, whole photo (n 79)": lambda m: get(c, "three_way", "two_part", "full", "val_eye", m, "acc"),
        "three-way call by classifier, whole photo": lambda m: get(c, "three_way", "two_part", "full", "val_clf", m, "acc"),
        "fluke covers a quarter (cut-outs, val)": lambda m: get(c, "cutout_needs", "val", m, "share_fluke_quarter"),
        "fluke notch inside the fluke (hr, clicks 03-18)": lambda m: get(c, "clicks_2025_03_18", m, "hr", "fluke_notch_in_fluke", "within"),
        "tail stock: body share in fluke views (hr, val, median)": lambda m: get(sh, "hr", m, "fluke", "body_median"),
    }
    names = [n for pair in ARMS.values() for n in pair]
    head = "| reading | " + " | ".join(f"{a} s0 | {a} s1" for a in ARMS) + " | W3e vs W3c | W3e vs W3d | W3d vs W3c |"
    lines = [head, "|" + "---|" * (1 + 2 * len(ARMS) + 3)]
    for k, fn in rows.items():
        v = {n: fn(n) for n in names}
        cells = [f"{v[n]:.3f}" if isinstance(v[n], (int, float)) else "-" for n in names]
        pa = lambda a: tuple(v[n] for n in ARMS[a])  # noqa: E731
        lines.append(f"| {k} | " + " | ".join(cells) + f" | {verdict(pa('W3c'), pa('W3e'))} | {verdict(pa('W3d'), pa('W3e'))} | {verdict(pa('W3c'), pa('W3d'))} |")
    lines += ["", __doc__.strip().split("\n\n")[-1].replace("\n", " ")]
    (E / "two_seeds_w3e.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
