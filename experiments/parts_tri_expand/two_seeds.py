"""W3c (two seeds) against W3c filled at prediction time (a) and W3d trained on the expanded triangle (b), with W3c's
measures (same scripts, same days) and the two new ones -> two_seeds.md.

    PYTHONPATH=. .venv/bin/python experiments/parts_tri_expand/two_seeds.py

Rule (fixed before any W3d or filled score was read; W3c's own rule from parts_weak/two_seeds.py): a reading of an
arm is SEPARABLE from W3c when both of its seeds lie on the same side of both W3c seeds and the gap between the
nearest seeds is larger than the larger of the two seed distances (W3c s0 - s1, the arm's own s0 - s1). Validation
days for the standing measures; held-out clicks of 2025-03-18 for the shape; check21 renders for the render IoU.
"""
from __future__ import annotations

import json
from pathlib import Path

E = Path(__file__).resolve().parent
C0, C1 = "parts_seg_r05_w3c", "parts_seg_r05_w3cs1"
ARMS = {"W3c": (C0, C1), "W3c filled (a)": (C0 + "_fill", C1 + "_fill"), "W3d (b)": ("parts_seg_r05_w3d", "parts_seg_r05_w3ds1")}


def verdict(ref, arm):
    if None in ref or None in arm:
        return "-"
    dist = max(abs(ref[0] - ref[1]), abs(arm[0] - arm[1]))
    if min(arm) > max(ref) and min(arm) - max(ref) > dist:
        return f"higher, separable (gap {min(arm) - max(ref):.3f} > {dist:.3f})"
    if max(arm) < min(ref) and min(ref) - max(arm) > dist:
        return f"lower, separable (gap {min(ref) - max(arm):.3f} > {dist:.3f})"
    return f"not separable (seed distance {dist:.3f})"


def get(d, *ks):
    try:
        for k in ks:
            d = d[k]
        return d
    except (KeyError, TypeError):
        return None


def main() -> int:
    c = json.loads((E / "compare_triexp.json").read_text())
    f = json.loads((E / "finless_triexp.json").read_text())
    fd = json.loads((E / "fin_definition_triexp.json").read_text())
    sh = json.loads((E / "shares_triexp.json").read_text())
    nm = json.loads((E / "new_measures.json").read_text())
    rows = {
        "fin region found, dorsal views (cut-outs, val)": lambda m: get(c, "cutout_needs", "val", m, "share_fin_region"),
        "fin tip inside the largest fin region (hr, clicks 03-18)": lambda m: get(c, "clicks_2025_03_18", m, "hr", "fin_tip_in_largest_fin_region", "within"),
        "lowest painted point, fin heights (clicks 03-18, median)": lambda m: get(fd, "real", "2025-03-18", m, "down", "median"),
        "lowest point of the connected region (band down the flank)": lambda m: get(fd, "real", "2025-03-18", m, "comp_down", "median"),
        "connected region over triangle area (median)": lambda m: get(fd, "real", "2025-03-18", m, "comp_area_over_triangle", "median"),
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
        "synthetic pectoral IoU (run05 val)": lambda m: get(c, "synthetic_val_iou", m, "pectoral_fin"),
        "synthetic dorsal IoU (run05 val, OLD render label)": lambda m: get(c, "synthetic_val_iou", m, "dorsal_fin"),
        "NEW: IoU with the expanded-triangle truth, held-out clicks 03-18": lambda m: get(nm, "clicks", "2025-03-18", m, "iou_expanded", "mean"),
        "NEW: IoU with the bare triangle, held-out clicks 03-18": lambda m: get(nm, "clicks", "2025-03-18", m, "iou_triangle", "mean"),
        "NEW: renders check21, IoU with the run06 fin + hump label": lambda m: get(nm, "renders", m, "fin_hump", "mean"),
        "NEW: renders check21, IoU with the fin-only label": lambda m: get(nm, "renders", m, "fin", "mean"),
    }
    names = [n for pair in ARMS.values() for n in pair]
    head = "| reading | " + " | ".join(f"{a} s0 | {a} s1" for a in ARMS) + " | (a) vs W3c | (b) vs W3c | (b) vs (a) |"
    lines = [head, "|" + "---|" * (1 + 2 * len(ARMS) + 3)]
    for k, fn in rows.items():
        v = {n: fn(n) for n in names}
        cells = [f"{v[n]:.3f}" if isinstance(v[n], (int, float)) else "-" for n in names]
        pa = lambda a: tuple(v[n] for n in ARMS[a])  # noqa: E731
        lines.append(f"| {k} | " + " | ".join(cells) + f" | {verdict(pa('W3c'), pa('W3c filled (a)'))} | {verdict(pa('W3c'), pa('W3d (b)'))} | "
                     f"{verdict(pa('W3c filled (a)'), pa('W3d (b)'))} |")
    lines += ["", __doc__.strip().split("\n\n")[-1].replace("\n", " ")]
    (E / "two_seeds.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
