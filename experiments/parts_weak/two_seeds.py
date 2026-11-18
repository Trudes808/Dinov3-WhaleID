"""The open W3c readings with both W3c seeds next to W2's two seeds and the baselines -> w3c_two_seeds.txt.

    PYTHONPATH=. .venv/bin/python experiments/parts_weak/two_seeds.py

Rule (fixed before the second seed is scored): a reading is SEPARABLE from W2 when both W3c seeds lie on the same
side of both W2 seeds and the gap between the nearest W3c seed and the nearest W2 seed is larger than the larger of
the two seed distances (W2 s0 - s1, baseline a - s1). Otherwise NOT SEPARABLE. Validation days only.
"""
from __future__ import annotations

import json
from pathlib import Path

E = Path(__file__).resolve().parent
A, S1, W2, W2S1, C0, C1 = ("parts_seg_r05_a", "parts_seg_r05_s1", "parts_seg_r05_w2", "parts_seg_r05_w2s1", "parts_seg_r05_w3c", "parts_seg_r05_w3cs1")


def verdict(v: dict) -> str:
    w = sorted([v[W2], v[W2S1]])
    c = sorted([v[C0], v[C1]])
    dist = max(abs(v[W2] - v[W2S1]), abs(v[A] - v[S1]))
    if c[0] > w[1] and c[0] - w[1] > dist:
        return f"SEPARABLE (higher; gap {c[0] - w[1]:.3f} > seed distance {dist:.3f})"
    if c[1] < w[0] and w[0] - c[1] > dist:
        return f"SEPARABLE (lower; gap {w[0] - c[1]:.3f} > seed distance {dist:.3f})"
    return f"not separable (seed distance {dist:.3f})"


def main() -> int:
    c = json.loads((E / "compare_w3c2.json").read_text())
    f = json.loads((E / "finless_w3c2.json").read_text())
    names = [A, S1, W2, W2S1, C0, C1]
    rows = {
        "finless backs called dorsal, by eye, crops (n 33)": {n: f["hr"][n]["t_eye"]["called_dorsal"] for n in names},
        "finless backs called dorsal, by classifier, crops (n 151)": {n: f["hr"][n]["t_vc"]["called_dorsal"] for n in names},
        "finless backs called dorsal, by classifier, whole photo": {n: f["full"][n]["t_vc"]["called_dorsal"] for n in names},
        "three-way call by eye, crops (n 79)": {n: c["three_way"]["two_part"]["hr"]["val_eye"][n]["acc"] for n in names},
        "fluke covers a quarter of the instance (cut-outs)": {n: c["cutout_needs"]["val"][n]["share_fluke_quarter"] for n in names},
        "synthetic pectoral IoU (run05 val)": {n: c["synthetic_val_iou"][n]["pectoral_fin"] for n in names},
    }
    head = f"{'reading':60s} {'a':>6s} {'s1':>6s} {'W2s0':>6s} {'W2s1':>6s} {'W3c0':>6s} {'W3c1':>6s}  verdict vs W2's two seeds"
    lines = [head, "-" * len(head)]
    for k, v in rows.items():
        lines.append(f"{k:60s} " + " ".join(f"{v[n]:6.3f}" for n in names) + "  " + verdict(v))
    lines.append("")
    lines.append(__doc__.strip().splitlines()[-3] + " " + " ".join(__doc__.strip().splitlines()[-2:]))
    (E / "w3c_two_seeds.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
