"""Report: run03 canon model with keypoint/state/deformation heads (canon_r03_kp) vs the plain run03
retrain (canon_r03_ft4), with the run02 model where numbers exist: synthetic test split overall and
by pose category, keypoint-head metrics, real photos (outline metric with and without keypoints,
side-flip rate).

    .venv/bin/python scripts/run03_kp_report.py --out experiments/run03_kp_report.md
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

EXP = Path("experiments"); REAL = EXP / "real_2024-12-18"
HARD = ["fin_tip", "pec_up", "roll_belly", "fluke_edge", "blow_sliver", "awash", "peduncle", "spyhop_low"]


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def f(v, nd=3):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--out", type=Path, required=True); p.add_argument("--a", default="canon_r03_ft4"); p.add_argument("--b", default="canon_r03_kp")
    a = p.parse_args()
    ma = load(EXP / a.a / "eval_test_kp" / "metrics.json") or load(EXP / a.a / "eval_test" / "metrics.json")
    mb = load(EXP / a.b / "eval_test_kp" / "metrics.json")
    oa, ob = (ma or {}).get("oracle_crops", {}), (mb or {}).get("oracle_crops", {})
    L = [f"# run03: keypoint heads vs plain retrain ({datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')})", "",
         f"Both models: same backbone, split, schedule; `{a.b}` adds 18 keypoint heatmaps, a visibility state per keypoint "
         "(visible / under water / hidden / out of frame) and the 7 deformation parameters. Synthetic numbers are on the run03 test split (oracle crops).", "",
         "## Synthetic test split, overall", "", f"| metric | {a.a} | {a.b} |", "|---|---|---|"]
    rows = [("surface error mean (m)", "surface_err_m_mean"), ("surface error median (m)", "surface_err_median_m_median"), ("visibility IoU", "vis_iou_mean"),
            ("side accuracy", "side_acc_mean"), ("rotation direct < 15 deg", "rot_direct_deg_under_15"), ("rotation direct median (deg)", "rot_direct_deg_median"),
            ("rotation PnP (dense map) < 15 deg", "rot_pnp_deg_under_15"), ("rotation PnP from detected keypoints < 15 deg", "rot_kppnp_deg_under_15"),
            ("images with a keypoint PnP", "rot_kppnp_images"), ("landmark PCK@0.1 from the map", "lm_map_pck_0.1_mean"),
            ("keypoint head PCK@0.05 / @0.1", ("kp_pck_0.05_mean", "kp_pck_0.1_mean")), ("keypoint state accuracy", "kp_state_acc_mean"),
            ("keypoint visible F1", "kp_visible_f1_mean"), ("deformation error (deg, mean abs)", "deform_err_deg_mean"), ("images", "images_evaluated")]
    for label, key in rows:
        get = lambda m: (" / ".join(f(m.get(k)) for k in key) if isinstance(key, tuple) else f(m.get(key)))
        L.append(f"| {label} | {get(oa)} | {get(ob)} |")
    L += ["", "## By pose category (oracle crops): hard views first", "",
          f"| category | n | surf. median m: {a.a} / {a.b} | rot direct <15: {a.a} / {a.b} | rot PnP(map) <15: {a.a} / {a.b} | rot PnP(keypoints) <15 ({a.b}) | kp PCK@0.1 | state acc |",
          "|---|---|---|---|---|---|---|---|"]
    ca, cb = oa.get("by_category", {}), ob.get("by_category", {})
    cats = HARD + sorted(set(cb) - set(HARD))
    for c in cats:
        x, y = ca.get(c, {}), cb.get(c, {})
        if not y:
            continue
        L.append(f"| {c} | {y.get('n')} | {f(x.get('surface_err_median_m'))} / {f(y.get('surface_err_median_m'))} | {f(x.get('rot_direct_deg_under_15'), 2)} / {f(y.get('rot_direct_deg_under_15'), 2)} | "
                 f"{f(x.get('rot_pnp_deg_under_15'), 2)} / {f(y.get('rot_pnp_deg_under_15'), 2)} | {f(y.get('rot_kppnp_deg_under_15'), 2)} (n={y.get('rot_kppnp_deg_n')}) | {f(y.get('kp_pck_0.1'), 2)} | {f(y.get('kp_state_acc'), 2)} |")
    L += ["", "## Real photos (2024-12-18)", "", "| outline metric | canon_r02_ft4 | canon_r03_ft4 | canon_r03_kp, dense map | canon_r03_kp, map + keypoints |", "|---|---|---|---|---|"]
    outs = [load(REAL / "outline_metric_canon_r02_ft4_v2.json"), load(REAL / f"outline_metric_{a.a}.json"), load(REAL / f"outline_metric_{a.b}.json"), load(REAL / f"outline_metric_{a.b}_kp.json")]
    def g(o, *path):
        d = (o or {}).get("summary", {})
        for k in path:
            d = d.get(k) if isinstance(d, dict) else None
        return d
    for label, path in (("IoU above water, median", ("iou_above_water", "median")), ("IoU above water, mean", ("iou_above_water", "mean")), ("photos with IoU >= 0.5", ("iou_above_0.5",)),
                        ("coverage, median", ("coverage", "median")), ("excess above water, median", ("excess_above_water", "median")), ("PnP inlier fraction, median", ("inlier_fraction", "median")),
                        ("median reprojection error (px)", ("median_reproj_px", "median")), ("photos with keypoints used", ("photos_with_keypoints",))):
        L.append(f"| {label} | " + " | ".join(f(g(o, *path)) for o in outs) + " |")
    oa_r, ob_r, ob_kp = outs[1], outs[2], outs[3]
    if oa_r and ob_r and ob_kp:
        pa, pb, pk = oa_r["photos"], ob_r["photos"], ob_kp["photos"]
        common = [s for s in pa if pa[s].get("ok") and pb.get(s, {}).get("ok") and pk.get(s, {}).get("ok")]
        if common:
            base = np.array([pa[s]["iou_above_water"] for s in common]); q = np.percentile(base, 25)
            hard = [s for s in common if pa[s]["iou_above_water"] <= q]
            L += ["", f"Hardest quartile of real photos by the plain model's outline IoU ({len(hard)} photos, IoU <= {q:.3f}):", "",
                  "| model | IoU above water, median | mean |", "|---|---|---|"]
            for name, d in ((a.a, pa), (f"{a.b}, dense map", pb), (f"{a.b}, map + keypoints", pk)):
                v = np.array([d[s]["iou_above_water"] for s in hard]); L.append(f"| {name} | {np.median(v):.3f} | {v.mean():.3f} |")
    cmp = REAL / f"compare_{a.b}.log"
    if cmp.exists():
        L += ["", "Side-flip (mirror) test on the real photos:", "", "```", cmp.read_text().strip(), "```"]
    L += ["", "## Files", "", f"- `experiments/{a.b}/eval_test_kp/metrics.json` (per-category under `by_category`), `experiments/{a.a}/eval_test_kp/metrics.json`",
          f"- `{REAL}/canon_{a.b}/` (keypoints in `points.jsonl` and the npz), `{REAL}/outline_metric_{a.b}.json`, `{REAL}/outline_metric_{a.b}_kp.json`"]
    a.out.write_text("\n".join(L) + "\n"); print("\n".join(L)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
