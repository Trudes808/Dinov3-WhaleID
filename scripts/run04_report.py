"""Metrics report for the run04 fine-tunes against the run03 baselines, as Markdown.

    .venv/bin/python scripts/run04_report.py --name canon_r04_ft4 --seg seg_r04_ft4 --baseline canon_r03_ft4 \
        --real experiments/real_2024-12-18 --out experiments/run04_report.md
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

EXP = Path("experiments")

CANON_ROWS = [
    ("surface error mean (m)", "surface_err_m_mean", "lower"),
    ("surface error median (m)", "surface_err_m_median", "lower"),
    ("torso / fluke / fin error (m)", ("surface_err_torso_m_mean", "surface_err_fluke_m_mean", "surface_err_fin_m_mean"), "lower"),
    ("visibility IoU", "vis_iou_mean", "higher"),
    ("mirror disagreement (m)", "flip_surface_disagreement_m_median", "lower"),
    ("side accuracy", "side_acc_mean", "higher"),
    ("side flip agreement", "side_flip_agreement_mean", "higher"),
    ("unstable crops (agreement < 0.8)", "side_unstable_crops_frac", "lower"),
    ("rotation direct median (deg)", "rot_direct_deg_median", "lower"),
    ("rotation direct < 15 deg", "rot_direct_deg_under_15", "higher"),
    ("rotation PnP median (deg)", "rot_pnp_deg_median", "lower"),
    ("rotation PnP < 15 deg", "rot_pnp_deg_under_15", "higher"),
    ("log-scale PnP error", "logscale_pnp_median", "lower"),
    ("PnP inlier fraction", "pnp_inlier_fraction_mean", "higher"),
    ("landmark PCK@0.1 (map)", "lm_map_pck_0.1_mean", "higher"),
    ("landmark PCK@0.1 (PnP)", "lm_pnp_pck_0.1_mean", "higher"),
]


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, (tuple, list)):
        return " / ".join(fmt(v) for v in value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def pick(metrics: dict | None, key):
    if not metrics:
        return None
    if isinstance(key, tuple):
        return tuple(metrics.get(k) for k in key)
    return metrics.get(key)


def canon_table(new: dict | None, old: dict | None, section: str) -> list[str]:
    lines = [f"| metric ({section}) | run04 | run03 baseline | better |", "|---|---|---|---|"]
    n, o = (new or {}).get(section), (old or {}).get(section)
    for label, key, better in CANON_ROWS:
        a, b = pick(n, key), pick(o, key)
        mark = ""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            mark = "run04" if (a < b) == (better == "lower") and a != b else ("run03" if a != b else "tie")
        lines.append(f"| {label} | {fmt(a)} | {fmt(b)} | {mark} |")
    if n and o:
        lines.append(f"| images evaluated | {n.get('images_evaluated')} | {o.get('images_evaluated')} | |")
    return lines


def seg_table(new: dict | None, old: dict | None) -> list[str]:
    lines = ["| segmentation (val) | run04 | run03 baseline |", "|---|---|---|"]
    for label, key in (("dataset IoU", "dataset_iou"), ("mean image IoU", "mean_image_iou"), ("median image IoU", "median_image_iou"),
                       ("precision", "precision"), ("recall", "recall"), ("IoU >= 0.5", "iou_at_0.5"), ("IoU >= 0.75", "iou_at_0.75"),
                       ("negatives clean", "negatives_clean"), ("images", "images")):
        lines.append(f"| {label} | {fmt((new or {}).get(key))} | {fmt((old or {}).get(key))} |")
    return lines


def outline_table(new: dict | None, old: dict | None) -> list[str]:
    lines = ["| real photos, label-free outline metric | run04 | run03 baseline |", "|---|---|---|"]
    ns, os_ = (new or {}).get("summary", {}), (old or {}).get("summary", {})
    for label, path in (("IoU above water, median", ("iou_above_water", "median")), ("IoU above water, mean", ("iou_above_water", "mean")),
                        ("photos with IoU >= 0.5", ("iou_above_0.5",)), ("coverage, median", ("coverage", "median")),
                        ("excess above water, median", ("excess_above_water", "median")), ("PnP inlier fraction, median", ("inlier_fraction", "median")),
                        ("median reprojection error (px), median", ("median_reproj_px", "median")), ("photos scored", ("scored",))):
        def get(d):
            for p in path:
                d = d.get(p) if isinstance(d, dict) else None
            return d
        lines.append(f"| {label} | {fmt(get(ns))} | {fmt(get(os_))} |")
    return lines


def side_flip(points_path: Path) -> dict | None:
    if not points_path.exists():
        return None
    rows = [json.loads(l) for l in open(points_path)]
    agree = np.array([r.get("side_flip_agreement") for r in rows if r.get("side_flip_agreement") is not None], float)
    dis = np.array([r.get("flip_disagreement_median_m") for r in rows if r.get("flip_disagreement_median_m") is not None], float)
    return {"crops": len(rows), "flip_rate_gt_20pct": float(np.mean((1.0 - agree) > 0.2)) if len(agree) else None,
            "side_flip_agreement_median": float(np.median(agree)) if len(agree) else None,
            "flip_disagreement_m_median": float(np.median(dis)) if len(dis) else None}


def training_summary(name: str) -> str:
    log = EXP / name / "log.jsonl"
    if not log.exists():
        return "no log"
    events = [json.loads(l) for l in open(log)]
    done = [e for e in events if e.get("event") == "done"]
    val = load_json(EXP / name / "val_metrics.json") or {}
    best = {k: v for k, v in val.items() if k in ("epoch", "score", "val_score", "surface_err_m_mean", "dataset_iou")}
    cfg = (load_json(EXP / name / "config.json") or {}).get("args", {})
    secs = done[0].get("seconds") or done[0].get("elapsed_s") if done else None
    return f"{name}: {'finished' if done else 'not finished'}" + (f" in {secs/3600:.1f} h" if isinstance(secs, (int, float)) else "") + \
        f", epochs {cfg.get('epochs')}, best {best}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True); p.add_argument("--seg", required=True); p.add_argument("--baseline", default="canon_r03_ft4")
    p.add_argument("--seg-baseline", default="seg_r03_ft4"); p.add_argument("--real", type=Path, default=Path("experiments/real_2024-12-18"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    new = load_json(EXP / a.name / "eval_test" / "metrics.json")
    old = load_json(EXP / a.baseline / "eval_test_v2" / "metrics.json") or load_json(EXP / a.baseline / "eval_test" / "metrics.json")
    lines = [f"# run04 fine-tunes vs run03 baselines ({datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')})", "",
             "Synthetic metrics are on each run's own test split (run04's has close-ups, rolled whales, calf-to-adult",
             "sizes and long-lens framing, so it is a harder set than run03's; compare like with like on the real photos).", "",
             "## Training", "", f"- {training_summary(a.seg)}", f"- {training_summary(a.name)}", "",
             "## Synthetic test split: canonical-surface model", ""]
    lines += canon_table(new, old, "oracle_crops") + [""] + canon_table(new, old, "seg_crops") + ["", "## Segmentation", ""]
    lines += seg_table(load_json(EXP / a.seg / "val_metrics.json"), load_json(EXP / a.seg_baseline / "val_metrics.json"))
    lines += ["", "## Real photos (2024-12-18, 334 photos)", ""]
    new_outline = load_json(a.real / f"outline_metric_{a.name}.json")
    old_outline = None
    for cand in (f"outline_metric_{a.baseline}_v2.json", f"outline_metric_{a.baseline}.json"):
        old_outline = old_outline or load_json(a.real / cand)
    lines += outline_table(new_outline, old_outline) + [""]
    sf_new, sf_old = side_flip(a.real / f"canon_{a.name}" / "points.jsonl"), side_flip(a.real / f"canon_{a.baseline}_v2" / "points.jsonl") or side_flip(a.real / f"canon_canon_{a.baseline}" / "points.jsonl")
    lines += ["| real photos, mirror test | run04 | run03 baseline |", "|---|---|---|"]
    for label, key in (("crops", "crops"), ("crops whose side flips > 20 % under mirroring", "flip_rate_gt_20pct"),
                       ("side flip agreement, median", "side_flip_agreement_median"), ("folded disagreement under mirroring (m), median", "flip_disagreement_m_median")):
        lines.append(f"| {label} | {fmt((sf_new or {}).get(key))} | {fmt((sf_old or {}).get(key))} |")
    compare_log = a.real / f"compare_{a.name}.log"
    if compare_log.exists():
        lines += ["", "```", compare_log.read_text().strip(), "```"]
    lines += ["", "## Files", "", f"- synthetic: `experiments/{a.name}/eval_test/metrics.json`", f"- real predictions: `{a.real}/canon_{a.name}/`",
              f"- outline metric: `{a.real}/outline_metric_{a.name}.json`", f"- checkpoints: `experiments/{a.name}/checkpoints/best.pt`, `experiments/{a.seg}/checkpoints/best.pt`"]
    a.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
