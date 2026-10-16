"""Metrics report for the run04 fine-tunes against the run03 baselines, as Markdown.

    .venv/bin/python scripts/run04_report.py [--suffix _smoke] --real experiments/real_2024-12-18 --out experiments/run04_report.md

Columns: run04 contour-weighted (seg_r04_cw / canon_r04_cw), run04 plain (seg_r04_ft4 / canon_r04_ft4: the run03 recipe
on the run04 data), the run03 models (seg_r03_ft4 / canon_r03_ft4). Synthetic numbers are on the run04 test split for all
three (the run03 models are cross-evaluated on it: eval_run04_test), with run03's own test split as a reference column.
Real-photo numbers are label-free and comparable across the board. Missing results print as "-".
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

EXP = Path("experiments")

CANON_ROWS = [
    ("surface error mean (m)", "surface_err_m_mean", "lower"),
    ("surface error median (m)", "surface_err_m_median", "lower"),
    ("torso / fluke / fin error (m)", ("surface_err_torso_m_mean", "surface_err_fluke_m_mean", "surface_err_fin_m_mean"), "lower"),
    ("visibility IoU", "vis_iou_mean", "higher"),
    ("visibility boundary F1, open air (pooled)", "vis_boundary_f1_open_pooled", "higher"),
    ("visibility boundary F1, at seams (pooled)", "vis_boundary_f1_seam_pooled", "higher"),
    ("visibility boundary F1, open air (image mean)", "vis_bf1_open_mean", "higher"),
    ("true-contour recall (pooled)", "vis_true_contour_recall_pooled", "higher"),
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
    ("images evaluated", "images_evaluated", None),
]
SEG_ROWS = [
    ("dataset IoU", "dataset_iou", "higher"), ("mean image IoU", "mean_image_iou", "higher"), ("median image IoU", "median_image_iou", "higher"),
    ("precision", "precision", "higher"), ("recall", "recall", "higher"), ("IoU >= 0.5", "iou_at_0.5", "higher"), ("IoU >= 0.75", "iou_at_0.75", "higher"),
    ("boundary F1 (all, image mean)", "boundary_f1", "higher"), ("boundary F1, open air (pooled)", "boundary_f1_open_pooled", "higher"),
    ("boundary precision / recall, open air (pooled)", ("boundary_precision_open_pooled", "boundary_recall_open_pooled"), "higher"),
    ("boundary F1, at seams (pooled)", "boundary_f1_seam_pooled", "higher"), ("boundary F1, open air (image mean)", "boundary_f1_open", "higher"),
    ("true-contour recall (pooled)", "true_contour_recall_pooled", "higher"), ("negatives clean", "negatives_clean", "higher"), ("images", "images", None),
]
OUTLINE_ROWS = [
    ("IoU above water, median", ("iou_above_water", "median"), "higher"), ("IoU above water, mean", ("iou_above_water", "mean"), "higher"),
    ("photos with IoU >= 0.5", ("iou_above_0.5",), "higher"), ("coverage, median", ("coverage", "median"), "higher"),
    ("excess above water, median", ("excess_above_water", "median"), "lower"), ("PnP inlier fraction, median", ("inlier_fraction", "median"), "higher"),
    ("median reprojection error (px), median", ("median_reproj_px", "median"), "lower"), ("photos scored", ("scored",), None),
]
MIRROR_ROWS = [("crops", "crops", None), ("crops whose side flips > 20 % under mirroring", "flip_rate_gt_20pct", "lower"),
               ("side flip agreement, median", "side_flip_agreement_median", "higher"), ("folded disagreement under mirroring (m), median", "flip_disagreement_m_median", "lower")]


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def first(*paths: Path):
    for p in paths:
        d = load_json(p)
        if d is not None:
            return d
    return None


def fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, (tuple, list)):
        return " / ".join(fmt(v) for v in value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def pick(metrics, key):
    if not metrics:
        return None
    if isinstance(key, tuple):
        if all(isinstance(k, str) and k in metrics for k in key) or any(isinstance(metrics.get(k), (int, float)) for k in key):
            return tuple(metrics.get(k) for k in key)
        d = metrics
        for k in key:
            d = d.get(k) if isinstance(d, dict) else None
        return d
    return metrics.get(key)


def table(title: str, columns: list[dict], rows: list[tuple], ranked: list[int]) -> list[str]:
    """One Markdown table; the best value among the ranked columns is marked with **bold**."""
    lines = [f"| {title} | " + " | ".join(c["label"] for c in columns) + " | best |", "|---|" + "---|" * (len(columns) + 1)]
    for label, key, better in rows:
        vals = [pick(c["metrics"], key) for c in columns]
        best = ""
        if better:
            scored = [(i, v) for i, v in enumerate(vals) if i in ranked and isinstance(v, (int, float)) and not isinstance(v, bool)]
            if len(scored) >= 2:
                i_best = (min if better == "lower" else max)(scored, key=lambda t: t[1])[0]
                best = columns[i_best]["label"]
        lines.append(f"| {label} | " + " | ".join(fmt(v) for v in vals) + f" | {best} |")
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
        return f"{name}: not trained"
    events = [json.loads(l) for l in open(log)]
    done = [e for e in events if e.get("event") == "done"]
    val = load_json(EXP / name / "val_metrics.json") or {}
    best = {k: round(v, 4) for k, v in val.items() if k in ("score", "surface_err_m_mean", "dataset_iou", "boundary_f1_open", "vis_bf1_open_mean")}
    cfg = (load_json(EXP / name / "config.json") or {}).get("args", {})
    contour = {k: v for k, v in cfg.items() if k.startswith("contour_") and v not in (1.0, 3)}
    secs = done[0].get("elapsed_s") if done else None
    return f"{name}: {'finished' if done else 'not finished'}" + (f" in {secs/3600:.1f} h" if isinstance(secs, (int, float)) else "") + \
        f", epochs {cfg.get('epochs')}, train images {(load_json(EXP / name / 'config.json') or {}).get('train_images')}, best val {best}" + \
        (f", contour weights {contour}" if contour else "")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suffix", default=""); p.add_argument("--real", type=Path, default=Path("experiments/real_2024-12-18"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    s = a.suffix
    models = [  # (column label, seg experiment, canon experiment, eval dir, real-photo result names)
        ("run04 contour-weighted", f"seg_r04_cw{s}", f"canon_r04_cw{s}", "eval_test", [f"canon_r04_cw{s}"]),
        ("run04 plain", f"seg_r04_ft4{s}", f"canon_r04_ft4{s}", "eval_test", [f"canon_r04_ft4{s}"]),
        ("run03 models", "seg_r03_ft4", "canon_r03_ft4", f"eval_run04{s}_test", ["canon_r03_ft4_v2", "canon_r03_ft4"]),
    ]
    reference = ("run03 (own test split)", "seg_r03_ft4", "canon_r03_ft4", "eval_test_v2|eval_test", [])
    ranked = [0, 1, 2]

    def canon_cols(section):
        cols = []
        for label, seg, canon, ev, _ in models + [reference]:
            m = first(*[EXP / canon / e / "metrics.json" for e in ev.split("|")])
            cols.append({"label": label, "metrics": (m or {}).get(section)})
        return cols

    def seg_cols():
        cols = []
        for label, seg, canon, ev, _ in models + [reference]:
            m = first(*[EXP / seg / e / "metrics.json" for e in ev.split("|")]) or (load_json(EXP / seg / "val_metrics.json") if label.startswith("run03 (own") else None)
            cols.append({"label": label, "metrics": m})
        return cols

    real_cols, real_direct_cols, mirror_cols = [], [], []
    for label, seg, canon, ev, names in models:
        for cols, suffix in ((real_cols, ""), (real_direct_cols, "_direct")):
            m = first(*[a.real / f"outline_metric_{n}{suffix}.json" for n in names])
            cols.append({"label": label, "metrics": m.get("summary") if m else None})
        sf = None
        for n in names:
            sf = sf or side_flip(a.real / f"canon_{n}" / "points.jsonl") or side_flip(a.real / f"canon_canon_{n}" / "points.jsonl")
        mirror_cols.append({"label": label, "metrics": sf})

    lines = [f"# run04 fine-tunes vs run03 ({datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')})", "",
             "Synthetic metrics are on the **run04 test split** for the three model columns (the run03 models are evaluated on it too, so",
             "the comparison is like for like: run04's split has the rigged whale, calves, close-ups and long-lens framing, a harder set",
             "than run03's); run03's own test split is the reference column. 'open air' / 'at seams' boundary scores use the run04 contour",
             "label: the whale's own silhouette against sky or water vs the waterline / foam / calf edges. Real-photo numbers are label-free.", "",
             "## Training", ""] + [f"- {training_summary(n)}" for n in (models[0][1], models[0][2], models[1][1], models[1][2], "seg_r03_ft4", "canon_r03_ft4")]
    lines += ["", "## Segmentation (test split)", ""] + table("segmentation", seg_cols(), SEG_ROWS, ranked)
    lines += ["", "## Canonical-surface model (test split)", "", "Oracle crops (ground-truth box):", ""] + table("oracle crops", canon_cols("oracle_crops"), CANON_ROWS, ranked)
    lines += ["", "Crops from the column's own segmentation model:", ""] + table("predicted crops", canon_cols("seg_crops"), CANON_ROWS, ranked)
    lines += ["", "## Real photos (2024-12-18, 334 photos)", "", "Rigid PnP on the dense map (rotation and translation from the map):", ""] + \
        table("label-free outline metric, PnP pose", real_cols, OUTLINE_ROWS, [0, 1, 2])
    lines += ["", "Rotation from the model's direct head, translation fitted on the map (a rigid PnP on a bent whale takes the rotation with it):", ""] + \
        table("label-free outline metric, direct rotation", real_direct_cols, OUTLINE_ROWS, [0, 1, 2])
    lines += [""] + table("mirror test", mirror_cols, MIRROR_ROWS, [0, 1, 2])
    for label, seg, canon, ev, names in models:
        for n in names:
            log = a.real / f"compare_{n}.log"
            if log.exists():
                lines += ["", f"<details><summary>side-flip comparison, {label}</summary>", "", "```", log.read_text().strip(), "```", "</details>"]
                break
    lines += ["", "## Files", ""]
    for label, seg, canon, ev, names in models:
        lines.append(f"- {label}: `experiments/{canon}/{ev.split('|')[0]}/metrics.json`, `experiments/{seg}/{ev.split('|')[0]}/metrics.json`, "
                     f"checkpoints `experiments/{canon}/checkpoints/best.pt`, `experiments/{seg}/checkpoints/best.pt`" +
                     (f", real predictions `{a.real}/canon_{names[0]}/`, outline metric `{a.real}/outline_metric_{names[0]}.json`" if names else ""))
    a.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
