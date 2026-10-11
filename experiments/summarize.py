"""Print a compact table of the validation curves and test metrics of the experiments.

    .venv/bin/python experiments/summarize.py [names...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def rows(path: Path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    names = sys.argv[1:] or sorted(p.name for p in ROOT.iterdir() if (p / "log.jsonl").exists())
    for name in names:
        exp = ROOT / name
        cfg = json.loads((exp / "config.json").read_text())
        vals = [r for r in rows(exp / "log.jsonl") if r["event"] == "val"]
        done = any(r["event"] == "done" for r in rows(exp / "log.jsonl"))
        print(f"== {name}  task={cfg['task']} unfreeze={cfg['model']['unfreeze_blocks']} train={cfg['train_images']} val={cfg['val_images']} "
              f"params={cfg['params']} epochs_logged={len(vals)} {'done' if done else 'RUNNING'}")
        if cfg["task"] == "segmentation":
            best = max(vals, key=lambda r: r["dataset_iou"]) if vals else None
            if best:
                print(f"   best val: epoch {best['epoch']} dataset_iou={best['dataset_iou']:.4f} mean_image_iou={best['mean_image_iou']:.4f} "
                      f"negatives_clean={best['negatives_clean']:.3f} P={best['precision']:.3f} R={best['recall']:.3f}")
        else:
            best = max(vals, key=lambda r: r["score"]) if vals else None
            if best:
                print(f"   best val: epoch {best['epoch']} u_mae={best['u_mae_mean']:.4f} angle_mae={best['angle_mae_deg_mean']:.1f} vis_iou={best['vis_iou_mean']:.3f} "
                      f"rot_direct_med={best['rot_direct_deg_median']:.1f} rot_pnp_med={best.get('rot_pnp_deg_median', float('nan')):.1f} "
                      f"pnp<15={best.get('rot_pnp_under_15deg')} direct<15={best.get('rot_direct_under_15deg')}")
        for ev in sorted(exp.glob("eval_*/metrics.json")):
            m = json.loads(ev.read_text())
            if cfg["task"] == "segmentation":
                print(f"   {ev.parent.name}: dataset_iou={m['dataset_iou']:.4f} mean_image_iou={m['mean_image_iou']:.4f} boundary_f1={m['boundary_f1']:.3f} "
                      f"negatives_clean={m['negatives_clean']:.3f} iou>=0.5: {m['iou_at_0.5']:.3f}")
            else:
                for tag in ("oracle_crops", "seg_crops"):
                    if tag in m:
                        t = m[tag]
                        print(f"   {ev.parent.name} [{tag}]: u_mae={t['u_mae_mean']:.4f} ({t['u_mae_metres_mean']:.2f} m) angle_mae={t['angle_mae_deg_mean']:.1f} "
                              f"vis_iou={t['vis_iou_mean']:.3f} rot_direct_med={t['rot_direct_deg_median']:.1f} rot_pnp_med={t.get('rot_pnp_deg_median')} "
                              f"pnp<15={t.get('rot_pnp_under_15deg')} pck0.1={t.get('pck_0.1_mean')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
