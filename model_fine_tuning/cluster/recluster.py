"""Re-cluster days from the saved burst score matrices of cluster_bursts.py (no embedding recompute) over a grid of
time weights and group rules, reporting the mean animal ARI / pair F1 on the val days and, separately, on the eval
days. Tuning reads the val days only; the eval numbers are printed for the record.

    python -m model_fine_tuning.cluster.recluster --root experiments/days_all --tag cluster_bursts_md --labels labels.json --days days.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cluster import cluster as agglomerate
from .cluster_bursts import calibrated_threshold, evaluate, groups_from_animals
from .match import time_term


def load_day(root: Path, day: str, tag: str, labels: dict):
    z = np.load(root / day / tag / "scores.npz", allow_pickle=True)
    bursts = json.loads((root / day / tag / "bursts.json").read_text())
    ts = json.loads((root / day / "timestamps.json").read_text())
    stems = [str(s) for s in z["stems"]]; pos = {s: i for i, s in enumerate(stems)}
    times = np.array([ts[s]["time"] for s in stems], float)
    cam = np.array([f"{ts[s].get('folder', '?')}|{ts[s].get('camera', '?')}" for s in stems]); same_cam = cam[:, None] == cam[None, :]
    L = labels["days"].get(day, {}).get("photos", {})
    gt_animal = [L.get(s.split("__")[-1], "?") for s in stems]; gt_group = [g.split("_")[-1][:3] if g != "?" else "?" for g in gt_animal]
    return dict(score=z["photo_score"], app=z["burst_app"], dt=z["burst_dt"], bursts=bursts, stems=stems, pos=pos, times=times, same_cam=same_cam, gt_animal=gt_animal, gt_group=gt_group)


def run(d: dict, time_weight: float, thr_offset: float, window: float, min_pairs: int) -> dict:
    cal = calibrated_threshold(d["score"], d["times"], d["same_cam"], time_weight)
    total = np.where(np.isfinite(d["app"]), d["app"], 0.0) + time_weight * time_term(d["dt"]); np.fill_diagonal(total, 0.0)
    pred_b = agglomerate(total, cal["threshold"] + thr_offset)
    pred = np.full(len(d["stems"]), -1)
    for b, c in zip(d["bursts"], pred_b):
        for s in b["photos"]:
            if s in d["pos"]:
                pred[d["pos"][s]] = c
    pred[pred < 0] = pred.max() + 1 + np.arange((pred < 0).sum())
    groups = groups_from_animals(pred, d["times"], window, min_pairs)
    return {"animals": evaluate(pred, d["gt_animal"]), "groups": evaluate(groups, d["gt_group"])}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--tag", required=True); p.add_argument("--labels", type=Path, required=True); p.add_argument("--days", type=Path, required=True)
    a = p.parse_args(argv)
    labels = json.loads(a.labels.read_text()); days = json.loads(a.days.read_text())
    data = {}
    for d in days["val"] + days["eval"]:
        if (a.root / d / a.tag / "scores.npz").exists():
            data[d] = load_day(a.root, d, a.tag, labels)
    print(f"days loaded: val {[d for d in days['val'] if d in data]} eval {[d for d in days['eval'] if d in data]}")
    rows = []
    for tw in (0.0, 0.25, 0.5, 1.0, 1.5):
        for off in (-0.5, -0.25, 0.0, 0.25, 0.5):
            res = {d: run(data[d], tw, off, 120.0, 2) for d in data}
            v = [res[d]["animals"] for d in days["val"] if d in res]; e = [res[d]["animals"] for d in days["eval"] if d in res]
            rows.append({"time_weight": tw, "thr_offset": off, "val_ari": float(np.mean([m["ari"] for m in v])), "val_f1": float(np.mean([m["pair_f1"] for m in v])),
                         "eval_ari": float(np.mean([m["ari"] for m in e])), "eval_f1": float(np.mean([m["pair_f1"] for m in e]))})
    rows.sort(key=lambda r: -r["val_ari"])
    print("animals: best settings by val ARI (eval shown for the record)")
    for r in rows[:6]:
        print(f"  tw {r['time_weight']:4.2f} thr {r['thr_offset']:+.2f}  val ARI {r['val_ari']:.3f} F1 {r['val_f1']:.3f}  | eval ARI {r['eval_ari']:.3f} F1 {r['eval_f1']:.3f}")
    base = [r for r in rows if r["time_weight"] == 1.0 and r["thr_offset"] == 0.0][0]
    print(f"  (current tw 1.00 thr +0.00  val ARI {base['val_ari']:.3f} F1 {base['val_f1']:.3f}  | eval ARI {base['eval_ari']:.3f} F1 {base['eval_f1']:.3f})")
    best = rows[0]
    grows = []
    for window in (30.0, 60.0, 120.0, 300.0):
        for mp in (1, 2, 5, 10):
            res = {d: run(data[d], best["time_weight"], best["thr_offset"], window, mp) for d in data}
            v = [res[d]["groups"] for d in days["val"] if d in res]; e = [res[d]["groups"] for d in days["eval"] if d in res]
            grows.append({"window": window, "min_pairs": mp, "val_ari": float(np.mean([m["ari"] for m in v])), "val_f1": float(np.mean([m["pair_f1"] for m in v])),
                          "eval_ari": float(np.mean([m["ari"] for m in e])), "eval_f1": float(np.mean([m["pair_f1"] for m in e]))})
    grows.sort(key=lambda r: -r["val_ari"])
    print("groups: best interleaving rules by val ARI (with the best animal settings)")
    for r in grows[:5]:
        print(f"  window {r['window']:5.0f}s min_pairs {r['min_pairs']:2d}  val ARI {r['val_ari']:.3f} F1 {r['val_f1']:.3f}  | eval ARI {r['eval_ari']:.3f} F1 {r['eval_f1']:.3f}")
    (a.root / f"recluster_{a.tag}.json").write_text(json.dumps({"animals": rows, "groups": grows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
