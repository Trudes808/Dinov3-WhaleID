"""Tune the v3 clustering rule (time weight, threshold offset, group rule) on the val days from the saved node
matrices of cluster_day.py (no embedding recompute); eval days are printed for the record only.

    python -m model_fine_tuning.cluster.tune_v3 --root experiments/days_all --tag cluster_v3 --days days.json --labels labels.json [--score-tag _x]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cluster_bursts import evaluate, groups_from_animals
from .linkage import constrained_average_linkage
from .match import time_term


def load(root: Path, day: str, tag: str, score_tag: str, labels: dict) -> dict | None:
    f = root / day / tag / f"scores{score_tag}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True); tracks = json.loads((root / day / tag / f"tracklets{score_tag}.json").read_text())
    ts = json.loads((root / day / "timestamps.json").read_text())
    keys = [str(k) for k in z["keys"]]; pos = {k: i for i, k in enumerate(keys)}
    L = labels["days"].get(day, {}).get("photos", {})
    main = [(k, i) for k, i in pos.items() if "~" not in k or k.endswith("~0")]
    stems = [k.rsplit("~", 1)[0] if "~" in k else k for k, _ in main]; midx = np.array([i for _, i in main])
    gt_a = [L.get(s.split("__")[-1], "?") for s in stems]; gt_g = [g.split("_")[-1][:3] if g != "?" else "?" for g in gt_a]
    return dict(app=z["node_app"], dt=z["node_dt"], same_burst=z["same_burst"], CL=z["cannot_link"], tracks=tracks, pos=pos, n=len(keys), midx=midx,
                ptimes=np.array([ts[s]["time"] for s in stems], float), gt_a=gt_a, gt_g=gt_g, near=float(z["near_median"]), far=float(z["far_median"]))


def run(d: dict, tw: float, off: float, window: float, min_pairs: int, fixed: float | None = None) -> dict:
    tt = time_term(d["dt"]); tt[d["same_burst"]] = 0.0
    total = np.where(np.isfinite(d["app"]), d["app"], 0.0) + tw * tt; np.fill_diagonal(total, 0.0)
    thr = 0.5 * (d["near"] + d["far"]) + tw * off if fixed is None else fixed
    lab = constrained_average_linkage(total, thr, d["CL"])
    pred = np.full(d["n"], -1)
    for t, c in zip(d["tracks"], lab):
        for k in t["keys"]:
            if k in d["pos"]:
                pred[d["pos"][k]] = c
    pred[pred < 0] = pred.max() + 1 + np.arange((pred < 0).sum())
    pc = pred[d["midx"]]
    return {"animals": evaluate(pc, d["gt_a"]), "groups": evaluate(groups_from_animals(pc, d["ptimes"], window, min_pairs), d["gt_g"])}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--tag", required=True); p.add_argument("--days", type=Path, required=True); p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--score-tag", default=""); p.add_argument("--fixed", type=float, nargs="*", default=[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], help="absolute thresholds to try as well")
    a = p.parse_args(argv)
    labels = json.loads(a.labels.read_text()); days = json.loads(a.days.read_text())
    data = {d: x for d in days["val"] + days["eval"] if (x := load(a.root, d, a.tag, a.score_tag, labels)) is not None}
    print(f"days: val {[d for d in days['val'] if d in data]} eval {[d for d in days['eval'] if d in data]}")
    rows = []
    for tw in (0.5, 1.0, 1.5, 2.0):
        for off in (-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0):
            res = {d: run(data[d], tw, off, 30.0, 10) for d in data}
            v = [res[d]["animals"] for d in days["val"] if d in res]; e = [res[d]["animals"] for d in days["eval"] if d in res]
            rows.append({"time_weight": tw, "thr_offset": off, "val_ari": float(np.mean([m["ari"] for m in v])) if v else np.nan, "val_f1": float(np.mean([m["pair_f1"] for m in v])) if v else np.nan,
                         "eval_ari": float(np.mean([m["ari"] for m in e])) if e else np.nan, "eval_f1": float(np.mean([m["pair_f1"] for m in e])) if e else np.nan,
                         "per_day": {d: round(res[d]["animals"]["ari"], 3) for d in res}})
    for tw in (0.5, 1.0, 1.5, 2.0):
        for thr in a.fixed:
            res = {d: run(data[d], tw, 0.0, 30.0, 10, fixed=thr) for d in data}
            v = [res[d]["animals"] for d in days["val"] if d in res]; e = [res[d]["animals"] for d in days["eval"] if d in res]
            rows.append({"time_weight": tw, "thr_offset": None, "fixed": thr, "val_ari": float(np.mean([m["ari"] for m in v])) if v else np.nan, "val_f1": float(np.mean([m["pair_f1"] for m in v])) if v else np.nan,
                         "eval_ari": float(np.mean([m["ari"] for m in e])) if e else np.nan, "eval_f1": float(np.mean([m["pair_f1"] for m in e])) if e else np.nan,
                         "per_day": {d: round(res[d]["animals"]["ari"], 3) for d in res}})
    rows.sort(key=lambda r: -r["val_ari"])
    print("animals: best by val ARI (eval for the record); 'fixed' = absolute threshold on the score scale")
    for r in rows[:10]:
        print(f"  tw {r['time_weight']:4.2f} {'fixed ' + format(r['fixed'], '+.2f') if r['thr_offset'] is None else 'off ' + format(r['thr_offset'], '+.2f'):11s}  val ARI {r['val_ari']:.3f} F1 {r['val_f1']:.3f} | eval ARI {r['eval_ari']:.3f} F1 {r['eval_f1']:.3f}  {r['per_day']}")
    best = rows[0]
    grows = []
    for window in (30.0, 60.0, 120.0):
        for mp in (2, 5, 10, 20):
            res = {d: run(data[d], best["time_weight"], best["thr_offset"] or 0.0, window, mp, fixed=best.get("fixed")) for d in data}
            v = [res[d]["groups"] for d in days["val"] if d in res]; e = [res[d]["groups"] for d in days["eval"] if d in res]
            grows.append({"window": window, "min_pairs": mp, "val_ari": float(np.mean([m["ari"] for m in v])), "eval_ari": float(np.mean([m["ari"] for m in e])) if e else np.nan})
    grows.sort(key=lambda r: -r["val_ari"])
    print("groups: best interleaving rule (with the best animal settings)")
    for r in grows[:4]:
        print(f"  window {r['window']:5.0f}s min_pairs {r['min_pairs']:2d}  val ARI {r['val_ari']:.3f} | eval ARI {r['eval_ari']:.3f}")
    (a.root / f"tune_{a.tag}{a.score_tag}.json").write_text(json.dumps({"animals": rows, "groups": grows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
