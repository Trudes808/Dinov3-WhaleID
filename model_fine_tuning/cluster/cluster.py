"""Cluster a day's photos by individual from the pairwise scores, and score the result against
the Lightroom labels (evaluation only).

    python -m model_fine_tuning.cluster.cluster --scores OUT/scores.npz --labels labels.json --out OUT \
        [--sources dino_1 colour] [--hypothesis best|same] [--time-weight 1.0] [--threshold auto]

Total score = mean over the chosen sources of the appearance z-score (per pair; the better of
the two side hypotheses when --hypothesis best) + time-weight x time term. Pairs with no
comparable region get the time term alone (they can still join a cluster through the photos
in between). Average-linkage agglomerative clustering on (max - score). The threshold is
self-calibrated from the data, never from the labels: pairs shot within 2 s on the same camera
are almost surely the same whale, pairs more than 90 min apart almost surely not (groups
change); the threshold is the midpoint between the medians of those two appearance-score
populations, applied to the total score. A label-based sweep is reported too, marked oracle.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def pair_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    same_p = pred[:, None] == pred[None, :]
    same_g = gt[:, None] == gt[None, :]
    off = ~np.eye(len(pred), dtype=bool)
    tp = np.sum(same_p & same_g & off) / 2; fp = np.sum(same_p & ~same_g & off) / 2; fn = np.sum(~same_p & same_g & off) / 2
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {"pair_precision": float(prec), "pair_recall": float(rec), "pair_f1": float(2 * prec * rec / max(prec + rec, 1e-9)),
            "ari": float(adjusted_rand_score(gt, pred)), "nmi": float(normalized_mutual_info_score(gt, pred)), "clusters": int(len(set(pred))), "individuals": int(len(set(gt)))}


def purity_report(pred: np.ndarray, labels: list[str]) -> dict:
    by_cluster = defaultdict(Counter)
    for c, l in zip(pred, labels):
        by_cluster[int(c)][l] += 1
    by_label = defaultdict(Counter)
    for c, l in zip(pred, labels):
        by_label[l][int(c)] += 1
    return {"cluster_purity": float(sum(max(v.values()) for v in by_cluster.values()) / len(pred)),
            "label_completeness": float(sum(max(v.values()) for v in by_label.values()) / len(pred)),
            "per_individual": {l: {"photos": sum(v.values()), "largest_cluster_share": round(max(v.values()) / sum(v.values()), 3), "clusters": len(v)} for l, v in sorted(by_label.items())},
            "per_cluster": {str(c): dict(v.most_common()) for c, v in sorted(by_cluster.items(), key=lambda kv: -sum(kv[1].values()))}}


def total_scores(z: dict, sources: list[str], hypothesis: str, time_weight: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (total score, appearance score, hypothesis used: 0 same / 1 flip / -1 none)."""
    app = []
    hyp_used = None
    for s in sources:
        same, flip = z[f"{s}__same"], z[f"{s}__flip"]
        if hypothesis == "same":
            a = same
        else:
            a = np.where(np.isnan(same), flip, np.where(np.isnan(flip), same, np.maximum(same, flip)))
        app.append(a)
    app = np.nanmean(np.stack(app), axis=0) if len(app) > 1 else app[0]
    with np.errstate(invalid="ignore"):
        best = np.where(np.isnan(app), -1, 0)
        if hypothesis != "same" and len(sources) == 1:
            s = sources[0]
            best = np.where(np.isnan(app), -1, np.where(np.nan_to_num(z[f"{s}__flip"], nan=-np.inf) > np.nan_to_num(z[f"{s}__same"], nan=-np.inf), 1, 0))
    total = np.where(np.isnan(app), 0.0, app) + time_weight * z["time_term"]
    return total, app, best


def self_calibrated_threshold(app: np.ndarray, dt: np.ndarray, same_cam: np.ndarray) -> dict:
    off = ~np.eye(len(app), dtype=bool)
    pos = app[(dt <= 2.0) & same_cam & off & np.isfinite(app)]
    neg = app[(dt >= 5400.0) & off & np.isfinite(app)]
    thr = float((np.median(pos) + np.median(neg)) / 2) if len(pos) and len(neg) else 0.0
    return {"threshold": thr, "burst_pairs": int(len(pos) // 2), "burst_median": float(np.median(pos)) if len(pos) else None,
            "far_pairs": int(len(neg) // 2), "far_median": float(np.median(neg)) if len(neg) else None,
            "burst_above_thr": float(np.mean(pos > thr)) if len(pos) else None, "far_below_thr": float(np.mean(neg < thr)) if len(neg) else None}


def cluster(total: np.ndarray, threshold: float) -> np.ndarray:
    d = np.max(total) - total
    np.fill_diagonal(d, 0.0)
    d = 0.5 * (d + d.T)
    model = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average", distance_threshold=max(float(np.max(total) - threshold), 1e-9))
    return model.fit_predict(d)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sources", nargs="*", default=["dino_1"])
    p.add_argument("--hypothesis", choices=["best", "same"], default="best")
    p.add_argument("--time-weight", type=float, default=1.0)
    p.add_argument("--threshold", default="auto")
    p.add_argument("--tag", default=None)
    args = p.parse_args(argv)
    z = dict(np.load(args.scores, allow_pickle=False))
    stems = [str(s) for s in z["stems"]]
    lab = json.loads(args.labels.read_text())["labels"]
    gt = np.array([lab[s][0] for s in stems])
    total, app, hyp = total_scores(z, args.sources, args.hypothesis, args.time_weight)
    cal = self_calibrated_threshold(app, z["dt_s"], z["same_camera"])
    thr = cal["threshold"] if args.threshold == "auto" else float(args.threshold)
    pred = cluster(total, thr)
    metrics = pair_metrics(pred, gt)
    metrics.update({"threshold": thr, "calibration": cal, "sources": args.sources, "hypothesis": args.hypothesis, "time_weight": args.time_weight,
                    "pairs_scored": int(np.isfinite(app).sum() // 2), "flip_hypothesis_used_frac": float(np.mean(hyp[np.triu_indices(len(stems), 1)] == 1)) if args.hypothesis == "best" else 0.0})
    # oracle sweep (labels used only to report, never to pick)
    sweep = []
    for t in np.linspace(thr - 2.0, thr + 2.0, 21):
        pr = cluster(total, float(t)); m = pair_metrics(pr, gt)
        sweep.append({"threshold": round(float(t), 3), "ari": round(m["ari"], 4), "pair_f1": round(m["pair_f1"], 4), "clusters": m["clusters"]})
    metrics["oracle_sweep"] = sweep
    metrics.update(purity_report(pred, list(gt)))
    tag = args.tag or ("+".join(args.sources) + f"_{args.hypothesis}_t{args.time_weight:g}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"clusters_{tag}.json").write_text(json.dumps({"stems": stems, "cluster": pred.tolist(), "label": gt.tolist(), "total_threshold": thr}, indent=1))
    (args.out / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=1))
    print(f"[{tag}] clusters={metrics['clusters']} (individuals {metrics['individuals']})  ARI={metrics['ari']:.3f} NMI={metrics['nmi']:.3f} "
          f"pair P/R/F1={metrics['pair_precision']:.3f}/{metrics['pair_recall']:.3f}/{metrics['pair_f1']:.3f}  purity={metrics['cluster_purity']:.3f} completeness={metrics['label_completeness']:.3f}  "
          f"thr={thr:.2f} (burst med {cal['burst_median']}, far med {cal['far_median']})  best oracle ARI={max(s['ari'] for s in sweep):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
