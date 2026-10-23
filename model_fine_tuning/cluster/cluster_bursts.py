"""Cluster a day's photos into animals and groups, starting from bursts and comparing like parts with like.

    python -m model_fine_tuning.cluster.cluster_bursts --day-dir DAY --parts DAY/parts --labels labels.json --out OUT \
        [--embedder megadescriptor | id:experiments/id_v2/checkpoints/best.pt] [--gap 1.5] [--time-weight 1.0] [--photo-level]

Nodes are bursts (cluster/bursts.py: one camera, gaps <= gap seconds; 98 % pure on 2024-12-18 at 1 s). Every chart
(body / fin / fluke, pose-normalised by cluster/parts.py from the CSE map) gets an embedding; photo-pair similarity
per part type is the cosine z-scored over all pairs of that part type ("how alike is alike for a fin"), and the
photo-pair appearance score is the best over the parts both photos show. The burst-pair appearance score is the
mean of the top-3 photo-pair scores between the two bursts (robust to one bad chart), the time term the step
function of the closest capture-time gap (match.py). Average-linkage agglomerative clustering on
appearance + time_weight x time, with the threshold self-calibrated as in cluster.py (same-burst photo pairs vs
pairs > 90 min apart; no labels). Groups: animal clusters whose photos interleave (>= group_min_pairs photo pairs within group_window s; defaults 10 and 30 s, time weight 1.5, chosen on the val days with cluster/recluster.py)
are joined. Scored against the catalog's animals (DAY_GxxAyy) and groups (Gxx): pair precision / recall / F1, ARI,
NMI, purity, plus an oracle threshold sweep for reference. --photo-level makes every photo its own node (baseline).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .bursts import make_bursts
from .cluster import cluster as agglomerate, pair_metrics, purity_report
from .match import time_term

PARTS = ("body", "fin", "fluke")


_MODELS: dict = {}


def embed_charts(chart_paths: list[Path], embedder: str, device, part: str = "body", batch: int = 32, center_crop: int = 0) -> np.ndarray:
    """L2-normalised embeddings of chart jpgs: "megadescriptor" (MegaDescriptor-L-384, no training) or
    "id:<checkpoint>" (cluster/train_id.py's WhaleIdModel, which takes the part as a token)."""
    import cv2
    if embedder == "megadescriptor":
        if "md" not in _MODELS:
            import timm
            _MODELS["md"] = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True, num_classes=0).to(device).eval()
        model, size, norm, part_idx = _MODELS["md"], 384, True, None
    elif embedder.startswith("id:"):
        if embedder not in _MODELS:
            from .train_id import PARTS as ID_PARTS, WhaleIdModel
            m, ckpt = WhaleIdModel.from_checkpoint(Path(embedder[3:]), device)
            _MODELS[embedder] = (m, int(ckpt["config"].get("args", {}).get("size", 448)), ID_PARTS)
        model, size, id_parts = _MODELS[embedder]; norm = False; part_idx = id_parts.index(part)
    else:
        raise SystemExit(f"unknown embedder {embedder}")
    out = []
    with torch.no_grad():
        for i in range(0, len(chart_paths), batch):
            imgs = []
            for f in chart_paths[i:i + batch]:
                im = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
                if center_crop and im.shape[0] > center_crop:
                    o = (im.shape[0] - center_crop) // 2; im = im[o:o + center_crop, o:o + center_crop]
                im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                imgs.append((im - 0.5) / 0.5 if norm else im)
            x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                e = model(x).float() if part_idx is None else model.embed(x, torch.full((len(imgs),), part_idx, dtype=torch.long, device=device)).float()
            out.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1))


def whole_crops(day_dir: Path, stems: list[str], margin: float = 0.15, min_px: int = 2000) -> dict:
    """Whole-whale crops (the WildFusion-style baseline input) from the day's segmentation masks, cached in day/crops."""
    import cv2
    out_dir = day_dir / "crops"; out_dir.mkdir(exist_ok=True); files = {}
    for s in stems:
        out = out_dir / f"{s}.jpg"
        if out.exists():
            files[s] = out; continue
        mp = day_dir / "canon" / f"{s}_mask.png"
        if not mp.exists():
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED) > 0
        if m.sum() < min_px:
            continue
        img = cv2.imread(str(day_dir / "images" / f"{s}.jpg")); H, W = img.shape[:2]; sy, sx = H / m.shape[0], W / m.shape[1]
        ys, xs = np.nonzero(m); x0, x1, y0, y1 = xs.min() * sx, (xs.max() + 1) * sx, ys.min() * sy, (ys.max() + 1) * sy
        side = max(x1 - x0, y1 - y0) * (1 + 2 * margin); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        xa, ya = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0)); xb, yb = int(min(cx + side / 2, W)), int(min(cy + side / 2, H))
        cv2.imwrite(str(out), img[ya:yb, xa:xb], [cv2.IMWRITE_JPEG_QUALITY, 92]); files[s] = out
    return files


def photo_scores(parts_dir: Path, stems: list[str], embedder: str, device, whole_crop_dir: Path | None = None) -> tuple[np.ndarray, dict]:
    """(n, n) photo-pair appearance score: best over shared part types of the per-type z-scored cosine; nan without a
    shared part. With whole_crop_dir the only "part" is the whole-whale crop (the WildFusion-style baseline)."""
    n = len(stems); pos = {s: i for i, s in enumerate(stems)}
    score = np.full((n, n), np.nan); per_part = {}
    for part in (("whole",) if whole_crop_dir is not None else PARTS):
        if part == "whole":
            crops = whole_crops(whole_crop_dir, stems); files = [(s, crops[s]) for s in stems if s in crops]
        else:
            files = [(s, parts_dir / "charts" / f"{s}__{part}.jpg") for s in stems]
            files = [(s, f) for s, f in files if f.exists()]
        if len(files) < 3:
            continue
        E = embed_charts([f for _, f in files], embedder, device, part=("body" if part == "whole" else part))
        S = E @ E.T; np.fill_diagonal(S, np.nan)
        Z = (S - np.nanmean(S)) / (np.nanstd(S) + 1e-9)
        idx = np.array([pos[s] for s, _ in files])
        cur = score[np.ix_(idx, idx)]
        score[np.ix_(idx, idx)] = np.where(np.isnan(cur), Z, np.fmax(cur, Z))
        per_part[part] = len(files)
    return score, per_part


def burst_scores(bursts: list[dict], score: np.ndarray, times: np.ndarray, pos: dict, topk: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Burst-pair appearance (mean of the top-k photo-pair scores) and closest capture-time gap (s)."""
    m = len(bursts); app = np.full((m, m), np.nan); dt = np.full((m, m), np.inf)
    idx = [np.array([pos[s] for s in b["photos"] if s in pos]) for b in bursts]
    for i in range(m):
        if len(idx[i]) == 0:
            continue
        ti = times[idx[i]]
        for j in range(i + 1, m):
            if len(idx[j]) == 0:
                continue
            block = score[np.ix_(idx[i], idx[j])].ravel(); block = block[np.isfinite(block)]
            if len(block):
                app[i, j] = app[j, i] = float(np.mean(np.sort(block)[-topk:]))
            d = float(np.min(np.abs(ti[:, None] - times[idx[j]][None, :]))); dt[i, j] = dt[j, i] = d
    return app, dt


def calibrated_threshold(score: np.ndarray, times: np.ndarray, same_cam: np.ndarray, time_weight: float) -> dict:
    """cluster.py's recipe at the photo level: same-burst pairs (<= 2 s, same camera) vs pairs > 90 min apart."""
    iu = np.triu_indices(len(times), 1)
    s = score[iu]; dt = np.abs(times[iu[0]] - times[iu[1]]); ok = np.isfinite(s)
    near = ok & (dt <= 2.0) & same_cam[iu]; far = ok & (dt > 5400)
    if near.sum() < 10 or far.sum() < 10:
        return {"threshold": 0.0, "burst_pairs": int(near.sum()), "far_pairs": int(far.sum())}
    bm, fm = float(np.median(s[near])), float(np.median(s[far]))
    return {"threshold": 0.5 * (bm + fm) + time_weight * 0.5 * (1.5 + (-0.5)), "burst_median": bm, "far_median": fm, "burst_pairs": int(near.sum()), "far_pairs": int(far.sum())}


def groups_from_animals(labels_pred: np.ndarray, times: np.ndarray, window: float = 30.0, min_pairs: int = 10) -> np.ndarray:
    """Join animal clusters whose photos interleave in time: >= min_pairs photo pairs within `window` seconds."""
    clusters = sorted(set(labels_pred.tolist())); k = {c: i for i, c in enumerate(clusters)}
    parent = list(range(len(clusters)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    members = {c: np.nonzero(labels_pred == c)[0] for c in clusters}
    for a in clusters:
        for b in clusters:
            if b <= a:
                continue
            d = np.abs(times[members[a]][:, None] - times[members[b]][None, :])
            if (d <= window).sum() >= min_pairs:
                ra, rb = find(k[a]), find(k[b])
                if ra != rb:
                    parent[ra] = rb
    return np.array([find(k[c]) for c in labels_pred])


def evaluate(pred: np.ndarray, gt: list[str]) -> dict:
    known = np.array([g != "?" for g in gt])
    if known.sum() < 10:
        return {}
    m = pair_metrics(pred[known], np.array(gt)[known]); m.update(purity_report(pred[known], [g for g in gt if g != "?"]))
    return m


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--day-dir", type=Path, required=True); p.add_argument("--parts", type=Path, default=None); p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--embedder", default="megadescriptor"); p.add_argument("--gap", type=float, default=1.5)
    p.add_argument("--time-weight", type=float, default=1.5); p.add_argument("--photo-level", action="store_true"); p.add_argument("--device", default="cuda")
    p.add_argument("--group-window", type=float, default=30.0); p.add_argument("--group-min-pairs", type=int, default=10)
    p.add_argument("--whole-crop", action="store_true", help="baseline: match whole-whale crops instead of part charts")
    a = p.parse_args(argv)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    day = a.day_dir.name; parts_dir = a.parts or (a.day_dir / "parts")
    ts = json.loads((a.day_dir / "timestamps.json").read_text())
    stems = sorted(s for s, t in ts.items() if t.get("time") is not None)
    pos = {s: i for i, s in enumerate(stems)}; times = np.array([ts[s]["time"] for s in stems], float)
    cam = np.array([f"{ts[s].get('folder', '?')}|{ts[s].get('camera', '?')}" for s in stems]); same_cam = cam[:, None] == cam[None, :]
    L = json.loads(a.labels.read_text())["days"].get(day, {}).get("photos", {}) if a.labels else {}
    gt_animal = [L.get(s.split("__")[-1], "?") for s in stems]; gt_group = [g.split("_")[-1][:3] if g != "?" else "?" for g in gt_animal]
    score, per_part = photo_scores(parts_dir, stems, a.embedder, device, whole_crop_dir=a.day_dir if a.whole_crop else None)
    print(f"{day}: {len(stems)} photos, charts {per_part}, photo pairs with a shared part {int(np.isfinite(score).sum() // 2)}", flush=True)
    cal = calibrated_threshold(score, times, same_cam, a.time_weight)
    bursts = [{"id": i, "photos": [s], "start": times[i], "end": times[i]} for i, s in enumerate(stems)] if a.photo_level else make_bursts(ts, a.gap)
    app, dt = burst_scores(bursts, score, times, pos)
    total = np.where(np.isfinite(app), app, 0.0) + a.time_weight * time_term(dt)
    np.fill_diagonal(total, 0.0)
    pred_b = agglomerate(total, cal["threshold"])
    pred = np.full(len(stems), -1)
    for b, c in zip(bursts, pred_b):
        for s in b["photos"]:
            if s in pos:
                pred[pos[s]] = c
    pred[pred < 0] = pred.max() + 1 + np.arange((pred < 0).sum())
    groups = groups_from_animals(pred, times, a.group_window, a.group_min_pairs)
    res = {"day": day, "photos": len(stems), "bursts": len(bursts), "embedder": a.embedder, "whole_crop": a.whole_crop, "gap_s": None if a.photo_level else a.gap, "time_weight": a.time_weight,
           "calibration": cal, "charts": per_part, "animals": evaluate(pred, gt_animal), "groups": evaluate(groups, gt_group)}
    sweep = []
    for thr in np.linspace(np.nanpercentile(total, 50), np.nanpercentile(total, 99.5), 25):
        pb = agglomerate(total, float(thr)); pp = np.full(len(stems), -1)
        for b, c in zip(bursts, pb):
            for s in b["photos"]:
                if s in pos:
                    pp[pos[s]] = c
        m = evaluate(pp, gt_animal)
        if m:
            sweep.append({"threshold": round(float(thr), 3), "ari": round(m["ari"], 4), "pair_f1": round(m["pair_f1"], 4), "clusters": m["clusters"]})
    res["oracle_sweep_animals"] = sweep
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "scores.npz", stems=np.array(stems), photo_score=score, burst_app=app, burst_dt=dt, pred_animal=pred, pred_group=groups)
    (a.out / "metrics.json").write_text(json.dumps(res, indent=1))
    (a.out / "bursts.json").write_text(json.dumps(bursts, indent=1))
    m, g = res["animals"], res["groups"]
    if m:
        print(f"  animals: clusters {m['clusters']} (truth {m['individuals']})  ARI {m['ari']:.3f}  pair P/R/F1 {m['pair_precision']:.3f}/{m['pair_recall']:.3f}/{m['pair_f1']:.3f}  purity {m['cluster_purity']:.3f}  thr {cal['threshold']:.2f}  best oracle ARI {max(s['ari'] for s in sweep):.3f}")
        print(f"  groups:  clusters {g['clusters']} (truth {g['individuals']})  ARI {g['ari']:.3f}  pair P/R/F1 {g['pair_precision']:.3f}/{g['pair_recall']:.3f}/{g['pair_f1']:.3f}  purity {g['cluster_purity']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
