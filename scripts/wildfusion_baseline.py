"""WildFusion baseline on one labelled day: whole-whale crops (from our segmentation masks), MegaDescriptor cosine
similarity, ALIKED + LightGlue local matching, isotonic calibration and fused scores (wildlife-tools), scored as
the identity pipeline is scored: ROC AUC over same-day photo pairs (same identity vs not).

    .venv/bin/python scripts/wildfusion_baseline.py --day-dir experiments/days/2024-12-18 --labels model_fine_tuning/splits/labels_all_days.json \
        --out experiments/days/2024-12-18/wildfusion [--current experiments/days/2024-12-18/cluster_id_smoke/scores.npz]

Calibration needs labelled pairs: the identities with >= 2 photos are split in half (A / B, deterministic); the
pipelines are calibrated on A's pairs (A1 vs A2, alternating photos per identity) and the fused score is evaluated
on all pairs within B. The uncalibrated single-pipeline scores need no labels and are also evaluated over all
pairs of the day, which is directly comparable to the current pipeline's "AUC all".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_auc_score


def make_crops(day_dir: Path, out_dir: Path, margin: float = 0.15, min_px: int = 2000) -> list[str]:
    """Whole-whale crops from the day's predicted masks (canon/<stem>_mask.png at the prediction scale)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = []
    for img_path in sorted((day_dir / "images").glob("*.jpg")):
        stem = img_path.stem
        mp = day_dir / "canon" / f"{stem}_mask.png"
        out = out_dir / f"{stem}.jpg"
        if out.exists():
            stems.append(stem); continue
        if not mp.exists():
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED) > 0
        if m.sum() < min_px:
            continue
        img = cv2.imread(str(img_path)); H, W = img.shape[:2]
        sy, sx = H / m.shape[0], W / m.shape[1]
        ys, xs = np.nonzero(m)
        x0, x1, y0, y1 = xs.min() * sx, (xs.max() + 1) * sx, ys.min() * sy, (ys.max() + 1) * sy
        side = max(x1 - x0, y1 - y0) * (1 + 2 * margin); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        xa, ya = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0)); xb, yb = int(min(cx + side / 2, W)), int(min(cy + side / 2, H))
        cv2.imwrite(str(out), img[ya:yb, xa:xb], [cv2.IMWRITE_JPEG_QUALITY, 92])
        stems.append(stem)
    return stems


def pair_auc(score: np.ndarray, labels: np.ndarray) -> float:
    iu = np.triu_indices(len(labels), 1)
    same = (labels[:, None] == labels[None, :])[iu]; s = score[iu]; ok = np.isfinite(s)
    return float(roc_auc_score(same[ok], s[ok])) if ok.sum() > 30 and same[ok].any() and (~same[ok]).any() else float("nan")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--day-dir", type=Path, required=True); p.add_argument("--labels", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--current", type=Path, default=None, help="scores.npz of the current pipeline (cluster/eval_id) for the same day")
    p.add_argument("--device", default="cuda"); p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    from wildlife_tools.data import ImageDataset
    from wildlife_tools.features import AlikedExtractor, DeepFeatures
    from wildlife_tools.similarity import CosineSimilarity, MatchLightGlue
    from wildlife_tools.similarity.calibration import IsotonicCalibration
    from wildlife_tools.similarity.wildfusion import SimilarityPipeline, WildFusion
    import timm

    day = a.day_dir.name
    L = json.loads(a.labels.read_text())["days"][day]["photos"]
    stems = make_crops(a.day_dir, a.out / "crops")
    rows = [{"stem": s, "path": str((a.out / "crops" / f"{s}.jpg").resolve()), "identity": L.get(s.split("__")[-1], "?")} for s in stems]
    df = pd.DataFrame(rows); df = df[df.identity != "?"].reset_index(drop=True)
    counts = df.identity.value_counts(); multi = sorted(counts[counts >= 2].index)
    rng = np.random.default_rng(0); perm = rng.permutation(len(multi)); A = set(np.array(multi)[perm[: len(multi) // 2]]); B = set(multi) - A
    dfA = df[df.identity.isin(A)].reset_index(drop=True); dfB = df[df.identity.isin(B)].reset_index(drop=True)
    dfA["k"] = dfA.groupby("identity").cumcount() % 2
    A1, A2 = dfA[dfA.k == 0].reset_index(drop=True), dfA[dfA.k == 1].reset_index(drop=True)
    print(f"{day}: {len(df)} labelled crops, {len(multi)} identities with >= 2 photos; calibration ids {len(A)} ({len(A1)}+{len(A2)} photos), eval ids {len(B)} ({len(dfB)} photos)", flush=True)
    mega_tf = T.Compose([T.Resize([384, 384]), T.ToTensor(), T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
    local_tf = T.Compose([T.Resize([512, 512]), T.ToTensor()])
    mega = SimilarityPipeline(matcher=CosineSimilarity(), extractor=DeepFeatures(timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True, num_classes=0), device=a.device, batch_size=16, num_workers=a.workers),
                              transform=mega_tf, calibration=IsotonicCalibration())
    local = SimilarityPipeline(matcher=MatchLightGlue(features="aliked", device=a.device), extractor=AlikedExtractor(device=a.device, num_workers=a.workers),
                               transform=local_tf, calibration=IsotonicCalibration())
    results = {"day": day, "crops": len(df), "identities_ge2": len(multi), "calibration_ids": len(A), "eval_ids": len(B), "eval_photos": len(dfB)}
    # uncalibrated scores over all labelled pairs of the day (the single pipelines' raw similarity)
    ds_all = ImageDataset(df, transform=None); labels_all = df.identity.to_numpy()
    raw = {}
    for name, pipe in (("megadescriptor_cosine", mega), ("aliked_lightglue", local)):
        pipe_nocal = SimilarityPipeline(matcher=pipe.matcher, extractor=pipe.extractor, transform=pipe.transform, calibration=None)
        s = np.asarray(pipe_nocal(ds_all, ds_all), float); np.fill_diagonal(s, np.nan)
        raw[name] = s; results[f"auc_all_{name}"] = pair_auc(s, labels_all)
        print(f"  {name:22s} AUC all pairs {results[f'auc_all_{name}']:.3f}", flush=True)
    # calibrated fusion: fit on A, evaluate on B (and the single calibrated pipelines on B)
    dsA1, dsA2 = ImageDataset(A1, transform=None), ImageDataset(A2, transform=None)
    dsB = ImageDataset(dfB, transform=None); labels_B = dfB.identity.to_numpy()
    fusion = WildFusion(calibrated_pipelines=[mega, local], priority_pipeline=mega)
    fusion.fit_calibration(dsA1, dsA2)
    idxB = df.index[df.identity.isin(B)].to_numpy()
    for name, pipe in (("megadescriptor_cosine", mega), ("aliked_lightglue", local)):
        s_raw = raw[name][np.ix_(idxB, idxB)]
        results[f"auc_evalB_{name}"] = pair_auc(s_raw, labels_B)
    fused = np.asarray(fusion(dsB, dsB, B=len(dsB)), float); np.fill_diagonal(fused, np.nan)
    results["auc_evalB_wildfusion"] = pair_auc(fused, labels_B)
    print(f"  eval half ({len(dfB)} photos): megadescriptor {results['auc_evalB_megadescriptor_cosine']:.3f}  lightglue {results['auc_evalB_aliked_lightglue']:.3f}  WildFusion {results['auc_evalB_wildfusion']:.3f}", flush=True)
    # the current pipeline on the same photos, if its score matrix is given
    if a.current and a.current.exists():
        z = np.load(a.current, allow_pickle=True); cstems = [str(s) for s in z["stems"]]
        key = [k for k in z.files if k.endswith("__same") and k.startswith("id__")] or [k for k in z.files if k.endswith("__same")]
        if key:
            same = np.asarray(z[key[0]], float); flip = np.asarray(z[key[0].replace("__same", "__flip")], float) if key[0].replace("__same", "__flip") in z.files else np.full_like(same, np.nan)
            cur = np.where(np.isnan(same), flip, np.where(np.isnan(flip), same, np.maximum(same, flip)))
            pos = {s: i for i, s in enumerate(cstems)}
            for tag, sub in (("all", df), ("evalB", dfB)):
                ii = [pos.get(s) for s in sub.stem]
                keep = [k for k, i in enumerate(ii) if i is not None]
                if len(keep) > 10:
                    m = cur[np.ix_([ii[k] for k in keep], [ii[k] for k in keep])].copy(); np.fill_diagonal(m, np.nan)
                    results[f"auc_{tag}_current_{key[0]}"] = pair_auc(m, sub.identity.to_numpy()[keep])
            print("  current pipeline:", {k: round(v, 3) for k, v in results.items() if "current" in k}, flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "scores.npz", stems=df.stem.to_numpy(), identity=labels_all, **{f"raw_{k}": v for k, v in raw.items()}, fused_evalB=fused, evalB_stems=dfB.stem.to_numpy())
    (a.out / "metrics.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
