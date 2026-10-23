"""Part-aligned identity scoring with the WildFusion matchers on the pose-normalised part charts (cluster/parts.py):
MegaDescriptor cosine and ALIKED + LightGlue per chart type (body, fin, fluke), pair AUC per part over the photo
pairs that both have the part, and a photo-pair score = best over the shared parts of the per-part z-scored score,
scored over all labelled pairs with at least one shared part. Run it on the charts built from the old map and on
those built from the CSE map to compare the maps with the same matchers.

    .venv/bin/python scripts/wildfusion_parts.py --parts experiments/days/2024-12-18/parts_cse --labels model_fine_tuning/splits/labels_all_days.json \
        --out experiments/days/2024-12-18/wildfusion_parts_cse
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score

PARTS = ("body", "fin", "fluke")


def pair_auc(score: np.ndarray, labels: np.ndarray, valid: np.ndarray | None = None) -> tuple[float, int]:
    iu = np.triu_indices(len(labels), 1)
    same = (labels[:, None] == labels[None, :])[iu]; s = score[iu]; ok = np.isfinite(s)
    if valid is not None:
        ok &= valid[iu]
    if ok.sum() < 30 or not same[ok].any() or not (~same[ok]).any():
        return float("nan"), int(ok.sum())
    return float(roc_auc_score(same[ok], s[ok])), int(ok.sum())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parts", type=Path, required=True); p.add_argument("--labels", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda"); p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    from wildlife_tools.data import ImageDataset
    from wildlife_tools.features import AlikedExtractor, DeepFeatures
    from wildlife_tools.similarity import CosineSimilarity, MatchLightGlue
    from wildlife_tools.similarity.wildfusion import SimilarityPipeline
    import timm

    day = a.parts.parent.name
    L = json.loads(a.labels.read_text())["days"][day]["photos"]
    charts = sorted((a.parts / "charts").glob("*.jpg"))
    rows = []
    for f in charts:
        stem, part = f.stem.rsplit("__", 1)
        if part not in PARTS:
            continue
        ident = L.get(stem.split("__")[-1], "?")
        if ident != "?":
            rows.append({"stem": stem, "part": part, "path": str(f.resolve()), "identity": ident})
    df = pd.DataFrame(rows)
    stems = sorted(df.stem.unique()); pos = {s: i for i, s in enumerate(stems)}; n = len(stems)
    labels = np.array([L[s.split("__")[-1]] for s in stems])
    print(f"{day}: {n} labelled photos with charts; per part: {df.part.value_counts().to_dict()}", flush=True)
    mega = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True, num_classes=0)
    pipes = {"megadescriptor": SimilarityPipeline(matcher=CosineSimilarity(), extractor=DeepFeatures(mega, device=a.device, batch_size=16, num_workers=a.workers),
                                                  transform=T.Compose([T.Resize([384, 384]), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)]), calibration=None),
             "lightglue": SimilarityPipeline(matcher=MatchLightGlue(features="aliked", device=a.device), extractor=AlikedExtractor(device=a.device, num_workers=a.workers),
                                             transform=T.Compose([T.Resize([512, 512]), T.ToTensor()]), calibration=None)}
    results = {"day": day, "photos": n, "charts": {k: int(v) for k, v in df.part.value_counts().items()}}
    full = {name: np.full((n, n), np.nan) for name in pipes}
    for part in PARTS:
        sub = df[df.part == part].reset_index(drop=True)
        if len(sub) < 10:
            continue
        ds = ImageDataset(sub, transform=None); idx = np.array([pos[s] for s in sub.stem])
        for name, pipe in pipes.items():
            s = np.asarray(pipe(ds, ds), float); np.fill_diagonal(s, np.nan)
            auc, pairs = pair_auc(s, sub.identity.to_numpy())
            results[f"auc_{part}_{name}"] = auc; results[f"pairs_{part}"] = pairs
            print(f"  {part:6s} {name:14s} AUC {auc:.3f} over {pairs} pairs", flush=True)
            z = (s - np.nanmean(s)) / (np.nanstd(s) + 1e-9)                   # z-score per part and matcher, then best over shared parts
            cur = full[name][np.ix_(idx, idx)]
            full[name][np.ix_(idx, idx)] = np.where(np.isnan(cur), z, np.fmax(cur, z))
    for name in pipes:
        auc, pairs = pair_auc(full[name], labels)
        results[f"auc_best_shared_part_{name}"] = auc; results["pairs_best_shared_part"] = pairs
        print(f"  best over shared parts, {name:14s} AUC {auc:.3f} over {pairs} pairs", flush=True)
    both = np.fmax(full["megadescriptor"], full["lightglue"])
    results["auc_best_shared_part_both"], _ = pair_auc(both, labels)
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "scores.npz", stems=np.array(stems), identity=labels, **{f"best_{k}": v for k, v in full.items()})
    (a.out / "metrics.json").write_text(json.dumps(results, indent=1)); print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
