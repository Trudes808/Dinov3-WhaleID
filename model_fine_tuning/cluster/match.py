"""Pairwise photo similarity from the canonical atlases: compare what the two photos show at
the same body locations, calibrated per location, under both side hypotheses, plus the capture
time difference.

    python -m model_fine_tuning.cluster.match --atlas DIR --timestamps timestamps.json --out OUT

Per feature source (e.g. dino_1 = frozen DINOv3 last block, canon_1 = fine-tuned backbone,
colour), and per surface cell, the cosine similarity between every pair of photos that show
that cell is z-scored against all pairs showing it ("how alike is alike for this spot"). A
pair's appearance score is the pixel-weighted mean z over the cells both photos show; cells on
the midline are shared, others must be on the same flank. Two hypotheses are scored: the two
photos' resolved sides as they are ("same"), and photo B mirrored ("flip"); the matcher may use
the better one, because the per-whale side decision is the least reliable step. The time term
is a step function of the capture-time difference (bursts count for a lot, hours for little).
Writes score matrices (npz) and the per-cell calibration needed to explain a link.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TIME_STEPS = [(2.0, 1.5), (30.0, 1.0), (300.0, 0.5), (3600.0, 0.0), (np.inf, -0.5)]      # (max seconds, term)


def time_term(dt_s: np.ndarray) -> np.ndarray:
    out = np.zeros_like(dt_s, dtype=np.float64)
    prev = -1.0
    for hi, val in TIME_STEPS:
        out[(dt_s > prev) & (dt_s <= hi)] = val
        prev = hi
    return out


def load_atlas(atlas_dir: Path) -> tuple[list[str], list[dict]]:
    stems, recs = [], []
    for f in sorted(atlas_dir.glob("*.npz")):
        z = np.load(f)
        recs.append({k: z[k] for k in z.files if k not in ("cell_map", "side_map")})
        stems.append(f.stem)
    return stems, recs


def feature_sources(rec: dict) -> list[str]:
    return sorted(k[len("feat_"):] for k in rec if k.startswith("feat_")) + ["colour", "colour_rel"]


def merge_cells(recs: list[dict], centres: np.ndarray, k2: int, seed: int = 0) -> list[dict]:
    """Re-express every atlas on a coarser grid: k-means of the cell centres into k2 groups,
    per-(group, side) pixel-weighted means of the features and colours."""
    from scipy.cluster.vq import kmeans2

    _, group = kmeans2(np.asarray(centres, np.float64), k2, minit="++", seed=np.random.default_rng(seed), iter=30)
    out = []
    for r in recs:
        g = group[r["cell"]]
        key = g.astype(np.int64) * 3 + (r["side"].astype(np.int64) + 1)
        uniq, inv = np.unique(key, return_inverse=True)
        w = r["count"].astype(np.float64)
        wsum = np.zeros(len(uniq)); np.add.at(wsum, inv, w)
        nr = {"cell": (uniq // 3).astype(np.int32), "side": (uniq % 3 - 1).astype(np.int8), "count": wsum.astype(np.int32)}
        for k, v in r.items():
            if k.startswith("feat_") or k in ("colour_lab", "disagree_m"):
                v2 = v if v.ndim == 2 else v[:, None]
                acc = np.zeros((len(uniq), v2.shape[1])); np.add.at(acc, inv, v2 * w[:, None])
                nr[k] = (acc / wsum[:, None]).astype(np.float32) if v.ndim == 2 else (acc[:, 0] / wsum).astype(np.float32)
        out.append(nr)
    return out


def relative_colour(recs: list[dict]) -> None:
    """Per photo, subtract the whale's mean LAB colour (pixel-weighted) so that lighting and
    exposure cancel and only the pigmentation pattern remains."""
    for r in recs:
        w = r["count"].astype(np.float64)[:, None]
        mean = (r["colour_lab"] * w).sum(0) / w.sum()
        r["colour_rel"] = (r["colour_lab"] - mean).astype(np.float32)


def cell_tables(recs: list[dict], source: str) -> dict:
    """cell -> (photo indices, sides, unit features) for one source."""
    tables = {}
    key = {"colour": "colour_lab", "colour_rel": "colour_rel"}.get(source, f"feat_{source}")
    for i, r in enumerate(recs):
        F = r[key].astype(np.float32)
        if not source.startswith("colour"):
            F = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-6)
        for j, c in enumerate(r["cell"]):
            tables.setdefault(int(c), ([], [], [], []))
            t = tables[int(c)]
            t[0].append(i); t[1].append(int(r["side"][j])); t[2].append(F[j]); t[3].append(float(r["count"][j]))
    return {c: (np.array(t[0]), np.array(t[1], np.int8), np.stack(t[2]), np.array(t[3])) for c, t in tables.items()}


def cell_similarity(F: np.ndarray, source: str) -> np.ndarray:
    if source.startswith("colour"):
        d = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=-1)
        return -d
    return F @ F.T


def score_source(recs: list[dict], source: str, n: int, min_cells: int = 8) -> dict:
    """Appearance z-score matrices under both hypotheses, the number of compared cells, and the
    per-cell calibration (mean, std of the similarity over all pairs showing the cell)."""
    tables = cell_tables(recs, source)
    S = {h: np.zeros((n, n)) for h in ("same", "flip")}
    W = {h: np.zeros((n, n)) for h in ("same", "flip")}
    N = {h: np.zeros((n, n), np.int32) for h in ("same", "flip")}
    calib = {}
    for c, (idx, side, F, cnt) in tables.items():
        if len(idx) < 3:
            continue
        sim = cell_similarity(F, source)
        off = ~np.eye(len(idx), dtype=bool)
        mu, sd = sim[off].mean(), sim[off].std() + 1e-6
        calib[c] = (float(mu), float(sd))
        z = (sim - mu) / sd
        w = np.sqrt(np.minimum(cnt[:, None], cnt[None, :]))
        same_side = (side[:, None] == side[None, :]) | (side[:, None] == 0) | (side[None, :] == 0)
        flip_side = (side[:, None] == -side[None, :]) | (side[:, None] == 0) | (side[None, :] == 0)
        for h, ok in (("same", same_side), ("flip", flip_side)):
            m = ok & off
            ii, jj = np.nonzero(m)
            np.add.at(S[h], (idx[ii], idx[jj]), z[ii, jj] * w[ii, jj])
            np.add.at(W[h], (idx[ii], idx[jj]), w[ii, jj])
            np.add.at(N[h], (idx[ii], idx[jj]), 1)
    out = {}
    for h in ("same", "flip"):
        score = np.where(W[h] > 0, S[h] / np.maximum(W[h], 1e-9), np.nan)
        score[N[h] < min_cells] = np.nan
        out[f"{h}"] = score
        out[f"n_{h}"] = N[h]
    out["calib"] = calib
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atlas", type=Path, required=True)
    p.add_argument("--timestamps", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-cells", type=int, default=8)
    p.add_argument("--merge-to", type=int, default=None, help="re-express the atlases on a coarser grid of this many cells")
    args = p.parse_args(argv)
    stems, recs = load_atlas(args.atlas)
    if args.merge_to:
        from .cells import cells_path, load_cells

        meta = json.loads((args.atlas / "meta.json").read_text())
        recs = merge_cells(recs, load_cells(cells_path(meta["k"]))["centres_m"], args.merge_to)
    relative_colour(recs)
    n = len(stems)
    ts = json.loads(args.timestamps.read_text())
    t = np.array([ts[s]["time"] for s in stems], float)
    cam = np.array([ts[s]["camera"] for s in stems])
    dt = np.abs(t[:, None] - t[None, :])
    args.out.mkdir(parents=True, exist_ok=True)
    matrices = {"dt_s": dt, "time_term": time_term(dt), "same_camera": cam[:, None] == cam[None, :]}
    calib_all = {}
    for source in feature_sources(recs[0]):
        res = score_source(recs, source, n, args.min_cells)
        for h in ("same", "flip"):
            matrices[f"{source}__{h}"] = res[h]
            matrices[f"{source}__n_{h}"] = res[f"n_{h}"]
        calib_all[source] = res["calib"]
        s = res["same"]; f = res["flip"]
        print(f"{source:10s} pairs with a score: {np.isfinite(s).sum() // 2:6d} (same-side) {np.isfinite(f).sum() // 2:6d} (flipped); "
              f"median cells compared {np.median(res['n_same'][res['n_same'] > 0]):.0f}", flush=True)
    np.savez_compressed(args.out / "scores.npz", stems=np.array(stems), **matrices)
    json.dump({"sources": feature_sources(recs[0]), "calib": {s: {str(c): v for c, v in d.items()} for s, d in calib_all.items()},
               "time_steps": [(None if np.isinf(h) else h, v) for h, v in TIME_STEPS], "min_cells": args.min_cells, "merge_to": args.merge_to}, open(args.out / "match_meta.json", "w"))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
