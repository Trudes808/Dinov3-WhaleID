"""Why were two photos linked? Heatmaps of the per-cell contributions on both crops, with the
time term, for the strongest links of every cluster; plus a contact sheet per cluster.

    python -m model_fine_tuning.cluster.explain --atlas DIR --scores OUT/scores.npz --clusters OUT/clusters_X.json --images DIR --out OUT/explain_X [--source dino_1]

For a pair, the per-cell z-score (how much more alike the two photos are at that spot than
photo pairs usually are) is painted on both crops: red = supports the link, blue = speaks
against it, grey = shown in only one photo. The header gives the appearance score, the number
of body cells compared, the side hypothesis, the time difference and its term. Links shown per
cluster: the edges of the maximum spanning tree of total scores (the chain that holds the
cluster together), strongest first.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import networkx as nx
import numpy as np

from ..common.data import warp_image, warp_params
from .match import cell_similarity


def load_photo(atlas_dir: Path, images_dir: Path, stem: str) -> dict:
    z = np.load(atlas_dir / f"{stem}.npz")
    x0, y0, s = z["x0y0s"]
    C = z["cell_map"].shape[0]
    img = cv2.cvtColor(cv2.imread(str(images_dir / f"{stem}.jpg"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    crop = warp_image(img, warp_params(x0, y0, s), (C, C))
    return {"crop": crop, "cell_map": z["cell_map"], "side_map": z["side_map"], "cell": z["cell"], "side": z["side"],
            "feat": {k[len("feat_"):]: z[k] for k in z.files if k.startswith("feat_")}, "colour": z["colour_lab"], "count": z["count"]}


def cell_contributions(a: dict, b: dict, source: str, calib: dict, flip_b: bool, midline: np.ndarray) -> dict:
    """cell -> z for the cells both photos show under the hypothesis."""
    if source.startswith("colour"):
        Fa, Fb = a["colour"], b["colour"]
        if source == "colour_rel":
            Fa = Fa - (Fa * a["count"][:, None]).sum(0) / a["count"].sum(); Fb = Fb - (Fb * b["count"][:, None]).sum(0) / b["count"].sum()
    else:
        Fa, Fb = a["feat"][source], b["feat"][source]
    if not source.startswith("colour"):
        Fa = Fa / np.maximum(np.linalg.norm(Fa, axis=1, keepdims=True), 1e-6); Fb = Fb / np.maximum(np.linalg.norm(Fb, axis=1, keepdims=True), 1e-6)
    sb = -b["side"] if flip_b else b["side"]
    ia = {(int(c), int(s)): i for i, (c, s) in enumerate(zip(a["cell"], a["side"]))}
    out = {}
    for j, (c, s) in enumerate(zip(b["cell"], sb)):
        c = int(c); s = int(s)
        cands = [(c, s)] if not midline[c] else [(c, 0), (c, 1), (c, -1)]
        if s == 0 and not midline[c]:
            cands = [(c, 0), (c, 1), (c, -1)]
        for cand in cands:
            if cand in ia and str(c) in calib:
                i = ia[cand]
                sim = float(Fa[i] @ Fb[j]) if not source.startswith("colour") else -float(np.linalg.norm(Fa[i] - Fb[j]))
                mu, sd = calib[str(c)]
                out[c] = (sim - mu) / sd
                break
    return out


def paint_patches(crop: np.ndarray, local: dict, vals: np.ndarray, patch: int, vmax: float = 0.15) -> np.ndarray:
    """Patch-level heatmap for the local matcher: red = this spot looks more like the same spot
    on the other photo than like other spots (supports the link), blue = less."""
    out = crop.astype(np.float32).copy()
    for (gx, gy), v in zip(local["patch_xy"], vals):
        sl = (slice(gy * patch, (gy + 1) * patch), slice(gx * patch, (gx + 1) * patch))
        if np.isnan(v):
            out[sl] = 0.6 * out[sl] + 0.4 * np.array([128, 128, 128]); continue
        t = float(np.clip(v / vmax, -1, 1))
        colour = np.array([255, 40, 40]) * t + np.array([255, 255, 255]) * (1 - t) if t >= 0 else np.array([40, 60, 255]) * (-t) + np.array([255, 255, 255]) * (1 + t)
        out[sl] = 0.35 * out[sl] + 0.65 * colour
    return out.astype(np.uint8)


def paint(photo: dict, contrib: dict, vmax: float = 3.0) -> np.ndarray:
    crop = photo["crop"].astype(np.float32)
    cm = photo["cell_map"]
    heat = np.full(cm.shape, np.nan, np.float32)
    for c, zv in contrib.items():
        heat[cm == c] = zv
    out = crop.copy()
    shown = cm >= 0
    only = shown & np.isnan(heat)
    out[only] = 0.6 * crop[only] + 0.4 * np.array([128, 128, 128])
    both = shown & ~np.isnan(heat)
    v = np.clip(heat[both] / vmax, -1, 1)
    colour = np.where(v[:, None] >= 0, np.array([255, 40, 40]) * v[:, None] + np.array([255, 255, 255]) * (1 - v[:, None]),
                      np.array([40, 60, 255]) * (-v[:, None]) + np.array([255, 255, 255]) * (1 + v[:, None]))
    out[both] = 0.35 * crop[both] + 0.65 * colour
    return out.astype(np.uint8)


def panel(a: dict, b: dict, contrib: dict, header: str) -> np.ndarray:
    pa, pb = paint(a, contrib), paint(b, contrib)
    strip = np.zeros((28, pa.shape[1] + pb.shape[1], 3), np.uint8)
    cv2.putText(strip, header, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([strip, np.concatenate([pa, pb], 1)], 0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atlas", type=Path, required=True)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--clusters", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--timestamps", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source", default="dino_1")
    p.add_argument("--local", type=Path, default=None, help="local_up2 dir: with --source local_dino, paint per-patch contributions")
    p.add_argument("--max-links", type=int, default=6)
    p.add_argument("--time-weight", type=float, default=1.0)
    args = p.parse_args(argv)
    z = dict(np.load(args.scores)); stems = [str(s) for s in z["stems"]]
    meta = json.loads((args.scores.parent / "match_meta.json").read_text()); calib = meta["calib"].get(args.source, {})
    local_mode = args.source == "local_dino"
    if local_mode:
        from .local import _load, patch_contributions
        local_data = _load(args.local, stems)
        local_raw = {s: np.load(args.local / f"{s}.npz") for s in stems}
    cl = json.loads(args.clusters.read_text()); ts = json.loads(args.timestamps.read_text())
    from .cells import cells_path, load_cells
    cells = load_cells(cells_path(512)); midline = cells["midline"]
    same, flip = z[f"{args.source}__same"], z[f"{args.source}__flip"]
    app = np.where(np.isnan(same), flip, np.where(np.isnan(flip), same, np.maximum(same, flip)))
    total = np.where(np.isnan(app), 0.0, app) + args.time_weight * z["time_term"]
    idx = {s: i for i, s in enumerate(stems)}
    groups = defaultdict(list)
    for s, c, l in zip(cl["stems"], cl["cluster"], cl["label"]):
        groups[c].append((s, l))
    args.out.mkdir(parents=True, exist_ok=True)
    cache = {}
    def photo(s):
        if s not in cache: cache[s] = load_photo(args.atlas, args.images, s)
        return cache[s]
    summary = []
    for c, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        names = [m[0] for m in members]
        g = nx.Graph()
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                g.add_edge(a, b, weight=float(total[idx[a], idx[b]]))
        edges = sorted(nx.maximum_spanning_tree(g).edges(data=True), key=lambda e: -e[2]["weight"])[: args.max_links] if len(names) > 1 else []
        cdir = args.out / f"cluster_{c:03d}_n{len(names)}"; cdir.mkdir(exist_ok=True)
        links = []
        for k, (a, b, d) in enumerate(edges):
            i, j = idx[a], idx[b]
            use_flip = np.nan_to_num(flip[i, j], nan=-np.inf) > np.nan_to_num(same[i, j], nan=-np.inf)
            dt = z["dt_s"][i, j]; tt = z["time_term"][i, j]
            if local_mode:
                va, vb = patch_contributions(local_data[a], local_data[b], bool(use_flip))
                n_matched = int(np.isfinite(va).sum() + np.isfinite(vb).sum())
                header = (f"local aligned-minus-misaligned z={app[i, j]:+.2f} over {n_matched} patches ({'B mirrored' if use_flip else 'sides as resolved'})  "
                          f"dt={dt:.1f}s time term={tt:+.1f}  total={total[i, j]:+.2f}   labels {dict(members)[a]} | {dict(members)[b]}")
                S = int(local_raw[a]["size"]); patch = 16
                crops = []
                for s_, vals in ((a, va), (b, vb)):
                    zz = local_raw[s_]; x0, y0, sc = zz["x0y0s"]
                    img = cv2.cvtColor(cv2.imread(str(args.images / f"{s_}.jpg"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    crop = warp_image(img, warp_params(x0, y0, sc), (S, S))
                    crops.append(cv2.resize(paint_patches(crop, zz, vals, patch), (512, 512)))
                strip = np.zeros((28, 1024, 3), np.uint8); cv2.putText(strip, header, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                img_panel = np.concatenate([strip, np.concatenate(crops, 1)], 0)
                contrib = {}
            else:
                contrib = cell_contributions(photo(a), photo(b), args.source, calib, use_flip, midline)
                header = (f"appearance z={app[i, j]:+.2f} over {len(contrib)} cells ({'B mirrored' if use_flip else 'sides as resolved'})  "
                          f"dt={dt:.1f}s time term={tt:+.1f}  total={total[i, j]:+.2f}   labels {dict(members)[a]} | {dict(members)[b]}")
                img_panel = panel(photo(a), photo(b), contrib, header)
            cv2.imwrite(str(cdir / f"link_{k:02d}.jpg"), cv2.cvtColor(img_panel, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            links.append({"a": a, "b": b, "appearance_z": float(app[i, j]) if np.isfinite(app[i, j]) else None, "cells": len(contrib), "flip": bool(use_flip), "dt_s": float(dt), "time_term": float(tt), "total": float(total[i, j]),
                          "labels": [dict(members)[a], dict(members)[b]]})
        # contact sheet: all members, time order, with labels
        order = sorted(names, key=lambda s: ts[s]["time"])
        tiles = []
        for s in order:
            t = cv2.resize(photo(s)["crop"], (192, 192))
            cv2.putText(t, dict(members)[s], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(t, ts[s]["datetime"][-8:], (4, 186), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(t)
        cols = 8
        rows = [np.concatenate(tiles[r: r + cols] + [np.zeros((192, 192, 3), np.uint8)] * (cols - len(tiles[r: r + cols])), 1) for r in range(0, len(tiles), cols)]
        cv2.imwrite(str(cdir / "members.jpg"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        label_counts = {l: sum(1 for m in members if m[1] == l) for l in set(m[1] for m in members)}
        summary.append({"cluster": int(c), "photos": len(names), "labels": dict(sorted(label_counts.items(), key=lambda kv: -kv[1])), "links": links})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("wrote", args.out, f"({len(summary)} clusters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
