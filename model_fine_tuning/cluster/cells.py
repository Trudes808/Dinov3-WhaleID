"""Canonical surface cells: the whale surface (folded mesh vertices) partitioned into K patches
by k-means, so that a photo can be summarised as "which cells are visible and what they look
like". Cells whose centre lies within `midline_m` of the sagittal plane are shared by both
sides (their side id is 0); all others carry the side (+1 right, -1 left) of the pixels that
map to them.

    python -m model_fine_tuning.cluster.cells --k 512      # writes assets/canonical_cells_512.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..common import geometry, paths


def build_cells(table: dict, k: int = 512, seed: int = 0, midline_m: float = 0.12) -> dict:
    from scipy.cluster.vq import kmeans2

    pts = np.asarray(table["folded_m"], np.float64)
    rng = np.random.default_rng(seed)
    centres, labels = kmeans2(pts, k, minit="++", seed=rng, iter=30)
    counts = np.bincount(labels, minlength=k)
    keep = counts > 0
    centres = centres[keep]
    region = geometry.mesh_region(centres, table)
    return {"centres_m": centres.astype(np.float32), "region": region.astype(np.int8), "midline": (centres[:, 1] < midline_m),
            "k": int(len(centres)), "length_m": table["length_m"], "x_min_m": table["x_min_m"]}


def save_cells(cells: dict, path: Path) -> None:
    np.savez_compressed(path, **cells)


def load_cells(path: Path) -> dict:
    from scipy.spatial import cKDTree

    with np.load(path) as data:
        cells = {k: (data[k].item() if data[k].ndim == 0 else data[k]) for k in data.files}
    cells["tree"] = cKDTree(cells["centres_m"])
    return cells


def assign_cells(folded_m: np.ndarray, cells: dict) -> np.ndarray:
    """(..., 3) folded points (metres) -> cell index (...)."""
    _, idx = cells["tree"].query(np.asarray(folded_m, float).reshape(-1, 3), k=1)
    return idx.reshape(np.shape(folded_m)[:-1])


def cells_path(k: int) -> Path:
    return paths.ASSETS_DIR / f"canonical_cells_{k}.npz"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    table = geometry.load_folded_table(paths.FOLDED_TABLE)
    cells = build_cells(table, args.k, args.seed)
    save_cells(cells, cells_path(args.k))
    print(f"{cells['k']} cells ({int(cells['midline'].sum())} on the midline; torso {int((cells['region']==0).sum())}, flukes {int((cells['region']==1).sum())}, fins {int((cells['region']==2).sum())}) -> {cells_path(args.k)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
