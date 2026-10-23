"""Per-pixel template-vertex labels for a factory run: vertex/<id>.png (uint16; 0 = no label, else the template
vertex index + 1), the nearest vertex of assets/humpback4k_template.npz to the canonical surface point of every
labelled pixel (canon/<id>.npz). Also adds the template's mirror table (the vertex each vertex maps to under a
left/right flip) if it is missing.

    python -m model_fine_tuning.common.vertex_labels RUN_DIR [--workers 12]
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from . import paths

TEMPLATE = paths.ASSETS_DIR / "humpback4k_template.npz"
CANON_OFFSET = 32768
_TREE = None


def ensure_mirror_table(path: Path = TEMPLATE) -> None:
    t = dict(np.load(path, allow_pickle=False))
    if "mirror" in t:
        return
    P = t["positions_cm"].astype(np.float64)
    t["mirror"] = cKDTree(P).query(P * np.array([1.0, -1.0, 1.0]))[1].astype(np.int32)
    np.savez_compressed(path, **t)


def _init():
    global _TREE
    _TREE = cKDTree(np.load(TEMPLATE)["positions_cm"].astype(np.float64))


def _job(args):
    canon_path, out_path = args
    if Path(out_path).exists():
        return 0
    with np.load(canon_path) as z:
        v = z["xyz_mm"]
    defined = v[..., 0] > 0
    out = np.zeros(v.shape[:2], np.uint16)
    if defined.any():
        cm = (v[defined].astype(np.float32) - CANON_OFFSET) / 10.0
        out[defined] = _TREE.query(cm)[1].astype(np.uint16) + 1
    Image.fromarray(out).save(out_path)
    return 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path); p.add_argument("--workers", type=int, default=12)
    a = p.parse_args(argv)
    ensure_mirror_table()
    canon_dir = a.run_dir / "dataset" / "canon"; out_dir = a.run_dir / "dataset" / "vertex"; out_dir.mkdir(exist_ok=True)
    jobs = [(str(f), str(out_dir / f"{f.stem}.png")) for f in sorted(canon_dir.glob("*.npz"))]
    done = 0
    with Pool(a.workers, initializer=_init) as pool:
        for k in pool.imap_unordered(_job, jobs, chunksize=16):
            done += k
    print(f"vertex labels: {done} written, {len(jobs)} shots -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
