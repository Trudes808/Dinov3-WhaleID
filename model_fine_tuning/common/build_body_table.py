"""Build the (u, theta) -> 3D lookup table from the humpback mesh and copy the landmarks.

    python -m model_fine_tuning.common.build_body_table [--mesh OBJ] [--run-dir RUN]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from . import geometry, paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=paths.WHALE_MESH_OBJ)
    parser.add_argument("--run-dir", type=Path, default=paths.WHALE_RUN_DIR)
    parser.add_argument("--out", type=Path, default=paths.BODY_TABLE)
    args = parser.parse_args(argv)
    verts = geometry.load_obj_vertices(args.mesh)
    table = geometry.build_body_table(verts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    geometry.save_body_table(table, args.out)
    print(f"table of {len(table['u'])} vertices, length {table['length_m']:.2f} m -> {args.out}")
    lm = args.run_dir / "dataset" / "landmarks_mesh_frame_cm.json"
    if lm.exists():
        shutil.copyfile(lm, args.out.parent / "landmarks_mesh_frame_cm.json")
        print("copied", lm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
