"""Grid of mesh fits: real photos beside synthetic shots of a similar pose class (fluke / back / head, from the body
fraction of the predicted whale pixels), each with the rigid PnP (orange), bend (green) and silhouette-refined
(white) above-water mesh outlines from fit_mesh.

    python -m model_fine_tuning.canon.fit_grid --real experiments/real_2024-12-18/fit_canon_r04_ft4 \
        --synthetic experiments/canon_r04_ft4/fit_synthetic --out experiments/real_2024-12-18/fit_grid_canon_r04_ft4.jpg --per-class 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

TILE = 400
CLASSES = ("back", "fluke", "head")


def tiles_for(fit_dir: Path, cls: str, n: int, synthetic: bool) -> list[np.ndarray]:
    d = json.loads((fit_dir / "fit.json").read_text())["photos"]
    out = []
    for key, r in d.items():
        if not r.get("ok") or r.get("pose_class") != cls:
            continue
        f = fit_dir / "overlays" / (f"{r['category']}_{key}.jpg" if synthetic else f"{key}.jpg")
        if not f.exists():
            continue
        img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        out.append(cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_AREA))
        if len(out) >= n:
            break
    while len(out) < n:
        out.append(np.full((TILE, TILE, 3), 20, np.uint8))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", type=Path, required=True); p.add_argument("--synthetic", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--per-class", type=int, default=4)
    a = p.parse_args(argv)
    rows = []
    for cls in CLASSES:
        left = tiles_for(a.real, cls, a.per_class, False); right = tiles_for(a.synthetic, cls, a.per_class, True)
        gap = np.full((TILE, 24, 3), 255, np.uint8)
        row = np.concatenate(left + [gap] + right, 1)
        band = np.full((28, row.shape[1], 3), 255, np.uint8)
        cv2.putText(band, f"{cls}: real photos (left, model mask tinted)  |  synthetic test shots (right, true mask tinted, model mask in cyan)   "
                          f"orange = rigid PnP, green = bend fit, white = silhouette-refined bend", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        rows.append(np.concatenate([band, row], 0))
    sheet = np.concatenate(rows, 0)
    cv2.imwrite(str(a.out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", a.out, sheet.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
