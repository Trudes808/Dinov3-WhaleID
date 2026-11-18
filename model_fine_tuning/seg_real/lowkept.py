"""The 20 view-labelled main whales (train + val days) where S4 or S5 keeps the least of the r04 blob; panels
r04 | S1 | S4 | S5 (whale = red fill, the r04 blob outlined yellow), caption = kept share per model.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.lowkept --names r04,s1,s4,s5 --pick s4,s5 -> SHEETS/lowkept_s4_s5.jpg
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from . import common as C
from .data import blob_1280
from .predict import load_prob, resize_1280


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--names", default="r04,s1,s4,s5"); ap.add_argument("--pick", default="s4,s5"); ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args(argv); names = a.names.split(","); pick = a.pick.split(",")
    PP = json.loads((C.CACHE / "per_photo_whales.json").read_text())
    kept = {n: {(r["day"], r["stem"], r["b"]): r for r in PP[n]} for n in set(names + pick) if n in PP}
    keys = sorted(kept[pick[0]], key=lambda k: min(kept[p][k]["kept"] for p in pick))[:a.n]
    rows = {}; tiles = []; legend = []
    for t, (day, stem, b) in enumerate(keys):
        if day not in rows:
            rows[day] = C.blob_rows(day)
        img = resize_1280(C.photo(day, stem)); H, W = img.shape[:2]; old = blob_1280(C.blob_image(day, stem), b, (H, W))
        ys, xs = np.nonzero(old); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max(); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w = max((x1 - x0) * 1.6, 160); h = max((y1 - y0) * 1.6, 107); w, h = (h * 1.5, h) if w / h < 1.5 else (w, w / 1.5); w, h = min(w, W), min(h, H)
        wx = int(min(max(cx - w / 2, 0), W - w)); wy = int(min(max(cy - h / 2, 0), H - h)); sl = (slice(wy, int(wy + h)), slice(wx, int(wx + w)))
        cs, _ = cv2.findContours(old[sl].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); panels = []; cap = {}
        for n in names:
            v = img[sl].copy(); m = load_prob(n, day, stem)[sl] >= 0.5
            v[m] = (0.55 * v[m] + 0.45 * np.array([0, 0, 255])).astype(np.uint8); cv2.drawContours(v, cs, -1, (0, 255, 255), 1)
            v = cv2.resize(v, (240, 160), interpolation=cv2.INTER_AREA if v.shape[1] > 240 else cv2.INTER_CUBIC)
            k = kept[n][(day, stem, b)]["kept"] if n in kept and (day, stem, b) in kept[n] else 1.0; cap[n] = k
            txt = f"{t + 1} {n} {k:.2f}"
            cv2.putText(v, txt, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA); cv2.putText(v, txt, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(v)
        tiles.append(cv2.copyMakeBorder(np.hstack(panels), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        legend.append({"tile": t + 1, "day": day, "stem": stem, "b": b, "kept": cap})
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    p = C.SHEETS / f"lowkept_{'_'.join(pick)}.jpg"
    cv2.imwrite(str(p), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 87])
    (C.SHEETS / f"lowkept_{'_'.join(pick)}_legend.json").write_text(json.dumps(legend, indent=0)); print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
