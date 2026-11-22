"""Rule A4 helper: share of the frame our live mask (seg_r04_ft4 canon mask) paints outside S6h (the shadow root's canon
mask, S6h + rule F), per training-day photo; days missing from the shadow root are listed as unknown (-1).

    .venv/bin/python experiments/sam3_labeller/live_minus_s6.py   -> OUT/lists/live_minus_s6.json
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lab_common as L  # noqa: E402

SHADOW = L.NW / "moonshot_cache" / "switch_root" / "days"


def one(args):
    day, stem = args
    a = cv2.imread(str(L.DAYS / day / "canon" / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(SHADOW / day / "canon" / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        return day, stem, -1.0
    a = cv2.resize(a, (640, 427), interpolation=cv2.INTER_NEAREST) > 0; b = cv2.resize(b, (640, 427), interpolation=cv2.INTER_NEAREST) > 0
    return day, stem, float((a & ~b).mean())


def main() -> None:
    todo = [(d, s) for d in L.TRAIN for s in L.day_stems(d)]
    with Pool(12) as P:
        res = P.map(one, todo, chunksize=32)
    out = {f"{d}|{s}": v for d, s, v in res}
    (L.OUT / "lists" / "live_minus_s6.json").write_text(json.dumps(out))
    v = np.array(list(out.values()))
    print(len(v), "photos; unknown", int((v < 0).sum()), "; >= 0.5 %", int((v >= 0.005).sum()), "; >= 2 %", int((v >= 0.02).sum()))


if __name__ == "__main__":
    main()
