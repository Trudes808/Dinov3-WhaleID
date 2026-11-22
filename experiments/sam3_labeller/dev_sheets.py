"""Plain thumbnails (no model overlay) of the dev pool photos for the by-eye tags of rule A2.2, 20 per sheet, numbered.
Sheets go to the scratch folder only (they show crew).

    .venv/bin/python experiments/sam3_labeller/dev_sheets.py   -> SCRATCH/dev_tag/tag_NN.jpg, OUT/dev/tag_order.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lab_common as L  # noqa: E402

TW, TH, COLS, ROWS = 384, 256, 5, 4


def main() -> None:
    pools = json.loads((L.OUT / "dev" / "pools.json").read_text()); rows = json.loads((L.OUT / "dev" / "shares.json").read_text())
    ids = sorted({k for c in pools.values() for k in c["in"] + c["out"]})
    random.Random(7).shuffle(ids)                     # order not by class, so the tag is blind to the pool
    out = L.SCRATCH / "dev_tag"; out.mkdir(parents=True, exist_ok=True)
    (L.OUT / "dev" / "tag_order.json").write_text(json.dumps(ids, indent=0))
    per = COLS * ROWS
    for s in range(0, len(ids), per):
        sheet = np.full((ROWS * (TH + 4), COLS * (TW + 4), 3), 255, np.uint8)
        for n, k in enumerate(ids[s:s + per]):
            im = cv2.resize(cv2.imread(rows[k]["path"]), (TW, TH), interpolation=cv2.INTER_AREA)
            cv2.putText(im, str(s + n), (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4); cv2.putText(im, str(s + n), (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            r, c = divmod(n, COLS); sheet[r * (TH + 4):r * (TH + 4) + TH, c * (TW + 4):c * (TW + 4) + TW] = im
        cv2.imwrite(str(out / f"tag_{s // per:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(len(ids), "photos,", (len(ids) + per - 1) // per, "sheets")


if __name__ == "__main__":
    main()
