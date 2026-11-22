"""Whale pixel scores on the frozen pixel truth with a band of r px (truth grid) around every truth boundary left out:
how much of the SAM 3 / S6h gap is outline placement (the truth was drawn with SAM 2.1 prompts) and how much is
whole regions. Also the gap split into pixels near (<= 10 px) and far from a truth boundary.

    PYTHONPATH=. .venv/bin/python experiments/sam3_compare/whale_band.py --settings SETTINGS.json --out OUT.json
"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import s3common as C
from analyze_test import check_frozen, PT, TH, TW

ap = argparse.ArgumentParser(); ap.add_argument("--settings", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args(); S = json.loads(a.settings.read_text()); man = check_frozen(); idx = C.prompt_index("test")
wj = [idx[p] for p in S["class_prompts"]["whale"]]; thr = S["class_thr"]["whale"]
R = [0, 2, 5, 10, 20]; acc = {m: {r: [0, 0, 0] for r in R} for m in ("live", "s6h", "sam3")}
for seq in man["sample"]:
    t = np.array(Image.open(PT / "labels" / f"{seq}.png")); valid = t != 255; tm = (t == 1)
    hw = (TH, TW); rec = C.load("test", seq)
    maps = {"live": C.test_live(seq, hw), "s6h": C.test_s6(seq, hw), "sam3": C.union(rec, wj, thr, hw)}
    for r in R:
        if r:
            k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
            band = cv2.dilate(t, k) != cv2.erode(t, k)
        else:
            band = np.zeros_like(valid)
        keep = valid & ~band
        for m, p in maps.items():
            A = acc[m][r]; A[0] += int((tm & p & keep).sum()); A[1] += int((~tm & p & keep).sum()); A[2] += int((tm & ~p & keep).sum())
res = {"freeze": man["frozen"]["time"]}
for m in acc:
    res[m] = {str(r): {"iou": round(tp / (tp + fp + fn), 4), "recall": round(tp / (tp + fn), 4), "precision": round(tp / (tp + fp), 4), "fp": fp, "fn": fn}
              for r, (tp, fp, fn) in acc[m].items()}
a.out.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=0))
