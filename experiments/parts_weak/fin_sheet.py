"""Sheet of clicked fins of 2025-03-18 (held-out clicks), whole hr window, one column per model (scratchpad only).

    PYTHONPATH=. .venv/bin/python experiments/parts_weak/fin_sheet.py --models parts_seg_r05_a parts_seg_r05_w2 parts_seg_r05_w3 parts_seg_r05_w3b \
        --name w3b_vs_w2_fins.jpg

Same 12 fins as w3_vs_w2_fins.jpg (seed 1 of the random pick). White triangle = the researcher's clicks; green = dorsal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import fin_definition as FD  # noqa: E402
from model_fine_tuning.parts_weak import DAYS, SHEETS  # noqa: E402

DAY = "2025-03-18"


def root(m):
    return FD.SHADOW if (FD.SHADOW / DAY / m / "instances.json").exists() else DAYS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    ck = FD.clicks(DAY)
    sel = ("parts_seg_r05_w2", "parts_seg_r05_w3")          # the pick of w3_vs_w2_fins.jpg
    ins = {m: json.loads((FD.SHADOW / DAY / m / "instances.json").read_text())["instances"] for m in sel}
    keys = sorted(k for k in ck if all(k in ins[m] and "hr_window" in ins[m][k] for m in sel))
    rng = np.random.default_rng(1)
    rows = []
    for key in rng.permutation(keys)[:12]:
        stem = key.rsplit("~", 1)[0]
        rgb = cv2.cvtColor(cv2.imread(str(DAYS / DAY / "images" / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        hx, hy, side, osz = ins["parts_seg_r05_w3"][key]["hr_window"]
        s = osz / side
        win = cv2.warpAffine(rgb, np.array([[s, 0, -hx * s], [0, s, -hy * s]]), (int(osz), int(osz)), flags=cv2.INTER_LINEAR)
        P = np.array([[(ck[key][n]["x"] * W - hx) * s, (ck[key][n]["y"] * H - hy) * s] for n in ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back")])
        q = np.round(P).astype(np.int32)
        ph = win.copy()
        cv2.polylines(ph, [q], True, (255, 255, 255), 2)
        panels = [cv2.resize(ph, (220, 220), interpolation=cv2.INTER_AREA)]
        for m in a.models:
            cm = np.asarray(Image.open(root(m) / DAY / m / "hr" / f"{key}.png"))
            o = FD.overlay(win, cm)
            cv2.polylines(o, [q], True, (255, 255, 255), 2)
            panels.append(cv2.resize(o, (220, 220), interpolation=cv2.INTER_AREA))
        rows.append(np.hstack([np.pad(p, ((2, 2), (2, 2), (0, 0)), constant_values=255) for p in panels]))
    sheet = np.vstack([np.hstack(rows[i:i + 2]) for i in range(0, len(rows) - 1, 2)])
    head = np.full((30, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(head, f"2025-03-18 clicked fins (white triangle): photo | {' | '.join(m.replace('parts_seg_r05_', '') for m in a.models)}   (green = dorsal)",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    p = SHEETS / a.name
    cv2.imwrite(str(p), cv2.cvtColor(np.vstack([head, sheet]), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
