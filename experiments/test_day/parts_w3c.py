"""Round 2: the part candidate W3c (NOT live) on the test day, applied as parts_seg/predict.py applies it (full-frame map at
1280 + per-instance hr windows, counts inside the instance mask). Rules: round2_rules.md, section 3.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock env PYTHONPATH=. .venv/bin/python experiments/test_day/parts_w3c.py \
        --arm live --day-dir LIVE_DAY --shadow OUT/live      (and --arm s6 --day-dir S6_DAY --shadow OUT/s6)
A shadow day folder (symlinks to images, cse, parts_hr of the given day) receives SHADOW/2026-04-07/w3c/{full,hr,instances.json};
nothing in the source day folders is written.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

from model_fine_tuning.common import geometry, paths
from model_fine_tuning.parts_seg import predict as PR
from model_fine_tuning.parts_seg.model import PartsSegModel

CKPT = Path("/var/tmp/sat3737-parts-cache/parts_weak/runs/parts_seg_r05_w3c/checkpoints/best.pt")


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--arm", required=True); p.add_argument("--day-dir", type=Path, required=True); p.add_argument("--shadow", type=Path, required=True)
    a = p.parse_args()
    day = a.day_dir.name; sh = a.shadow / day; sh.mkdir(parents=True, exist_ok=True)
    for sub in ("images", "cse", "parts_hr"):
        if not (sh / sub).exists():
            os.symlink(a.day_dir / sub, sh / sub)
    PR.DAYS = a.shadow
    t = geometry.load_folded_table(paths.FOLDED_TABLE); table = {"x_min_m": float(t["x_min_m"]), "length_m": float(t["length_m"])}
    dev = torch.device("cuda"); model, _ = PartsSegModel.from_checkpoint(CKPT, dev); t0 = time.time()
    if not (sh / "w3c" / "instances.json").exists():
        res = PR.process_day(model, day, "w3c", table, dev, workers=4)
        print(a.arm, len(res), "instances", f"{time.time() - t0:.0f}s", flush=True)
    # process_day maps only photos listed in parts_hr (photos with an instance); the "no gate" table needs every photo
    import json
    import cv2
    from model_fine_tuning.common.data import pad_to_multiple, to_tensor
    n = 0
    for l in open(sh / "images" / "index.jsonl"):
        stem = Path(json.loads(l)["image"]).stem; f = sh / "w3c" / "full" / f"{stem}.png"
        if f.exists():
            continue
        rgb = cv2.cvtColor(cv2.imread(str(sh / "images" / f"{stem}.jpg")), cv2.COLOR_BGR2RGB); H, W = rgb.shape[:2]; sc = 1280 / max(H, W)
        small = cv2.resize(rgb, (int(round(W * sc)), int(round(H * sc))), interpolation=cv2.INTER_AREA)
        padded, (h, w) = pad_to_multiple(small); PR.save_palette(PR.run_model(model, to_tensor(padded), dev)[:h, :w], f); n += 1
    print(a.arm, n, "full-frame maps added for photos without an instance", flush=True)


if __name__ == "__main__":
    main()
