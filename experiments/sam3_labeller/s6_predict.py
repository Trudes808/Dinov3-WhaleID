"""P(whale) of a binary whale checkpoint (default S6 = seg_splash_s6 last.pt) at long side 1280 on whole days, saved as
uint8 PNG (round(255 P)) to moonshot_cache/sam3_labeller/<name>/<day>/<stem>.png. Resumable, time budget per lock hold.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock env PYTHONPATH=. .venv/bin/python experiments/sam3_labeller/s6_predict.py \
        --days train,val [--name s6prob] [--budget 1500]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import lab_common as L  # noqa: E402
from model_fine_tuning.common.data import pad_to_multiple, to_tensor  # noqa: E402
from model_fine_tuning.segmentation.model import WhaleSegModel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--days", default="train,val"); ap.add_argument("--name", default="s6prob")
    ap.add_argument("--checkpoint", type=Path, default=L.S6); ap.add_argument("--budget", type=float, default=1500)
    a = ap.parse_args()
    days = []
    for g in a.days.split(","):
        days += L.TRAIN if g == "train" else L.VAL if g == "val" else [g]
    todo = [(d, s) for d in days for s in L.day_stems_any(d) if not (L.OUT / a.name / d / f"{s}.png").exists()]
    print(len(todo), "to do", flush=True)
    if not todo:
        (L.OUT / a.name / "_DONE").touch(); return
    dev = torch.device("cuda"); model, _ = WhaleSegModel.from_checkpoint(a.checkpoint, dev); t0 = time.time()

    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(todo)

        def __getitem__(self, j):
            d, s = todo[j]
            return j, L.resize_1280(cv2.cvtColor(cv2.imread(str(L.photo_path(d, s))), cv2.COLOR_BGR2RGB))

    n = 0
    with torch.no_grad():
        for j, small in torch.utils.data.DataLoader(DS(), batch_size=None, num_workers=6):
            small = small.numpy(); padded, (h, w) = pad_to_multiple(small)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = torch.sigmoid(model(to_tensor(padded)[None].to(dev)).float())[0, :h, :w].cpu().numpy()
            d, s = todo[int(j)]; o = L.OUT / a.name / d / f"{s}.png"; o.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(o), np.round(p * 255).astype(np.uint8)); n += 1
            if n % 1000 == 0:
                print(n, f"{time.time() - t0:.0f} s", flush=True)
            if time.time() - t0 > a.budget:
                print("budget reached", flush=True); break
    left = sum(1 for d, s in todo if not (L.OUT / a.name / d / f"{s}.png").exists())
    print(f"done {n}, left {left}, {time.time() - t0:.0f} s", flush=True)
    if not left:
        (L.OUT / a.name / "_DONE").touch()


if __name__ == "__main__":
    main()
