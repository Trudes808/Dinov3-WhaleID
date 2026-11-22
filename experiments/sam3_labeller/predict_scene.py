"""Scene model prediction (rules_labeller.md A7) on whole days, and the cost measure (C4).

    flock LOCK env PYTHONPATH=. .venv/bin/python experiments/sam3_labeller/predict_scene.py --ckpt CKPT --name NAME --days val[,2026-04-07]
        -> OUT/pred/NAME/<day>/<stem>.png   our classes (0 water / sky, 1 whale, 2 person, 3 rail, 4 tape, 5 deck, 6 land) at 1280
    flock LOCK env PYTHONPATH=. .venv/bin/python experiments/sam3_labeller/predict_scene.py --cost --ckpt CKPT --n 100
Whale = hysteresis 0.6 / 0.1 on P(whale); every other pixel = argmax over the non-whale classes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import lab_common as L  # noqa: E402
from model_fine_tuning.common.data import pad_to_multiple, to_tensor  # noqa: E402
from model_fine_tuning.seg_multi.model import SceneSegModel  # noqa: E402
from model_fine_tuning.segmentation.model import WhaleSegModel  # noqa: E402


def to_map(prob: np.ndarray) -> np.ndarray:
    """prob (K, H, W) softmax -> class map with the A7 whale rule."""
    whale = L.hyst(prob[1]); q = prob.copy(); q[1] = -1
    lab = q.argmax(0).astype(np.uint8); lab[whale] = 1
    return lab


def run_days(a) -> None:
    days = []
    for g in a.days.split(","):
        days += L.VAL if g == "val" else [g]
    todo = [(d, s) for d in days for s in L.day_stems_any(d) if not (L.OUT / "pred" / a.name / d / f"{s}.png").exists()]
    print(len(todo), "to do", flush=True)
    dev = torch.device("cuda"); model, _ = SceneSegModel.from_checkpoint(a.ckpt, dev)

    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(todo)

        def __getitem__(self, j):
            d, s = todo[j]
            return j, L.resize_1280(cv2.cvtColor(cv2.imread(str(L.photo_path(d, s))), cv2.COLOR_BGR2RGB))

    with torch.no_grad():
        for j, small in torch.utils.data.DataLoader(DS(), batch_size=None, num_workers=6):
            small = small.numpy(); padded, (h, w) = pad_to_multiple(small)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(to_tensor(padded)[None].to(dev))
            prob = torch.softmax(lg.float(), 1)[0, :, :h, :w].cpu().numpy()
            d, s = todo[int(j)]; o = L.OUT / "pred" / a.name / d / f"{s}.png"; o.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(o), to_map(prob))
    print("done", flush=True)


def cost(a) -> None:
    dev = torch.device("cuda"); stems = [(d, s) for d in L.VAL for s in L.day_stems(d)][:: max(1, 884 // a.n)][: a.n]
    imgs = [L.resize_1280(cv2.cvtColor(cv2.imread(str(L.photo_path(d, s))), cv2.COLOR_BGR2RGB)) for d, s in stems]
    res = {}
    for name, loader in [("s6h", lambda: WhaleSegModel.from_checkpoint(L.S6, dev)[0]), ("scene", lambda: SceneSegModel.from_checkpoint(a.ckpt, dev)[0])]:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); model = loader(); ts = []
        with torch.no_grad():
            for k, small in enumerate(imgs):
                padded, (h, w) = pad_to_multiple(small); torch.cuda.synchronize(); t = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = model(to_tensor(padded)[None].to(dev))
                if name == "s6h":
                    p = torch.sigmoid(lg.float())[0, :h, :w].cpu().numpy(); L.hyst(p)
                else:
                    to_map(torch.softmax(lg.float(), 1)[0, :, :h, :w].cpu().numpy())
                torch.cuda.synchronize()
                if k >= 5:
                    ts.append(time.time() - t)
        res[name] = {"s_per_photo_median": float(np.median(ts)), "s_per_photo_mean": float(np.mean(ts)), "n": len(ts),
                     "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1), "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2)}
        del model
    print(json.dumps(res, indent=1)); (L.OUT / "cost.json").write_text(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", type=Path, required=True); ap.add_argument("--name", default=None)
    ap.add_argument("--days", default="val"); ap.add_argument("--cost", action="store_true"); ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    cost(a) if a.cost else run_days(a)


if __name__ == "__main__":
    main()
