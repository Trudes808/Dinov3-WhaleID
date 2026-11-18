"""Segmentation probabilities of a checkpoint on the real photos the seg_real scoring needs (cached, never in the repo).

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock env PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.predict \
        --checkpoint CKPT --name NAME [--set labelled,views,clicks]
        -> CACHE/pred/NAME/<day>/<stem>.png   uint8 = round(255 x P(whale)) at long side 1280 (the model's input scale;
           segmentation.predict.segment() runs there too and only upsamples afterwards)

Photo sets (training and validation days only; evaluation days refused by common.check_day):
  labelled  every photo with a blob in experiments/seg_splash/blob_labels.json
  views     photos whose main whale is certain: a view label F / D / O (view_class/view_labels.json) on its instance
  clicks    the researcher's clicked photos of 2025-01-02 and 2025-03-18 (Whale-Review-Web pair_marks.json, read only)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import common as C

PRED = C.CACHE / "pred"
CLICK_DAYS = ["2025-01-02", "2025-03-18"]


def load_marks(day: str):
    """experiments/instances_v2/common.load_marks (imported read-only): per old key the main whale's keypoints, and
    the other whales marked on that photo."""
    sys.path.insert(0, str(C.REPO / "experiments" / "instances_v2"))
    import common as IC          # noqa: E402  (the instances_v2 helper module)
    C.check_day(day)
    return IC.load_marks(day)


def view_keys() -> dict:
    """old instance key -> (day, code) for view labels F / D / O on train / val days."""
    v = json.loads((C.REPO / "experiments" / "view_class" / "view_labels.json").read_text()); out = {}
    for k, r in v.items():
        if r["day"] in C.EVAL or r["day"] not in C.TRAIN + C.VAL:
            continue
        if r["code"] and r["code"][0] in "FDO":
            out[k] = (r["day"], r["code"])
    return out


def photo_set(names: list) -> list:
    S = set()
    if "labelled" in names:
        S |= {(b["day"], b["stem"]) for b in C.load_labels()}
    if "views" in names:
        S |= {(d, k.rsplit("~", 1)[0]) for k, (d, _) in view_keys().items()}
    if "clicks" in names:
        for day in CLICK_DAYS:
            photos, _ = load_marks(day)
            S |= {(day, k.rsplit("~", 1)[0]) for k in photos}
    for d, _ in S:
        C.check_day(d)
    return sorted(S)


def resize_1280(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]; s = 1280 / max(h, w)
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def pred_path(name: str, day: str, stem: str) -> Path:
    return PRED / name / day / f"{stem}.png"


def load_prob(name: str, day: str, stem: str) -> np.ndarray:
    """float32 P(whale) at long side 1280."""
    return cv2.imread(str(pred_path(name, day, stem)), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0


def main(argv=None) -> int:
    import torch
    from ..common.data import pad_to_multiple, to_tensor
    from ..segmentation.model import WhaleSegModel
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--set", default="labelled,views,clicks"); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    todo = [(d, s) for d, s in photo_set(a.set.split(",")) if not pred_path(a.name, d, s).exists()]
    print(f"{a.name}: {len(todo)} photos to segment", flush=True); t = time.time()
    if not todo:
        return 0
    dev = torch.device("cuda"); model, _ = WhaleSegModel.from_checkpoint(Path(a.checkpoint), dev)

    class Photos(torch.utils.data.Dataset):
        def __len__(self):
            return len(todo)

        def __getitem__(self, j):
            d, s = todo[j]
            return j, resize_1280(cv2.cvtColor(C.photo(d, s), cv2.COLOR_BGR2RGB))

    dl = torch.utils.data.DataLoader(Photos(), batch_size=None, num_workers=a.workers)
    with torch.no_grad():
        for n, (j, small) in enumerate(dl):
            small = small.numpy(); padded, (h, w) = pad_to_multiple(small)
            x = to_tensor(padded)[None].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prob = torch.sigmoid(model(x).float())[0, :h, :w].cpu().numpy()
            d, s = todo[int(j)]; p = pred_path(a.name, d, s); p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), np.round(prob * 255).astype(np.uint8))
            if n % 500 == 0:
                print(f"  {n} / {len(todo)} ({time.time() - t:.0f} s)", flush=True)
    print(f"{a.name}: done ({time.time() - t:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
