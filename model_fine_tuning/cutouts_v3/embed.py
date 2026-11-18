"""Fingerprints of cut-out sets with an identity checkpoint, as production makes them (jpg -> 512 px INTER_AREA ->
WhaleIdModel.embed in bfloat16 -> unit vector), for the keys the Labeling Assistant counts (CACHE/points/<day>.json).

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock env PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_v3.embed \
        --net G=experiments/pair_student_g/checkpoints/best.pt --days train,val [--sets up,fluke,finB,finC]

Sets: up = DAY/parts_up/charts/<key>__<part>.jpg (all three parts); fluke / finB / finC = CACHE/<day>/<set>/<key>.jpg.
-> CACHE/emb/<net>/<day>.npz with "<part>|<set>|keys" and "<part>|<set>|E" (a key whose file is missing is left out).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import CACHE, DAYS, ROOT, day_list
from .points import load as load_points

NET = 512
SET_PART = {"fluke": "fluke", "finB": "fin", "finC": "fin", "finP": "fin", "finP4": "fin", "finP6": "fin"}     # finP / finP4 / finP6: point-based fins (pose_side/make_finP.py) at 1.6 / 4.1 / 6.0 x the triangle


def path_of(day: str, part: str, s: str, key: str) -> Path:
    if s == "up":
        return DAYS / day / "parts_up" / "charts" / f"{key}__{part}.jpg"
    return CACHE / day / s / f"{key}.jpg"


class Files(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        im = cv2.cvtColor(cv2.imread(str(self.files[i])), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (NET, NET), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))).float() / 255.0


@torch.no_grad()
def embed_files(model, files, part_idx, dev, bs=32, workers=6):
    out = []
    for x in DataLoader(Files(files), batch_size=bs, num_workers=workers):
        x = x.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = model.embed(x, torch.full((len(x),), part_idx, dtype=torch.long, device=dev)).float()
        out.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--net", required=True, help="NAME=checkpoint"); ap.add_argument("--days", default="train,val")
    ap.add_argument("--sets", default="up,fluke,finB,finC"); ap.add_argument("--all-keys", action="store_true", help="every key with a file, not only the assistant's points")
    a = ap.parse_args(argv)
    from ..cluster.train_id import PARTS, WhaleIdModel
    name, ck = a.net.split("=", 1); ck = Path(ck) if Path(ck).is_absolute() else ROOT / ck
    dev = torch.device("cuda"); model, _ = WhaleIdModel.from_checkpoint(ck, dev)
    for day in day_list(a.days):
        t0 = time.time(); pts = load_points(day); res = {}
        for s in a.sets.split(","):
            for part in (("body", "fin", "fluke") if s == "up" else (SET_PART[s],)):
                keys = sorted(pts[part])
                if a.all_keys and s != "up":
                    keys = sorted({p.stem for p in (CACHE / day / s).glob("*.jpg")})
                keys = [k for k in keys if path_of(day, part, s, k).exists()]
                E = embed_files(model, [path_of(day, part, s, k) for k in keys], PARTS.index(part), dev)
                res[f"{part}|{s}|keys"] = np.array(keys); res[f"{part}|{s}|E"] = E.astype(np.float32)
        f = CACHE / "emb" / name / f"{day}.npz"; f.parent.mkdir(parents=True, exist_ok=True)
        old = dict(np.load(f)) if f.exists() else {}
        old.update(res); np.savez(f.with_suffix(".tmp.npz"), **old); f.with_suffix(".tmp.npz").replace(f)
        print(f"{name} {day}: " + ", ".join(f"{k.rsplit('|', 1)[0]} {len(v)}" for k, v in res.items() if k.endswith("|keys")) + f" ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
