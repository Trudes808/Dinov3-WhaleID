"""Fingerprints of parts_up and the W3c fin cut-outs with a frozen identity network, as cutouts_v3/embed.py makes them
(jpg -> 512 px INTER_AREA -> WhaleIdModel.embed in bfloat16 -> unit vector), for the keys the Labeling Assistant
counts (cut-outs v3 points cache).

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock env PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_w3c.embed \
        --net G=experiments/pair_student_g/checkpoints/best.pt --days val --sets up,s0_wide,s0_tight

Sets: up = DAY/parts_up/charts/<key>__<part>.jpg (body, fin, fluke; re-embedded here so the baseline is reproduced,
not copied); anything else = CACHE/<day>/<set>/<key>.jpg (fins). -> CACHE/emb/<net>/<day>.npz ("<part>|<set>|keys", "|E").
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import CACHE, DAYS, ROOT, day_list
from ..cutouts_v3.embed import embed_files
from ..cutouts_v3.points import load as load_points


def path_of(day: str, part: str, s: str, key: str) -> Path:
    if s == "up":
        return DAYS / day / "parts_up" / "charts" / f"{key}__{part}.jpg"
    return CACHE / day / s / f"{key}.jpg"


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--net", required=True); ap.add_argument("--days", default="val"); ap.add_argument("--sets", default="up,s0_wide,s0_tight")
    a = ap.parse_args(argv)
    from ..cluster.train_id import PARTS, WhaleIdModel
    name, ck = a.net.split("=", 1); ck = Path(ck) if Path(ck).is_absolute() else ROOT / ck
    dev = torch.device("cuda"); model, _ = WhaleIdModel.from_checkpoint(ck, dev)
    for day in day_list(a.days):
        t0 = time.time(); pts = load_points(day); res = {}
        for s in a.sets.split(","):
            for part in (("body", "fin", "fluke") if s == "up" else ("fin",)):
                keys = [k for k in sorted(pts[part]) if path_of(day, part, s, k).exists()]
                E = embed_files(model, [path_of(day, part, s, k) for k in keys], PARTS.index(part), dev)
                res[f"{part}|{s}|keys"] = np.array(keys); res[f"{part}|{s}|E"] = E.astype(np.float32)
        f = CACHE / "emb" / name / f"{day}.npz"; f.parent.mkdir(parents=True, exist_ok=True)
        old = dict(np.load(f)) if f.exists() else {}
        old.update(res); np.savez(f.with_suffix(".tmp.npz"), **old); f.with_suffix(".tmp.npz").replace(f)
        print(f"{name} {day}: " + ", ".join(f"{k.rsplit('|', 1)[0]} {len(v)}" for k, v in res.items() if k.endswith("|keys")) + f" ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
