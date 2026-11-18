"""Train the parts model (model_fine_tuning/parts_seg, unchanged) with weak real-photo crops added to every batch.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock env PYTHONPATH=. .venv/bin/python -m model_fine_tuning.parts_weak.train \
        --n-real 2 --w-need 0.5 -- --name parts_seg_r05_w1 --init-weights /var/tmp/sat3737-parts-cache/phase1_last.pt \
        --epochs 10 --eval-every 5 --warmup-steps 100 --workers 5

Everything after "--" goes to parts_seg.train unchanged (recipe, schedule, validation on the run05 render frames and
checkpoint selection). Two names inside parts_seg.train are replaced, nothing else:
  DataLoader   the TRAINING loader (PartsCropDataset with train=True, built exactly as before: same dataset, batch 8,
               shuffle from the same torch seed) is wrapped: after each render batch, --n-real real crops (data.py) are
               appended. The real crops come from their own loader with its own torch generator and sampler, so the
               render stream (order and crops) is the baseline's.
  parts_loss   on the first (render) rows the original parts_loss, unchanged; on the appended real rows the weak
               loss (losses.py) x --lambda-real, added.
--workers 5 (render) + 2 real workers + the main process = 8 processes; worker counts do not change any sample.
Writes experiments/<name>/parts_weak_config.json next to parts_seg.train's own files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler

from . import INDEX
from .data import POOLS, WEIGHTS, RealWeakCrops
from .losses import weak_loss

CUR: dict = {}


class EpochSampler(Sampler):
    """Global sample numbers epoch * per_epoch + j (the real crop of a number is fixed, data.py)."""

    def __init__(self, per_epoch: int):
        self.per_epoch, self.epoch = per_epoch, 0

    def __len__(self):
        return self.per_epoch

    def __iter__(self):
        base = self.epoch * self.per_epoch
        return iter(range(base, base + self.per_epoch))


class JointLoader:
    def __init__(self, render_dl: DataLoader, real_ds: RealWeakCrops, n_real: int, workers: int, seed: int):
        self.render_dl, self.n_real, self.epoch = render_dl, n_real, 0
        self.sampler = EpochSampler(len(render_dl) * n_real)
        self.real_dl = DataLoader(real_ds, batch_size=n_real, sampler=self.sampler, num_workers=workers, pin_memory=True,
                                  persistent_workers=workers > 0, generator=torch.Generator().manual_seed(seed + 12345))
        self.dataset = render_dl.dataset

    def __len__(self):
        return len(self.render_dl)

    def __iter__(self):
        self.sampler.epoch = self.epoch
        self.epoch += 1
        real_it = iter(self.real_dl)
        for rb in self.render_dl:
            xb = next(real_it)
            CUR.update(n_render=rb["image"].shape[0], inst=xb["inst"], cons=xb["cons"], ids=xb["id"], whale=xb.get("whale"))
            yield {"id": list(rb["id"]) + list(xb["id"]), "image": torch.cat([rb["image"], xb["image"]]),
                   "label": torch.cat([rb["label"], xb["label"]])}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    k = argv.index("--") if "--" in argv else len(argv)
    mine, rest = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-real", type=int, default=2, help="real crops appended to each render batch")
    ap.add_argument("--lambda-real", type=float, default=1.0)
    ap.add_argument("--w-need", type=float, default=0.5, help="weight of the 'some fluke / fin is here' terms (0 = ablation)")
    ap.add_argument("--p-hr", type=float, default=0.6)
    ap.add_argument("--real-workers", type=int, default=2)
    ap.add_argument("--weights", type=float, nargs=5, default=list(WEIGHTS), help=f"pool weights {POOLS}")
    ap.add_argument("--index", type=Path, default=INDEX)
    ap.add_argument("--maps-dir", type=Path, default=None, help="W3: weak maps with the whole clicked triangle (build_tri.py)")
    ap.add_argument("--no-need-fin", action="store_true", help="drop the 'some fin is here' term only (w2b)")
    ap.add_argument("--w-whale", type=float, default=0.0, help="W2: weight of 'no background on certain-whale pixels' (0 = W1)")
    ap.add_argument("--whale-dir", type=Path, default=Path("/var/tmp/sat3737-parts-cache/parts_weak/whale"))
    a = ap.parse_args(mine)

    from ..parts_seg import train as T
    from ..parts_seg.data import PartsCropDataset

    targs = T.parse(rest)
    real_ds = RealWeakCrops(a.index, crop=targs.crop, seed=targs.seed, p_hr=a.p_hr, photometric_strength=targs.photometric, weights=a.weights,
                            whale_dir=a.whale_dir if a.w_whale > 0 else None, maps_dir=a.maps_dir)
    out = targs.out_root / targs.name
    out.mkdir(parents=True, exist_ok=True)
    summ = a.index.with_name(a.index.stem + "_summary.json")
    (out / "parts_weak_config.json").write_text(json.dumps({
        "n_real_per_batch": a.n_real, "lambda_real": a.lambda_real, "w_need": a.w_need, "w_whale": a.w_whale, "need_fin": not a.no_need_fin, "maps_dir": str(a.maps_dir) if a.maps_dir else None,
        "whale_dir": str(a.whale_dir) if a.w_whale > 0 else None, "p_hr": a.p_hr, "pool_weights": dict(zip(POOLS, a.weights)),
        "pool_sizes": real_ds.counts(), "index": str(a.index), "index_summary": json.loads(summ.read_text()) if summ.exists() else None,
        "train_args": rest, "code": "model_fine_tuning/parts_weak (build.py, data.py, losses.py, train.py)"}, indent=1))

    orig_loss = T.parts_loss

    def loader(ds, *args, **kw):
        dl = DataLoader(ds, *args, **kw)
        if isinstance(ds, PartsCropDataset) and ds.train and a.n_real > 0:
            return JointLoader(dl, real_ds, a.n_real, a.real_workers, targs.seed)
        return dl

    def loss(logits, target, class_weight=None, dice_weight=1.0):
        n = CUR.get("n_render")
        if n is None or logits.shape[0] <= n:
            return orig_loss(logits, target, class_weight, dice_weight)
        l_r, parts = orig_loss(logits[:n], target[:n], class_weight, dice_weight)
        dev = logits.device
        wh = CUR.get("whale")
        l_w, wp = weak_loss(logits[n:], target[n:], CUR["inst"].to(dev, non_blocking=True), CUR["cons"].to(dev, non_blocking=True), w_need=a.w_need,
                            whale=wh.to(dev, non_blocking=True) if wh is not None else None, w_whale=a.w_whale,
                            need_fin=not a.no_need_fin)
        parts.update(wp)
        parts["render_loss"] = float(l_r.item())
        return l_r + a.lambda_real * l_w, parts

    T.DataLoader = loader
    T.parts_loss = loss
    return T.main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
