"""Scene segmentation (7 classes) with the recipe of seg_splash_s6 (experiments/sam3_labeller/rules_labeller.md A6).

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock env PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_multi.train \
        --name seg_scene_sam3_s0 --seed 0 --pseudo-dir PSEUDO --real-dir REAL_S6MC [--budget 1700]

Resumable: after every epoch the full state goes to /var/tmp/sat3737-parts-cache/seg_multi/<name>_resume.pt; a run that
passes --budget seconds stops after the current epoch and the next call continues. At the end the last state is saved
as experiments/<name>/checkpoints/best.pt (the kept checkpoint = last epoch, as S6h uses last.pt), the resume file is
deleted, and the run05 validation renders are scored for the whale (P(whale) >= 0.5, dataset IoU as the recipe).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..common import paths
from ..common.data import IGNORE, SegCropDataset, open_runs
from ..common.train_utils import Logger, apply_lr, count_params, load_checkpoint, lr_at, move, save_checkpoint, set_seed
from ..seg_real import common as SC
from ..seg_real.data import RealBlobCrops, SynthCrops
from .data import Mixed3, PseudoCrops
from .model import SceneSegModel, scene_loss

RESUME = Path("/var/tmp/sat3737-parts-cache/seg_multi")


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pseudo-dir", type=Path, required=True); p.add_argument("--real-dir", type=Path, required=True)
    p.add_argument("--init", type=Path, default=SC.SEG_R05)
    p.add_argument("--p-s6", type=float, default=0.2); p.add_argument("--p-pseudo", type=float, default=0.3); p.add_argument("--p-extra", type=float, default=0.6)
    p.add_argument("--balanced", action="store_true", help="v2: class-balanced centring of pseudo crops")
    p.add_argument("--epochs", type=int, default=8); p.add_argument("--batch-size", type=int, default=8); p.add_argument("--crop", type=int, default=512)
    p.add_argument("--lr-head", type=float, default=3e-4); p.add_argument("--lr-backbone", type=float, default=3e-5)
    p.add_argument("--layer-decay", type=float, default=0.8); p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=200); p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=6); p.add_argument("--budget", type=float, default=1700)
    p.add_argument("--max-steps", type=int, default=None)
    return p.parse_args(argv)


@torch.no_grad()
def eval_whale(model, run, dev, n=None) -> dict:
    ds = SegCropDataset(run, run.ids("val")[:n], train=False); dl = DataLoader(ds, batch_size=4, num_workers=4)
    I = U = 0; model.eval()
    for b in dl:
        b = move(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(b["image"])
        p = torch.softmax(lg.float(), 1)[:, 1].cpu().numpy(); m = b["mask"].cpu().numpy()
        for i in range(len(p)):
            h, w = b["orig_hw"][i].tolist(); pr = p[i, :h, :w] >= 0.5; gt = m[i, :h, :w] == 1; ok = m[i, :h, :w] != IGNORE
            I += int((pr & gt & ok).sum()); U += int(((pr | gt) & ok).sum())
    return {"dataset_iou": I / max(1, U), "n": len(ds)}


def main(argv=None) -> int:
    a = parse(argv); set_seed(a.seed); dev = torch.device("cuda"); out = paths.EXPERIMENTS_DIR / a.name; out.mkdir(parents=True, exist_ok=True)
    RESUME.mkdir(parents=True, exist_ok=True); rpath = RESUME / f"{a.name}_resume.pt"; t0 = time.time()
    run = open_runs(SC.RUN05, paths.SPLITS_DIR / "run05.json")
    synth = SynthCrops(run, run.ids("train"), crop=a.crop, scale_range=(0.5, 1.5), whale_bias=0.7, photometric_strength=1.0, train=True, seed=a.seed)
    real = RealBlobCrops(crop=a.crop, seed=a.seed, real_dir=a.real_dir); pseudo = PseudoCrops(a.pseudo_dir, crop=a.crop, p_extra=a.p_extra, seed=a.seed, balanced=a.balanced)
    ds = Mixed3(synth, real, pseudo, a.p_s6, a.p_pseudo, seed=a.seed)
    model = SceneSegModel().to(dev)
    config = {"task": "scene_segmentation", "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, "model": model.config,
              "train_synthetic": len(synth), "real_s6mc_centres": len(real), "pseudo_photos": len(pseudo), "params": count_params(model),
              "recipe": "experiments/seg_splash_s6/config.json + 7-class head (rules_labeller.md A6)"}
    opt = torch.optim.AdamW(model.param_groups(a.lr_head, a.lr_backbone, a.layer_decay, a.weight_decay), betas=(0.9, 0.999))
    start_epoch, step = 0, 0
    if rpath.exists():
        st = load_checkpoint(rpath); model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start_epoch, step = st["epoch"] + 1, st["step"]
        print(f"resumed after epoch {st['epoch']} (step {step})", flush=True)
    else:
        model.init_from_binary(load_checkpoint(a.init)["model"])
        (out / "config.json").write_text(json.dumps(config, indent=1))
    log = Logger(out); log.log(event="start" if start_epoch == 0 else "resume", epoch=start_epoch, **{k: v for k, v in config.items() if k != "args"})
    tl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True, persistent_workers=True)
    total = a.epochs * len(tl)
    for epoch in range(start_epoch, a.epochs):
        model.train(); ds.set_epoch(epoch); te = time.time(); run_l = []; src = {"real": 0, "pseudo": 0, "synth": 0}
        torch.manual_seed(a.seed * 1000 + epoch)
        for batch in tl:
            for i in batch["id"]:
                src["real" if str(i).startswith("real:") else "pseudo" if str(i).startswith("pseudo:") else "synth"] += 1
            f = lr_at(step, total, a.warmup_steps, 1.0); apply_lr(opt, f); batch = move(batch, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch["image"])
            loss, parts = scene_loss(logits.float(), batch["mask"], a.dice_weight)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0); opt.step()
            run_l.append(loss.item()); step += 1
            if step % 50 == 0:
                log.log(event="train", epoch=epoch, step=step, loss=float(np.mean(run_l[-50:])), **parts, lr_factor=f, sec_per_step=round((time.time() - te) / len(run_l), 3))
            if a.max_steps and step >= a.max_steps:
                break
        log.log(event="epoch", epoch=epoch, step=step, loss=float(np.mean(run_l)), seconds=round(time.time() - te, 1), sources=src)
        last = epoch == a.epochs - 1 or bool(a.max_steps and step >= a.max_steps)
        if last:
            save_checkpoint(out / "checkpoints" / "best.pt", model, config, {"step": step, "epoch": epoch})
            m = eval_whale(model, run, dev); log.log(event="val", epoch=epoch, step=step, **m); (out / "val_metrics.json").write_text(json.dumps(m, indent=1))
            rpath.unlink(missing_ok=True); log.log(event="done", seconds_this_hold=round(time.time() - t0, 1)); (out / "DONE").touch()
            return 0
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "step": step}, rpath)
        if time.time() - t0 + (time.time() - te) > a.budget:
            log.log(event="pause", epoch=epoch, step=step, seconds_this_hold=round(time.time() - t0, 1)); return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
