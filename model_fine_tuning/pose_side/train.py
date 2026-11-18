"""Train the keypoint + state head (frozen DINOv3 ViT-B/16).

    # renders only (run05 train split), synthetic test on run05 val + run06 r06r_test_final
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.pose_side.train --out /var/tmp/sat3737-parts-cache/pose_side/A --steps 5000
    # + the clicks of ONE click day (half of each batch), starting from A; tested later on the other click day
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.pose_side.train --out .../B_0102 --init .../A/head.pt --real-day 2025-01-02 --steps 800 --lr 3e-4
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import FACTORY, guard
from .data import KP, STATES, RealCrops, RenderCrops, render_items
from .model import Backbone, Head, decode, heat_loss
from .realdata import clicks


def loader(ds, bs, shuffle=True, workers=14):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, drop_last=shuffle, persistent_workers=False, pin_memory=True)


def synth_eval(bb, head, items, dev, bs=32):
    ds = RenderCrops(items, train=False); errs = {n: [] for n in KP}; st_ok = []; st_conf = np.zeros((5, 5), int)
    head.eval()
    with torch.no_grad():
        for b in loader(ds, bs, shuffle=False):
            hm, st = head(bb(b["img"].to(dev, non_blocking=True))); P, _ = decode(hm.float())
            P = P.cpu().numpy(); T = b["pts"].numpy(); H = b["heat"].numpy(); S = b["state"].numpy(); SP = st.argmax(-1).cpu().numpy()
            for i in range(len(P)):
                fb = np.linalg.norm(T[i, 3] - T[i, 4]); fs = np.linalg.norm(T[i, 0] - T[i, 1])
                for k, n in enumerate(KP):
                    u = fs if k < 3 else fb
                    if H[i, k] and u > 8:
                        errs[n].append(float(np.linalg.norm(P[i, k] - T[i, k]) / u))
                    st_conf[S[i, k], SP[i, k]] += 1
    head.train()
    out = {n: dict(n=len(e), median=float(np.median(e)) if e else None, within_025=float(np.mean(np.array(e) < 0.25)) if e else None) for n, e in errs.items()}
    out["state_confusion_rows_true"] = dict(states=STATES, m=st_conf.tolist()); out["state_acc"] = float(np.trace(st_conf) / st_conf.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--steps", type=int, default=5000); ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--init", type=Path); ap.add_argument("--real-day"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-synth-eval", type=int, default=400)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed); dev = torch.device("cuda")
    t0 = time.time()
    r05 = FACTORY / "run05" / "dataset"
    tr_items = render_items(r05, "train"); va_items = render_items(r05, "val"); random.Random(0).shuffle(va_items); va_items = va_items[:a.n_synth_eval]
    r06_items = render_items(FACTORY / "r06r_test_final" / "dataset", None)
    print(f"render crops: train {len(tr_items)}, run05 val {len(va_items)}, r06r_test_final {len(r06_items)} ({time.time() - t0:.0f} s)", flush=True)
    synth = RenderCrops(tr_items, train=True, seed=a.seed)
    real = None
    if a.real_day:
        rd = guard(a.real_day.split(",")); real = RealCrops([(d, k, v) for d in rd for k, v in clicks(d).items()], seed=a.seed)
        print(f"real crops of {a.real_day}: {len(real)}", flush=True)
    bb = Backbone().to(dev); head = Head().to(dev)
    if a.init:
        head.load_state_dict(torch.load(a.init, map_location=dev))
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05 if not a.init else 0.1)
    bs_s = a.bs if real is None else a.bs // 2
    it_s = iter(loader(synth, bs_s)); it_r = iter(loader(real, a.bs - bs_s, workers=6)) if real is not None else None
    ep_s = ep_r = 0; log = []
    for step in range(1, a.steps + 1):
        def nxt(it, ds, ep, bs, w):
            try:
                return next(it), it, ep
            except StopIteration:
                ds.epoch = ep + 1; it = iter(loader(ds, bs, workers=w)); return next(it), it, ep + 1
        b, it_s, ep_s = nxt(it_s, synth, ep_s, bs_s, 14)
        if it_r is not None:
            br, it_r, ep_r = nxt(it_r, real, ep_r, a.bs - bs_s, 6)
            b = {k: torch.cat([b[k], br[k]]) for k in b}
        f = bb(b["img"].to(dev, non_blocking=True))
        hm, st = head(f)
        lh = heat_loss(hm.float(), b["pts"].to(dev), b["heat"].to(dev))
        sm = b["smask"].to(dev); ls = (torch.nn.functional.cross_entropy(st.float().flatten(0, 1), b["state"].to(dev).flatten(), reduction="none") * sm.flatten().float()).sum() / sm.float().sum().clamp_min(1)
        loss = lh + 0.5 * ls
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sched.step()
        if step % 100 == 0 or step == 1:
            r = dict(step=step, heat=float(lh), state=float(ls), lr=sched.get_last_lr()[0], t=round(time.time() - t0)); log.append(r); print(r, flush=True)
    torch.save(head.state_dict(), a.out / "head.pt")
    res = dict(args={k: str(v) for k, v in vars(a).items()}, log=log, run05_val=synth_eval(bb, head, va_items, dev), r06r_test_final=synth_eval(bb, head, r06_items, dev),
               minutes=round((time.time() - t0) / 60, 1))
    (a.out / "metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("run05_val", "r06r_test_final")}, indent=1))
    (a.out / "done").write_text("ok\n")


if __name__ == "__main__":
    main()
