"""DROPPED 2026-11-17: the swap / paste route was dropped after the user's review of the pastes and the rw = sc result;
the sw arm was never trained and the composite cache (realwater/sw/) was deleted.

Swap arm (parts_seg_r05_sw): run05 renders pasted where a real whale of the same view and size sat, keeping the
real photo's own wash (paste06.py, method "swapfront", colour transfer ON for run05 whales, grain matched, labels by
the corrected ignore rule of paste06.paste).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.synth_aug.swap_run05 precompute [--variants 2 --workers 12]
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.synth_aug.swap_run05 train --p 0.5 -- --name parts_seg_r05_sw \
        --init-weights /var/tmp/sat3737-parts-cache/phase1_last.pt --epochs 10 --eval-every 5 --warmup-steps 100 --workers 6

Frames: the real photo at 1280 px wide (the frame width of the render and of the model's full-frame input), so the
whale's size in the frame is the real whale's. Partners: paste06.match with the tight gate (silhouette IoU >= 0.55,
render box >= 0.9 of the real box in width and height), scale 0.33 - 4 (run05 whales are about half the real size),
real whales from the FULL training-day pool (paste06 pool). Up to --variants composites per render, each with a
different random passing partner (seeded), precomputed on CPU into /var/tmp/sat3737-parts-cache/realwater/sw/.
Training: the same wrapper as the other arms (train_rw.CompositeCropDataset: decided per sample from (seed, epoch,
index), before the parts agent's own crop / flip / photometric code); with probability p a render with composites is
replaced by one of them (random variant), a render without (no partner, whale cut by the frame, negative) gets the
sc arm's treatment (real scale, render water). Validation = plain render frames as in every arm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..common import paths
from ..common.data import WhaleRun
from ..parts_seg import data as pdata
from . import paste06 as P6
from . import plates as P
from .composite import Options
from .train_rw import CompositeCropDataset

RUN = paths.WHALE_FACTORY_ROOT / "data" / "renders" / "run05"
SPLIT = paths.SPLITS_DIR / "run05.json"
OUT = P.CACHE / "sw"
INDEX = OUT / "index.json"
FRAME_W = 1280


def render_dict(run: WhaleRun, sid: str) -> dict | None:
    label = pdata.load_label(run, sid, pdata.DEFAULT_CACHE)
    m = run.mask(sid)
    mask = np.where(m == 255, 2, m).astype(np.uint8)
    ip = run.dataset_dir / "instances" / f"{sid}.png"
    if ip.exists():
        inst = np.asarray(Image.open(ip))
        inst = inst[..., 0] if inst.ndim == 3 else inst
        inst = np.where(mask == 1, np.where(inst >= 2, 2, 1), 0).astype(np.uint8)
    else:
        inst = (mask == 1).astype(np.uint8)
    vis = (inst == 1) & (mask == 1)
    if vis.sum() < 300 or not ((label >= 1) & (label <= 5)).any():
        return None
    allv = mask == 1
    if allv[:, :2].any() or allv[:, -2:].any() or allv[:2].any() or allv[-2:].any():
        return None                                              # whale cut by the frame: sc treatment
    return dict(ds=str(run.dataset_dir), id=sid, img=run.image(sid), mask=mask, inst=inst, parts=label, meta={})


def load_photo_w(q: dict, width: int = FRAME_W):
    """the pool photo at `width` px wide, with its masks and the pool record's box scaled to it."""
    ph = P6.load_photo(q)
    H, W = ph["img"].shape[:2]
    s = width / W
    size = (width, int(round(H * s)))
    img = cv2.resize(ph["img"], size, interpolation=cv2.INTER_AREA)
    this = cv2.resize(ph["this"].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    others = cv2.resize(ph["others"].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    sb = width / q["size"][0]
    box = [v * sb for v in q["box"]]
    return dict(img=img, this=this, others=others, box=box), dict(q, box=box)


def _scaled_pool(recs):
    return [dict(q, box=[v * FRAME_W / q["size"][0] for v in q["box"]]) for q in recs]


_G: dict = {}


def _init():
    recs, masks = P6.load_pool()
    _G.update(recs=recs, recs_s=_scaled_pool(recs), masks=masks, run=WhaleRun(RUN, SPLIT))


def _job(args):
    sid, n_var, seed = args
    try:
        r = render_dict(_G["run"], sid)
        if r is None:
            return sid, [], "no usable whale"
        ps = P6.match(r, _G["recs_s"], _G["masks"], scale_range=(0.33, 4.0), all_ok=True)
        if not ps:
            return sid, [], "no partner"
        rng = np.random.default_rng([seed, int(sid)])
        pick = rng.choice(len(ps), min(n_var, len(ps)), replace=False)
        out = []
        for v, j in enumerate(pick):
            i, flip, iou, f = ps[j]
            q = _G["recs"][i]
            ph, q2 = load_photo_w(q)
            bg = P6.clean_photo(ph, wash=False)
            res = P6.paste(r, ph, q2, flip, "swapfront", bg, colour=True, seed=int(sid) * 10 + v)
            ip, lp = OUT / "img" / f"{sid}_{v}.jpg", OUT / "lab" / f"{sid}_{v}.png"
            cv2.imwrite(str(ip), cv2.cvtColor(res["img"], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])
            cv2.imwrite(str(lp), res["parts"])
            out.append(dict(v=v, img=str(ip), lab=str(lp), real=q["key"], day=q["day"], iou=round(float(iou), 3),
                            scale=round(float(res["scale"]), 3), n_partners=len(ps), ignore_px=res["safety_px"]))
        return sid, out, None
    except Exception as exc:  # noqa: BLE001
        return sid, [], f"error {exc!r}"[:200]


def precompute(n_var: int = 2, workers: int = 12, seed: int = 0, limit: int | None = None) -> None:
    from multiprocessing import Pool

    (OUT / "img").mkdir(parents=True, exist_ok=True)
    (OUT / "lab").mkdir(parents=True, exist_ok=True)
    run = WhaleRun(RUN, SPLIT)
    ids = run.ids("train")[:limit]
    index, why = {}, {}
    with Pool(workers, initializer=_init) as pool:
        for k, (sid, out, err) in enumerate(pool.imap_unordered(_job, [(s, n_var, seed) for s in ids], chunksize=4)):
            if out:
                index[sid] = out
            else:
                why[err] = why.get(err, 0) + 1
            if (k + 1) % 500 == 0:
                print(f"{k + 1}/{len(ids)} renders, {len(index)} with composites", why, flush=True)
    INDEX.write_text(json.dumps({"variants": n_var, "renders": len(ids), "with_composites": len(index), "without": why,
                                 "composites": index}))
    (OUT / "precompute.done").write_text(json.dumps({"renders": len(ids), "with_composites": len(index), "without": why}))
    print("done", len(ids), "renders,", len(index), "with composites;", why)


class SwapCropDataset(CompositeCropDataset):
    """p: swap composite (precomputed) when the render has one, else the sc treatment (real scale, render water)."""

    def __init__(self, run, ids, cache=pdata.DEFAULT_CACHE, *, p: float = 0.5, **kw):
        super().__init__(run, ids, cache, p=p, opts=Options(real_water=False, real_scale=True), **kw)
        self.swaps = json.loads(INDEX.read_text())["composites"]

    def composite(self, index: int):
        rng = np.random.default_rng([self.seed, self.epoch, index, 7919])
        if not self.train or rng.random() >= self.p:
            return None
        sid = self.ids[index]
        vs = self.swaps.get(sid)
        if not vs:
            return super().composite(index)                    # same seed -> same decision; sc treatment
        v = vs[int(rng.integers(len(vs)))]
        img = cv2.cvtColor(cv2.imread(v["img"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        lab = cv2.imread(v["lab"], cv2.IMREAD_UNCHANGED)
        return img, lab, {"kind": "swap", "real": v["real"]}


def train_main(argv) -> int:
    if "--" in argv:
        k = argv.index("--")
        mine, rest = argv[:k], argv[k + 1:]
    else:
        mine, rest = argv, []
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=0.5)
    a = ap.parse_args(mine)
    from ..parts_seg import train as T

    targs = T.parse(rest)
    out = targs.out_root / targs.name
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((OUT / "precompute.done").read_text())
    (out / "realwater_config.json").write_text(json.dumps({"mode": "swap (paste06 swapfront, colour on, corrected labels)",
                                                           "p": a.p, "precompute": meta, "index": str(INDEX), "train_args": rest}, indent=1))

    def factory(run, ids, cache=pdata.DEFAULT_CACHE, train=True, **kw):
        if train:
            return SwapCropDataset(run, ids, cache, p=a.p, train=True, **kw)
        return pdata.PartsCropDataset(run, ids, cache, train=False, **kw)

    T.PartsCropDataset = factory
    return T.main(rest)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd, rest = argv[0], argv[1:]
    if cmd == "precompute":
        ap = argparse.ArgumentParser()
        ap.add_argument("--variants", type=int, default=2)
        ap.add_argument("--workers", type=int, default=12)
        ap.add_argument("--limit", type=int, default=None)
        a = ap.parse_args(rest)
        precompute(a.variants, a.workers, limit=a.limit)
        return 0
    if cmd == "train":
        return train_main(rest)
    raise SystemExit("precompute | train")


if __name__ == "__main__":
    raise SystemExit(main())
