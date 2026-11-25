"""W3e's training command with the pilot r06_pilot22 mixed into the TRAINING render stream (P_add / P_only).

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock env PYTHONPATH=. .venv/bin/python -m model_fine_tuning.pilot22_train.train \
        --pilot-share 0.3333333 -- <parts_weak.train arguments exactly as W3e: --n-real 2 --w-need 0.5 --w-whale 1.0 \
        --maps-dir .../maps_tri3 -- --name NAME --out-root DIR --cache .../run05/labels_triA --init-weights ... --seed S ...>

Everything after the first "--" goes to parts_weak.train unchanged (which hands its own "--" part to parts_seg.train).
Two names are replaced before that, nothing else:
  parts_seg.train.PartsCropDataset   for the TRAINING dataset only (train=True) a MixDataset: a PartsCropDataset
      subclass (so parts_weak still appends its real crops) holding the run05 dataset exactly as W3e builds it and a
      second PartsCropDataset over the pilot cache (prep.py) with the same crop / scale / bias / photometric arguments.
      Validation (run05 val, rule-A cache, checkpoint choice) and the class weights (run05 train labels) are W3e's.
  parts_weak.train.DataLoader        for a MixDataset the shuffle is replaced by MixSampler; every other loader
      (validation, parts_weak's real-crop loader) is built exactly as before.
MixSampler: per epoch the same number of render samples as W3e (len(run05 train) = 6440, i.e. 805 batches of 8, so the
schedule, warm-up, real crops per batch and steps are W3e's); round(share x 6440) of them are pilot shots, the rest
run05 shots. Both are drawn as streams of permutations (every shot once before any shot twice), mixed and shuffled with
numpy rng (seed, epoch). run05 crops: as in W3e (PartsCropDataset with its epoch left at 0, which is what W3e's
persistent workers see); pilot crops: the pass number through the pilot is the crop rng's epoch, so a pilot shot drawn
again gets a new crop. --pilot-share 1 = P_only (no run05 shots in the training stream).
Writes <out-root>/<name>/pilot22_config.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Sampler

from ..parts_seg.data import PartsCropDataset
from .prep import CACHE

PILOT_BASE = 10_000_000          # global sample index >= this: pilot, (index - base) = pass * n_pilot + i


class PilotRun:
    """the pieces of WhaleRun that PartsCropDataset touches, over the prep.py cache"""

    def __init__(self, cache: Path):
        self.cache = cache
        self.dataset_dir = cache          # labels are always in the cache; derive() is never reached

    def image(self, shot_id: str) -> np.ndarray:
        bgr = cv2.imread(str(self.cache / "images" / f"{shot_id}.png"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(shot_id)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class MixSampler(Sampler):
    def __init__(self, n_r05: int, n_pilot: int, share: float, seed: int):
        self.n_r05, self.n_pilot, self.share, self.seed = n_r05, n_pilot, share, seed
        self.per_epoch = n_r05
        self.n_p = int(round(share * self.per_epoch)) if n_pilot else 0
        self.n_r = self.per_epoch - self.n_p
        self.epoch = 0
        self.r_stream, self.p_stream, self.p_pass = [], [], -1
        self.rng = np.random.default_rng([seed, 2211])
        self.log: list[dict] = []

    def __len__(self):
        return self.per_epoch

    def _take_r(self, k):
        out = []
        while len(out) < k:
            if not self.r_stream:
                self.r_stream = list(self.rng.permutation(self.n_r05))
            out.append(int(self.r_stream.pop()))
        return out

    def _take_p(self, k):
        out = []
        while len(out) < k:
            if not self.p_stream:
                self.p_pass += 1
                self.p_stream = list(self.rng.permutation(self.n_pilot))
            out.append(PILOT_BASE + self.p_pass * self.n_pilot + int(self.p_stream.pop()))
        return out

    def __iter__(self):
        idx = np.array(self._take_r(self.n_r) + self._take_p(self.n_p), np.int64)
        idx = idx[np.random.default_rng([self.seed, 2212, self.epoch]).permutation(len(idx))]
        self.log.append({"epoch": self.epoch, "run05": self.n_r, "pilot": self.n_p, "pilot_pass": self.p_pass})
        self.epoch += 1
        return iter(idx.tolist())


class MixDataset(PartsCropDataset):
    def __init__(self, run, ids, cache, *, pilot_ids: list[str], pilot_cache: Path, share: float, **kw):
        super().__init__(run, ids, cache, **kw)
        self.r05 = PartsCropDataset(run, ids, cache, **kw)
        self.pil = PartsCropDataset(PilotRun(pilot_cache), pilot_ids, pilot_cache / "labels", **kw)
        self.share = share
        self.sampler = MixSampler(len(ids), len(pilot_ids), share, kw.get("seed", 0))

    def set_epoch(self, epoch: int):                       # run05 crops keep epoch 0 (W3e's persistent workers)
        self.epoch = epoch

    def __len__(self):
        return self.sampler.per_epoch

    def __getitem__(self, index: int):
        if index < PILOT_BASE:
            return self.r05[index]
        g = index - PILOT_BASE
        p, i = divmod(g, len(self.pil.ids))
        self.pil.epoch = p
        item = self.pil[i]
        item["id"] = "p22_" + item["id"]
        return item


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    k = argv.index("--") if "--" in argv else len(argv)
    mine, rest = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot-share", type=float, required=True, help="share of the training render samples from the pilot (1 = P_only)")
    ap.add_argument("--pilot-cache", type=Path, default=CACHE)
    a = ap.parse_args(mine)
    ids = json.loads((a.pilot_cache / "ids.json").read_text())
    pilot_ids = ids["train"]

    from ..parts_seg import train as T
    from ..parts_weak import train as PW

    made: dict = {}

    def pcd(run, ids_, cache, **kw):
        if kw.get("train", True):
            ds = MixDataset(run, ids_, cache, pilot_ids=pilot_ids, pilot_cache=a.pilot_cache, share=a.pilot_share, **kw)
            made["ds"] = ds
            return ds
        return PartsCropDataset(run, ids_, cache, **kw)

    torch_dl = PW.DataLoader

    def dl(ds, *args, **kw):
        if isinstance(ds, MixDataset):
            kw.pop("shuffle", None)
            kw["sampler"] = ds.sampler
        return torch_dl(ds, *args, **kw)

    T.PartsCropDataset = pcd
    PW.DataLoader = dl
    # where parts_seg.train writes (its own --name / --out-root, parsed the same way)
    k2 = rest.index("--") if "--" in rest else len(rest)
    targs = T.parse(rest[k2 + 1:])
    out = targs.out_root / targs.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"pilot_share": a.pilot_share, "pilot_cache": str(a.pilot_cache), "pilot_train_shots": len(pilot_ids),
           "pilot_dropped": ids["dropped"], "excluded_individual": "i12", "args": argv,
           "code": "model_fine_tuning/pilot22_train/train.py"}
    (out / "pilot22_config.json").write_text(json.dumps(cfg, indent=1))
    rc = PW.main(rest)
    ds = made.get("ds")
    if ds is not None:
        cfg["sampler_log"] = ds.sampler.log
        cfg["per_epoch"] = {"run05": ds.sampler.n_r, "pilot": ds.sampler.n_p}
        (out / "pilot22_config.json").write_text(json.dumps(cfg, indent=1))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
