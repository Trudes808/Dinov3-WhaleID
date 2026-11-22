"""Training samples for the scene model: run05 renders (seg_real.data.SynthCrops, labels 0 / 1 / ignore), S6's real
partial labels with every non-whale SAM 3 region set to ignore (real_s6mc), and the SAM 3 pseudo-labels (pseudo).
Label folders are built by experiments/sam3_labeller/build_labels.py (untracked, moonshot_cache/sam3_labeller/)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..common.data import IGNORE, photometric, to_tensor, warp_image, warp_nearest, warp_params


class PseudoCrops(Dataset):
    """512-px crops of the pseudo-labelled photos (1280 px), scale 0.5-1.5 as the recipe; centred on a stored point of an
    extra class with probability p_extra (when the photo has one), else of the whale or anywhere."""

    def __init__(self, root: Path, crop: int = 512, scale_range=(0.5, 1.5), p_extra: float = 0.6, seed: int = 0, balanced: bool = False):
        self.balanced = balanced      # v2: pick the extra class uniformly among those present, then a point of it
        self.root = Path(root); self.index = json.loads((self.root / "index.json").read_text())
        self.crop, self.scale_range, self.p_extra, self.seed, self.epoch = crop, scale_range, p_extra, seed, 0

    def __len__(self):
        return len(self.index)

    def set_epoch(self, e):
        self.epoch = e

    def get(self, i: int, rng: np.random.Generator):
        cls_pick = None
        if self.balanced and rng.random() < self.p_extra:
            # v2: an extra class uniformly, then a photo that holds it, then one of its points
            if not hasattr(self, "_by_cls"):
                self._by_cls = {}
                for k, r_ in enumerate(self.index):
                    for c in (r_.get("pts_cls") or {}):
                        self._by_cls.setdefault(c, []).append(k)
            ks = sorted(self._by_cls); cls_pick = ks[int(rng.integers(len(ks)))]
            i = self._by_cls[cls_pick][int(rng.integers(len(self._by_cls[cls_pick])))]
        r = self.index[i]
        img = cv2.cvtColor(cv2.imread(str(self.root / r["day"] / f"{r['stem']}.jpg")), cv2.COLOR_BGR2RGB)
        lab = cv2.imread(str(self.root / r["day"] / f"{r['stem']}.png"), cv2.IMREAD_UNCHANGED)
        H, W = lab.shape; C = self.crop
        s = float(np.exp(rng.uniform(np.log(self.scale_range[0]), np.log(self.scale_range[1])))); win = C / s
        pts_x, pts_w = r.get("pts_extra", []), r.get("pts_whale", [])
        if cls_pick is not None:
            pp = r["pts_cls"][cls_pick]; px, py = pp[int(rng.integers(len(pp)))]
        elif not self.balanced and pts_x and rng.random() < self.p_extra:
            px, py = pts_x[int(rng.integers(len(pts_x)))]
        elif pts_w and rng.random() < 0.7:
            px, py = pts_w[int(rng.integers(len(pts_w)))]
        else:
            px, py = rng.uniform(0, W), rng.uniform(0, H)
        M = warp_params(px - rng.uniform(0.1, 0.9) * win, py - rng.uniform(0.1, 0.9) * win, s)
        im = warp_image(img, M, (C, C)); m = warp_nearest(lab, M, (C, C), IGNORE)
        if rng.random() < 0.5:
            im, m = im[:, ::-1], m[:, ::-1]
        im = photometric(np.ascontiguousarray(im), rng, 1.0)
        return {"id": f"pseudo:{r['day']}|{r['stem']}", "image": to_tensor(im), "mask": torch.from_numpy(np.ascontiguousarray(m).astype(np.int64)),
                "contours": torch.zeros((C, C), dtype=torch.uint8)}


class Mixed3(Dataset):
    """One epoch = len(synth) samples; each sample is real_s6mc with p_s6, pseudo with p_pseudo, else synthetic."""

    def __init__(self, synth, real, pseudo, p_s6: float, p_pseudo: float, seed: int = 0):
        self.synth, self.real, self.pseudo, self.p6, self.pp, self.seed, self.epoch = synth, real, pseudo, p_s6, p_pseudo, seed, 0

    def __len__(self):
        return len(self.synth)

    def set_epoch(self, e):
        self.epoch = e; self.synth.set_epoch(e); self.real.set_epoch(e); self.pseudo.set_epoch(e)

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i, 99]); u = rng.random()
        if u < self.p6:
            return self.real.get(int(rng.integers(len(self.real))), rng)
        if u < self.p6 + self.pp:
            return self.pseudo.get(int(rng.integers(len(self.pseudo))), rng)
        return self.synth[i]
