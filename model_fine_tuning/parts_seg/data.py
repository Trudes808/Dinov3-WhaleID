"""Training / validation crops for the part segmentation (labels from labels.py, cached as palette PNGs).

Crops as the whale segmentation's SegCropDataset (common/data.py): random 512 windows at 0.5-1.5 x native scale,
biased to contain whale pixels, horizontal flip (the classes are left / right symmetric), the same photometric
augmentation (exposure, white balance, saturation, blur, noise, JPEG). Additionally half of the whale-biased crops are
centred on a pixel of a rare part (fluke, dorsal fin, pectoral fin) when the shot has one. Validation = full frames.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..common.data import IGNORE, pad_to_multiple, photometric, to_tensor, warp_image, warp_nearest, warp_params
from .labels import derive

DEFAULT_CACHE = Path("/var/tmp/sat3737-parts-cache/run05/labels")


def load_label(run, shot_id: str, cache: Path | None) -> np.ndarray:
    if cache is not None:
        p = Path(cache) / f"{shot_id}.png"
        if p.exists():
            return np.asarray(Image.open(p))
    return derive(run.dataset_dir, shot_id)


class PartsCropDataset(Dataset):
    def __init__(self, run, ids: list[str], cache: Path | None = DEFAULT_CACHE, crop: int = 512, scale_range=(0.5, 1.5),
                 whale_bias: float = 0.7, rare_bias: float = 0.5, flip: bool = True, photometric_strength: float = 1.0,
                 train: bool = True, seed: int = 0):
        self.run, self.ids, self.cache, self.crop = run, list(ids), cache, crop
        self.scale_range, self.whale_bias, self.rare_bias, self.flip = scale_range, whale_bias, rare_bias, flip
        self.photometric_strength, self.train, self.seed = photometric_strength, train, seed
        self.epoch = 0

    def __len__(self):
        return len(self.ids)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __getitem__(self, index: int):
        shot_id = self.ids[index]
        image = self.run.image(shot_id)
        label = load_label(self.run, shot_id, self.cache)
        if not self.train:
            img, (h, w) = pad_to_multiple(image)
            m, _ = pad_to_multiple(label, fill=IGNORE)
            return {"id": shot_id, "image": to_tensor(img), "label": torch.from_numpy(m.astype(np.int64)), "orig_hw": torch.tensor([h, w])}
        rng = np.random.default_rng([self.seed, self.epoch, index])
        H, W = label.shape
        C = self.crop
        s = float(np.exp(rng.uniform(np.log(self.scale_range[0]), np.log(self.scale_range[1]))))
        win = C / s
        whale = (label > 0) & (label != IGNORE)
        rare = (label >= 1) & (label <= 3)
        if whale.any() and rng.random() < self.whale_bias:
            pool = rare if rare.any() and rng.random() < self.rare_bias else whale
            ys, xs = np.nonzero(pool)
            k = rng.integers(len(xs))
            x0 = xs[k] - rng.uniform(0.1, 0.9) * win
            y0 = ys[k] - rng.uniform(0.1, 0.9) * win
        else:
            x0 = rng.uniform(-0.2 * win, W - 0.8 * win)
            y0 = rng.uniform(-0.2 * win, H - 0.8 * win)
        M = warp_params(x0, y0, s)
        img = warp_image(image, M, (C, C))
        m = warp_nearest(label, M, (C, C), IGNORE)
        if self.flip and rng.random() < 0.5:
            img, m = img[:, ::-1], m[:, ::-1]
        img = photometric(np.ascontiguousarray(img), rng, self.photometric_strength)
        return {"id": shot_id, "image": to_tensor(img), "label": torch.from_numpy(np.ascontiguousarray(m).astype(np.int64))}
