"""Train the parts model (model_fine_tuning/parts_seg, unchanged) with rendered whales composited into real water.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.synth_aug.train_rw --rw-mode rw_rs --rw-p 0.5 -- \
        --name parts_seg_r05_rw --init-weights /var/tmp/sat3737-parts-cache/phase1_last.pt --epochs 10 --eval-every 5 \
        --warmup-steps 100 --workers 6

Everything after "--" goes to parts_seg.train unchanged (same recipe, schedule, validation and checkpoint selection on
plain render frames). The only change: parts_seg.train builds its TRAINING dataset through CompositeCropDataset,
which with probability --rw-p (decided per sample from (seed, epoch, index), so deterministic) replaces the render
frame and its part label by a composite (composite.py) BEFORE the parts agent's own crop / flip / photometric code
runs on it (PartsCropDataset.__getitem__, called as is).

Modes: rw_rs = real water + real scale; rs = real scale only (render water kept); rw = real water, render scale.
--rw-index swaps the plate set (e.g. the modelled-sea plates of fw_plates.py) with everything else unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from ..parts_seg import data as pdata
from ..parts_seg.data import PartsCropDataset
from .composite import Compositor, Options

MIN_SIDE = 640
MODES = {"rw_rs": dict(real_water=True, real_scale=True), "rs": dict(real_water=False, real_scale=True),
         "rw": dict(real_water=True, real_scale=False)}

_orig_load_label = pdata.load_label


class _RunProxy:
    """WhaleRun stand-in: serves the composite image while one is pending, everything else from the real run."""

    def __init__(self, run, owner):
        self._run, self._owner = run, owner

    def image(self, shot_id):
        ov = self._owner._override
        if ov is not None and ov[0] == shot_id:
            return ov[1]
        return self._run.image(shot_id)

    def __getattr__(self, name):
        return getattr(self._run, name)


def _load_label(run, shot_id, cache):
    if isinstance(run, _RunProxy):
        ov = run._owner._override
        if ov is not None and ov[0] == shot_id:
            return ov[2]
        run = run._run
    return _orig_load_label(run, shot_id, cache)


pdata.load_label = _load_label            # PartsCropDataset.__getitem__ looks the name up in parts_seg.data at call time


class CompositeCropDataset(PartsCropDataset):
    def __init__(self, run, ids, cache=pdata.DEFAULT_CACHE, *, p: float = 0.5, opts: Options | None = None,
                 index: Path | None = None, **kw):
        super().__init__(run, ids, cache, **kw)
        self.index = index                                # plate index (default: the real-water plates)
        self.base_run = run
        self.run = _RunProxy(run, self)
        self.p = p
        self.opts = opts or Options()
        self._comp = None
        self._override = None

    @property
    def comp(self) -> Compositor:
        if self._comp is None:                           # built lazily in each worker
            self._comp = Compositor(self.opts, index=self.index) if self.index else Compositor(self.opts)
        return self._comp

    def composite(self, index: int):
        """(image, label, info) of the composite for this index and epoch, or None when the sample stays a render."""
        rng = np.random.default_rng([self.seed, self.epoch, index, 7919])
        if not self.train or rng.random() >= self.p:
            return None
        sid = self.ids[index]
        image = self.base_run.image(sid)
        label = _orig_load_label(self.base_run, sid, self.cache)
        mask = self.base_run.mask(sid)
        img, lab, info = self.comp(image, label, mask, rng)
        # the parts crop draws windows up to 1024 px (512 / 0.5) and needs each side >= 0.6 x the window: small
        # composites (a short plate, a zoomed-out render) are padded to 640 px, image reflected, label ignore
        h, w = lab.shape
        ph, pw = max(0, MIN_SIDE - h), max(0, MIN_SIDE - w)
        if ph or pw:
            img = cv2.copyMakeBorder(img, ph // 2, ph - ph // 2, pw // 2, pw - pw // 2, cv2.BORDER_REFLECT)
            lab = cv2.copyMakeBorder(lab, ph // 2, ph - ph // 2, pw // 2, pw - pw // 2, cv2.BORDER_CONSTANT, value=255)
            info["padded"] = [int(ph), int(pw)]
        return img, lab, info

    def __getitem__(self, index: int):
        c = self.composite(index)
        if c is None:
            return super().__getitem__(index)
        self._override = (self.ids[index], c[0], c[1])
        try:
            return super().__getitem__(index)
        finally:
            self._override = None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        k = argv.index("--")
        mine, rest = argv[:k], argv[k + 1:]
    else:
        mine, rest = argv, []
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rw-mode", choices=sorted(MODES), required=True)
    ap.add_argument("--rw-p", type=float, default=0.5)
    ap.add_argument("--rw-no-colour", action="store_true")
    ap.add_argument("--rw-band", type=float, default=1.0)
    ap.add_argument("--rw-index", type=Path, default=None, help="another plate index (e.g. the modelled-sea plates, fw_plates.py)")
    a = ap.parse_args(mine)
    opts = Options(colour_match=not a.rw_no_colour, band=a.rw_band, **MODES[a.rw_mode])

    from ..parts_seg import train as T

    targs = T.parse(rest)
    out = targs.out_root / targs.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "realwater_config.json").write_text(json.dumps({"mode": a.rw_mode, "p": a.rw_p, "options": opts.to_dict(),
                                                           "plate_index": str(a.rw_index) if a.rw_index else "real (plates.INDEX)",
                                                           "train_args": rest}, indent=1))

    def factory(run, ids, cache=pdata.DEFAULT_CACHE, train=True, **kw):
        if train:
            return CompositeCropDataset(run, ids, cache, p=a.rw_p, opts=opts, index=a.rw_index, train=True, **kw)
        return PartsCropDataset(run, ids, cache, train=False, **kw)

    T.PartsCropDataset = factory
    return T.main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
