"""Training data for the whale / water segmentation with real partial labels (S1), real splash pasted into renders
(S2) and real-water composites (S3, synth_aug, imported unchanged).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.data build      (CPU, needs the r04 predictions)
        -> CACHE/real/<day>/<stem>.jpg   the photo at long side 1280 (INTER_AREA, as segment() feeds the model)
           CACHE/real/<day>/<stem>.png   partial label at 1280: 0 background, 1 whale, 255 ignore
           CACHE/real/index.json         per photo the labelled blobs used as crop centres (role, box at 1280)
           CACHE/splash_bank.npz         real splash / wake cut-outs (rgb, soft alpha, reference whale length)
        TRAINING days only; validation-day pixels never enter.
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.data preview --mode s2 --n 24    (scratchpad sheet)

Partial label of a real photo (blob_labels.json role):
  background  blob pixels = 0 (splash, wake, shore, boat, person, pole, drone, bird, other, sure water)
  whale       blob eroded by 2 px = 1; a 3-px band either side of its edge = ignore (whale with confidence 1 only)
  ignore      whale_splash, unclear, unsure whale, unsure water
  elsewhere   ignore, except open water: farther than FAR_PX (40 px at 1280) from every pixel that seg_r04_ft4 (the
              live model, whose blobs are the label units) gives P >= 0.25, which is background.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..common.data import IGNORE, SegCropDataset, photometric, to_tensor, warp_image, warp_nearest, warp_params
from . import common as C
from .predict import load_prob, resize_1280

REAL = C.CACHE / "real"
BANK = C.CACHE / "splash_bank.npz"
FAR_PX = 40
ANY_P = 0.25


def disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def blob_1280(lab2560: np.ndarray, b: int, shape) -> np.ndarray:
    m = cv2.resize((lab2560 == b).astype(np.uint8) * 255, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return m >= 128


def white_pixels(img_bgr: np.ndarray) -> np.ndarray:
    """Bright, unsaturated pixels (white water), grown by 2 px."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return cv2.dilate(((hsv[..., 2] > 190) & (hsv[..., 1] < 45)).astype(np.uint8), disk(2)) > 0


RING = {"in": 3, "out": 12, "erode": 2, "thin": 0, "ring_p": ANY_P, "below_in": 0}
# S4: defaults; S5: --ring-in 5 --erode 1 --out real_s5; S6: + --thin 12 --ring-p 0.05 --below-in 10 --out real_s6


def label_map(day: str, stem: str, blobs: list, shape, strict: bool = False, img=None) -> np.ndarray:
    """strict (S4 labels): an UNSURE splash / wake blob is background on its white pixels only (the rest ignore), an
    unsure water / other blob is ignore (the unsure "wake" calls include dark waterline strips that are low backs);
    and a ring 3-12 px outside every sure whale blob is background where r04 gives P < 0.25 (tight outline)."""
    p = load_prob("r04", day, stem)
    near = cv2.dilate((p >= ANY_P).astype(np.uint8), disk(FAR_PX)) > 0
    lab = np.full(shape, IGNORE, np.uint8); lab[~near] = 0
    L = C.blob_image(day, stem)
    ms = {b["b"]: blob_1280(L, b["b"], shape) for b in blobs}
    wp = white_pixels(img) if strict else None
    for b in blobs:
        if b["role"] == "background":
            if strict and not b["sure"] and b["label"] in ("splash", "wake"):
                lab[ms[b["b"]] & wp] = 0
            elif strict and not b["sure"] and b["label"] in ("water", "other"):
                pass
            else:
                lab[ms[b["b"]]] = 0
    for b in blobs:
        if b["role"] == "ignore":
            lab[cv2.dilate(ms[b["b"]].astype(np.uint8), disk(2)) > 0] = IGNORE
    if strict:
        # tight outline (S4): a ring 3-12 px outside a sure whale blob is background where r04 itself says no whale
        # (P < 0.25); without it the 40-px ignore zone lets the model grow a halo into the water (S1: +12 % area)
        for b in blobs:
            if b["role"] == "whale":
                m = ms[b["b"]].astype(np.uint8)
                ring = (cv2.dilate(m, disk(RING["out"])) > 0) & ~(cv2.dilate(m, disk(RING["in"])) > 0) & (p < RING["ring_p"]) & (lab == IGNORE)
                if RING["below_in"]:          # S6: on the waterline side (below the blob's lower edge in a column) the ring starts later
                    ys, xs = np.nonzero(m); bot = np.full(m.shape[1], -1); np.maximum.at(bot, xs, ys)
                    near = cv2.dilate(m, disk(RING["below_in"])) > 0; ry, rx = np.nonzero(ring & near)
                    under = (bot[rx] >= 0) & (ry > bot[rx]); ring[ry[under], rx[under]] = False
                lab[ring] = 0
    for b in blobs:
        if b["role"] == "whale":
            m = ms[b["b"]].astype(np.uint8)
            lab[cv2.dilate(m, disk(RING["in"] if strict else 3)) > 0] = IGNORE
            core = cv2.erode(m, disk(RING["erode"] if strict else 2)) > 0
            if strict and RING["thin"]:       # S6: thin structures (local thickness < thin px) are positive without erosion
                dt = cv2.distanceTransform(m, cv2.DIST_L2, 5); th = 2 * cv2.dilate(dt, np.ones((15, 15), np.uint8))
                core |= (m > 0) & (th < RING["thin"])
            lab[core] = 1
    return lab


def cmd_build(a) -> None:
    global REAL, BANK
    if a.strict:
        REAL, BANK = C.CACHE / (a.out or "real_strict"), C.CACHE / f"splash_bank_{a.out or 'strict'}.npz"
        RING.update({"in": a.ring_in, "erode": a.erode, "thin": a.thin, "ring_p": a.ring_p, "below_in": a.below_in})
    B = [b for b in C.load_labels() if b["split"] == "train"]
    for b in B:
        assert b["day"] in C.TRAIN
    by = defaultdict(list)
    for b in B:
        by[(b["day"], b["stem"])].append(b)
    index, bank = [], {"rgb": [], "alpha": [], "ref": [], "label": [], "src": []}
    for n, ((day, stem), blobs) in enumerate(sorted(by.items())):
        img = resize_1280(C.photo(day, stem)); shape = img.shape[:2]
        lab = label_map(day, stem, blobs, shape, strict=a.strict, img=img)
        d = REAL / day; d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95]); cv2.imwrite(str(d / f"{stem}.png"), lab)
        rows = C.blob_rows(day)[stem]; L = None
        cen = []
        for b in blobs:
            if b["role"] == "ignore":
                continue
            x0, y0, x1, y1 = (v / 2 for v in b["bbox"])
            cen.append({"b": b["b"], "role": b["role"], "label": b["label"], "box": [x0, y0, x1, y1]})
            if b["label"] in ("splash", "wake"):
                L = C.blob_image(day, stem) if L is None else L
                m = blob_1280(L, b["b"], shape).astype(np.uint8)
                if m.sum() < 30:
                    continue
                md = cv2.dilate(m, disk(2)).astype(np.float32)
                alpha = cv2.GaussianBlur(md, (0, 0), 1.5)
                ys, xs = np.nonzero(alpha > 0.02)
                bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
                main = rows["blobs"][0] if rows["blobs"] else None
                ref = main["long"] / 2 if (main and main["b"] != b["b"]) else max(bx1 - bx0, by1 - by0) * 2.0
                bank["rgb"].append(cv2.cvtColor(img[by0:by1, bx0:bx1], cv2.COLOR_BGR2RGB)); bank["alpha"].append(alpha[by0:by1, bx0:bx1].astype(np.float16))
                bank["ref"].append(float(ref)); bank["label"].append(b["label"]); bank["src"].append(f"{day}|{stem}|{b['b']}")
        if cen:
            index.append({"day": day, "stem": stem, "H": shape[0], "W": shape[1], "blobs": cen})
        if n % 200 == 0:
            print(n, len(by), flush=True)
    (REAL / "index.json").write_text(json.dumps(index))
    obj = np.empty(len(bank["rgb"]), object); obj[:] = bank["rgb"]; al = np.empty(len(bank["alpha"]), object); al[:] = bank["alpha"]
    np.savez(BANK, rgb=obj, alpha=al, ref=np.array(bank["ref"]), label=np.array(bank["label"]), src=np.array(bank["src"]))
    print(f"{len(index)} training photos, {sum(len(r['blobs']) for r in index)} crop centres; splash bank {len(bank['rgb'])} cut-outs "
          f"({sum(l == 'splash' for l in bank['label'])} splash, {sum(l == 'wake' for l in bank['label'])} wake)")


# ------------------------------------------------------------------------------------------------- real crops
class RealBlobCrops(Dataset):
    """512-px crops of the training-day photos at the synthetic recipe's scale jitter (512 / s source px of the
    1280-px photo, s log-uniform 0.5-1.5), centred near a random pixel of a labelled blob."""

    def __init__(self, crop: int = 512, scale_range=(0.5, 1.5), photometric_strength: float = 1.0, seed: int = 0, real_dir: Path | None = None):
        self.real = Path(real_dir) if real_dir else REAL
        self.index = json.loads((self.real / "index.json").read_text())
        self.items = [(i, j) for i, r in enumerate(self.index) for j in range(len(r["blobs"]))]
        self.crop, self.scale_range, self.ph, self.seed, self.epoch = crop, scale_range, photometric_strength, seed, 0

    def __len__(self):
        return len(self.items)

    def set_epoch(self, e):
        self.epoch = e

    def get(self, index: int, rng: np.random.Generator):
        i, j = self.items[index]; r = self.index[i]; b = r["blobs"][j]
        img = cv2.cvtColor(cv2.imread(str(self.real / r["day"] / f"{r['stem']}.jpg")), cv2.COLOR_BGR2RGB)
        lab = cv2.imread(str(self.real / r["day"] / f"{r['stem']}.png"), cv2.IMREAD_UNCHANGED)
        C_ = self.crop; s = float(np.exp(rng.uniform(np.log(self.scale_range[0]), np.log(self.scale_range[1])))); win = C_ / s
        x0, y0, x1, y1 = b["box"]
        px, py = rng.uniform(x0, x1), rng.uniform(y0, y1)
        M = warp_params(px - rng.uniform(0.1, 0.9) * win, py - rng.uniform(0.1, 0.9) * win, s)
        im = warp_image(img, M, (C_, C_)); m = warp_nearest(lab, M, (C_, C_), IGNORE)
        if rng.random() < 0.5:
            im, m = im[:, ::-1], m[:, ::-1]
        im = photometric(np.ascontiguousarray(im), rng, self.ph)
        return {"id": f"real:{r['day']}|{r['stem']}|{b['b']}", "image": to_tensor(im), "mask": torch.from_numpy(np.ascontiguousarray(m).astype(np.int64)),
                "contours": torch.zeros((C_, C_), dtype=torch.uint8)}


# ------------------------------------------------------------------------------------------------ splash paste
class SplashPaster:
    """Pastes 1-3 real splash / wake cut-outs into a 512-px synthetic crop, next to and partly over the rendered whale,
    scaled by (rendered whale length in the crop) / (the real photo's main whale length) x U(0.6, 1.4). Opaque pasted
    pixels (alpha >= 0.5) = background, 0.15 <= alpha < 0.5 = ignore."""

    def __init__(self, path: Path = BANK):
        z = np.load(path, allow_pickle=True)
        self.rgb, self.alpha, self.ref = list(z["rgb"]), list(z["alpha"]), z["ref"]

    def __call__(self, img: np.ndarray, m: np.ndarray, rng: np.random.Generator, s: float):
        img = img.astype(np.float32); m = m.copy(); H, W = m.shape
        whale = m == 1; ys, xs = np.nonzero(whale)
        water = img[m == 0]; wl = float(np.median(water.mean(1))) if len(water) > 100 else None
        n = 1 + int(rng.random() < 0.4) + int(rng.random() < 0.2)
        if len(xs) > 200:
            wlen = max(xs.max() - xs.min(), ys.max() - ys.min())
            edge = whale & ~(cv2.erode(whale.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0); ey, ex = np.nonzero(edge)
        for _ in range(n):
            k = int(rng.integers(len(self.rgb))); rgb = self.rgb[k].astype(np.float32); al = self.alpha[k].astype(np.float32)
            f = (wlen / self.ref[k] if len(xs) > 200 else s) * float(rng.uniform(0.6, 1.4))
            f = float(np.clip(f, 0.2, 3.0)); h, w = al.shape
            f = min(f, 0.7 * W / max(w, h)); nh, nw = max(2, int(round(h * f))), max(2, int(round(w * f)))
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
            al = np.clip(cv2.resize(al, (nw, nh), interpolation=cv2.INTER_LINEAR), 0, 1)
            if wl is not None:                     # bring the cut-out halfway to the crop's water brightness
                ring = (al > 0.02) & (al < 0.3)
                src_l = float(np.median(rgb[ring].mean(1))) if ring.sum() > 10 else wl
                rgb = rgb * float(np.clip(np.sqrt(wl / max(src_l, 1.0)), 0.6, 1.25))
            if len(xs) > 200:
                e = int(rng.integers(len(ex))); cx = ex[e] + rng.uniform(-0.5, 0.5) * nw; cy = ey[e] + rng.uniform(-0.5, 0.5) * nh
            else:
                cx, cy = rng.uniform(0, W), rng.uniform(0, H)
            x0, y0 = int(round(cx - nw / 2)), int(round(cy - nh / 2))
            sx0, sy0, sx1, sy1 = max(0, -x0), max(0, -y0), min(nw, W - x0), min(nh, H - y0)
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            a = al[sy0:sy1, sx0:sx1]; tgt = (slice(y0 + sy0, y0 + sy1), slice(x0 + sx0, x0 + sx1))
            img[tgt] = img[tgt] * (1 - a[..., None]) + rgb[sy0:sy1, sx0:sx1] * a[..., None]
            mm = m[tgt]; mm[a >= 0.5] = 0; mm[(a >= 0.15) & (a < 0.5)] = IGNORE; m[tgt] = mm
        return np.clip(img + 0.5, 0, 255).astype(np.uint8), m


class SynthCrops(SegCropDataset):
    """common.data.SegCropDataset (the seg_r05_ft4 recipe: scale 0.5-1.5, whale bias 0.7, flip, photometric) with two
    optional steps: a real-water composite of the whole frame (synth_aug.composite, prob rw_p) before the crop and
    real splash pasted into the crop (prob paste_p) before flip / photometric."""

    def __init__(self, run, ids, paste_p: float = 0.0, rw_p: float = 0.0, **kw):
        super().__init__(run, ids, **kw)
        self.paste_p, self.rw_p = paste_p, rw_p
        self._paste = None; self._comp = None

    def __getitem__(self, index: int):
        if not self.train:
            return super().__getitem__(index)
        shot_id = self.ids[index]
        rng = np.random.default_rng([self.seed, self.epoch, index])
        rng2 = np.random.default_rng([self.seed, self.epoch, index, 17])
        image = self.run.image(shot_id); mask = self.run.mask(shot_id)
        if self.rw_p > 0 and rng2.random() < self.rw_p:
            if self._comp is None:
                from ..synth_aug.composite import Compositor, Options
                self._comp = Compositor(Options())
            image, mask, _ = self._comp(image, mask.copy(), mask, rng2)
            ph, pw = max(0, 640 - mask.shape[0]), max(0, 640 - mask.shape[1])     # windows reach 1024 px (synth_aug.train_rw MIN_SIDE)
            if ph or pw:
                image = cv2.copyMakeBorder(image, ph // 2, ph - ph // 2, pw // 2, pw - pw // 2, cv2.BORDER_REFLECT)
                mask = cv2.copyMakeBorder(mask, ph // 2, ph - ph // 2, pw // 2, pw - pw // 2, cv2.BORDER_CONSTANT, value=IGNORE)
        H, W = mask.shape; C_ = self.crop
        s = float(np.exp(rng.uniform(np.log(self.scale_range[0]), np.log(self.scale_range[1])))); win = C_ / s
        ys, xs = np.nonzero(mask == 1)
        if len(xs) and rng.random() < self.whale_bias:
            k = rng.integers(len(xs)); x0 = xs[k] - rng.uniform(0.1, 0.9) * win; y0 = ys[k] - rng.uniform(0.1, 0.9) * win
        else:
            x0 = rng.uniform(-0.2 * win, W - 0.8 * win); y0 = rng.uniform(-0.2 * win, H - 0.8 * win)
        M = warp_params(x0, y0, s)
        img = warp_image(image, M, (C_, C_)); m = warp_nearest(mask, M, (C_, C_), IGNORE)
        if self.paste_p > 0 and rng2.random() < self.paste_p:
            if self._paste is None:
                self._paste = SplashPaster()
            img, m = self._paste(img, m, rng2, s)
        if self.flip and rng.random() < 0.5:
            img, m = img[:, ::-1], m[:, ::-1]
        img = photometric(np.ascontiguousarray(img), rng, self.photometric_strength)
        return {"id": shot_id, "image": to_tensor(img), "mask": torch.from_numpy(np.ascontiguousarray(m).astype(np.int64)),
                "contours": torch.zeros((C_, C_), dtype=torch.uint8)}


class Mixed(Dataset):
    """One epoch = len(synth) samples; each is a real crop with probability p_real (decided from seed, epoch, index)."""

    def __init__(self, synth, real, p_real: float, seed: int = 0):
        self.synth, self.real, self.p, self.seed, self.epoch = synth, real, p_real, seed, 0

    def __len__(self):
        return len(self.synth)

    def set_epoch(self, e):
        self.epoch = e; self.synth.set_epoch(e)
        if self.real is not None:
            self.real.set_epoch(e)

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i, 99])
        if self.real is not None and rng.random() < self.p:
            return self.real.get(int(rng.integers(len(self.real))), rng)
        return self.synth[i]


# ---------------------------------------------------------------------------------------------------- preview
def cmd_preview(a) -> None:
    from ..common import paths
    from ..common.data import open_runs
    rng = np.random.default_rng(a.seed); tiles = []
    if a.mode == "real":
        ds = RealBlobCrops(); items = [ds.get(int(rng.integers(len(ds))), rng) for _ in range(a.n)]
    else:
        run = open_runs(C.RUN05, paths.SPLITS_DIR / "run05.json")
        ds = SynthCrops(run, run.ids("train"), paste_p=1.0 if a.mode == "s2" else 0.0, rw_p=1.0 if a.mode == "s3" else 0.0, crop=512, train=True, seed=a.seed)
        items = [ds[int(rng.integers(len(ds)))] for _ in range(a.n)]
    for it in items:
        im = (it["image"].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8); m = it["mask"].numpy()
        ov = im.copy(); ov[m == 1] = (0.5 * ov[m == 1] + [0, 110, 0]).astype(np.uint8); ov[m == 0] = (0.75 * ov[m == 0] + [60, 0, 0]).astype(np.uint8)
        ov[m == IGNORE] = (0.6 * ov[m == IGNORE]).astype(np.uint8)
        t = np.hstack([cv2.resize(im, (300, 300)), cv2.resize(ov, (300, 300))]); tiles.append(cv2.cvtColor(t, cv2.COLOR_RGB2BGR))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles) - len(tiles) % 3, 3)]
    out = C.SHEETS / f"preview_{a.mode}.jpg"; cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85]); print(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build"); p.add_argument("--strict", action="store_true", help="S4 labels -> CACHE/real_strict")
    p.add_argument("--ring-in", type=int, default=3); p.add_argument("--erode", type=int, default=2); p.add_argument("--out", default=None)
    p.add_argument("--thin", type=int, default=0); p.add_argument("--ring-p", type=float, default=ANY_P); p.add_argument("--below-in", type=int, default=0); p.set_defaults(fn=cmd_build)
    p = sub.add_parser("preview"); p.add_argument("--mode", default="real", choices=["real", "s2", "s3"]); p.add_argument("--n", type=int, default=18)
    p.add_argument("--seed", type=int, default=0); p.set_defaults(fn=cmd_preview)
    a = ap.parse_args(argv); a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
