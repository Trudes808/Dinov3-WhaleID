"""Composite a rendered whale (with a feathered band of its own rendered water) into a real-water plate.

    from model_fine_tuning.synth_aug.composite import Compositor, Options
    comp = Compositor(Options(real_water=True, real_scale=True, colour_match=True, band=1.0))
    image, label, info = comp(image, label, mask, rng)          # deterministic for a given np.random.Generator

    image  uint8 RGB (H, W, 3), the render (common.data.WhaleRun.image)
    label  uint8 (H, W): 0 background, 1..5 whale classes (a binary whale label 0 / 1 / 255 works the same), 255 ignore
    mask   uint8 (H, W): the factory mask as WhaleRun.mask gives it (0, 1 visible whale, 255 = class 2 under water)
    -> image uint8 RGB (h, w, 3), label uint8 (h, w) -- the size changes (a plate window / a rescaled frame, at most
       1280 x ~850), info dict (plate, scale, length shares, colour statistics). Build the Compositor lazily in each
       DataLoader worker (it reads a 2.2k-line index and caches 64 plates); ~0.1 s per call on one CPU core.
    Stable interface: Compositor(opts=None, index=INDEX, plate_cache=64), Compositor.__call__(image, label, mask, rng),
    Options fields as below. Training wrapper: synth_aug/train_rw.py (CompositeCropDataset).

Inputs: the render (uint8 RGB HxW), its part label (labels.py: 0 bg, 1..5 parts, 255 ignore) and its factory mask
(0 / 1 visible whale / 255 = class 2, whale just under water; common.data.WhaleRun.mask). Output: a new frame and its
label, transformed identically (image bilinear / area, label nearest).

The patch = the visible whale (mask 1, calf included) with a 2-px rim and the render's white water next to it
(bright pixels connected to the whale within 3 x band: splash, waterline foam), opaque; around it a BAND of the
render's own water (band x max(6 px, 0.06 x the whale's visible length), render pixels) whose opacity falls
linearly from 1 at the whale to 0, so rendered and real water mix there (the feather). The rendered under-water
zone (mask 2) is not carried over (its colour-shifted sheets showed as pink / banded patches): the plate shows there.
Labels: the patch's own labels where the patch is opaque; the under-water zone and calf stay 255 wherever they land
(whatever pixels show there); everything else (band, feather, plate) is background 0.

Ingredients (Options):
  real_water   paste into a real plate (plates.py); off = keep the render's own water
  real_scale   scale so the whale's visible length / frame width follows the real distribution: the length share
               of a random training-day photo's main whale (real_framing.json, all 7.9k photos, x lognormal 0.1),
               clipped to a scale of 0.33-4; without real water the whale centre goes where that photo's whale
               sat in its frame. off = render size.
               With real water the plate is drawn among plates whose real whale was within 30 % of the pasted
               whale's length (else the 50 nearest) and the whale is pasted where that real whale sat (the plate
               came from those rows), so wave size and whale size agree as they did in the real photo.
  colour_match the band's rendered water (and the under-water zone) moved to the plate's water statistics in CIELAB
               (mean, std ratio clipped 0.6-1.6) measured in a ring around the paste; visible whale pixels untouched
  band         band-width multiplier
Whales the render frame cuts (24 % of run05 shots touch an edge) are placed so the cut stays on a frame edge: the
output frame is clipped to the transformed render frame on every side the whale touches.
Negatives (no whale): real water -> a plate, all background; scale only -> unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

import json

from .plates import CACHE, FRAMING, INDEX, WORK_W, load_index

IGNORE = 255


@dataclass
class Options:
    real_water: bool = True
    real_scale: bool = True
    colour_match: bool = True
    band: float = 1.0
    max_up: float = 4.0            # largest scale factor (upscaling a 1280x720 render softens it)
    min_scale: float = 0.33
    jitter: float = 0.1            # lognormal sigma on the target length, and the paste-position jitter (x length)

    def to_dict(self):
        return asdict(self)


def _lab(rgb_f32: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb_f32 / 255.0, cv2.COLOR_RGB2Lab)


def _rgb(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(lab, cv2.COLOR_Lab2RGB) * 255.0


class Compositor:
    def __init__(self, opts: Options | None = None, index: Path = INDEX, plate_cache: int = 64):
        self.opts = opts or Options()
        # the scale-only control needs no plates (it never shows one)
        self.plates = load_index(index) if self.opts.real_water else []
        self.lengths = np.array([p["length_px"] for p in self.plates], np.float64)
        self.hole_w = np.array([(p["hole_box"][2] - p["hole_box"][0]) if p.get("hole_box") else 0 for p in self.plates], np.float64)
        self.framing = json.loads(FRAMING.read_text())
        self._cache: dict[int, np.ndarray] = {}
        self._cache_n = plate_cache

    # -------------------------------------------------------------------------------------------- helpers
    def plate(self, k: int) -> np.ndarray:
        if k not in self._cache:
            if len(self._cache) >= self._cache_n:
                self._cache.pop(next(iter(self._cache)))
            bgr = cv2.imread(str(CACHE / self.plates[k]["file"]), cv2.IMREAD_COLOR)
            self._cache[k] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._cache[k]

    def pick_plate(self, rng, length: float) -> int:
        """a plate whose real whale was about this long (within 30 %, else one of the 50 nearest): the waves
        around the pasted whale then have the size they had around a real whale of that size; among those, plates
        whose mirror-filled hole is at most 1.6 x the pasted length wide (the whale covers most of the fill)"""
        rel = np.abs(np.log(self.lengths / max(length, 1.0)))
        cand = np.nonzero((rel < np.log(1.3)) & (self.hole_w <= 1.6 * length))[0]     # the paste covers the filled hole
        if len(cand) < 20:
            cand = np.nonzero(rel < np.log(1.3))[0]
        if len(cand) == 0:
            cand = np.argsort(rel)[:50]
        return int(cand[rng.integers(len(cand))])

    @staticmethod
    def whale_box(label: np.ndarray):
        vis = (label >= 1) & (label <= 5)
        if not vis.any():
            return None
        ys, xs = np.nonzero(vis)
        return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)

    def patch_alpha(self, image: np.ndarray, label: np.ndarray, mask: np.ndarray, length: float):
        """(alpha float32 HxW in render pixels, band px, keep bool). alpha = 1 on the visible whale (incl. calf),
        a 2-px rim and the render's white water next to it (splash, waterline foam), then falls linearly to 0 over
        the band: the rendered water fades out from the waterline. The under-water zone (mask class 2) is not
        carried over: there the plate shows, and the label stays ignore."""
        band = max(6.0, 0.06 * length) * self.opts.band
        vis = mask == 1
        keep = label != 0
        core = cv2.dilate(vis.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        k = int(round(band))
        L = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)[..., 0].astype(np.float32)        # 8-bit L (0..255)
        ring = cv2.distanceTransform((~vis).astype(np.uint8), cv2.DIST_L2, 3) <= 3 * k
        water = (mask == 0) & ~ring
        if water.sum() < 500:
            water = mask == 0
        if water.sum() > 500:
            lv = float(np.median(L[water]))
            up = float(np.percentile(L[water], 95)) - lv
            white = (cv2.GaussianBlur(L, (0, 0), 1.5) > lv + max(25.0, 1.5 * up)) & ring & (mask == 0)
            if white.any():
                n, cc = cv2.connectedComponents((white | core).astype(np.uint8))
                ids = np.unique(cc[core])
                core |= np.isin(cc, ids[ids > 0])
        dist = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5)
        alpha = np.clip(1.0 - dist / band, 0.0, 1.0).astype(np.float32)
        # the rendered under-water zone is not carried at all (its colour-shifted sheet showed as pink / banded
        # patches below the whale): the plate shows there, softened over 2 px; its label stays ignore
        uw = ((mask == 255) & ~core).astype(np.float32)
        if uw.any():
            alpha *= 1.0 - np.clip(cv2.GaussianBlur(uw, (0, 0), 2.0) * 2.0, 0.0, 1.0) * (~core)
        return alpha, band, keep

    # -------------------------------------------------------------------------------------------- main
    def __call__(self, image: np.ndarray, label: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
        o = self.opts
        H, W = label.shape
        box = self.whale_box(label)
        info = {"opts": o.to_dict()}
        if box is None:                                   # negative render
            if not o.real_water:
                return image, label, dict(info, kind="negative_unchanged")
            k = int(rng.integers(len(self.plates)))
            p = self.plate(k)
            return p.copy(), np.zeros(p.shape[:2], np.uint8), dict(info, kind="negative_plate", plate=k)
        length = box[2] - box[0]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        fr = self.framing[int(rng.integers(len(self.framing)))]
        if o.real_scale:
            target = fr["length_share"] * WORK_W * float(np.exp(rng.normal(0, o.jitter)))
            f = float(np.clip(target / max(length, 1.0), o.min_scale, o.max_up))
        else:
            f = 1.0
        k = self.pick_plate(rng, f * length) if o.real_water else -1
        pr = self.plates[k] if o.real_water else {"day": None, "stem": None}
        # ---- where the whale centre goes (canvas coordinates) and the canvas
        if o.real_water:
            wb = pr["whale_box"]
            canvas_img = self.plate(k)
            # plates that start at the horizon line: drop another 5 % of the frame height (shore strips the
            # horizon finder left)
            skip = 0
            if pr["horizon_top"] > 0 and pr["rect"][1] - pr["horizon_top"] < 0.05 * pr["frame"][1]:
                skip = int(min(max(0.0, pr["horizon_top"] + 0.05 * pr["frame"][1] - pr["rect"][1]), 0.5 * canvas_img.shape[0]))
            canvas_img = canvas_img[skip:]
            wb = [wb[0], wb[1] - skip, wb[2], wb[3] - skip]
            ch, cw = canvas_img.shape[:2]
            px = (wb[0] + wb[2]) / 2 + rng.normal(0, o.jitter) * f * length
            py = (wb[1] + wb[3]) / 2 + rng.normal(0, o.jitter * 0.5) * f * (box[3] - box[1])
            px, py = float(np.clip(px, 0, cw - 1)), float(np.clip(py, 0, ch - 1))
        else:
            # framing of a real photo: the whale centre where that photo's whale sat in its frame
            rx, ry = fr["cx"], fr["cy"]
            cw, ch = int(min(WORK_W, round(W * f))), int(min(720, round(H * f)))
            px, py = rx * cw, ry * ch
        M = np.array([[f, 0, px - f * cx], [0, f, py - f * cy]], np.float64)
        # output window: clip to the transformed render frame (render water) or on the sides the whale touches
        x0, y0, x1, y1 = 0.0, 0.0, float(cw), float(ch)
        fx0, fy0, fx1, fy1 = M[0, 2], M[1, 2], M[0, 2] + f * W, M[1, 2] + f * H
        keep = mask == 1                                  # the visible whale decides which cut edges must stay edges
        if not o.real_water:
            x0, y0, x1, y1 = max(x0, fx0), max(y0, fy0), min(x1, fx1), min(y1, fy1)
        else:
            if keep[:, :2].any():
                x0 = max(x0, fx0)
            if keep[:, -2:].any():
                x1 = min(x1, fx1)
            if keep[:2].any():
                y0 = max(y0, fy0)
            if keep[-2:].any():
                y1 = min(y1, fy1)
        x0, y0 = int(np.ceil(x0)), int(np.ceil(y0))
        x1, y1 = int(np.floor(x1)), int(np.floor(y1))
        if x1 - x0 < 32 or y1 - y0 < 32:
            return image, label, dict(info, kind="fallback_window")
        ow, oh = x1 - x0, y1 - y0
        Mo = M.copy()
        Mo[0, 2] -= x0
        Mo[1, 2] -= y0
        interp = cv2.INTER_AREA if f < 0.8 else cv2.INTER_LINEAR
        info.update(kind="composite" if o.real_water else "rescaled", plate=k, plate_day=pr["day"], plate_stem=pr["stem"],
                    scale=f, render_length_share=length / W, out_length_share=f * length / WORK_W, window=[x0, y0, x1, y1],
                    size=[ow, oh])
        if not o.real_water:
            out = cv2.warpAffine(image, Mo, (ow, oh), flags=interp, borderMode=cv2.BORDER_REFLECT)
            lab = cv2.warpAffine(label, Mo, (ow, oh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=IGNORE)
            return out, lab, info
        # ---- real water: patch from the render
        alpha, band, keep = self.patch_alpha(image, label, mask, length)
        # everything below works on the patch's bounding box only (speed)
        ys, xs = np.nonzero(alpha > 0)
        bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        T = np.array([[1, 0, bx0], [0, 1, by0], [0, 0, 1]], np.float64)
        Mb = (np.vstack([Mo, [0, 0, 1]]) @ T)[:2]
        src = image[by0:by1, bx0:bx1].astype(np.float32)
        a_b, m_b, k_b = alpha[by0:by1, bx0:bx1], mask[by0:by1, bx0:bx1], keep[by0:by1, bx0:bx1]
        if o.colour_match:
            water = (a_b > 0) & ~k_b & (m_b == 0)
            if water.sum() > 50:
                # target statistics: the plate's water in a 20-px ring around where the patch lands (window coords)
                a_c = cv2.warpAffine(a_b, Mb, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.01
                ring = cv2.distanceTransform((~a_c).astype(np.uint8), cv2.DIST_L2, 3)
                ring = (ring > 0) & (ring <= 20)
                if ring.sum() < 500:
                    ring = ~a_c
                pv = canvas_img[y0:y1, x0:x1][ring]
                if len(pv) > 6000:
                    pv = pv[rng.choice(len(pv), 6000, replace=False)]
                plab = _lab(pv.astype(np.float32)[:, None, :])[:, 0]
                rlab = _lab(src)
                rw = rlab[water]
                rw = rw[rw[:, 0] <= np.percentile(rw[:, 0], 90)]               # band water without its foam
                plab = plab[plab[:, 0] <= np.percentile(plab[:, 0], 90)]
                mu_s, sd_s = np.median(rw, 0), 1.4826 * np.median(np.abs(rw - np.median(rw, 0)), 0) + 0.5
                mu_t, sd_t = np.median(plab, 0), 1.4826 * np.median(np.abs(plab - np.median(plab, 0)), 0) + 0.5
                ratio = np.ones(3)
                ratio[0] = np.clip(sd_t[0] / sd_s[0], 0.6, 1.6)               # contrast on L* only; a*, b* shifted
                                                                             # (their spread in water is tiny, a ratio
                                                                             # there tints the under-water zone)
                sel = (m_b == 0) & (a_b > 0)                     # rendered water only; whale and under-water rim untouched
                rlab[sel] = (rlab[sel] - mu_s) * ratio + mu_t
                src = np.where(sel[..., None], np.clip(_rgb(rlab), 0, 255), src).astype(np.float32)
                info["colour"] = {"src_mean": mu_s.round(2).tolist(), "dst_mean": mu_t.round(2).tolist(), "std_ratio": ratio.round(3).tolist()}
        img_w = cv2.warpAffine(src, Mb, (ow, oh), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        a_w = cv2.warpAffine(alpha[by0:by1, bx0:bx1], Mb, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        lab_w = cv2.warpAffine(label[by0:by1, bx0:bx1], Mb, (ow, oh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        plate = canvas_img[y0:y1, x0:x1].astype(np.float32)
        out = plate * (1 - a_w[..., None]) + img_w * a_w[..., None]
        out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        lab_full = cv2.warpAffine(label, Mo, (ow, oh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        lab = np.zeros((oh, ow), np.uint8)
        opaque = a_w >= 0.999
        lab[opaque] = lab_w[opaque]
        lab[lab_full == IGNORE] = IGNORE                 # under-water zone / calf: ignore, whatever pixels show there
        # a whale / ignore label pixel never falls in the feather (the core is the label grown by the band), but if
        # resampling put one there, ignore it rather than call it background
        stray = (lab_w != 0) & ~opaque
        lab[stray] = IGNORE
        info["band_px"] = band
        return out, lab, info
