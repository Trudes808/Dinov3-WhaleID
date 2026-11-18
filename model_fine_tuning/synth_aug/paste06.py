"""DROPPED 2026-11-17: the swap / paste route was dropped after the user's review of the pastes and the rw = sc result.

Paste run06 rendered whales (2560 x 1707) into REAL sea at full resolution, three ways, with exact labels.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.synth_aug.paste06 pool                 # real-whale pool (cached, once)
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.synth_aug.paste06 sheets --datasets \
        ../Whale-Image-Data-Factory/data/renders/r06r_test_r3/dataset [more datasets] --out SCRATCH/realwater
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.synth_aug.paste06 judge --datasets ... --out SCRATCH/realwater --n 60
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.synth_aug.paste06 failures --notes notes.json --out SCRATCH/realwater
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.synth_aug.paste06 r4b --datasets DATASET --out SCRATCH/realwater
        (every usable shot: render | transfer off | transfer on | real; judge grid; labels with the safety zone; 2 full)
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.synth_aug.paste06 partners --datasets D1 D2 ...   (match gate counts)

Every composite uses one real photo of a TRAINING day (never validation / evaluation days) at its own 2560-px size:
  the render's whale goes where the photo's own whale sat, scaled to that whale's box (geometric mean of width and
  height ratio, bottoms aligned = same waterline row), so whale size against wave size is the photo's.
  The render is chosen per real whale by silhouette: same view (fluke / dorsal fin / other: real = the view
  classifier DAY/safe/view_class.json at P >= 0.9, render = its own part shares), best IoU of the bbox-normalised
  masks (render flipped if better), penalised for aspect and scale far from 1.
Methods (same render, same photo, same place):
  A  "band"  : the earlier method: whale + a feathered band of its own rendered water (colour-matched to the photo's
               water in CIELAB), on the photo with the real whale AND its wash removed (mirror fill from the same rows,
               plates._hole / plates.mirror_fill, as for the real-water plates)
  B  "foam"  : whale + only the render's white water (splash, foam, streams) with its coverage as alpha (unmixed from
               the rendered water around it), no rendered water, on the same cleaned photo
  C  "swap"  : whale only, on the photo with ONLY the real whale's mask (+3 px) filled: the real wash, wake and
               splash of the real whale stay around the rendered one
  Cf "swap + own foam + colour": C plus the render's own white water in front of / around the whale (B's foam
               alpha), so streams and splash the render made stay, over the real wash; whale colour as Cc
  Ch "swap + foam in front + colour": C plus the render's white water only where it lies in front of the whale
               (inside the whale's closed silhouette), so splash over the body / streams across the fluke show as
               white water instead of see-through holes; outside the whale the real wash only; colour as Cc
  Cc "swap + colour": C with the whale's dark skin matched in CIELAB (mean / std of L*, a*, b* of the darker 60 %)
               to the real whale that was in the photo; white patterns are left alone (weight falls to 0 above them)
Finishing for all: a waterline feather (alpha ramps 0.5 -> 1 over a few px where the whale meets its under-water
zone), grain added to match the photo's (robust std of the high-pass of the water), a slight blur when the render
was downscaled (it is sharper than the photo then).
Labels (exact, transformed with the whale, nearest): instances (1.. per pasted whale), parts (1 fluke, 2 dorsal fin,
3 pectoral fin, 4 head, 5 body) where the render is opaque. Ignore (255), swap methods (rule of 2026-11-17 13:00):
  (a) the real whale's mask grown by max(4 px, 2 % of its length) where the opaque render does not cover it, and a
      4-px ring round the filled hole;  (b) the real foam put back on top;  (c) other real whales;
  (d) a band below the REAL whale's waterline: per column, UW_BAND (0.15) x the real box height (>= 4 px) below
      the real mask's lowest pixel;  (e) the render's under-water class only where render pixels are pasted.
Everything else -- real water right below and beside the pasted whale, the filled hole -- is background 0
(a model that never sees background under the waterline grows a rim there). Methods A / B: the render's under-water
class is ignore wherever it lands (their rendered water is pasted).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

from ..common import paths
from . import plates as P

CACHE = P.CACHE / "paste06"
POOL = CACHE / "real_pool.json"
POOL_MASKS = CACHE / "real_pool_masks.npz"
NORM = 48
FLUKE_CATS = {"fluke_up", "fluke_down", "fluke_rollout", "fluke_sinking", "tail_slap", "peduncle_throw"}
PART_COL = {1: (230, 60, 60), 2: (60, 220, 60), 3: (240, 200, 40), 4: (70, 120, 255), 5: (200, 200, 200)}


# --------------------------------------------------------------------------------------------------------- real pool
def norm_mask(m: np.ndarray) -> tuple[np.ndarray, float]:
    ys, xs = np.nonzero(m)
    crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
    ar = crop.shape[1] / max(crop.shape[0], 1)
    return cv2.resize(crop * 255, (NORM, NORM), interpolation=cv2.INTER_AREA) > 127, ar


def view_of_vc(v) -> str | None:
    if not v:
        return None
    pf, pd = v
    return "fluke" if pf >= 0.9 else ("dorsal" if pd >= 0.9 else ("other" if pf <= 0.1 and pd <= 0.1 else None))


def build_pool(per_day: int | None = None, seed: int = 0) -> None:
    """main (largest kept) whale of every photo (or up to per_day photos) per training day, with its view and
    bbox-normalised mask. The first pool (2026-11-17 morning) used per_day=120: real_pool_120.json."""
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    recs, masks = [], []
    for day in P.TRAIN_DAYS:
        d = P.DAYS / day
        vc = json.loads((d / "safe" / "view_class.json").read_text())["views"] if (d / "safe" / "view_class.json").exists() else {}
        rows = [r for r in P.day_records(day) if r["main"] is not None]
        rng.shuffle(rows)
        n = 0
        for r in rows:
            if per_day is not None and n >= per_day:
                break
            view = view_of_vc(vc.get(r["main"]["key"]))
            if view is None:
                continue
            m = cv2.imread(str(d / "canon" / f"{r['stem']}_mask.png"), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in r["main"]["bbox"]]
            sc = m.shape[1] / 2560.0
            box = [int(x0 * sc), int(y0 * sc), int(math.ceil(x1 * sc)), int(math.ceil(y1 * sc))]
            sub = m[box[1]:box[3], box[0]:box[2]] > 0
            if sub.sum() < 400:
                continue
            nm, ar = norm_mask(sub)
            recs.append(dict(day=day, stem=r["stem"], key=r["main"]["key"], view=view, box=box, ar=ar, size=list(m.shape[::-1]),
                             n_inst=len([i for i in r["instances"] if not i["drop"]]), length_share=(x1 - x0) / 2560.0))
            masks.append(nm)
            n += 1
        print(day, n, flush=True)
    POOL.write_text(json.dumps(recs))
    np.savez_compressed(POOL_MASKS, masks=np.packbits(np.stack(masks), axis=-1))
    print("pool", len(recs))


def load_pool():
    recs = json.loads(POOL.read_text())
    masks = np.unpackbits(np.load(POOL_MASKS)["masks"], axis=-1)[..., :NORM].astype(bool)
    assert not {r["day"] for r in recs} & P.FORBIDDEN
    return recs, masks


# --------------------------------------------------------------------------------------------------------- renders
def read_png(p: Path) -> np.ndarray:
    from PIL import Image
    a = np.asarray(Image.open(p))
    return a[..., 0] if a.ndim == 3 else a


def load_render(ds: Path, sid: str) -> dict:
    img = cv2.cvtColor(cv2.imread(str(ds / "images" / f"{sid}.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mask = read_png(ds / "masks" / f"{sid}.png")
    inst = read_png(ds / "instances" / f"{sid}.png")
    parts = read_png(ds / "parts" / f"{sid}.png")
    meta = json.loads((ds / "meta" / f"{sid}.json").read_text())
    return dict(ds=str(ds), id=sid, img=img, mask=mask, inst=inst, parts=parts, meta=meta)


def render_view(r: dict) -> str:
    vis = (r["inst"] == 1) & (r["mask"] == 1)
    n = max(int(vis.sum()), 1)
    fl = int(((r["parts"] == 1) & vis).sum()) / n
    fin = int(((r["parts"] == 2) & vis).sum()) / n
    return "fluke" if fl >= 0.15 else ("dorsal" if fin >= 0.01 else "other")


MIN_IOU, MIN_COVER = 0.55, 0.9
UW_BAND = 0.15                            # ignore band below the real whale's waterline, x the real whale's box height           # tightened 2026-11-17 (were 0.4 / none)


def match(r: dict, recs, masks, view=None, length_range=None, n_inst_max=None, exclude=(), min_iou=None, scale_range=(0.5, 2.0),
          min_cover=None, all_ok=False):
    """best real whale for render r: same view, silhouette IoU (flip allowed) >= min_iou, scale within scale_range,
    the scaled render box at least min_cover of the real box in width AND height (so the render hides the real
    whale), aspect and scale penalties. all_ok=True returns the list of every passing (i, flip, iou, f) instead."""
    min_iou = MIN_IOU if min_iou is None else min_iou
    min_cover = MIN_COVER if min_cover is None else min_cover
    passing = []
    vis = (r["inst"] == 1) & (r["mask"] == 1)
    nm, ar = norm_mask(vis)
    ys, xs = np.nonzero(vis)
    rw, rh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    view = view or render_view(r)
    best = None
    for i, q in enumerate(recs):
        if q["view"] != view or q["key"] in exclude:
            continue
        if length_range and not (length_range[0] <= q["length_share"] <= length_range[1]):
            continue
        if n_inst_max is not None and q["n_inst"] > n_inst_max:
            continue
        qw, qh = q["box"][2] - q["box"][0], q["box"][3] - q["box"][1]
        f = math.sqrt(qw * qh / (rw * rh))
        if not (scale_range[0] <= f <= scale_range[1]):
            continue
        if f * rw < min_cover * qw or f * rh < min_cover * qh:
            continue
        for flip in (False, True):
            m = nm[:, ::-1] if flip else nm
            iou = (m & masks[i]).sum() / max((m | masks[i]).sum(), 1)
            if iou < min_iou:
                continue
            s = iou - 0.5 * abs(math.log(ar / q["ar"])) - 0.15 * abs(math.log2(f))
            passing.append((i, flip, iou, f))
            if best is None or s > best[0]:
                best = (s, i, flip, iou, f)
    return passing if all_ok else best


# --------------------------------------------------------------------------------------------------------- photos
def load_photo(q: dict) -> dict:
    d = P.DAYS / q["day"]
    P._check(q["day"])
    img = cv2.cvtColor(cv2.imread(str(d / "images" / f"{q['stem']}.jpg"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    m = cv2.imread(str(d / "canon" / f"{q['stem']}_mask.png"), cv2.IMREAD_GRAYSCALE)
    if m.shape != img.shape[:2]:
        m = cv2.resize(m, img.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    allw = m > 0
    x0, y0, x1, y1 = q["box"]
    pad = 0.03 * max(x1 - x0, y1 - y0) + 3
    inbox = np.zeros_like(allw)
    inbox[max(int(y0 - pad), 0):int(y1 + pad), max(int(x0 - pad), 0):int(x1 + pad)] = True
    this = allw & inbox
    return dict(img=img, this=this, others=allw & ~this, box=q["box"])


def clean_photo(ph: dict, wash: bool) -> np.ndarray:
    """the photo with its whale removed: wash=False -> only the mask (+3 px) filled (C keeps the real wash);
    wash=True -> the generous plates hole (mask + splash / wake / dark parts) filled (A, B)."""
    img = ph["img"]
    if wash:
        lab = P.to_lab(img)
        x0, y0, x1, y1 = ph["box"]
        wlen = x1 - x0
        hole = P._hole(ph["this"], [], lab, int(max(20, 0.10 * wlen)), wlen)
    else:
        hole = cv2.dilate(ph["this"].astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    ys, xs = np.nonzero(hole)
    pad = 8
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, img.shape[0])
    filled, _ = P.mirror_fill(img[y0:y1], hole[y0:y1])          # rows of the hole, full width (sources left / right)
    out = img.copy()
    out[y0:y1] = filled
    ph["hole"] = hole                                            # for the label safety zone (paste)
    return out


def grain_sigma(img: np.ndarray, m: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.2)
    v = hp[m]
    if v.size < 100:
        return 0.0
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


# --------------------------------------------------------------------------------------------------------- paste
def white_alpha(r: dict, band: float) -> tuple[np.ndarray, np.ndarray]:
    """(alpha, foam colour) of the render's own white water near the whales (mask 0 pixels), unmixed from the
    rendered water: alpha = (L - water level) / (white - water level) on the smoothed L*, colour = the pixel with the
    water part removed."""
    img = r["img"].astype(np.float32)
    vis = r["mask"] == 1
    L = cv2.cvtColor(r["img"], cv2.COLOR_RGB2Lab)[..., 0].astype(np.float32) * 100 / 255
    near = cv2.distanceTransform((~vis).astype(np.uint8), cv2.DIST_L2, 3) <= 6 * band
    water = (r["mask"] == 0) & ~near
    if water.sum() < 1000:
        water = r["mask"] == 0
    lv = float(np.median(L[water])) if water.any() else 50.0
    up = float(np.percentile(L[water], 95) - lv) if water.any() else 10.0
    t0 = lv + max(8.0, 1.2 * up)
    white = 92.0
    a = np.clip((cv2.GaussianBlur(L, (0, 0), 1.0) - t0) / max(white - t0, 5.0), 0, 1) * near * (r["mask"] == 0)
    # keep only foam connected to a whale (not distant whitecaps of the rendered sea)
    seed = cv2.dilate(vis.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    n, cc = cv2.connectedComponents(((a > 0.05) | seed).astype(np.uint8))
    ids = np.unique(cc[seed])
    a = a * np.isin(cc, ids[ids > 0])
    bg = cv2.medianBlur(np.clip(np.where(water[..., None], img, np.median(img[water], 0)), 0, 255).astype(np.uint8), 21).astype(np.float32)
    col = np.clip((img - (1 - a[..., None]) * bg) / np.maximum(a[..., None], 0.05), 0, 255)
    return a.astype(np.float32), col


def colour_match_whale(whale_rgb: np.ndarray, wmask: np.ndarray, ref_rgb: np.ndarray, rmask: np.ndarray) -> np.ndarray:
    """dark-skin statistics of the pasted whale -> those of the real whale (L*, a*, b* mean / std of the darker 60 %);
    weight 1 up to the 60th percentile of L*, 0 from 25 L* above it (white patterns untouched)."""
    lab = cv2.cvtColor(whale_rgb.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    rl = cv2.cvtColor(ref_rgb.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    w_px, r_px = lab[wmask], rl[rmask]
    if len(w_px) < 50 or len(r_px) < 50:
        return whale_rgb
    pw, pr = np.percentile(w_px[:, 0], 60), np.percentile(r_px[:, 0], 60)
    ws, rs = w_px[w_px[:, 0] <= pw], r_px[r_px[:, 0] <= pr]
    if rs[:, 0].mean() > 45:            # the real reference is not dark skin (a white pectoral / belly): leave it
        return whale_rgb
    mu_s, sd_s = ws.mean(0), ws.std(0) + 0.5
    mu_t, sd_t = rs.mean(0), rs.std(0) + 0.5
    ratio = np.clip(sd_t / sd_s, 0.5, 2.0)
    moved = (lab - mu_s) * ratio + mu_t
    wt = np.clip((pw + 25 - lab[..., 0]) / 25, 0, 1)[..., None] * wmask[..., None]
    out = lab * (1 - wt) + moved * wt
    return np.clip(cv2.cvtColor(out.astype(np.float32), cv2.COLOR_Lab2RGB) * 255, 0, 255)


def real_waterline_foam(ph: dict) -> np.ndarray:
    """alpha (H, W) of the real photo's own white water inside the real whale's mask along its lower edge (a strip
    max(3, min(0.06 x box height, 18)) px above the mask's bottom in each column): white = L* > max(70, water median
    + 20), chroma < 18. A wider strip (0.25 x height) picked up white skin of real flukes / pectorals."""
    img, this = ph["img"], ph["this"]
    H, W = this.shape
    x0, y0, x1, y1 = ph["box"]
    bh = max(y1 - y0, 4)
    ys, xs = np.nonzero(this)
    if len(xs) == 0:
        return np.zeros((H, W), np.float32)
    bottom = np.full(W, -1)
    np.maximum.at(bottom, xs, ys)
    rows = np.arange(H)[:, None]
    band = max(3.0, min(0.06 * bh, 18.0))                          # a thin strip: the waterline, not the whale's white skin
    low = this & (bottom[None, :] >= 0) & (rows >= bottom[None, :] - band)
    sub = slice(max(int(y0 - bh), 0), min(int(y1 + bh), H))
    lab = cv2.cvtColor(img[sub], cv2.COLOR_RGB2Lab).astype(np.float32)
    L = lab[..., 0] * 100 / 255
    C = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    ring = ~cv2.dilate(this[sub].astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    lv = float(np.median(L[ring])) if ring.any() else 50.0
    white = (L > max(70.0, lv + 20)) & (C < 18)
    a = np.zeros((H, W), np.float32)
    a[sub] = cv2.GaussianBlur((white & low[sub]).astype(np.float32), (0, 0), 1.0)
    return np.clip(a * 1.5, 0, 1) * low


def paste(r: dict, ph: dict, q: dict, flip: bool, method: str, bg: np.ndarray, colour: bool = False, seed: int = 0,
          real_foam: bool = True, safety: bool = True) -> dict:
    """composite at the photo's size. method in band / foam / swap / swapfront.
    real_foam: the photo's own white water at the real whale's waterline goes back on top of the render (label 255).
    safety: every pixel of the real whale's mask (grown) not covered by the opaque render, and a ring round the
    filled hole, is ignore (255): a leftover piece of a real whale must never be labelled background."""
    H, W = bg.shape[:2]
    rng = np.random.default_rng(seed)
    img, mask, inst, parts = r["img"], r["mask"], r["inst"], r["parts"]
    if flip:
        img, mask, inst, parts = img[:, ::-1], mask[:, ::-1], inst[:, ::-1], parts[:, ::-1]
    vis1 = (inst == 1) & (mask == 1)
    ys, xs = np.nonzero(vis1)
    rx0, ry0, rx1, ry1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    qx0, qy0, qx1, qy1 = q["box"]
    f = math.sqrt((qx1 - qx0) * (qy1 - qy0) / ((rx1 - rx0) * (ry1 - ry0)))
    tx = (qx0 + qx1) / 2 - f * (rx0 + rx1) / 2
    ty = qy1 - f * ry1                                                  # bottoms aligned: same waterline row
    M = np.array([[f, 0, tx], [0, f, ty]], np.float64)
    interp = cv2.INTER_AREA if f < 0.8 else cv2.INTER_LINEAR
    vis = (mask == 1)
    length = (rx1 - rx0)
    band = max(6.0, 0.06 * length)
    # alpha in render pixels
    core = cv2.dilate(vis.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    src = img.astype(np.float32)
    if method == "band":
        _, _, keepw = None, None, None
        a_white, _ = white_alpha(dict(r, img=img, mask=mask), band)
        core2 = core | (a_white > 0.3)
        dist = cv2.distanceTransform((~core2).astype(np.uint8), cv2.DIST_L2, 5)
        alpha = np.clip(1.0 - dist / band, 0, 1).astype(np.float32)
        alpha *= 1 - ((mask == 2) & ~core2)                             # rendered under-water sheet not carried
        # colour-match the band water to the photo's water near the target
        water = (alpha > 0) & (mask == 0)
        if water.sum() > 100:
            ring = np.zeros((H, W), bool)
            ring[max(int(qy0 - 0.3 * (qy1 - qy0)), 0):min(int(qy1 + 0.5 * (qy1 - qy0)), H),
                 max(int(qx0 - 0.3 * (qx1 - qx0)), 0):min(int(qx1 + 0.3 * (qx1 - qx0)), W)] = True
            pv = bg[ring].astype(np.float32)
            plab = cv2.cvtColor(pv[:, None] / 255, cv2.COLOR_RGB2Lab)[:, 0]
            rlab = cv2.cvtColor(src / 255, cv2.COLOR_RGB2Lab)
            rw = rlab[water]
            mu_s, sd_s = np.median(rw, 0), 1.4826 * np.median(np.abs(rw - np.median(rw, 0)), 0) + 0.5
            mu_t, sd_t = np.median(plab, 0), 1.4826 * np.median(np.abs(plab - np.median(plab, 0)), 0) + 0.5
            ratio = np.ones(3)
            ratio[0] = np.clip(sd_t[0] / sd_s[0], 0.6, 1.6)
            rlab[water] = (rlab[water] - mu_s) * ratio + mu_t
            src = np.where(water[..., None], np.clip(cv2.cvtColor(rlab, cv2.COLOR_Lab2RGB) * 255, 0, 255), src)
    elif method == "foam":
        a_white, col = white_alpha(dict(r, img=img, mask=mask), band)
        alpha = np.maximum(core.astype(np.float32) * (mask == 1 | core), a_white)
        alpha = np.where(core, 1.0, a_white).astype(np.float32)
        src = np.where(core[..., None], src, col)
    elif method == "swapfront":
        # whale + the render's own white water only where it lies IN FRONT of the whale (inside the whale's closed
        # silhouette: splash over the body, streams between fluke and body), so the whale is not see-through there
        k = int(max(5, 0.04 * length)) | 1
        closed = cv2.morphologyEx(vis.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
        from scipy import ndimage as _nd
        front = (_nd.binary_fill_holes(closed) & ~vis)
        a_white, col = white_alpha(dict(r, img=img, mask=mask), band)
        a_front = np.where(front, np.maximum(a_white, 0.0), 0.0)
        alpha = np.where(core, 1.0, a_front).astype(np.float32)
        src = np.where(core[..., None], src, col)
    else:  # swap
        alpha = core.astype(np.float32)
    # waterline feather: where the whale meets its under-water zone, alpha 0.5 -> 1 over w px inside the whale
    uw = mask == 2
    if uw.any():
        w = max(2.0, 0.004 * length / max(f, 1e-3) * f)
        d_uw = cv2.distanceTransform((~uw).astype(np.uint8), cv2.DIST_L2, 3)
        ramp = np.clip(0.5 + 0.5 * d_uw / w, 0, 1)
        alpha = np.where(core & (d_uw < w), alpha * ramp, alpha)
    # whale finishing: colour match (variant), done in render space on the whale pixels
    if colour:
        real = ph["img"]
        src = np.where(vis[..., None], colour_match_whale(src, vis, real, ph["this"]), src)
    # warp everything into the photo
    out_w = cv2.warpAffine(src, M, (W, H), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a_w = cv2.warpAffine(alpha, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if f < 0.8:                                                         # downscaled render is sharper than the photo
        out_w = cv2.GaussianBlur(out_w, (0, 0), 0.6)
    m_w = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    i_w = cv2.warpAffine(inst, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    p_w = cv2.warpAffine(parts, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # grain: add noise so the whale's high-pass matches the photo's water
    wm = (m_w == 1) & (a_w > 0.99)
    water_ref = np.zeros((H, W), bool)
    water_ref[max(int(qy0 - (qy1 - qy0)), 0):min(int(qy1 + (qy1 - qy0)), H)] = True
    water_ref &= ~(cv2.dilate((a_w > 0).astype(np.uint8) | ph["this"].astype(np.uint8) | ph["others"].astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
    s_p, s_w = grain_sigma(bg, water_ref), grain_sigma(np.clip(out_w, 0, 255).astype(np.uint8), wm)
    add = math.sqrt(max(s_p ** 2 - s_w ** 2, 0.0))
    if add > 0:
        out_w = out_w + rng.normal(0, add, (H, W, 1)).astype(np.float32) * (a_w[..., None] > 0)
    comp = bg.astype(np.float32) * (1 - a_w[..., None]) + out_w * a_w[..., None]
    a_rf = real_waterline_foam(ph) if real_foam and method in ("swap", "swapfront") else np.zeros((H, W), np.float32)
    if a_rf.any():
        comp = comp * (1 - a_rf[..., None]) + ph["img"].astype(np.float32) * a_rf[..., None]
    comp = np.clip(comp + 0.5, 0, 255).astype(np.uint8)
    # labels
    opaque = (a_w >= 0.5) & (m_w == 1)
    lab_inst = np.zeros((H, W), np.uint8)
    lab_parts = np.zeros((H, W), np.uint8)
    lab_inst[opaque] = i_w[opaque]
    lab_parts[opaque] = p_w[opaque]
    ign = np.zeros((H, W), bool)
    ign |= ph["others"]                                                 # (c) other real whales
    if method == "swap" or method == "swapfront":
        ign |= (m_w == 2) & (a_w > 0)                                   # (e) render under-water class only where pasted
    else:
        ign |= (m_w == 2) & ~opaque                                     # A / B: the rendered water around it is pasted
    if safety:
        x0, y0, x1, y1 = q["box"]
        bh = max(y1 - y0, 1)
        g = int(max(4, 0.02 * max(x1 - x0, y1 - y0)))
        grown = cv2.dilate(ph["this"].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))) > 0
        ign |= grown & ~opaque                                          # (a) real whale not hidden by the render
        if "hole" in ph:
            h = ph["hole"].astype(np.uint8)
            ring = (cv2.dilate(h, np.ones((9, 9), np.uint8)) > 0) & ~(cv2.erode(h, np.ones((9, 9), np.uint8)) > 0)
            ign |= ring & ~opaque                                       # (a) 4-px ring round the filled hole
        # (d) a band below the REAL whale's waterline (its submerged body may show): per column of the real mask,
        # UW_BAND x the real box height (at least 4 px) below the mask's lowest pixel
        ys_, xs_ = np.nonzero(ph["this"])
        if len(xs_):
            bottom = np.full(W, -1)
            np.maximum.at(bottom, xs_, ys_)
            depth = max(4.0, UW_BAND * bh)
            rows = np.arange(H)[:, None]
            band = (bottom[None, :] >= 0) & (rows > bottom[None, :]) & (rows <= bottom[None, :] + depth)
            ign |= band & ~opaque
    ign |= (a_rf > 0.2)                                                 # (b) real foam put back on top
    safe = ign
    for lab_ in (lab_inst, lab_parts):
        lab_[ign] = 255
    return dict(img=comp, inst=lab_inst, parts=lab_parts, scale=f, grain_added=add, method=method, colour=colour,
                safety_px=int(safe.sum()), real_foam_px=int((a_rf > 0.2).sum()))


# --------------------------------------------------------------------------------------------------------- sheets
def crop_box(box, W, H, ctx=2.6, aspect=850 / 600):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = max((x1 - x0) * ctx, (y1 - y0) * ctx * aspect, 400)
    h = w / aspect
    if h > H:
        h, w = H, H * aspect
    if w > W:
        w, h = W, W / aspect
    xa = int(np.clip(cx - w / 2, 0, W - w))
    ya = int(np.clip(cy - h / 2, 0, H - h))
    return xa, ya, int(xa + w), int(ya + h)


def tile(img, box, tw=850, th=600, text=None):
    xa, ya, xb, yb = box
    t = cv2.resize(img[ya:yb, xa:xb], (tw, th), interpolation=cv2.INTER_AREA)
    if text:
        cv2.putText(t, text, (8, 26), 0, 0.75, (0, 0, 0), 4)
        cv2.putText(t, text, (8, 26), 0, 0.75, (255, 255, 0), 2)
    return t


def outline(img, inst, parts):
    o = img.copy()
    for k, col in PART_COL.items():
        m = (parts == k).astype(np.uint8)
        if m.any():
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(o, cs, -1, col, 2)
    ig = (inst == 255).astype(np.uint8)
    if ig.any():
        cs, _ = cv2.findContours(ig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(o, cs, -1, (255, 0, 255), 2)
    for k in range(1, 10):
        m = (inst == k).astype(np.uint8)
        if m.any():
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(o, cs, -1, (255, 255, 255) if k == 1 else (0, 255, 255), 1)
    return o


def save(path: Path, rows):
    sheet = np.vstack(rows)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(path, flush=True)


def candidates(datasets: list[Path]):
    out = []
    for ds in datasets:
        for line in open(ds / "manifest.jsonl"):
            m = json.loads(line)
            if m.get("rejected") or m.get("negative"):
                continue
            out.append((ds, m))
    return out


SLOTS = [  # (name, categories, extra filter)
    ("back with dorsal fin", {"back_arch", "surface"}, None),
    ("back with dorsal fin", {"back_arch", "surface", "peduncle"}, None),
    ("fluke up from behind", {"fluke_up"}, None),
    ("fluke up from behind", {"fluke_up", "fluke_down"}, None),
    ("fluke rolling out, water streaming", {"fluke_rollout"}, None),
    ("fluke rolling out, water streaming", {"fluke_rollout", "fluke_sinking"}, None),
    ("pec slap / breach with splash", {"pec_slap"}, None),
    ("pec slap / breach with splash", {"breach", "head_rise", "tail_slap"}, None),
    ("two whales in one frame", None, "two"),
    ("two whales in one frame", None, "two"),
    ("small far whale", None, "small"),
    ("small far whale", None, "small"),
]


def pick_renders(datasets, seed=0, recs=None, masks=None):
    """one render per slot whose whale(s) are inside the render frame and that has a real match (match())."""
    rng = random.Random(seed)
    cands = candidates(datasets)
    rng.shuffle(cands)
    used, picks, used_keys = set(), [], set()
    for name, cats, extra in SLOTS:
        for ds, m in cands:
            if (str(ds), m["id"]) in used:
                continue
            if cats and m["category"] not in cats:
                continue
            if extra == "two" and m.get("n_animals", 1) < 2:
                continue
            vp = m.get("visible_pixels", 0)
            if vp < 3000:
                continue
            r = load_render(ds, m["id"])
            vis = (r["inst"] == 1) & (r["mask"] == 1)
            if vis.sum() < 1500:
                continue
            ys, xs = np.nonzero(vis)
            wshare = (xs.max() - xs.min()) / r["img"].shape[1]
            if extra == "two" and len(set(np.unique(r["inst"][r["mask"] == 1])) - {0}) < 2:
                continue
            if extra == "small" and wshare > 0.12:
                continue
            if extra != "small" and wshare < 0.08:
                continue
            allv = r["mask"] == 1
            if allv[:, :3].any() or allv[:, -3:].any() or allv[:3].any():
                continue                                                  # a whale cut by the render frame
            if extra == "two":
                b2 = (r["inst"] == 2) & allv
                if b2.sum() < 800:
                    continue
                y2, x2 = np.nonzero(b2)
                if not (1 / 3 <= (x2.max() - x2.min()) / max(xs.max() - xs.min(), 1) <= 3):
                    continue
            if recs is not None:
                lr = (0.0, 0.14) if extra == "small" else (0.15, 0.95)
                b = match(r, recs, masks, length_range=lr, n_inst_max=None if extra == "two" else 1, exclude=used_keys)
                if b is None:
                    continue
                used_keys.add(recs[b[1]]["key"])
                r["_match"] = b
            used.add((str(ds), m["id"]))
            picks.append((name, r))
            break
    return picks


def run_sheets(datasets: list[Path], out: Path, seed: int = 0, n_judge: int = 60):
    out.mkdir(parents=True, exist_ok=True)
    if str(out.resolve()).startswith(str(paths.REPO_ROOT.resolve())):
        raise SystemExit("sheets show real photos: write them outside the repository")
    recs, masks = load_pool()
    picks = pick_renders(datasets, seed, recs, masks)
    print("picked", [(n, r["id"]) for n, r in picks], flush=True)
    results = []
    for name, r in picks:
        s_, i, flip, iou, f = r["_match"]
        q = recs[i]
        ph = load_photo(q)
        bg_ab = clean_photo(ph, wash=True)
        bg_c = clean_photo(ph, wash=False)                   # last: ph["hole"] = the swap's hole
        res = {m: paste(r, ph, q, flip, meth, bg, colour=col, seed=seed)
               for m, meth, bg, col in (("A", "band", bg_ab, False), ("B", "foam", bg_ab, False), ("C", "swap", bg_c, False),
                                        ("Cc", "swap", bg_c, True), ("Cf", "foam", bg_c, True), ("Ch", "swapfront", bg_c, True))}
        results.append(dict(name=name, r=r, q=q, ph=ph, res=res, iou=iou, flip=flip))
        print(name, r["id"], "->", q["day"], q["stem"][-20:], f"iou {iou:.2f} scale {res['C']['scale']:.2f}", flush=True)
    return results


def make_all(results, out: Path, best: str = "Cc"):
    tw, th = 850, 600
    # examples: render | paste | real photo of the same view at the same scale (the photo the whale went into)
    rows = []
    for k, x in enumerate(results):
        r, ph, res = x["r"], x["ph"], x["res"][best]
        vis = r["mask"] == 1
        ys, xs = np.nonzero(vis)
        rb = crop_box([xs.min(), ys.min(), xs.max(), ys.max()], *r["img"].shape[1::-1])
        pb = crop_box(x["q"]["box"], *ph["img"].shape[1::-1])
        rows.append(np.hstack([tile(r["img"], rb, tw, th, f"render {Path(r['ds']).parent.name}/{r['id']} {r['meta']['category']}"),
                               tile(res["img"], pb, tw, th, f"paste ({best}) x{res['scale']:.2f}: {x['name']}"),
                               tile(ph["img"], pb, tw, th, f"real {x['q']['day']} (the photo it went into)")]))
    save(out / "paste_examples_01.jpg", rows[:6])
    if len(rows) > 6:
        save(out / "paste_examples_02.jpg", rows[6:12])
    # methods: 6 whales x A, B, C, Cc
    rows = []
    for x in results[::2][:6]:
        pb = crop_box(x["q"]["box"], *x["ph"]["img"].shape[1::-1])
        rows.append(np.hstack([tile(x["res"][m]["img"], pb, 470, 332, lbl) for m, lbl in
                               (("A", "A band"), ("B", "B whale + own foam"), ("C", "C swap, real wash"), ("Cc", "C + colour"),
                                ("Cf", "C + all own foam + colour"), ("Ch", "C + foam in front + colour"))]))
    save(out / "paste_methods.jpg", rows)
    # full frames
    for k, x in enumerate(results[:6]):
        p = out / f"paste_full_{k + 1:02d}.jpg"
        cv2.imwrite(str(p), cv2.cvtColor(x["res"][best]["img"], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(p)
    # labels
    rows = []
    for x in results[:6]:
        res = x["res"][best]
        pb = crop_box(x["q"]["box"], *x["ph"]["img"].shape[1::-1])
        rows.append(np.hstack([tile(res["img"], pb, tw, th, "paste"),
                               tile(outline(res["img"], res["inst"], res["parts"]), pb, tw, th,
                                    "parts: fluke red, fin green, pec yellow, head blue, body grey; ignore magenta")]))
    save(out / "paste_labels.jpg", rows)


def judge(datasets: list[Path], out: Path, n: int = 60, seed: int = 1, method: str = "swapfront", colour: bool = True):
    """n random renders (any category, whale not cut by the render frame), each into its best-matching real photo
    with the chosen method; a numbered grid to judge by eye, and per paste three cached tiles (render | paste |
    real) in CACHE/judge/ for the failures sheet."""
    recs, masks = load_pool()
    rng = random.Random(seed)
    cands = candidates(datasets)
    rng.shuffle(cands)
    jd = CACHE / "judge"
    jd.mkdir(parents=True, exist_ok=True)
    info, tiles, used = [], [], set()
    for ds, m in cands:
        if len(info) >= n:
            break
        if m.get("visible_pixels", 0) < 3000:
            continue
        r = load_render(ds, m["id"])
        vis = (r["inst"] == 1) & (r["mask"] == 1)
        if vis.sum() < 1500 or vis[:, :3].any() or vis[:, -3:].any() or vis[:3].any():
            continue
        b = match(r, recs, masks, exclude=used)
        if b is None:
            continue
        s_, i, flip, iou, f = b
        q = recs[i]
        used.add(q["key"])
        ph = load_photo(q)
        bg = clean_photo(ph, wash=method not in ("swap", "swapfront"))
        res = paste(r, ph, q, flip, method, bg, colour=colour, seed=seed)
        k = len(info)
        ys, xs = np.nonzero(r["mask"] == 1)
        rb = crop_box([xs.min(), ys.min(), xs.max(), ys.max()], *r["img"].shape[1::-1])
        pb = crop_box(q["box"], *ph["img"].shape[1::-1])
        t = [tile(r["img"], rb, 640, 452), tile(res["img"], pb, 640, 452), tile(ph["img"], pb, 640, 452)]
        for j, tt in enumerate(t):
            cv2.imwrite(str(jd / f"{k:03d}_{j}.jpg"), cv2.cvtColor(tt, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        info.append(dict(k=k, ds=str(ds), id=m["id"], category=m["category"], real=q["key"], day=q["day"], iou=round(float(iou), 3),
                         scale=round(res["scale"], 3), flip=flip, grain_added=round(res["grain_added"], 2)))
        g = cv2.resize(t[1], (400, 282))
        cv2.putText(g, f"{k} {m['category']}", (6, 24), 0, 0.7, (0, 0, 0), 4)
        cv2.putText(g, f"{k} {m['category']}", (6, 24), 0, 0.7, (255, 255, 0), 2)
        tiles.append(g)
        print(k, m["id"], m["category"], q["day"], f"iou {iou:.2f} x{res['scale']:.2f}", flush=True)
    (jd / "info.json").write_text(json.dumps(info, indent=1))
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    save(out / "paste_judge_60.jpg", rows)


def failures(out: Path, notes: dict):
    jd = CACHE / "judge"
    info = {x["k"]: x for x in json.loads((jd / "info.json").read_text())}
    rows = []
    for k, note in notes.items():
        k = int(k)
        t = [cv2.cvtColor(cv2.imread(str(jd / f"{k:03d}_{j}.jpg")), cv2.COLOR_BGR2RGB) for j in range(3)]
        row = np.hstack(t)
        txt = f"#{k} {info[k]['category']}: {note}"
        cv2.putText(row, txt, (8, 440), 0, 0.8, (0, 0, 0), 5)
        cv2.putText(row, txt, (8, 440), 0, 0.8, (255, 255, 0), 2)
        rows.append(row)
    save(out / "paste_failures.jpg", rows)


def usable(ds: Path):
    """non-rejected, non-negative shots whose whales are all inside the render frame"""
    out = []
    for line in open(ds / "manifest.jsonl"):
        m = json.loads(line)
        if m.get("rejected") or m.get("negative") or m.get("visible_pixels", 0) < 3000:
            continue
        r = load_render(ds, m["id"])
        allv = r["mask"] == 1
        if ((r["inst"] == 1) & allv).sum() < 1500 or allv[:, :3].any() or allv[:, -3:].any() or allv[:3].any():
            continue
        out.append(r)
    return out


def partner_counts(datasets: list[Path]) -> dict:
    """how many renders find a partner under the tight gate, and how many pool whales are someone's partner"""
    recs, masks = load_pool()
    res = {}
    for gate in ("loose", "tight"):
        kw = dict(min_iou=0.4, min_cover=0.0) if gate == "loose" else {}
        n_r, n_ok, hit = 0, 0, set()
        for ds in datasets:
            for r in usable(ds):
                n_r += 1
                ps = match(r, recs, masks, all_ok=True, **kw)
                if ps:
                    n_ok += 1
                    hit |= {p[0] for p in ps}
        res[gate] = {"renders": n_r, "renders_with_partner": n_ok, "pool": len(recs), "pool_whales_with_partner": len(hit)}
        print(gate, res[gate], flush=True)
    return res


def r4b(ds: Path, out: Path, seed: int = 0):
    """paste_r4b_examples / _judge / _labels / _full_01, _02 for one dataset (every usable shot)."""
    recs, masks = load_pool()
    rs = usable(ds)
    used, rows, judge_tiles, lab_rows, done, nomatch = set(), [], [], [], [], []
    for r in rs:
        b = match(r, recs, masks, exclude=used)
        if b is None:
            nomatch.append(r["id"])
            continue
        _, i, flip, iou, f = b
        q = recs[i]
        used.add(q["key"])
        ph = load_photo(q)
        bg = clean_photo(ph, wash=False)
        off = paste(r, ph, q, flip, "swapfront", bg, colour=False, seed=seed)
        on = paste(r, ph, q, flip, "swapfront", bg, colour=True, seed=seed)
        k = len(done)
        vis = r["mask"] == 1
        ys, xs = np.nonzero(vis)
        rb = crop_box([xs.min(), ys.min(), xs.max(), ys.max()], *r["img"].shape[1::-1])
        pb = crop_box(q["box"], *ph["img"].shape[1::-1])
        tw, th = 640, 452
        rows.append(np.hstack([tile(r["img"], rb, tw, th, f"#{k} render {r['id']} {r['meta']['category']}"),
                               tile(off["img"], pb, tw, th, f"paste, transfer OFF x{f:.2f} iou {iou:.2f}"),
                               tile(on["img"], pb, tw, th, "paste, transfer ON"),
                               tile(ph["img"], pb, tw, th, f"real {q['day']}")]))
        for v, lbl in ((off, "off"), (on, "on")):
            t = cv2.resize(tile(v["img"], pb, tw, th), (400, 282))
            cv2.putText(t, f"{k}{lbl[1]} {r['meta']['category']} {lbl}", (6, 24), 0, 0.65, (0, 0, 0), 4)
            cv2.putText(t, f"{k}{lbl[1]} {r['meta']['category']} {lbl}", (6, 24), 0, 0.65, (255, 255, 0), 2)
            judge_tiles.append(t)
        ov = outline(off["img"], off["inst"], off["parts"])
        ig = (off["inst"] == 255)
        ov[ig] = (0.55 * ov[ig] + 0.45 * np.array([255, 0, 255])).astype(np.uint8)
        lab_rows.append(np.hstack([tile(off["img"], pb, 850, 600, f"#{k} paste"),
                                   tile(ov, pb, 850, 600, f"labels; magenta fill = ignore ({off['safety_px']} px ignore, {off['real_foam_px']} px real foam)")]))
        done.append(dict(k=k, id=r["id"], category=r["meta"]["category"], real=q["key"], iou=round(float(iou), 3), scale=round(f, 3),
                         safety_px=off["safety_px"], real_foam_px=off["real_foam_px"], full=off["img"]))
        print(k, r["id"], r["meta"]["category"], q["day"], f"iou {iou:.2f} x{f:.2f}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    save(out / "paste_r4b_examples.jpg", rows)
    while len(judge_tiles) % 6:
        judge_tiles.append(np.zeros_like(judge_tiles[0]))
    save(out / "paste_r4b_judge.jpg", [np.hstack(judge_tiles[i:i + 6]) for i in range(0, len(judge_tiles), 6)])
    save(out / "paste_r4b_labels.jpg", lab_rows[:8])
    big = sorted(done, key=lambda d: -d["scale"])
    for j, d in enumerate(done[:2] if len(done) < 2 else [done[0], done[len(done) // 2]]):
        p = out / f"paste_r4b_full_{j + 1:02d}.jpg"
        cv2.imwrite(str(p), cv2.cvtColor(d["full"], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(p)
    info = [{k: v for k, v in d.items() if k != "full"} for d in done]
    (CACHE / "r4b_info.json").write_text(json.dumps({"usable": len(rs), "pasted": info, "no_partner": nomatch}, indent=1))
    print("usable", len(rs), "pasted", len(done), "no partner", nomatch)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["pool", "sheets", "judge", "failures", "r4b", "partners"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--notes", type=Path, default=None, help="failures: json {judge number: note}")
    ap.add_argument("--datasets", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--best", default="Ch")
    a = ap.parse_args(argv)
    if a.cmd == "pool":
        build_pool()
        return 0
    if a.cmd == "r4b":
        r4b(a.datasets[0], a.out, a.seed)
        return 0
    if a.cmd == "partners":
        print(json.dumps(partner_counts(a.datasets), indent=1))
        return 0
    if a.cmd == "judge":
        a.out.mkdir(parents=True, exist_ok=True)
        judge(a.datasets, a.out, a.n, a.seed + 1)
        return 0
    if a.cmd == "failures":
        failures(a.out, json.loads(a.notes.read_text()))
        return 0
    results = run_sheets(a.datasets, a.out, a.seed)
    make_all(results, a.out, a.best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
