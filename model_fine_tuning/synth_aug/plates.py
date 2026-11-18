"""Real-water plates from the group's own photographs (TRAINING days only).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.synth_aug.plates build [--per-day 150] [--workers 6]
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.synth_aug.plates sheet --out SCRATCH/plates_sheet.jpg

A plate is a rectangle of pure water cut from a real photo at "work scale" (the photo resized to 1280 px wide, so the
frame width = 1280 px exactly as a 1280x720 render), taken from the rows the photo's main whale sat in, with the real
whale (and its splash / wake) removed:

  1. whale pixels = DAY/canon/<stem>_mask.png > 0, plus the whole boxes of the dropped not-a-whale instances; the hole = those, dilated by max(10 px, 0.10 x the main whale's box length), plus bright water (foam,
     splash, wake) and dark blobs (whale parts the mask missed) connected to them within 0.6 x the whale length
     sideways / 3 x the radius up and down (quarter scale, smoothed so single wave crests / troughs do not count),
     dilated again by 4 px.
  2. "bad" = not water: above the detected horizon (experiments/realism_gap/horizon.py) + 3 % of the frame, with a
     second pass below it for a shore line (land between sky and sea); dark blobs (smoothed L* well below the local
     water level: whales the mask missed, boats, people); colour outliers (a* / b* far from the water's: land,
     vegetation, buoys, life jackets); smooth bright areas and whole smooth rows (sky / haze); large bright blobs
     (splash). Bad blobs touching the hole and lying wholly in the whale's rows join the hole instead (wake, the
     rest of the whale just under the surface).
  3. rectangle: the widest bad-free column interval containing the hole's box over its rows, then rows grown up and
     down while they stay bad-free (16-px blocks); minimum max(384 px, 1.4 x whale length) wide and
     max(160 px, 1.6 x whale box height) tall.
  4. the hole is filled column-mirrored from the water left and right of it in the SAME rows (perspective kept),
     the two sides split in proportion to the water available (a side may repeat, but must hold >= 0.3 of the width
     it fills; holes wider than 0.85 of the plate are refused), meeting with a 24-px cross-fade; the fill's local mean
     is pulled to what the surroundings predict (Gaussian 24 px), blended in with a 4-px feather.
  5. refused if, after an opening that removes sun glitter and small whitecaps, > 2 % of the plate is white water.

Photos with status "no_whale" (4 on the training days) give a plate below the horizon with the nearest-in-time whale
photo of the same photographer folder supplying the whale row and length (if within 120 s and the same focal length).

Index (jsonl, one plate per line): day, stem, camera, focal_mm, time, kind, rect (x0, y0, x1, y1 work px), plate size,
whale box in plate coords, whale length share (box width / frame width; the realism_gap definition), hole box,
water L*a*b* statistics. Plates + index live in /var/tmp/sat3737-parts-cache/realwater/ (never in the repository).
The validation and evaluation days are refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from ..common import paths

DAYS = paths.EXPERIMENTS_DIR / "days_all"
CACHE = Path("/var/tmp/sat3737-parts-cache/realwater")
PLATES = CACHE / "plates"
INDEX = CACHE / "plates_index.jsonl"
FRAMING = CACHE / "real_framing.json"
WORK_W = 1280
BLOCK = 16
SPLIT = json.loads((paths.SPLITS_DIR / "days.json").read_text())
TRAIN_DAYS = list(SPLIT["train"])
FORBIDDEN = set(SPLIT["eval"]) | set(SPLIT["val"])

sys.path.insert(0, str(paths.EXPERIMENTS_DIR / "realism_gap"))
import horizon  # noqa: E402  (experiments/realism_gap/horizon.py, read only)


def _check(day: str) -> None:
    if day in FORBIDDEN or day not in TRAIN_DAYS:
        raise RuntimeError(f"plates come from training days only, not {day}")


def to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


# ------------------------------------------------------------------------------------------------------ loading
def day_records(day: str) -> list[dict]:
    _check(day)
    d = DAYS / day
    drop = set(json.loads((d / "safe" / "not_whale.json").read_text()).get("drop", [])) if (d / "safe" / "not_whale.json").exists() else set()
    ts = json.loads((d / "timestamps.json").read_text()) if (d / "timestamps.json").exists() else {}
    status = {}
    for line in open(d / "canon" / "points.jsonl"):
        if line.strip():
            p = json.loads(line)
            status[Path(p["image"]).stem] = p["status"]
    recs = []
    seen = set()
    for line in open(d / "parts_hr" / "index.jsonl"):
        if not line.strip():
            continue
        r = json.loads(line)
        inst = [dict(key=i["key"], bbox=[float(v) for v in i["bbox_2560"]], drop=i["key"] in drop) for i in r.get("instances", []) if i.get("bbox_2560")]
        keep = [i for i in inst if not i["drop"]]
        main = max(keep, key=lambda i: (i["bbox"][2] - i["bbox"][0]) * (i["bbox"][3] - i["bbox"][1])) if keep else None
        t = ts.get(r["stem"], {})
        recs.append(dict(day=day, stem=r["stem"], instances=inst, main=main, status=status.get(r["stem"], "ok"), time=t.get("time"),
                         camera=t.get("camera"), focal_mm=t.get("focal_mm"), folder=t.get("folder")))
        seen.add(r["stem"])
    for stem, st in status.items():                     # no_whale photos are not in parts_hr
        if st == "no_whale" and stem not in seen:
            t = ts.get(stem, {})
            recs.append(dict(day=day, stem=stem, instances=[], main=None, status=st, time=t.get("time"), camera=t.get("camera"),
                             focal_mm=t.get("focal_mm"), folder=t.get("folder")))
    return recs


# ------------------------------------------------------------------------------------------------------ analysis
def _anomalies(lab: np.ndarray, hole: np.ndarray, rows: tuple[int, int]) -> tuple[np.ndarray, dict]:
    """bool map of not-water pixels (hole excluded from the statistics), at work scale."""
    H, W = hole.shape
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    ok = ~hole
    # water reference: the whale's rows (the plate's own rows), outside the hole
    r0, r1 = rows
    ref = np.zeros_like(ok)
    ref[r0:r1] = True
    ref &= ok
    if ref.sum() < 2000:
        ref = ok
    a_med, b_med = float(np.median(A[ref])), float(np.median(B[ref]))
    a_mad = float(np.median(np.abs(A[ref] - a_med))) + 1.0
    b_mad = float(np.median(np.abs(B[ref] - b_med))) + 1.0
    # quarter-scale maps
    q = 4
    h4, w4 = H // q, W // q
    Ls = cv2.resize(L, (w4, h4), interpolation=cv2.INTER_AREA)
    As = cv2.resize(A, (w4, h4), interpolation=cv2.INTER_AREA)
    Bs = cv2.resize(B, (w4, h4), interpolation=cv2.INTER_AREA)
    oks = cv2.resize(ok.astype(np.uint8), (w4, h4), interpolation=cv2.INTER_NEAREST) > 0
    # local water level: masked large box mean (hole left out)
    k = 41
    num = cv2.boxFilter(np.where(oks, Ls, 0).astype(np.float32), -1, (k, k), normalize=False)
    den = cv2.boxFilter(oks.astype(np.float32), -1, (k, k), normalize=False)
    bg = num / np.maximum(den, 1)
    Lsm = cv2.GaussianBlur(Ls, (0, 0), 1.5)
    resid = (Ls - bg)[oks]
    sd = float(np.std(resid)) + 1.0
    dark = (Lsm < bg - max(7.0, 2.0 * sd)) & oks
    splash = (Lsm > bg + max(15.0, 3.0 * sd)) & oks
    Asm, Bsm = cv2.GaussianBlur(As, (0, 0), 1.5), cv2.GaussianBlur(Bs, (0, 0), 1.5)
    colour = ((np.abs(Asm - a_med) > max(8.0, 5 * a_mad)) | ((Bsm - b_med) > max(10.0, 5 * b_mad))) & oks
    # texture: water carries ripples; sky / haze / land at a distance is smooth AND brighter than the water
    tex = cv2.GaussianBlur(np.abs(Ls - cv2.GaussianBlur(Ls, (0, 0), 1.0)), (0, 0), 3.0)
    tex_ref = float(np.median(tex[cv2.resize(ref.astype(np.uint8), (w4, h4), interpolation=cv2.INTER_NEAREST) > 0])) if ref.any() else float(np.median(tex))
    ref4 = cv2.resize(ref.astype(np.uint8), (w4, h4), interpolation=cv2.INTER_NEAREST) > 0
    L_ref = float(np.median(Ls[ref4])) if ref4.any() else float(np.median(Ls))
    texs = cv2.GaussianBlur(tex, (0, 0), 4.0)
    # sky / haze / distant land the horizon finder missed: smooth (texture < 0.35 x the whale rows') and brighter
    # than the whale rows' water; also whole rows that are this smooth (the horizon band)
    smooth_bright = (texs < 0.35 * tex_ref) & (Lsm > L_ref + 0.5 * sd) & oks
    rowtex = np.median(texs, 1)
    smooth_bright[rowtex < 0.3 * tex_ref] = True
    bad4 = np.zeros((h4, w4), bool)
    for m, min_area in ((dark, 25), (colour, 25), (smooth_bright, 150), (splash, 120)):
        lab_, n = ndimage.label(m)
        if n:
            sizes = ndimage.sum(np.ones_like(m, np.float32), lab_, index=np.arange(1, n + 1))
            big = np.nonzero(sizes >= min_area)[0] + 1
            bad4 |= np.isin(lab_, big)
    bad4 = cv2.dilate(bad4.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    bad = cv2.resize(bad4.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    return bad, {"a_med": a_med, "b_med": b_med, "sd_L4": sd, "tex_ref": tex_ref}


def _hole(mask: np.ndarray, boxes: list, lab: np.ndarray, radius: int, wlen: float) -> np.ndarray:
    H, W = mask.shape
    m = mask.copy()
    for (x0, y0, x1, y1) in boxes:
        m[max(int(y0), 0):min(int(np.ceil(y1)), H), max(int(x0), 0):min(int(np.ceil(x1)), W)] = True
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    core = cv2.dilate(m.astype(np.uint8), k) > 0
    # bright water (foam / splash / wake, which runs along the waterline) and dark blobs (whale parts the mask
    # missed) connected to the whale: quarter scale, smoothed (single wave troughs / crests vanish), searched in a
    # ring 0.6 x the whale length sideways, 3 x radius up and down
    q = 4
    h4, w4 = H // q, W // q
    L = cv2.GaussianBlur(cv2.resize(lab[..., 0], (w4, h4), interpolation=cv2.INTER_AREA), (0, 0), 1.5)
    m4 = cv2.resize(m.astype(np.uint8), (w4, h4), interpolation=cv2.INTER_AREA) > 0
    c4 = cv2.resize(core.astype(np.uint8), (w4, h4), interpolation=cv2.INTER_NEAREST) > 0
    rx, ry = int(0.6 * wlen + radius) // q, 3 * radius // q
    ring = cv2.dilate(m4.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rx + 1, 2 * ry + 1))) > 0
    water = ~c4
    lvl = float(np.median(L[water])) if water.any() else 50.0
    up = float(np.percentile(L[water], 90) - lvl) if water.any() else 10.0
    dn = float(lvl - np.percentile(L[water], 10)) if water.any() else 10.0
    bright = (L > lvl + max(6.0, 1.5 * up)) & ring
    dark = (L < lvl - max(8.0, 2.0 * dn)) & ring
    g4 = c4.copy()
    for extra in (bright, dark):
        lab_, n = ndimage.label(extra | c4)
        if n:
            keep = np.unique(lab_[c4])
            g4 |= np.isin(lab_, keep[keep > 0])
    grown = core | (cv2.resize(g4.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0)
    return cv2.dilate(grown.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0


def _fold(idx: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """reflect integer indices into [lo, hi) (triangle wave)."""
    n = hi - lo
    if n <= 0:
        return np.full_like(idx, lo)
    p = (idx - lo) % (2 * n)
    return lo + np.where(p < n, p, 2 * n - 1 - p)


def mirror_fill(img: np.ndarray, hole: np.ndarray) -> tuple[np.ndarray, bool]:
    """Fill every connected hole column-mirrored from the water left and right of it in the same rows."""
    out = img.astype(np.float32).copy()
    H, W = hole.shape
    lab_, n = ndimage.label(hole)
    ok = True
    for sl in ndimage.find_objects(lab_):
        ya, yb = sl[0].start, sl[0].stop
        xa, xb = max(sl[1].start - 2, 0), min(sl[1].stop + 2, W)
        w = xb - xa
        xs = np.arange(xa, xb)
        # left mirror covers [xa, xa + wl), right mirror the rest, split in proportion to the water available on
        # each side; a side may repeat (folded mirror) but must hold at least 0.3 of the width it fills
        if w > 0.85 * W:
            return out.astype(np.uint8), False
        avl, avr = xa, W - xb
        wl = float(w * avl / max(avl + avr, 1))
        if (avl and avl < 0.3 * wl) or (avr and avr < 0.3 * (w - wl)):
            return out.astype(np.uint8), False
        left_ok, right_ok = wl >= 1, w - wl >= 1
        srcL = _fold(2 * xa - 1 - xs, 0, xa) if left_ok else None
        srcR = _fold(2 * xb - 1 - xs, xb, W) if right_ok else None
        if left_ok and right_ok:
            t = np.clip((xs - (xa + wl)) / 24.0 + 0.5, 0, 1)           # 24-px cross-fade where the halves meet
        else:
            t = np.zeros(w) if left_ok else np.ones(w)
        rows = slice(ya, yb)
        fill = np.zeros((yb - ya, w, 3), np.float32)
        wsum = np.zeros((yb - ya, w, 1), np.float32)
        for src, wt in ((srcL, 1 - t), (srcR, t)):
            if src is None:
                continue
            good = ~hole[rows][:, src]                                    # never copy another hole
            ww = wt[None, :] * good
            fill += out[rows][:, src] * ww[..., None]
            wsum += ww[..., None]
        bad = wsum[..., 0] < 1e-3
        region = lab_[rows, xa:xb] > 0
        if (bad & region).any():
            ok = False
        fill = fill / np.maximum(wsum, 1e-3)
        reg = out[rows, xa:xb]
        reg[region] = fill[region]
        out[rows, xa:xb] = reg
    # low-frequency correction: the fill's local mean is pulled to what the surroundings predict (sigma 24 px)
    wgt = (~hole).astype(np.float32)
    num = cv2.GaussianBlur(out * wgt[..., None], (0, 0), 24)
    den = cv2.GaussianBlur(wgt, (0, 0), 24)[..., None]
    expect = num / np.maximum(den, 1e-3)
    have = cv2.GaussianBlur(out, (0, 0), 24)
    corr = np.where((den > 0.05) & hole[..., None], expect - have, 0)
    out = out + cv2.GaussianBlur(corr.astype(np.float32), (0, 0), 6)
    # feather the hole border
    a = cv2.GaussianBlur((cv2.dilate(hole.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0).astype(np.float32), (0, 0), 2.0)
    a = np.maximum(a, hole.astype(np.float32))[..., None]
    res = img.astype(np.float32) * (1 - a) + out * a
    return np.clip(res + 0.5, 0, 255).astype(np.uint8), ok


def best_rect(bad: np.ndarray, hole: np.ndarray, wbox: list[float], min_w: int, min_h: int, top: int):
    """(x0, y0, x1, y1) bad-free rectangle containing the whale box (block grid), or None."""
    H, W = bad.shape
    hb, wb = H // BLOCK, W // BLOCK
    blk = bad[: hb * BLOCK, : wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK).mean((1, 3)) > 0.02
    blk[: int(np.ceil(top / BLOCK))] = True
    x0, y0, x1, y1 = wbox
    bx0, bx1 = int(x0 // BLOCK), min(int(np.ceil(x1 / BLOCK)), wb)
    by0, by1 = int(y0 // BLOCK), min(int(np.ceil(y1 / BLOCK)), hb)
    if blk[by0:by1, bx0:bx1].any():
        return None
    rowbad = blk[by0:by1].any(0)
    a = bx0
    while a > 0 and not rowbad[a - 1]:
        a -= 1
    b = bx1
    while b < wb and not rowbad[b]:
        b += 1
    if (b - a) * BLOCK < min_w:
        return None
    r0, r1 = by0, by1
    while r0 > 0 and not blk[r0 - 1, a:b].any():
        r0 -= 1
    while r1 < hb and not blk[r1, a:b].any():
        r1 += 1
    if (r1 - r0) * BLOCK < min_h:
        return None
    return a * BLOCK, r0 * BLOCK, b * BLOCK, r1 * BLOCK


def make_plate(rec: dict, partner: dict | None = None, debug: bool = False):
    """-> (plate rgb, index record) or (None, reason)."""
    d = DAYS / rec["day"]
    bgr = cv2.imread(str(d / "images" / f"{rec['stem']}.jpg"), cv2.IMREAD_COLOR)
    if bgr is None:
        return None, "no image"
    H0, W0 = bgr.shape[:2]
    s = WORK_W / W0
    Hs = int(round(H0 * s))
    rgb = cv2.cvtColor(cv2.resize(bgr, (WORK_W, Hs), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    mp = d / "canon" / f"{rec['stem']}_mask.png"
    m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp.exists() else None
    mask = (cv2.resize(m, (WORK_W, Hs), interpolation=cv2.INTER_NEAREST) > 0) if m is not None else np.zeros((Hs, WORK_W), bool)
    sc = s * W0 / 2560.0                      # 2560-px coordinates -> work
    # whales: their mask pixels (dilated below); not-a-whale instances (boats, birds, people): their whole box
    boxes = [[v * sc for v in i["bbox"]] for i in rec["instances"] if i["drop"]]
    src = rec if rec["main"] is not None else partner
    if src is None or src["main"] is None:
        return None, "no whale row"
    wb = [v * sc for v in src["main"]["bbox"]]
    length_share = (src["main"]["bbox"][2] - src["main"]["bbox"][0]) / 2560.0
    wlen = wb[2] - wb[0]
    lab = to_lab(rgb)
    radius = int(max(10, 0.10 * wlen))
    hole = _hole(mask, boxes, lab, radius, wlen) if (mask.any() or boxes) else np.zeros_like(mask)
    hy, _ = horizon.detect(lab, hole, whale_box=wb if rec["main"] is not None else None)
    top = 0 if hy is None else int(np.clip(np.max(hy) + 0.03 * Hs, 0, Hs))
    for _ in range(2):                                   # a shore line below the sky line (land between sky and sea)
        inv = hole.copy()
        inv[:top] = True
        hy2, _ = horizon.detect(lab, inv, whale_box=wb if rec["main"] is not None else None)
        if hy2 is None or np.max(hy2) <= top:
            break
        top = int(np.clip(np.max(hy2) + 0.03 * Hs, 0, Hs))
    rows = (int(max(wb[1] - wlen * 0.3, top)), int(min(wb[3] + wlen * 0.3, Hs)))
    bad, st = _anomalies(lab, hole, rows)
    if rec["main"] is not None and bad.any():
        lab_b, nb = ndimage.label(bad)
        near = cv2.dilate(hole.astype(np.uint8), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)) > 0
        band = np.zeros_like(hole)
        band[max(int(wb[1] - radius), 0):int(wb[3] + 2 * radius)] = True
        touch = np.unique(lab_b[near & bad])
        for k in touch[touch > 0]:
            comp = lab_b == k
            if not (comp & ~band).any() and comp.sum() <= hole.sum():          # wholly inside the whale's rows: wake / unmasked whale part
                hole |= comp
                bad &= ~comp
        hole = cv2.dilate(hole.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if hy is not None:
        bad[: top] = True
    # the whale box grown by the hole (the area the paste and the fill occupy)
    if rec["main"] is not None:
        lab_h, _ = ndimage.label(hole)
        cx, cy = int(np.clip((wb[0] + wb[2]) / 2, 0, WORK_W - 1)), int(np.clip((wb[1] + wb[3]) / 2, 0, Hs - 1))
        k = lab_h[cy, cx] if lab_h[cy, cx] > 0 else None
        if k is not None:
            yy, xx = np.nonzero(lab_h == k)
            seed = [float(xx.min()), float(yy.min()), float(xx.max() + 1), float(yy.max() + 1)]
        else:
            seed = wb
    else:
        seed = wb
    seed = [max(seed[0], 0), max(seed[1], top), min(seed[2], WORK_W), min(seed[3], Hs)]
    if seed[3] - seed[1] < 4:
        return None, "whale above horizon"
    min_w = int(max(384, 1.4 * wlen))
    min_h = int(max(160, 1.6 * (wb[3] - wb[1])))
    rect = best_rect(bad, hole, seed, min(min_w, WORK_W), min_h, top)
    if rect is None:
        return (None, "no clean rectangle") if not debug else (None, {"bad": bad, "hole": hole, "rgb": rgb, "top": top, "hy": hy})
    x0, y0, x1, y1 = rect
    filled, ok = mirror_fill(rgb[y0:y1, x0:x1], hole[y0:y1, x0:x1])
    if not ok:
        return None, "fill without source"
    # white water left in the plate (a breach / fluke splash that is not connected to the mask): large bright
    # regions after an opening that removes sun glitter and small whitecaps
    Lp = lab[y0:y1, x0:x1, 0]
    thr = max(80.0, float(np.median(Lp)) + 30.0)
    white = cv2.morphologyEx((cv2.cvtColor(filled, cv2.COLOR_RGB2Lab)[..., 0].astype(np.float32) * 100 / 255 > thr).astype(np.uint8),
                             cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    white_frac = float(white.mean())
    if white_frac > 0.02:
        return None, "white water in plate"
    hole_p = hole[y0:y1, x0:x1]
    labp = lab[y0:y1, x0:x1][~hole_p]
    hb = None
    if hole_p.any():
        yy, xx = np.nonzero(hole_p)
        hb = [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)]
    idx = dict(day=rec["day"], stem=rec["stem"], kind="whale_photo" if rec["main"] is not None else "no_whale",
               partner=None if rec["main"] is not None else src["stem"], camera=rec.get("camera"), focal_mm=rec.get("focal_mm"),
               time=rec.get("time"), work_scale=s, frame=[WORK_W, Hs], rect=[x0, y0, x1, y1], size=[x1 - x0, y1 - y0],
               whale_box=[wb[0] - x0, wb[1] - y0, wb[2] - x0, wb[3] - y0], length_share=length_share, length_px=wlen,
               hole_box=hb, hole_frac=float(hole_p.mean()), white_frac=white_frac, horizon_top=top,
               lab_mean=[float(v) for v in labp.mean(0)], lab_std=[float(v) for v in labp.std(0)],
               n_instances=len(rec["instances"]))
    if debug:
        idx["_dbg"] = {"bad": bad, "hole": hole, "rgb": rgb, "top": top, "hy": hy}
    return filled, idx


# ------------------------------------------------------------------------------------------------------ build
def _job(args):
    rec, partner = args
    try:
        plate, info = make_plate(rec, partner)
    except Exception as exc:  # noqa: BLE001
        return None, f"error {exc!r}"[:200], rec["stem"]
    if plate is None:
        return None, info, rec["stem"]
    out = PLATES / rec["day"] / f"{rec['stem']}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(plate, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])
    info["file"] = str(out.relative_to(CACHE))
    return info, None, rec["stem"]


def candidates(day: str, per_day: int) -> list[tuple[dict, dict | None]]:
    recs = day_records(day)
    whale = sorted([r for r in recs if r["main"] is not None], key=lambda r: (r["time"] or 0, r["stem"]))
    out = []
    # no_whale photos: partner = nearest-in-time whale photo of the same folder and focal length, within 120 s
    for r in recs:
        if r["status"] == "no_whale" and r["time"]:
            cand = [w for w in whale if w["folder"] == r["folder"] and w["focal_mm"] == r["focal_mm"] and w["time"] and abs(w["time"] - r["time"]) <= 120]
            if cand:
                out.append((r, min(cand, key=lambda w: abs(w["time"] - r["time"]))))
    # whale photos evenly over the day (twice the quota, the build stops a day at per_day accepted plates)
    n = len(whale)
    if n:
        k = min(n, 2 * per_day)
        picks = np.unique(np.linspace(0, n - 1, k).round().astype(int))
        out += [(whale[i], None) for i in picks]
    return out


def write_framing() -> int:
    # the real framing distribution: main whale length share and centre (share of the frame) of EVERY photo with a
    # whale on the training days (plates with very long whales rarely survive, so plates alone would be biased short)
    framing = []
    for day in TRAIN_DAYS:
        for r in day_records(day):
            if r["main"] is not None:
                x0, y0, x1, y1 = r["main"]["bbox"]
                framing.append({"day": day, "length_share": (x1 - x0) / 2560.0, "cx": (x0 + x1) / 2 / 2560.0, "cy": (y0 + y1) / 2 / (2560.0 / 1.5)})   # 3:2 frames
    CACHE.mkdir(parents=True, exist_ok=True)
    FRAMING.write_text(json.dumps(framing))
    return len(framing)


def build(per_day: int, workers: int) -> None:
    from multiprocessing import Pool

    CACHE.mkdir(parents=True, exist_ok=True)
    rows, reasons = [], {}
    with Pool(workers) as pool:
        for day in TRAIN_DAYS:
            jobs = candidates(day, per_day)
            got = 0
            day_reasons = {}
            for info, why, stem in pool.imap(_job, jobs, chunksize=4):
                if info is not None and got < per_day:
                    rows.append(info)
                    got += 1
                elif info is not None:
                    (CACHE / info["file"]).unlink(missing_ok=True)
                else:
                    day_reasons[why] = day_reasons.get(why, 0) + 1
            reasons[day] = {"candidates": len(jobs), "plates": got, "rejected": day_reasons}
            print(day, len(jobs), "candidates", got, "plates", day_reasons, flush=True)
    write_framing()
    with open(INDEX, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (CACHE / "plates_build.json").write_text(json.dumps({"per_day": per_day, "days": reasons, "plates": len(rows)}, indent=1))
    print("plates", len(rows), "->", INDEX)


def load_index(path: Path = INDEX) -> list[dict]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    assert not {r["day"] for r in rows} & FORBIDDEN
    return rows


# ------------------------------------------------------------------------------------------------------ sheets
def sheet(out: Path, n: int = 48, seed: int = 0, cols: int = 4, tile_w: int = 480) -> None:
    """random plates, each: the source photo (work scale, rect in yellow, hole in red, bad in blue) | the plate."""
    rows = load_index()
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(rows), min(n, len(rows)), replace=False)
    tiles = []
    for i in pick:
        r = rows[i]
        recs = {x["stem"]: x for x in day_records(r["day"])}
        rec = recs[r["stem"]]
        partner = recs.get(r["partner"]) if r.get("partner") else None
        _, dbg = make_plate(rec, partner, debug=True)
        g = dbg["_dbg"]
        vis = g["rgb"].copy()
        vis[g["bad"]] = (0.5 * vis[g["bad"]] + [0, 0, 120]).astype(np.uint8)
        vis[g["hole"]] = (0.5 * vis[g["hole"]] + [120, 0, 0]).astype(np.uint8)
        x0, y0, x1, y1 = r["rect"]
        cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 0), 4)
        plate = cv2.cvtColor(cv2.imread(str(CACHE / r["file"])), cv2.COLOR_BGR2RGB)
        a = cv2.resize(vis, (tile_w, int(vis.shape[0] * tile_w / vis.shape[1])))
        b = np.zeros_like(a)
        ph = int(plate.shape[0] * tile_w / WORK_W)
        pw = int(plate.shape[1] * tile_w / WORK_W)
        b[: min(ph, a.shape[0]), :pw] = cv2.resize(plate, (pw, ph))[: a.shape[0]]
        t = np.hstack([a, b])
        cv2.putText(t, f"{r['day']} L={r['length_share']:.2f} {r['kind']}", (6, 22), 0, 0.6, (255, 255, 255), 2)
        t = cv2.resize(t, (2 * tile_w, 340))
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--per-day", type=int, default=150)
    b.add_argument("--workers", type=int, default=6)
    sub.add_parser("framing")
    s = sub.add_parser("sheet")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--n", type=int, default=48)
    s.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.per_day, a.workers)
    elif a.cmd == "framing":
        print("framing records", write_framing())
    else:
        if str(a.out.resolve()).startswith(str(paths.REPO_ROOT.resolve())):
            raise SystemExit("sheets show real photos: write them outside the repository")
        sheet(a.out, a.n, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
