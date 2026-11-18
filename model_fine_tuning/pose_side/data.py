"""Training crops for the keypoint + state head.

Renders (run05 primary whale; run06 test sets per animal): a square around the animal's visible mask (1.3-1.8 x its
long side, centre jittered), 512 px, rotation +-12 deg, mirror with p 0.5 (left / right fluke tips swapped: a mirrored
whale is a valid whale), and the occlusions the real photos have:
  waterline   p 0.4: everything below a slightly tilted line somewhere across the whale is replaced by water taken from
              the same frame beside the whale (feathered edge, a thin foam line half the time); points below it that
              were visible become 'under_water'
  splash      p 0.5: 1-3 real splash / wake cut-outs of training days (seg_splash bank; the click days are left out)
              on the whale's outline; a visible point under alpha >= 0.5 becomes 'hidden'
  photometric brightness / contrast / colour, blur, noise, jpeg
Targets per keypoint: position in the crop (amodal: points under water or hidden keep their true position), a state
(STATES) and two masks: heat (train the position) and state (train the state).
Position is trained for visible / hidden / occluded points inside the crop, and for under-water points inside the crop
only if a point of the same group (fin or fluke) is visible or hidden (nobody can guess a fluke that is all under water).

Real crops (clicks of one click day): the part model's per-instance window (hr_window, 1.5 x the instance's long side),
the same jitter; clicked points are 'visible' with heat and state trained, unclicked points carry no loss.
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import CLICK_DAYS, DAYS, FACTORY, guard

KP = ["fluke_tip_left", "fluke_tip_right", "fluke_notch", "dorsal_fin_front", "dorsal_fin_back", "dorsal_fin_tip"]
GROUP = [0, 0, 0, 1, 1, 1]
FLIP = [1, 0, 2, 3, 4, 5]
STATES = ["visible", "under_water", "hidden", "occluded", "out_of_frame"]
S_ID = {s: i for i, s in enumerate(STATES)}
CROP = 512
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
BANK = Path("/var/tmp/sat3737-parts-cache/seg_splash/splash_bank.npz")


def render_items(dataset: Path, split: str | None, min_px: int = 600):
    """[(dataset, id, instance_id, {name: (u, v, state)})]"""
    ds = Path(dataset); items = []
    ids = json.loads((ds / "splits.json").read_text())[split] if split else sorted(p.stem for p in (ds / "meta").glob("*.json"))
    for i in ids:
        m = json.loads((ds / "meta" / f"{i}.json").read_text())
        if m.get("negative") or m.get("no_whale_in_frame"):
            continue
        animals = m.get("animals") or [dict(instance_id=1, keypoints=m["keypoints"], visible=m.get("visible") or {})]
        for a in animals:
            vis = (a.get("visible") or {}).get("visible_pixels", 0)
            if vis < min_px:
                continue
            kp = {n: (q["uv"][0], q["uv"][1], q.get("state", "out_of_frame")) for n, q in a["keypoints"].items() if n in KP}
            if not any(s == "visible" for _, _, s in kp.values()):
                continue
            items.append((str(ds), i, int(a.get("instance_id", 1)), kp))
    return items


def _square(box, scale, jitter, rng):
    x0, y0, x1, y1 = box; side = max(x1 - x0, y1 - y0, 24) * scale
    cx = (x0 + x1) / 2 + rng.uniform(-jitter, jitter) * side; cy = (y0 + y1) / 2 + rng.uniform(-jitter, jitter) * side
    return cx, cy, side


def _affine(cx, cy, side, rot_deg, flip):
    s = CROP / side; a = np.deg2rad(rot_deg); c, d = np.cos(a) * s, np.sin(a) * s
    M = np.array([[c, -d, 0.0], [d, c, 0.0]]); M[:, 2] = np.array([CROP / 2, CROP / 2]) - M[:, :2] @ np.array([cx, cy])
    if flip:
        M = np.array([[-1.0, 0, CROP], [0, 1.0, 0]]) @ np.vstack([M, [0, 0, 1]])
    return M


def photometric(img, rng):
    img = img.astype(np.float32)
    img = img * rng.uniform(0.7, 1.3) + rng.uniform(-20, 20)
    mean = img.mean((0, 1), keepdims=True); img = (img - mean) * rng.uniform(0.75, 1.25) + mean
    img = img * rng.uniform(0.9, 1.1, size=(1, 1, 3))
    grey = img.mean(2, keepdims=True); img = grey + (img - grey) * rng.uniform(0.6, 1.3)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.4))
    if rng.random() < 0.5:
        img = img + rng.normal(0, rng.uniform(1, 5), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(55, 95))]); img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    return img


class SplashBank:
    def __init__(self, path=BANK, exclude_days=CLICK_DAYS):
        z = np.load(path, allow_pickle=True)
        keep = [i for i, s in enumerate(z["src"]) if s.split("|")[0] not in exclude_days]
        self.rgb = [z["rgb"][i] for i in keep]; self.alpha = [z["alpha"][i] for i in keep]; self.ref = z["ref"][keep]

    def paste(self, img, whale, rng):
        """img RGB uint8 crop, whale bool mask (crop) -> img, alpha_total"""
        img = img.astype(np.float32); H, W = whale.shape; atot = np.zeros((H, W), np.float32)
        ys, xs = np.nonzero(whale)
        if len(xs) < 50:
            return img.astype(np.uint8), atot
        wlen = max(xs.max() - xs.min(), ys.max() - ys.min())
        edge = whale & ~(cv2.erode(whale.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0); ey, ex = np.nonzero(edge)
        for _ in range(1 + int(rng.random() < 0.4) + int(rng.random() < 0.2)):
            k = int(rng.integers(len(self.rgb))); rgb = self.rgb[k].astype(np.float32); al = self.alpha[k].astype(np.float32)
            f = float(np.clip(wlen / self.ref[k] * rng.uniform(0.6, 1.6), 0.2, 3.0)); h, w = al.shape
            f = min(f, 0.6 * W / max(w, h)); nh, nw = max(2, int(h * f)), max(2, int(w * f))
            rgb = cv2.resize(rgb, (nw, nh)); al = np.clip(cv2.resize(al, (nw, nh)), 0, 1)
            e = int(rng.integers(len(ex))); cx = ex[e] + rng.uniform(-0.4, 0.4) * nw; cy = ey[e] + rng.uniform(-0.4, 0.4) * nh
            x0, y0 = int(cx - nw / 2), int(cy - nh / 2)
            sx0, sy0, sx1, sy1 = max(0, -x0), max(0, -y0), min(nw, W - x0), min(nh, H - y0)
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            a = al[sy0:sy1, sx0:sx1]; t = (slice(y0 + sy0, y0 + sy1), slice(x0 + sx0, x0 + sx1))
            img[t] = img[t] * (1 - a[..., None]) + rgb[sy0:sy1, sx0:sx1] * a[..., None]; atot[t] = np.maximum(atot[t], a)
        return np.clip(img, 0, 255).astype(np.uint8), atot


def waterline(frame, M, whale_full, crop, whale_c, rng):
    """Cut the crop with a water surface: returns crop, function below(x, y) -> bool. None if no clean water found."""
    ys, xs = np.nonzero(whale_c)
    if len(xs) < 50:
        return None
    top, bot = ys.min(), ys.max(); y_cut = rng.uniform(top + 0.15 * (bot - top), bot + 1); slope = np.tan(np.deg2rad(rng.uniform(-5, 5)))
    H, W = frame.shape[:2]
    for _ in range(6):                                    # water from the same frame, shifted sideways (and a bit vertically)
        dx = rng.choice([-1, 1]) * rng.uniform(0.6, 1.4) * CROP / abs(M[0, 0] + 1e-9) * 0.9; dy = rng.uniform(-0.1, 0.1) * CROP / abs(M[0, 0] + 1e-9)
        Ms = M.copy(); Ms[:, 2] = M[:, 2] - M[:, :2] @ np.array([dx, dy])
        wc = cv2.warpAffine(whale_full.astype(np.uint8), Ms, (CROP, CROP), flags=cv2.INTER_NEAREST, borderValue=1)
        if wc.mean() < 0.02:
            water = cv2.warpAffine(frame, Ms, (CROP, CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            break
    else:
        return None
    yy, xx = np.mgrid[0:CROP, 0:CROP].astype(np.float32)
    line = y_cut + slope * (xx - CROP / 2) + 2.0 * np.sin(xx / rng.uniform(15, 40) + rng.uniform(0, 6))
    feather = rng.uniform(1.0, 3.0); a = np.clip((yy - line) / feather + 0.5, 0, 1)[..., None]
    out = crop.astype(np.float32) * (1 - a) + water.astype(np.float32) * a
    if rng.random() < 0.5:                               # a thin foam line at the cut
        foam = np.exp(-((yy - line) ** 2) / (2 * rng.uniform(0.8, 2.5) ** 2)) * (whale_c | (np.abs(yy - line) < 3)) * rng.uniform(0.3, 0.8)
        out = out * (1 - foam[..., None]) + 235 * foam[..., None]
    return np.clip(out, 0, 255).astype(np.uint8), (lambda x, y: y > y_cut + slope * (x - CROP / 2))


def normalise(img):
    return torch.from_numpy(((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1).copy())


class RenderCrops(Dataset):
    def __init__(self, items, train=True, p_water=0.4, p_splash=0.5, seed=0):
        self.items, self.train, self.pw, self.ps, self.seed = items, train, p_water, p_splash, seed
        self.bank = SplashBank() if train and p_splash > 0 else None
        self.epoch = 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ds, i, inst_id, kp = self.items[idx]
        rng = np.random.default_rng((self.seed, self.epoch, idx) if self.train else (12345, idx))
        frame = cv2.cvtColor(cv2.imread(f"{ds}/images/{i}.png"), cv2.COLOR_BGR2RGB)
        inst = np.array(Image.open(f"{ds}/instances/{i}.png")); whale_full = inst == inst_id
        ys, xs = np.nonzero(whale_full)
        box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else (0, 0, frame.shape[1], frame.shape[0])
        if self.train:
            cx, cy, side = _square(box, rng.uniform(1.3, 1.8), 0.12, rng); rot = rng.uniform(-12, 12); flip = rng.random() < 0.5
        else:
            cx, cy, side = _square(box, 1.5, 0.0, rng); rot = 0.0; flip = False
        M = _affine(cx, cy, side, rot, flip)
        crop = cv2.warpAffine(frame, M, (CROP, CROP), flags=cv2.INTER_AREA if side > CROP else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        whale_c = cv2.warpAffine(whale_full.astype(np.uint8), M, (CROP, CROP), flags=cv2.INTER_NEAREST) > 0
        names = [KP[FLIP[k]] if flip else KP[k] for k in range(len(KP))]
        pts = np.zeros((len(KP), 2), np.float32); st = np.full(len(KP), S_ID["out_of_frame"], np.int64)
        for k, n in enumerate(names):
            if n in kp:
                u, v, s = kp[n]; pts[k] = M[:, :2] @ np.array([u, v]) + M[:, 2]; st[k] = S_ID.get(s, S_ID["out_of_frame"])
        inside = (pts[:, 0] >= 0) & (pts[:, 0] < CROP) & (pts[:, 1] >= 0) & (pts[:, 1] < CROP)
        st[~inside] = S_ID["out_of_frame"]
        if self.train and rng.random() < self.pw:
            r = waterline(frame, M, whale_full, crop, whale_c, rng)
            if r is not None:
                crop, below = r
                for k in range(len(KP)):
                    if st[k] in (S_ID["visible"], S_ID["hidden"]) and below(*pts[k]):
                        st[k] = S_ID["under_water"]
        if self.bank is not None and rng.random() < self.ps:
            crop, al = self.bank.paste(crop, whale_c, rng)
            for k in range(len(KP)):
                if st[k] == S_ID["visible"] and al[int(np.clip(pts[k, 1], 0, CROP - 1)), int(np.clip(pts[k, 0], 0, CROP - 1))] >= 0.5:
                    st[k] = S_ID["hidden"]
        if self.train:
            crop = photometric(crop, rng)
        seen = {g: any(st[k] in (S_ID["visible"], S_ID["hidden"]) for k in range(len(KP)) if GROUP[k] == g) for g in (0, 1)}
        heat = np.array([inside[k] and (st[k] in (S_ID["visible"], S_ID["hidden"], S_ID["occluded"]) or (st[k] == S_ID["under_water"] and seen[GROUP[k]]))
                         for k in range(len(KP))])
        return dict(img=normalise(crop), pts=torch.from_numpy(pts), state=torch.from_numpy(st), heat=torch.from_numpy(heat),
                    smask=torch.ones(len(KP), dtype=torch.bool))


# ------------------------------------------------------------------------------------------------ real photos
def real_window(day, key):
    """square window (x0, y0, side) in jpg pixels around the instance: the part model's hr window (1.5 x long side)."""
    z = _inst_json(day)
    r = z.get(key)
    if r and r.get("hr_window"):
        x0, y0, side, _ = r["hr_window"]; return x0 + side / 2, y0 + side / 2, side          # (x0, y0, side, out size), 2560 px
    return None


_IJ = {}


def _inst_json(day):
    if day not in _IJ:
        guard([day]); _IJ[day] = json.loads((DAYS / day / "parts_seg_r05_a" / "instances.json").read_text())["instances"]
    return _IJ[day]


def load_jpg(day, stem):
    guard([day]); return cv2.cvtColor(cv2.imread(str(DAYS / day / "images" / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)


class RealCrops(Dataset):
    """clicked photos of the given days: [(day, key, {name: (x, y)})]"""
    def __init__(self, rows, seed=0):
        self.rows = [r for r in rows if real_window(r[0], r[1]) is not None]; self.seed = seed; self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        day, key, kp = self.rows[idx]; rng = np.random.default_rng((self.seed, self.epoch, idx, 7))
        frame = load_jpg(day, key.rsplit("~", 1)[0]); cx, cy, side = real_window(day, key)
        side *= rng.uniform(0.9, 1.15); cx += rng.uniform(-0.08, 0.08) * side; cy += rng.uniform(-0.08, 0.08) * side
        flip = rng.random() < 0.5; M = _affine(cx, cy, side, rng.uniform(-8, 8), flip)
        crop = photometric(cv2.warpAffine(frame, M, (CROP, CROP), flags=cv2.INTER_AREA if side > CROP else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), rng)
        pts = np.zeros((len(KP), 2), np.float32); st = np.zeros(len(KP), np.int64); heat = np.zeros(len(KP), bool)
        for k in range(len(KP)):
            n = KP[FLIP[k]] if flip else KP[k]
            if n in kp:
                pts[k] = M[:, :2] @ np.array(kp[n]) + M[:, 2]
                heat[k] = (0 <= pts[k, 0] < CROP) and (0 <= pts[k, 1] < CROP)
        return dict(img=normalise(crop), pts=torch.from_numpy(pts), state=torch.from_numpy(st), heat=torch.from_numpy(heat),
                    smask=torch.from_numpy(heat.copy()))
