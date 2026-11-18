"""Realism gap, shared parts: where things live, which samples to take from each world, how to load
one sample (image + whale mask + "ignore" mask), the two views (whale-centred square crop, full frame)
and the region map of a view.

Region ids in every label map:
    0 ignore (padding, other whales / calves, instances the not-whale filter dropped)
    1 whale (the sample's main whale, visible part)
    2 band  (water within about one whale-height of the whale mask, below the horizon)
    3 far   (water farther than that, below the horizon)
    4 above (above the detected horizon: sky, or land where there is land)

Real photos: train + val days only (model_fine_tuning/splits/days.json); the eval days are refused.
Renders: any Whale-Image-Data-Factory dataset folder (images/, masks/, optional instances/, meta/ or
manifest.jsonl); render 'mask' class 2 (whale just under water) counts as water, as the same pixels do
in a real photo's mask.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
DAYS_ALL = REPO / "experiments" / "days_all"
SPLITS = REPO / "model_fine_tuning" / "splits" / "days.json"
SCRATCH = Path(os.environ.get("REALISM_GAP_OUT", "/tmp/claude-1072903897/-home-sat3737-NOAA-whales/"
                              "4fbba7ff-3ba5-4b50-9cb7-970c81250409/scratchpad/realism_gap"))
CACHE = SCRATCH / "cache"
DEFAULT_SCENARIOS = Path("/tmp/claude-1072903897/-home-sat3737-NOAA-whales/4fbba7ff-3ba5-4b50-9cb7-970c81250409/"
                         "scratchpad/hard/scenarios.json")
RUN05 = Path("/home/sat3737/NOAA_whales/Whale-Image-Data-Factory/data/renders/run05/dataset")

CROP = 512          # whale-centred square crop side after resizing
CROP_FACTOR = 1.6   # crop side = longer side of the whale box x this
FRAME_LONG = 1280   # full frame resized to this long side
REGIONS = {1: "whale", 2: "band", 3: "far", 4: "above"}

_splits = json.loads(SPLITS.read_text())
EVAL_DAYS = set(_splits["eval"])
REAL_DAYS = list(_splits["train"]) + list(_splits["val"])
assert not EVAL_DAYS & set(REAL_DAYS)


def _check_day(day: str) -> None:
    if day in EVAL_DAYS:
        raise RuntimeError(f"evaluation day {day} must not be read")


# ------------------------------------------------------------------------------------------ samples
def _day_instances(day: str):
    """Photos of one day with their instances (not-whale instances flagged)."""
    _check_day(day)
    root = DAYS_ALL / day
    drop = set()
    nw = root / "safe" / "not_whale.json"
    if nw.exists():
        drop = set(json.loads(nw.read_text()).get("drop", []))
    out = []
    for line in (root / "parts_hr" / "index.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        stem = r["stem"]
        img, msk = root / "images" / f"{stem}.jpg", root / "canon" / f"{stem}_mask.png"
        if not (img.exists() and msk.exists()):
            continue
        inst = [dict(key=i["key"], bbox=list(map(float, i["bbox_2560"])), area=float(i.get("area", 0)), drop=i["key"] in drop)
                for i in r.get("instances", []) if i.get("bbox_2560")]
        out.append(dict(stem=stem, img=str(img), mask=str(msk), instances=inst))
    return out


def real_samples(n: int = 1500, seed: int = 0) -> list[dict]:
    """About n photos, evenly over the train + val days (a day with fewer photos gives all of them),
    the main instance = the largest instance that is not a 'drop'."""
    rng = random.Random(seed)
    per_day = {}
    for day in REAL_DAYS:
        photos = []
        for p in _day_instances(day):
            keep = [i for i in p["instances"] if not i["drop"]]
            if not keep:
                continue
            main = max(keep, key=lambda i: (i["bbox"][2] - i["bbox"][0]) * (i["bbox"][3] - i["bbox"][1]))
            photos.append(dict(world="real", id=main["key"], key=main["key"], day=day, group=day, stem=p["stem"], img=p["img"],
                               mask=p["mask"], bbox=main["bbox"], others=[i for i in p["instances"] if i["key"] != main["key"]]))
        rng.shuffle(photos)
        per_day[day] = photos
    quota = _water_fill({d: len(v) for d, v in per_day.items()}, n)
    out = [s for d in REAL_DAYS for s in per_day[d][:quota[d]]]
    return out


def _water_fill(counts: dict, n: int) -> dict:
    """Equal shares, a group with fewer items gives all and the rest is shared by the others."""
    quota = {k: 0 for k in counts}
    left = dict(counts)
    budget = n
    while budget > 0 and any(left.values()):
        active = [k for k in left if left[k] > 0]
        share = max(budget // len(active), 1)
        for k in active:
            take = min(share, left[k], budget)
            quota[k] += take; left[k] -= take; budget -= take
            if budget <= 0:
                break
    return quota


def scenario_samples(path: Path, per: int = 60, seed: int = 0) -> list[dict]:
    """Instances of the hard-situation lists (train / val days only), up to `per` per situation."""
    d = json.loads(Path(path).read_text())
    rng = random.Random(seed)
    cache: dict[str, dict] = {}
    out = []
    for name, keys in d["scenarios"].items():
        keys = [k for k in keys if d["day"].get(k) in REAL_DAYS]
        rng.shuffle(keys)
        got = 0
        for key in keys:
            if got >= per:
                break
            day = d["day"][key]
            if day not in cache:
                cache[day] = {i["key"]: (p, i) for p in _day_instances(day) for i in p["instances"]}
            if key not in cache[day]:
                continue
            p, inst = cache[day][key]
            out.append(dict(world="real", id=f"{name}|{key}", key=key, scenario=name, day=day, group=day, stem=p["stem"],
                            img=p["img"], mask=p["mask"], bbox=inst["bbox"],
                            others=[i for i in p["instances"] if i["key"] != key]))
            got += 1
    return out


def _render_groups(dataset: Path, metas: dict) -> dict:
    """Render batch per shot id: <dataset>/../batches/batch_*.json when present, else sea state + look."""
    groups = {}
    bdir = dataset.parent / "batches"
    if bdir.is_dir():
        for f in sorted(bdir.glob("batch_*.json")):
            try:
                b = json.loads(f.read_text())
            except Exception:
                continue
            for s in b.get("shots", []):
                groups[str(s["id"])] = f"batch{int(b.get('index', 0)):04d}"
    for i, m in metas.items():
        groups.setdefault(i, f"{m.get('sea_state', '?')}_{m.get('look', '?')}")
    return groups


def render_samples(dataset: Path, n: int = 1500, seed: int = 0, min_visible: int = 300) -> list[dict]:
    """About n shots with a visible whale, stratified over pose categories in proportion to the run
    (at least 20 per category where it has them)."""
    dataset = Path(dataset)
    rows = {}
    man = dataset / "manifest.jsonl"
    if man.exists():
        for line in man.read_text().splitlines():
            if line.strip():
                r = json.loads(line); rows[str(r["id"])] = r
    else:
        for f in sorted((dataset / "meta").glob("*.json")):
            m = json.loads(f.read_text())
            rows[str(m["id"])] = dict(id=m["id"], category=m.get("category"), negative=m.get("negative", False),
                                     sea_state=m.get("sea_state"), look=m.get("look"),
                                     visible_pixels=(m.get("visible") or {}).get("visible_pixels", 0))
    ok = {i: r for i, r in rows.items() if not r.get("negative") and (r.get("visible_pixels") or 0) >= min_visible
          and (dataset / "images" / f"{i}.png").exists()}
    groups = _render_groups(dataset, ok)
    bycat = defaultdict(list)
    for i, r in ok.items():
        bycat[r.get("category") or "?"].append(i)
    rng = random.Random(seed)
    total = sum(len(v) for v in bycat.values())
    out = []
    for cat in sorted(bycat):
        ids = sorted(bycat[cat]); rng.shuffle(ids)
        k = min(len(ids), max(20, round(n * len(ids) / max(total, 1))))
        for i in ids[:k]:
            r = ok[i]
            out.append(dict(world="render", id=i, group=groups.get(i, "?"), category=cat, sea_state=r.get("sea_state"),
                            look=r.get("look"), img=str(dataset / "images" / f"{i}.png"), mask=str(dataset / "masks" / f"{i}.png"),
                            inst=str(dataset / "instances" / f"{i}.png") if (dataset / "instances" / f"{i}.png").exists() else None))
    return out


def tag_of(dataset: Path, n: int, seed: int) -> str:
    """Cache name of a render dataset: path, sample size, and the images folder's file count and
    modification time, so a test re-rendered into the same folder gets fresh features."""
    dataset = Path(dataset).resolve()
    img = dataset / "images"
    stamp = f"{len(os.listdir(img))}|{int(img.stat().st_mtime)}" if img.is_dir() else ""
    h = hashlib.sha1(f"{dataset}|{n}|{seed}|{stamp}".encode()).hexdigest()[:8]
    return f"{dataset.parent.name}_{dataset.name}_{h}"


# ------------------------------------------------------------------------------------------ loading
def load_sample(s: dict):
    """-> rgb uint8 (H,W,3), whale bool (H,W), ignore bool (H,W), box xyxy (main whale, native px)."""
    rgb = cv2.cvtColor(cv2.imread(s["img"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    if s["world"] == "real":
        m = cv2.imread(s["mask"], cv2.IMREAD_GRAYSCALE)
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        m = m > 0
        sc = W / 2560.0
        x0, y0, x1, y1 = [v * sc for v in s["bbox"]]
        pw, ph = 0.03 * (x1 - x0) + 2, 0.03 * (y1 - y0) + 2
        inbox = np.zeros((H, W), bool)
        inbox[max(int(y0 - ph), 0):min(int(np.ceil(y1 + ph)), H), max(int(x0 - pw), 0):min(int(np.ceil(x1 + pw)), W)] = True
        whale = m & inbox
        ignore = m & ~whale
        for o in s.get("others", []):            # a dropped (not-whale) instance: ignore its whole box
            if o.get("drop"):
                a0, b0, a1, b1 = [int(round(v * sc)) for v in o["bbox"]]
                ignore[max(b0, 0):b1, max(a0, 0):a1] = True
        ignore &= ~whale
    else:
        cls = np.asarray(Image.open(s["mask"]))
        if cls.ndim == 3:
            cls = cls[..., 0]
            cls = np.where(cls > 200, 1, np.where(cls > 64, 2, 0))
        if s.get("inst"):
            inst = np.asarray(Image.open(s["inst"]))
            if inst.ndim == 3:
                inst = inst[..., 0]; inst = np.where(inst > 200, 1, np.where(inst > 64, 2, 0))
            whale = inst == 1
            ignore = (inst > 1) | ((cls == 1) & ~whale)
        else:
            whale = cls == 1
            ignore = np.zeros_like(whale)
        x0 = y0 = x1 = y1 = None
    if whale.any():
        ys, xs = np.nonzero(whale)
        box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    else:
        box = [x0, y0, x1, y1] if x0 is not None else [W * 0.45, H * 0.45, W * 0.55, H * 0.55]
    return rgb, whale, ignore, box


# ------------------------------------------------------------------------------------------ views
def crop_params(box, W, H):
    """Square side = 1.6 x the box's longer side, centred on the box, shifted to stay inside the photo
    where it fits (padding only where the square is larger than the photo)."""
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * CROP_FACTOR
    side = max(side, 16.0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sx = cx - side / 2; sy = cy - side / 2
    if side <= W:
        sx = min(max(sx, 0), W - side)
    else:
        sx = (W - side) / 2
    if side <= H:
        sy = min(max(sy, 0), H - side)
    else:
        sy = (H - side) / 2
    return sx, sy, side


def warp_square(arr, sx, sy, side, out, interp, border=0):
    s = out / side
    M = np.array([[s, 0, -sx * s], [0, s, -sy * s]], np.float32)
    if interp is None:
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return cv2.warpAffine(arr, M, (out, out), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border), M


def resize_any(arr, size, nearest=False):
    w, h = size
    if nearest:
        return cv2.resize(arr.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    s = w / arr.shape[1]
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


def region_map(whale, ignore, valid, horizon_y, band_px):
    """Label map (uint8) from whale / ignore / valid masks of one view, a horizon row per column
    (None = no horizon: everything water) and the band width in pixels."""
    H, W = whale.shape
    lab = np.zeros((H, W), np.uint8)
    above = np.zeros((H, W), bool)
    if horizon_y is not None:
        rows = np.arange(H)[:, None]
        above = rows < horizon_y[None, :]
    water = valid & ~whale & ~ignore & ~above
    if whale.any():
        dist = cv2.distanceTransform((~whale).astype(np.uint8), cv2.DIST_L2, 5)
        band = water & (dist <= band_px)
    else:
        band = np.zeros_like(water)
    lab[water] = 3
    lab[band] = 2
    lab[valid & above & ~whale & ~ignore] = 4
    lab[whale & valid] = 1
    return lab


def ensure_dirs():
    for d in (SCRATCH, CACHE):
        d.mkdir(parents=True, exist_ok=True)
