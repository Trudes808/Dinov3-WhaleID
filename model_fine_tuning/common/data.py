"""Reading a Whale-Image-Data-Factory run and building training crops.

Dataset layout (RUN/dataset): images/<id>.png (RGBA), masks/<id>.png (palette, 0/1 visible
whale), bodycoords/<id>.png (R = fraction along body, G = angle/360, B = 255 where defined,
over the *amodal* whale), meta/<id>.json (pose, intrinsics, keypoints...), manifest.jsonl.

Two crop strategies:
  * SegCropDataset   random crops at roughly native scale (scale jitter), biased to contain
                     whale pixels; evaluation uses the full frame.
  * BodyCropDataset  square crops centred on the visible whale with a margin, resized to a
                     fixed size, so the whale fills a consistent fraction of the input;
                     at test time the crop comes from the segmentation model's mask.
Padding outside the source image is labelled IGNORE (255) in masks and invalid in body coords.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from . import geometry
from .splits import load_split

IGNORE = 255
CANON_OFFSET = 32768   # canon npz value = mm + offset (factory.postprocess)


class WhaleRun:
    def __init__(self, run_dir: Path, split_path: Path | None = None):
        self.run_dir = Path(run_dir)
        self.dataset_dir = self.run_dir / "dataset"
        with open(self.dataset_dir / "manifest.jsonl") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        self.by_id = {r["id"]: r for r in self.rows}
        self.split = load_split(split_path) if split_path else None
        self._meta: dict[str, dict] = {}
        lm = self.dataset_dir / "landmarks_mesh_frame_cm.json"
        self.landmarks_m = geometry.load_landmarks(lm) if lm.exists() else None

    def ids(self, split: str | None = None) -> list[str]:
        if split is None or split == "all":
            return [r["id"] for r in self.rows]
        if self.split is None:
            raise ValueError("no split file loaded")
        return [i for i in self.split[split] if i in self.by_id]

    def meta(self, shot_id: str) -> dict:
        if shot_id not in self._meta:
            self._meta[shot_id] = json.loads((self.dataset_dir / "meta" / f"{shot_id}.json").read_text())
        return self._meta[shot_id]

    def image(self, shot_id: str) -> np.ndarray:
        bgr = cv2.imread(str(self.dataset_dir / "images" / f"{shot_id}.png"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(shot_id)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def mask(self, shot_id: str) -> np.ndarray:
        """uint8 label: 0 background, 1 visible whale, IGNORE where the factory wrote class 2
        (whale just below the surface, which may show through the water: not scored)."""
        # Palette PNG: PIL returns the class ids (cv2 would expand the palette to colours).
        m = np.asarray(Image.open(self.dataset_dir / "masks" / f"{shot_id}.png"))
        if m.ndim == 3:
            m = m[..., 0]
        if m.max(initial=0) > 2:                     # a colour-expanded or 0/255 mask
            m = np.where(m >= 200, 1, np.where(m > 0, 2, 0))
        out = (m == 1).astype(np.uint8)
        out[m == 2] = IGNORE
        return out

    def visible(self, shot_id: str) -> np.ndarray:
        return self.mask(shot_id) == 1

    def primary(self, shot_id: str, mask: np.ndarray) -> np.ndarray:
        """The shot's own whale (for the crop box): instances/<id>.png == 1 when the shot has a companion
        whale (run04 mom + calf, factory.postprocess), otherwise the visible-whale class of the mask."""
        path = self.dataset_dir / "instances" / f"{shot_id}.png"
        if path.exists():
            inst = np.asarray(Image.open(path))
            if (inst == 1).any():
                return inst == 1
        return mask == 1

    def contours(self, shot_id: str) -> np.ndarray | None:
        """Boundary classes of the visible mask (factory.postprocess): 1 true silhouette (the whale ends there,
        e.g. the top contour), 2 waterline / foam edge (the body continues under water: unreliable in real
        photos), 3 edge against a calf; None for datasets without the label. For contour-weighted losses and
        for silhouette fitting that uses the top contour only."""
        path = self.dataset_dir / "contours" / f"{shot_id}.png"
        if not path.exists():
            return None
        return np.asarray(Image.open(path))

    def contours_or_zeros(self, shot_id: str) -> np.ndarray:
        """contours() or an all-zero map of the mask's shape (runs before the label: no re-weighting, no split metrics)."""
        c = self.contours(shot_id)
        if c is not None:
            return c.astype(np.uint8)
        m = self.mask(shot_id)
        return np.zeros(m.shape, np.uint8)

    def bodycoords(self, shot_id: str):
        bgr = cv2.imread(str(self.dataset_dir / "bodycoords" / f"{shot_id}.png"), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return decode_bodycoords(rgb)

    def has_canon(self, shot_id: str) -> bool:
        return (self.dataset_dir / "canon" / f"{shot_id}.npz").exists()

    def canon(self, shot_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Canonical surface point per pixel (cm, undeformed mesh frame: x tail -> snout, y right,
        z up) as float32 (H, W, 3) with nan outside the whale, and the defined mask."""
        with np.load(self.dataset_dir / "canon" / f"{shot_id}.npz") as data:
            values = data["xyz_mm"]
        defined = values[..., 0] > 0
        cm = (values.astype(np.float32) - CANON_OFFSET) / 10.0
        cm[~defined] = np.nan
        return cm, defined

    def is_deformed(self, shot_id: str) -> bool:
        """True when the factory bent the body for this shot (run02+): the rigid pose and the
        canonical body table are then only approximate."""
        return any(abs(float(v)) > 1e-9 for v in (self.meta(shot_id).get("deform") or {}).values())

    def has_bodycoords(self, shot_id: str) -> bool:
        return (self.dataset_dir / "bodycoords" / f"{shot_id}.png").exists()


class CombinedRun:
    """Several factory runs as one dataset (run03 + run04: the flat-whale and the rigged-whale data together). Ids
    are "<k>:<id>" with k the run's index, every accessor dispatches on the prefix, the splits are the unions of
    the runs' splits. landmarks_m and dataset_dir are the first run's (evaluation stays per run)."""

    def __init__(self, runs: list[WhaleRun]):
        self.runs = runs
        self.rows = [dict(r, id=f"{k}:{r['id']}") for k, run in enumerate(runs) for r in run.rows]
        self.by_id = {r["id"]: r for r in self.rows}
        self.run_dir, self.dataset_dir, self.landmarks_m = runs[0].run_dir, runs[0].dataset_dir, runs[0].landmarks_m
        self.split = None

    def _run(self, shot_id: str) -> tuple[WhaleRun, str]:
        k, i = shot_id.split(":", 1)
        return self.runs[int(k)], i

    def ids(self, split: str | None = None) -> list[str]:
        return [f"{k}:{i}" for k, run in enumerate(self.runs) for i in run.ids(split)]

    def meta(self, shot_id): run, i = self._run(shot_id); return run.meta(i)
    def image(self, shot_id): run, i = self._run(shot_id); return run.image(i)
    def mask(self, shot_id): run, i = self._run(shot_id); return run.mask(i)
    def visible(self, shot_id): run, i = self._run(shot_id); return run.visible(i)
    def primary(self, shot_id, mask): run, i = self._run(shot_id); return run.primary(i, mask)
    def contours(self, shot_id): run, i = self._run(shot_id); return run.contours(i)
    def contours_or_zeros(self, shot_id): run, i = self._run(shot_id); return run.contours_or_zeros(i)
    def bodycoords(self, shot_id): run, i = self._run(shot_id); return run.bodycoords(i)
    def has_canon(self, shot_id): run, i = self._run(shot_id); return run.has_canon(i)
    def canon(self, shot_id): run, i = self._run(shot_id); return run.canon(i)
    def is_deformed(self, shot_id): run, i = self._run(shot_id); return run.is_deformed(i)
    def has_bodycoords(self, shot_id): run, i = self._run(shot_id); return run.has_bodycoords(i)


def open_runs(run_dir, split, extra_dirs=None, extra_splits=None):
    """WhaleRun, or a CombinedRun when extra run directories (with their split files) are given."""
    run = WhaleRun(run_dir, split)
    if not extra_dirs:
        return run
    extra_splits = extra_splits or []
    return CombinedRun([run] + [WhaleRun(Path(d), Path(s) if s else None) for d, s in zip(extra_dirs, extra_splits + [None] * (len(extra_dirs) - len(extra_splits)))])


def decode_bodycoords(rgb: np.ndarray):
    u = rgb[..., 0].astype(np.float32) / 255.0
    theta = rgb[..., 1].astype(np.float32) / 255.0 * 360.0
    defined = rgb[..., 2] > 127
    return u, theta, defined


# ---- geometric warps ---------------------------------------------------------------------------
def warp_params(x0: float, y0: float, scale: float) -> np.ndarray:
    """Affine mapping source pixel -> crop pixel: crop = (src - (x0, y0)) * scale."""
    return np.array([[scale, 0.0, -x0 * scale], [0.0, scale, -y0 * scale]], np.float64)


def warp_image(image: np.ndarray, M: np.ndarray, size: tuple[int, int], interpolation=cv2.INTER_LINEAR, border=0) -> np.ndarray:
    flags = interpolation
    return cv2.warpAffine(image, M, size, flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def warp_nearest(label: np.ndarray, M: np.ndarray, size: tuple[int, int], fill) -> np.ndarray:
    return cv2.warpAffine(label, M, size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


# ---- photometric augmentation -------------------------------------------------------------------
def photometric(image: np.ndarray, rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    """Augmentations aimed at the sim-to-real gap: exposure, white balance, saturation, blur,
    sensor noise, JPEG, occasional greyscale. Input/output uint8 RGB."""
    img = image.astype(np.float32) / 255.0
    s = strength
    # exposure / contrast / gamma
    img = img * rng.uniform(1 - 0.35 * s, 1 + 0.35 * s)
    img = (img - 0.5) * rng.uniform(1 - 0.3 * s, 1 + 0.3 * s) + 0.5
    img = np.clip(img, 0, 1) ** rng.uniform(1 - 0.3 * s, 1 + 0.3 * s)
    # white balance
    img = img * rng.uniform(1 - 0.12 * s, 1 + 0.12 * s, size=(1, 1, 3))
    # saturation
    grey = img.mean(axis=2, keepdims=True)
    img = grey + (img - grey) * rng.uniform(1 - 0.5 * s, 1 + 0.3 * s)
    if rng.random() < 0.05 * s:
        img = np.repeat(img.mean(axis=2, keepdims=True), 3, axis=2)
    img = np.clip(img, 0, 1)
    if rng.random() < 0.3 * s:
        k = int(rng.choice([3, 5, 7]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.5 * s:
        img = img + rng.normal(0, rng.uniform(0.0, 0.03 * s), img.shape).astype(np.float32)
    out = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if rng.random() < 0.3 * s:
        q = int(rng.uniform(35, 95))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return out


def to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0


def pad_to_multiple(image: np.ndarray, multiple: int = 16, fill=0):
    """Pad bottom/right so H and W are multiples of ``multiple``. Returns (padded, (H, W) original)."""
    h, w = image.shape[:2]
    H = int(math.ceil(h / multiple) * multiple)
    W = int(math.ceil(w / multiple) * multiple)
    if (H, W) == (h, w):
        return image, (h, w)
    pad = ((0, H - h), (0, W - w)) + ((0, 0),) * (image.ndim - 2)
    return np.pad(image, pad, constant_values=fill), (h, w)


def groups_for(row: dict) -> dict:
    from .metrics import size_bin

    return {"category": row["category"], "sea_state": row["sea_state"], "look": row["look"], "size_bin": size_bin(row["visible_pixels"]),
            "negative": bool(row["negative"])}


# ---- segmentation ------------------------------------------------------------------------------
class SegCropDataset(Dataset):
    def __init__(self, run: WhaleRun, ids: list[str], crop: int = 512, scale_range=(0.5, 1.5), whale_bias: float = 0.7,
                 flip: bool = True, photometric_strength: float = 1.0, train: bool = True, seed: int = 0):
        self.run, self.ids, self.crop = run, list(ids), crop
        self.scale_range, self.whale_bias, self.flip = scale_range, whale_bias, flip
        self.photometric_strength, self.train, self.seed = photometric_strength, train, seed
        self.epoch = 0

    def __len__(self):
        return len(self.ids)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __getitem__(self, index: int):
        shot_id = self.ids[index]
        image = self.run.image(shot_id)
        mask = self.run.mask(shot_id)
        contours = self.run.contours_or_zeros(shot_id)
        if not self.train:
            img, (h, w) = pad_to_multiple(image)
            m, _ = pad_to_multiple(mask, fill=IGNORE)
            c, _ = pad_to_multiple(contours, fill=0)
            return {"id": shot_id, "image": to_tensor(img), "mask": torch.from_numpy(m.astype(np.int64)), "orig_hw": torch.tensor([h, w]),
                    "contours": torch.from_numpy(c.astype(np.uint8))}
        rng = np.random.default_rng([self.seed, self.epoch, index])
        H, W = mask.shape
        C = self.crop
        s = float(np.exp(rng.uniform(np.log(self.scale_range[0]), np.log(self.scale_range[1]))))
        win = C / s
        ys, xs = np.nonzero(mask)
        if len(xs) and rng.random() < self.whale_bias:
            k = rng.integers(len(xs))
            x0 = xs[k] - rng.uniform(0.1, 0.9) * win
            y0 = ys[k] - rng.uniform(0.1, 0.9) * win
        else:
            x0 = rng.uniform(-0.2 * win, W - 0.8 * win)
            y0 = rng.uniform(-0.2 * win, H - 0.8 * win)
        M = warp_params(x0, y0, s)
        img = warp_image(image, M, (C, C))
        m = warp_nearest(mask, M, (C, C), IGNORE)
        c = warp_nearest(contours, M, (C, C), 0)
        if self.flip and rng.random() < 0.5:
            img, m, c = img[:, ::-1], m[:, ::-1], c[:, ::-1]
        img = photometric(np.ascontiguousarray(img), rng, self.photometric_strength)
        return {"id": shot_id, "image": to_tensor(img), "mask": torch.from_numpy(np.ascontiguousarray(m).astype(np.int64)),
                "contours": torch.from_numpy(np.ascontiguousarray(c).astype(np.uint8))}


# ---- body coordinates + pose ------------------------------------------------------------------------
def bbox_of(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def square_crop_params(bbox, crop: int, margin: float, jitter: float, rng: np.random.Generator | None, min_side: float = 160.0):
    """Square source window around a bbox -> (x0, y0, scale) mapping it to crop x crop pixels."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * margin
    side = max(side, min_side)
    if rng is not None and jitter > 0:
        cx += rng.uniform(-jitter, jitter) * side
        cy += rng.uniform(-jitter, jitter) * side
    scale = crop / side
    return cx - side / 2.0, cy - side / 2.0, scale


class BodyCropDataset(Dataset):
    """Whale-centred crops with dense body-coordinate targets and the rigid pose."""

    def __init__(self, run: WhaleRun, ids: list[str], crop: int = 512, min_visible: int = 300, margin_range=(1.25, 2.2),
                 jitter: float = 0.12, flip: bool = True, photometric_strength: float = 1.0, train: bool = True, seed: int = 0,
                 eval_margin: float = 1.6, bboxes: dict[str, list] | None = None):
        self.run, self.crop, self.train, self.seed = run, crop, train, seed
        self.margin_range, self.jitter, self.flip, self.photometric_strength = margin_range, jitter, flip, photometric_strength
        self.eval_margin, self.bboxes = eval_margin, bboxes
        self.ids = [i for i in ids if run.by_id[i]["visible_pixels"] >= min_visible and run.has_bodycoords(i)]
        if bboxes is not None:
            self.ids = [i for i in self.ids if i in bboxes and bboxes[i] is not None]
        self.epoch = 0

    def __len__(self):
        return len(self.ids)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __getitem__(self, index: int):
        shot_id = self.ids[index]
        run = self.run
        image = run.image(shot_id)
        mask = run.mask(shot_id)
        u, theta, defined = run.bodycoords(shot_id)
        meta = run.meta(shot_id)
        rng = np.random.default_rng([self.seed, self.epoch, index])
        C = self.crop
        if self.train:
            margin = rng.uniform(*self.margin_range)
            x0, y0, s = square_crop_params(bbox_of(run.primary(shot_id, mask)), C, margin, self.jitter, rng)
        else:
            bbox = self.bboxes[shot_id] if self.bboxes is not None else bbox_of(run.primary(shot_id, mask))
            x0, y0, s = square_crop_params(bbox, C, self.eval_margin, 0.0, None)
        M = warp_params(x0, y0, s)
        img = warp_image(image, M, (C, C))
        m = warp_nearest(mask, M, (C, C), IGNORE)
        uc = warp_nearest(u, M, (C, C), 0.0)
        tc = warp_nearest(theta, M, (C, C), 0.0)
        dc = warp_nearest(defined.astype(np.uint8), M, (C, C), 0)
        R, t = geometry.gt_camera_from_whale(meta)
        ppm = float(meta["pixels_per_metre_at_whale"]) * s * float(meta.get("body_scale", 1.0))   # per unit-mesh metre
        flipped = False
        if self.train and self.flip and rng.random() < 0.5:
            flipped = True
            img, m, uc, tc, dc = img[:, ::-1], m[:, ::-1], uc[:, ::-1], tc[:, ::-1], dc[:, ::-1]
            tc = (360.0 - tc) % 360.0
            R = geometry.flip_rotation(R)
        if self.train:
            img = photometric(np.ascontiguousarray(img), rng, self.photometric_strength)
        valid = (m == 1) & (dc > 0)
        rad = np.radians(tc)
        sample = {
            "id": shot_id,
            "image": to_tensor(img),
            "mask": torch.from_numpy(np.ascontiguousarray(m).astype(np.int64)),
            "u": torch.from_numpy(np.ascontiguousarray(uc)).float(),
            "theta_deg": torch.from_numpy(np.ascontiguousarray(tc)).float(),
            "sincos": torch.from_numpy(np.ascontiguousarray(np.stack([np.sin(rad), np.cos(rad)], 0))).float(),
            "valid": torch.from_numpy(np.ascontiguousarray(valid)),
            "R": torch.from_numpy(R).float(),
            "d6": torch.from_numpy(geometry.rotation_to_6d(R)),
            "log_ppm": torch.tensor(math.log(ppm), dtype=torch.float32),
            "crop_x0y0s": torch.tensor([x0, y0, s], dtype=torch.float64),
            "flipped": flipped,
        }
        return sample


# ---- run03 keypoint labels (see run03_labels.py) ----
KP_STATE_CLASS = {"visible": 0, "under_water": 1, "hidden": 2, "occluded": 2, "out_of_frame": 3}   # the model's 4 classes; missing -> -1
DEFORM_FLIP_SIGN = np.array([1, 1, 1, -1, -1, 1, 1], np.float32)   # arch, tail_curl, head_pitch, lateral, fluke_twist, fin_raise, fin_sweep


def keypoint_targets(meta: dict, x0: float, y0: float, s: float, crop: int, flipped: bool, heat_size: int, sigma: float = 2.5):
    """Keypoint targets for a crop: uv (K, 2) crop pixels (nan when missing), state class (K,) with
    -1 = missing, Gaussian heatmaps (K, heat_size, heat_size) for the visible keypoints inside the crop,
    on the stride crop / heat_size grid. Mirrored crops swap the left/right keypoints."""
    from .run03_labels import FLIP_PAIRS, KEYPOINTS

    K = len(KEYPOINTS)
    uv = np.full((K, 2), np.nan, np.float32); st = np.full(K, -1, np.int64)
    kps = meta.get("keypoints") or {}
    for i, name in enumerate(KEYPOINTS):
        k = kps.get(name)
        if not k or "state" not in k:
            continue
        uv[i] = ((k["uv"][0] - x0) * s, (k["uv"][1] - y0) * s)
        st[i] = KP_STATE_CLASS.get(k["state"], -1)
    if flipped:
        uv[:, 0] = crop - uv[:, 0]
        for a, b in FLIP_PAIRS:
            i, j = KEYPOINTS.index(a), KEYPOINTS.index(b)
            uv[[i, j]] = uv[[j, i]]; st[[i, j]] = st[[j, i]]
    inside = np.isfinite(uv).all(1) & (uv >= 0).all(1) & (uv < crop).all(1)
    st[(st == 0) & ~inside] = 3                                  # a visible keypoint cut off by the crop is out of frame
    stride = crop / heat_size
    heat = np.zeros((K, heat_size, heat_size), np.float32)
    gy, gx = np.mgrid[0:heat_size, 0:heat_size]
    for i in np.nonzero(st == 0)[0]:
        cx, cy = uv[i] / stride - 0.5
        heat[i] = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sigma ** 2))
    return uv, st, heat


def deform_target(meta: dict, flipped: bool, scale_deg: float = 30.0) -> np.ndarray:
    from .run03_labels import DEFORM_PARAMS

    d = meta.get("deform") or {}
    v = np.array([float(d.get(k, 0.0)) for k in DEFORM_PARAMS], np.float32) / scale_deg
    return v * DEFORM_FLIP_SIGN if flipped else v


class CanonCropDataset(Dataset):
    """Whale-centred crops with the canonical surface point per pixel as the dense target.

    Targets per pixel (where the visible whale has a label): "xyz" = the folded, normalised
    canonical point ((x - x_min) / L, |y| / L, z / L) with L the body length, "side" = sign of
    y (+1 right, -1 left, 0 on the midline), "valid". The side is NOT part of the regression
    target: a mirrored image of this bilaterally symmetric body has the same folded coordinates,
    so a horizontal flip leaves "xyz" unchanged and only mirrors "side" and the rotation. Also
    returns the rigid pose R, log pixels-per-metre and the crop transform, like BodyCropDataset."""

    def __init__(self, run: WhaleRun, ids: list[str], table: dict, crop: int = 512, min_visible: int = 300, margin_range=(1.25, 2.2),
                 jitter: float = 0.12, flip: bool = True, photometric_strength: float = 1.0, train: bool = True, seed: int = 0,
                 eval_margin: float = 1.6, bboxes: dict[str, list] | None = None, keypoints: bool = False, heat_stride: int = 4):
        self.run, self.crop, self.train, self.seed = run, crop, train, seed
        self.margin_range, self.jitter, self.flip, self.photometric_strength = margin_range, jitter, flip, photometric_strength
        self.eval_margin, self.bboxes = eval_margin, bboxes
        self.keypoints, self.heat_stride = keypoints, heat_stride     # run03 targets: heatmaps, keypoint states, deformation
        self.x_min_cm, self.length_cm = float(table["x_min_m"]) * 100.0, float(table["length_m"]) * 100.0
        self.ids = [i for i in ids if run.by_id[i]["visible_pixels"] >= min_visible and run.has_canon(i)]
        if bboxes is not None:
            self.ids = [i for i in self.ids if i in bboxes and bboxes[i] is not None]
        self.epoch = 0

    def __len__(self):
        return len(self.ids)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def fold(self, canon_cm: np.ndarray) -> np.ndarray:
        """(H, W, 3) cm -> (3, H, W) folded, normalised target (nan -> 0)."""
        x = (canon_cm[..., 0] - self.x_min_cm) / self.length_cm
        y = np.abs(canon_cm[..., 1]) / self.length_cm
        z = canon_cm[..., 2] / self.length_cm
        return np.nan_to_num(np.stack([x, y, z], 0)).astype(np.float32)

    def __getitem__(self, index: int):
        shot_id = self.ids[index]
        run = self.run
        image = run.image(shot_id)
        mask = run.mask(shot_id)
        canon_cm, defined = run.canon(shot_id)
        contours = run.contours_or_zeros(shot_id)
        meta = run.meta(shot_id)
        rng = np.random.default_rng([self.seed, self.epoch, index])
        C = self.crop
        if self.train:
            margin = rng.uniform(*self.margin_range)
            x0, y0, s = square_crop_params(bbox_of(run.primary(shot_id, mask)), C, margin, self.jitter, rng)
        else:
            bbox = self.bboxes[shot_id] if self.bboxes is not None else bbox_of(run.primary(shot_id, mask))
            x0, y0, s = square_crop_params(bbox, C, self.eval_margin, 0.0, None)
        M = warp_params(x0, y0, s)
        img = warp_image(image, M, (C, C))
        m = warp_nearest(mask, M, (C, C), IGNORE)
        ct = warp_nearest(contours, M, (C, C), 0)
        cc = np.stack([warp_nearest(np.nan_to_num(canon_cm[..., k]), M, (C, C), 0.0) for k in range(3)], -1)
        dc = warp_nearest(defined.astype(np.uint8), M, (C, C), 0) > 0
        R, t = geometry.gt_camera_from_whale(meta)
        ppm = float(meta["pixels_per_metre_at_whale"]) * s * float(meta.get("body_scale", 1.0))   # per unit-mesh metre
        flipped = False
        if self.train and self.flip and rng.random() < 0.5:
            flipped = True
            img, m, cc, dc, ct = img[:, ::-1], m[:, ::-1], cc[:, ::-1], dc[:, ::-1], ct[:, ::-1]
            cc = cc * np.array([1.0, -1.0, 1.0], np.float32)       # the mirror image shows the mirrored body point
            R = geometry.flip_rotation(R)
        if self.train:
            img = photometric(np.ascontiguousarray(img), rng, self.photometric_strength)
        valid = (m == 1) & dc
        side = np.sign(cc[..., 1]).astype(np.int8)
        side[np.abs(cc[..., 1]) < 2.0] = 0                           # within 2 cm of the midline: no side
        side[~dc] = 0
        sample = {
            "id": shot_id,
            "image": to_tensor(np.ascontiguousarray(img)),
            "mask": torch.from_numpy(np.ascontiguousarray(m).astype(np.int64)),
            "contours": torch.from_numpy(np.ascontiguousarray(ct).astype(np.uint8)),
            "xyz": torch.from_numpy(self.fold(np.where(dc[..., None], cc, np.nan))),
            "side": torch.from_numpy(np.ascontiguousarray(side)),
            "valid": torch.from_numpy(np.ascontiguousarray(valid)),
            "R": torch.from_numpy(R).float(),
            "d6": torch.from_numpy(geometry.rotation_to_6d(R)),
            "log_ppm": torch.tensor(math.log(ppm), dtype=torch.float32),
            "crop_x0y0s": torch.tensor([x0, y0, s], dtype=torch.float64),
            "flipped": flipped,
        }
        if self.keypoints:
            uv, st, heat = keypoint_targets(meta, x0, y0, s, C, flipped, C // self.heat_stride)
            sample.update({"kp_uv": torch.from_numpy(uv), "kp_state": torch.from_numpy(st), "kp_heat": torch.from_numpy(heat),
                           "deform": torch.from_numpy(deform_target(meta, flipped))})
        return sample


class RealCropDataset(Dataset):
    """Unlabelled real photos with predicted masks, cropped like the synthetic training crops
    (mask bbox, random margin and jitter, photometric augmentation). Used for the label-free
    mirror-consistency loss: the folded canonical coordinates of a crop and of its mirror image
    must agree. Returns the crop and the mask warped into it."""

    def __init__(self, images_dir: Path, masks_dir: Path, crop: int = 512, margin_range=(1.25, 2.2), jitter: float = 0.12,
                 photometric_strength: float = 1.0, min_pixels: int = 2000, seed: int = 0):
        self.images_dir, self.masks_dir, self.crop = Path(images_dir), Path(masks_dir), crop
        self.margin_range, self.jitter, self.photometric_strength, self.seed = margin_range, jitter, photometric_strength, seed
        self.items = []
        for f in sorted(self.images_dir.glob("*.jpg")):
            m = self.masks_dir / f"{f.stem}_mask.png"
            if m.exists():
                mask = cv2.imread(str(m), cv2.IMREAD_UNCHANGED)
                if mask is not None and (mask > 0).sum() >= min_pixels:
                    self.items.append((f, m))
        self.epoch = 0

    def __len__(self):
        return len(self.items)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __getitem__(self, index: int):
        f, m = self.items[index]
        image = cv2.cvtColor(cv2.imread(str(f), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(m), cv2.IMREAD_UNCHANGED) > 0
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        rng = np.random.default_rng([self.seed + 7919, self.epoch, index])
        C = self.crop
        x0, y0, s = square_crop_params(bbox_of(mask), C, rng.uniform(*self.margin_range), self.jitter, rng)
        M = warp_params(x0, y0, s)
        img = photometric(np.ascontiguousarray(warp_image(image, M, (C, C))), rng, self.photometric_strength)
        mc = cv2.warpAffine(mask.astype(np.uint8), M, (C, C), flags=cv2.INTER_NEAREST) > 0
        return {"id": f.stem, "image": to_tensor(img), "whale": torch.from_numpy(np.ascontiguousarray(mc))}

