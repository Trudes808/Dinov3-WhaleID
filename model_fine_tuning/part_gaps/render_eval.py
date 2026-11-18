"""Run a part model on RENDERS and summarise it per rendered whale, in the same form as the real-photo instances
(full frame at long side 1280; per-whale window 1.5 x its box at up to 768 px), so the same three-way rule and the
same taxonomy can be read on renders as on real photos.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.part_gaps.render_eval --dataset <renders>/r06r_test_final/dataset \
        [--model parts_seg_r05_a] [--n 300] [--split val] [--device cpu --threads 12]

Truth per whale: run06 datasets = dataset/parts/<id>.png restricted to the whale's id in dataset/instances/<id>.png
(factory parts: 1 fluke, 2 dorsal fin incl. hump, 3 pectoral, 4 head, 5 body); run05 = the parts_seg label cache
(/var/tmp/sat3737-parts-cache/run05/labels, the labels the model was trained on; 255 ignore) with instances id 1.
Per whale: meta category / pose / scene, the visible box, truth pixels per class, predicted pixels per class inside
the whale mask grown by 2 px (as "pixels_near" on real photos) and the share of the whale's pixels, for both variants.
Output: /var/tmp/sat3737-parts-cache/part_gaps/renders/<dataset tag>/<model>.jsonl (one line per whale). Resumable.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from ..common import paths
from ..common.data import pad_to_multiple, to_tensor
from ..parts_seg.model import PartsSegModel
from ..parts_seg.predict import run_model
from . import common as C

K = 6
RUN05_LABELS = Path("/var/tmp/sat3737-parts-cache/run05/labels")


def dataset_tag(ds: Path) -> str:
    return ds.parent.name if ds.name == "dataset" else ds.name


def shot_ids(ds: Path, n: int | None, split: str | None, seed: int = 0) -> list[str]:
    man = [json.loads(l) for l in open(ds / "manifest.jsonl") if l.strip()]
    ids = [m["id"] for m in man if not m.get("negative") and not m.get("rejected") and (m.get("visible_pixels", 1) or 0) > 0]
    if split:
        sp = json.loads((paths.SPLITS_DIR / f"{dataset_tag(ds)}.json").read_text()) if (paths.SPLITS_DIR / f"{dataset_tag(ds)}.json").exists() else None
        if sp is not None and split in sp:
            keep = set(sp[split]) if isinstance(sp[split], list) else set(sp[split].get("ids", []))
            if keep:
                ids = [i for i in ids if i in keep]
    if n and len(ids) > n:
        ids = sorted(random.Random(seed).sample(ids, n))
    return ids


def truth_maps(ds: Path, sid: str, H: int, W: int):
    """(parts uint8 HxW with 255 ignore, instances uint8 HxW)"""
    inst = np.asarray(Image.open(ds / "instances" / f"{sid}.png"))
    if inst.ndim == 3:
        inst = inst[..., 0]
    if (ds / "parts" / f"{sid}.png").exists():
        parts = np.asarray(Image.open(ds / "parts" / f"{sid}.png"))
    else:
        parts = np.asarray(Image.open(RUN05_LABELS / f"{sid}.png"))
    if parts.ndim == 3:
        parts = parts[..., 0]
    if parts.shape != (H, W):
        parts = cv2.resize(parts, (W, H), interpolation=cv2.INTER_NEAREST)
    if inst.shape != (H, W):
        inst = cv2.resize(inst, (W, H), interpolation=cv2.INTER_NEAREST)
    parts = blade_only(ds, sid, parts, inst)
    return parts, inst


HUMP_STATS: dict = {}


def blade_only(ds: Path, sid: str, parts: np.ndarray, inst: np.ndarray) -> np.ndarray:
    """From r06r_test_r6b on, the factory's dorsal class is fin + hump, with the blade as a sub-label a{k}_blade in
    animals/<id>.npz (cropped to a{k}_box, k = instance id). The r05 part models were trained on the blade (+ a thin ridge
    strip), so for the three-way truth and the taxonomy the hump goes to body; the hump pixels per whale are kept in
    HUMP_STATS[(sid, k)] = (fin+hump px, blade px). Datasets without a{k}_blade are returned unchanged."""
    p = ds / "animals" / f"{sid}.npz"
    if not p.exists():
        return parts
    z = np.load(p)
    if not any(k.endswith("_blade") for k in z.files):
        return parts
    parts = parts.copy()
    for k in range(1, int(z["n_animals"]) + 1):
        if f"a{k}_blade" not in z.files:
            continue
        x0, y0, x1, y1 = [int(v) for v in z[f"a{k}_box"]]
        bl = np.zeros(parts.shape, bool)
        b = z[f"a{k}_blade"].astype(bool)
        bl[y0:y0 + b.shape[0], x0:x0 + b.shape[1]] = b[: parts.shape[0] - y0, : parts.shape[1] - x0]
        fin = (parts == 2) & (inst == k)
        HUMP_STATS[(sid, k)] = (int(fin.sum()), int((fin & bl).sum()))
        parts[fin & ~bl] = 5
    return parts


def animals_meta(meta: dict) -> dict:
    out = {}
    if meta.get("animals"):
        for a in meta["animals"]:
            out[int(a["instance_id"])] = {"category": a.get("category"), "role": a.get("role"),
                                         "pose": (a.get("pose") or {}).get("description"), "orientation": a.get("orientation"),
                                         "visible_fraction": a.get("visible_fraction_of_whale"),
                                         "parts_visible": a.get("parts_visible")}
    else:
        out[1] = {"category": meta.get("category"), "role": "primary", "pose": (meta.get("pose") or {}).get("description") if isinstance(meta.get("pose"), dict) else meta.get("pose"),
                  "orientation": meta.get("orientation"), "visible_fraction": meta.get("visible_fraction_of_whale"), "parts_visible": None}
    return out


def counts(cls_map: np.ndarray, mask: np.ndarray) -> list[int]:
    return np.bincount(cls_map[mask], minlength=256)[:K].astype(int).tolist()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--model", default="parts_seg_r05_a")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--split", default=None, help="for run05: 'val' = the shots the model never trained on")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--min-px", type=int, default=400, help="whales with fewer visible pixels are skipped")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    ds = a.dataset
    tag = dataset_tag(ds)
    out = C.CACHE / "renders" / tag / f"{a.model}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["shot"] for l in open(out) if l.strip()}
    ids = [i for i in shot_ids(ds, a.n, a.split) if i not in done]
    ck = a.checkpoint or (paths.EXPERIMENTS_DIR / a.model / "checkpoints" / "best.pt")
    device = torch.device(a.device)
    model, _ = PartsSegModel.from_checkpoint(ck, device)
    model.eval()
    t0 = time.time()
    with open(out, "a") as f:
        for n, sid in enumerate(ids):
            meta = json.loads((ds / "meta" / f"{sid}.json").read_text())
            rgb = cv2.cvtColor(cv2.imread(str(ds / "images" / f"{sid}.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            H, W = rgb.shape[:2]
            parts, inst = truth_maps(ds, sid, H, W)
            sc = 1280 / max(H, W)
            small = cv2.resize(rgb, (int(round(W * sc)), int(round(H * sc))), interpolation=cv2.INTER_AREA)
            padded, (h, w) = pad_to_multiple(small)
            full = run_model(model, to_tensor(padded), device)[:h, :w]
            full_up = cv2.resize(full, (W, H), interpolation=cv2.INTER_NEAREST)
            am = animals_meta(meta)
            k2560 = 2560.0 / W                     # native px -> the 2560-wide frame of the real photos
            iids = [int(v) for v in np.unique(inst) if v > 0]
            if not meta.get("animals"):             # run04 / run05: id 2 is not a separate whale: primary only
                iids = [i for i in iids if i == 1]
            for iid in iids:
                m = inst == iid
                npx = int(m.sum())
                if npx < a.min_px:
                    continue
                ys, xs = np.nonzero(m)
                box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                near = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                rec = {"shot": sid, "iid": iid, "W": W, "H": H, "box": box, "L2560": max(box[2] - box[0], box[3] - box[1]) * k2560,
                       "px": npx, "touches_border": bool(box[0] <= 0 or box[1] <= 0 or box[2] >= W or box[3] >= H),
                       "n_animals": len(am), "scene": (meta.get("scene") or {}).get("type") if isinstance(meta.get("scene"), dict) else meta.get("scene"),
                       **{f"meta_{k}": v for k, v in (am.get(iid) or {}).items()},
                       "truth": counts(parts, m), "fin_hump_blade_px": HUMP_STATS.get((sid, iid)), "full": counts(full_up, near)}
                # hr window: 1.5 x the box, at least 192 px of the 2560 frame, output <= 768 and <= 2x the 2560 resolution
                x1, y1, x2, y2 = box
                side = max(max(x2 - x1, y2 - y1) * 1.5, 192.0 / k2560)
                osz = int(min(768, max(256, round(side * k2560 * 2 / 16) * 16)))
                hx, hy = (x1 + x2) / 2 - side / 2, (y1 + y2) / 2 - side / 2
                M = np.array([[osz / side, 0, -hx * osz / side], [0, osz / side, -hy * osz / side]])
                crop = cv2.warpAffine(rgb, M, (osz, osz), flags=cv2.INTER_AREA if side > osz else cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                hr = run_model(model, to_tensor(crop), device)
                Mi = cv2.invertAffineTransform(M)
                hr_up = cv2.warpAffine(hr, Mi, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
                rec["hr"] = counts(hr_up, near & (hr_up != 255))
                rec["hr_window"] = [float(hx), float(hy), float(side), osz]
                f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 10 == 0:
                print(f"{tag} {n + 1}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
