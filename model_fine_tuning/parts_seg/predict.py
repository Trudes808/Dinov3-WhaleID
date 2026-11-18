"""Run the part segmentation on the real photos of whole days and summarise it per whale instance.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.parts_seg.predict --checkpoint experiments/parts_seg_r05_a/checkpoints/best.pt \
        --days train,val [--out-name parts_seg_r05_a]

Two variants per photo:
  full  the whole 2560-px jpg at long side 1280 (as the whale segmentation runs), one class map per photo:
        DAY/<out-name>/full/<stem>.png (palette, 1280 px wide)
  hr    per whale instance, a square around its box (1.5 x the long side, at least 192 px of the 2560 image) at
        768 px (never more than 2 x the 2560 image's resolution): DAY/<out-name>/hr/<stem>~k.png, with the window in
        the json
The instances are those of the pipeline: the blobs of DAY/cse/<stem>_canon.npz "use" split by cluster.parts.split_instances
with the body fraction u (as experiments/bodymap_review/bodymap_stats.py), keys "<stem>~k" as in parts_hr/index.jsonl.
Per day DAY/<out-name>/instances.json: per instance key and variant the predicted pixels per class inside the instance
(pixels of that variant's class map: "pixels" inside the instance mask warped into the map, "pixels_near" inside it
grown by one cse-crop pixel + 2 px; "map_px_per_2560px" gives the scale), and the box (2560 px) and centroid of the
largest dorsal-fin region and of the fluke region. Nothing else in the day folder is written.
--stats-only rebuilds instances.json from the saved class maps without the model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..common import geometry, paths
from ..common.data import pad_to_multiple, to_tensor
from .labels import CLASSES, PALETTE
from .model import PartsSegModel

DAYS = paths.EXPERIMENTS_DIR / "days_all"
EVAL_DAYS = {"2024-12-18", "2024-12-26", "2025-02-19", "2025-03-21"}
K = len(CLASSES)
HR_SIDE = 768


def day_list(spec: str) -> list[str]:
    split = json.loads((paths.SPLITS_DIR / "days.json").read_text())
    days = [d for x in spec.split(",") for d in (split[x] if x in split else [x])]
    bad = [d for d in days if d in EVAL_DAYS]
    if bad:
        raise SystemExit(f"evaluation days must not be read: {bad}")
    return days


def save_palette(label: np.ndarray, path: Path) -> None:
    im = Image.fromarray(label.astype(np.uint8), mode="P")
    pal = PALETTE + [0, 0, 0] * (256 - len(PALETTE) // 3)
    im.putpalette(pal)
    im.save(path, optimize=True)


def instances_of(day: str, stem: str, table: dict):
    from ..cluster.parts import split_instances

    z = np.load(DAYS / day / "cse" / f"{stem}_canon.npz")
    use = z["use"]
    x0, y0, s = [float(v) for v in z["x0y0s"]]
    if use.sum() < 400:
        return [], (x0, y0, s)
    signed = z["signed_mm"].astype(np.float32) / 1000.0
    u = (signed[..., 0] - table["x_min_m"]) / table["length_m"]
    return split_instances(use, 0.1, u=u), (x0, y0, s)


def hr_window(bbox: list[float], W: int, H: int) -> tuple[float, float, float, int]:
    """(x0, y0, side in 2560 px, output size) of the high-resolution window around an instance box."""
    x1, y1, x2, y2 = bbox
    side = max(max(x2 - x1, y2 - y1) * 1.5, 192.0)
    out = int(min(HR_SIDE, max(256, round(side * 2 / 16) * 16)))
    return (x1 + x2) / 2 - side / 2, (y1 + y2) / 2 - side / 2, side, out


class DayPhotos(Dataset):
    def __init__(self, day: str, recs: list[dict]):
        self.day, self.recs = day, recs

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        rec = self.recs[i]
        bgr = cv2.imread(str(DAYS / self.day / "images" / f"{rec['stem']}.jpg"), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        sc = 1280 / max(H, W)
        small = cv2.resize(rgb, (int(round(W * sc)), int(round(H * sc))), interpolation=cv2.INTER_AREA)
        crops = []
        for inst in rec.get("instances", []):
            if not inst.get("bbox_2560"):
                crops.append(None)
                continue
            hx, hy, side, out = hr_window(inst["bbox_2560"], W, H)
            M = np.array([[out / side, 0, -hx * out / side], [0, out / side, -hy * out / side]])
            crops.append((to_tensor(cv2.warpAffine(rgb, M, (out, out), flags=cv2.INTER_AREA if side > out else cv2.INTER_LINEAR,
                                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)), [hx, hy, side, out]))
        return {"stem": rec["stem"], "small": small, "scale": sc, "hw": (H, W), "crops": crops}


def _collate(b):
    return b[0]


@torch.no_grad()
def run_model(model, img01: torch.Tensor, device) -> np.ndarray:
    x = img01[None].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(x).float()
    return logits[0].argmax(0).to(torch.uint8).cpu().numpy()


def region_stats(cls_map: np.ndarray, inst_crop: np.ndarray, x0: float, y0: float, s: float, a: float, ox: float, oy: float) -> dict:
    """Class counts of one instance, in the class map's own frame (map px = (p2560 - (ox, oy)) * a), and the fin /
    fluke regions (box and centroid in 2560 px). The instance mask (cse crop frame, crop = (p2560 - (x0, y0)) * s) is
    warped into the map's frame; "pixels" counts inside it, "pixels_near" inside it grown by one crop pixel + 2 px
    (thin parts such as a fin seen from behind stick out of the coarse mask). Shares for the decision use pixels_near."""
    H, W = cls_map.shape
    M = np.array([[a / s, 0, (x0 - ox) * a], [0, a / s, (y0 - oy) * a]])
    inst = cv2.warpAffine(inst_crop.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0
    r = int(np.ceil(a / s)) + 2
    near = cv2.dilate(inst.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
    known = cls_map != 255
    c_in = np.bincount(cls_map[inst & known], minlength=K)[:K]
    c_near = np.bincount(cls_map[near & known], minlength=K)[:K]
    out = {"pixels": {CLASSES[k]: int(c_in[k]) for k in range(K)}, "pixels_near": {CLASSES[k]: int(c_near[k]) for k in range(K)},
           "area": int(inst.sum()), "covered": int((near & known).sum()), "map_px_per_2560px": float(a)}

    def to2560(xs, ys):
        return [float(xs.min() / a + ox), float(ys.min() / a + oy), float((xs.max() + 1) / a + ox), float((ys.max() + 1) / a + oy)], \
               [float(xs.mean() / a + ox), float(ys.mean() / a + oy)]

    fin = ((cls_map == 2) & near).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fin, connectivity=8)
    if n > 1:
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        ys, xs = np.nonzero(lab == k)
        box, c = to2560(xs, ys)
        out["fin"] = {"box": box, "centroid": c, "px": int(len(xs)), "regions": int(n - 1)}
    fl = ((cls_map == 1) & near).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fl, connectivity=8)
    if n > 1:
        areas = st[1:, cv2.CC_STAT_AREA]
        keep = 1 + np.nonzero(areas >= 0.1 * areas.max())[0]
        ys, xs = np.nonzero(np.isin(lab, keep))
        box, c = to2560(xs, ys)
        out["fluke"] = {"box": box, "centroid": c, "px": int(len(xs)), "regions": int(len(keep))}
    return out


def instance_records(day: str, stem: str, table: dict, full: np.ndarray, full_a: float, hr_maps: dict, windows: dict) -> dict:
    """Per instance key: stats of the full-frame map and of its high-resolution map."""
    try:
        masks, (x0, y0, s) = instances_of(day, stem, table)
    except FileNotFoundError:
        return {}
    res = {}
    for k, m in enumerate(masks):
        key = f"{stem}~{k}"
        rec = {"full": region_stats(full, m, x0, y0, s, full_a, 0.0, 0.0), "cse_x0y0s": [x0, y0, s]}
        if key in hr_maps:
            hx, hy, side, osz = windows[key]
            rec["hr"] = region_stats(hr_maps[key], m, x0, y0, s, osz / side, hx, hy)
            rec["hr_window"] = [float(hx), float(hy), float(side), int(osz)]
        res[key] = rec
    return res


def stats_from_pngs(day: str, out_name: str, table: dict) -> dict:
    """Rebuild DAY/<out-name>/instances.json from the saved class maps (no model needed)."""
    d = DAYS / day
    out = d / out_name
    recs = [json.loads(l) for l in open(d / "parts_hr" / "index.jsonl") if l.strip()]
    result = {}
    for rec in recs:
        stem = rec["stem"]
        p = out / "full" / f"{stem}.png"
        if not p.exists():
            continue
        full = np.asarray(Image.open(p))
        W = Image.open(d / "images" / f"{stem}.jpg").size[0]
        hr_maps, windows = {}, {}
        for inst in rec.get("instances", []):
            hp = out / "hr" / f"{inst['key']}.png"
            if hp.exists() and inst.get("bbox_2560"):
                hr_maps[inst["key"]] = np.asarray(Image.open(hp))
                hx, hy, side, osz = hr_window(inst["bbox_2560"], W, 0)
                windows[inst["key"]] = (hx, hy, side, osz)
        result.update(instance_records(day, stem, table, full, full.shape[1] / W, hr_maps, windows))
    (out / "instances.json").write_text(json.dumps({"day": day, "model": out_name, "classes": CLASSES, "instances": result}))
    return result


def process_day(model, day: str, out_name: str, table: dict, device, workers: int = 4, limit: int | None = None) -> dict:
    d = DAYS / day
    out = d / out_name
    (out / "full").mkdir(parents=True, exist_ok=True)
    (out / "hr").mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in open(d / "parts_hr" / "index.jsonl") if l.strip()][:limit]
    recs_by_stem = {r["stem"]: r for r in recs}
    loader = DataLoader(DayPhotos(day, recs), batch_size=1, num_workers=workers, collate_fn=_collate)
    result = {}
    for item in loader:
        stem = item["stem"]
        small = item["small"]
        padded, (h, w) = pad_to_multiple(small)
        full = run_model(model, to_tensor(padded), device)[:h, :w]
        save_palette(full, out / "full" / f"{stem}.png")
        hr_maps, windows = {}, {}
        for k, inst in enumerate(recs_by_stem[stem].get("instances", [])):
            crop = item["crops"][k] if k < len(item["crops"]) else None
            if crop is None:
                continue
            t, win = crop
            hr = run_model(model, t, device)
            save_palette(hr, out / "hr" / f"{inst['key']}.png")
            hr_maps[inst["key"]], windows[inst["key"]] = hr, win
        result.update(instance_records(day, stem, table, full, item["scale"], hr_maps, windows))
    (out / "instances.json").write_text(json.dumps({"day": day, "model": out_name, "classes": CLASSES, "instances": result}))
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--stats-only", action="store_true", help="rebuild instances.json from the saved class maps")
    ap.add_argument("--days", default="train,val")
    ap.add_argument("--out-name", default="parts_seg_r05_a")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="debug: photos per day")
    a = ap.parse_args(argv)
    t = geometry.load_folded_table(paths.FOLDED_TABLE)
    table = {"x_min_m": float(t["x_min_m"]), "length_m": float(t["length_m"])}
    if not a.stats_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = PartsSegModel.from_checkpoint(a.checkpoint, device)
    for day in day_list(a.days):
        t0 = time.time()
        res = stats_from_pngs(day, a.out_name, table) if a.stats_only else process_day(model, day, a.out_name, table, device, a.workers, a.limit)
        print(day, len(res), "instances", f"{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
