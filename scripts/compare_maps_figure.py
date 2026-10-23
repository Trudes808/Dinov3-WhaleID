"""Qualitative comparison of the body-coordinate estimates: the same real photos and synthetic test crops, one column
per approach, all drawn in one colour code. Two figures: body fraction (tail blue -> snout red, left flank darker)
and body part (fluke / peduncle / dorsal back / mid body / head, left flank darker). Synthetic rows end with the truth.

    .venv/bin/python scripts/compare_maps_figure.py --out experiments/real_2024-12-18/compare_maps [--real 6] [--synthetic 4]

Regression models (canon_r03_ft4, canon_r04_ft4, canon_r0304_ft4): folded xyz -> body fraction, side by the pipeline's
own side resolution (choose_prior_side + resolve_side). CSE models: template vertex -> body fraction and side after the
ambiguity-aware vote. Crops of the real photos are the ones of canon_canon_r04_ft4 (its masks).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from model_fine_tuning.common import geometry, paths
from model_fine_tuning.common.data import IGNORE, CanonCropDataset, WhaleRun
from model_fine_tuning.canon.cse import Template, WhaleCSEModel, predict_vertices, side_vote
from model_fine_tuning.canon.evaluate import predict_crop
from model_fine_tuning.canon.model import WhaleCanonModel
from model_fine_tuning.canon.predict_cse import crop_from, to_tensor

PART_COLOURS = np.array([[0, 0, 0], [70, 130, 255], [60, 200, 220], [60, 230, 90], [250, 200, 60], [250, 80, 80]], np.float32)
BODY_PARTS = [(0.0, 0.14), (0.14, 0.32), (0.32, 0.55), (0.55, 0.72), (0.72, 1.01)]
TILE = 300


def frac_colour(frac: np.ndarray) -> np.ndarray:
    """body fraction 0..1 -> blue..cyan..green..yellow..red"""
    c = np.array([[60, 60, 255], [60, 200, 255], [60, 255, 120], [255, 200, 60], [255, 60, 60]], np.float32)
    f = np.clip(frac, 0, 0.9999) * 4
    i = f.astype(int); t = (f - i)[:, None]
    return c[i] * (1 - t) + c[i + 1] * t


def paint(crop: np.ndarray, use: np.ndarray, frac: np.ndarray, side: np.ndarray, mode: str) -> np.ndarray:
    img = cv2.resize(crop, (TILE, TILE), interpolation=cv2.INTER_AREA).astype(np.float32)
    m = cv2.resize(use.astype(np.uint8), (TILE, TILE), interpolation=cv2.INTER_NEAREST) > 0
    fr = cv2.resize(frac.astype(np.float32), (TILE, TILE), interpolation=cv2.INTER_NEAREST)
    sd = cv2.resize(side.astype(np.float32), (TILE, TILE), interpolation=cv2.INTER_NEAREST)
    if mode == "fraction":
        col = frac_colour(fr[m])
    else:
        part = np.zeros(fr[m].shape, int)
        for k, (lo, hi) in enumerate(BODY_PARTS):
            part[(fr[m] >= lo) & (fr[m] < hi)] = k + 1
        col = PART_COLOURS[part]
    col = col * np.where(sd[m] < 0, 0.5, 1.0)[:, None]
    img[m] = 0.3 * img[m] + 0.7 * col
    return img.astype(np.uint8)


@torch.no_grad()
def regression_map(model, crop_t, use, Kc, table, device):
    pred = predict_crop(model, crop_t, device)
    pm = geometry.unfold_norm(np.moveaxis(pred["xyz"][0], 0, -1), table)
    prior, pconf = geometry.choose_prior_side(pred["R"][0], pred["xyz"][0], use)
    res = geometry.resolve_side(pm, use, Kc, R_init=pred["R"][0], table=table, prior_side=prior, prior_confidence=pconf)
    frac = (pm[..., 0] - table["x_min_m"]) / table["length_m"]
    return frac, res["side"].astype(np.float32), pred["vis"][0]


@torch.no_grad()
def cse_map(model, template, crop_t, use, device):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        a = model(crop_t)
    emb = a["emb"][0].float(); R = a["R"][0].float().cpu().numpy()
    use_t = torch.from_numpy(use).to(device)
    ids, conf, mp = predict_vertices(model, emb, use_t, template)
    _, ids_v = side_vote(ids, conf, template, R, mp)
    y, x = torch.nonzero(use_t, as_tuple=True)
    frac = np.zeros(use.shape, np.float32); side = np.zeros(use.shape, np.float32)
    frac[y.cpu().numpy(), x.cpu().numpy()] = template.body_fraction[ids_v].cpu().numpy()
    side[y.cpu().numpy(), x.cpu().numpy()] = template.side[ids_v].float().cpu().numpy()
    return frac, side, (a["vis_logit"][0].float() > 0).cpu().numpy()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--real", type=int, default=6); p.add_argument("--synthetic", type=int, default=4)
    p.add_argument("--canon-dir", type=Path, default=Path("experiments/real_2024-12-18/canon_canon_r04_ft4")); p.add_argument("--images", type=Path, default=Path("experiments/real_2024-12-18/images"))
    p.add_argument("--run-dir", type=Path, default=Path("/home/sat3737/NOAA_whales/Whale-Image-Data-Factory/data/renders/run04")); p.add_argument("--split", type=Path, default=Path("model_fine_tuning/splits/run04.json"))
    a = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = geometry.load_folded_table(paths.FOLDED_TABLE); template = Template(device=device)
    models = [("run03 map", "reg", WhaleCanonModel.from_checkpoint("experiments/canon_r03_ft4/checkpoints/best.pt", device)[0]),
              ("run04 map", "reg", WhaleCanonModel.from_checkpoint("experiments/canon_r04_ft4/checkpoints/best.pt", device)[0]),
              ("run03+04 map", "reg", WhaleCanonModel.from_checkpoint("experiments/canon_r0304_ft4/checkpoints/best.pt", device)[0]),
              ("CSE frozen", "cse", WhaleCSEModel.from_checkpoint("experiments/cse_r04_frozen/checkpoints/best.pt", device)[0]),
              ("CSE 2 blocks", "cse", WhaleCSEModel.from_checkpoint("experiments/cse_r04_ft2/checkpoints/best.pt", device)[0])]
    rows = []            # (label, crop, use, [(name, frac, side)], truth or None)
    # real photos: pick by the CSE predictor's pose class when its rows exist, else the first N with enough pixels
    cse_rows = Path("experiments/real_2024-12-18/cse_r04_ft2_canon/rows.jsonl")
    order = []
    if cse_rows.exists():
        recs = [json.loads(l) for l in open(cse_rows)]
        by = {}
        for r in recs:
            mix = r.get("part_mix") or {}
            cls = "fluke" if mix.get("fluke", 0) > 0.5 else ("head" if mix.get("head", 0) > 0.5 else "back")
            by.setdefault(cls, []).append(r["stem"])
        want = {"back": max(a.real - 3, 2), "fluke": 2, "head": 1}
        for cls, k in want.items():
            order += by.get(cls, [])[:k]
    files = sorted(a.canon_dir.glob("*_canon.npz"))
    stems = order or [f.name[: -len("_canon.npz")] for f in files[: a.real]]
    ts = json.loads((a.images.parent / "timestamps.json").read_text()) if (a.images.parent / "timestamps.json").exists() else {}
    from model_fine_tuning.canon.outline_metric import SENSOR_WIDTH_MM
    for stem in stems[: a.real]:
        z = np.load(a.canon_dir / f"{stem}_canon.npz"); use = z["use"]
        if use.sum() < 300:
            continue
        crop = crop_from(a.images / f"{stem}.jpg", z["x0y0s"], use.shape[0]); x0, y0, s = z["x0y0s"]
        img = cv2.imread(str(a.images / f"{stem}.jpg")); H, W = img.shape[:2]; meta = ts.get(stem, {}); focal = float(meta["focal_mm"]) if meta.get("focal_mm") else None
        Kc = geometry.intrinsics_for_crop(geometry.intrinsics_guess(W, H, focal, SENSOR_WIDTH_MM.get(meta.get("camera"), 36.0)), x0, y0, s)
        crop_t = to_tensor(crop, device); maps = []
        for name, kind, model in models:
            frac, side, _ = regression_map(model, crop_t, use, Kc, table, device) if kind == "reg" else cse_map(model, template, crop_t, use, device)
            maps.append((name, frac, side))
        rows.append((f"real {stem[-8:]}", crop, use, maps, None))
    # synthetic test crops with truth, one per chosen category
    run = WhaleRun(a.run_dir, a.split)
    cats = ["back_arch", "fluke_up", "blow", "logging", "fluke_down", "pec_up"][: a.synthetic]
    ids = []
    for c in cats:
        pool = [i for i in run.ids("test") if run.by_id[i]["category"] == c and run.by_id[i]["visible_pixels"] > 3000 and run.has_canon(i)]
        if pool:
            ids.append(pool[len(pool) // 2])
    ds = CanonCropDataset(run, ids, table, crop=512, min_visible=300, train=False)
    for batch in DataLoader(ds, batch_size=1):
        sid = batch["id"][0]; meta = run.meta(sid); mask = batch["mask"][0].numpy(); use = mask == 1
        crop = (batch["image"][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8); x0, y0, s = batch["crop_x0y0s"][0].numpy()
        Kc = geometry.intrinsics_for_crop(geometry.intrinsics_from_meta(meta), x0, y0, s); crop_t = batch["image"].to(device); maps = []
        for name, kind, model in models:
            frac, side, _ = regression_map(model, crop_t, use, Kc, table, device) if kind == "reg" else cse_map(model, template, crop_t, use, device)
            maps.append((name, frac, side))
        gt_frac = batch["xyz"][0, 0].numpy(); gt_side = batch["side"][0].numpy().astype(np.float32)
        rows.append((f"synthetic {meta['category']}", crop, use, maps, (gt_frac, gt_side)))
    a.out.mkdir(parents=True, exist_ok=True)
    header = ["photo + mask"] + [n for n, _, _ in models] + ["truth"]
    for mode in ("fraction", "parts"):
        grid = []
        for label, crop, use, maps, truth in rows:
            base = cv2.resize(crop, (TILE, TILE), interpolation=cv2.INTER_AREA).astype(np.float32)
            m = cv2.resize(use.astype(np.uint8), (TILE, TILE), interpolation=cv2.INTER_NEAREST) > 0
            base[m] = 0.6 * base[m] + 0.4 * np.array([255, 60, 60]); base = base.astype(np.uint8)
            cv2.putText(base, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles = [base] + [paint(crop, use, frac, side, mode) for _, frac, side in maps]
            tiles.append(paint(crop, use, truth[0], truth[1], mode) if truth is not None else np.full((TILE, TILE, 3), 25, np.uint8))
            grid.append(np.concatenate(tiles, 1))
        band = np.full((26, TILE * len(header), 3), 255, np.uint8)
        for i, h in enumerate(header):
            cv2.putText(band, h, (i * TILE + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        legend = np.full((26, TILE * len(header), 3), 255, np.uint8)
        txt = ("body fraction: blue = flukes .. cyan = peduncle .. green = dorsal back .. yellow = mid body .. red = head; darker = left flank" if mode == "fraction"
               else "parts: blue fluke, cyan peduncle, green dorsal back, yellow mid body, red head; darker = left flank")
        cv2.putText(legend, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        sheet = np.concatenate([band] + grid + [legend], 0)
        out = a.out / f"compare_maps_{mode}.jpg"
        cv2.imwrite(str(out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88]); print("wrote", out, sheet.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
