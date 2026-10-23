"""Fin / fluke landmark detector trained on factory part crops.

One network for both parts: a DINOv3 ViT-B/16 backbone with the shared dense decoder, predicting on a crop
around the dorsal fin or the flukes
  * a heatmap per landmark (fin: tip / front / back + the trailing- and leading-edge chains fin_te_XX, fin_le_XX;
    fluke: tips / notch / tail base + the trailing-edge chain fluke_te_XX) on the stride-4 grid,
  * a visibility state per landmark (visible / under water / hidden / out of frame),
  * three dense maps at full crop resolution: whale mask, the whale's own silhouette edge (the factory's
    contour class 1: where the body really ends) and the water / foam / calf seam (classes 2-3).
The edge chains and spine_end (the dorsal midline point at the fluke root) exist in run05 renders and in the refreshed
run04 labels (2026-10-21); where a landmark is absent from a meta its channel is "missing" and carries no loss.
Note: parts_v1 (2026-10-20) was trained on run04 labels whose dorsal_fin_tip pointed at a phantom fin fabricated by the
canonical warp; from parts_v2 on, the labels come from the rig's real fin blade.

    python -m model_fine_tuning.landmarks.parts train --run-dir .../run04 --split splits/run04.json --name parts_v1
    python -m model_fine_tuning.landmarks.parts predict --ckpt experiments/part_landmarks/parts_v1/best.pt --day 2025-02-27
    python -m model_fine_tuning.landmarks.parts sheet --ckpt ... --run-dir .../run04 --split ... --out docs/part_landmarks_val.png

Why a dedicated detector: the signature / shape experiments (cluster/edge_match.py) showed the researchers' cue,
the trailing edge of fin and fluke, cannot be read from segmentation masks whose endpoints (tip, base) jump
frame to frame. The factory rig gives exact landmarks and the true silhouette for free.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from ..common import paths
from ..common.backbone import DinoFeatures, normalize_image
from ..common.data import IGNORE, WhaleRun, load_split, photometric, square_crop_params, to_tensor, warp_image, warp_nearest, warp_params
from ..common.heads import DenseDecoder

STATES = ["visible", "under_water", "hidden", "out_of_frame"]
STATE_CLASS = {"visible": 0, "under_water": 1, "hidden": 2, "occluded": 2, "out_of_frame": 3}
PARTS = {
    "fin": {"points": ["dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back"], "chains": [("fin_te", 12, False), ("fin_le", 12, False)]},
    "fluke": {"points": ["fluke_tip_left", "fluke_tip_right", "fluke_notch", "tail_base_top", "tail_base_bottom", "spine_end"], "chains": [("fluke_te", 24, True)]},
}
FLIP_PAIRS = [("fluke_tip_left", "fluke_tip_right")]
HEAT_STRIDE = 4
DENSE = ["mask", "own_edge", "seam_edge"]


def landmark_names(part: str) -> list[str]:
    spec = PARTS[part]
    return list(spec["points"]) + [f"{prefix}_{i:02d}" for prefix, n, _ in spec["chains"] for i in range(n)]


ALL_NAMES = landmark_names("fin") + landmark_names("fluke")
NAME_INDEX = {n: i for i, n in enumerate(ALL_NAMES)}
PART_OF = {n: "fin" for n in landmark_names("fin")} | {n: "fluke" for n in landmark_names("fluke")}


def flip_permutation() -> np.ndarray:
    """Index permutation of ALL_NAMES under a horizontal flip: left/right pairs swap, span-indexed chains reverse."""
    perm = np.arange(len(ALL_NAMES))
    for a, b in FLIP_PAIRS:
        perm[NAME_INDEX[a]], perm[NAME_INDEX[b]] = NAME_INDEX[b], NAME_INDEX[a]
    for spec in PARTS.values():
        for prefix, n, reverses in spec["chains"]:
            if reverses:
                idx = [NAME_INDEX[f"{prefix}_{i:02d}"] for i in range(n)]
                perm[idx] = idx[::-1]
    return perm


FLIP_PERM = flip_permutation()


# ----------------------------------------------------------------------------------------------- data
def part_index(run: WhaleRun, ids: list[str], part: str, min_extent_px: float = 28.0, min_visible: int = 300, cache: Path | None = None) -> list[dict]:
    """Frames showing the part: at least one of its landmarks visible, the landmarks' in-frame extent at least
    min_extent_px. Returns [{id, bbox, extent}] (bbox over the in-frame landmarks of the part, frame pixels)."""
    if cache is not None and cache.exists():
        rows = json.loads(cache.read_text())
        keep = set(ids)
        return [r for r in rows if r["id"] in keep]
    names = landmark_names(part)
    rows = []
    for shot_id in ids:
        if run.by_id[shot_id]["visible_pixels"] < min_visible or run.by_id[shot_id].get("negative"):
            continue
        kps = run.meta(shot_id).get("keypoints") or {}
        pts = [(k["uv"], k["state"]) for n in names if (k := kps.get(n)) and k.get("in_frame")]
        if not any(s == "visible" for _, s in pts):
            continue
        uv = np.array([p for p, _ in pts], float)
        x0, y0 = uv.min(0); x1, y1 = uv.max(0)
        extent = float(max(x1 - x0, y1 - y0))
        if extent < min_extent_px:
            continue
        rows.append({"id": shot_id, "bbox": [float(x0), float(y0), float(x1), float(y1)], "extent": extent, "part": part})
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows))
    return rows


def crop_targets(meta: dict, x0: float, y0: float, s: float, crop: int, flipped: bool, heat_size: int, sigma: float = 2.5):
    """uv (K, 2) crop px (nan = missing), state (K,) in STATES / -1 missing, heat (K, h, h) for the visible ones."""
    K = len(ALL_NAMES)
    uv = np.full((K, 2), np.nan, np.float32); st = np.full(K, -1, np.int64)
    kps = meta.get("keypoints") or {}
    for i, name in enumerate(ALL_NAMES):
        k = kps.get(name)
        if not k or "state" not in k:
            continue
        uv[i] = ((k["uv"][0] - x0) * s, (k["uv"][1] - y0) * s)
        st[i] = STATE_CLASS.get(k["state"], -1)
    if flipped:
        uv[:, 0] = crop - uv[:, 0]
        uv, st = uv[FLIP_PERM], st[FLIP_PERM]
    inside = np.isfinite(uv).all(1) & (uv >= 0).all(1) & (uv < crop).all(1)
    st[(st == 0) & ~inside] = 3
    stride = crop / heat_size
    heat = np.zeros((K, heat_size, heat_size), np.float32)
    gy, gx = np.mgrid[0:heat_size, 0:heat_size]
    for i in np.nonzero(st == 0)[0]:
        cx, cy = uv[i] / stride - 0.5
        heat[i] = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sigma ** 2))
    return uv, st, heat


class PartCropDataset(Dataset):
    def __init__(self, run: WhaleRun, rows: list[dict], crop: int = 512, train: bool = True, margin_range=(1.4, 3.0), eval_margin: float = 2.0,
                 jitter: float = 0.1, photometric_strength: float = 1.0, seed: int = 0):
        self.run, self.rows, self.crop, self.train = run, rows, crop, train
        self.margin_range, self.eval_margin, self.jitter, self.photometric_strength, self.seed = margin_range, eval_margin, jitter, photometric_strength, seed
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __getitem__(self, index: int):
        row = self.rows[index]; shot_id = row["id"]; run = self.run; C = self.crop
        image = run.image(shot_id); mask = run.mask(shot_id); contours = run.contours_or_zeros(shot_id); meta = run.meta(shot_id)
        rng = np.random.default_rng([self.seed, self.epoch, index])
        bx0, by0, bx1, by1 = row["bbox"]
        H, W = image.shape[:2]
        # the window: extent x margin, at least 1.4 x the extent, at most 1.15 x the frame (a fluke spanning the frame)
        margin = rng.uniform(*self.margin_range) if self.train else self.eval_margin
        margin = min(margin, 1.15 * max(W, H) / max(row["extent"], 1.0))
        min_side = max(64.0, row["extent"] * 1.4)
        if self.train:
            x0, y0, s = square_crop_params([bx0, by0, bx1, by1], C, margin, self.jitter, rng, min_side=min_side)
        else:
            x0, y0, s = square_crop_params([bx0, by0, bx1, by1], C, margin, 0.0, None, min_side=min_side)
        contours = contours.copy(); contours[:2, :] = 0; contours[-2:, :] = 0; contours[:, :2] = 0; contours[:, -2:] = 0   # the frame border is not a silhouette
        M = warp_params(x0, y0, s)
        img = warp_image(image, M, (C, C))
        m = warp_nearest(mask, M, (C, C), IGNORE)
        ct = warp_nearest(contours, M, (C, C), 0)
        flipped = bool(self.train and rng.random() < 0.5)
        if flipped:
            img, m, ct = img[:, ::-1], m[:, ::-1], ct[:, ::-1]
        if self.train:
            img = photometric(np.ascontiguousarray(img), rng, self.photometric_strength)
        uv, st, heat = crop_targets(meta, x0, y0, s, C, flipped, C // HEAT_STRIDE)
        # the crop's other part is not supervised: its landmarks are "missing"
        other = [i for i, n in enumerate(ALL_NAMES) if PART_OF[n] != row["part"]]
        st[other] = -1; heat[other] = 0.0; uv[other] = np.nan
        own = (ct == 1).astype(np.uint8); seam = ((ct == 2) | (ct == 3)).astype(np.uint8)
        k3 = np.ones((3, 3), np.uint8)
        dense = np.stack([(m == 1).astype(np.float32), cv2.dilate(own, k3).astype(np.float32), cv2.dilate(seam, k3).astype(np.float32)], 0)
        valid = (m != IGNORE).astype(np.float32)
        return {"id": shot_id, "part": row["part"], "image": to_tensor(np.ascontiguousarray(img)), "dense": torch.from_numpy(dense), "valid": torch.from_numpy(valid),
                "uv": torch.from_numpy(uv), "state": torch.from_numpy(st), "heat": torch.from_numpy(heat), "flipped": flipped,
                "extent_px": torch.tensor(float(row["extent"]) * s, dtype=torch.float32), "crop_x0y0s": torch.tensor([x0, y0, s], dtype=torch.float64)}


# ----------------------------------------------------------------------------------------------- model
class PartNet(nn.Module):
    def __init__(self, arch: str = "vitb16", unfreeze_blocks: int = 4, layers=None, decoder_width: int = 256, pretrained: bool = True):
        super().__init__()
        self.config = {"arch": arch, "unfreeze_blocks": unfreeze_blocks, "layers": layers, "decoder_width": decoder_width, "names": ALL_NAMES, "task": "part_landmarks"}
        self.backbone = DinoFeatures(arch, layers=layers, pretrained=pretrained)
        self.backbone.set_trainable(unfreeze_blocks)
        self.decoder = DenseDecoder(self.backbone.embed_dim, len(self.backbone.layers), len(DENSE), width=decoder_width)
        fine = self.decoder.out.in_channels; K = len(ALL_NAMES)
        self.kp_head = nn.Sequential(nn.Conv2d(fine, fine, 3, padding=1), nn.GELU(), nn.Conv2d(fine, K, 1))
        self.aux = nn.Sequential(nn.Linear(decoder_width + self.backbone.embed_dim, 512), nn.GELU(), nn.Dropout(0.1))
        self.state = nn.Linear(512, K * len(STATES))
        with torch.no_grad():
            self.kp_head[-1].bias.fill_(-4.0)

    def forward(self, image01: torch.Tensor) -> dict:
        x = normalize_image(image01)
        feats, cls = self.backbone(x)
        dense_logit, coarse, fine = self.decoder.forward_full(feats, x)
        heat = torch.sigmoid(self.kp_head(fine))
        pooled = torch.cat([coarse.mean((2, 3)), cls], 1)
        state_logit = self.state(self.aux(pooled)).view(-1, len(ALL_NAMES), len(STATES))
        return {"heat": heat, "state_logit": state_logit, "dense_logit": dense_logit}

    def param_groups(self, lr_head: float, lr_backbone: float, layer_decay: float, weight_decay: float) -> list[dict]:
        groups = self.backbone.param_groups(lr_backbone, layer_decay, weight_decay)
        heads = [p for mod in (self.decoder, self.kp_head, self.aux, self.state) for p in mod.parameters() if p.requires_grad]
        groups.append({"params": heads, "lr": lr_head, "weight_decay": weight_decay, "name": "heads"})
        return groups

    @classmethod
    def load(cls, ckpt_path: Path, device="cuda") -> "PartNet":
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        model = cls(arch=cfg["arch"], unfreeze_blocks=0, layers=cfg.get("layers"), decoder_width=cfg.get("decoder_width", 256), pretrained=False)
        model.load_state_dict(ckpt["model"]); model.to(device).eval()
        return model


def losses(out: dict, batch: dict, w_heat: float = 100.0, w_state: float = 0.5, w_dense: float = 1.0, edge_pos_weight: float = 10.0) -> tuple[torch.Tensor, dict]:
    st = batch["state"]; known = st >= 0                                                     # (B, K)
    heat_err = ((out["heat"].float() - batch["heat"]) ** 2).mean((2, 3))                      # (B, K)
    heat = (heat_err * known).sum() / known.sum().clamp(min=1)
    state = F.cross_entropy(out["state_logit"].float()[known], st[known]) if known.any() else out["state_logit"].sum() * 0
    dl = out["dense_logit"].float(); tgt = batch["dense"]; valid = batch["valid"][:, None]
    bce_mask = (F.binary_cross_entropy_with_logits(dl[:, :1], tgt[:, :1], reduction="none") * valid).sum() / valid.sum().clamp(min=1)
    pw = torch.tensor(edge_pos_weight, device=dl.device)
    bce_edge = F.binary_cross_entropy_with_logits(dl[:, 1:], tgt[:, 1:], pos_weight=pw)
    p = torch.sigmoid(dl[:, 1:]); dice = 1 - (2 * (p * tgt[:, 1:]).sum((2, 3)) + 1) / (p.sum((2, 3)) + tgt[:, 1:].sum((2, 3)) + 1)
    dense = bce_mask + bce_edge + dice.mean()
    total = w_heat * heat + w_state * state + w_dense * dense
    return total, {"heat": heat.item(), "state": state.item(), "dense": dense.item(), "dice": dice.mean().item()}


def decode_heatmaps(heat: np.ndarray, stride: int = HEAT_STRIDE) -> tuple[np.ndarray, np.ndarray]:
    """(K, h, w) -> positions (K, 2) crop px (peak refined by its 3x3 centroid) and peak confidences (K,)."""
    K, h, w = heat.shape
    flat = heat.reshape(K, -1); idx = flat.argmax(1); conf = flat.max(1)
    py, px = idx // w, idx % w
    uv = np.zeros((K, 2), np.float32)
    for k in range(K):
        y0, y1 = max(py[k] - 1, 0), min(py[k] + 2, h); x0, x1 = max(px[k] - 1, 0), min(px[k] + 2, w)
        patch = heat[k, y0:y1, x0:x1]; ws = patch.sum()
        gy, gx = np.mgrid[y0:y1, x0:x1]
        cy, cx = ((gy * patch).sum() / ws, (gx * patch).sum() / ws) if ws > 0 else (py[k], px[k])
        uv[k] = ((cx + 0.5) * stride, (cy + 0.5) * stride)
    return uv, conf


# ----------------------------------------------------------------------------------------------- train / eval
def move(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, amp: bool) -> dict:
    model.eval()
    rows = []
    for batch in loader:
        b = move(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(b["image"])
        heat = out["heat"].float().cpu().numpy(); state = out["state_logit"].float().argmax(-1).cpu().numpy()
        dense = (out["dense_logit"].float() > 0).cpu().numpy(); tgt = batch["dense"].numpy() > 0.5
        for i in range(heat.shape[0]):
            uv, conf = decode_heatmaps(heat[i]); st = batch["state"][i].numpy(); uv_gt = batch["uv"][i].numpy(); part = batch["part"][i]
            own = [NAME_INDEX[n] for n in landmark_names(part)]; vis = (st == 0); vis[[j for j in range(len(st)) if j not in own]] = False
            row = {"part": part}
            ext = float(batch["extent_px"][i])
            if vis.any() and ext > 1:
                err = np.linalg.norm(uv[vis] - uv_gt[vis], axis=1) / ext
                row["pck05"] = float(np.mean(err < 0.05)); row["pck10"] = float(np.mean(err < 0.1)); row["n_vis"] = int(vis.sum())
            known = st >= 0
            if known.any():
                row["state_acc"] = float(np.mean(state[i][known] == st[known]))
                tp = np.sum((state[i] == 0) & vis); fp = np.sum((state[i] == 0) & known & ~vis); fn = np.sum((state[i] != 0) & vis)
                row["vis_f1"] = float(2 * tp / max(2 * tp + fp + fn, 1))
            k5 = np.ones((5, 5), np.uint8)
            for c, name in enumerate(DENSE):
                p, t = dense[i, c], tgt[i, c]
                if c == 0:
                    inter = (p & t).sum(); row["mask_iou"] = float(inter / max((p | t).sum(), 1))
                elif t.any():
                    pd, td = cv2.dilate(p.astype(np.uint8), k5) > 0, cv2.dilate(t.astype(np.uint8), k5) > 0
                    prec = (p & td).sum() / max(p.sum(), 1); rec = (t & pd).sum() / max(t.sum(), 1)
                    row[f"{name}_f1"] = float(2 * prec * rec / max(prec + rec, 1e-6))
            rows.append(row)
    out = {}
    for part in PARTS:
        sub = [r for r in rows if r["part"] == part]
        for key in ("pck05", "pck10", "state_acc", "vis_f1", "mask_iou", "own_edge_f1", "seam_edge_f1"):
            vals = [r[key] for r in sub if key in r]
            if vals:
                out[f"{part}/{key}"] = float(np.mean(vals))
        out[f"{part}/n"] = len(sub)
    out["pck05"] = float(np.mean([out[f"{p}/pck05"] for p in PARTS if f"{p}/pck05" in out] or [0]))
    return out


def build_rows(run: WhaleRun, split_ids: dict, cache_dir: Path) -> dict:
    rows = {}
    for split, ids in split_ids.items():
        rows[split] = []
        for part in PARTS:
            cache = cache_dir / f"index_{run.run_dir.name}_{part}.json"
            all_rows = part_index(run, run.ids("all"), part, cache=cache)      # cached over the whole run, then filtered
            keep = set(ids)
            rows[split] += [r for r in all_rows if r["id"] in keep]
    return rows


def cmd_train(args) -> int:
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda")
    out_dir = args.out_root / "part_landmarks" / args.name; out_dir.mkdir(parents=True, exist_ok=True)
    runs = [WhaleRun(args.run_dir, args.split)] + [WhaleRun(d, s) for d, s in zip(args.extra_run_dir or [], args.extra_split or [])]
    train_sets, val_sets = [], []
    for run in runs:
        rows = build_rows(run, {"train": run.ids("train"), "val": run.ids("val")}, out_dir.parent / "index")
        tr, va = rows["train"][: args.max_train], rows["val"][: args.max_val]
        print(f"{run.run_dir.name}: train {len(tr)} part crops ({sum(r['part'] == 'fin' for r in tr)} fin / {sum(r['part'] == 'fluke' for r in tr)} fluke), val {len(va)}", flush=True)
        train_sets.append(PartCropDataset(run, tr, crop=args.crop, train=True, seed=args.seed, margin_range=(args.margin_min, args.margin_max), jitter=args.jitter))
        val_sets.append(PartCropDataset(run, va, crop=args.crop, train=False))
    train_ds, val_ds = ConcatDataset(train_sets), ConcatDataset(val_sets)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    model = PartNet(arch=args.arch, unfreeze_blocks=args.unfreeze_blocks, layers=args.layers).to(device)
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False); missing = model.load_state_dict(ck["model"], strict=False); print("init from", args.init_from, missing)
    optimizer = torch.optim.AdamW(model.param_groups(args.lr_head, args.lr_backbone, args.layer_decay, args.weight_decay))
    base_lr = [g["lr"] for g in optimizer.param_groups]
    total_steps = args.epochs * len(train_loader); step = 0; best = -1.0
    log = open(out_dir / "log.jsonl", "a")
    (out_dir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    for epoch in range(args.epochs):
        for ds in train_sets:
            ds.set_epoch(epoch)
        model.train(); t0 = time.time(); agg = {}
        for batch in train_loader:
            step += 1
            warm = min(1.0, step / args.warmup_steps); cos = 0.5 * (1 + math.cos(math.pi * min(step / total_steps, 1.0)))
            for g, lr0 in zip(optimizer.param_groups, base_lr):
                g["lr"] = lr0 * warm * cos
            b = move(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not args.no_amp):
                out = model(b["image"])
            loss, parts = losses(out, b, w_heat=args.w_heat, w_state=args.w_state, w_dense=args.w_dense)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 1.0); optimizer.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            agg["loss"] = agg.get("loss", 0.0) + loss.item(); agg["n"] = agg.get("n", 0) + 1
            if step % 50 == 0:
                print(f"ep {epoch} step {step}/{total_steps} loss {agg['loss'] / agg['n']:.4f} heat {agg['heat'] / agg['n']:.5f} state {agg['state'] / agg['n']:.3f} dense {agg['dense'] / agg['n']:.3f} ({(time.time() - t0) / agg['n']:.2f} s/it)", flush=True)
        metrics = evaluate(model, val_loader, device, not args.no_amp)
        rec = {"epoch": epoch, "step": step, "train": {k: v / agg["n"] for k, v in agg.items() if k != "n"}, "val": metrics, "time": time.time() - t0}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"== epoch {epoch}: val {json.dumps({k: round(v, 3) for k, v in metrics.items()})}", flush=True)
        ck = {"model": model.state_dict(), "config": model.config, "epoch": epoch, "val": metrics}
        torch.save(ck, out_dir / "last.pt")
        if metrics["pck05"] > best:
            best = metrics["pck05"]; torch.save(ck, out_dir / "best.pt"); print(f"   best so far (pck05 {best:.3f})", flush=True)
    return 0


# ----------------------------------------------------------------------------------------------- predict (real charts) / sheets
def draw_prediction(img: np.ndarray, uv: np.ndarray, conf: np.ndarray, state: np.ndarray, part: str, dense: np.ndarray | None, scale: float = 1.0,
                    uv_gt: np.ndarray | None = None, st_gt: np.ndarray | None = None) -> np.ndarray:
    """Overlay: predicted own-edge (magenta) and seam (cyan) maps, landmarks (circle, green = predicted visible),
    chains as polylines, ground truth (if given) as small white dots."""
    vis = img.copy()
    if dense is not None:
        own = cv2.resize(dense[1].astype(np.float32), (vis.shape[1], vis.shape[0])) > 0.5
        seam = cv2.resize(dense[2].astype(np.float32), (vis.shape[1], vis.shape[0])) > 0.5
        vis[own] = (0.4 * vis[own] + 0.6 * np.array([255, 0, 255])).astype(np.uint8)
        vis[seam] = (0.5 * vis[seam] + 0.5 * np.array([0, 220, 255])).astype(np.uint8)
    names = landmark_names(part); spec = PARTS[part]
    for prefix, n, _ in spec["chains"]:
        pts = [(uv[NAME_INDEX[f"{prefix}_{i:02d}"]] * scale, state[NAME_INDEX[f"{prefix}_{i:02d}"]]) for i in range(n)]
        for (p, s), (q, t) in zip(pts[:-1], pts[1:]):
            if s == 0 and t == 0:
                cv2.line(vis, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), (255, 255, 0), 1)
    for name in spec["points"]:
        i = NAME_INDEX[name]; p = tuple(np.round(uv[i] * scale).astype(int))
        col = (80, 255, 80) if state[i] == 0 else (90, 90, 90)
        cv2.circle(vis, p, 5, col, 1 if state[i] else 2)
        cv2.putText(vis, name.replace("dorsal_fin_", "").replace("fluke_", "").replace("tail_base_", "tb_")[:6], (p[0] + 6, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
    if uv_gt is not None:
        for name in names:
            i = NAME_INDEX[name]
            if st_gt is not None and st_gt[i] == 0 and np.isfinite(uv_gt[i]).all():
                cv2.circle(vis, tuple(np.round(uv_gt[i] * scale).astype(int)), 2, (255, 255, 255), -1)
    return vis


def cmd_sheet(args) -> int:
    """Val-split check: predicted vs ground-truth landmarks and edges on factory part crops."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PartNet.load(args.ckpt, device)
    run = WhaleRun(args.run_dir, args.split)
    rows = build_rows(run, {"val": run.ids("val")}, args.out_root / "part_landmarks" / "index")["val"]
    rng = np.random.default_rng(0); rows = [rows[i] for i in rng.permutation(len(rows))[: args.count]]
    ds = PartCropDataset(run, rows, crop=args.crop, train=False)
    tiles = []
    for i in range(len(ds)):
        s = ds[i]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(s["image"][None].to(device))
        heat = out["heat"].float()[0].cpu().numpy(); uv, conf = decode_heatmaps(heat); st = out["state_logit"].float()[0].argmax(-1).cpu().numpy()
        dense = (out["dense_logit"].float()[0] > 0).cpu().numpy()
        img = (s["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)[..., ::-1].copy()
        tile = draw_prediction(img, uv, conf, st, s["part"], dense, uv_gt=s["uv"].numpy(), st_gt=s["state"].numpy())
        cv2.putText(tile, f"{s['id']} {s['part']}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(cv2.resize(tile, (args.tile, args.tile)))
    while len(tiles) % args.cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.concatenate([np.concatenate(tiles[i:i + args.cols], 1) for i in range(0, len(tiles), args.cols)], 0)
    args.out.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(args.out), sheet); print(f"{len(rows)} tiles -> {args.out}")
    return 0


def chart_files(day_dir: Path, tag: str) -> list[tuple[Path, str]]:
    """Real part charts: parts_<tag>/charts/<key>__fin.jpg / __fluke.jpg (cluster/parts.py)."""
    out = []
    for part in PARTS:
        out += [(f, part) for f in sorted((day_dir / f"parts_{tag}" / "charts").glob(f"*__{part}.jpg"))]
    return out


def cmd_predict(args) -> int:
    """Real photos: run the detector on each fin / fluke chart of the day, write part_landmarks/<key>__<part>.json
    (landmarks in chart pixels with confidence and state, own-edge map as a PNG) and an overview sheet."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PartNet.load(args.ckpt, device)
    day_dir = args.days_root / args.day; out_dir = day_dir / f"part_landmarks{args.suffix}"; out_dir.mkdir(exist_ok=True)
    files = chart_files(day_dir, args.parts_tag)
    if args.limit:
        files = files[: args.limit]
    tiles = []
    for f, part in files:
        bgr = cv2.imread(str(f)); H, W = bgr.shape[:2]; C = args.crop
        # the native charts frame the part tightly; pad so the part fills the crop like the training windows did
        pad = int(round(args.pad * max(H, W)))
        if pad > 0:
            bgr = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        Hp, Wp = bgr.shape[:2]
        img = cv2.resize(bgr, (C, C), interpolation=cv2.INTER_AREA)[..., ::-1]
        x = to_tensor(np.ascontiguousarray(img))[None].to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(x)
        heat = out["heat"].float()[0].cpu().numpy(); uv, conf = decode_heatmaps(heat); st = out["state_logit"].float()[0].argmax(-1).cpu().numpy()
        prob = torch.sigmoid(out["dense_logit"].float()[0]).cpu().numpy()
        sx, sy = Wp / C, Hp / C
        rec = {"chart": f.name, "part": part, "size": [W, H], "pad": pad, "landmarks": {n: {"uv": [float(uv[NAME_INDEX[n], 0] * sx - pad), float(uv[NAME_INDEX[n], 1] * sy - pad)], "conf": float(conf[NAME_INDEX[n]]),
                                                                                "state": STATES[int(st[NAME_INDEX[n]])]} for n in landmark_names(part)}}
        (out_dir / f"{f.stem}.json").write_text(json.dumps(rec))
        cv2.imwrite(str(out_dir / f"{f.stem}__edges.png"), (np.stack([prob[0], prob[1], prob[2]], -1) * 255).astype(np.uint8))
        if len(tiles) < args.sheet_count:
            tile = draw_prediction(np.ascontiguousarray(img[..., ::-1]), uv, conf, st, part, prob > 0.5)
            cv2.putText(tile, f.stem.split("__")[-2][-18:] + " " + part, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            tiles.append(cv2.resize(tile, (args.tile, args.tile)))
    if tiles:
        while len(tiles) % args.cols:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.concatenate([np.concatenate(tiles[i:i + args.cols], 1) for i in range(0, len(tiles), args.cols)], 0)
        cv2.imwrite(str(out_dir / "sheet.png"), sheet)
    print(f"{len(files)} charts -> {out_dir}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--name", required=True); t.add_argument("--run-dir", type=Path, default=paths.WHALE_RUN_DIR); t.add_argument("--split", type=Path, default=paths.SPLITS_DIR / "run04.json")
    t.add_argument("--extra-run-dir", type=Path, nargs="*", default=None); t.add_argument("--extra-split", type=Path, nargs="*", default=None)
    t.add_argument("--out-root", type=Path, default=paths.EXPERIMENTS_DIR); t.add_argument("--arch", default="vitb16"); t.add_argument("--unfreeze-blocks", type=int, default=4)
    t.add_argument("--layers", type=int, nargs="*", default=None); t.add_argument("--epochs", type=int, default=8); t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--crop", type=int, default=512); t.add_argument("--margin-min", type=float, default=1.4); t.add_argument("--margin-max", type=float, default=3.0); t.add_argument("--jitter", type=float, default=0.1)
    t.add_argument("--lr-head", type=float, default=3e-4); t.add_argument("--lr-backbone", type=float, default=3e-5); t.add_argument("--layer-decay", type=float, default=0.8); t.add_argument("--weight-decay", type=float, default=0.05)
    t.add_argument("--warmup-steps", type=int, default=200); t.add_argument("--w-heat", type=float, default=100.0); t.add_argument("--w-state", type=float, default=0.5); t.add_argument("--w-dense", type=float, default=1.0)
    t.add_argument("--workers", type=int, default=8); t.add_argument("--seed", type=int, default=0); t.add_argument("--max-train", type=int, default=None); t.add_argument("--max-val", type=int, default=None)
    t.add_argument("--init-from", type=Path, default=None); t.add_argument("--no-amp", action="store_true")
    s = sub.add_parser("sheet")
    s.add_argument("--ckpt", type=Path, required=True); s.add_argument("--run-dir", type=Path, default=paths.WHALE_RUN_DIR); s.add_argument("--split", type=Path, default=paths.SPLITS_DIR / "run04.json")
    s.add_argument("--out-root", type=Path, default=paths.EXPERIMENTS_DIR); s.add_argument("--out", type=Path, required=True); s.add_argument("--count", type=int, default=48); s.add_argument("--cols", type=int, default=8)
    s.add_argument("--tile", type=int, default=384); s.add_argument("--crop", type=int, default=512)
    r = sub.add_parser("predict")
    r.add_argument("--ckpt", type=Path, required=True); r.add_argument("--day", required=True); r.add_argument("--days-root", type=Path, default=paths.EXPERIMENTS_DIR / "days_all")
    r.add_argument("--parts-tag", default="xr"); r.add_argument("--suffix", default=""); r.add_argument("--crop", type=int, default=512); r.add_argument("--limit", type=int, default=0)
    r.add_argument("--pad", type=float, default=0.15, help="border added around the chart (fraction of its side) before the crop")
    r.add_argument("--sheet-count", type=int, default=64); r.add_argument("--cols", type=int, default=8); r.add_argument("--tile", type=int, default=384)
    args = p.parse_args(argv)
    return {"train": cmd_train, "sheet": cmd_sheet, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
