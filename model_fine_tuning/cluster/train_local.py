"""Train a local mark descriptor from cross-burst positive pairs and same-region negatives of other animals.

    python -m model_fine_tuning.cluster.train_local --name local_v1 --data-root experiments/days_all [--epochs 8]

A positive pair = two charts of the same part of the same animal (catalog identity) taken in DIFFERENT bursts (so the
model cannot match on the frame-to-frame sameness of one burst). Both charts are pose-normalised, so a token at a
chart position has its counterpart near the same position in the other chart: for an anchor whale token of chart A1
the positives are the whale tokens of A2 within r_pos of its chart position (a soft set: the best of them counts) that
also agree in canonical body coordinates within 1 m where the CSE map gives them (the "same region" sanity check);
the negatives are every whale token of the batch's OTHER animals (the same region of a different animal is in there,
that is the hard case), and the same animal's tokens far from the anchor's region (so the descriptor encodes the
mark, not the whale). InfoNCE with temperature 0.1 over the anchor's row. Batches: one part type, P animals x 2 charts
from different bursts, drawn from one day at a time (the animals photographed together are the negatives that matter),
1024-px crops at the chart's native scale (fin / fluke from parts_xr at 3072 px, body from parts_hr at 1536 px) with a
shared window plus small independent jitter, light photometric jitter, small rotation and scale; no flips.
Model: DINOv3 ViT-B/16, last `unfreeze_blocks` tuned, token head LayerNorm-Linear-GELU-Linear -> 128-d unit vectors.
Validation on the val days: chart-pair AUC where a pair's score is the mean over A's whale tokens of the best cosine
within r_pos in B (positives: same animal, other burst; negatives: other animals of the same day), per part.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..common import paths
from ..common.backbone import ARCHS, DinoFeatures, normalize_image
from ..common.train_utils import Logger, count_params, save_checkpoint, set_seed
from .bursts import make_bursts

PARTS = ("body", "fin", "fluke")
PATCH = 16


# ------------------------------------------------------------------ data
def scan_pairs(data_root: Path, labels: dict, days: list[str], xr_tag: str = "parts_xr", hr_tag: str = "parts_hr") -> dict:
    """Per day: charts {(ident, part): {burst: [chart record]}} for the labelled main instances."""
    out = {}
    for day in days:
        d = data_root / day
        if not (d / "timestamps.json").exists():
            continue
        ts = json.loads((d / "timestamps.json").read_text()); photos = labels["days"].get(day, {}).get("photos", {})
        burst_of = {s: b["id"] for b in make_bursts(ts, 1.5) for s in b["photos"]}
        groups: dict = {}
        for part in PARTS:
            src = d / (hr_tag if part == "body" else xr_tag)
            if not (src / "index.jsonl").exists():
                src = d / hr_tag
            for npz in src.glob(f"*__{part}.npz"):
                key = npz.name[: -len(f"__{part}.npz")]
                stem = key.rsplit("~", 1)[0] if "~" in key else key
                if "~" in key and not key.endswith("~0"):
                    continue
                ident = photos.get(stem.split("__")[-1])
                if ident is None or ident.split("_")[1].startswith("O") or stem not in burst_of:
                    continue
                groups.setdefault((ident, part), {}).setdefault(burst_of[stem], []).append({"key": key, "part": part, "jpg": src / "charts" / f"{key}__{part}.jpg", "npz": npz, "day": day, "stem": stem, "ident": ident})
        out[day] = {k: v for k, v in groups.items() if len(v) >= 2}                 # at least two bursts
    return out


def canon_grid(rec: dict, day_dir: Path, grid: int = 48) -> np.ndarray | None:
    """Canonical coordinates (m) on a grid x grid lattice of the chart (NaN off the whale), cached next to the chart."""
    cache = rec["npz"].with_suffix(".canon48.npy")
    if cache.exists():
        try:
            return np.load(cache)
        except Exception:  # noqa: BLE001  (a half-written cache from another worker: recompute)
            pass
    try:
        from .signature import token_canon_coords

        z = np.load(rec["npz"]); size = int(z["whale"].shape[0])
        cp = token_canon_coords(day_dir, rec["npz"].parent, rec["key"], rec["part"], grid, chart_size=size).reshape(grid, grid, 3).astype(np.float32)
        tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp.npy")
        np.save(tmp, cp); os.replace(tmp, cache)                                   # atomic: readers never see a partial file
        return cp
    except Exception:  # noqa: BLE001
        return None


class PairBatches(Dataset):
    """Each item = one batch: P animals x 2 charts (different bursts) of one part, cropped to 1024 px at native scale."""

    def __init__(self, groups: dict, data_root: Path, P: int = 8, crop: int = 1024, n_batches: int = 300, seed: int = 0, train: bool = True):
        self.groups, self.root, self.P, self.crop, self.n, self.train = groups, data_root, P, crop, n_batches, train
        self.rng = random.Random(seed); self.np_rng = np.random.default_rng(seed)
        self.days = [d for d, g in groups.items() if g]
        self.weights = [len(groups[d]) for d in self.days]

    def __len__(self):
        return self.n

    def _load(self, rec: dict, window: tuple[int, int, float], ang: float):
        img = cv2.cvtColor(cv2.imread(str(rec["jpg"]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB); mask = np.load(rec["npz"])["whale"].astype(np.uint8)
        S = img.shape[0]; x0, y0, scale = window
        cx, cy = x0 + self.crop / 2, y0 + self.crop / 2
        M = cv2.getRotationMatrix2D((cx, cy), ang, scale); M[:, 2] += np.array([self.crop / 2 - cx, self.crop / 2 - cy])
        crop = cv2.warpAffine(img, M, (self.crop, self.crop), flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        m = cv2.warpAffine(mask, M, (self.crop, self.crop), flags=cv2.INTER_NEAREST)
        if self.train:
            rng = self.np_rng; crop = crop.astype(np.float32)
            crop = crop * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15); crop = crop * rng.uniform(0.93, 1.07, size=(1, 1, 3))
            if rng.random() < 0.25:
                crop = cv2.GaussianBlur(crop, (0, 0), rng.uniform(0.4, 1.2))
            if rng.random() < 0.25:
                crop = crop + rng.normal(0, 3, crop.shape)
            crop = np.clip(crop, 0, 255)
        g = self.crop // PATCH
        tok_mask = m.reshape(g, PATCH, g, PATCH).mean((1, 3)) >= 0.3
        # mark-likeness of every token: local contrast (|Laplacian| of the grey crop averaged per 16-px patch), scaled by
        # the crop's 90th percentile over whale tokens -> [0, 1]; smooth grey skin ~0, notches / barnacles / scars high
        gray = cv2.cvtColor(np.clip(crop, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 1.0), cv2.CV_32F)).reshape(g, PATCH, g, PATCH).mean((1, 3))
        ref = np.percentile(lap[tok_mask], 90) if tok_mask.any() else 1.0
        contrast = np.clip(lap / max(float(ref), 1e-3), 0, 1).astype(np.float32)
        # chart coordinates of every token centre (inverse of M), for the region matching
        ii, jj = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
        pc = np.stack([(jj + 0.5) * PATCH, (ii + 0.5) * PATCH], -1).reshape(-1, 2)
        A = M[:, :2]; t = M[:, 2]; chart_xy = np.linalg.solve(A, (pc - t).T).T.astype(np.float32)
        canon = canon_grid(rec, self.root / rec["day"])
        if canon is not None:
            gi = np.clip((chart_xy[:, 1] / S * canon.shape[0]).astype(int), 0, canon.shape[0] - 1); gj = np.clip((chart_xy[:, 0] / S * canon.shape[1]).astype(int), 0, canon.shape[1] - 1)
            cxyz = canon[gi, gj]
        else:
            cxyz = np.full((g * g, 3), np.nan, np.float32)
        x = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1) / 255.0)).float()
        return x, torch.from_numpy(tok_mask.reshape(-1)), torch.from_numpy(chart_xy / S), torch.from_numpy(cxyz), torch.from_numpy(contrast.reshape(-1))      # positions as chart fractions

    def __getitem__(self, i):
        rng = self.rng
        for _ in range(20):
            day = rng.choices(self.days, weights=self.weights)[0]; part = rng.choice(PARTS)
            cands = [(k, v) for k, v in self.groups[day].items() if k[1] == part]
            if len(cands) >= 2:
                break
        rng.shuffle(cands); chosen = cands[: self.P]
        if len(chosen) < self.P:                                                     # fill from another day (easy negatives)
            other = rng.choices(self.days, weights=self.weights)[0]
            extra = [(k, v) for k, v in self.groups[other].items() if k[1] == part and k not in dict(chosen)]
            rng.shuffle(extra); chosen += extra[: self.P - len(chosen)]
        xs, masks, poss, canons, ids, cons = [], [], [], [], [], []
        # ONE crop window for the whole batch (as a chart fraction): positives and negatives share the crop geometry, so
        # the position information the ViT tokens carry cannot tell the animals apart (it could when each animal had its
        # own window: 0.97 AUC on crops, 0.6 on full charts). Every chart gets its own small jitter on top.
        bx = 0.5 + rng.uniform(-0.2, 0.2) if part != "body" else rng.uniform(0.0, 1.0)
        by = 0.5 + rng.uniform(-0.2, 0.2) if part != "body" else rng.uniform(0.0, 1.0)
        for n, ((ident, _), bursts) in enumerate(chosen):
            b1, b2 = rng.sample(list(bursts), 2)
            r1, r2 = rng.choice(bursts[b1]), rng.choice(bursts[b2])
            for rec in (r1, r2):
                Sz = int(np.load(rec["npz"])["whale"].shape[0]); span = max(Sz - self.crop, 0)
                jit = 0.03 * Sz if self.train else 0.0
                scale = rng.uniform(0.9, 1.1) if self.train else 1.0
                x0 = float(np.clip(bx * span + rng.uniform(-jit, jit), 0, span)); y0 = float(np.clip(by * span + rng.uniform(-jit, jit), 0, span))
                ang = rng.uniform(-5, 5) if self.train else 0.0
                x, m, pos, cxyz, con = self._load(rec, (x0, y0, scale), ang)
                xs.append(x); masks.append(m); poss.append(pos); canons.append(cxyz); ids.append(n); cons.append(con)
        return {"image": torch.stack(xs), "mask": torch.stack(masks), "pos": torch.stack(poss), "canon": torch.stack(canons), "animal": torch.tensor(ids), "part": part, "contrast": torch.stack(cons)}


# ------------------------------------------------------------------ model
class ConvHead(nn.Module):
    """Token head with context: a 3 x 3 convolution over the token grid, so a descriptor covers a 48 x 48 px region
    (a whole notch or barnacle patch with its surround) instead of one 16-px token."""

    def __init__(self, d: int, dim: int, hidden: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(d); self.c1 = nn.Conv2d(d, hidden, 3, padding=1); self.c2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:                           # (B, h, w, d) -> (B, h, w, dim)
        x = self.norm(t).permute(0, 3, 1, 2)
        return self.c2(F.gelu(self.c1(x))).permute(0, 2, 3, 1)


class LocalDescModel(nn.Module):
    def __init__(self, dim: int = 128, unfreeze_blocks: int = 2, arch: str = "vitb16", pretrained: bool = True, head: str = "mlp"):
        super().__init__()
        self.config = {"arch": arch, "dim": dim, "unfreeze_blocks": unfreeze_blocks, "task": "local_desc", "head": head}
        self.backbone = DinoFeatures(arch, layers=[ARCHS[arch]["depth"] - 1], pretrained=pretrained)
        self.backbone.set_trainable(unfreeze_blocks)
        d = self.backbone.embed_dim
        self.head = ConvHead(d, dim) if head == "conv" else nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(), nn.Linear(256, dim))

    def tokens(self, image01: torch.Tensor) -> torch.Tensor:
        """(B, h, w, dim) unit descriptors."""
        feats, _ = self.backbone(normalize_image(image01))
        t = feats[-1].permute(0, 2, 3, 1)
        return F.normalize(self.head(t), dim=-1)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        from ..common.train_utils import load_checkpoint

        ckpt = load_checkpoint(path); cfg = ckpt["config"]["model"]
        model = cls(cfg["dim"], unfreeze_blocks=0, arch=cfg["arch"], pretrained=False, head=cfg.get("head", "mlp"))
        model.load_state_dict(ckpt["model"], strict=True)
        return model.to(device).eval(), ckpt


def region_loss(desc: torch.Tensor, mask: torch.Tensor, pos: torch.Tensor, canon: torch.Tensor, animal: torch.Tensor, r_pos: float, r_far: float,
                tau: float = 0.1, anchors_per_crop: int = 384, canon_tol: float = 1.0, contrast: torch.Tensor | None = None, anchor_weight: str = "none") -> tuple[torch.Tensor, dict]:
    """desc (B, N, D) unit, mask (B, N) bool, pos (B, N, 2) chart fractions, canon (B, N, 3) m or NaN, animal (B,). B = 2P, chart 2k and 2k+1 are a pair.
    anchor_weight "contrast": anchors are drawn with probability 0.1 + contrast and their losses weighted by 0.2 + contrast
    (smooth grey skin carries no mark information: its anchors only add irreducible noise)."""
    B, N, D = desc.shape
    flat = desc.reshape(B * N, D); fmask = mask.reshape(-1); fpos = pos.reshape(B * N, 2); fan = animal[:, None].expand(B, N).reshape(-1)
    fchart = torch.arange(B, device=desc.device)[:, None].expand(B, N).reshape(-1); fcanon = canon.reshape(B * N, 3)
    fcon = contrast.reshape(-1) if contrast is not None else torch.zeros(B * N, device=desc.device)
    losses, stats = [], {"anchors": 0, "pos_tokens": 0.0, "hard_neg_acc": 0.0, "hard_neg_acc_marks": 0.0}
    cand = fmask.nonzero()[:, 0]                                                   # all whale tokens of the batch
    n_marks = 0
    for b in range(B):
        partner = b + 1 if b % 2 == 0 else b - 1
        own = (fchart == b) & fmask
        idx = own.nonzero()[:, 0]
        if len(idx) == 0:
            continue
        if anchor_weight == "contrast" and len(idx) > anchors_per_crop:
            idx = idx[torch.multinomial(0.1 + fcon[idx], anchors_per_crop, replacement=False)]
        else:
            idx = idx[torch.randperm(len(idx), device=desc.device)[:anchors_per_crop]]
        sim = flat[idx] @ flat[cand].T / tau                                       # (A, C)
        d = torch.cdist(fpos[idx], fpos[cand])                                     # chart-fraction distance
        c_chart = fchart[cand]; c_an = fan[cand]
        same_animal = (c_an == animal[b])[None, :]                                 # (1, C)
        in_partner = (c_chart == partner)[None, :]
        pos_set = in_partner & (d <= r_pos)
        # sanity check: canonical coordinates must agree within canon_tol where both are known
        ca, cc = fcanon[idx], fcanon[cand]
        both = torch.isfinite(ca[:, 0])[:, None] & torch.isfinite(cc[:, 0])[None, :]
        cd = torch.cdist(torch.nan_to_num(ca), torch.nan_to_num(cc))
        pos_set = pos_set & ~(both & (cd > canon_tol))
        neg_set = (~same_animal) | (same_animal & (d > r_far) & ~(c_chart == b)[None, :])
        neg_set = neg_set & ~pos_set
        has = pos_set.any(1)
        if not has.any():
            continue
        sim = sim[has]; pos_set = pos_set[has]; neg_set = neg_set[has]; d = d[has]; con = fcon[idx][has]
        neg_inf = torch.full_like(sim, -1e4)
        lp = torch.logsumexp(torch.where(pos_set, sim, neg_inf), 1)
        la = torch.logsumexp(torch.where(pos_set | neg_set, sim, neg_inf), 1)
        per_anchor = la - lp
        if anchor_weight == "contrast":
            w = 0.2 + con; losses.append((per_anchor * w).sum() / w.sum())
        else:
            losses.append(per_anchor.mean())
        stats["anchors"] += int(has.sum()); stats["pos_tokens"] += float(pos_set.sum(1).float().mean())
        hard = (~same_animal) & (d <= r_pos) & neg_set
        best_pos = torch.where(pos_set, sim, neg_inf).max(1).values; best_hard = torch.where(hard, sim, neg_inf).max(1).values
        ok = hard.any(1)
        if ok.any():
            stats["hard_neg_acc"] += float((best_pos[ok] > best_hard[ok]).float().mean())
            marks = ok & (con >= 0.5)
            if marks.any():
                stats["hard_neg_acc_marks"] += float((best_pos[marks] > best_hard[marks]).float().mean()); n_marks += 1
    n = max(len(losses), 1)
    stats["pos_tokens"] /= n; stats["hard_neg_acc"] /= n; stats["hard_neg_acc_marks"] /= max(n_marks, 1)
    return (torch.stack(losses).mean() if losses else desc.sum() * 0.0), stats


# ------------------------------------------------------------------ validation
@torch.no_grad()
def pair_score(model, ds: PairBatches, device, r_pos: float, n_batches: int, topk_frac: float = 0.2) -> dict:
    """AUC per part over chart pairs: score = mean over A's whale tokens of the best cosine within r_pos in B (auc_all_*),
    and the mean of the top topk_frac of those (auc_*: a few strongly matching regions decide, as a mark match should)."""
    from sklearn.metrics import roc_auc_score

    scores, labels_, parts = [], [], []
    for i in range(n_batches):
        b = ds[i]
        x = b["image"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            desc = model.tokens(x).float()
        B = desc.shape[0]; desc = desc.reshape(B, -1, desc.shape[-1]); mask = b["mask"].to(device); pos = b["pos"].to(device)
        for a in range(B):
            for c in range(B):
                if a == c or (a // 2 == c // 2 and c != a + 1 and c != a - 1):
                    continue
                if a // 2 == c // 2 and c < a:
                    continue
                ia = mask[a].nonzero()[:, 0]; ic = mask[c].nonzero()[:, 0]
                if len(ia) < 8 or len(ic) < 8:
                    continue
                sim = desc[a][ia] @ desc[c][ic].T; near = torch.cdist(pos[a][ia], pos[c][ic]) <= r_pos
                v = torch.where(near, sim, torch.full_like(sim, -1.0)).max(1).values
                v = v[v > -1.0]
                if len(v) == 0:
                    continue
                k = max(10, int(topk_frac * len(v)))
                scores.append((float(v.mean()), float(torch.topk(v, min(k, len(v))).values.mean()))); labels_.append(int(b["animal"][a] == b["animal"][c])); parts.append(b["part"])
    out = {}
    s, y, pt = np.array(scores), np.array(labels_), np.array(parts)
    for part in PARTS:
        sel = pt == part
        if sel.sum() > 20 and y[sel].any() and (~y[sel]).any():
            out[f"auc_all_{part}"] = float(roc_auc_score(y[sel], s[sel, 0])); out[f"auc_{part}"] = float(roc_auc_score(y[sel], s[sel, 1])); out[f"pairs_{part}"] = int(sel.sum())
    vals = [v for k, v in out.items() if k.startswith("auc_") and not k.startswith("auc_all")]
    out["auc_mean"] = float(np.mean(vals)) if vals else 0.0                        # selection on the top-k score (the best-matching regions decide)
    vals_all = [v for k, v in out.items() if k.startswith("auc_all_")]
    out["auc_all_mean"] = float(np.mean(vals_all)) if vals_all else 0.0
    return out


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True); p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=paths.SPLITS_DIR / "labels_all_days.json"); p.add_argument("--days", type=Path, default=paths.SPLITS_DIR / "days.json")
    p.add_argument("--out-root", type=Path, default=paths.EXPERIMENTS_DIR); p.add_argument("--train-days", nargs="*", default=None); p.add_argument("--val-days", nargs="*", default=None)
    p.add_argument("--dim", type=int, default=128); p.add_argument("--unfreeze-blocks", type=int, default=2); p.add_argument("--P", type=int, default=6); p.add_argument("--crop", type=int, default=1024)
    p.add_argument("--batches-per-epoch", type=int, default=300); p.add_argument("--epochs", type=int, default=8); p.add_argument("--val-batches", type=int, default=40)
    p.add_argument("--lr-head", type=float, default=3e-4); p.add_argument("--lr-backbone", type=float, default=1e-5); p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--r-pos", type=float, default=0.05, help="positive radius as a fraction of the chart (0.05 x 3072 = 154 px, ~10 native tokens; the crop jitter is 3 %)")
    p.add_argument("--r-far", type=float, default=0.12, help="same-animal tokens beyond this chart fraction are negatives")
    p.add_argument("--workers", type=int, default=6); p.add_argument("--seed", type=int, default=0); p.add_argument("--max-steps", type=int, default=None); p.add_argument("--allow-eval-days", action="store_true")
    p.add_argument("--head", default="mlp", choices=["mlp", "conv"], help="conv: 3 x 3 context over the token grid")
    p.add_argument("--anchor-weight", default="none", choices=["none", "contrast"], help="contrast: draw and weight anchors by local image contrast (mark-likeness)")
    p.add_argument("--topk-frac", type=float, default=0.2, help="validation pair score = mean of the top fraction of anchor matches")
    args = p.parse_args(argv)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = json.loads(args.labels.read_text()); days = json.loads(args.days.read_text())
    train_days = args.train_days or days["train"]; val_days = args.val_days or days["val"]
    if not args.allow_eval_days:
        assert not set(train_days) & set(days["eval"]) and not set(val_days) & set(days["eval"]), "eval days must stay held out"
    tr = scan_pairs(args.data_root, labels, train_days); va = scan_pairs(args.data_root, labels, val_days)
    n_groups = sum(len(g) for g in tr.values()); n_charts = sum(len(c) for g in tr.values() for v in g.values() for c in v.values())
    print(f"train: {n_groups} (animal, part) groups with >= 2 bursts, {n_charts} charts, {len(tr)} days; val: {sum(len(g) for g in va.values())} groups", flush=True)
    out = args.out_root / args.name; out.mkdir(parents=True, exist_ok=True)
    model = LocalDescModel(args.dim, args.unfreeze_blocks, head=args.head).to(device)
    config = {"task": "local_desc", "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "model": model.config, "params": count_params(model)}
    (out / "config.json").write_text(json.dumps(config, indent=1)); logger = Logger(out)
    ds = PairBatches(tr, args.data_root, args.P, args.crop, args.batches_per_epoch, seed=args.seed, train=True)
    vds = PairBatches(va, args.data_root, args.P, args.crop, args.val_batches, seed=123, train=False)
    loader = DataLoader(ds, batch_size=None, num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=True)
    head_params = [p_ for n, p_ in model.named_parameters() if not n.startswith("backbone.") and p_.requires_grad]
    groups = model.backbone.param_groups(args.lr_backbone, 0.8, 0.05) + [{"params": head_params, "lr": args.lr_head, "weight_decay": 1e-4}]
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups)
    total = args.epochs * len(ds); step, best = 0, -1.0
    for epoch in range(args.epochs):
        model.train(); t0 = time.time(); losses, accs = [], []
        for b in loader:
            lr_factor = min(1.0, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / max(total, 1)))
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_factor
            x = b["image"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                desc = model.tokens(x)
            desc = desc.float(); Bn = desc.shape[0]; desc = desc.reshape(Bn, -1, desc.shape[-1])
            loss, st = region_loss(desc, b["mask"].to(device), b["pos"].to(device), b["canon"].to(device), b["animal"].to(device), args.r_pos, args.r_far, args.tau,
                                   contrast=b["contrast"].to(device), anchor_weight=args.anchor_weight)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
            losses.append(loss.item()); accs.append(st["hard_neg_acc"]); step += 1
            if step % 25 == 0:
                logger.log(event="train", epoch=epoch, step=step, loss=float(np.mean(losses[-25:])), hard_neg_acc=float(np.mean(accs[-25:])), hard_neg_acc_marks=round(st["hard_neg_acc_marks"], 3), anchors=st["anchors"], pos_tokens=round(st["pos_tokens"], 1), seconds=round(time.time() - t0, 1))
            if args.max_steps and step >= args.max_steps:
                break
        logger.log(event="epoch", epoch=epoch, step=step, loss=float(np.mean(losses)) if losses else None, hard_neg_acc=float(np.mean(accs)) if accs else None, seconds=round(time.time() - t0, 1))
        model.eval(); metrics = pair_score(model, vds, device, args.r_pos, args.val_batches, args.topk_frac)
        logger.log(event="val", epoch=epoch, step=step, **metrics)
        if metrics["auc_mean"] > best:
            best = metrics["auc_mean"]; save_checkpoint(out / "checkpoints" / "best.pt", model, config, {"epoch": epoch, "val": metrics}); (out / "val_metrics.json").write_text(json.dumps(metrics, indent=1))
        save_checkpoint(out / "checkpoints" / "last.pt", model, config, {"epoch": epoch, "val": metrics})
        if args.max_steps and step >= args.max_steps:
            break
    logger.log(event="done", best_val_auc=best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
