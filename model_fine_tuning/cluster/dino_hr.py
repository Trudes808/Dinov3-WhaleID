"""Hi-res DINO identity head: the whale region at (close to) full resolution, one DINOv3 pass per chart, two outputs.

    python -m model_fine_tuning.cluster.dino_hr train --name dino_hr --data-root experiments/days_all --parts-tag parts_hr --size 1024
    python -m model_fine_tuning.cluster.dino_hr extract --checkpoint experiments/dino_hr/checkpoints/best.pt --parts DAY/parts_hr --out DAY/dino_hr

Input: an instance chart from cluster/parts.py --instances at 1536 px (body / fin / fluke, pose-normalised, cut from the
RAW), resized to `size` (1024: a 16-px patch is ~25-40 full-resolution pixels, 10-20 cm on the whale, so scars, barnacle
patches and notches are patch-level features instead of being averaged away at 448 px). DINOv3 ViT-B/16 with RoPE runs
at that resolution directly (SDPA attention); the last `unfreeze_blocks` blocks are tuned.
  1. Global embedding: the whale tokens (chart mask at token resolution) pooled by a learned per-part attention query
     plus their masked mean plus CLS -> 512-d, trained like cluster/train_id.py (ArcFace per part + supervised
     contrastive within part, group-hard P x K batches). Background tokens never enter the pooling.
  2. Local descriptors: the same last-block tokens, PCA-compressed at extraction time (64-d, fp16) with the token mask,
     for the spatially verified patch matching in cluster/local_hr.py (mutual nearest neighbours in chart coordinates).
Model selection on the val days by same-day pair AUC of the global embedding per part. No horizontal flips (charts are
mirrored to a common flank). Labels: training and selection only.
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
from torch.utils.data import DataLoader, Dataset

from ..common import paths
from ..common.backbone import ARCHS, DinoFeatures, normalize_image
from ..common.train_utils import Logger, count_params, save_checkpoint, set_seed
from .train_id import PARTS, ArcMargin, PKSampler, pair_auc_by_part, supcon

PATCH = 16


# ------------------------------------------------------------------ data
class HRChartDataset(Dataset):
    """Chart jpg + whale mask (from the chart npz) -> image (3, S, S) in [0, 1] and the whale fraction per token (S/16)^2."""

    def __init__(self, recs: list[dict], size: int = 1024, train: bool = True, seed: int = 0):
        self.recs, self.size, self.train = recs, size, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = cv2.cvtColor(cv2.imread(str(r["path"]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = np.load(r["npz"])["whale"].astype(np.uint8)
        S = img.shape[0]
        if self.train:
            rng = self.rng
            scale = rng.uniform(0.75, 1.0); ang = rng.uniform(-8, 8)
            cx, cy = S / 2 + rng.uniform(-0.06, 0.06) * S, S / 2 + rng.uniform(-0.06, 0.06) * S
            M = cv2.getRotationMatrix2D((cx, cy), ang, self.size / (S * scale))
            M[:, 2] += np.array([self.size / 2 - cx, self.size / 2 - cy])
            img = cv2.warpAffine(img, M, (self.size, self.size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (self.size, self.size), flags=cv2.INTER_NEAREST)
            img = img.astype(np.float32)
            img = img * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)
            img = img * rng.uniform(0.9, 1.1, size=(1, 1, 3))
            if rng.random() < 0.3:
                img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
            img = np.clip(img, 0, 255)
        else:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32)
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        g = self.size // PATCH
        tok_mask = mask.reshape(g, PATCH, g, PATCH).mean((1, 3)).astype(np.float32)
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1) / 255.0)).float()
        return {"image": x, "mask": torch.from_numpy(tok_mask.reshape(-1)), "part": PARTS.index(r["part"]), "ident": r["ident"], "day": r["day"], "stem": r["stem"]}


# ------------------------------------------------------------------ model
class DinoHRModel(nn.Module):
    def __init__(self, n_classes: int, dim: int = 512, unfreeze_blocks: int = 2, arch: str = "vitb16", pretrained: bool = True):
        super().__init__()
        self.config = {"arch": arch, "dim": dim, "unfreeze_blocks": unfreeze_blocks, "n_classes": n_classes, "task": "identity_hr"}
        self.backbone = DinoFeatures(arch, layers=[ARCHS[arch]["depth"] - 1], pretrained=pretrained)
        self.backbone.set_trainable(unfreeze_blocks)
        d = self.backbone.embed_dim
        self.part_query = nn.Embedding(len(PARTS), d)
        self.kv_norm = nn.LayerNorm(d)
        self.k_proj = nn.Linear(d, d); self.v_proj = nn.Linear(d, d)
        self.neck = nn.Sequential(nn.Linear(3 * d, 1024), nn.GELU(), nn.Dropout(0.1), nn.Linear(1024, dim))
        self.arc = nn.ModuleList([ArcMargin(dim, n_classes) for _ in PARTS])

    def tokens(self, image01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats, cls = self.backbone(normalize_image(image01))
        t = feats[-1]                                                               # (B, d, h, w)
        return t.flatten(2).transpose(1, 2), cls                                   # (B, N, d), (B, d)

    def pool(self, tok: torch.Tensor, cls: torch.Tensor, part: torch.Tensor, mask: torch.Tensor, min_frac: float = 0.3) -> torch.Tensor:
        m = mask > min_frac
        empty = m.sum(1) == 0
        m[empty] = True                                                             # no mask: pool everything
        q = self.part_query(part)                                                   # (B, d)
        kn = self.kv_norm(tok)
        logits = (self.k_proj(kn) @ q[:, :, None]).squeeze(-1) / math.sqrt(tok.shape[-1])
        attn = torch.softmax(logits.masked_fill(~m, -1e4), dim=1)
        att = (attn[:, :, None] * self.v_proj(kn)).sum(1)
        mean = (tok * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        return self.neck(torch.cat([att, mean, cls], 1))

    def embed(self, image01: torch.Tensor, part: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tok, cls = self.tokens(image01)
        return self.pool(tok, cls, part, mask)

    def forward(self, image01, part, mask):
        return self.embed(image01, part, mask)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        from ..common.train_utils import load_checkpoint

        ckpt = load_checkpoint(path)
        cfg = ckpt["config"]["model"]
        model = cls(cfg["n_classes"], cfg["dim"], unfreeze_blocks=0, arch=cfg["arch"], pretrained=False)
        model.load_state_dict(ckpt["model"], strict=True)
        return model.to(device).eval(), ckpt


@torch.no_grad()
def embed_records(model, recs: list[dict], device, size: int, batch: int = 8, workers: int = 4) -> np.ndarray:
    loader = DataLoader(HRChartDataset(recs, size=size, train=False), batch_size=batch, num_workers=workers)
    out = []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            z = model.embed(b["image"].to(device), b["part"].to(device), b["mask"].to(device))
        out.append(F.normalize(z.float(), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.config["dim"]))


# ------------------------------------------------------------------ train
def train(args) -> int:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = json.loads(args.labels.read_text()); days = json.loads(args.days.read_text())
    train_days = args.train_days or days["train"]; val_days = args.val_days or days["val"]
    if not args.allow_eval_days:
        assert not set(train_days) & set(days["eval"]) and not set(val_days) & set(days["eval"]), "eval days must stay held out"
    train_recs, val_recs = _scan_tag(args.data_root, labels, train_days, args.parts_tag), _scan_tag(args.data_root, labels, val_days, args.parts_tag)
    idents = sorted({r["ident"] for r in train_recs}); cls_index = {i: k for k, i in enumerate(idents)}
    print(f"train: {len(train_recs)} charts, {len(idents)} identities, {len({r['day'] for r in train_recs})} days; val: {len(val_recs)} charts", flush=True)
    for part in PARTS:
        print(f"  {part}: {sum(r['part'] == part for r in train_recs)} train charts", flush=True)
    out = args.out_root / args.name; out.mkdir(parents=True, exist_ok=True)
    model = DinoHRModel(len(idents), args.dim, args.unfreeze_blocks, arch=args.arch).to(device)
    for m in model.arc:
        m.m = args.arc_margin
    config = {"task": "identity_hr", "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "func"}, "model": model.config,
              "train_charts": len(train_recs), "val_charts": len(val_recs), "identities": idents, "params": count_params(model)}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    logger = Logger(out)
    ds = HRChartDataset(train_recs, size=args.size, train=True, seed=args.seed)
    sampler = PKSampler(train_recs, args.P, args.K, args.batches_per_epoch, seed=args.seed, mix_prob=args.mix_prob, group_hard=args.group_hard)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    head_params = [p_ for n, p_ in model.named_parameters() if not n.startswith("backbone.") and p_.requires_grad]
    groups = model.backbone.param_groups(args.lr_backbone, 0.8, 0.05) + [{"params": head_params, "lr": args.lr_head, "weight_decay": 1e-4}]
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups)
    total = args.epochs * len(sampler); step, best = 0, -1.0
    for epoch in range(args.epochs):
        model.train(); t0 = time.time(); losses = []
        for b in loader:
            lr_factor = min(1.0, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / max(total, 1)))
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_factor
            x = b["image"].to(device, non_blocking=True); part = b["part"].to(device); mask = b["mask"].to(device)
            y = torch.tensor([cls_index[i] for i in b["ident"]], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                z = model.embed(x, part, mask)
            z = z.float()
            arc = sum(model.arc[k](z[part == k], y[part == k]) for k in range(len(PARTS)) if (part == k).sum() > 1)
            loss = args.w_arc * arc + args.w_supcon * supcon(z, y, part)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
            losses.append(loss.item()); step += 1
            if step % 50 == 0:
                logger.log(event="train", epoch=epoch, step=step, loss=float(np.mean(losses[-50:])), lr_factor=lr_factor, seconds=round(time.time() - t0, 1))
            if args.max_steps and step >= args.max_steps:
                break
        logger.log(event="epoch", epoch=epoch, step=step, loss=float(np.mean(losses)) if losses else None, seconds=round(time.time() - t0, 1))
        model.eval()
        metrics = pair_auc_by_part(embed_records(model, val_recs, device, args.size, workers=min(args.workers, 4)), val_recs) if val_recs else {"auc_mean": 0.0}
        logger.log(event="val", epoch=epoch, step=step, **metrics)
        if metrics["auc_mean"] > best:
            best = metrics["auc_mean"]
            save_checkpoint(out / "checkpoints" / "best.pt", model, config, {"epoch": epoch, "val": metrics})
            (out / "val_metrics.json").write_text(json.dumps(metrics, indent=1))
        save_checkpoint(out / "checkpoints" / "last.pt", model, config, {"epoch": epoch, "val": metrics})
        if args.max_steps and step >= args.max_steps:
            break
    logger.log(event="done", best_val_auc=best)
    return 0


def _scan_tag(data_root: Path, labels: dict, days: list[str], tag: str) -> list[dict]:
    """scan_charts over DATA/<day>/<tag>/ instead of DATA/<day>/parts/."""
    recs = []
    for day in days:
        if (data_root / day / tag).exists():
            recs += _scan_one(data_root / day / tag, labels["days"].get(day, {}).get("photos", {}), day)
    counts = {}
    for r in recs:
        counts[r["ident"]] = counts.get(r["ident"], 0) + 1
    return [r for r in recs if counts[r["ident"]] >= 2]


def _scan_one(parts_dir: Path, photos: dict, day: str) -> list[dict]:
    recs = []
    for npz in parts_dir.glob("*__*.npz"):
        stem, part = npz.name[:-4].rsplit("__", 1)
        if part not in PARTS:
            continue
        base = stem.split("__")[-1]
        if "~" in base:
            base, inst = base.rsplit("~", 1)
            if inst != "0":
                continue
        ident = photos.get(base)
        if ident is None or ident.split("_")[1].startswith("O"):
            continue
        recs.append({"path": parts_dir / "charts" / f"{stem}__{part}.jpg", "npz": npz, "part": part, "ident": ident, "day": day, "stem": stem,
                     "group": f"{day}_{ident.split('_')[1][:3]}"})
    return recs


# ------------------------------------------------------------------ extract
@torch.no_grad()
def extract(args) -> int:
    """Global embeddings + PCA-compressed local tokens for every chart (all instances) of a day's parts directory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.frozen:                                                                 # zero-shot baseline: pretrained backbone, masked-mean global
        model = DinoHRModel(1, unfreeze_blocks=0, pretrained=True).to(device).eval(); size = args.size
        if args.layer is not None:                                                  # tokens of an intermediate block (more local, less semantic)
            model.backbone.layers = [args.layer]
    else:
        model, ckpt = DinoHRModel.from_checkpoint(args.checkpoint, device)
        size = int(ckpt["config"]["args"].get("size", 1024))
    g = size // PATCH
    files = sorted((args.parts / "charts").glob("*__*.jpg"))
    recs = []
    for f in files:
        stem, part = f.stem.rsplit("__", 1)
        if part in PARTS and (args.parts / f"{stem}__{part}.npz").exists():
            recs.append({"path": f, "npz": args.parts / f"{stem}__{part}.npz", "part": part, "ident": "?", "day": args.parts.parent.name, "stem": stem})
    loader = DataLoader(HRChartDataset(recs, size=size, train=False), batch_size=args.batch, num_workers=args.workers)
    glob_, toks, masks = [], [], []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            tok, cls = model.tokens(b["image"].to(device))
            if args.frozen:
                m = (b["mask"].to(device) > 0.3).float(); m[m.sum(1) == 0] = 1.0
                z = (tok.float() * m[..., None]).sum(1) / m.sum(1, keepdim=True)
            else:
                z = model.pool(tok, cls, b["part"].to(device), b["mask"].to(device))
        glob_.append(F.normalize(z.float(), dim=1).cpu().numpy())
        toks.append(tok.float().cpu().numpy().astype(np.float16)); masks.append((b["mask"] > 0.3).numpy())
    G = np.concatenate(glob_); T = np.concatenate(toks); Mk = np.concatenate(masks)
    # PCA of the whale tokens (fit on a sample), applied to all tokens; descriptors unit-normalised afterwards
    rng = np.random.default_rng(0)
    sample = T[Mk].astype(np.float32)
    sample = sample[rng.choice(len(sample), min(len(sample), 60000), replace=False)] if len(sample) else T.reshape(-1, T.shape[-1])[:1000].astype(np.float32)
    mean = sample.mean(0); U, S, Vt = np.linalg.svd(sample - mean, full_matrices=False)
    P = Vt[: args.local_dim].T.astype(np.float32)                                  # (d, local_dim)
    L = np.empty((T.shape[0], T.shape[1], args.local_dim), np.float16)
    for i in range(0, len(T), 64):
        x = (T[i:i + 64].astype(np.float32) - mean) @ P
        L[i:i + 64] = (x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)).astype(np.float16)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "features.npz", stems=np.array([r["stem"] for r in recs]), parts=np.array([r["part"] for r in recs]), glob=G, local=L, mask=Mk,
             grid=g, pca_mean=mean, pca_P=P, explained=float((S[: args.local_dim] ** 2).sum() / (S ** 2).sum()))
    print(f"{args.parts.parent.name}: {len(recs)} charts -> {args.out} (local PCA-{args.local_dim} keeps {(S[: args.local_dim] ** 2).sum() / (S ** 2).sum():.2f} of the variance)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--name", required=True); t.add_argument("--data-root", type=Path, required=True); t.add_argument("--parts-tag", default="parts_hr")
    t.add_argument("--labels", type=Path, default=paths.SPLITS_DIR / "labels_all_days.json"); t.add_argument("--days", type=Path, default=paths.SPLITS_DIR / "days.json")
    t.add_argument("--train-days", nargs="*", default=None); t.add_argument("--val-days", nargs="*", default=None); t.add_argument("--out-root", type=Path, default=paths.EXPERIMENTS_DIR)
    t.add_argument("--arch", default="vitb16"); t.add_argument("--size", type=int, default=1024); t.add_argument("--dim", type=int, default=512); t.add_argument("--unfreeze-blocks", type=int, default=2)
    t.add_argument("--P", type=int, default=6); t.add_argument("--K", type=int, default=3); t.add_argument("--batches-per-epoch", type=int, default=300); t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--lr-head", type=float, default=3e-4); t.add_argument("--lr-backbone", type=float, default=2e-5); t.add_argument("--w-arc", type=float, default=1.0); t.add_argument("--w-supcon", type=float, default=1.0)
    t.add_argument("--arc-margin", type=float, default=0.4); t.add_argument("--workers", type=int, default=6); t.add_argument("--seed", type=int, default=0); t.add_argument("--max-steps", type=int, default=None)
    t.add_argument("--allow-eval-days", action="store_true"); t.add_argument("--group-hard", action="store_true"); t.add_argument("--mix-prob", type=float, default=0.3)
    t.set_defaults(func=train)
    e = sub.add_parser("extract")
    e.add_argument("--checkpoint", type=Path, default=None); e.add_argument("--parts", type=Path, required=True); e.add_argument("--out", type=Path, required=True)
    e.add_argument("--frozen", action="store_true", help="no checkpoint: pretrained DINOv3, masked-mean global (zero-shot baseline)"); e.add_argument("--size", type=int, default=1024)
    e.add_argument("--layer", type=int, default=None, help="frozen mode: block index whose tokens are kept (default the last)")
    e.add_argument("--batch", type=int, default=8); e.add_argument("--workers", type=int, default=4); e.add_argument("--local-dim", type=int, default=64)
    e.set_defaults(func=extract)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
