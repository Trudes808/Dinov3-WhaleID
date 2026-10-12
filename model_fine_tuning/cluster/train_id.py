"""Identity embedding for whale part charts, trained on the labelled days.

    python -m model_fine_tuning.cluster.train_id --name id_v1 --data-root DATA --labels splits/labels_all_days.json --days splits/days.json

DATA/<day>/parts/ holds the charts made by cluster.parts for each processed day. One DINOv3
ViT-B/16 backbone (last blocks fine-tuned) + a 512-d embedding, trained with ArcFace (one
class per identity = (day, animal) after the catalog's "same as" merges) and a supervised
contrastive term, both computed within part type (a fin is never pushed towards a fluke).
Batches are P identities x K charts drawn mostly from the same day, so that the negatives
are the animals photographed alongside each other (the clustering problem) rather than
whales from another month. No horizontal flips (the charts are mirrored to a common flank
and marks are asymmetric); random resized crops, small rotations, photometric jitter.
Model selection on the val days by part-wise pair AUC.
"""

from __future__ import annotations

import argparse
import json
import math
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
from ..common.backbone import DinoFeatures, normalize_image
from ..common.train_utils import Logger, count_params, save_checkpoint, set_seed

PARTS = ("body", "fin", "fluke")


# ------------------------------------------------------------------ data
def scan_charts(data_root: Path, labels: dict, days: list[str], min_photos: int = 2) -> list[dict]:
    """One record per chart: path, part, identity id, day, stem. Identities with fewer than
    min_photos charts (any part) are dropped as classes but kept out of the data entirely."""
    recs = []
    for day in days:
        parts_dir = data_root / day / "parts"
        if not parts_dir.exists():
            continue
        photos = labels["days"].get(day, {}).get("photos", {})
        for npz in parts_dir.glob("*__*.npz"):
            stem, part = npz.name[: -4].rsplit("__", 1)
            base = stem.split("__")[-1]                       # convert_raw names: <folder>__<baseName>
            ident = photos.get(base)
            if ident is None or ident.split("_")[1].startswith("O"):
                continue
            recs.append({"path": parts_dir / "charts" / f"{stem}__{part}.jpg", "npz": npz, "part": part, "ident": ident, "day": day, "stem": stem})
    counts = {}
    for r in recs:
        counts[r["ident"]] = counts.get(r["ident"], 0) + 1
    return [r for r in recs if counts[r["ident"]] >= min_photos]


class ChartDataset(Dataset):
    def __init__(self, recs: list[dict], size: int = 448, train: bool = True, seed: int = 0):
        self.recs, self.size, self.train = recs, size, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = cv2.cvtColor(cv2.imread(str(r["path"]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        S = img.shape[0]
        if self.train:
            rng = self.rng
            scale = rng.uniform(0.75, 1.0); ang = rng.uniform(-8, 8)
            cx, cy = S / 2 + rng.uniform(-0.06, 0.06) * S, S / 2 + rng.uniform(-0.06, 0.06) * S
            M = cv2.getRotationMatrix2D((cx, cy), ang, self.size / (S * scale))
            M[:, 2] += np.array([self.size / 2 - cx, self.size / 2 - cy])
            img = cv2.warpAffine(img, M, (self.size, self.size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            img = img.astype(np.float32)
            img = img * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)                     # exposure / offset
            img = img * rng.uniform(0.9, 1.1, size=(1, 1, 3))                            # white balance
            if rng.random() < 0.3:
                img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
            img = np.clip(img, 0, 255)
        else:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32)
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1) / 255.0)).float()
        return {"image": x, "part": PARTS.index(r["part"]), "ident": r["ident"], "day": r["day"], "stem": r["stem"]}


class PKSampler:
    """Batches of P identities x K charts, identities drawn from one day at a time (with a
    chance of mixing in a second day), so most negatives are the same day's other animals."""

    def __init__(self, recs: list[dict], P: int, K: int, batches_per_epoch: int, seed: int = 0, mix_prob: float = 0.3):
        self.by_day = {}
        for i, r in enumerate(recs):
            self.by_day.setdefault(r["day"], {}).setdefault(r["ident"], []).append(i)
        self.days = [d for d in self.by_day if len(self.by_day[d]) >= 2]
        self.P, self.K, self.n, self.mix_prob = P, K, batches_per_epoch, mix_prob
        self.rng = random.Random(seed)
        self.weights = [sum(len(v) for v in self.by_day[d].values()) for d in self.days]

    def __iter__(self):
        for _ in range(self.n):
            day = self.rng.choices(self.days, weights=self.weights)[0]
            pool = dict(self.by_day[day])
            if self.rng.random() < self.mix_prob:
                other = self.rng.choices(self.days, weights=self.weights)[0]
                pool.update(self.by_day[other])
            idents = self.rng.sample(list(pool), min(self.P, len(pool)))
            batch = []
            for ident in idents:
                items = pool[ident]
                batch += self.rng.sample(items, self.K) if len(items) >= self.K else self.rng.choices(items, k=self.K)
            yield batch

    def __len__(self):
        return self.n


# ------------------------------------------------------------------ model
class ArcMargin(nn.Module):
    def __init__(self, dim: int, n_classes: int, s: float = 30.0, m: float = 0.4):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.s, self.m = s, m

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cos = F.linear(F.normalize(z), F.normalize(self.W)).clamp(-1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(cos)
        target = torch.cos(theta + self.m)
        onehot = F.one_hot(y, cos.shape[1]).bool()
        logits = torch.where(onehot, target, cos) * self.s
        return F.cross_entropy(logits, y)


def supcon(z: torch.Tensor, y: torch.Tensor, part: torch.Tensor, t: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss within part type."""
    z = F.normalize(z)
    sim = z @ z.T / t
    n = len(z)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    same_part = part[:, None] == part[None, :]
    pos = (y[:, None] == y[None, :]) & same_part & ~eye
    valid = same_part & ~eye
    sim = sim.masked_fill(~valid, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    has_pos = pos.sum(1) > 0
    if not has_pos.any():
        return z.sum() * 0.0
    return -((log_prob * pos).sum(1)[has_pos] / pos.sum(1)[has_pos]).mean()


class WhaleIdModel(nn.Module):
    def __init__(self, n_classes: int, dim: int = 512, unfreeze_blocks: int = 4, arch: str = "vitb16", pretrained: bool = True):
        super().__init__()
        self.config = {"arch": arch, "dim": dim, "unfreeze_blocks": unfreeze_blocks, "n_classes": n_classes, "task": "identity"}
        self.backbone = DinoFeatures(arch, layers=None, pretrained=pretrained)
        self.backbone.set_trainable(unfreeze_blocks)
        d = self.backbone.embed_dim
        self.part_token = nn.Embedding(len(PARTS), d)
        self.neck = nn.Sequential(nn.Linear(2 * d, 1024), nn.GELU(), nn.Dropout(0.1), nn.Linear(1024, dim))
        self.arc = nn.ModuleList([ArcMargin(dim, n_classes) for _ in PARTS])      # one class-centre set per part

    def embed(self, image01: torch.Tensor, part: torch.Tensor) -> torch.Tensor:
        feats, cls = self.backbone(normalize_image(image01))
        pooled = feats[-1].mean((2, 3))                                            # (B, d) GAP of the last block
        x = torch.cat([cls + self.part_token(part), pooled], 1)
        return self.neck(x)

    def forward(self, image01, part):
        return self.embed(image01, part)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        from ..common.train_utils import load_checkpoint

        ckpt = load_checkpoint(path)
        cfg = ckpt["config"]["model"]
        model = cls(cfg["n_classes"], cfg["dim"], unfreeze_blocks=0, arch=cfg["arch"], pretrained=False)
        model.load_state_dict(ckpt["model"], strict=True)
        return model.to(device).eval(), ckpt


# ------------------------------------------------------------------ evaluation on val days
@torch.no_grad()
def embed_records(model, recs: list[dict], device, size: int, batch: int = 32, workers: int = 4) -> np.ndarray:
    ds = ChartDataset(recs, size=size, train=False)
    loader = DataLoader(ds, batch_size=batch, num_workers=workers)
    out = []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            z = model.embed(b["image"].to(device), b["part"].to(device))
        out.append(F.normalize(z.float(), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.config["dim"]))


def pair_auc_by_part(Z: np.ndarray, recs: list[dict]) -> dict:
    from sklearn.metrics import roc_auc_score

    out = {}
    for part in PARTS:
        idx = [i for i, r in enumerate(recs) if r["part"] == part]
        if len(idx) < 10:
            continue
        z = Z[idx]; ident = np.array([recs[i]["ident"] for i in idx]); day = np.array([recs[i]["day"] for i in idx])
        sim = z @ z.T
        iu = np.triu_indices(len(idx), 1)
        same_day = (day[:, None] == day[None, :])[iu]
        y = (ident[:, None] == ident[None, :])[iu]
        s = sim[iu]
        m = same_day
        if y[m].any() and (~y[m]).any():
            out[f"auc_{part}_same_day"] = float(roc_auc_score(y[m], s[m]))
            out[f"pairs_{part}"] = int(m.sum())
    vals = [v for k, v in out.items() if k.startswith("auc_")]
    out["auc_mean"] = float(np.mean(vals)) if vals else 0.0
    return out


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=paths.SPLITS_DIR / "labels_all_days.json")
    p.add_argument("--days", type=Path, default=paths.SPLITS_DIR / "days.json")
    p.add_argument("--train-days", nargs="*", default=None, help="override the train days (smoke tests)")
    p.add_argument("--val-days", nargs="*", default=None)
    p.add_argument("--out-root", type=Path, default=paths.EXPERIMENTS_DIR)
    p.add_argument("--size", type=int, default=448)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--unfreeze-blocks", type=int, default=4)
    p.add_argument("--P", type=int, default=8); p.add_argument("--K", type=int, default=4)
    p.add_argument("--batches-per-epoch", type=int, default=300)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr-head", type=float, default=3e-4); p.add_argument("--lr-backbone", type=float, default=2e-5)
    p.add_argument("--w-arc", type=float, default=1.0); p.add_argument("--w-supcon", type=float, default=1.0)
    p.add_argument("--arc-margin", type=float, default=0.4)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--allow-eval-days", action="store_true", help="smoke tests only: permit eval days in train/val")
    args = p.parse_args(argv)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = json.loads(args.labels.read_text()); days = json.loads(args.days.read_text())
    train_days = args.train_days or days["train"]; val_days = args.val_days or days["val"]
    if not args.allow_eval_days:
        assert not set(train_days) & set(days["eval"]) and not set(val_days) & set(days["eval"]), "eval days must stay held out"
    train_recs = scan_charts(args.data_root, labels, train_days)
    val_recs = scan_charts(args.data_root, labels, val_days)
    idents = sorted({r["ident"] for r in train_recs}); cls_index = {i: k for k, i in enumerate(idents)}
    print(f"train: {len(train_recs)} charts, {len(idents)} identities, {len({r['day'] for r in train_recs})} days; val: {len(val_recs)} charts", flush=True)
    for part in PARTS:
        print(f"  {part}: {sum(r['part'] == part for r in train_recs)} train charts", flush=True)
    out = args.out_root / args.name; out.mkdir(parents=True, exist_ok=True)
    model = WhaleIdModel(len(idents), args.dim, args.unfreeze_blocks).to(device)
    for m in model.arc:
        m.m = args.arc_margin
    config = {"task": "identity", "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "model": model.config,
              "train_charts": len(train_recs), "val_charts": len(val_recs), "identities": idents, "params": count_params(model)}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    logger = Logger(out)
    ds = ChartDataset(train_recs, size=args.size, train=True, seed=args.seed)
    sampler = PKSampler(train_recs, args.P, args.K, args.batches_per_epoch, seed=args.seed)
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
            x = b["image"].to(device, non_blocking=True); part = b["part"].to(device)
            y = torch.tensor([cls_index[i] for i in b["ident"]], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                z = model.embed(x, part)
            z = z.float()
            arc = sum(model.arc[k](z[part == k], y[part == k]) for k in range(len(PARTS)) if (part == k).sum() > 1)
            loss = args.w_arc * arc + args.w_supcon * supcon(z, y, part)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
            losses.append(loss.item()); step += 1
            if step % 50 == 0:
                logger.log(event="train", epoch=epoch, step=step, loss=float(np.mean(losses[-50:])), lr_factor=lr_factor)
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


if __name__ == "__main__":
    raise SystemExit(main())
