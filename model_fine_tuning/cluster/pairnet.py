"""Pair verifier on labelled same-part chart pairs (teacher) and its distillation into a single-chart embedding (student).

    python -m model_fine_tuning.cluster.pairnet train   --name pair_teacher --data-root experiments/days_all
    python -m model_fine_tuning.cluster.pairnet distill --name pair_student --teacher experiments/pair_teacher/checkpoints/best.pt --data-root experiments/days_all
    python -m model_fine_tuning.cluster.pairnet eval    --teacher ... [--student ...] [--reference experiments/id_v4/checkpoints/best.pt] --days val|eval

Teacher: both charts of a pair are seen at once. DINOv3 ViT-B/16 tokens of each chart (512 px, last blocks tuned) are
projected to 256-d with a chart-position encoding and a part token, then L layers of self-attention within each chart
and cross-attention between the two (shared weights, whale tokens only), and a symmetric head on the pooled tokens
gives P(same animal). Trained with binary cross-entropy on every pair of a batch of P animals x 2 charts of one part
drawn from one day (positives = the same animal in two different bursts; negatives = the other animals of that day:
the clustering problem), light augmentation, no flips. This is the "purely pairwise" model the embeddings have never
been compared with: it can align regions of the two charts itself instead of relying on a fixed radius or a cosine.

Student: a single-chart embedding (cluster/train_id.py's WhaleIdModel, the id_v4 architecture) trained so that
sigmoid(a cos(z_i, z_j) + b) reproduces the teacher's P(same) on every pair of the same batches (relational
distillation; optional ArcFace term), i.e. the teacher's pairwise judgement is forced into an embedding space where
same-individual charts align, which then works for all-pairs clustering at embedding cost.

eval: pair AUC per part on identical pairs for the teacher, the student and a reference embedding (id_v4): cross-burst
positives vs other animals of the same day on the val days (model selection) or the eval days (final numbers).
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
from ..common.backbone import ARCHS, DinoFeatures, normalize_image
from ..common.train_utils import Logger, count_params, load_checkpoint, save_checkpoint, set_seed
from .train_id import PARTS, WhaleIdModel
from .train_local import scan_pairs

PATCH = 16


# ------------------------------------------------------------------ data
class PairBatchDataset(Dataset):
    """One item = one batch: P animals x 2 charts (different bursts) of one part type, from one day (hard negatives)."""

    def __init__(self, groups: dict, P: int = 6, size: int = 512, n_batches: int = 300, seed: int = 0, train: bool = True, mix_prob: float = 0.2, group_hard: bool = False):
        """group_hard: fill the batch with animals of ONE catalog group first (the animals photographed together: the pairs
        the day clustering gets wrong), then the day's other animals."""
        self.groups, self.P, self.size, self.n, self.train, self.mix_prob, self.group_hard = groups, P, size, n_batches, train, mix_prob, group_hard
        self.rng = random.Random(seed); self.np_rng = np.random.default_rng(seed)
        self.days = [d for d, g in groups.items() if g]; self.weights = [len(groups[d]) for d in self.days]

    def __len__(self):
        return self.n

    def _load(self, rec: dict):
        img = cv2.cvtColor(cv2.imread(str(rec["jpg"]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB); mask = np.load(rec["npz"])["whale"].astype(np.uint8)
        S = img.shape[0]
        if self.train:
            rng = self.np_rng; scale = rng.uniform(0.8, 1.0); ang = rng.uniform(-8, 8)
            cx, cy = S / 2 + rng.uniform(-0.05, 0.05) * S, S / 2 + rng.uniform(-0.05, 0.05) * S
            M = cv2.getRotationMatrix2D((cx, cy), ang, self.size / (S * scale)); M[:, 2] += np.array([self.size / 2 - cx, self.size / 2 - cy])
            img = cv2.warpAffine(img, M, (self.size, self.size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT); mask = cv2.warpAffine(mask, M, (self.size, self.size), flags=cv2.INTER_NEAREST)
            img = img.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15); img = img * rng.uniform(0.93, 1.07, size=(1, 1, 3))
            if rng.random() < 0.3:
                img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.4))
            img = np.clip(img, 0, 255)
        else:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32); mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        g = self.size // PATCH
        tok_mask = mask.reshape(g, PATCH, g, PATCH).mean((1, 3)) >= 0.3
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1) / 255.0)).float()
        return x, torch.from_numpy(tok_mask.reshape(-1))

    def __getitem__(self, i):
        rng = self.rng
        for _ in range(30):
            day = rng.choices(self.days, weights=self.weights)[0]; part = rng.choice(PARTS)
            cands = [(k, v) for k, v in self.groups[day].items() if k[1] == part]
            if len(cands) >= 3:
                break
        rng.shuffle(cands)
        if self.group_hard:                                                     # same catalog group first (ident = day_GxxAyy)
            by_g: dict = {}
            for k, v in cands:
                by_g.setdefault(k[0].split("_")[1][:3], []).append((k, v))
            gs = [g for g in by_g.values() if len(g) >= 2]; rng.shuffle(gs)
            ordered = [kv for g in gs for kv in g] + [kv for g in by_g.values() if len(g) < 2 for kv in g]
            cands = ordered
        chosen = cands[: self.P]
        if len(chosen) < self.P or rng.random() < self.mix_prob:
            other = rng.choices(self.days, weights=self.weights)[0]
            extra = [(k, v) for k, v in self.groups[other].items() if k[1] == part and k not in dict(chosen)]; rng.shuffle(extra)
            chosen = (chosen + extra)[: self.P]
        xs, ms, ids, keys = [], [], [], []
        for n, ((ident, _), bursts) in enumerate(chosen):
            b1, b2 = rng.sample(list(bursts), 2)
            for rec in (rng.choice(bursts[b1]), rng.choice(bursts[b2])):
                x, m = self._load(rec); xs.append(x); ms.append(m); ids.append(n); keys.append(rec["key"])
        grp = [k[0].split("_")[1][:3] for (k, _) in chosen for _ in (0, 1)]
        return {"image": torch.stack(xs), "mask": torch.stack(ms), "animal": torch.tensor(ids), "part": PARTS.index(part), "keys": keys, "day": day, "group": grp}


# ------------------------------------------------------------------ teacher
def fourier_pos(g: int, device, bands: int = 8) -> torch.Tensor:
    """(g*g, 4*bands) sinusoidal encoding of the token grid positions as chart fractions."""
    ii, jj = torch.meshgrid(torch.arange(g, device=device), torch.arange(g, device=device), indexing="ij")
    p = torch.stack([(jj.reshape(-1) + 0.5) / g, (ii.reshape(-1) + 0.5) / g], 1)
    freqs = (2.0 ** torch.arange(bands, device=device)) * math.pi
    ang = p[:, :, None] * freqs[None, None, :]
    return torch.cat([torch.sin(ang).reshape(len(p), -1), torch.cos(ang).reshape(len(p), -1)], 1)


class MatchLayer(nn.Module):
    def __init__(self, d: int, heads: int = 4):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.1)
        self.n2 = nn.LayerNorm(d); self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.1)
        self.n3 = nn.LayerNorm(d); self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(0.1), nn.Linear(4 * d, d))

    def forward(self, a, ma, b, mb):
        """a, b (B, N, d); ma, mb (B, N) True = whale token. Returns updated a, b."""
        ha = self.n1(a); a = a + self.self_attn(ha, ha, ha, key_padding_mask=~ma, need_weights=False)[0]
        hb = self.n1(b); b = b + self.self_attn(hb, hb, hb, key_padding_mask=~mb, need_weights=False)[0]
        ca, cb = self.n2(a), self.n2(b)
        a2 = a + self.cross_attn(ca, cb, cb, key_padding_mask=~mb, need_weights=False)[0]
        b2 = b + self.cross_attn(cb, ca, ca, key_padding_mask=~ma, need_weights=False)[0]
        a2 = a2 + self.ff(self.n3(a2)); b2 = b2 + self.ff(self.n3(b2))
        return a2, b2


class PairNet(nn.Module):
    """Pair verifier. With a reference identity model (residual mode) the logit is scale * cos(reference embeddings) + bias
    plus a cross-attention correction whose last layer starts at zero: training begins at the embedding's own decision
    and can only add what seeing both charts together adds (a matcher trained from nothing stayed at chance: 20k pairs
    per epoch are far too few for it)."""

    def __init__(self, dim: int = 256, layers: int = 3, unfreeze_blocks: int = 2, arch: str = "vitb16", pretrained: bool = True, size: int = 512, reference: str | None = None):
        super().__init__()
        self.config = {"arch": arch, "dim": dim, "layers": layers, "unfreeze_blocks": unfreeze_blocks, "size": size, "task": "pair_verifier", "reference": reference}
        self.backbone = DinoFeatures(arch, layers=[ARCHS[arch]["depth"] - 1], pretrained=pretrained); self.backbone.set_trainable(unfreeze_blocks)
        d_in = self.backbone.embed_dim
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, dim))
        self.pos = nn.Linear(32, dim); self.part_emb = nn.Embedding(len(PARTS), dim)
        self.layers = nn.ModuleList([MatchLayer(dim) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim, 1))
        self.ref = None
        if reference:
            self.ref, _ = WhaleIdModel.from_checkpoint(reference, "cpu")
            for p_ in self.ref.parameters():
                p_.requires_grad_(False)
            self.ref_scale = nn.Parameter(torch.tensor([10.0])); self.ref_bias = nn.Parameter(torch.tensor([-5.0]))
            nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def tokens(self, image01: torch.Tensor, part: torch.Tensor) -> torch.Tensor:
        """(B, N, dim) chart tokens with position and part encodings."""
        feats, _ = self.backbone(normalize_image(image01))
        t = feats[-1].flatten(2).transpose(1, 2)                                   # (B, N, d_in)
        g = feats[-1].shape[-1]
        return self.proj(t) + self.pos(fourier_pos(g, t.device))[None] + self.part_emb(part)[:, None, :]

    @torch.no_grad()
    def ref_embed(self, image01: torch.Tensor, part: torch.Tensor) -> torch.Tensor | None:
        if self.ref is None:
            return None
        self.ref.eval()
        return F.normalize(self.ref.embed(image01, part).float(), dim=1)

    def match(self, a: torch.Tensor, ma: torch.Tensor, b: torch.Tensor, mb: torch.Tensor, ra: torch.Tensor | None = None, rb: torch.Tensor | None = None) -> torch.Tensor:
        """Logit of 'same animal' for each pair (a_i, b_i); masks True on whale tokens (an empty mask keeps all);
        ra, rb = reference embeddings of the two charts (residual mode)."""
        ma = torch.where(ma.any(1, keepdim=True), ma, torch.ones_like(ma)); mb = torch.where(mb.any(1, keepdim=True), mb, torch.ones_like(mb))
        for layer in self.layers:
            a, b = layer(a, ma, b, mb)
        a, b = self.norm(a), self.norm(b)
        pa = (a * ma[..., None]).sum(1) / ma.sum(1, keepdim=True); pb = (b * mb[..., None]).sum(1) / mb.sum(1, keepdim=True)
        logit = self.head(torch.cat([pa * pb, (pa - pb).abs(), pa + pb], 1)).squeeze(-1)
        if self.ref is not None and ra is not None:
            logit = logit + self.ref_scale * (ra * rb).sum(1) + self.ref_bias
        return logit

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        ckpt = load_checkpoint(path); cfg = ckpt["config"]["model"]
        model = cls(cfg["dim"], cfg["layers"], unfreeze_blocks=0, arch=cfg["arch"], pretrained=False, size=cfg.get("size", 512), reference=cfg.get("reference"))
        model.load_state_dict(ckpt["model"], strict=True)
        return model.to(device).eval(), ckpt


def all_pairs(n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    iu = torch.triu_indices(n, n, 1, device=device)
    return iu[0], iu[1]


def teacher_logits(model: PairNet, tok: torch.Tensor, mask: torch.Tensor, i: torch.Tensor, j: torch.Tensor, chunk: int = 48, ref: torch.Tensor | None = None) -> torch.Tensor:
    """Symmetrised logits for pairs (i, j) of a token batch (ref = reference embeddings of the charts, residual mode)."""
    out = []
    for s in range(0, len(i), chunk):
        ii, jj = i[s:s + chunk], j[s:s + chunk]
        ra, rb = (ref[ii], ref[jj]) if ref is not None else (None, None)
        l1 = model.match(tok[ii], mask[ii], tok[jj], mask[jj], ra, rb); l2 = model.match(tok[jj], mask[jj], tok[ii], mask[ii], rb, ra)
        out.append(0.5 * (l1 + l2))
    return torch.cat(out)


# ------------------------------------------------------------------ evaluation on identical pairs
@torch.no_grad()
def eval_pairs(ds: PairBatchDataset, device, n_batches: int, teacher: PairNet | None = None, embedders: dict | None = None, ref_size: int = 512) -> dict:
    """Pair AUC per part for the teacher and for every embedding model (cosine) on the same pairs."""
    from sklearn.metrics import roc_auc_score

    rows = {"teacher": [], **{k: [] for k in (embedders or {})}}; labels_, parts, within = [], [], []
    for bi in range(n_batches):
        b = ds[bi]; x = b["image"].to(device); m = b["mask"].to(device); n = x.shape[0]
        i, j = all_pairs(n, device); y = (b["animal"][i.cpu()] == b["animal"][j.cpu()]).numpy()
        g = np.array(b.get("group", ["?"] * n)); within.append(g[i.cpu().numpy()] == g[j.cpu().numpy()])
        labels_.append(y); parts.append(np.full(len(y), b["part"]))
        part = torch.full((n,), b["part"], dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if teacher is not None:
                rows["teacher"].append(teacher_logits(teacher, teacher.tokens(x, part), m, i, j, ref=teacher.ref_embed(x, part)).float().cpu().numpy())
            for name, emb in (embedders or {}).items():
                z = F.normalize(emb.embed(x, part).float(), dim=1); rows[name].append((z[i] * z[j]).sum(1).cpu().numpy())
    y = np.concatenate(labels_); pt = np.concatenate(parts); wg = np.concatenate(within); out = {}
    for name, vals in rows.items():
        if not vals:
            continue
        s = np.concatenate(vals)
        for k, part in enumerate(PARTS):
            sel = pt == k
            if sel.sum() >= 20 and y[sel].any() and (~y[sel]).any():
                out[f"auc_{name}_{part}"] = float(roc_auc_score(y[sel], s[sel])); out[f"pairs_{part}"] = int(sel.sum())
            w = sel & wg
            if w.sum() >= 20 and y[w].any() and (~y[w]).any():
                out[f"auc_{name}_{part}_within_group"] = float(roc_auc_score(y[w], s[w]))
        aucs = [v for kk, v in out.items() if kk.startswith(f"auc_{name}_") and not kk.endswith("_within_group")]
        out[f"auc_{name}_mean"] = float(np.mean(aucs)) if aucs else float("nan")
        waucs = [v for kk, v in out.items() if kk.startswith(f"auc_{name}_") and kk.endswith("_within_group")]
        out[f"auc_{name}_within_group_mean"] = float(np.mean(waucs)) if waucs else float("nan")
    out["pairs"] = int(len(y)); out["positives"] = int(y.sum())
    return out


# ------------------------------------------------------------------ commands
def common_setup(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = json.loads(args.labels.read_text()); days = json.loads(args.days.read_text())
    train_days = args.train_days or days["train"]; val_days = args.val_days or days["val"]
    if not args.allow_eval_days:
        assert not set(train_days) & set(days["eval"]) and not set(val_days) & set(days["eval"]), "eval days must stay held out"
    tr = scan_pairs(args.data_root, labels, train_days, xr_tag="parts_hr"); va = scan_pairs(args.data_root, labels, val_days, xr_tag="parts_hr")
    print(f"train: {sum(len(g) for g in tr.values())} (animal, part) groups with >= 2 bursts over {len(tr)} days; val: {sum(len(g) for g in va.values())} groups", flush=True)
    return device, tr, va, days


def lr_schedule(step, total):
    return min(1.0, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / max(total, 1)))


def cmd_train(args) -> int:
    device, tr, va, _ = common_setup(args)
    out = args.out_root / args.name; out.mkdir(parents=True, exist_ok=True)
    model = PairNet(args.dim, args.layers, args.unfreeze_blocks, size=args.size, reference=str(args.residual_ref) if args.residual_ref else None).to(device)
    if args.init_backbone:                                                      # start from an identity model's fine-tuned ViT (same DINOv3 weights layout)
        state = {k: v for k, v in load_checkpoint(args.init_backbone)["model"].items() if k.startswith("backbone.vit.")}
        missing = model.load_state_dict(state, strict=False); print(f"backbone initialised from {args.init_backbone} ({len(state)} tensors)", flush=True)
    config = {"task": "pair_verifier", "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "func"}, "model": model.config, "params": count_params(model)}
    (out / "config.json").write_text(json.dumps(config, indent=1)); logger = Logger(out)
    ds = PairBatchDataset(tr, args.P, args.size, args.batches_per_epoch, seed=args.seed, train=True, group_hard=args.group_hard)
    vds = PairBatchDataset(va, args.P, args.size, args.val_batches, seed=123, train=False, group_hard=True)
    loader = DataLoader(ds, batch_size=None, num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=True)
    head_params = [p_ for n, p_ in model.named_parameters() if not n.startswith("backbone.") and not n.startswith("ref.") and p_.requires_grad]
    groups = model.backbone.param_groups(args.lr_backbone, 0.8, 0.05) + [{"params": head_params, "lr": args.lr_head, "weight_decay": 1e-4}]
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups); total = args.epochs * len(ds); step, best = 0, -1.0
    ref = {}
    if args.reference:
        ref["reference"], _ = WhaleIdModel.from_checkpoint(args.reference, device)
    for epoch in range(args.epochs):
        model.train()
        if model.ref is not None:
            model.ref.eval()
        t0 = time.time(); losses, accs = [], []
        for b in loader:
            f = lr_schedule(step, total)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            x = b["image"].to(device, non_blocking=True); m = b["mask"].to(device); n = x.shape[0]
            part = torch.full((n,), b["part"], dtype=torch.long, device=device)
            i, j = all_pairs(n, device); y = (b["animal"].to(device)[i] == b["animal"].to(device)[j]).float()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                tok = model.tokens(x, part); r = model.ref_embed(x, part)
                logit = model.match(tok[i], m[i], tok[j], m[j], None if r is None else r[i], None if r is None else r[j]).float()
            pos_w = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1.0)], device=device)
            loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_w)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
            losses.append(loss.item()); accs.append(float(((logit > 0).float() == y).float().mean())); step += 1
            if step % 25 == 0:
                logger.log(event="train", epoch=epoch, step=step, loss=float(np.mean(losses[-25:])), pair_acc=float(np.mean(accs[-25:])), seconds=round(time.time() - t0, 1))
            if args.max_steps and step >= args.max_steps:
                break
        logger.log(event="epoch", epoch=epoch, step=step, loss=float(np.mean(losses)) if losses else None, seconds=round(time.time() - t0, 1))
        model.eval(); metrics = eval_pairs(vds, device, args.val_batches, teacher=model, embedders=ref)
        logger.log(event="val", epoch=epoch, step=step, **metrics)
        if metrics.get("auc_teacher_mean", 0.0) > best:
            best = metrics["auc_teacher_mean"]; save_checkpoint(out / "checkpoints" / "best.pt", model, config, {"epoch": epoch, "val": metrics}); (out / "val_metrics.json").write_text(json.dumps(metrics, indent=1))
        save_checkpoint(out / "checkpoints" / "last.pt", model, config, {"epoch": epoch, "val": metrics})
        print(f"epoch {epoch}: teacher val AUC {metrics.get('auc_teacher_mean', float('nan')):.3f} (within-group {metrics.get('auc_teacher_within_group_mean', float('nan')):.3f})" + (f"  reference {metrics.get('auc_reference_mean', float('nan')):.3f} ({metrics.get('auc_reference_within_group_mean', float('nan')):.3f})" if ref else ""), flush=True)
        if args.max_steps and step >= args.max_steps:
            break
    logger.log(event="done", best_val_auc=best)
    return 0


def cmd_distill(args) -> int:
    device, tr, va, _ = common_setup(args)
    out = args.out_root / args.name; out.mkdir(parents=True, exist_ok=True)
    teacher, _ = PairNet.from_checkpoint(args.teacher, device)
    idents = sorted({r["ident"] for g in tr.values() for v in g.values() for recs in v.values() for r in recs}); cls_index = {k: n for n, k in enumerate(idents)}
    student = WhaleIdModel(len(idents), args.dim, args.unfreeze_blocks).to(device)
    if args.init:
        state = {k: v for k, v in load_checkpoint(args.init)["model"].items() if not k.startswith("arc.")}
        student.load_state_dict(state, strict=False); print("student initialised from", args.init, flush=True)
    scale = nn.Parameter(torch.tensor([8.0], device=device)); bias = nn.Parameter(torch.tensor([-3.0], device=device))
    config = {"task": "identity", "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "func"}, "model": student.config, "identities": idents, "params": count_params(student),
              "distilled_from": str(args.teacher)}
    (out / "config.json").write_text(json.dumps(config, indent=1)); logger = Logger(out)
    ds = PairBatchDataset(tr, args.P, args.size, args.batches_per_epoch, seed=args.seed, train=True, group_hard=args.group_hard)
    vds = PairBatchDataset(va, args.P, args.size, args.val_batches, seed=123, train=False, group_hard=True)
    loader = DataLoader(ds, batch_size=None, num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=True)
    head_params = [p_ for n, p_ in student.named_parameters() if not n.startswith("backbone.") and p_.requires_grad] + [scale, bias]
    groups = student.backbone.param_groups(args.lr_backbone, 0.8, 0.05) + [{"params": head_params, "lr": args.lr_head, "weight_decay": 1e-4}]
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups); total = args.epochs * len(ds); step, best = 0, -1.0
    ref = {}
    if args.reference:
        ref["reference"], _ = WhaleIdModel.from_checkpoint(args.reference, device)
    for epoch in range(args.epochs):
        student.train(); t0 = time.time(); losses = []
        for b in loader:
            f = lr_schedule(step, total)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            x = b["image"].to(device, non_blocking=True); m = b["mask"].to(device); n = x.shape[0]
            part = torch.full((n,), b["part"], dtype=torch.long, device=device)
            i, j = all_pairs(n, device); y = (b["animal"].to(device)[i] == b["animal"].to(device)[j])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                p_t = torch.sigmoid(teacher_logits(teacher, teacher.tokens(x, part), m, i, j, ref=teacher.ref_embed(x, part)).float())
            if args.teacher_mix > 0:                                                   # pull the soft targets toward the labels (the teacher is not perfect)
                p_t = (1 - args.teacher_mix) * p_t + args.teacher_mix * y.float()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                z = student.embed(x, part)
            z = F.normalize(z.float(), dim=1); cos = (z[i] * z[j]).sum(1)
            loss = F.binary_cross_entropy_with_logits(scale * cos + bias, p_t)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0); opt.step()
            losses.append(loss.item()); step += 1
            if step % 25 == 0:
                logger.log(event="train", epoch=epoch, step=step, loss=float(np.mean(losses[-25:])), scale=float(scale.item()), bias=float(bias.item()), seconds=round(time.time() - t0, 1))
            if args.max_steps and step >= args.max_steps:
                break
        logger.log(event="epoch", epoch=epoch, step=step, loss=float(np.mean(losses)) if losses else None, seconds=round(time.time() - t0, 1))
        student.eval(); metrics = eval_pairs(vds, device, args.val_batches, teacher=None, embedders={"student": student, **ref})
        logger.log(event="val", epoch=epoch, step=step, **metrics)
        if metrics.get("auc_student_mean", 0.0) > best:
            best = metrics["auc_student_mean"]; save_checkpoint(out / "checkpoints" / "best.pt", student, config, {"epoch": epoch, "val": metrics}); (out / "val_metrics.json").write_text(json.dumps(metrics, indent=1))
        save_checkpoint(out / "checkpoints" / "last.pt", student, config, {"epoch": epoch, "val": metrics})
        print(f"epoch {epoch}: student val AUC {metrics.get('auc_student_mean', float('nan')):.3f} (within-group {metrics.get('auc_student_within_group_mean', float('nan')):.3f})" + (f"  reference {metrics.get('auc_reference_mean', float('nan')):.3f} ({metrics.get('auc_reference_within_group_mean', float('nan')):.3f})" if ref else ""), flush=True)
        if args.max_steps and step >= args.max_steps:
            break
    logger.log(event="done", best_val_auc=best)
    return 0


def cmd_eval(args) -> int:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = json.loads(args.labels.read_text()); days = json.loads(args.days.read_text())
    use_days = days[args.split] if args.split in days else args.split.split(",")
    groups = scan_pairs(args.data_root, labels, use_days, xr_tag="parts_hr")
    ds = PairBatchDataset(groups, args.P, args.size, args.val_batches, seed=123, train=False, group_hard=True)
    teacher = PairNet.from_checkpoint(args.teacher, device)[0] if args.teacher else None
    emb = {}
    if args.student:
        emb["student"], _ = WhaleIdModel.from_checkpoint(args.student, device)
    if args.reference:
        emb["reference"], _ = WhaleIdModel.from_checkpoint(args.reference, device)
    metrics = eval_pairs(ds, device, args.val_batches, teacher=teacher, embedders=emb)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=1))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--data-root", type=Path, required=True); sp.add_argument("--labels", type=Path, default=paths.SPLITS_DIR / "labels_all_days.json")
        sp.add_argument("--days", type=Path, default=paths.SPLITS_DIR / "days.json"); sp.add_argument("--train-days", nargs="*", default=None); sp.add_argument("--val-days", nargs="*", default=None)
        sp.add_argument("--out-root", type=Path, default=paths.EXPERIMENTS_DIR); sp.add_argument("--P", type=int, default=6); sp.add_argument("--size", type=int, default=512)
        sp.add_argument("--batches-per-epoch", type=int, default=300); sp.add_argument("--epochs", type=int, default=10); sp.add_argument("--val-batches", type=int, default=60)
        sp.add_argument("--lr-head", type=float, default=3e-4); sp.add_argument("--lr-backbone", type=float, default=2e-5); sp.add_argument("--workers", type=int, default=6)
        sp.add_argument("--seed", type=int, default=0); sp.add_argument("--max-steps", type=int, default=None); sp.add_argument("--allow-eval-days", action="store_true")
        sp.add_argument("--reference", type=Path, default=None, help="identity checkpoint scored on the same pairs (id_v4)")
        sp.add_argument("--group-hard", action="store_true", help="batches filled with animals of one catalog group first")
    t = sub.add_parser("train"); common(t)
    t.add_argument("--name", required=True); t.add_argument("--dim", type=int, default=256); t.add_argument("--layers", type=int, default=3); t.add_argument("--unfreeze-blocks", type=int, default=2)
    t.add_argument("--init-backbone", type=Path, default=None, help="identity checkpoint whose ViT weights initialise the teacher's backbone (id_v4)")
    t.add_argument("--residual-ref", type=Path, default=None, help="identity checkpoint whose cosine is the teacher's starting decision (cross-attention learns the correction)")
    t.set_defaults(func=cmd_train)
    d = sub.add_parser("distill"); common(d)
    d.add_argument("--name", required=True); d.add_argument("--teacher", type=Path, required=True); d.add_argument("--dim", type=int, default=512); d.add_argument("--unfreeze-blocks", type=int, default=4)
    d.add_argument("--init", type=Path, default=None, help="start the student from an identity checkpoint"); d.add_argument("--teacher-mix", type=float, default=0.3, help="weight of the hard label in the soft target")
    d.set_defaults(func=cmd_distill)
    e = sub.add_parser("eval")
    e.add_argument("--data-root", type=Path, required=True); e.add_argument("--labels", type=Path, default=paths.SPLITS_DIR / "labels_all_days.json"); e.add_argument("--days", type=Path, default=paths.SPLITS_DIR / "days.json")
    e.add_argument("--split", default="val", help="val | eval | comma list of days"); e.add_argument("--teacher", type=Path, default=None); e.add_argument("--student", type=Path, default=None); e.add_argument("--reference", type=Path, default=None)
    e.add_argument("--P", type=int, default=6); e.add_argument("--size", type=int, default=512); e.add_argument("--val-batches", type=int, default=80); e.add_argument("--seed", type=int, default=0); e.add_argument("--out", type=Path, default=None)
    e.set_defaults(func=cmd_eval)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
