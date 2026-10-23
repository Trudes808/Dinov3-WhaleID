"""Continuous-surface-embedding head: which template vertex every whale pixel shows.

A per-pixel embedding (EMB_DIM) from the dense decoder is matched against a learned embedding per template vertex
(assets/humpback4k_template.npz: 4 000 vertices of the 4k canonical mesh with parts, sides, geodesics and
soft-target neighbours). Training: cross-entropy over the vertices on a sample of labelled whale pixels per crop,
with geodesic-Gaussian soft targets (the template's k nearest geodesic neighbours), plus the visibility BCE and
the direct pose head of the canon model. The template is not folded: left and right vertices are distinct, a
mirrored crop maps every label to its mirror vertex, and inference votes the whale's side once (rotation head +
pixel majority) and snaps inconsistent pixels to their mirror vertex. Body bend never enters: the head answers
"which surface point", however the body is curved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..common import paths
from ..common.backbone import DinoFeatures, normalize_image
from ..common.data import IGNORE
from ..common.geometry import rotation_from_6d_torch
from ..common.heads import DenseDecoder

TEMPLATE_PATH = paths.ASSETS_DIR / "humpback4k_template.npz"
EMB_DIM = 32


class Template:
    """The template tables as tensors (geodesics float32 cm, parts, sides, knn soft targets, mirror)."""

    def __init__(self, path: Path = TEMPLATE_PATH, device="cpu"):
        t = np.load(path, allow_pickle=False)
        self.n = int(len(t["positions_cm"]))
        self.positions_cm = torch.from_numpy(t["positions_cm"].astype(np.float32)).to(device)
        self.part = torch.from_numpy(t["part"].astype(np.int64)).to(device)
        self.part_names = [str(s) for s in t["part_names"]]
        self.side = torch.from_numpy(t["side"].astype(np.int64)).to(device)
        self.surface = torch.from_numpy(t["surface"].astype(np.int64)).to(device)
        self.body_fraction = torch.from_numpy(t["body_fraction"].astype(np.float32)).to(device)
        self.geodesic_cm = torch.from_numpy(t["geodesic_cm"].astype(np.float32)).to(device)
        self.knn_ids = torch.from_numpy(t["knn_ids"].astype(np.int64)).to(device)
        self.knn_weights = torch.from_numpy(t["knn_weights"].astype(np.float32)).to(device)
        self.mirror = torch.from_numpy(t["mirror"].astype(np.int64)).to(device)
        self.mesh = str(t["mesh"])


class WhaleCSEModel(nn.Module):
    """DINOv3 + dense decoder -> per-pixel embedding (EMB_DIM) and a whale logit; a learned vertex embedding table;
    the attention-pooled pose head (6D rotation, log pixels-per-metre) as in WhaleCanonModel."""

    def __init__(self, n_vertices: int, arch: str = "vitb16", unfreeze_blocks: int = 0, layers=None, weights=None, repo=None,
                 pretrained: bool = True, decoder_width: int = 256, emb_dim: int = EMB_DIM, tau: float = 0.07):
        super().__init__()
        self.config = {"arch": arch, "unfreeze_blocks": unfreeze_blocks, "layers": layers, "decoder_width": decoder_width, "task": "cse",
                       "n_vertices": int(n_vertices), "emb_dim": int(emb_dim), "tau": float(tau)}
        self.backbone = DinoFeatures(arch, weights=weights, repo=repo, layers=layers, pretrained=pretrained)
        self.backbone.set_trainable(unfreeze_blocks)
        self.decoder = DenseDecoder(self.backbone.embed_dim, len(self.backbone.layers), emb_dim + 1, width=decoder_width)
        self.vertex_emb = nn.Parameter(torch.randn(n_vertices, emb_dim) * 0.02)
        self.tau = tau
        self.pool_logit = nn.Conv2d(decoder_width, 1, 1)
        self.pose = nn.Sequential(nn.Linear(decoder_width + self.backbone.embed_dim, 512), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 7))
        with torch.no_grad():
            self.pose[-1].weight.mul_(0.1)
            self.pose[-1].bias.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 3.5], dtype=torch.float32))

    def forward(self, image01: torch.Tensor) -> dict:
        x = normalize_image(image01)
        feats, cls = self.backbone(x)
        dense, coarse, _ = self.decoder.forward_full(feats, x)
        emb = F.normalize(dense[:, :-1], dim=1)                     # (B, D, H, W), unit length
        vis_logit = dense[:, -1]
        att = torch.softmax(self.pool_logit(coarse).flatten(1), dim=1)
        pooled = (coarse.flatten(2) * att[:, None, :]).sum(-1)
        pose = self.pose(torch.cat([pooled, cls], 1))
        return {"emb": emb, "vis_logit": vis_logit, "R": rotation_from_6d_torch(pose[:, :6]), "d6": pose[:, :6], "log_ppm": pose[:, 6]}

    def vertex_logits(self, emb_pixels: torch.Tensor) -> torch.Tensor:
        """(N, D) unit embeddings -> (N, V) logits against the unit vertex table."""
        return emb_pixels @ F.normalize(self.vertex_emb, dim=1).t() / self.tau

    def param_groups(self, lr_head: float, lr_backbone: float, layer_decay: float, weight_decay: float) -> list[dict]:
        groups = self.backbone.param_groups(lr_backbone, layer_decay, weight_decay)
        decay, no_decay = [], []
        for module in (self.decoder, self.pool_logit, self.pose):
            for name, p in module.named_parameters():
                (no_decay if p.ndim <= 1 else decay).append(p)
        groups.append({"params": decay, "lr": lr_head, "weight_decay": weight_decay, "name": "head_wd"})
        groups.append({"params": no_decay + [self.vertex_emb], "lr": lr_head, "weight_decay": 0.0, "name": "head_nd"})
        for g in groups:
            g["base_lr"] = g["lr"]
        return groups

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        from ..common.train_utils import load_checkpoint

        ckpt = load_checkpoint(path)
        cfg = ckpt["config"]["model"]
        model = cls(cfg["n_vertices"], arch=cfg["arch"], unfreeze_blocks=0, layers=cfg.get("layers"), decoder_width=cfg.get("decoder_width", 256),
                    pretrained=False, emb_dim=cfg.get("emb_dim", EMB_DIM), tau=cfg.get("tau", 0.07))
        model.load_state_dict(ckpt["model"], strict=True)
        return model.to(device).eval(), ckpt


def sample_pixels(vertex: torch.Tensor, n: int, generator=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Up to n labelled pixels per crop: (batch index, row, col) of pixels with a vertex label."""
    b, y, x = torch.nonzero(vertex > 0, as_tuple=True)
    out_b, out_y, out_x = [], [], []
    for i in range(int(vertex.shape[0])):
        sel = torch.nonzero(b == i, as_tuple=True)[0]
        if len(sel) > n:
            sel = sel[torch.randperm(len(sel), generator=generator, device=sel.device)[:n]]
        out_b.append(b[sel]); out_y.append(y[sel]); out_x.append(x[sel])
    return torch.cat(out_b), torch.cat(out_y), torch.cat(out_x)


def cse_loss(out: dict, batch: dict, model: WhaleCSEModel, template: Template, weights: dict, pixels_per_crop: int = 4096) -> tuple[torch.Tensor, dict]:
    """Soft cross-entropy over the template vertices on sampled labelled pixels (geodesic-Gaussian targets over the
    label's k nearest neighbours), visibility BCE outside the ignore class, Frobenius on the rotation, L1 on log scale."""
    mask = batch["mask"]
    not_ignored = (mask != IGNORE).float()
    vis = F.binary_cross_entropy_with_logits(out["vis_logit"], (mask == 1).float(), reduction="none")
    vis = (vis * not_ignored).sum() / not_ignored.sum().clamp(min=1)
    vertex = batch["vertex"]
    b, y, x = sample_pixels(vertex, pixels_per_crop)
    if len(b):
        emb = out["emb"][b, :, y, x]                                   # (N, D)
        logits = model.vertex_logits(emb)                              # (N, V)
        gt = vertex[b, y, x] - 1                                       # (N,)
        logp = F.log_softmax(logits, dim=1)
        ids, w = template.knn_ids[gt], template.knn_weights[gt]        # (N, K)
        vert = -(w * logp.gather(1, ids)).sum(1).mean()
        with torch.no_grad():
            pred = logits.argmax(1)
            geo = template.geodesic_cm[pred, gt]
            acc30 = (geo < 30).float().mean().item()
            part_acc = (template.part[pred] == template.part[gt]).float().mean().item()
            side_known = template.side[gt] != 0
            side_acc = (template.side[pred][side_known] == template.side[gt][side_known]).float().mean().item() if side_known.any() else float("nan")
    else:
        vert = out["emb"].sum() * 0; acc30 = part_acc = side_acc = float("nan")
    rot = ((out["R"] - batch["R"]) ** 2).sum(dim=(1, 2)).mean()
    scale = (out["log_ppm"] - batch["log_ppm"]).abs().mean()
    loss = weights["vis"] * vis + weights["vertex"] * vert + weights["rot"] * rot + weights["scale"] * scale
    return loss, {"vis": vis.item(), "vertex": vert.item(), "rot": rot.item(), "scale": scale.item(), "acc30cm": acc30, "part_acc": part_acc, "side_acc": side_acc}


@torch.no_grad()
def predict_vertices(model: WhaleCSEModel, emb: torch.Tensor, use: torch.Tensor, template: Template | None = None, chunk: int = 65536):
    """emb (D, H, W), use (H, W) bool -> (vertex id, confidence) per used pixel in nonzero order of `use`; with a
    template also the probability of each pixel's mirror vertex (how ambiguous its side is)."""
    y, x = torch.nonzero(use, as_tuple=True)
    e = emb[:, y, x].t()
    ids, conf, mirror_p = [], [], []
    for s in range(0, len(e), chunk):
        p = torch.softmax(model.vertex_logits(e[s:s + chunk]), 1)
        c, i = p.max(1)
        ids.append(i); conf.append(c)
        if template is not None:
            mirror_p.append(p.gather(1, template.mirror[i][:, None])[:, 0])
    empty = torch.zeros(0, dtype=torch.long, device=emb.device), torch.zeros(0, device=emb.device)
    if not ids:
        return empty if template is None else (*empty, torch.zeros(0, device=emb.device))
    if template is None:
        return torch.cat(ids), torch.cat(conf)
    return torch.cat(ids), torch.cat(conf), torch.cat(mirror_p)


@torch.no_grad()
def side_vote(ids: torch.Tensor, conf: torch.Tensor, template: Template, R: np.ndarray | None = None, mirror_p: torch.Tensor | None = None,
              ambiguity: float = 0.3) -> tuple[int, torch.Tensor]:
    """The whale's facing flank from the confidence-weighted pixel majority plus the rotation head (weighted like 20 % of
    the pixels). Only *ambiguous* pixels move: those on the other side whose mirror vertex scores at least `ambiguity`
    times their own vertex (a back seen from behind or a fluke shows both flanks legitimately; forcing every pixel to
    one side cost 9 points of side accuracy on run04 val). Without mirror probabilities nothing moves."""
    side = template.side[ids].float()
    vote = float((side * conf).sum())
    if R is not None:
        from ..common.geometry import initial_side_from_rotation
        vote += 0.2 * float(conf.sum()) * float(initial_side_from_rotation(R))
    s = 1 if vote >= 0 else -1
    out = ids.clone()
    if mirror_p is not None:
        move = (template.side[ids] == -s) & (mirror_p >= ambiguity * conf)
        out[move] = template.mirror[ids[move]]
    return s, out


@torch.no_grad()
def crop_metrics(ids: torch.Tensor, gt: torch.Tensor, use_pred: torch.Tensor, gt_vertex: torch.Tensor, template: Template) -> dict:
    """Per-crop metrics: geodesic error (cm) of the predicted vertex on labelled pixels where the model predicts whale,
    part and side accuracy, and the coverage error: |predicted visible fraction - true visible fraction| per part
    (the visible vertex set is the set of vertices any pixel maps to)."""
    row = {}
    if len(ids):
        geo = template.geodesic_cm[ids, gt]
        row["geodesic_cm_median"] = float(geo.median()); row["geodesic_cm_mean"] = float(geo.mean())
        row["within_30cm"] = float((geo < 30).float().mean()); row["within_60cm"] = float((geo < 60).float().mean())
        row["part_acc"] = float((template.part[ids] == template.part[gt]).float().mean())
        known = template.side[gt] != 0
        row["side_acc"] = float((template.side[ids][known] == template.side[gt][known]).float().mean()) if known.any() else None
    pv = torch.zeros(template.n, dtype=torch.bool, device=ids.device); pv[ids] = True
    gv = torch.zeros(template.n, dtype=torch.bool, device=ids.device); g = gt_vertex[gt_vertex > 0] - 1; gv[g] = True
    err = []
    for k in range(1, len(template.part_names) + 1):
        sel = template.part == k
        err.append(abs(float(pv[sel].float().mean()) - float(gv[sel].float().mean())))
    row["coverage_err_mean"] = float(np.mean(err)); row["visible_vertices_pred"] = int(pv.sum()); row["visible_vertices_gt"] = int(gv.sum())
    return row
