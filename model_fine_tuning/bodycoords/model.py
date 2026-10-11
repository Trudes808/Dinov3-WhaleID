"""Body-coordinate + rigid-pose model.

Dense head (per pixel, DensePose-style): u = fraction along the body (tail 0 -> snout 1),
(sin, cos) of the angle around the body axis, and a visibility logit (visible whale or not).
Pose head (per crop): a 6D rotation -> camera-from-whale matrix in the Unreal camera frame,
and log pixels-per-metre at the whale (the image scale of the whale). The pose head pools
the fused stride-16 features with learned, visibility-weighted attention plus the CLS token.

Why the same backbone as segmentation: dense correspondence to a canonical body is exactly
what DINOv3 patch features encode well (the paper's dense-matching results); a CNN or a
dedicated pose network would start from weaker features. The rigid 3D fit then comes from
the dense predictions by PnP (geometry.fit_pose_pnp); the direct rotation head is a
cheap second estimate and a fallback when few whale pixels are visible.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..common.backbone import DinoFeatures, normalize_image
from ..common.data import IGNORE
from ..common.geometry import rotation_from_6d_torch
from ..common.heads import DenseDecoder


class WhaleBodyModel(nn.Module):
    def __init__(self, arch: str = "vitb16", unfreeze_blocks: int = 4, layers=None, weights=None, repo=None, pretrained: bool = True,
                 decoder_width: int = 256):
        super().__init__()
        self.config = {"arch": arch, "unfreeze_blocks": unfreeze_blocks, "layers": layers, "decoder_width": decoder_width}
        self.backbone = DinoFeatures(arch, weights=weights, repo=repo, layers=layers, pretrained=pretrained)
        self.backbone.set_trainable(unfreeze_blocks)
        self.decoder = DenseDecoder(self.backbone.embed_dim, len(self.backbone.layers), 4, width=decoder_width)
        self.pool_logit = nn.Conv2d(decoder_width, 1, 1)
        self.pose = nn.Sequential(
            nn.Linear(decoder_width + self.backbone.embed_dim, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.GELU(),
            nn.Linear(256, 7),
        )
        with torch.no_grad():  # start near identity rotation and a plausible scale
            self.pose[-1].weight.mul_(0.1)
            self.pose[-1].bias.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 3.5], dtype=torch.float32))

    def forward(self, image01: torch.Tensor) -> dict:
        x = normalize_image(image01)
        feats, cls = self.backbone(x)
        dense, coarse = self.decoder(feats, x)
        u = torch.sigmoid(dense[:, 0])
        sincos = F.normalize(dense[:, 1:3], dim=1, eps=1e-6)
        vis_logit = dense[:, 3]
        att = torch.softmax(self.pool_logit(coarse).flatten(1), dim=1)
        pooled = (coarse.flatten(2) * att[:, None, :]).sum(-1)
        pose = self.pose(torch.cat([pooled, cls], 1))
        R = rotation_from_6d_torch(pose[:, :6])
        return {"u": u, "sincos": sincos, "vis_logit": vis_logit, "R": R, "d6": pose[:, :6], "log_ppm": pose[:, 6]}

    def param_groups(self, lr_head: float, lr_backbone: float, layer_decay: float, weight_decay: float) -> list[dict]:
        groups = self.backbone.param_groups(lr_backbone, layer_decay, weight_decay)
        decay, no_decay = [], []
        for module in (self.decoder, self.pool_logit, self.pose):
            for name, p in module.named_parameters():
                (no_decay if p.ndim <= 1 else decay).append(p)
        groups.append({"params": decay, "lr": lr_head, "weight_decay": weight_decay, "name": "head_wd"})
        groups.append({"params": no_decay, "lr": lr_head, "weight_decay": 0.0, "name": "head_nd"})
        for g in groups:
            g["base_lr"] = g["lr"]
        return groups

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        from ..common.train_utils import load_checkpoint

        ckpt = load_checkpoint(path)
        cfg = ckpt["config"]["model"]
        model = cls(arch=cfg["arch"], unfreeze_blocks=0, layers=cfg.get("layers"), decoder_width=cfg.get("decoder_width", 256), pretrained=False)
        model.load_state_dict(ckpt["model"], strict=True)
        return model.to(device).eval(), ckpt


def body_loss(out: dict, batch: dict, weights: dict | None = None) -> tuple[torch.Tensor, dict]:
    w = {"vis": 1.0, "u": 10.0, "angle": 5.0, "rot": 2.0, "scale": 1.0}
    w.update(weights or {})
    mask = batch["mask"]
    not_ignored = mask != IGNORE
    vis_t = (mask == 1).float()
    vis = F.binary_cross_entropy_with_logits(out["vis_logit"], vis_t, reduction="none")
    vis = (vis * not_ignored).sum() / not_ignored.sum().clamp(min=1)
    valid = batch["valid"]
    n = valid.sum().clamp(min=1)
    u_l1 = ((out["u"] - batch["u"]).abs() * valid).sum() / n
    cos_sim = (out["sincos"] * batch["sincos"]).sum(1)
    angle = ((1.0 - cos_sim) * valid).sum() / n
    rot = ((out["R"] - batch["R"]) ** 2).sum(dim=(1, 2)).mean()
    scale = (out["log_ppm"] - batch["log_ppm"]).abs().mean()
    loss = w["vis"] * vis + w["u"] * u_l1 + w["angle"] * angle + w["rot"] * rot + w["scale"] * scale
    return loss, {"vis": vis.item(), "u_l1": u_l1.item(), "angle": angle.item(), "rot": rot.item(), "scale": scale.item()}
