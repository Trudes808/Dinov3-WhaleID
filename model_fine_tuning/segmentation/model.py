"""Whale / background segmentation: DINOv3 ViT + dense decoder, binary logits."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..common.backbone import DinoFeatures, normalize_image
from ..common.data import IGNORE
from ..common.heads import DenseDecoder


class WhaleSegModel(nn.Module):
    def __init__(self, arch: str = "vitb16", unfreeze_blocks: int = 4, layers=None, weights=None, repo=None, pretrained: bool = True,
                 decoder_width: int = 256):
        super().__init__()
        self.config = {"arch": arch, "unfreeze_blocks": unfreeze_blocks, "layers": layers, "decoder_width": decoder_width}
        self.backbone = DinoFeatures(arch, weights=weights, repo=repo, layers=layers, pretrained=pretrained)
        self.backbone.set_trainable(unfreeze_blocks)
        self.decoder = DenseDecoder(self.backbone.embed_dim, len(self.backbone.layers), 1, width=decoder_width)

    def forward(self, image01: torch.Tensor) -> torch.Tensor:
        x = normalize_image(image01)
        feats, _ = self.backbone(x)
        logits, _ = self.decoder(feats, x)
        return logits[:, 0]

    def param_groups(self, lr_head: float, lr_backbone: float, layer_decay: float, weight_decay: float) -> list[dict]:
        groups = self.backbone.param_groups(lr_backbone, layer_decay, weight_decay)
        decay, no_decay = [], []
        for name, p in self.decoder.named_parameters():
            (no_decay if p.ndim <= 1 else decay).append(p)
        groups.append({"params": decay, "lr": lr_head, "weight_decay": weight_decay, "name": "head_wd"})
        groups.append({"params": no_decay, "lr": lr_head, "weight_decay": 0.0, "name": "head_nd"})
        for g in groups:
            g["base_lr"] = g["lr"]
        return groups

    @classmethod
    def from_checkpoint(cls, path, device="cpu", weights=None, repo=None):
        from ..common.train_utils import load_checkpoint

        ckpt = load_checkpoint(path)
        cfg = ckpt["config"]["model"]
        model = cls(arch=cfg["arch"], unfreeze_blocks=0, layers=cfg.get("layers"), decoder_width=cfg.get("decoder_width", 256), pretrained=False)
        model.load_state_dict(ckpt["model"], strict=True)
        return model.to(device).eval(), ckpt


def seg_loss(logits: torch.Tensor, target: torch.Tensor, dice_weight: float = 1.0, weight: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """BCE + soft Dice over non-ignored pixels. target int64 with IGNORE for padding. weight: optional (B, H, W)
    per-pixel factor on the BCE (common.contours.contour_weight_map: up at the true silhouette, down at the
    waterline / foam seams); the Dice term is the unweighted region overlap."""
    valid = target != IGNORE
    t = (target == 1).float()
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    w = valid.float() if weight is None else valid.float() * weight
    bce = (bce * w).sum() / w.sum().clamp(min=1)
    p = torch.sigmoid(logits) * valid
    inter = (p * t).sum(dim=(1, 2))
    denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
    dice = 1 - (2 * inter + 1.0) / (denom + 1.0)
    dice = dice.mean()
    loss = bce + dice_weight * dice
    return loss, {"bce": bce.item(), "dice": dice.item()}
