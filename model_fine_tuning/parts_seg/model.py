"""Part segmentation: DINOv3 ViT + the dense decoder of the whale segmentation, K-class logits.

Classes (labels.CLASSES): 0 background, 1 fluke, 2 dorsal fin, 3 pectoral fin, 4 head, 5 body.
Loss: class-weighted cross-entropy + soft multi-class Dice over the whale classes, ignore index 255.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..common.backbone import DinoFeatures, normalize_image
from ..common.data import IGNORE
from ..common.heads import DenseDecoder
from .labels import CLASSES


class PartsSegModel(nn.Module):
    def __init__(self, arch: str = "vitb16", unfreeze_blocks: int = 4, layers=None, weights=None, repo=None, pretrained: bool = True,
                 decoder_width: int = 256, num_classes: int = len(CLASSES)):
        super().__init__()
        self.config = {"arch": arch, "unfreeze_blocks": unfreeze_blocks, "layers": layers, "decoder_width": decoder_width, "num_classes": num_classes}
        self.backbone = DinoFeatures(arch, weights=weights, repo=repo, layers=layers, pretrained=pretrained)
        self.backbone.set_trainable(unfreeze_blocks)
        self.decoder = DenseDecoder(self.backbone.embed_dim, len(self.backbone.layers), num_classes, width=decoder_width)

    def forward(self, image01: torch.Tensor) -> torch.Tensor:
        x = normalize_image(image01)
        feats, _ = self.backbone(x)
        logits, _ = self.decoder(feats, x)
        return logits

    def init_from_binary(self, path) -> dict:
        """Start from the whale / background segmentation checkpoint (backbone + decoder except the output layer)."""
        from ..common.train_utils import load_checkpoint

        state = load_checkpoint(path)["model"]
        own = self.state_dict()
        take = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape and not k.startswith("decoder.out.")}
        missing = [k for k in own if k not in take]
        self.load_state_dict(take, strict=False)
        return {"loaded": len(take), "not_loaded": missing}

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
    def from_checkpoint(cls, path, device="cpu"):
        from ..common.train_utils import load_checkpoint

        ckpt = load_checkpoint(path)
        cfg = ckpt["config"]["model"]
        model = cls(arch=cfg["arch"], unfreeze_blocks=0, layers=cfg.get("layers"), decoder_width=cfg.get("decoder_width", 256),
                    num_classes=cfg.get("num_classes", len(CLASSES)), pretrained=False)
        model.load_state_dict(ckpt["model"], strict=True)
        return model.to(device).eval(), ckpt


def parts_loss(logits: torch.Tensor, target: torch.Tensor, class_weight: torch.Tensor | None = None, dice_weight: float = 1.0):
    """Weighted CE + soft Dice (classes 1.. only, over the batch, classes present in target or prediction)."""
    ce = F.cross_entropy(logits, target, weight=class_weight, ignore_index=IGNORE)
    valid = (target != IGNORE).unsqueeze(1).float()
    prob = torch.softmax(logits, 1) * valid
    K = logits.shape[1]
    onehot = F.one_hot(target.clamp(max=K - 1) * (target != IGNORE), K).permute(0, 3, 1, 2).float() * valid
    inter = (prob * onehot).sum(dim=(0, 2, 3))
    denom = prob.sum(dim=(0, 2, 3)) + onehot.sum(dim=(0, 2, 3))
    dice_c = 1 - (2 * inter + 1.0) / (denom + 1.0)
    present = onehot.sum(dim=(0, 2, 3)) > 0
    sel = present.clone(); sel[0] = False
    dice = dice_c[sel].mean() if sel.any() else logits.sum() * 0
    loss = ce + dice_weight * dice
    return loss, {"ce": float(ce.item()), "dice": float(dice.item())}
