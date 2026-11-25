"""DINOv3 ViT + dense decoder (as segmentation.model.WhaleSegModel) with a K-class head; softmax over classes."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..common.backbone import DinoFeatures, normalize_image
from ..common.data import IGNORE
from ..common.heads import DenseDecoder

CLASSES = ["water_sky", "whale", "person", "rail", "tape", "deck", "land"]


class SceneSegModel(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES), arch: str = "vitb16", unfreeze_blocks: int = 4, layers=None, pretrained: bool = True,
                 decoder_width: int = 256):
        super().__init__()
        self.config = {"arch": arch, "unfreeze_blocks": unfreeze_blocks, "layers": layers, "decoder_width": decoder_width, "n_classes": n_classes,
                       "classes": CLASSES[:n_classes]}
        self.backbone = DinoFeatures(arch, layers=layers, pretrained=pretrained)
        self.backbone.set_trainable(unfreeze_blocks)
        self.decoder = DenseDecoder(self.backbone.embed_dim, len(self.backbone.layers), n_classes, width=decoder_width)

    def forward(self, image01: torch.Tensor) -> torch.Tensor:
        x = normalize_image(image01)
        feats, _ = self.backbone(x)
        logits, _ = self.decoder(feats, x)
        return logits

    def param_groups(self, lr_head, lr_backbone, layer_decay, weight_decay):
        groups = self.backbone.param_groups(lr_backbone, layer_decay, weight_decay)
        decay, no_decay = [], []
        for _, p in self.decoder.named_parameters():
            (no_decay if p.ndim <= 1 else decay).append(p)
        groups.append({"params": decay, "lr": lr_head, "weight_decay": weight_decay, "name": "head_wd"})
        groups.append({"params": no_decay, "lr": lr_head, "weight_decay": 0.0, "name": "head_nd"})
        for g in groups:
            g["base_lr"] = g["lr"]
        return groups

    def init_from_binary(self, state: dict, extra_bias: float = -3.0) -> None:
        """Copy a WhaleSegModel state (1-channel head): every shared weight as is; head channel 1 (whale) = the binary
        logit, channel 0 (water / sky) = 0, the extra classes zero weights and bias extra_bias."""
        own = self.state_dict()
        for k, v in state.items():
            if k in ("decoder.out.weight", "decoder.out.bias"):
                continue
            own[k] = v
        w = torch.zeros_like(own["decoder.out.weight"]); b = torch.full_like(own["decoder.out.bias"], extra_bias)
        w[1] = state["decoder.out.weight"][0]; b[1] = state["decoder.out.bias"][0]; b[0] = 0.0
        own["decoder.out.weight"] = w; own["decoder.out.bias"] = b
        self.load_state_dict(own, strict=True)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        from ..common.train_utils import load_checkpoint
        ck = load_checkpoint(path); cfg = ck["config"]["model"]
        m = cls(n_classes=cfg["n_classes"], arch=cfg["arch"], unfreeze_blocks=0, layers=cfg.get("layers"), decoder_width=cfg.get("decoder_width", 256), pretrained=False)
        m.load_state_dict(ck["model"], strict=True)
        return m.to(device).eval(), ck


NOT_WHALE = 254      # v3 (rules_scene_v3.md S2a): any class but whale


def scene_loss(logits: torch.Tensor, target: torch.Tensor, dice_weight: float = 1.0) -> tuple[torch.Tensor, dict]:
    """Cross-entropy over non-ignored pixels + soft Dice averaged over the classes present in the batch's target.
    Pixels labelled NOT_WHALE add -log(1 - P(whale)) to the cross-entropy mean and count as whale background in Dice."""
    nw = target == NOT_WHALE
    if nw.any():
        tgt = target.masked_fill(nw, IGNORE)
        ce_sum = F.cross_entropy(logits, tgt, ignore_index=IGNORE, reduction="sum")
        lse = torch.logsumexp(logits, 1)
        others = torch.cat([logits[:, :1], logits[:, 2:]], 1)
        nw_sum = (lse - torch.logsumexp(others, 1))[nw].sum()
        ce = (ce_sum + nw_sum) / ((tgt != IGNORE).sum() + nw.sum()).clamp(min=1)
        valid = (tgt != IGNORE).unsqueeze(1).float().repeat(1, logits.shape[1], 1, 1)
        valid[:, 1] = torch.maximum(valid[:, 1], nw.float())
        target = tgt
    else:
        ce = F.cross_entropy(logits, target, ignore_index=IGNORE)
        valid = (target != IGNORE).unsqueeze(1).float()
    p = torch.softmax(logits, 1) * valid
    t = F.one_hot(target.clamp(max=logits.shape[1] - 1), logits.shape[1]).permute(0, 3, 1, 2).float() * valid
    inter = (p * t).sum((0, 2, 3)); den = p.sum((0, 2, 3)) + t.sum((0, 2, 3))
    present = t.sum((0, 2, 3)) > 0
    dice = (1 - (2 * inter + 1.0) / (den + 1.0))[present].mean() if present.any() else logits.sum() * 0
    if torch.isnan(ce):
        ce = logits.sum() * 0
    return ce + dice_weight * dice, {"ce": float(ce.item()), "dice": float(dice.item())}
