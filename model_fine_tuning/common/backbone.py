"""DINOv3 ViT backbone: loading from the local facebookresearch/dinov3 checkout, multi-layer
patch features, and selective unfreezing with layer-wise learning-rate decay.

Fine-tuning policy: ``unfreeze_blocks=0`` keeps the backbone frozen (the DINOv3 paper's
default for dense heads); ``N`` trains the last N transformer blocks and the final norm;
``-1`` trains everything including the patch embedding. Layer-wise LR decay gives earlier
blocks smaller learning rates, which protects the pretrained features.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from . import paths

ARCHS = {
    "vits16": {"fn": "dinov3_vits16", "embed_dim": 384, "depth": 12},
    "vits16plus": {"fn": "dinov3_vits16plus", "embed_dim": 384, "depth": 12},
    "vitb16": {"fn": "dinov3_vitb16", "embed_dim": 768, "depth": 12},
    "vitl16": {"fn": "dinov3_vitl16", "embed_dim": 1024, "depth": 24},
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_backbone(arch: str = "vitb16", weights: str | Path | None = None, repo: str | Path | None = None, pretrained: bool = True):
    repo = Path(repo or paths.DINOV3_REPO)
    if not (repo / "hubconf.py").exists():
        raise FileNotFoundError(f"DINOv3 checkout not found at {repo} (set DINOV3_REPO)")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from dinov3.hub import backbones  # noqa: WPS433  (import from the checkout)

    fn = getattr(backbones, ARCHS[arch]["fn"])
    if not pretrained:
        return fn(pretrained=False)
    weights = Path(weights or paths.DINOV3_WEIGHTS)
    if not weights.exists():
        raise FileNotFoundError(f"backbone weights not found at {weights} (set DINOV3_WEIGHTS)")
    return fn(pretrained=True, weights=str(weights))


def default_layers(depth: int, count: int = 4) -> list[int]:
    """Evenly spaced block indices ending at the last block, e.g. [2, 5, 8, 11] for depth 12."""
    step = depth // count
    return [depth - 1 - step * (count - 1 - i) for i in range(count)]


class DinoFeatures(nn.Module):
    """Wraps a DINOv3 ViT; returns patch-feature maps (B, C, H/16, W/16) from several blocks
    plus the last requested block's CLS token."""

    def __init__(self, arch: str = "vitb16", weights=None, repo=None, layers: list[int] | None = None, pretrained: bool = True):
        super().__init__()
        self.arch = arch
        self.vit = load_backbone(arch, weights, repo, pretrained)
        self.embed_dim = ARCHS[arch]["embed_dim"]
        self.depth = ARCHS[arch]["depth"]
        self.layers = list(layers) if layers else default_layers(self.depth)
        self.patch_size = self.vit.patch_size
        self.unfreeze_blocks = 0
        self.set_trainable(0)

    # ---- freezing -------------------------------------------------------------------------
    def set_trainable(self, unfreeze_blocks: int) -> None:
        self.unfreeze_blocks = unfreeze_blocks
        for p in self.vit.parameters():
            p.requires_grad_(False)
        if unfreeze_blocks == 0:
            return
        first = 0 if unfreeze_blocks < 0 else max(self.depth - unfreeze_blocks, 0)
        for i, blk in enumerate(self.vit.blocks):
            if i >= first:
                for p in blk.parameters():
                    p.requires_grad_(True)
        for p in self.vit.norm.parameters():
            p.requires_grad_(True)
        if unfreeze_blocks < 0:
            for name, p in self.vit.named_parameters():
                if not name.startswith("blocks.") and not name.startswith("norm."):
                    p.requires_grad_(True)

    def param_groups(self, lr: float, layer_decay: float = 0.8, weight_decay: float = 0.05) -> list[dict]:
        """Per-block learning rates: block i gets lr * layer_decay ** (depth - 1 - i)."""
        groups: dict[str, dict] = {}
        for name, p in self.vit.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("blocks."):
                idx = int(name.split(".")[1])
                scale = layer_decay ** (self.depth - 1 - idx)
            elif name.startswith("norm."):
                scale = 1.0
            else:
                scale = layer_decay ** self.depth
            no_decay = p.ndim <= 1 or name.endswith("token") or "tokens" in name
            key = f"{scale:.6f}_{'nd' if no_decay else 'wd'}"
            group = groups.setdefault(key, {"params": [], "lr": lr * scale, "weight_decay": 0.0 if no_decay else weight_decay, "name": f"backbone_{key}"})
            group["params"].append(p)
        return list(groups.values())

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.unfreeze_blocks == 0:
            self.vit.eval()
        return self

    # ---- forward --------------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        if H % self.patch_size or W % self.patch_size:
            raise ValueError(f"input {H}x{W} must be a multiple of the patch size {self.patch_size}")
        if self.unfreeze_blocks == 0:
            with torch.no_grad():
                outs = self.vit.get_intermediate_layers(x, n=self.layers, reshape=True, return_class_token=True, norm=True)
        else:
            outs = self.vit.get_intermediate_layers(x, n=self.layers, reshape=True, return_class_token=True, norm=True)
        feats = [o[0] for o in outs]
        cls = outs[-1][1]
        return feats, cls


def normalize_image(x: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) float in [0,1] -> ImageNet-normalised."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std
