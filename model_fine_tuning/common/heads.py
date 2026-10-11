"""Dense decoder shared by both models.

Four DINOv3 feature maps (stride 16) are projected to a common width and summed, refined
with two 3x3 convolutions, upsampled x4, and fused with a shallow stride-4 stem computed
from the RGB image. The stem carries the edge detail the 16-px patches lose, which
matters for the whale/water boundary. Output is predicted at stride 4 and bilinearly
upsampled to full resolution.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def conv_gn_gelu(cin: int, cout: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.GroupNorm(32 if cout % 32 == 0 else 8, cout), nn.GELU())


class DenseDecoder(nn.Module):
    def __init__(self, in_dim: int, n_layers: int, out_channels: int, width: int = 256, stem_width: int = 64, fine_width: int = 128):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(in_dim, width, 1) for _ in range(n_layers)])
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        self.coarse = nn.Sequential(conv_gn_gelu(width, width), conv_gn_gelu(width, width))
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_width // 2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(stem_width // 2, stem_width, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(stem_width, stem_width, 3, padding=1), nn.GELU(),
        )
        self.fine = nn.Sequential(conv_gn_gelu(width + stem_width, fine_width), conv_gn_gelu(fine_width, fine_width))
        self.out = nn.Conv2d(fine_width, out_channels, 1)
        self.width = width

    def forward(self, feats: list[torch.Tensor], image: torch.Tensor):
        """feats: list of (B, C, h, w) at stride 16; image: normalised (B, 3, H, W).
        Returns (full-res output (B, out, H, W), coarse fused features (B, width, h, w))."""
        w = torch.softmax(self.layer_weights, 0)
        x = sum(w[i] * self.proj[i](f) for i, f in enumerate(feats))
        x = self.coarse(x)
        coarse = x
        H, W = image.shape[-2:]
        x = F.interpolate(x, size=(H // 4, W // 4), mode="bilinear", align_corners=False)
        s = self.stem(image)
        x = self.fine(torch.cat([x, s], 1))
        out = self.out(x)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out, coarse
