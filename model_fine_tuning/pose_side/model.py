"""Keypoint + state head on frozen DINOv3 ViT-B/16 patch features (blocks 5, 8, 11; stride 16), the recipe of the
human top-down models (ViTPose: plain ViT + light decoder to a stride-4 heatmap) with two additions from the occlusion
literature: a per-keypoint state (visible / under water / hidden by splash / behind a body / outside the crop; as
ProbPose's presence + visibility) and amodal positions for points under water or behind splash (trained from renders).
Heatmap loss: cross-entropy of a spatial soft-max against a Gaussian (sigma 1.5 cells = 6 px at 512)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..common.backbone import load_backbone

LAYERS = [5, 8, 11]


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = load_backbone("vitb16").eval()
        for p in self.vit.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fs = self.vit.get_intermediate_layers(x, n=LAYERS, reshape=True, norm=True)
        return torch.cat([f.float() for f in fs], 1)


def block(c):
    return nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(16, c), nn.GELU())


class Head(nn.Module):
    def __init__(self, cin=768 * len(LAYERS), K=6, S=5, c=256):
        super().__init__()
        self.K = K
        self.proj = nn.Sequential(nn.Conv2d(cin, c, 1), nn.GroupNorm(16, c), nn.GELU())
        self.body = nn.Sequential(block(c), block(c), block(c))
        self.up = nn.Sequential(nn.ConvTranspose2d(c, 128, 4, 2, 1), nn.GroupNorm(16, 128), nn.GELU(),
                                nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GroupNorm(16, 64), nn.GELU())
        self.hm = nn.Conv2d(64, K, 3, padding=1)
        self.kemb = nn.Parameter(torch.zeros(K, c))
        self.state = nn.Sequential(nn.Linear(2 * c, 256), nn.GELU(), nn.Linear(256, S))

    def forward(self, f):
        h = self.body(self.proj(f))                                  # B c 32 32
        hm = self.hm(self.up(h))                                      # B K 128 128
        att = F.adaptive_avg_pool2d(hm, h.shape[-2:]).flatten(2).softmax(-1)          # B K 1024
        pooled = torch.einsum("bkn,bcn->bkc", att, h.flatten(2))                       # B K c
        glob = h.mean((2, 3))[:, None].expand(-1, self.K, -1)
        st = self.state(torch.cat([pooled + self.kemb, glob], -1))                     # B K S
        return hm, st


def heat_loss(hm, pts, mask, sigma=1.5, stride=4):
    B, K, H, W = hm.shape
    ys = torch.arange(H, device=hm.device, dtype=torch.float32) * stride + (stride - 1) / 2
    xs = torch.arange(W, device=hm.device, dtype=torch.float32) * stride + (stride - 1) / 2
    d2 = (xs[None, None, None, :] - pts[..., 0, None, None]) ** 2 + (ys[None, None, :, None] - pts[..., 1, None, None]) ** 2
    t = torch.exp(-d2 / (2 * (sigma * stride) ** 2)); t = t / t.sum((2, 3), keepdim=True).clamp_min(1e-8)
    logp = hm.flatten(2).log_softmax(-1).view_as(hm)
    ce = -(t * logp).sum((2, 3))                                     # B K
    m = mask.float(); return (ce * m).sum() / m.sum().clamp_min(1)


def decode(hm, stride=4):
    """B K H W logits -> B K 2 (crop px), B K peak probability"""
    B, K, H, W = hm.shape
    p = hm.flatten(2).softmax(-1); conf, idx = p.max(-1)
    iy, ix = idx // W, idx % W
    pv = p.view(B, K, H, W); out = torch.zeros(B, K, 2, device=hm.device)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy, xx = (iy + dy).clamp(0, H - 1), (ix + dx).clamp(0, W - 1)
            w = pv.gather(2, yy[..., None, None].expand(-1, -1, 1, W)).squeeze(2).gather(2, xx[..., None]).squeeze(2)
            out[..., 0] += w * (xx.float() * stride + (stride - 1) / 2); out[..., 1] += w * (yy.float() * stride + (stride - 1) / 2)
    wsum = torch.zeros_like(conf)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy, xx = (iy + dy).clamp(0, H - 1), (ix + dx).clamp(0, W - 1)
            wsum += pv.gather(2, yy[..., None, None].expand(-1, -1, 1, W)).squeeze(2).gather(2, xx[..., None]).squeeze(2)
    return out / wsum[..., None].clamp_min(1e-9), conf
