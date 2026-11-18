"""HW-b6 split into a frozen front (no grad, no stored activations) and a trainable back. BatchNorm statistics stay as
in the public checkpoint (the whole net runs in eval mode, also while training): batches here are 16 crops of one
catalogue group, far too few and too alike for batch statistics."""
import torch

import common  # noqa: F401  (sets sys.path for bench / extract)
from extract import HappyWhaleB6


class HW(torch.nn.Module):
    def __init__(self, unfreeze_from=5):
        super().__init__()
        self.net = HappyWhaleB6()
        self.k = unfreeze_from                                      # blocks[k:] + conv_head + bn2 + GeM + BN embedding train
        b = self.net.backbone
        front = [b.conv_stem, b.bn1] + list(b.blocks[:self.k])
        for m in front:
            for p in m.parameters():
                p.requires_grad_(False)
        self.front = torch.nn.Sequential(*front)
        self.back = torch.nn.Sequential(*b.blocks[self.k:], b.conv_head, b.bn2)

    def forward(self, x):
        with torch.no_grad():                                      # in chunks: the 1024-px front's activations are large
            h = torch.cat([self.front(c) for c in x.split(4)])
        return self.net.bn(self.net.pool(self.back(h)).flatten(1))

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]
