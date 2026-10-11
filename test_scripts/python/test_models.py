"""Model construction and forward shapes (random weights, no checkpoint needed; GPU optional)."""

from __future__ import annotations

import unittest

import torch

from model_fine_tuning.bodycoords.model import WhaleBodyModel, body_loss
from model_fine_tuning.common.data import IGNORE
from model_fine_tuning.segmentation.model import WhaleSegModel, seg_loss


class ModelTest(unittest.TestCase):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_seg_forward_and_loss(self):
        model = WhaleSegModel(arch="vits16", unfreeze_blocks=2, pretrained=False).to(self.device)
        x = torch.rand(2, 3, 128, 192, device=self.device)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 128, 192))
        target = torch.randint(0, 2, (2, 128, 192), device=self.device)
        target[:, :8] = IGNORE
        loss, parts = seg_loss(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        trainable = [n for n, p in model.backbone.vit.named_parameters() if p.requires_grad]
        self.assertTrue(all(n.startswith(("blocks.10.", "blocks.11.", "norm.")) for n in trainable), trainable[:3])

    def test_frozen_backbone_has_no_grad(self):
        model = WhaleSegModel(arch="vits16", unfreeze_blocks=0, pretrained=False).to(self.device)
        self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
        groups = model.param_groups(1e-3, 1e-5, 0.8, 0.05)
        self.assertTrue(all(g["name"].startswith("head") for g in groups))

    def test_body_forward_and_loss(self):
        model = WhaleBodyModel(arch="vits16", unfreeze_blocks=0, pretrained=False).to(self.device)
        x = torch.rand(2, 3, 128, 128, device=self.device)
        out = model(x)
        self.assertEqual(tuple(out["u"].shape), (2, 128, 128))
        self.assertEqual(tuple(out["sincos"].shape), (2, 2, 128, 128))
        self.assertEqual(tuple(out["R"].shape), (2, 3, 3))
        det = torch.linalg.det(out["R"])
        self.assertTrue(torch.allclose(det, torch.ones_like(det), atol=1e-4))
        batch = {"mask": torch.randint(0, 2, (2, 128, 128), device=self.device), "u": torch.rand(2, 128, 128, device=self.device),
                 "sincos": torch.nn.functional.normalize(torch.randn(2, 2, 128, 128, device=self.device), dim=1),
                 "valid": torch.rand(2, 128, 128, device=self.device) > 0.5, "R": torch.eye(3, device=self.device).expand(2, 3, 3),
                 "log_ppm": torch.tensor([3.0, 4.0], device=self.device)}
        loss, parts = body_loss(out, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(parts), {"vis", "u_l1", "angle", "rot", "scale"})


if __name__ == "__main__":
    unittest.main()
