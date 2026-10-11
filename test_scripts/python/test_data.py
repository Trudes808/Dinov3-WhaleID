"""Dataset reading and augmentation invariants (needs the run on disk)."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from model_fine_tuning.common import paths
from model_fine_tuning.common.data import IGNORE, BodyCropDataset, SegCropDataset, WhaleRun, decode_bodycoords, pad_to_multiple


class DataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = WhaleRun(paths.WHALE_RUN_DIR, paths.SPLITS_DIR / "run01.json")

    def test_split_is_a_partition(self):
        train, val, test = (set(self.data.ids(s)) for s in ("train", "val", "test"))
        self.assertFalse(train & val or train & test or val & test)
        self.assertEqual(len(train | val | test), len(self.data.rows))

    def test_seg_train_sample(self):
        ds = SegCropDataset(self.data, self.data.ids("train")[:4], crop=256, train=True)
        s = ds[0]
        self.assertEqual(tuple(s["image"].shape), (3, 256, 256))
        self.assertEqual(tuple(s["mask"].shape), (256, 256))
        self.assertTrue(set(torch.unique(s["mask"]).tolist()) <= {0, 1, IGNORE})
        self.assertTrue(0.0 <= s["image"].min() and s["image"].max() <= 1.0)

    def test_seg_eval_sample_is_full_frame(self):
        ds = SegCropDataset(self.data, self.data.ids("val")[:1], train=False)
        s = ds[0]
        self.assertEqual(tuple(s["image"].shape[1:]), (720, 1280))
        self.assertEqual(s["orig_hw"].tolist(), [720, 1280])

    def test_body_sample_targets(self):
        ids = [i for i in self.data.ids("train") if self.data.by_id[i]["visible_pixels"] > 2000][:3]
        ds = BodyCropDataset(self.data, ids, crop=256, train=True)
        for i in range(len(ds)):
            s = ds[i]
            v = s["valid"]
            self.assertGreater(int(v.sum()), 0)
            self.assertTrue(((s["u"][v] >= 0) & (s["u"][v] <= 1)).all())
            norms = torch.linalg.norm(s["sincos"], dim=0)[v]
            self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-4))
            self.assertAlmostEqual(float(torch.det(s["R"])), 1.0, places=4)
            self.assertTrue((s["mask"][v] == 1).all())

    def test_bodycoords_decode(self):
        rgb = np.zeros((2, 2, 3), np.uint8)
        rgb[0, 0] = (255, 128, 255)
        u, theta, defined = decode_bodycoords(rgb)
        self.assertAlmostEqual(float(u[0, 0]), 1.0)
        self.assertAlmostEqual(float(theta[0, 0]), 128 / 255 * 360, places=4)
        self.assertTrue(defined[0, 0] and not defined[1, 1])

    def test_pad_to_multiple(self):
        img = np.zeros((701, 1001, 3), np.uint8)
        padded, (h, w) = pad_to_multiple(img, 16)
        self.assertEqual(padded.shape[:2], (704, 1008))
        self.assertEqual((h, w), (701, 1001))


if __name__ == "__main__":
    unittest.main()
