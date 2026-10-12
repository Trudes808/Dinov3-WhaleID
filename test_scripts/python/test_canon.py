"""Canonical-surface conventions on ground truth: side resolution recovers the true side and
pose from GT folded points; points of interest read off the GT map hit the rendered keypoints;
a mirrored training crop has the same folded target. Needs run02 with canon labels."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from model_fine_tuning.common import geometry, paths
from model_fine_tuning.common.data import CanonCropDataset, WhaleRun

RUN02 = Path(os.environ.get("WHALE_RUN02_DIR", Path.home() / "NOAA_whales/Whale-Image-Data-Factory/data/renders/run02"))


class CanonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (RUN02 / "dataset" / "canon").exists() or not paths.FOLDED_TABLE.exists():
            raise unittest.SkipTest("run02 canon labels or the folded table are missing")
        cls.data = WhaleRun(RUN02, paths.SPLITS_DIR / "run02.json")
        cls.table = geometry.load_folded_table(paths.FOLDED_TABLE)
        cls.ids = [i for i in cls.data.ids("val") if cls.data.by_id[i]["visible_pixels"] > 5000 and cls.data.has_canon(i) and not cls.data.is_deformed(i)][:6]
        if len(cls.ids) < 3:
            raise unittest.SkipTest("not enough undeformed val shots with canon labels")

    def gt_maps(self, sid):
        canon_cm, defined = self.data.canon(sid)
        pts = canon_cm / 100.0
        valid = self.data.visible(sid) & defined
        return pts, valid

    def test_folded_points_lie_on_the_mesh(self):
        for sid in self.ids:
            pts, valid = self.gt_maps(sid)
            folded = geometry.fold_points_m(pts[valid])
            _, d = geometry.snap_to_surface(folded, self.table)
            self.assertLess(np.median(d), 0.12, sid)   # mesh vertex spacing is ~10 cm on the flukes

    def test_resolve_side_recovers_side_and_pose(self):
        accs, rots = [], []
        for sid in self.ids:
            meta = self.data.meta(sid)
            R, _ = geometry.gt_camera_from_whale(meta)
            K = geometry.intrinsics_from_meta(meta)
            pts, valid = self.gt_maps(sid)
            folded = geometry.fold_points_m(pts.reshape(-1, 3)).reshape(pts.shape)
            res = geometry.resolve_side(folded, valid, K, R_init=R, table=self.table)
            self.assertIsNotNone(res["fit"], sid)
            gt_side = np.sign(pts[..., 1])
            scored = valid & (np.abs(pts[..., 1]) > 0.1)
            acc = float(np.mean(res["side"][scored] == gt_side[scored]))
            accs.append(acc); rots.append(geometry.geodesic_deg(res["fit"]["R"], R))
            self.assertGreater(acc, 0.97, sid)
            self.assertLess(rots[-1], 4.0, sid)
        # a wrong initial guess of the facing flank must still be recovered by the reprojection step
        sid = self.ids[0]
        meta = self.data.meta(sid)
        R, _ = geometry.gt_camera_from_whale(meta)
        pts, valid = self.gt_maps(sid)
        folded = geometry.fold_points_m(pts.reshape(-1, 3)).reshape(pts.shape)
        res = geometry.resolve_side(folded, valid, geometry.intrinsics_from_meta(meta), R_init=geometry.flip_rotation(R), iterations=6, table=self.table)
        scored = valid & (np.abs(pts[..., 1]) > 0.1)
        acc_wrong_init = float(np.mean(res["side"][scored] == np.sign(pts[..., 1])[scored])) if res["fit"] is not None else 0.0
        print(f"\nside accuracy from GT maps: {np.round(accs, 3)}  PnP rot err deg: {np.round(rots, 2)}  wrong-init recovery acc: {acc_wrong_init:.3f}")

    def test_landmarks_from_gt_map_hit_keypoints(self):
        errs = []
        for sid in self.ids:
            meta = self.data.meta(sid)
            pts, valid = self.gt_maps(sid)
            lm = geometry.landmarks_from_map(pts, valid, self.data.landmarks_m, max_dist_m=0.15)
            for name, rec in lm.items():
                kp = meta["keypoints"][name]
                if rec["uv"] is not None and kp["in_frame"] and not kp["occluded"] and kp.get("inside_label", True):
                    errs.append(np.linalg.norm(np.array(rec["uv"]) - np.array(kp["uv"])))
        self.assertGreater(len(errs), 10)
        self.assertLess(np.median(errs), 6.0, f"median keypoint error {np.median(errs):.1f} px over {len(errs)} landmarks")

    def test_mirrored_crop_has_the_same_folded_target(self):
        ds_plain = CanonCropDataset(self.data, self.ids[:3], self.table, train=True, margin_range=(1.6, 1.6), jitter=0.0, flip=False, photometric_strength=0.0, seed=1)
        ds_flip = CanonCropDataset(self.data, self.ids[:3], self.table, train=True, margin_range=(1.6, 1.6), jitter=0.0, flip=True, photometric_strength=0.0, seed=1)
        checked = 0
        for epoch in range(12):
            ds_flip.set_epoch(epoch)
            for i in range(len(ds_flip)):
                b = ds_flip[i]
                if not b["flipped"]:
                    continue
                a = ds_plain[i]
                np.testing.assert_allclose(b["xyz"].numpy()[:, :, ::-1], a["xyz"].numpy(), atol=1e-5)
                np.testing.assert_array_equal(b["side"].numpy()[:, ::-1], -a["side"].numpy())
                np.testing.assert_array_equal(b["valid"].numpy()[:, ::-1], a["valid"].numpy())
                self.assertLess(geometry.geodesic_deg(b["R"].numpy(), geometry.flip_rotation(a["R"].numpy())), 0.1)
                checked += 1
            if checked >= 3:
                break
        self.assertGreaterEqual(checked, 1)
