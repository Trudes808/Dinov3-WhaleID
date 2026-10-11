"""Geometry conventions: GT pose reprojects landmarks onto the stored keypoints; PnP on the
ground-truth body coordinates recovers the ground-truth pose; training-time flips keep
coordinates and rotation consistent. Needs the run and the body table."""

from __future__ import annotations

import unittest

import numpy as np

from model_fine_tuning.common import geometry, paths
from model_fine_tuning.common.data import BodyCropDataset, WhaleRun


class GeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = WhaleRun(paths.WHALE_RUN_DIR, paths.SPLITS_DIR / "run01.json")
        cls.table = geometry.load_body_table(paths.BODY_TABLE)
        cls.landmarks = geometry.load_landmarks(paths.ASSETS_DIR / "landmarks_mesh_frame_cm.json")
        # undeformed shots only: with body deformation (run02+) the rigid pose is approximate
        cls.ids = [i for i in cls.data.ids("val") if cls.data.by_id[i]["visible_pixels"] > 2000 and not cls.data.is_deformed(i)][:10]

    def test_rotation_6d_roundtrip(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            w, x, y, z = q
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            self.assertLess(geometry.geodesic_deg(geometry.rotation_from_6d_np(geometry.rotation_to_6d(R)), R), 1e-3)

    def test_gt_pose_projects_keypoints(self):
        for sid in self.ids:
            meta = self.data.meta(sid)
            R, t = geometry.gt_camera_from_whale(meta)
            K = geometry.intrinsics_from_meta(meta)
            names = [n for n in self.landmarks if meta["keypoints"][n]["in_frame"]]
            uv, _ = geometry.project_ue(np.stack([self.landmarks[n] for n in names]), R, t, K)
            gt = np.array([meta["keypoints"][n]["uv"] for n in names])
            self.assertLess(np.abs(uv - gt).max(), 0.5, sid)

    def test_pnp_recovers_gt_pose(self):
        for sid in self.ids:
            meta = self.data.meta(sid)
            R, t = geometry.gt_camera_from_whale(meta)
            K = geometry.intrinsics_from_meta(meta)
            u, th, d = self.data.bodycoords(sid)
            fit = geometry.fit_pose_pnp(u, th, self.data.visible(sid) & d, K, self.table)
            self.assertIsNotNone(fit, sid)
            self.assertLess(geometry.geodesic_deg(fit["R"], R), 4.0, sid)
            self.assertLess(abs(np.log(fit["ppm"] / meta["pixels_per_metre_at_whale"])), 0.15, sid)

    def test_flip_consistency(self):
        ds = BodyCropDataset(self.data, self.ids, train=True, photometric_strength=0.0, seed=3)
        seen_flip = False
        for i in range(len(ds)):
            s = ds[i]
            meta = self.data.meta(s["id"])
            x0, y0, sc = s["crop_x0y0s"].numpy()
            Kc = geometry.intrinsics_for_crop(geometry.intrinsics_from_meta(meta), x0, y0, sc)
            u, th, v = s["u"].numpy(), s["theta_deg"].numpy(), s["valid"].numpy()
            if s["flipped"]:
                seen_flip = True
                # undo the mirror on the arrays; the stored R must then be the flipped GT
                u, th, v = u[:, ::-1], (360 - th) % 360, v[:, ::-1]
                th = th[:, ::-1]
                fit = geometry.fit_pose_pnp(np.ascontiguousarray(u), np.ascontiguousarray(th), np.ascontiguousarray(v), Kc, self.table)
                self.assertLess(geometry.geodesic_deg(geometry.flip_rotation(fit["R"]), s["R"].numpy()), 5.0)
            else:
                fit = geometry.fit_pose_pnp(u, th, v, Kc, self.table)
                self.assertLess(geometry.geodesic_deg(fit["R"], s["R"].numpy()), 5.0)
            self.assertLess(abs(np.log(fit["ppm"] / np.exp(float(s["log_ppm"])))), 0.15)
        self.assertTrue(seen_flip)


if __name__ == "__main__":
    unittest.main()
