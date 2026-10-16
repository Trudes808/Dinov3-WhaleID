"""Metrics for the two heads."""

from __future__ import annotations

import numpy as np

from .contours import pooled_boundary_scores


def binary_confusion(pred: np.ndarray, gt: np.ndarray, ignore: np.ndarray | None = None) -> dict:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if ignore is not None:
        keep = ~ignore
        pred, gt = pred[keep], gt[keep]
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    tn = int(np.sum(~pred & ~gt))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def iou_from_confusion(c: dict) -> float | None:
    denom = c["tp"] + c["fp"] + c["fn"]
    return None if denom == 0 else c["tp"] / denom


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance_px: int = 3) -> float | None:
    """F-score of boundary pixels within a tolerance (Perazzi et al.); None if no boundaries."""
    import cv2

    def boundary(m):
        m = m.astype(np.uint8)
        er = cv2.erode(m, np.ones((3, 3), np.uint8))
        return (m - er).astype(bool)

    bp, bg = boundary(pred), boundary(gt)
    if not bp.any() and not bg.any():
        return None
    if not bp.any() or not bg.any():
        return 0.0
    k = np.ones((2 * tolerance_px + 1, 2 * tolerance_px + 1), np.uint8)
    dp = cv2.dilate(bp.astype(np.uint8), k).astype(bool)
    dg = cv2.dilate(bg.astype(np.uint8), k).astype(bool)
    precision = np.sum(bp & dg) / max(np.sum(bp), 1)
    recall = np.sum(bg & dp) / max(np.sum(bg), 1)
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


class SegAccumulator:
    """Dataset-level and per-image segmentation metrics, with group breakdowns."""

    def __init__(self):
        self.rows = []

    def add(self, shot_id: str, pred: np.ndarray, gt: np.ndarray, groups: dict | None = None, boundary: bool = True,
            ignore: np.ndarray | None = None, contours: np.ndarray | None = None) -> dict:
        """ignore: pixels left out of the confusion counts (label class 2, padding). For the
        boundary score they are treated as background in both pred and gt. contours: the factory's
        contour classes (run04+) for the boundary score split into open-air silhouette vs seams."""
        c = binary_confusion(pred, gt, ignore)
        row = {"id": shot_id, **c, "iou": iou_from_confusion(c), "gt_pixels": int(gt.sum()), "pred_pixels": int(pred.sum())}
        if ignore is not None and ignore.any():
            row["ignored_pixels"] = int(ignore.sum())
            pred, gt = pred & ~ignore, gt & ~ignore
        row["bf1"] = boundary_f1(pred, gt) if boundary else None
        if contours is not None and (contours > 0).any():
            from .contours import boundary_f1_split

            row.update(boundary_f1_split(pred, gt, contours, ignore=ignore))
        row.update(groups or {})
        self.rows.append(row)
        return row

    def summary(self, group_keys=("category", "sea_state", "size_bin")) -> dict:
        rows = self.rows
        out = {"images": len(rows)}
        out.update(self._agg(rows))
        for key in group_keys:
            values = sorted({r[key] for r in rows if key in r})
            if values:
                out[f"by_{key}"] = {v: self._agg([r for r in rows if r.get(key) == v]) for v in values}
        return out

    @staticmethod
    def _agg(rows: list[dict]) -> dict:
        tp = sum(r["tp"] for r in rows)
        fp = sum(r["fp"] for r in rows)
        fn = sum(r["fn"] for r in rows)
        with_gt = [r for r in rows if r["gt_pixels"] > 0]
        without = [r for r in rows if r["gt_pixels"] == 0]
        ious = [r["iou"] for r in with_gt]
        bfs = [r["bf1"] for r in with_gt if r["bf1"] is not None]
        return {
            "n": len(rows),
            "dataset_iou": (tp / (tp + fp + fn)) if tp + fp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "mean_image_iou": float(np.mean(ious)) if ious else None,
            "median_image_iou": float(np.median(ious)) if ious else None,
            "iou_at_0.5": float(np.mean([i >= 0.5 for i in ious])) if ious else None,
            "iou_at_0.75": float(np.mean([i >= 0.75 for i in ious])) if ious else None,
            "boundary_f1": float(np.mean(bfs)) if bfs else None,
            **{f"boundary_f1_{k}": (float(np.mean(v)) if v else None) for k in ("open", "seam")
               for v in [[r[f"bf1_{k}"] for r in with_gt if r.get(f"bf1_{k}") is not None]]},
            "true_contour_recall": float(np.mean([r["true_contour_recall"] for r in with_gt if r.get("true_contour_recall") is not None]))
            if any(r.get("true_contour_recall") is not None for r in with_gt) else None,
            **pooled_boundary_scores(with_gt),
            "negatives": len(without),
            "negatives_clean": float(np.mean([r["pred_pixels"] == 0 for r in without])) if without else None,
            "negatives_fp_pixels_mean": float(np.mean([r["pred_pixels"] for r in without])) if without else None,
        }


def size_bin(visible_pixels: int) -> str:
    if visible_pixels == 0:
        return "0_none"
    if visible_pixels < 500:
        return "1_tiny<500px"
    if visible_pixels < 3000:
        return "2_small<3k"
    if visible_pixels < 20000:
        return "3_medium<20k"
    return "4_large"


def angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (a - b + 180.0) % 360.0 - 180.0
    return np.abs(d)


def summarize_values(values, prefix: str = "") -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return {f"{prefix}n": 0}
    return {f"{prefix}n": int(len(v)), f"{prefix}mean": float(v.mean()), f"{prefix}median": float(np.median(v)),
            f"{prefix}p90": float(np.percentile(v, 90))}
