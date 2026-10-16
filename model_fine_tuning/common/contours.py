"""Contour classes of the factory's visible mask and what the models do with them.

Whale-Image-Data-Factory (run04+) labels every boundary pixel of the visible mask (contours/<id>.png):
1 = the whale's own silhouette (the body ends there, against sky or water: open air), 2 = a seam, where the
body continues under the water surface or behind foam / spray, 3 = the edge against the calf. A real
photo's top contour is the trustworthy one; the waterline and foam edges are not, so:

  * training (contour_weight_map): the segmentation / visibility BCE and the canonical-coordinate L1
    are re-weighted in a band around the contours, up at the true silhouette, down at the seams;
  * evaluation (boundary_f1_split): the boundary F-score is reported separately away from the seams
    ("open", the part a mesh fit should match) and at the seams.

Runs without the label (run01-03) give all-zero contour maps, which leave both unchanged.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

CONTOUR_NONE, CONTOUR_TRUE, CONTOUR_SEAM, CONTOUR_CALF = 0, 1, 2, 3


def contour_weight_map(contours: torch.Tensor, band_px: int = 3, w_true: float = 1.0, w_seam: float = 1.0) -> torch.Tensor:
    """(B, H, W) contour classes -> (B, H, W) float32 loss weights: w_true within band_px of a true-silhouette
    pixel, w_seam within band_px of a seam / calf-edge pixel (where the two bands overlap the silhouette wins),
    1 everywhere else. band_px < 0 or unit weights return all ones."""
    ones = torch.ones(contours.shape, dtype=torch.float32, device=contours.device)
    if band_px < 0 or (w_true == 1.0 and w_seam == 1.0):
        return ones
    k = 2 * band_px + 1
    c = contours[:, None]
    near_true = F.max_pool2d((c == CONTOUR_TRUE).float(), k, stride=1, padding=band_px)[:, 0] > 0
    near_seam = F.max_pool2d((c >= CONTOUR_SEAM).float(), k, stride=1, padding=band_px)[:, 0] > 0
    w = torch.where(near_seam, torch.full_like(ones, float(w_seam)), ones)
    return torch.where(near_true, torch.full_like(ones, float(w_true)), w)


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool)


def _f1(bp: np.ndarray, bg: np.ndarray, tolerance_px: int) -> float | None:
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


def _counts(bp: np.ndarray, bg: np.ndarray, tolerance_px: int) -> dict:
    """Boundary-match counts (pred pixels, pred pixels within tolerance of a gt boundary pixel, gt pixels, gt pixels
    within tolerance of a pred boundary pixel); F-scores are computed from these, per image or pooled."""
    k = np.ones((2 * tolerance_px + 1, 2 * tolerance_px + 1), np.uint8)
    dp = cv2.dilate(bp.astype(np.uint8), k).astype(bool) if bp.any() else np.zeros_like(bp)
    dg = cv2.dilate(bg.astype(np.uint8), k).astype(bool) if bg.any() else np.zeros_like(bg)
    return {"pred": int(bp.sum()), "pred_hit": int(np.sum(bp & dg)), "gt": int(bg.sum()), "gt_hit": int(np.sum(bg & dp))}


def f1_from_counts(c: dict, min_gt: int = 1) -> float | None:
    """Boundary F-score from match counts; None when there is no gt boundary (or fewer than min_gt pixels of it)."""
    if c["gt"] < max(min_gt, 1):
        return None
    if c["pred"] == 0:
        return 0.0
    precision, recall = c["pred_hit"] / c["pred"], c["gt_hit"] / c["gt"]
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


MIN_OPEN_PX = 40   # per-image open-air F-score only when the gt has at least this many open-air boundary pixels
OPEN_NEAR_PX = 8   # predicted boundary pixels count towards the open-air score only within this distance of the gt silhouette


def boundary_f1_split(pred: np.ndarray, gt: np.ndarray, contours: np.ndarray | None, tolerance_px: int = 3,
                      ignore: np.ndarray | None = None) -> dict:
    """Boundary F-score (Perazzi et al., tolerance in pixels) of a predicted mask against the visible-whale mask,
    split by the factory contour classes. "bf1_open" scores the whale's own silhouette: the gt boundary pixels
    further than tolerance + 1 px from any seam / calf-edge pixel, against the predicted boundary pixels within
    OPEN_NEAR_PX of them (a predicted boundary that strays past the waterline or a foam edge by more than the seam
    zone is a seam error, not a silhouette error, and is left out here; a bulge past the silhouette costs recall).
    "bf1_seam" scores the boundary pixels of both masks within the seam zone. The per-image
    open score is None when the gt has fewer than MIN_OPEN_PX open-air boundary pixels (an elevated camera sees
    the far waterline, so a surfacing whale's back can be all seam: nothing to score). The match counts
    ("open_counts", "seam_counts": pred / pred_hit / gt / gt_hit) are returned for pooling over a dataset, which
    is the robust aggregate. Also the recall of the true-silhouette pixels themselves ("true_contour_recall": a
    predicted boundary within tolerance) and their count. ignore: the don't-care pixels (the factory's through-water
    class, padding): boundary pixels of either mask within tolerance + 1 px of them are left out, otherwise a
    prediction that paints whale into the through-water zone grows a false boundary at that zone's outer edge.
    Without a contour label every key is None."""
    out = {"bf1_open": None, "bf1_seam": None, "true_contour_recall": None, "true_contour_px": 0, "open_counts": None, "seam_counts": None}
    if contours is None or not (contours > 0).any():
        return out
    contours = contours.copy()                        # the frame edge is no contour (older labels marked it as one)
    contours[0, :] = contours[-1, :] = 0; contours[:, 0] = contours[:, -1] = 0
    seam = contours >= CONTOUR_SEAM
    k = np.ones((2 * tolerance_px + 3, 2 * tolerance_px + 3), np.uint8)
    seam_zone = cv2.dilate(seam.astype(np.uint8), k).astype(bool)
    bp, bg = _boundary(pred), _boundary(gt)
    if ignore is not None and ignore.any():
        ignore_zone = cv2.dilate(ignore.astype(np.uint8), k).astype(bool)
        bp, bg = bp & ~ignore_zone, bg & ~ignore_zone
    open_gt = bg & ~seam_zone
    near_open = cv2.dilate(open_gt.astype(np.uint8), np.ones((2 * OPEN_NEAR_PX + 1, 2 * OPEN_NEAR_PX + 1), np.uint8)).astype(bool)
    out["open_counts"] = _counts(bp & near_open & ~seam_zone, open_gt, tolerance_px)
    out["seam_counts"] = _counts(bp & seam_zone, bg & seam_zone, tolerance_px)
    out["bf1_open"] = f1_from_counts(out["open_counts"], MIN_OPEN_PX)
    out["bf1_seam"] = f1_from_counts(out["seam_counts"])
    true_px = contours == CONTOUR_TRUE
    out["true_contour_px"] = int(true_px.sum())
    if true_px.any():
        kt = np.ones((2 * tolerance_px + 1, 2 * tolerance_px + 1), np.uint8)
        near_pred = cv2.dilate(bp.astype(np.uint8), kt).astype(bool)
        out["true_contour_hit"] = int(np.sum(true_px & near_pred))
        out["true_contour_recall"] = float(out["true_contour_hit"] / true_px.sum())
    return out


def pooled_boundary_scores(rows: list[dict]) -> dict:
    """Dataset-level boundary scores from per-image boundary_f1_split rows: the F-score of the pooled match counts
    away from the seams and at them, and the pooled recall of the true-silhouette pixels."""
    out = {}
    for key in ("open", "seam"):
        cs = [r[f"{key}_counts"] for r in rows if r.get(f"{key}_counts")]
        if cs:
            tot = {k: sum(c[k] for c in cs) for k in ("pred", "pred_hit", "gt", "gt_hit")}
            out[f"boundary_f1_{key}_pooled"] = f1_from_counts(tot)
            out[f"boundary_precision_{key}_pooled"] = tot["pred_hit"] / tot["pred"] if tot["pred"] else None
            out[f"boundary_recall_{key}_pooled"] = tot["gt_hit"] / tot["gt"] if tot["gt"] else None
    n_true = sum(r.get("true_contour_px", 0) for r in rows if r.get("true_contour_px"))
    if n_true:
        out["true_contour_recall_pooled"] = sum(r.get("true_contour_hit", 0) for r in rows) / n_true
    return out
