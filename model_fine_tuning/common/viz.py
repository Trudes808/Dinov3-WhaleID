"""Overlays for qualitative checks."""

from __future__ import annotations

import numpy as np


def seg_overlay(image: np.ndarray, pred: np.ndarray, gt: np.ndarray | None = None, alpha: float = 0.5) -> np.ndarray:
    """Pred-only: red. With GT: TP green, FP red, FN blue."""
    out = image.astype(np.float32).copy()
    pred = pred.astype(bool)
    if gt is None:
        out[pred] = (1 - alpha) * out[pred] + alpha * np.array([255, 0, 0])
    else:
        gt = gt.astype(bool)
        for region, colour in ((pred & gt, (0, 220, 0)), (pred & ~gt, (255, 0, 0)), (~pred & gt, (0, 90, 255))):
            out[region] = (1 - alpha) * out[region] + alpha * np.array(colour)
    return out.astype(np.uint8)


def bodycoord_rgb(u: np.ndarray, theta_deg: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Same encoding as the dataset's bodycoords PNGs: R = u, G = theta/360, B = 255 where valid."""
    out = np.zeros(u.shape + (3,), np.uint8)
    out[..., 0] = np.clip(np.round(u * 255), 0, 255)
    out[..., 1] = np.clip(np.round((theta_deg % 360.0) / 360.0 * 255), 0, 255)
    out[..., 2] = 255
    out[~valid.astype(bool)] = 0
    return out


def bodycoord_overlay(image: np.ndarray, u: np.ndarray, theta_deg: np.ndarray, valid: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    colours = bodycoord_rgb(u, theta_deg, valid).astype(np.float32)
    out = image.astype(np.float32).copy()
    v = valid.astype(bool)
    out[v] = (1 - alpha) * out[v] + alpha * colours[v]
    return out.astype(np.uint8)


def draw_points(image: np.ndarray, points: dict, colour=(255, 255, 0), radius: int = 4) -> np.ndarray:
    import cv2

    out = np.ascontiguousarray(image.copy())
    for name, (x, y) in points.items():
        if not (np.isfinite(x) and np.isfinite(y)) or abs(x) > 1e5 or abs(y) > 1e5:
            continue  # behind the camera or wildly off-image
        cx, cy = int(round(float(x))), int(round(float(y)))
        cv2.circle(out, (cx, cy), radius, colour, -1)
        cv2.putText(out, name, (cx + 5, cy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
    return out


def hstack(images: list[np.ndarray]) -> np.ndarray:
    h = max(i.shape[0] for i in images)
    padded = [np.pad(i, ((0, h - i.shape[0]), (0, 0), (0, 0))) for i in images]
    return np.concatenate(padded, axis=1)
