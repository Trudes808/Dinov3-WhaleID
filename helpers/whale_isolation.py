"""Generate whale masks and crops from cached previews.

This module isolates whale-body pixels with deterministic classical image
processing. It writes a full-image mask, a padded review crop, a background-muted
masked crop for feature extraction, and metadata useful for review.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


ISOLATION_METHOD = "dark_anchor_local_growth_v4"


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80]


def isolation_paths_for_image(
    image_path: str | Path,
    crop_dir: str | Path,
    mask_dir: str | Path,
) -> tuple[Path, Path, Path, Path]:
    """Return deterministic review crop, masked crop, mask, and metadata paths."""
    source_path = Path(image_path)
    digest = hashlib.sha1(str(source_path.resolve()).encode("utf-8")).hexdigest()[:14]
    stem = f"{digest}_{_safe_stem(source_path.stem)}"
    return (
        Path(crop_dir) / f"{stem}_crop.jpg",
        Path(crop_dir) / f"{stem}_masked.jpg",
        Path(mask_dir) / f"{stem}_mask.png",
        Path(mask_dir) / f"{stem}.json",
    )


def _load_rgb(image_path: str | Path, max_side: int | None = None) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    if max_side and max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def _border_sample(array: np.ndarray, border_fraction: float = 0.08) -> np.ndarray:
    height, width = array.shape[:2]
    border = max(2, int(min(height, width) * border_fraction))
    samples = [
        array[:border, :, :],
        array[-border:, :, :],
        array[:, :border, :],
        array[:, -border:, :],
    ]
    return np.concatenate([sample.reshape(-1, array.shape[-1]) for sample in samples], axis=0)


def _odd_kernel(value: float | int, minimum: int = 3) -> int:
    size = max(minimum, int(value))
    return size if size % 2 == 1 else size + 1


def _robust_normalize(values: np.ndarray, low_percentile: float = 2.0, high_percentile: float = 98.0) -> np.ndarray:
    low = float(np.percentile(values, low_percentile))
    high = float(np.percentile(values, high_percentile))
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _estimate_waterline(gray: np.ndarray) -> int:
    height = gray.shape[0]
    if cv2 is None:
        return int(height * 0.12)
    gray_uint8 = (np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8)
    edges = cv2.Canny(gray_uint8, 40, 130).astype(np.float32) / 255.0
    row_texture = edges.mean(axis=1) + (0.45 * gray.std(axis=1))
    smooth_window = max(9, int(height * 0.035))
    kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
    smoothed = np.convolve(row_texture, kernel, mode="same")
    search_start = int(height * 0.08)
    search_end = int(height * 0.58)
    if search_end <= search_start:
        return int(height * 0.12)
    threshold = float(np.percentile(smoothed[search_start:], 58))
    candidates = np.where(smoothed[search_start:search_end] >= threshold)[0]
    return int(search_start + candidates[0]) if len(candidates) else int(height * 0.12)


def _foreground_scores(image: Image.Image) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    height, width = gray.shape

    water_rgb = np.median(_border_sample(rgb), axis=0)
    water_gray = float(np.median(_border_sample(gray[:, :, None])[:, 0]))
    blue_green_excess = np.clip(((rgb[:, :, 1] + rgb[:, :, 2]) / 2.0) - rgb[:, :, 0], 0.0, 1.0)
    water_distance_rgb = np.linalg.norm(rgb - water_rgb, axis=2)
    low_saturation = np.clip(1.0 - hsv[:, :, 1], 0.0, 1.0)
    channel_spread = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    neutral_dark = np.clip(1.0 - (channel_spread * 3.4), 0.0, 1.0)
    dark = np.clip((water_gray + 0.12 - gray) / 0.42, 0.0, 1.0)
    bright = np.clip((gray - water_gray + 0.02) / 0.38, 0.0, 1.0)

    if cv2 is not None:
        gray_uint8 = (np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8)
        local_mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(width, height) / 70)
        local_dark_contrast = np.clip(local_mean - gray, 0.0, 1.0)
        local_bright_contrast = np.clip(gray - local_mean, 0.0, 1.0)
        edges = cv2.Canny(gray_uint8, 32, 115).astype(np.float32) / 255.0
        edges = cv2.GaussianBlur(edges, (0, 0), 1.0)
        texture = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        lab = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
        water_lab = np.median(_border_sample(lab), axis=0)
        water_distance_lab = np.linalg.norm(lab - water_lab, axis=2)
        water_distance = _robust_normalize((0.55 * water_distance_rgb) + (0.45 * water_distance_lab))
    else:
        local_dark_contrast = dark
        local_bright_contrast = bright
        edges = np.asarray(image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
        texture = edges
        water_distance = _robust_normalize(water_distance_rgb)

    waterline = _estimate_waterline(gray)
    yy, xx = np.indices((height, width), dtype=np.float32)
    below_waterline = 1.0 / (1.0 + np.exp(-(yy - waterline) / max(height * 0.03, 1.0)))
    center_prior = 1.0 - np.clip(np.abs((xx / max(width - 1, 1)) - 0.5) / 0.62, 0.0, 1.0)
    vertical_prior = 1.0 - np.clip(np.abs((yy / max(height - 1, 1)) - 0.55) / 0.62, 0.0, 1.0)

    dark_anchor = (
        (1.65 * dark)
        + (0.95 * local_dark_contrast)
        + (0.55 * neutral_dark * dark)
        + (0.38 * water_distance)
        + (0.25 * edges)
        - (1.05 * blue_green_excess)
    )
    bright_body = (
        (0.95 * bright * low_saturation)
        + (0.75 * local_bright_contrast)
        + (0.62 * water_distance * low_saturation)
        + (0.28 * edges)
        - (0.55 * texture)
        - (0.70 * blue_green_excess)
    )
    local_whale = np.maximum(dark_anchor, bright_body)
    dark_anchor *= 0.44 + (0.56 * below_waterline)
    local_whale *= 0.48 + (0.52 * below_waterline)
    local_whale *= 0.74 + (0.16 * center_prior) + (0.10 * vertical_prior)
    dark_anchor *= 0.75 + (0.15 * center_prior) + (0.10 * vertical_prior)

    return {
        "dark_anchor": _robust_normalize(dark_anchor),
        "local_whale": _robust_normalize(local_whale),
    }, {
        "estimated_waterline_y": int(waterline),
        "water_gray": water_gray,
        "water_rgb": [float(value) for value in water_rgb],
    }


def _component_candidates(mask: np.ndarray, score: np.ndarray) -> tuple[np.ndarray | None, list[dict[str, Any]]]:
    if cv2 is None:
        rows, columns = np.where(mask)
        if len(rows) == 0:
            return None, []
        return None, [
            {
                "label": 1,
                "area": int(mask.sum()),
                "bbox": (int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1),
                "score": float(score[mask].mean()),
                "centroid": (float(columns.mean()), float(rows.mean())),
                "area_fraction": float(mask.mean()),
            }
        ]

    height, width = mask.shape
    image_area = height * width
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    candidates: list[dict[str, Any]] = []
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(10, int(image_area * 0.000012)):
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area_fraction = area / max(image_area, 1)
        width_fraction = component_width / max(width, 1)
        height_fraction = component_height / max(height, 1)
        if y <= int(height * 0.02) or width_fraction > 0.55 or height_fraction > 0.42 or area_fraction > 0.13:
            continue
        component_mask = labels == label
        mean_score = float(score[component_mask].mean())
        centroid_x, centroid_y = centroids[label]
        center_prior = 1.0 - min(1.0, abs((centroid_x / max(width - 1, 1)) - 0.5) + (0.65 * abs((centroid_y / max(height - 1, 1)) - 0.55)))
        area_prior = min(1.0, area_fraction / 0.018)
        shape_prior = min(1.0, (component_width / max(component_height, 1)) / 8.0)
        combined_score = (0.58 * mean_score) + (0.20 * center_prior) + (0.14 * area_prior) + (0.08 * shape_prior)
        candidates.append(
            {
                "label": label,
                "area": area,
                "bbox": (x, y, x + component_width, y + component_height),
                "score": combined_score,
                "mean_score": mean_score,
                "centroid": (float(centroid_x), float(centroid_y)),
                "area_fraction": area_fraction,
            }
        )
    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    return labels, candidates


def _select_anchor(mask: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    labels, candidates = _component_candidates(mask, score)
    if not candidates:
        fallback = score >= float(np.percentile(score, 99.2))
        labels, candidates = _component_candidates(fallback, score)
        if not candidates:
            return fallback, {"body_component_count": 0, "body_candidate_count": 0, "anchor_missing": True}
        mask = fallback

    anchor = candidates[0]
    if labels is None:
        return mask, {"body_component_count": 1, "body_candidate_count": len(candidates), "body_anchor_score": float(anchor["score"])}

    anchor_x0, anchor_y0, anchor_x1, anchor_y1 = anchor["bbox"]
    anchor_cx, anchor_cy = anchor["centroid"]
    height, width = mask.shape
    expansion_x = max(width * 0.09, (anchor_x1 - anchor_x0) * 2.8, 35)
    expansion_y = max(height * 0.055, (anchor_y1 - anchor_y0) * 2.4, 24)
    keep_labels = {anchor["label"]}
    for candidate in candidates[1:8]:
        candidate_cx, candidate_cy = candidate["centroid"]
        close_to_anchor = abs(candidate_cx - anchor_cx) <= expansion_x and abs(candidate_cy - anchor_cy) <= expansion_y
        similar_strength = candidate["score"] >= anchor["score"] * 0.58
        if close_to_anchor and similar_strength:
            keep_labels.add(candidate["label"])
    selected = np.isin(labels, list(keep_labels))
    min_side = min(height, width)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd_kernel(min_side * 0.006), _odd_kernel(min_side * 0.006)))
    selected = cv2.morphologyEx((selected.astype(np.uint8) * 255), cv2.MORPH_CLOSE, close_kernel) > 0
    return selected, {
        "body_component_count": len(keep_labels),
        "body_candidate_count": len(candidates),
        "body_anchor_score": float(anchor["score"]),
        "body_anchor_area_fraction": float(anchor.get("area_fraction", 0.0)),
    }


def _local_growth_mask(image: Image.Image) -> tuple[np.ndarray, dict[str, Any]]:
    scores, metadata = _foreground_scores(image)
    dark_score = scores["dark_anchor"]
    local_score = scores["local_whale"]
    height, width = dark_score.shape
    image_area = height * width

    anchor_threshold = max(float(np.percentile(dark_score, 98.45)), float(dark_score.mean() + 2.05 * dark_score.std()))
    anchor_mask = dark_score >= anchor_threshold
    if anchor_mask.mean() < 0.00008:
        anchor_mask = dark_score >= float(np.percentile(dark_score, 99.15))
    if cv2 is not None:
        min_side = min(height, width)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd_kernel(min_side * 0.0025), _odd_kernel(min_side * 0.0025)))
        anchor_mask = cv2.morphologyEx((anchor_mask.astype(np.uint8) * 255), cv2.MORPH_OPEN, open_kernel) > 0
    anchor_mask, anchor_metadata = _select_anchor(anchor_mask, dark_score)
    rows, columns = np.where(anchor_mask)
    if len(rows) == 0:
        return anchor_mask, {**metadata, **anchor_metadata, "anchor_mask_fraction": 0.0, "mask_fraction": 0.0}

    x0 = int(columns.min())
    x1 = int(columns.max()) + 1
    y0 = int(rows.min())
    y1 = int(rows.max()) + 1
    pad_x = int(max(width * 0.075, (x1 - x0) * 2.6, 24))
    pad_y = int(max(height * 0.055, (y1 - y0) * 2.8, 20))
    region_x0 = max(0, x0 - pad_x)
    region_x1 = min(width, x1 + pad_x)
    region_y0 = max(0, y0 - pad_y)
    region_y1 = min(height, y1 + pad_y)
    region_mask = np.zeros_like(anchor_mask, dtype=bool)
    region_mask[region_y0:region_y1, region_x0:region_x1] = True

    local_values = local_score[region_mask]
    if local_values.size:
        local_threshold = max(float(np.percentile(local_values, 78.5)), float(local_values.mean() + 0.18 * local_values.std()))
        grown = (local_score >= local_threshold) & region_mask
        grown |= anchor_mask
    else:
        grown = anchor_mask

    if cv2 is not None:
        min_side = min(height, width)
        anchor_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd_kernel(min_side * 0.030), _odd_kernel(min_side * 0.030)))
        dilated_anchor = cv2.dilate(anchor_mask.astype(np.uint8), anchor_dilate_kernel, iterations=1) > 0
        component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(grown.astype(np.uint8), connectivity=8)
        keep_labels: set[int] = set()
        anchor_center_x = float(columns.mean())
        anchor_center_y = float(rows.mean())
        for label in range(1, component_count):
            component = labels == label
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < max(8, int(image_area * 0.00001)) or area / max(image_area, 1) > 0.11:
                continue
            centroid_x, centroid_y = centroids[label]
            close_to_anchor = abs(centroid_x - anchor_center_x) <= pad_x * 1.15 and abs(centroid_y - anchor_center_y) <= pad_y * 1.15
            overlaps_anchor = bool((component & dilated_anchor).any())
            strong_component = bool(local_values.size and float(local_score[component].mean()) >= float(local_values.mean()))
            if overlaps_anchor or (close_to_anchor and strong_component):
                keep_labels.add(label)
        if keep_labels:
            grown = np.isin(labels, list(keep_labels))
        grown |= anchor_mask
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd_kernel(min_side * 0.010), _odd_kernel(min_side * 0.010)))
        grown = cv2.morphologyEx((grown.astype(np.uint8) * 255), cv2.MORPH_CLOSE, close_kernel) > 0
    if grown.mean() > 0.13:
        grown = anchor_mask
    return grown, {
        **metadata,
        **anchor_metadata,
        "anchor_threshold": anchor_threshold,
        "anchor_mask_fraction": float(anchor_mask.mean()),
        "mask_fraction": float(grown.mean()),
        "local_region_fraction": float(region_mask.mean()),
        "local_growth_used": True,
    }


def _bbox_from_mask(mask: np.ndarray, image_size: tuple[int, int], padding_fraction: float) -> tuple[int, int, int, int] | None:
    rows, columns = np.where(mask)
    if len(rows) == 0 or len(columns) == 0:
        return None
    mask_height, mask_width = mask.shape
    image_width, image_height = image_size
    x0 = int(columns.min() * image_width / mask_width)
    x1 = int((columns.max() + 1) * image_width / mask_width)
    y0 = int(rows.min() * image_height / mask_height)
    y1 = int((rows.max() + 1) * image_height / mask_height)
    padding = int(max(x1 - x0, y1 - y0) * padding_fraction)
    return (max(0, x0 - padding), max(0, y0 - padding), min(image_width, x1 + padding), min(image_height, y1 + padding))


def propose_whale_crop(image_path: str | Path, padding_fraction: float = 0.18) -> dict[str, Any]:
    """Return a local-growth whale mask, padded crop box, and quality metadata."""
    image = _load_rgb(image_path)
    working_image = image.copy()
    scale = 1.0
    max_side = 1250
    if max(working_image.size) > max_side:
        scale = max_side / max(working_image.size)
        working_image = working_image.resize((max(1, int(working_image.width * scale)), max(1, int(working_image.height * scale))), Image.Resampling.LANCZOS)
    working_mask, metadata = _local_growth_mask(working_image)
    if scale != 1.0:
        mask_image = Image.fromarray((working_mask.astype(np.uint8) * 255), mode="L").resize(image.size, Image.Resampling.NEAREST)
        mask = np.asarray(mask_image) > 0
    else:
        mask = working_mask
    bbox = _bbox_from_mask(mask, image.size, padding_fraction)
    if bbox is None:
        width, height = image.size
        bbox = (0, 0, width, height)
        quality_score = 0.0
    else:
        x0, y0, x1, y1 = bbox
        image_area = max(image.size[0] * image.size[1], 1)
        bbox_fraction = ((x1 - x0) * (y1 - y0)) / image_area
        mask_fraction = float(mask.mean())
        crop_foreground_fraction = mask_fraction / max(bbox_fraction, 1e-6)
        rows, columns = np.where(mask)
        center_x = float(columns.mean() / max(mask.shape[1] - 1, 1)) if len(columns) else 0.5
        center_y = float(rows.mean() / max(mask.shape[0] - 1, 1)) if len(rows) else 0.5
        centeredness = 1.0 - min(1.0, abs(center_x - 0.5) + (0.75 * abs(center_y - 0.55)))
        size_score = min(1.0, mask_fraction / 0.020)
        density_score = 1.0 - min(1.0, abs(crop_foreground_fraction - 0.24) / 0.24)
        quality_score = float(np.clip((0.35 * size_score) + (0.32 * density_score) + (0.22 * centeredness) + (0.11 * min(1.0, bbox_fraction / 0.16)), 0.0, 1.0))
    return {"image": image, "mask": mask, "bbox": bbox, "crop_quality_score": quality_score, "isolation_method": ISOLATION_METHOD, **metadata}


def _masked_crop(image: Image.Image, mask: np.ndarray, bbox: tuple[int, int, int, int]) -> Image.Image:
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(image.size, Image.Resampling.NEAREST)
    rgb_array = np.asarray(image, dtype=np.uint8)
    mask_array = np.asarray(mask_image, dtype=np.uint8) > 0
    background_color = np.median(rgb_array[~mask_array], axis=0).astype(np.uint8) if (~mask_array).any() else np.array([0, 0, 0], dtype=np.uint8)
    masked_array = np.empty_like(rgb_array)
    masked_array[:, :] = background_color
    masked_array[mask_array] = rgb_array[mask_array]
    return Image.fromarray(masked_array, mode="RGB").crop(bbox)


def create_whale_crop(
    image_path: str | Path,
    crop_path: str | Path,
    masked_crop_path: str | Path,
    mask_path: str | Path,
    metadata_path: str | Path | None = None,
    padding_fraction: float = 0.18,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create review crop, local-growth mask, masked crop, and metadata."""
    crop_output = Path(crop_path)
    masked_crop_output = Path(masked_crop_path)
    mask_output = Path(mask_path)
    metadata_output = Path(metadata_path) if metadata_path else None
    if crop_output.exists() and mask_output.exists() and masked_crop_output.exists() and not overwrite:
        if metadata_output and metadata_output.exists():
            metadata = json.loads(metadata_output.read_text(encoding="utf-8"))
            metadata["crop_path"] = str(crop_output)
            metadata["masked_crop_path"] = str(masked_crop_output)
            metadata["mask_path"] = str(mask_output)
            metadata["crop_status"] = "ready"
            return metadata
        return {"crop_path": str(crop_output), "masked_crop_path": str(masked_crop_output), "mask_path": str(mask_output), "crop_status": "ready"}
    proposal = propose_whale_crop(image_path, padding_fraction=padding_fraction)
    image: Image.Image = proposal["image"]
    bbox = tuple(int(value) for value in proposal["bbox"])
    crop_output.parent.mkdir(parents=True, exist_ok=True)
    masked_crop_output.parent.mkdir(parents=True, exist_ok=True)
    mask_output.parent.mkdir(parents=True, exist_ok=True)
    image.crop(bbox).save(crop_output, quality=92)
    _masked_crop(image, proposal["mask"], bbox).save(masked_crop_output, quality=92)
    resized_mask = Image.fromarray((proposal["mask"].astype(np.uint8) * 255), mode="L").resize(image.size, Image.Resampling.NEAREST)
    resized_mask.save(mask_output)
    x0, y0, x1, y1 = bbox
    image_area = max(image.size[0] * image.size[1], 1)
    bbox_fraction = ((x1 - x0) * (y1 - y0)) / image_area
    mask_fraction = float(proposal["mask"].mean())
    metadata = {
        "crop_path": str(crop_output),
        "masked_crop_path": str(masked_crop_output),
        "mask_path": str(mask_output),
        "crop_status": "ready",
        "isolation_method": proposal.get("isolation_method", ISOLATION_METHOD),
        "whale_bbox_x": x0,
        "whale_bbox_y": y0,
        "whale_bbox_width": x1 - x0,
        "whale_bbox_height": y1 - y0,
        "crop_quality_score": proposal["crop_quality_score"],
        "mask_fraction": mask_fraction,
        "bbox_fraction": float(bbox_fraction),
        "crop_foreground_fraction": float(mask_fraction / max(bbox_fraction, 1e-6)),
    }
    for key in ("estimated_waterline_y", "anchor_threshold", "anchor_mask_fraction", "local_region_fraction", "local_growth_used", "body_component_count", "body_candidate_count", "body_anchor_score", "body_anchor_area_fraction", "water_gray", "water_rgb"):
        if key in proposal:
            metadata[key] = proposal[key]
    if metadata_output:
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def ensure_whale_crops(
    records: list[dict[str, Any]],
    crop_dir: str | Path,
    mask_dir: str | Path,
    image_key: str = "preview_path",
    padding_fraction: float = 0.18,
    overwrite: bool = False,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach local-growth whale masks, review crops, and masked crops."""
    updated_records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"ready": 0, "failed": 0, "skipped_by_limit": 0, "errors": [], "method": ISOLATION_METHOD}
    processed_count = 0
    for record in records:
        updated_record = dict(record)
        if limit is not None and processed_count >= limit:
            updated_record["crop_status"] = "not_requested"
            summary["skipped_by_limit"] += 1
            updated_records.append(updated_record)
            continue
        processed_count += 1
        source_value = record.get(image_key) or record.get("preview_path") or record.get("image_path")
        source_path = Path(str(source_value)) if source_value else None
        if not source_path or not source_path.exists():
            updated_record["crop_status"] = "source_missing"
            summary["failed"] += 1
            updated_records.append(updated_record)
            continue
        crop_path, masked_crop_path, mask_path, metadata_path = isolation_paths_for_image(source_path, crop_dir, mask_dir)
        try:
            metadata = create_whale_crop(source_path, crop_path, masked_crop_path, mask_path, metadata_path=metadata_path, padding_fraction=padding_fraction, overwrite=overwrite)
            updated_record.update(metadata)
            summary["ready"] += 1
        except Exception as exc:
            updated_record["crop_status"] = f"failed: {exc}"
            summary["failed"] += 1
            if len(summary["errors"]) < 12:
                summary["errors"].append({"image_path": str(source_path), "error": str(exc)})
        updated_records.append(updated_record)
    return updated_records, summary
