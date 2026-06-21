"""Plotting helpers for whale clustering review notebooks."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


def _import_matplotlib():
    import matplotlib.pyplot as plt  # type: ignore

    return plt


def _records_from(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        if "records" in value:
            return value["records"]
        if "series" in value:
            return value["series"]
    return value or []


def _image_path(record: dict[str, Any]) -> str | None:
    for key in ("representative_image_path", "crop_path", "preview_path", "image_path"):
        if record.get(key) and Path(str(record[key])).exists():
            return str(record[key])
    return None


def plot_manifest_summary(manifest: dict[str, Any] | list[dict[str, Any]]):
    """Plot image counts by day and Lightroom match status."""
    plt = _import_matplotlib()
    records = _records_from(manifest)
    day_counts = Counter(record.get("day_label") or "unknown" for record in records)
    matched = sum(bool(record.get("lightroom_matched")) for record in records)
    unmatched = len(records) - matched

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    day_labels = [label for label, _ in day_counts.most_common(12)]
    axes[0].bar(day_labels, [day_counts[label] for label in day_labels], color="#2f6f73")
    axes[0].set_title("Images by day")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Images")
    axes[1].bar(["matched", "unmatched"], [matched, unmatched], color=["#4c8c4a", "#a64d4d"])
    axes[1].set_title("Lightroom join")
    axes[1].set_ylabel("Images")
    figure.tight_layout()
    return figure


def plot_series_grid(series: dict[str, Any] | list[dict[str, Any]], max_series: int = 12, thumb_size: int = 160):
    """Show one representative image per series."""
    plt = _import_matplotlib()
    records = _records_from(series)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        series_id = str(record.get("validated_series_id") or record.get("series_id") or "unknown_series")
        grouped[series_id].append(record)

    selected = list(grouped.items())[:max_series]
    if not selected:
        figure = plt.figure(figsize=(4, 2))
        plt.text(0.5, 0.5, "No series to plot", ha="center", va="center")
        plt.axis("off")
        return figure

    columns = min(4, len(selected))
    rows = (len(selected) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 3.2, rows * 3.0))
    axes_list = list(getattr(axes, "flat", [axes]))
    for axis, (series_id, image_records) in zip(axes_list, selected):
        best_record = sorted(image_records, key=lambda record: float(record.get("crop_quality_score") or 0.0), reverse=True)[0]
        path = _image_path(best_record)
        if path:
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            image.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            axis.imshow(image)
        axis.set_title(f"{series_id}\n{len(image_records)} images", fontsize=8)
        axis.axis("off")
    for axis in axes_list[len(selected) :]:
        axis.axis("off")
    figure.tight_layout()
    return figure


def plot_image_grid(
    records_or_manifest: dict[str, Any] | list[dict[str, Any]],
    max_images: int = 12,
    thumb_size: int = 160,
):
    """Show a simple thumbnail grid from manifest or artifact records."""
    plt = _import_matplotlib()
    records = _records_from(records_or_manifest)[:max_images]
    if not records:
        figure = plt.figure(figsize=(4, 2))
        plt.text(0.5, 0.5, "No images to plot", ha="center", va="center")
        plt.axis("off")
        return figure

    columns = min(4, len(records))
    rows = (len(records) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 3.0, rows * 3.0))
    axes_list = list(getattr(axes, "flat", [axes]))
    for axis, record in zip(axes_list, records):
        path = _image_path(record)
        if path:
            try:
                image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
                image.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                axis.imshow(image)
            except Exception:
                axis.text(0.5, 0.5, "preview unavailable", ha="center", va="center", fontsize=8)
        else:
            axis.text(0.5, 0.5, "missing image", ha="center", va="center", fontsize=8)
        axis.set_title(str(record.get("filename") or record.get("series_id") or "image")[:36], fontsize=8)
        axis.axis("off")
    for axis in axes_list[len(records) :]:
        axis.axis("off")
    figure.tight_layout()
    return figure


def plot_cluster_review(
    clusters: dict[str, Any] | list[dict[str, Any]],
    manifest: dict[str, Any] | list[dict[str, Any]] | None = None,
    ground_truth: dict[str, Any] | list[dict[str, Any]] | None = None,
    max_clusters: int = 8,
    max_images_per_cluster: int = 6,
    thumb_size: int = 150,
):
    """Show image thumbnails grouped by predicted whale cluster."""
    plt = _import_matplotlib()
    records = _records_from(clusters)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("predicted_whale_cluster_id") or "unclustered")].append(record)
    selected = list(grouped.items())[:max_clusters]
    if not selected:
        figure = plt.figure(figsize=(4, 2))
        plt.text(0.5, 0.5, "No clusters to plot", ha="center", va="center")
        plt.axis("off")
        return figure

    rows = len(selected)
    columns = max_images_per_cluster
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2.2, rows * 2.4))
    if rows == 1:
        axes = [axes]
    for row_index, (cluster_id, cluster_records) in enumerate(selected):
        row_axes = axes[row_index]
        sorted_records = sorted(cluster_records, key=lambda record: float(record.get("crop_quality_score") or 0.0), reverse=True)
        truth_labels = sorted({str(record.get("ground_truth_whale_id")) for record in cluster_records if record.get("ground_truth_whale_id")})
        for column_index in range(columns):
            axis = row_axes[column_index]
            if column_index < len(sorted_records):
                path = _image_path(sorted_records[column_index])
                if path:
                    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
                    image.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                    axis.imshow(image)
                if column_index == 0:
                    axis.set_ylabel(f"{cluster_id}\n{', '.join(truth_labels[:3])}", fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.tight_layout()
    return figure