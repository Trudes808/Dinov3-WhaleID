"""Create cached RGB previews for RAW/JPEG images."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


DISPLAY_EXTENSIONS = {"jpg", "jpeg", "png", "tif", "tiff"}


class PreviewError(RuntimeError):
    """Raised when a preview cannot be created for an image."""


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80]


def preview_path_for_image(image_path: str | Path, preview_dir: str | Path) -> Path:
    """Return deterministic preview path for an image."""
    source_path = Path(image_path)
    digest = hashlib.sha1(str(source_path.resolve()).encode("utf-8")).hexdigest()[:14]
    return Path(preview_dir) / f"{digest}_{_safe_stem(source_path.stem)}.jpg"


def _open_with_rawpy(image_path: Path) -> Image.Image:
    try:
        import rawpy  # type: ignore
    except ImportError as exc:
        raise PreviewError("rawpy is required to create previews for RAW files") from exc
    with rawpy.imread(str(image_path)) as raw_image:
        rgb_array = raw_image.postprocess(use_camera_wb=True, output_bps=8)
    return Image.fromarray(rgb_array)


def create_preview(
    image_path: str | Path,
    output_path: str | Path,
    max_size: int = 1600,
    overwrite: bool = False,
) -> Path:
    """Create one RGB JPEG preview and return its path."""
    source_path = Path(image_path)
    preview_path = Path(output_path)
    if preview_path.exists() and not overwrite:
        return preview_path
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    extension = source_path.suffix.lower().lstrip(".")
    if extension in DISPLAY_EXTENSIONS:
        image = ImageOps.exif_transpose(Image.open(source_path)).convert("RGB")
    else:
        try:
            image = ImageOps.exif_transpose(Image.open(source_path)).convert("RGB")
        except Exception:
            image = _open_with_rawpy(source_path).convert("RGB")

    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    image.save(preview_path, quality=92)
    return preview_path


def ensure_preview_cache(
    records: list[dict[str, Any]],
    preview_dir: str | Path,
    max_size: int = 1600,
    overwrite: bool = False,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach preview paths/status to manifest records."""
    updated_records: list[dict[str, Any]] = []
    summary = {"created_or_existing": 0, "failed": 0, "skipped_by_limit": 0, "errors": []}
    processed_count = 0
    for record in records:
        updated_record = dict(record)
        if limit is not None and processed_count >= limit:
            updated_record["preview_status"] = "not_requested"
            summary["skipped_by_limit"] += 1
            updated_records.append(updated_record)
            continue
        processed_count += 1
        source_path = Path(str(record.get("image_path")))
        output_path = preview_path_for_image(source_path, preview_dir)
        updated_record["preview_path"] = str(output_path)
        if not source_path.exists():
            updated_record["preview_status"] = "source_missing"
            summary["failed"] += 1
            updated_records.append(updated_record)
            continue
        try:
            create_preview(source_path, output_path, max_size=max_size, overwrite=overwrite)
            updated_record["preview_status"] = "ready"
            summary["created_or_existing"] += 1
        except Exception as exc:
            updated_record["preview_status"] = f"failed: {exc}"
            summary["failed"] += 1
            if len(summary["errors"]) < 12:
                summary["errors"].append({"image_path": str(source_path), "error": str(exc)})
        updated_records.append(updated_record)
    return updated_records, summary
