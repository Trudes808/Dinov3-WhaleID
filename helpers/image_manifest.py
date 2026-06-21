"""Build and join image manifests for whale clustering experiments."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from .lrcat_parser import normalize_date


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".cr2",
    ".cr3",
    ".dng",
    ".nef",
    ".arw",
    ".orf",
    ".rw2",
}

CAMERA_RE = re.compile(r"\bC\d+\b", re.IGNORECASE)
SEQUENCE_RE = re.compile(r"-(?P<sequence>\d{1,4})-[^-]+$")
DATE_TEXT_RE = re.compile(r"20\d{2}[-_]\d{1,2}[-_]\d{1,2}")


def iter_image_paths(image_root: str | Path) -> list[Path]:
    """Return supported image paths below the data root."""
    root = Path(image_root).expanduser()
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def parse_sequence_number(filename_stem: str) -> int | None:
    """Parse the field sequence number from the common NOAA filename pattern."""
    match = SEQUENCE_RE.search(filename_stem)
    if not match:
        return None
    try:
        return int(match.group("sequence"))
    except ValueError:
        return None


def parse_camera_id(*text_values: str | None) -> str | None:
    """Return C# camera ID from folder or filename text."""
    for text_value in text_values:
        if not text_value:
            continue
        match = CAMERA_RE.search(text_value)
        if match:
            return match.group(0).upper()
    return None


def parse_photographer(folder_name: str, filename_stem: str) -> str | None:
    """Best-effort photographer parser from folder/filename conventions."""
    cleaned_folder = DATE_TEXT_RE.sub("", folder_name)
    cleaned_folder = CAMERA_RE.sub("", cleaned_folder)
    cleaned_folder = re.sub(r"\bR\d+\b", "", cleaned_folder, flags=re.IGNORECASE)
    cleaned_folder = re.sub(r"[_,-]+", " ", cleaned_folder)
    tokens = [token for token in cleaned_folder.split() if token.lower() not in {"trip", "uh"}]
    if tokens:
        return " ".join(tokens).strip() or None

    filename_parts = filename_stem.split("-")
    if len(filename_parts) >= 4:
        candidate = filename_parts[3].replace("_", " ").strip()
        if candidate and not candidate.upper().startswith("NOAA"):
            return candidate
    return None


def parse_image_record(path: Path, image_root: Path) -> dict[str, Any]:
    """Create a normalized manifest record for one image file."""
    resolved_path = path.resolve()
    relative_path = path.relative_to(image_root)
    parent_folder = path.parent.name
    path_text = str(relative_path)
    day_label = normalize_date(path_text) or normalize_date(path.stem)
    camera_id = parse_camera_id(parent_folder, path.stem)
    return {
        "image_id": str(resolved_path),
        "image_path": str(resolved_path),
        "relative_path": str(relative_path),
        "folder_relative_path": str(relative_path.parent),
        "filename": path.name,
        "image_stem": path.stem,
        "extension": path.suffix.lower().lstrip("."),
        "day_label": day_label,
        "camera_id": camera_id,
        "photographer": parse_photographer(parent_folder, path.stem),
        "sequence_number": parse_sequence_number(path.stem),
        "source_exists": path.exists(),
        "is_raw": path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"},
    }


def _ground_truth_indexes(ground_truth_records: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[tuple[str | None, str], dict[str, Any]], dict[tuple[str | None, str], dict[str, Any]]]:
    by_path: dict[str, dict[str, Any]] = {}
    by_date_filename: dict[tuple[str | None, str], dict[str, Any]] = {}
    by_date_stem: dict[tuple[str | None, str], dict[str, Any]] = {}
    for record in ground_truth_records:
        image_path = record.get("image_path")
        if image_path:
            by_path[str(Path(image_path).expanduser().resolve()).casefold()] = record
        filename = str(record.get("filename") or "").casefold()
        image_stem = str(record.get("image_stem") or "").casefold()
        day_label = record.get("day_label") or record.get("capture_date")
        if filename:
            by_date_filename[(day_label, filename)] = record
        if image_stem:
            by_date_stem[(day_label, image_stem)] = record
    return by_path, by_date_filename, by_date_stem


def join_ground_truth(record: dict[str, Any], ground_truth_record: dict[str, Any] | None) -> dict[str, Any]:
    """Attach Lightroom label fields to one manifest record."""
    joined_record = dict(record)
    joined_record["lightroom_matched"] = bool(ground_truth_record)
    if not ground_truth_record:
        joined_record.update(
            {
                "ground_truth_whale_id": None,
                "ground_truth_whale_ids": None,
                "is_scarring_study_green": False,
                "is_other_study_yellow": False,
                "color_label": None,
            }
        )
        return joined_record

    for key, value in ground_truth_record.items():
        if key in {"image_path", "filename", "image_stem", "extension", "day_label"}:
            joined_record[f"lightroom_{key}"] = value
        else:
            joined_record[key] = value
    return joined_record


def build_image_manifest(
    image_root: str | Path,
    ground_truth_records: list[dict[str, Any]] | None = None,
    target_day: str | None = None,
    max_images: int | None = None,
) -> list[dict[str, Any]]:
    """Scan the image tree and join Lightroom labels when available."""
    root = Path(image_root).expanduser().resolve()
    normalized_target_day = normalize_date(target_day) if target_day else None
    by_path, by_date_filename, by_date_stem = _ground_truth_indexes(ground_truth_records or [])
    records: list[dict[str, Any]] = []
    for image_path in iter_image_paths(root):
        record = parse_image_record(image_path, root)
        if normalized_target_day and record.get("day_label") != normalized_target_day:
            continue
        path_key = str(Path(record["image_path"]).resolve()).casefold()
        date_filename_key = (record.get("day_label"), str(record.get("filename") or "").casefold())
        date_stem_key = (record.get("day_label"), str(record.get("image_stem") or "").casefold())
        ground_truth_record = by_path.get(path_key) or by_date_filename.get(date_filename_key) or by_date_stem.get(
            date_stem_key
        )
        records.append(join_ground_truth(record, ground_truth_record))
        if max_images and len(records) >= max_images:
            break
    return records


def summarize_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return counts for quick notebook printouts."""
    day_counts = Counter(record.get("day_label") or "unknown" for record in records)
    extension_counts = Counter(record.get("extension") or "unknown" for record in records)
    camera_counts = Counter(record.get("camera_id") or "unknown" for record in records)
    return {
        "images": len(records),
        "days": len(day_counts),
        "lightroom_matched": sum(bool(record.get("lightroom_matched")) for record in records),
        "labeled_images": sum(bool(record.get("ground_truth_whale_id")) for record in records),
        "raw_images": sum(bool(record.get("is_raw")) for record in records),
        "green_scarring_images": sum(bool(record.get("is_scarring_study_green")) for record in records),
        "yellow_study_images": sum(bool(record.get("is_other_study_yellow")) for record in records),
        "top_days": day_counts.most_common(12),
        "extensions": extension_counts.most_common(),
        "cameras": camera_counts.most_common(),
    }


def available_days(records: list[dict[str, Any]]) -> list[tuple[str, int, int]]:
    """Return day, image count, labeled count sorted by day."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        day_label = record.get("day_label") or "unknown"
        grouped.setdefault(day_label, []).append(record)
    return [
        (
            day_label,
            len(day_records),
            sum(bool(record.get("ground_truth_whale_id")) for record in day_records),
        )
        for day_label, day_records in sorted(grouped.items())
    ]
