"""Detect and summarize photo series within whale field days."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

import numpy as np


def parse_capture_time(value: str | None) -> datetime | None:
    """Parse Lightroom ISO-ish capture timestamps."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _stream_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record.get("day_label") or "unknown_day"),
        str(record.get("camera_id") or "unknown_camera"),
        str(record.get("photographer") or "unknown_photographer"),
        str(record.get("folder_relative_path") or "unknown_folder"),
    )


def _series_prefix(stream_key: tuple[str, str, str, str]) -> str:
    day_label, camera_id, photographer, folder_name = stream_key
    digest = hashlib.sha1("|".join(stream_key).encode("utf-8")).hexdigest()[:6]
    clean_camera = camera_id.replace(" ", "_")
    clean_photographer = "_".join(photographer.split())[:24]
    return f"{day_label}_{clean_camera}_{clean_photographer}_{digest}"


def _sort_key(record: dict[str, Any]) -> tuple[str, int, str]:
    capture_time = parse_capture_time(record.get("capture_time"))
    capture_sort = capture_time.isoformat() if capture_time else "9999"
    sequence_number = record.get("sequence_number")
    sequence_sort = int(sequence_number) if sequence_number is not None else 10**9
    return capture_sort, sequence_sort, str(record.get("filename") or "")


def detect_provisional_series(
    records: list[dict[str, Any]],
    sequence_gap: int = 5,
    time_gap_seconds: int = 300,
) -> list[dict[str, Any]]:
    """Group images into provisional series using metadata continuity."""
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_stream_key(record)].append(record)

    updated_records: list[dict[str, Any]] = []
    for stream_key, stream_records in sorted(grouped.items()):
        sorted_records = sorted(stream_records, key=_sort_key)
        series_index = 1
        previous_record: dict[str, Any] | None = None
        prefix = _series_prefix(stream_key)
        for record in sorted_records:
            break_reasons: list[str] = []
            if previous_record is not None:
                current_sequence = record.get("sequence_number")
                previous_sequence = previous_record.get("sequence_number")
                if current_sequence is not None and previous_sequence is not None:
                    sequence_delta = int(current_sequence) - int(previous_sequence)
                    if sequence_delta > sequence_gap:
                        break_reasons.append(f"sequence_gap_{sequence_delta}")
                    elif sequence_delta <= 0:
                        break_reasons.append("non_monotonic_sequence")
                current_time = parse_capture_time(record.get("capture_time"))
                previous_time = parse_capture_time(previous_record.get("capture_time"))
                if current_time and previous_time:
                    time_delta_seconds = (current_time - previous_time).total_seconds()
                    if time_delta_seconds > time_gap_seconds:
                        break_reasons.append(f"time_gap_{int(time_delta_seconds)}s")
            if previous_record is not None and break_reasons:
                series_index += 1
            updated_record = dict(record)
            updated_record["provisional_series_id"] = f"{prefix}_S{series_index:04d}"
            updated_record["validated_series_id"] = updated_record["provisional_series_id"]
            updated_record["series_break_reason"] = ";".join(break_reasons) if break_reasons else "continuation"
            updated_records.append(updated_record)
            previous_record = record
    return sorted(updated_records, key=lambda record: str(record.get("image_path") or ""))


def _cosine_similarity(left_vector: Any, right_vector: Any) -> float | None:
    if left_vector is None or right_vector is None:
        return None
    left_array = np.asarray(left_vector, dtype=np.float32)
    right_array = np.asarray(right_vector, dtype=np.float32)
    left_norm = float(np.linalg.norm(left_array))
    right_norm = float(np.linalg.norm(right_array))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return float(np.dot(left_array, right_array) / (left_norm * right_norm))


def validate_series_by_adjacent_similarity(
    records: list[dict[str, Any]],
    feature_key: str = "feature_vector",
    min_similarity: float = 0.70,
) -> list[dict[str, Any]]:
    """Split provisional series when adjacent visual features change abruptly."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("provisional_series_id") or "unknown_series")].append(record)

    updated_records: list[dict[str, Any]] = []
    for provisional_series_id, series_records in sorted(grouped.items()):
        sorted_records = sorted(series_records, key=_sort_key)
        split_index = 1
        previous_record: dict[str, Any] | None = None
        for record in sorted_records:
            similarity = None
            split_reason = "metadata_series"
            if previous_record is not None:
                similarity = _cosine_similarity(previous_record.get(feature_key), record.get(feature_key))
                if similarity is not None and similarity < min_similarity:
                    split_index += 1
                    split_reason = f"visual_split_similarity_{similarity:.3f}"
                elif similarity is not None:
                    split_reason = f"visual_continuity_{similarity:.3f}"
                else:
                    split_reason = "visual_similarity_unavailable"
            updated_record = dict(record)
            updated_record["adjacent_similarity"] = similarity
            updated_record["validated_series_id"] = f"{provisional_series_id}_V{split_index:02d}"
            updated_record["visual_boundary_reason"] = split_reason
            updated_records.append(updated_record)
            previous_record = record
    return sorted(updated_records, key=lambda record: str(record.get("image_path") or ""))


def summarize_series(records: list[dict[str, Any]], series_key: str = "validated_series_id") -> list[dict[str, Any]]:
    """Return one summary record per series."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get(series_key) or "unknown_series")].append(record)

    summaries: list[dict[str, Any]] = []
    for series_id, series_records in sorted(grouped.items()):
        sorted_records = sorted(series_records, key=_sort_key)
        truth_counts = Counter(
            record.get("ground_truth_whale_id") for record in sorted_records if record.get("ground_truth_whale_id")
        )
        summaries.append(
            {
                "series_id": series_id,
                "image_count": len(sorted_records),
                "day_label": sorted_records[0].get("day_label"),
                "camera_id": sorted_records[0].get("camera_id"),
                "photographer": sorted_records[0].get("photographer"),
                "first_filename": sorted_records[0].get("filename"),
                "last_filename": sorted_records[-1].get("filename"),
                "first_sequence": sorted_records[0].get("sequence_number"),
                "last_sequence": sorted_records[-1].get("sequence_number"),
                "first_capture_time": sorted_records[0].get("capture_time"),
                "last_capture_time": sorted_records[-1].get("capture_time"),
                "labeled_images": sum(bool(record.get("ground_truth_whale_id")) for record in sorted_records),
                "dominant_ground_truth_whale_id": truth_counts.most_common(1)[0][0] if truth_counts else None,
                "ground_truth_whale_ids": "|".join(sorted(str(label) for label in truth_counts)),
                "preview_ready_images": sum(record.get("preview_status") == "ready" for record in sorted_records),
                "crop_ready_images": sum(record.get("crop_status") == "ready" for record in sorted_records),
                "feature_ready_images": sum(record.get("feature_status") == "ready" for record in sorted_records),
            }
        )
    return summaries
