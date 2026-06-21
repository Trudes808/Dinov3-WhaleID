"""Series representation, pair scoring, and day-level clustering helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

import numpy as np


def cosine_similarity(left_vector: Any, right_vector: Any) -> float | None:
    """Return cosine similarity for two vectors, or None when unavailable."""
    if left_vector is None or right_vector is None:
        return None
    left_array = np.asarray(left_vector, dtype=np.float32)
    right_array = np.asarray(right_vector, dtype=np.float32)
    left_norm = float(np.linalg.norm(left_array))
    right_norm = float(np.linalg.norm(right_array))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return float(np.dot(left_array, right_array) / (left_norm * right_norm))


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def build_series_representations(
    records: list[dict[str, Any]],
    series_key: str = "validated_series_id",
    feature_key: str = "feature_vector",
) -> list[dict[str, Any]]:
    """Aggregate per-image records into one representation per series."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get(series_key) or "unknown_series")].append(record)

    series_records: list[dict[str, Any]] = []
    for series_id, image_records in sorted(grouped.items()):
        feature_vectors = [np.asarray(record[feature_key], dtype=np.float32) for record in image_records if record.get(feature_key) is not None]
        aggregate_vector = _normalize(np.mean(feature_vectors, axis=0)) if feature_vectors else None
        truth_counts = Counter(record.get("ground_truth_whale_id") for record in image_records if record.get("ground_truth_whale_id"))
        quality_scores = [float(record.get("crop_quality_score") or 0.0) for record in image_records]
        series_records.append(
            {
                "series_id": series_id,
                "day_label": image_records[0].get("day_label"),
                "camera_id": image_records[0].get("camera_id"),
                "photographer": image_records[0].get("photographer"),
                "image_count": len(image_records),
                "feature_count": len(feature_vectors),
                "image_ids": [record.get("image_id") for record in image_records],
                "filenames": [record.get("filename") for record in image_records],
                "representative_image_path": _best_representative_path(image_records),
                "mean_crop_quality_score": float(np.mean(quality_scores)) if quality_scores else None,
                "dominant_ground_truth_whale_id": truth_counts.most_common(1)[0][0] if truth_counts else None,
                "ground_truth_whale_ids": "|".join(sorted(str(label) for label in truth_counts)),
                "series_feature_vector": aggregate_vector.astype(float).tolist() if aggregate_vector is not None else None,
            }
        )
    return series_records


def _best_representative_path(records: list[dict[str, Any]]) -> str | None:
    sorted_records = sorted(records, key=lambda record: float(record.get("crop_quality_score") or 0.0), reverse=True)
    for record in sorted_records:
        for key in ("crop_path", "preview_path", "image_path"):
            if record.get(key):
                return str(record[key])
    return None


def score_series_pairs(
    series_records: list[dict[str, Any]],
    same_day_only: bool = True,
) -> list[dict[str, Any]]:
    """Compute pairwise series similarities."""
    pair_scores: list[dict[str, Any]] = []
    for left, right in combinations(series_records, 2):
        if same_day_only and left.get("day_label") != right.get("day_label"):
            continue
        similarity = cosine_similarity(left.get("series_feature_vector"), right.get("series_feature_vector"))
        if similarity is None:
            continue
        quality_values = [value for value in (left.get("mean_crop_quality_score"), right.get("mean_crop_quality_score")) if value is not None]
        quality_weight = float(np.mean(quality_values)) if quality_values else 1.0
        pair_scores.append(
            {
                "day_label": left.get("day_label"),
                "left_series_id": left["series_id"],
                "right_series_id": right["series_id"],
                "similarity": similarity,
                "quality_weight": quality_weight,
                "weighted_score": similarity * quality_weight,
                "left_image_count": left.get("image_count"),
                "right_image_count": right.get("image_count"),
            }
        )
    return sorted(pair_scores, key=lambda record: (str(record.get("day_label")), -float(record.get("weighted_score") or 0.0)))


class _UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def cluster_series_by_threshold(
    series_records: list[dict[str, Any]],
    pair_scores: list[dict[str, Any]],
    threshold: float = 0.82,
    score_key: str = "similarity",
) -> list[dict[str, Any]]:
    """Cluster series into day-level connected components."""
    series_by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for series_record in series_records:
        series_by_day[str(series_record.get("day_label") or "unknown_day")].append(series_record)

    pair_scores_by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair_score in pair_scores:
        pair_scores_by_day[str(pair_score.get("day_label") or "unknown_day")].append(pair_score)

    clustered: list[dict[str, Any]] = []
    for day_label, day_series in sorted(series_by_day.items()):
        series_ids = [str(series_record["series_id"]) for series_record in day_series]
        union_find = _UnionFind(series_ids)
        for pair_score in pair_scores_by_day.get(day_label, []):
            score = pair_score.get(score_key)
            if score is not None and float(score) >= threshold:
                union_find.union(str(pair_score["left_series_id"]), str(pair_score["right_series_id"]))

        root_to_cluster: dict[str, str] = {}
        next_index = 1
        for series_record in sorted(day_series, key=lambda record: str(record["series_id"])):
            root = union_find.find(str(series_record["series_id"]))
            if root not in root_to_cluster:
                root_to_cluster[root] = f"{day_label}_W{next_index:03d}"
                next_index += 1
            updated = dict(series_record)
            updated["predicted_whale_cluster_id"] = root_to_cluster[root]
            updated["cluster_threshold"] = threshold
            clustered.append(updated)
    return clustered


def assign_clusters_to_records(
    records: list[dict[str, Any]],
    clustered_series: list[dict[str, Any]],
    series_key: str = "validated_series_id",
) -> list[dict[str, Any]]:
    """Attach predicted whale cluster IDs to image-level records."""
    cluster_by_series = {
        str(series_record["series_id"]): series_record.get("predicted_whale_cluster_id") for series_record in clustered_series
    }
    updated_records: list[dict[str, Any]] = []
    for record in records:
        updated = dict(record)
        updated["predicted_whale_cluster_id"] = cluster_by_series.get(str(record.get(series_key)))
        updated_records.append(updated)
    return updated_records


def summarize_clusters(clustered_series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one compact summary row per predicted cluster."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for series_record in clustered_series:
        grouped[str(series_record.get("predicted_whale_cluster_id") or "unknown_cluster")].append(series_record)

    summaries: list[dict[str, Any]] = []
    for cluster_id, series_group in sorted(grouped.items()):
        truth_counts = Counter(record.get("dominant_ground_truth_whale_id") for record in series_group if record.get("dominant_ground_truth_whale_id"))
        summaries.append(
            {
                "predicted_whale_cluster_id": cluster_id,
                "day_label": series_group[0].get("day_label"),
                "series_count": len(series_group),
                "image_count": sum(int(record.get("image_count") or 0) for record in series_group),
                "dominant_ground_truth_whale_id": truth_counts.most_common(1)[0][0] if truth_counts else None,
                "ground_truth_whale_ids": "|".join(sorted(str(label) for label in truth_counts)),
                "series_ids": "|".join(str(record.get("series_id")) for record in series_group),
            }
        )
    return summaries