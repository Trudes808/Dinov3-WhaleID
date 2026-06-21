"""Notebook-facing orchestration helpers for the whale clustering pipeline."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers.clustering import assign_clusters_to_records, build_series_representations, cluster_series_by_threshold
from helpers.clustering import score_series_pairs, summarize_clusters
from helpers.evaluation import evaluate_records
from helpers.feature_extraction import ensure_feature_cache
from helpers.image_manifest import available_days, build_image_manifest, summarize_manifest
from helpers.lrcat_parser import collapse_ground_truth_by_image, inspect_schema, parse_lrcat_ground_truth, summarize_ground_truth
from helpers.raw_previews import ensure_preview_cache
from helpers.series_detection import detect_provisional_series, summarize_series, validate_series_by_adjacent_similarity
from helpers.whale_isolation import ensure_whale_crops


@dataclass
class WhaleClusteringConfig:
    """Configuration for one whale clustering run."""

    image_root: str = "/home/sat3737/Test/Lightroom images"
    lrcat_path: str = "/home/sat3737/Test/Test.lrcat"
    run_dir: str = "../runs/whale_clustering/baseline_zero_shot"
    target_day: str | None = "2025-01-15"
    max_images: int | None = None
    preview_max_size: int = 1600
    preview_limit: int | None = None
    crop_limit: int | None = None
    feature_limit: int | None = None
    feature_method: str = "color_texture"
    feature_image_size: int = 224
    sequence_gap: int = 5
    time_gap_seconds: int = 300
    visual_split_min_similarity: float = 0.70
    cluster_similarity_threshold: float = 0.82
    overwrite_artifacts: bool = False

    @property
    def resolved_run_dir(self) -> Path:
        path = Path(self.run_dir).expanduser()
        if not path.is_absolute():
            if path.parts and path.parts[0] == "..":
                path = (REPO_ROOT / "notebooks" / path).resolve()
            else:
                path = (REPO_ROOT / path).resolve()
        return path

    def artifact_dir(self, name: str) -> Path:
        return self.resolved_run_dir / name


def _write_json(value: Any, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_records_csv(records: list[dict[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {}
            for key in fieldnames:
                value = record.get(key)
                if isinstance(value, (list, tuple, dict)):
                    row[key] = json.dumps(value, sort_keys=True)
                else:
                    row[key] = value
            writer.writerow(row)
    return path


def initialize_run(config: WhaleClusteringConfig) -> dict[str, Path]:
    """Create run directories and persist the config."""
    directories = {
        "run": config.resolved_run_dir,
        "manifests": config.artifact_dir("manifests"),
        "raw_catalog_exports": config.artifact_dir("raw_catalog_exports"),
        "previews": config.artifact_dir("previews"),
        "crops": config.artifact_dir("crops"),
        "masks": config.artifact_dir("masks"),
        "embeddings": config.artifact_dir("embeddings"),
        "series": config.artifact_dir("series"),
        "clusters": config.artifact_dir("clusters"),
        "reports": config.artifact_dir("reports"),
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    _write_json(asdict(config), config.resolved_run_dir / "config.json")
    return directories


def load_ground_truth(config: WhaleClusteringConfig) -> dict[str, Any]:
    """Inspect the Lightroom catalog and export normalized labels."""
    initialize_run(config)
    schema = inspect_schema(config.lrcat_path)
    collection_records = parse_lrcat_ground_truth(config.lrcat_path, image_root=config.image_root)
    collapsed_records = collapse_ground_truth_by_image(collection_records)
    summary = summarize_ground_truth(collapsed_records)

    _write_json(schema, config.artifact_dir("raw_catalog_exports") / "lightroom_schema.json")
    _write_records_csv(collection_records, config.artifact_dir("raw_catalog_exports") / "lightroom_collection_memberships.csv")
    _write_records_csv(collapsed_records, config.artifact_dir("raw_catalog_exports") / "ground_truth_by_image.csv")
    _write_json(summary, config.artifact_dir("reports") / "ground_truth_summary.json")
    return {"schema": schema, "collection_records": collection_records, "records": collapsed_records, "summary": summary}


def build_manifest(config: WhaleClusteringConfig, ground_truth: dict[str, Any] | list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build the filesystem manifest and join Lightroom labels."""
    initialize_run(config)
    ground_truth_records = ground_truth.get("records", []) if isinstance(ground_truth, dict) else (ground_truth or [])
    records = build_image_manifest(
        config.image_root,
        ground_truth_records=ground_truth_records,
        target_day=config.target_day,
        max_images=config.max_images,
    )
    summary = summarize_manifest(records)
    days = available_days(records)
    _write_records_csv(records, config.artifact_dir("manifests") / "image_manifest.csv")
    _write_json(summary, config.artifact_dir("reports") / "manifest_summary.json")
    _write_json(days, config.artifact_dir("reports") / "available_days.json")
    return {"records": records, "summary": summary, "available_days": days}


def detect_series(manifest: dict[str, Any] | list[dict[str, Any]], config: WhaleClusteringConfig) -> dict[str, Any]:
    """Run metadata-based provisional series detection."""
    initialize_run(config)
    records = manifest.get("records", []) if isinstance(manifest, dict) else manifest
    series_records = detect_provisional_series(
        records,
        sequence_gap=config.sequence_gap,
        time_gap_seconds=config.time_gap_seconds,
    )
    summary = summarize_series(series_records, series_key="validated_series_id")
    _write_records_csv(series_records, config.artifact_dir("series") / "series_manifest.csv")
    _write_records_csv(summary, config.artifact_dir("series") / "series_summary.csv")
    return {"records": series_records, "summary": summary}


def compute_artifacts(series: dict[str, Any] | list[dict[str, Any]], config: WhaleClusteringConfig) -> dict[str, Any]:
    """Create previews, crops, embeddings, and visual-boundary series summaries."""
    initialize_run(config)
    records = series.get("records", []) if isinstance(series, dict) else series
    records, preview_summary = ensure_preview_cache(
        records,
        config.artifact_dir("previews"),
        max_size=config.preview_max_size,
        overwrite=config.overwrite_artifacts,
        limit=config.preview_limit,
    )
    records, crop_summary = ensure_whale_crops(
        records,
        config.artifact_dir("crops"),
        config.artifact_dir("masks"),
        overwrite=config.overwrite_artifacts,
        limit=config.crop_limit,
    )
    records, feature_summary = ensure_feature_cache(
        records,
        config.artifact_dir("embeddings"),
        image_key="masked_crop_path",
        method=config.feature_method,
        image_size=config.feature_image_size,
        overwrite=config.overwrite_artifacts,
        limit=config.feature_limit,
    )
    records = validate_series_by_adjacent_similarity(
        records,
        feature_key="feature_vector",
        min_similarity=config.visual_split_min_similarity,
    )
    series_summary = summarize_series(records, series_key="validated_series_id")
    summary = {"previews": preview_summary, "crops": crop_summary, "features": feature_summary}
    _write_records_csv(records, config.artifact_dir("manifests") / "artifact_manifest.csv")
    _write_records_csv(series_summary, config.artifact_dir("series") / "validated_series_summary.csv")
    _write_json(summary, config.artifact_dir("reports") / "artifact_summary.json")
    return {"records": records, "series_summary": series_summary, "summary": summary}


def score_series_matches(artifacts: dict[str, Any] | list[dict[str, Any]], config: WhaleClusteringConfig) -> dict[str, Any]:
    """Build series representations and score same-day series pairs."""
    initialize_run(config)
    records = artifacts.get("records", []) if isinstance(artifacts, dict) else artifacts
    series_records = build_series_representations(records)
    pair_scores = score_series_pairs(series_records, same_day_only=True)
    _write_records_csv(series_records, config.artifact_dir("series") / "series_representations.csv")
    _write_records_csv(pair_scores, config.artifact_dir("series") / "series_pair_scores.csv")
    return {"records": records, "series": series_records, "pair_scores": pair_scores}


def cluster_day(matches: dict[str, Any], config: WhaleClusteringConfig) -> dict[str, Any]:
    """Cluster scored series within each day and attach cluster IDs to images."""
    initialize_run(config)
    clustered_series = cluster_series_by_threshold(
        matches["series"],
        matches["pair_scores"],
        threshold=config.cluster_similarity_threshold,
    )
    records = assign_clusters_to_records(matches["records"], clustered_series)
    cluster_summary = summarize_clusters(clustered_series)
    _write_records_csv(clustered_series, config.artifact_dir("clusters") / "clustered_series.csv")
    _write_records_csv(cluster_summary, config.artifact_dir("clusters") / "cluster_summary.csv")
    _write_records_csv(records, config.artifact_dir("clusters") / "clustered_images.csv")
    return {"records": records, "series": clustered_series, "summary": cluster_summary, "pair_scores": matches["pair_scores"]}


def evaluate_clusters(
    clusters: dict[str, Any],
    ground_truth: dict[str, Any] | list[dict[str, Any]] | None,
    config: WhaleClusteringConfig,
) -> dict[str, Any]:
    """Evaluate predicted image clusters against joined Lightroom labels."""
    initialize_run(config)
    metrics = evaluate_records(clusters["records"])
    _write_json(metrics, config.artifact_dir("reports") / "cluster_metrics.json")
    return metrics
