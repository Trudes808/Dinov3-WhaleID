"""Feature extraction and embedding cache helpers for clustering baselines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def _image_signature(image_path: str | Path) -> str:
    path = Path(image_path).expanduser().resolve()
    try:
        stat = path.stat()
        payload = f"{path}|{stat.st_mtime_ns}|{stat.st_size}"
    except FileNotFoundError:
        payload = str(path)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def embedding_path_for_image(
    image_path: str | Path,
    embedding_dir: str | Path,
    method: str = "color_texture",
    image_size: int = 224,
) -> Path:
    """Return deterministic embedding cache path for an image/method."""
    digest = hashlib.sha1(f"{_image_signature(image_path)}|{method}|{image_size}".encode("utf-8")).hexdigest()[:18]
    return Path(embedding_dir) / method / f"{digest}.npz"


def _open_feature_image(image_path: str | Path, image_size: int) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    offset = ((image_size - image.width) // 2, (image_size - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def compute_color_texture_embedding(image_path: str | Path, image_size: int = 224) -> np.ndarray:
    """Compute a lightweight deterministic visual descriptor for one image."""
    image = _open_feature_image(image_path, image_size)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    gray_image = image.convert("L")
    gray = np.asarray(gray_image, dtype=np.float32) / 255.0

    features: list[np.ndarray] = []
    for channel_index in range(3):
        histogram, _ = np.histogram(rgb[:, :, channel_index], bins=16, range=(0.0, 1.0), density=True)
        features.append(histogram.astype(np.float32))
    for channel_index in range(3):
        histogram, _ = np.histogram(hsv[:, :, channel_index], bins=12, range=(0.0, 1.0), density=True)
        features.append(histogram.astype(np.float32))

    grid_features: list[float] = []
    grid_size = 4
    cell_height = image_size // grid_size
    cell_width = image_size // grid_size
    for row_index in range(grid_size):
        for column_index in range(grid_size):
            cell = rgb[
                row_index * cell_height : (row_index + 1) * cell_height,
                column_index * cell_width : (column_index + 1) * cell_width,
                :,
            ]
            grid_features.extend(cell.mean(axis=(0, 1)).tolist())
            grid_features.append(float(cell.std()))
    features.append(np.asarray(grid_features, dtype=np.float32))

    edge_image = gray_image.filter(ImageFilter.FIND_EDGES)
    edges = np.asarray(edge_image, dtype=np.float32) / 255.0
    gradient_histogram, _ = np.histogram(edges, bins=16, range=(0.0, 1.0), density=True)
    features.append(gradient_histogram.astype(np.float32))
    features.append(np.asarray([gray.mean(), gray.std(), edges.mean(), edges.std()], dtype=np.float32))
    return _normalize_vector(np.concatenate(features))


def _batched(values: list[str | Path], batch_size: int) -> Iterable[list[str | Path]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def compute_dinov3_embeddings(
    image_paths: list[str | Path],
    model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
    batch_size: int = 8,
    device: str | None = None,
) -> list[np.ndarray]:
    """Compute DINOv3 class-token embeddings when torch/transformers are available."""
    try:
        import torch  # type: ignore
        from transformers import AutoImageProcessor, AutoModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError("DINOv3 embeddings require torch and transformers") from exc

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(selected_device)
    model.eval()

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for batch_paths in _batched(image_paths, batch_size):
            images = [ImageOps.exif_transpose(Image.open(path)).convert("RGB") for path in batch_paths]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(selected_device) for key, value in inputs.items()}
            outputs = model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                batch_embeddings = outputs.pooler_output
            else:
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
            for vector in batch_embeddings.detach().cpu().numpy():
                embeddings.append(_normalize_vector(np.asarray(vector, dtype=np.float32)))
    return embeddings


def compute_embedding(
    image_path: str | Path,
    method: str = "color_texture",
    image_size: int = 224,
) -> np.ndarray:
    """Compute one embedding using a supported method."""
    if method == "color_texture":
        return compute_color_texture_embedding(image_path, image_size=image_size)
    if method == "dinov3":
        return compute_dinov3_embeddings([image_path], batch_size=1)[0]
    raise ValueError(f"Unsupported feature method: {method}")


def save_embedding(path: str | Path, vector: np.ndarray, metadata: dict[str, Any]) -> Path:
    """Write an embedding vector and JSON metadata to an NPZ cache file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, vector=np.asarray(vector, dtype=np.float32), metadata=json.dumps(metadata, sort_keys=True))
    return output_path


def load_embedding(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load an embedding vector and metadata from cache."""
    with np.load(path, allow_pickle=False) as data:
        vector = np.asarray(data["vector"], dtype=np.float32)
        metadata = json.loads(str(data["metadata"]))
    return vector, metadata


def ensure_feature_cache(
    records: list[dict[str, Any]],
    embedding_dir: str | Path,
    image_key: str = "crop_path",
    method: str = "color_texture",
    image_size: int = 224,
    overwrite: bool = False,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach embedding vectors and cache paths to manifest records."""
    updated_records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"ready": 0, "failed": 0, "skipped_by_limit": 0, "method": method, "errors": []}
    processed_count = 0
    for record in records:
        updated_record = dict(record)
        if limit is not None and processed_count >= limit:
            updated_record["feature_status"] = "not_requested"
            summary["skipped_by_limit"] += 1
            updated_records.append(updated_record)
            continue
        processed_count += 1

        source_value = record.get(image_key) or record.get("crop_path") or record.get("preview_path") or record.get("image_path")
        source_path = Path(str(source_value)) if source_value else None
        if not source_path or not source_path.exists():
            updated_record["feature_status"] = "source_missing"
            summary["failed"] += 1
            updated_records.append(updated_record)
            continue

        embedding_path = embedding_path_for_image(source_path, embedding_dir, method=method, image_size=image_size)
        try:
            if embedding_path.exists() and not overwrite:
                vector, metadata = load_embedding(embedding_path)
            else:
                vector = compute_embedding(source_path, method=method, image_size=image_size)
                metadata = {
                    "image_path": str(source_path),
                    "method": method,
                    "image_size": image_size,
                    "source_signature": _image_signature(source_path),
                }
                save_embedding(embedding_path, vector, metadata)
            updated_record["feature_status"] = "ready"
            updated_record["feature_method"] = metadata.get("method", method)
            updated_record["feature_path"] = str(embedding_path)
            updated_record["feature_vector"] = vector.astype(float).tolist()
            summary["ready"] += 1
        except Exception as exc:
            updated_record["feature_status"] = f"failed: {exc}"
            summary["failed"] += 1
            if len(summary["errors"]) < 12:
                summary["errors"].append({"image_path": str(source_path), "error": str(exc)})
        updated_records.append(updated_record)
    return updated_records, summary