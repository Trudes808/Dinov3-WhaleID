"""Evaluate predicted whale clusters against Lightroom ground truth."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any


def _comb2(value: int) -> int:
    return value * (value - 1) // 2


def _labeled_records(records: list[dict[str, Any]], truth_key: str, prediction_key: str) -> list[dict[str, Any]]:
    return [record for record in records if record.get(truth_key) and record.get(prediction_key)]


def pairwise_scores(records: list[dict[str, Any]], truth_key: str, prediction_key: str) -> dict[str, float | int]:
    """Return pairwise precision/recall/F1 for labeled records."""
    true_positive = false_positive = false_negative = true_negative = 0
    for left, right in combinations(records, 2):
        same_truth = left.get(truth_key) == right.get(truth_key)
        same_prediction = left.get(prediction_key) == right.get(prediction_key)
        if same_truth and same_prediction:
            true_positive += 1
        elif same_prediction and not same_truth:
            false_positive += 1
        elif same_truth and not same_prediction:
            false_negative += 1
        else:
            true_negative += 1
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pair_count": true_positive + false_positive + false_negative + true_negative,
        "true_positive_pairs": true_positive,
        "false_positive_pairs": false_positive,
        "false_negative_pairs": false_negative,
        "true_negative_pairs": true_negative,
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_f1": f1,
    }


def adjusted_rand_index(records: list[dict[str, Any]], truth_key: str, prediction_key: str) -> float:
    """Compute adjusted Rand index without requiring sklearn."""
    if len(records) < 2:
        return 0.0
    truth_counts = Counter(record[truth_key] for record in records)
    prediction_counts = Counter(record[prediction_key] for record in records)
    contingency = Counter((record[truth_key], record[prediction_key]) for record in records)
    sum_comb_c = sum(_comb2(count) for count in contingency.values())
    sum_comb_truth = sum(_comb2(count) for count in truth_counts.values())
    sum_comb_prediction = sum(_comb2(count) for count in prediction_counts.values())
    total_pairs = _comb2(len(records))
    if total_pairs == 0:
        return 0.0
    expected_index = (sum_comb_truth * sum_comb_prediction) / total_pairs
    max_index = (sum_comb_truth + sum_comb_prediction) / 2
    denominator = max_index - expected_index
    if denominator == 0:
        return 1.0 if sum_comb_c == max_index else 0.0
    return (sum_comb_c - expected_index) / denominator


def normalized_mutual_info(records: list[dict[str, Any]], truth_key: str, prediction_key: str) -> float:
    """Compute normalized mutual information using arithmetic mean entropy."""
    total = len(records)
    if total == 0:
        return 0.0
    truth_counts = Counter(record[truth_key] for record in records)
    prediction_counts = Counter(record[prediction_key] for record in records)
    contingency = Counter((record[truth_key], record[prediction_key]) for record in records)

    mutual_info = 0.0
    for (truth_label, prediction_label), count in contingency.items():
        mutual_info += (count / total) * math.log((count * total) / (truth_counts[truth_label] * prediction_counts[prediction_label]))
    truth_entropy = -sum((count / total) * math.log(count / total) for count in truth_counts.values())
    prediction_entropy = -sum((count / total) * math.log(count / total) for count in prediction_counts.values())
    denominator = (truth_entropy + prediction_entropy) / 2
    return mutual_info / denominator if denominator else 1.0


def cluster_purity(records: list[dict[str, Any]], truth_key: str, prediction_key: str) -> float:
    """Return weighted cluster purity for labeled records."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[prediction_key])].append(record)
    correct = 0
    for cluster_records in grouped.values():
        truth_counts = Counter(record[truth_key] for record in cluster_records)
        correct += truth_counts.most_common(1)[0][1]
    return correct / len(records) if records else 0.0


def split_merge_counts(records: list[dict[str, Any]], truth_key: str, prediction_key: str) -> dict[str, int]:
    """Count ground-truth over-splits and predicted over-merges."""
    clusters_by_truth: dict[str, set[str]] = defaultdict(set)
    truth_by_cluster: dict[str, set[str]] = defaultdict(set)
    for record in records:
        truth_label = str(record[truth_key])
        prediction_label = str(record[prediction_key])
        clusters_by_truth[truth_label].add(prediction_label)
        truth_by_cluster[prediction_label].add(truth_label)
    return {
        "over_split_truth_whales": sum(len(cluster_ids) > 1 for cluster_ids in clusters_by_truth.values()),
        "over_merged_predicted_clusters": sum(len(truth_ids) > 1 for truth_ids in truth_by_cluster.values()),
    }


def evaluate_records(
    records: list[dict[str, Any]],
    truth_key: str = "ground_truth_whale_id",
    prediction_key: str = "predicted_whale_cluster_id",
    day_key: str = "day_label",
) -> dict[str, Any]:
    """Evaluate image-level cluster assignments overall and by day."""
    labeled_records = _labeled_records(records, truth_key, prediction_key)
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in labeled_records:
        by_day[str(record.get(day_key) or "unknown_day")].append(record)

    day_metrics = []
    for day_label, day_records in sorted(by_day.items()):
        day_metrics.append(_evaluate_group(day_records, truth_key, prediction_key, day_label=day_label))

    overall = _evaluate_group(labeled_records, truth_key, prediction_key, day_label="overall")
    overall["by_day"] = day_metrics
    return overall


def _evaluate_group(
    records: list[dict[str, Any]],
    truth_key: str,
    prediction_key: str,
    day_label: str,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "day_label": day_label,
        "labeled_images": len(records),
        "ground_truth_whales": len({record[truth_key] for record in records}) if records else 0,
        "predicted_clusters": len({record[prediction_key] for record in records}) if records else 0,
    }
    metrics.update(pairwise_scores(records, truth_key, prediction_key))
    metrics["adjusted_rand_index"] = adjusted_rand_index(records, truth_key, prediction_key)
    metrics["normalized_mutual_info"] = normalized_mutual_info(records, truth_key, prediction_key)
    metrics["cluster_purity"] = cluster_purity(records, truth_key, prediction_key)
    metrics.update(split_merge_counts(records, truth_key, prediction_key))
    return metrics