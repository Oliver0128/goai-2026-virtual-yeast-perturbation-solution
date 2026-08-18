from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
)


def paired_finite(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != pred.shape:
        raise ValueError(f"Shape mismatch: truth={truth.shape}, prediction={pred.shape}")
    mask = np.isfinite(truth) & np.isfinite(pred)
    return truth[mask], pred[mask]


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth, pred = paired_finite(y_true, y_pred)
    if truth.size < 2:
        return float("nan")
    truth_centered = truth - truth.mean()
    pred_centered = pred - pred.mean()
    denominator = np.sqrt(
        np.dot(truth_centered, truth_centered) * np.dot(pred_centered, pred_centered)
    )
    if denominator <= 0 or not np.isfinite(denominator):
        return float("nan")
    return float(np.dot(truth_centered, pred_centered) / denominator)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth, pred = paired_finite(y_true, y_pred)
    if truth.size < 2:
        return float("nan")
    denominator = np.square(truth - truth.mean()).sum()
    if denominator <= 0 or not np.isfinite(denominator):
        return float("nan")
    return float(1.0 - np.square(truth - pred).sum() / denominator)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth, pred = paired_finite(y_true, y_pred)
    if truth.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(truth - pred))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth, pred = paired_finite(y_true, y_pred)
    if truth.size == 0:
        return float("nan")
    return float(np.mean(np.abs(truth - pred)))


def vector_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    truth, _ = paired_finite(y_true, y_pred)
    return {
        "pcc": pearson(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "paired_proteins": int(truth.size),
    }


def finite_aggregate(values: Iterable[float], method: str) -> tuple[float, int, int]:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    undefined = int(array.size - finite.size)
    if finite.size == 0:
        return float("nan"), 0, undefined
    if method == "mean":
        value = np.mean(finite)
    elif method == "median":
        value = np.median(finite)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    return float(value), int(finite.size), undefined


def zero_filled_aggregate(values: Iterable[float], method: str) -> float:
    """Aggregate a score-bearing sample metric with undefined samples set to zero.

    The finite-only raw summary remains available separately through ``finite_aggregate``.
    An empty candidate set remains undefined and is converted to zero only by the module
    normalization layer.
    """

    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return float("nan")
    filled = np.where(np.isfinite(array), array, 0.0)
    if method == "mean":
        value = np.mean(filled)
    elif method == "median":
        value = np.median(filled)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    return float(value)


def normalize_score(value: float, policy: str) -> float:
    if not np.isfinite(value):
        return 0.0
    if policy == "clip_0_1":
        return float(np.clip(value, 0.0, 1.0))
    if policy == "correlation_to_unit":
        return float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))
    if policy == "identity":
        return float(value)
    raise ValueError(f"Unknown score normalization policy: {policy}")


def recall_at_k(truth_positive: np.ndarray, scores: np.ndarray, k: int) -> float:
    truth = np.asarray(truth_positive, dtype=bool)
    rank_scores = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(rank_scores)
    truth = truth[mask]
    rank_scores = rank_scores[mask]
    positives = int(truth.sum())
    if positives == 0 or k <= 0 or truth.size == 0:
        return float("nan")
    k = min(int(k), int(truth.size))
    order = np.argsort(-rank_scores, kind="stable")[:k]
    return float(truth[order].sum() / positives)


def dep_metrics(
    delta_true: np.ndarray,
    delta_pred: np.ndarray,
    threshold: float,
    fixed_ks: list[int],
    sample_slices: list[slice],
    aggregation: str,
) -> dict[str, Any]:
    truth = np.asarray(delta_true, dtype=np.float64)
    pred = np.asarray(delta_pred, dtype=np.float64)
    if truth.shape != pred.shape or truth.ndim != 1:
        raise ValueError("DEP inputs must be equal-length flattened arrays")
    mask = np.isfinite(truth) & np.isfinite(pred)
    truth = truth[mask]
    pred = pred[mask]
    truth_positive = np.abs(truth) > float(threshold)
    pred_positive = np.abs(pred) > float(threshold)
    support = int(truth_positive.sum())
    predicted_support = int(pred_positive.sum())
    tp = int(np.sum(truth_positive & pred_positive))
    fp = int(np.sum(~truth_positive & pred_positive))
    fn = int(np.sum(truth_positive & ~pred_positive))
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else float("nan")
    f1 = float(2 * precision * recall / (precision + recall)) if np.isfinite(recall) and precision + recall else 0.0

    if support and support < truth.size:
        average_precision = float(average_precision_score(truth_positive, np.abs(pred)))
        precision_curve, recall_curve, _ = precision_recall_curve(truth_positive, np.abs(pred))
        pr_auc_trapezoid = float(auc(recall_curve[::-1], precision_curve[::-1]))
    else:
        average_precision = float("nan")
        pr_auc_trapezoid = float("nan")

    high_truth = truth[truth_positive]
    high_pred = pred[truth_positive]
    direction_accuracy = (
        float(np.mean(np.sign(high_truth) == np.sign(high_pred))) if high_truth.size else float("nan")
    )
    high_effect_pcc_pooled = pearson(high_truth, high_pred)

    # Per-sample high-effect PCC and Recall@K are calculated before the global finite mask.
    original_truth = np.asarray(delta_true, dtype=np.float64)
    original_pred = np.asarray(delta_pred, dtype=np.float64)
    per_sample_pcc: list[float] = []
    recall_truth_count: list[float] = []
    fixed_recall: dict[int, list[float]] = {int(k): [] for k in fixed_ks}
    for sample_slice in sample_slices:
        yt = original_truth[sample_slice]
        yp = original_pred[sample_slice]
        finite = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[finite]
        yp = yp[finite]
        positive = np.abs(yt) > float(threshold)
        per_sample_pcc.append(pearson(yt[positive], yp[positive]))
        k_truth = int(positive.sum())
        recall_truth_count.append(recall_at_k(positive, np.abs(yp), k_truth))
        for k, values in fixed_recall.items():
            values.append(recall_at_k(positive, np.abs(yp), int(k)))

    pcc_sample, pcc_n, pcc_undefined = finite_aggregate(per_sample_pcc, aggregation)
    pcc_sample_proxy = zero_filled_aggregate(per_sample_pcc, aggregation)
    recall_k_truth, recall_n, recall_undefined = finite_aggregate(recall_truth_count, aggregation)
    fixed_summary = {}
    for k, values in fixed_recall.items():
        value, n, undefined = finite_aggregate(values, aggregation)
        fixed_summary[f"recall_at_{k}"] = {
            "value": value,
            "evaluated_samples": n,
            "undefined_samples": undefined,
        }

    return {
        "threshold": float(threshold),
        "paired_values": int(truth.size),
        "truth_high_effect_values": support,
        "predicted_high_effect_values": predicted_support,
        "prevalence": float(support / truth.size) if truth.size else float("nan"),
        "direction_accuracy": direction_accuracy,
        "high_effect_pcc_per_sample": pcc_sample,
        "high_effect_pcc_per_sample_proxy": pcc_sample_proxy,
        "high_effect_pcc_pooled": high_effect_pcc_pooled,
        "high_effect_pcc_evaluated_samples": pcc_n,
        "high_effect_pcc_undefined_samples": pcc_undefined,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_precision": average_precision,
        "pr_auc_trapezoid": pr_auc_trapezoid,
        "recall_at_truth_count": recall_k_truth,
        "recall_at_truth_count_evaluated_samples": recall_n,
        "recall_at_truth_count_undefined_samples": recall_undefined,
        "fixed_k": fixed_summary,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": int(truth.size - tp - fp - fn)},
    }
