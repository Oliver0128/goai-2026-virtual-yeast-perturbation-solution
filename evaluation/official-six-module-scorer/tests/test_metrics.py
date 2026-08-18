from __future__ import annotations

import numpy as np
import pytest
from goai_scorer.metrics import (
    dep_metrics,
    finite_aggregate,
    mae,
    normalize_score,
    pearson,
    r2,
    recall_at_k,
    rmse,
    zero_filled_aggregate,
)
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@pytest.mark.parametrize(
    ("truth", "pred", "expected"),
    [
        ([1, 2, 3], [1, 2, 3], 1.0),
        ([1, 2, 3], [3, 2, 1], -1.0),
        ([1, 2, 3], [11, 12, 13], 1.0),
        ([1, np.nan, 3], [1, 999, 3], 1.0),
        ([0, 1, 4, 9], [0.2, 0.8, 3.0, 10.0], pearsonr([0, 1, 4, 9], [0.2, 0.8, 3.0, 10.0]).statistic),
    ],
)
def test_pearson_matches_hand_and_scipy(truth, pred, expected):
    assert pearson(np.array(truth), np.array(pred)) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("truth", "pred"),
    [([1, 1, 1], [1, 2, 3]), ([1], [1]), ([np.nan, 1], [2, np.nan])],
)
def test_pearson_undefined_cases(truth, pred):
    assert np.isnan(pearson(np.array(truth), np.array(pred)))


@pytest.mark.parametrize(
    ("truth", "pred"),
    [
        ([1, 2, 4], [1, 2, 4]),
        ([1, 2, 4], [2, 3, 5]),
        ([1, 2, 4, 8], [0, 3, 5, 7]),
        ([-2, 0, 3], [-1, 1, 2]),
    ],
)
def test_r2_matches_sklearn(truth, pred):
    assert r2(np.array(truth), np.array(pred)) == pytest.approx(r2_score(truth, pred), abs=1e-12)


def test_r2_constant_truth_is_undefined():
    assert np.isnan(r2(np.array([2, 2, 2]), np.array([2, 2, 2])))


@pytest.mark.parametrize(
    ("truth", "pred"),
    [([0, 0, 0], [1, 1, 1]), ([1, 2, 3], [2, 0, 5]), ([1, np.nan, 3], [2, 5, 3])],
)
def test_rmse_and_mae_match_standard_definitions(truth, pred):
    yt = np.asarray(truth, dtype=float)
    yp = np.asarray(pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    assert rmse(yt, yp) == pytest.approx(np.sqrt(mean_squared_error(yt[mask], yp[mask])))
    assert mae(yt, yp) == pytest.approx(mean_absolute_error(yt[mask], yp[mask]))


def test_rmse_does_not_scale_like_euclidean_with_feature_count():
    assert rmse(np.zeros(3), np.ones(3)) == pytest.approx(1.0)
    assert rmse(np.zeros(300), np.ones(300)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("value", "policy", "expected"),
    [(-2.0, "clip_0_1", 0.0), (0.4, "clip_0_1", 0.4), (2.0, "clip_0_1", 1.0), (-1.0, "correlation_to_unit", 0.0), (0.0, "correlation_to_unit", 0.5), (1.0, "correlation_to_unit", 1.0), (np.nan, "clip_0_1", 0.0)],
)
def test_normalize_score(value, policy, expected):
    assert normalize_score(value, policy) == pytest.approx(expected)


def test_finite_aggregate_tracks_undefined():
    value, evaluated, undefined = finite_aggregate([1.0, np.nan, 3.0], "mean")
    assert value == pytest.approx(2.0)
    assert evaluated == 2
    assert undefined == 1


def test_zero_filled_aggregate_counts_undefined_samples_as_zero():
    assert zero_filled_aggregate([1.0, np.nan, 0.5], "mean") == pytest.approx(0.5)
    assert zero_filled_aggregate([1.0, np.nan, 0.5], "median") == pytest.approx(0.5)


def test_recall_at_k_known_ranking():
    truth = np.array([True, False, True, False])
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    assert recall_at_k(truth, scores, 2) == pytest.approx(0.5)
    assert recall_at_k(truth, scores, 3) == pytest.approx(1.0)


def test_dep_threshold_is_strictly_greater_than_one():
    truth = np.array([1.0, -1.0, 1.0001, -1.0001, 0.0])
    pred = truth.copy()
    result = dep_metrics(truth, pred, 1.0, [2], [slice(0, 5)], "mean")
    assert result["truth_high_effect_values"] == 2


def test_dep_perfect_prediction_core_metrics():
    truth = np.array([2.0, -2.0, 0.1, 1.5, -1.2, 0.2])
    result = dep_metrics(truth, truth.copy(), 1.0, [2], [slice(0, 6)], "mean")
    assert result["direction_accuracy"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)
    assert result["average_precision"] == pytest.approx(1.0)
    assert result["recall_at_truth_count"] == pytest.approx(1.0)
