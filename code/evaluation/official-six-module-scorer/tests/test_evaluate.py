from __future__ import annotations

import copy
from dataclasses import replace

import pytest
from goai_scorer.evaluate import evaluate_validation


def test_perfect_prediction_scores_one_on_all_modules(scoring_data, config):
    result = evaluate_validation(scoring_data, config).result
    assert result["provisional_weighted_proxy_100"] == pytest.approx(100.0)
    for module in result["modules"].values():
        assert module["normalized_score"] == pytest.approx(1.0)


def test_official_weights_are_exact(scoring_data, config):
    result = evaluate_validation(scoring_data, config).result
    weights = [result["modules"][f"m{i}"]["official_weight"] for i in range(1, 7)]
    assert weights == [0.20, 0.25, 0.20, 0.20, 0.10, 0.05]
    assert sum(weights) == pytest.approx(1.0)


def test_weighted_points_sum_to_total(scoring_data, config):
    result = evaluate_validation(scoring_data, config).result
    points = sum(module["weighted_points"] for module in result["modules"].values())
    assert points == pytest.approx(result["provisional_weighted_proxy_100"])


def test_output_contains_total_modules_and_all_metric_groups(scoring_data, config):
    result = evaluate_validation(scoring_data, config).result
    assert set(result["modules"]) == {"m1", "m2", "m3", "m4", "m5", "m6"}
    assert set(result["metrics"]) == {
        "absolute_all",
        "matched_control_fc_all",
        "context_residual_val_chem_only",
        "drug_residual_val_strain_only",
        "dep",
        "by_split",
    }
    assert set(result["metrics"]["by_split"]) == set(config["validation_splits"])


def test_time_is_labeled_interpolation(scoring_data, config):
    result = evaluate_validation(scoring_data, config).result
    assert result["modules"]["m5"]["coverage"]["val_time"]["interpretation"] == "temporal_interpolation"


def test_reference_coverage_is_split_specific(scoring_data, config):
    result = evaluate_validation(scoring_data, config).result
    assert result["coverage"]["context_reference_rate_val_chem_only"] == pytest.approx(1.0)
    assert result["coverage"]["drug_reference_rate_val_strain_only"] == pytest.approx(1.0)


def test_negative_raw_pcc_is_preserved_but_proxy_clipped(scoring_data, config):
    changed = copy.deepcopy(scoring_data)
    changed.prediction_log2.loc["vc"] = changed.prediction_log2.loc["vc"].to_numpy()[::-1]
    result = evaluate_validation(changed, config).result
    assert result["metrics"]["by_split"]["val_chem_only"]["absolute"]["sample_pcc"]["value"] < 0
    assert result["modules"]["m1"]["normalized_score"] >= 0


def test_missing_context_reference_fails_closed_without_global_fallback(scoring_data, config):
    changed = copy.deepcopy(scoring_data)
    changed.metadata.loc["vc", "Yeast_cell_plate"] = "UNSEEN_PLATE"
    # Add a scoring control for the altered plate, so only the train context reference is missing.
    changed.metadata.loc["tr_ctrl_water", "Yeast_cell_plate"] = "UNSEEN_PLATE"
    result = evaluate_validation(changed, config).result
    assert result["modules"]["m3"]["coverage"]["matched_reference_samples"] == 0
    assert result["modules"]["m3"]["normalized_score"] == 0.0


def test_partially_missing_reference_is_zero_in_proxy_not_dropped(scoring_data, config):
    changed = copy.deepcopy(scoring_data)
    changed.metadata.loc["vc", "Yeast_cell_plate"] = "UNSEEN_PLATE"
    changed.metadata.loc["tr_ctrl_water", "Yeast_cell_plate"] = "UNSEEN_PLATE"
    changed.metadata.loc["vc2"] = changed.metadata.loc["vc"].copy()
    changed.metadata.loc["vc2", "Yeast_cell_plate"] = "P1"
    changed.truth_log2.loc["vc2"] = changed.truth_log2.loc["vc"].copy()
    changed.prediction_log2.loc["vc2"] = changed.prediction_log2.loc["vc"].copy()
    changed = replace(
        changed,
        target_ids=changed.target_ids.append(type(changed.target_ids)(["vc2"])),
    )
    result = evaluate_validation(changed, config).result
    module = result["modules"]["m3"]
    assert module["coverage"]["matched_reference_samples"] == 1
    assert module["raw_score"] == pytest.approx(1.0)
    assert module["normalized_score"] == pytest.approx(0.5)


def test_sample_metrics_have_per_sample_components(scoring_data, config):
    artifacts = evaluate_validation(scoring_data, config)
    required = {
        "absolute_pcc",
        "absolute_r2",
        "fc_pcc",
        "context_residual_pcc",
        "drug_residual_pcc",
        "control_matched",
    }
    assert required.issubset(artifacts.sample_metrics.columns)


def test_reference_manifest_is_train_only(scoring_data, config):
    manifest = evaluate_validation(scoring_data, config).reference_manifest
    assert manifest["fit_split"] == "train"
    assert manifest["reference_sha256"]


def test_median_profile_is_supported(scoring_data, config):
    config["aggregation"]["per_sample"] = "median"
    config["control_matching"]["aggregation"] = "median"
    result = evaluate_validation(scoring_data, config).result
    assert result["assumptions"]["per_sample_aggregation"] == "median"
    assert 0 <= result["provisional_weighted_proxy"] <= 1
