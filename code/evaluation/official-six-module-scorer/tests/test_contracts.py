from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
from goai_scorer.cli import _json_safe
from goai_scorer.contracts import load_config, load_scoring_data


def test_load_contract_success(synthetic_files, config):
    metadata, proteome, prediction = synthetic_files
    data = load_scoring_data(metadata, proteome, prediction, config)
    assert len(data.target_ids) == 4
    assert len(data.retained_proteins) == 6
    assert np.isfinite(data.prediction_log2.to_numpy()).all()
    assert all(dtype == np.dtype("float64") for dtype in data.truth_log2.dtypes)
    assert all(dtype == np.dtype("float64") for dtype in data.prediction_log2.dtypes)


def test_json_safe_preserves_boolean_type():
    payload = _json_safe({"python": True, "numpy": np.bool_(False)})
    assert payload == {"python": True, "numpy": False}
    assert all(isinstance(value, bool) for value in payload.values())


def test_rejects_separate_evaluation_truth_path(synthetic_files, config, tmp_path):
    metadata, proteome, prediction = synthetic_files
    forbidden = tmp_path / "proteome_raw_test.csv"
    forbidden.write_bytes(proteome.read_bytes())
    with pytest.raises(ValueError, match="Validation-only scorer rejected"):
        load_scoring_data(metadata, forbidden, prediction, config)


def test_missing_prediction_protein_fails(synthetic_files, config):
    metadata, proteome, prediction = synthetic_files
    frame = pd.read_csv(prediction).drop(columns=["P6"])
    frame.to_csv(prediction, index=False)
    with pytest.raises(ValueError, match="missing retained proteins"):
        load_scoring_data(metadata, proteome, prediction, config)


def test_nonfinite_prediction_fails(synthetic_files, config):
    metadata, proteome, prediction = synthetic_files
    frame = pd.read_csv(prediction)
    frame.loc[0, "P1"] = np.nan
    frame.to_csv(prediction, index=False)
    with pytest.raises(ValueError, match="Predictions must be finite"):
        load_scoring_data(metadata, proteome, prediction, config)


def test_missing_target_sample_fails(synthetic_files, config):
    metadata, proteome, prediction = synthetic_files
    frame = pd.read_csv(prediction).iloc[1:]
    frame.to_csv(prediction, index=False)
    with pytest.raises(ValueError, match="missing validation targets"):
        load_scoring_data(metadata, proteome, prediction, config)


def test_extra_official_protein_column_is_allowed(synthetic_files, config):
    metadata, proteome, prediction = synthetic_files
    raw = pd.read_csv(proteome)
    raw["P_EXTRA"] = 2.0
    raw.to_csv(proteome, index=False)
    pred = pd.read_csv(prediction)
    pred["P_EXTRA"] = 1.0
    pred.to_csv(prediction, index=False)
    config["input_contract"]["expected_raw_proteins"] = 7
    data = load_scoring_data(metadata, proteome, prediction, config)
    assert "P_EXTRA" in data.retained_proteins


def test_unknown_prediction_protein_fails(synthetic_files, config):
    metadata, proteome, prediction = synthetic_files
    pred = pd.read_csv(prediction)
    pred["NOT_IN_OFFICIAL_PROTEOME"] = 1.0
    pred.to_csv(prediction, index=False)
    with pytest.raises(ValueError, match="unknown protein columns"):
        load_scoring_data(metadata, proteome, prediction, config)


def test_unknown_prediction_sample_fails(synthetic_files, config):
    metadata, proteome, prediction = synthetic_files
    pred = pd.read_csv(prediction)
    extra = pred.iloc[[0]].copy()
    extra["sample_ID"] = "unknown_sample"
    pd.concat([pred, extra], ignore_index=True).to_csv(prediction, index=False)
    with pytest.raises(ValueError, match="unknown sample IDs"):
        load_scoring_data(metadata, proteome, prediction, config)


def test_positive_infinity_in_raw_proteome_fails(synthetic_files, config):
    metadata, proteome, prediction = synthetic_files
    raw = pd.read_csv(proteome)
    raw.loc[0, "P1"] = np.inf
    raw.to_csv(proteome, index=False)
    with pytest.raises(ValueError, match=r"not \+/-inf"):
        load_scoring_data(metadata, proteome, prediction, config)


def test_embedded_test_split_fails_even_with_train_val_filename(synthetic_files, config):
    metadata, proteome, prediction = synthetic_files
    meta = pd.read_csv(metadata)
    raw = pd.read_csv(proteome)
    test_meta = meta.iloc[[0]].copy()
    test_meta["sample_ID"] = "hidden_test_truth"
    test_meta["split_final"] = "test_both"
    test_raw = raw.iloc[[0]].copy()
    test_raw["sample_ID"] = "hidden_test_truth"
    pd.concat([meta, test_meta], ignore_index=True).to_csv(metadata, index=False)
    pd.concat([raw, test_raw], ignore_index=True).to_csv(proteome, index=False)
    with pytest.raises(ValueError, match="forbidden or unknown split_final"):
        load_scoring_data(metadata, proteome, prediction, config)


def test_frozen_input_hash_mismatch_fails(synthetic_files, config):
    metadata, proteome, prediction = synthetic_files
    config["input_contract"]["expected_metadata_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Metadata SHA-256"):
        load_scoring_data(metadata, proteome, prediction, config)


def test_train_only_filter_ignores_validation_missingness(synthetic_files, config):
    metadata, proteome, prediction = synthetic_files
    first = load_scoring_data(metadata, proteome, prediction, config)
    raw = pd.read_csv(proteome)
    meta = pd.read_csv(metadata).set_index("sample_ID")
    val_ids = meta.index[meta["split_final"].ne("train")]
    raw.loc[raw["sample_ID"].isin(val_ids), "P1"] = np.nan
    raw.to_csv(proteome, index=False)
    second = load_scoring_data(metadata, proteome, prediction, config)
    assert first.retained_proteins.tolist() == second.retained_proteins.tolist()


def test_train_missingness_can_remove_protein(synthetic_files, config):
    metadata, proteome, prediction = synthetic_files
    raw = pd.read_csv(proteome)
    meta = pd.read_csv(metadata).set_index("sample_ID")
    train_ids = meta.index[meta["split_final"].eq("train")]
    raw.loc[raw["sample_ID"].isin(train_ids), "P6"] = np.nan
    raw.to_csv(proteome, index=False)
    data = load_scoring_data(metadata, proteome, prediction, config)
    assert "P6" not in data.retained_proteins


def test_configured_module_weights_must_sum_to_one(tmp_path, config):
    bad = copy.deepcopy(config)
    bad["module_weights"]["m1"] = 0.5
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="sum to 1"):
        load_config(path)


def test_config_rejects_nonofficial_module_weights(tmp_path, config):
    bad = copy.deepcopy(config)
    bad["module_weights"]["m1"] = 0.19
    bad["module_weights"]["m2"] = 0.26
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="official weights"):
        load_config(path)


@pytest.mark.parametrize("section", ["protein_filter", "references"])
def test_config_rejects_nontrain_fit_split(tmp_path, config, section):
    bad = copy.deepcopy(config)
    bad[section]["fit_split"] = "val_chem_only"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="must be fit on split_final=train"):
        load_config(path)
