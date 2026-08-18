from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
from goai_scorer.controls import match_control_profiles


def _match(metadata, truth, target_ids, control_ids, config):
    settings = config["control_matching"]
    return match_control_profiles(
        metadata,
        truth,
        target_ids,
        control_ids,
        settings["keys"],
        settings["required_equal_fields"],
        settings["aggregation"],
    )


def test_exact_control_match_and_mixed_solvent_flag(synthetic_frames, config):
    metadata, truth, _, _ = synthetic_frames
    result = _match(
        metadata,
        truth,
        pd.Index(["vc"]),
        pd.Index(["tr_ctrl_water", "tr_ctrl_dmso"]),
        config,
    )
    assert bool(result.details.loc["vc", "matched"])
    assert result.details.loc["vc", "candidate_count"] == 2
    assert bool(result.details.loc["vc", "mixed_control_labels"])
    expected = truth.loc[["tr_ctrl_water", "tr_ctrl_dmso"]].mean(axis=0)
    np.testing.assert_allclose(result.profiles.loc["vc"], expected)


@pytest.mark.parametrize(
    "field",
    ["data_source", "Strains", "Medium", "Temperature", "pert_time", "instrument", "Yeast_cell_plate", "pert_time_unit"],
)
def test_each_matching_field_is_strict(synthetic_frames, config, field):
    metadata, truth, _, _ = synthetic_frames
    changed = metadata.copy()
    if field in {"Temperature", "pert_time"}:
        changed.loc["tr_ctrl_water", field] = int(changed.loc["tr_ctrl_water", field]) + 1
    else:
        changed.loc["tr_ctrl_water", field] = f"DIFFERENT_{field}"
    result = _match(
        changed,
        truth,
        pd.Index(["vc"]),
        pd.Index(["tr_ctrl_water"]),
        config,
    )
    assert not bool(result.details.loc["vc", "matched"])


def test_mean_and_median_control_aggregation_differ(synthetic_frames, config):
    metadata, truth, _, _ = synthetic_frames
    extra = truth.loc["tr_ctrl_water"].copy() + 100
    extra.name = "extra_ctrl"
    truth = pd.concat([truth, extra.to_frame().T])
    metadata = pd.concat([metadata, metadata.loc[["tr_ctrl_water"]].rename(index={"tr_ctrl_water": "extra_ctrl"})])
    controls = pd.Index(["tr_ctrl_water", "tr_ctrl_dmso", "extra_ctrl"])
    mean_result = _match(metadata, truth, pd.Index(["vc"]), controls, config)
    median_config = copy.deepcopy(config)
    median_config["control_matching"]["aggregation"] = "median"
    median_result = _match(metadata, truth, pd.Index(["vc"]), controls, median_config)
    assert not np.allclose(mean_result.profiles.loc["vc"], median_result.profiles.loc["vc"])


def test_unmatched_control_profile_is_nan(synthetic_frames, config):
    metadata, truth, _, _ = synthetic_frames
    result = _match(metadata, truth, pd.Index(["vb"]), pd.Index(["tr_ctrl_water"]), config)
    assert result.profiles.loc["vb"].isna().all()
    assert result.details.loc["vb", "finite_control_proteins"] == 0
