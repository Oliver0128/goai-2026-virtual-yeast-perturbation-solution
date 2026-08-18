from __future__ import annotations

import numpy as np
from goai_scorer.references import fit_train_references


def test_reference_sources_are_train_only(synthetic_frames, config):
    metadata, truth, _, _ = synthetic_frames
    refs = fit_train_references(metadata, truth, config)
    assert set(metadata.loc[refs.source_sample_ids, "split_final"]) == {"train"}
    assert set(metadata.loc[refs.source_control_sample_ids, "split_final"]) == {"train"}


def test_validation_truth_mutation_does_not_change_reference_hash(synthetic_frames, config):
    metadata, truth, _, _ = synthetic_frames
    first = fit_train_references(metadata, truth, config)
    changed = truth.copy()
    changed.loc[["vc", "vs", "vb", "vt"]] += 999.0
    second = fit_train_references(metadata, changed, config)
    assert first.sha256 == second.sha256


def test_train_truth_mutation_changes_reference_hash(synthetic_frames, config):
    metadata, truth, _, _ = synthetic_frames
    first = fit_train_references(metadata, truth, config)
    changed = truth.copy()
    changed.loc["tr_d1", "P1"] += 1.0
    second = fit_train_references(metadata, changed, config)
    assert first.sha256 != second.sha256


def test_context_reference_is_mean_of_train_drug_deltas(synthetic_frames, config):
    metadata, truth, _, _ = synthetic_frames
    refs = fit_train_references(metadata, truth, config)
    context = refs.context_for(metadata.loc["vc"])
    ctrl = truth.loc[["tr_ctrl_water", "tr_ctrl_dmso"]].mean(axis=0).to_numpy()
    expected = np.mean(
        np.vstack([truth.loc["tr_d1"].to_numpy() - ctrl, truth.loc["tr_d2"].to_numpy() - ctrl]),
        axis=0,
    )
    np.testing.assert_allclose(context, expected)


def test_drug_reference_is_train_delta_for_same_drug(synthetic_frames, config):
    metadata, truth, _, _ = synthetic_frames
    refs = fit_train_references(metadata, truth, config)
    drug = refs.drug_for(metadata.loc["vs"])
    ctrl = truth.loc[["tr_ctrl_water", "tr_ctrl_dmso"]].mean(axis=0).to_numpy()
    np.testing.assert_allclose(drug, truth.loc["tr_d1"].to_numpy() - ctrl)


def test_unseen_drug_has_no_drug_reference(synthetic_frames, config):
    metadata, truth, _, _ = synthetic_frames
    refs = fit_train_references(metadata, truth, config)
    assert refs.drug_for(metadata.loc["vc"]) is None
