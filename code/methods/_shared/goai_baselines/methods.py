from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .core import (
    CONTROL_LABELS,
    QC_LABEL,
    BaselineData,
    design_matrix,
    exact_control_profiles,
    hierarchical_context_prediction,
    impute_targets,
    multioutput_ridge,
    repeat_profile,
    treatment_mask,
)


def protein_mean(data: BaselineData, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    return repeat_profile(data.train_mean, data.target_ids), {"fit_rows": len(data.train_ids)}


def train_control_mean(data: BaselineData, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = data.metadata.loc[data.train_ids, "perturbation_no_concentration"].astype(str)
    control_ids = data.train_ids[labels.isin(CONTROL_LABELS)]
    profile = data.truth_log2.loc[control_ids].mean(axis=0).fillna(data.train_mean)
    return repeat_profile(profile, data.target_ids), {"fit_rows": len(control_ids)}


def hierarchical_context(data: BaselineData, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction, sample_levels = hierarchical_context_prediction(data, config["hierarchical_levels"])
    counts = pd.Series(sample_levels).value_counts().sort_index().to_dict()
    return prediction, {"fit_rows": len(data.train_ids), "sample_fallback_level_counts": counts}


def biological_ridge(data: BaselineData, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    fit_ids = data.train_ids[data.metadata.loc[data.train_ids, "perturbation_no_concentration"].ne(QC_LABEL)]
    x_fit, x_target, design = design_matrix(
        data.metadata.loc[fit_ids],
        data.metadata.loc[data.target_ids],
        config["categorical_columns"],
        config["numeric_columns"],
    )
    y_fit, target_means = impute_targets(data.truth_log2.loc[fit_ids], data.train_mean)
    coefficients = multioutput_ridge(x_fit, y_fit, config["ridge_alpha"])
    prediction = pd.DataFrame(x_target @ coefficients, index=data.target_ids, columns=data.retained_proteins)
    return prediction, {
        "fit_rows": len(fit_ids),
        "ridge_alpha": config["ridge_alpha"],
        "design": design,
        "target_imputation_mean_sha256": __import__("hashlib").sha256(target_means.to_numpy().tobytes()).hexdigest(),
    }


def control_residual_ridge(data: BaselineData, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_treatment_ids = data.train_ids[treatment_mask(data.metadata.loc[data.train_ids])]
    labels = data.metadata.loc[data.train_ids, "perturbation_no_concentration"].astype(str)
    train_control_ids = data.train_ids[labels.isin(CONTROL_LABELS)]
    train_controls, matched = exact_control_profiles(data, train_treatment_ids, train_control_ids)
    matched_ids = train_treatment_ids[matched.to_numpy()]
    train_controls = train_controls.loc[matched_ids]
    residual = data.truth_log2.loc[matched_ids] - train_controls
    residual_fallback = residual.mean(axis=0).fillna(0.0)
    y_fit, residual_means = impute_targets(residual, residual_fallback)
    x_fit, x_target, design = design_matrix(
        data.metadata.loc[matched_ids],
        data.metadata.loc[data.target_ids],
        config["categorical_columns"],
        config["numeric_columns"],
    )
    coefficients = multioutput_ridge(x_fit, y_fit, config["ridge_alpha"])
    delta_prediction = x_target @ coefficients
    base_prediction, fallback_levels = hierarchical_context_prediction(
        data,
        config["base_hierarchical_levels"],
        fit_ids=train_control_ids,
    )
    prediction = pd.DataFrame(
        base_prediction.to_numpy(dtype=np.float64) + delta_prediction,
        index=data.target_ids,
        columns=data.retained_proteins,
    )
    return prediction, {
        "fit_rows": len(matched_ids),
        "unmatched_train_treatments": int((~matched).sum()),
        "train_control_rows": len(train_control_ids),
        "ridge_alpha": config["ridge_alpha"],
        "design": design,
        "base_fallback_level_counts": pd.Series(fallback_levels).value_counts().sort_index().to_dict(),
        "residual_imputation_mean_sha256": __import__("hashlib").sha256(residual_means.to_numpy().tobytes()).hexdigest(),
    }


def matched_control_oracle(data: BaselineData, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if config.get("diagnostic_only") is not True or config.get("deployable") is not False:
        raise ValueError("Oracle requires diagnostic_only=true and deployable=false")
    labels = data.metadata["perturbation_no_concentration"].astype(str)
    all_control_ids = data.metadata.index[labels.isin(CONTROL_LABELS)]
    validation_control_ids = all_control_ids[
        data.metadata.loc[all_control_ids, "split_final"].isin(config["validation_splits"])
    ]
    profiles, matched = exact_control_profiles(data, data.target_ids, all_control_ids)
    prediction = profiles.fillna(data.train_mean)
    return prediction, {
        "all_control_rows_read": len(all_control_ids),
        "validation_control_rows_read": len(validation_control_ids),
        "matched_targets": int(matched.sum()),
        "unmatched_targets_fallback_to_train_mean": int((~matched).sum()),
        "truth_boundary_exception": "may use released validation control truth; diagnostic oracle; never test inference",
    }


METHODS = {
    "b0-protein-mean": protein_mean,
    "b1-train-control-mean": train_control_mean,
    "b2-hierarchical-context-mean": hierarchical_context,
    "b3-biological-metadata-ridge": biological_ridge,
    "b4-control-residual-ridge": control_residual_ridge,
    "d0-matched-control-oracle": matched_control_oracle,
}
