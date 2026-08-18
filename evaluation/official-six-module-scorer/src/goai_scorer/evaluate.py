from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import __version__
from .contracts import ScoringData
from .controls import ControlMatches, match_control_profiles
from .metrics import (
    dep_metrics,
    finite_aggregate,
    normalize_score,
    vector_metrics,
    zero_filled_aggregate,
)
from .references import TrainReferences, fit_train_references

MODULE_NAMES = {
    "m1": "absolute_fidelity",
    "m2": "matched_control_raw_fc",
    "m3": "context_mean_residual",
    "m4": "drug_mean_residual",
    "m5": "dual_unseen_and_time_interpolation",
    "m6": "high_effect_and_dep_detection",
}


@dataclass(frozen=True)
class EvaluationArtifacts:
    result: dict[str, Any]
    sample_metrics: pd.DataFrame
    control_matches: pd.DataFrame
    reference_manifest: dict[str, Any]


def _metric_aggregate(
    frame: pd.DataFrame,
    column: str,
    method: str,
    *,
    score_bearing: bool,
) -> dict[str, Any]:
    value, evaluated, undefined = finite_aggregate(frame[column].tolist(), method)
    result = {
        "value": value,
        "aggregation": f"per_sample_{method}",
        "evaluated_samples": evaluated,
        "undefined_samples": undefined,
    }
    if score_bearing:
        result["proxy_value"] = zero_filled_aggregate(frame[column].tolist(), method)
        result["proxy_undefined_policy"] = "zero"
    return result


def _absolute_summary(frame: pd.DataFrame, method: str) -> dict[str, Any]:
    return {
        "sample_pcc": _metric_aggregate(
            frame, "absolute_pcc", method, score_bearing=True
        ),
        "sample_r2": _metric_aggregate(frame, "absolute_r2", method, score_bearing=True),
        "sample_rmse": _metric_aggregate(
            frame, "absolute_rmse", method, score_bearing=False
        ),
        "sample_mae": _metric_aggregate(frame, "absolute_mae", method, score_bearing=False),
        "samples": len(frame),
        "paired_values": int(frame["absolute_paired_proteins"].sum()),
    }


def _effect_summary(frame: pd.DataFrame, prefix: str, method: str) -> dict[str, Any]:
    return {
        "pcc": _metric_aggregate(frame, f"{prefix}_pcc", method, score_bearing=True),
        "r2": _metric_aggregate(frame, f"{prefix}_r2", method, score_bearing=False),
        "rmse": _metric_aggregate(frame, f"{prefix}_rmse", method, score_bearing=False),
        "mae": _metric_aggregate(frame, f"{prefix}_mae", method, score_bearing=False),
        "candidate_samples": len(frame),
        "matched_samples": int(frame[f"{prefix}_pcc"].notna().sum()),
    }


def _weighted_raw(values: dict[str, float], weights: dict[str, float]) -> float:
    if not values:
        return float("nan")
    total = 0.0
    for key, weight in weights.items():
        value = float(values.get(key, float("nan")))
        if not np.isfinite(value):
            return float("nan")
        total += float(weight) * value
    return float(total)


def _module_record(
    module_id: str,
    official_weight: float,
    internal_values: dict[str, float],
    internal_proxy_values: dict[str, float],
    internal_weights: dict[str, float],
    normalization: str,
    coverage: dict[str, Any],
) -> dict[str, Any]:
    raw_score = _weighted_raw(internal_values, internal_weights)
    proxy_score_before_normalization = _weighted_raw(internal_proxy_values, internal_weights)
    normalized_components = {
        key: normalize_score(float(internal_proxy_values.get(key, float("nan"))), normalization)
        for key in internal_weights
    }
    normalized_score = float(
        sum(float(internal_weights[key]) * normalized_components[key] for key in internal_weights)
    )
    return {
        "module_id": module_id,
        "name": MODULE_NAMES[module_id],
        "official_weight": float(official_weight),
        "raw_score": raw_score,
        "proxy_score_before_normalization": proxy_score_before_normalization,
        "normalized_score": normalized_score,
        "weighted_points": float(100.0 * official_weight * normalized_score),
        "internal_values": internal_values,
        "internal_proxy_values": internal_proxy_values,
        "internal_weights": internal_weights,
        "normalized_components": normalized_components,
        "normalization": normalization,
        "coverage": coverage,
    }


def _build_sample_metrics(
    data: ScoringData,
    controls: ControlMatches,
    references: TrainReferences,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[slice]]:
    records: list[dict[str, Any]] = []
    dep_truth_chunks: list[np.ndarray] = []
    dep_pred_chunks: list[np.ndarray] = []
    sample_slices: list[slice] = []
    offset = 0
    for sample_id in data.target_ids:
        truth = data.truth_log2.loc[sample_id].to_numpy(dtype=np.float64, copy=False)
        prediction = data.prediction_log2.loc[sample_id].to_numpy(dtype=np.float64, copy=False)
        absolute = vector_metrics(truth, prediction)
        row = data.metadata.loc[sample_id]
        record: dict[str, Any] = {
            "sample_ID": str(sample_id),
            "split": str(row["split_final"]),
            "absolute_pcc": absolute["pcc"],
            "absolute_r2": absolute["r2"],
            "absolute_rmse": absolute["rmse"],
            "absolute_mae": absolute["mae"],
            "absolute_paired_proteins": absolute["paired_proteins"],
            "control_matched": bool(controls.details.loc[str(sample_id), "matched"]),
            "context_reference_matched": False,
            "drug_reference_matched": False,
        }
        for prefix in ("fc", "context_residual", "drug_residual"):
            for metric in ("pcc", "r2", "rmse", "mae"):
                record[f"{prefix}_{metric}"] = float("nan")

        if record["control_matched"]:
            control = controls.profiles.loc[sample_id].to_numpy(dtype=np.float64, copy=False)
            delta_true = truth - control
            delta_pred = prediction - control
            fc = vector_metrics(delta_true, delta_pred)
            for metric in ("pcc", "r2", "rmse", "mae"):
                record[f"fc_{metric}"] = fc[metric]

            context_ref = references.context_for(row)
            if context_ref is not None:
                record["context_reference_matched"] = True
                context_metrics = vector_metrics(delta_true - context_ref, delta_pred - context_ref)
                for metric in ("pcc", "r2", "rmse", "mae"):
                    record[f"context_residual_{metric}"] = context_metrics[metric]

            drug_ref = references.drug_for(row)
            if drug_ref is not None:
                record["drug_reference_matched"] = True
                drug_metrics = vector_metrics(delta_true - drug_ref, delta_pred - drug_ref)
                for metric in ("pcc", "r2", "rmse", "mae"):
                    record[f"drug_residual_{metric}"] = drug_metrics[metric]

            dep_truth_chunks.append(delta_true)
            dep_pred_chunks.append(delta_pred)
            sample_slices.append(slice(offset, offset + delta_true.size))
            offset += delta_true.size
        records.append(record)
    sample_frame = pd.DataFrame(records).set_index("sample_ID", drop=False)
    if dep_truth_chunks:
        dep_truth = np.concatenate(dep_truth_chunks)
        dep_pred = np.concatenate(dep_pred_chunks)
    else:
        dep_truth = np.asarray([], dtype=np.float64)
        dep_pred = np.asarray([], dtype=np.float64)
    return sample_frame, dep_truth, dep_pred, sample_slices


def evaluate_validation(data: ScoringData, config: dict[str, Any]) -> EvaluationArtifacts:
    matching = config["control_matching"]
    control_labels = set(map(str, config["control_labels"]))
    all_control_ids = data.metadata.index[
        data.metadata["perturbation_no_concentration"].astype(str).isin(control_labels)
    ]
    controls = match_control_profiles(
        metadata=data.metadata,
        truth_log2=data.truth_log2,
        target_ids=data.target_ids,
        control_ids=all_control_ids,
        keys=list(matching["keys"]),
        required_equal_fields=list(matching.get("required_equal_fields", [])),
        aggregation=str(matching["aggregation"]),
    )
    references = fit_train_references(data.metadata, data.truth_log2, config)
    sample_frame, dep_truth, dep_pred, sample_slices = _build_sample_metrics(
        data, controls, references
    )

    aggregation = str(config["aggregation"]["per_sample"])
    normalization = str(config["aggregation"]["score_normalization"])
    validation_splits = list(map(str, config["validation_splits"]))
    by_split: dict[str, Any] = {}
    for split in validation_splits:
        split_frame = sample_frame.loc[sample_frame["split"].eq(split)]
        by_split[split] = {
            "absolute": _absolute_summary(split_frame, aggregation),
            "fc": _effect_summary(split_frame, "fc", aggregation),
        }

    absolute_all = _absolute_summary(sample_frame, aggregation)
    fc_all = _effect_summary(sample_frame, "fc", aggregation)
    chem_frame = sample_frame.loc[sample_frame["split"].eq("val_chem_only")]
    strain_frame = sample_frame.loc[sample_frame["split"].eq("val_strain_only")]
    context_summary = _effect_summary(chem_frame, "context_residual", aggregation)
    drug_summary = _effect_summary(strain_frame, "drug_residual", aggregation)

    dep_cfg = config["dep"]
    dep = dep_metrics(
        delta_true=dep_truth,
        delta_pred=dep_pred,
        threshold=float(dep_cfg["absolute_delta_gt"]),
        fixed_ks=[int(x) for x in dep_cfg["fixed_recall_ks"]],
        sample_slices=sample_slices,
        aggregation=aggregation,
    )

    internal_weights = config["module_internal_weights"]
    official_weights = {key: float(value) for key, value in config["module_weights"].items()}
    modules: dict[str, dict[str, Any]] = {}
    modules["m1"] = _module_record(
        "m1",
        official_weights["m1"],
        {
            "sample_pcc": float(absolute_all["sample_pcc"]["value"]),
            "sample_r2": float(absolute_all["sample_r2"]["value"]),
        },
        {
            "sample_pcc": float(absolute_all["sample_pcc"]["proxy_value"]),
            "sample_r2": float(absolute_all["sample_r2"]["proxy_value"]),
        },
        {key: float(value) for key, value in internal_weights["m1"].items()},
        normalization,
        {"target_samples": len(sample_frame)},
    )
    modules["m2"] = _module_record(
        "m2",
        official_weights["m2"],
        {"fc_pcc": float(fc_all["pcc"]["value"])},
        {"fc_pcc": float(fc_all["pcc"]["proxy_value"])},
        {key: float(value) for key, value in internal_weights["m2"].items()},
        normalization,
        {
            "target_samples": len(sample_frame),
            "matched_control_samples": int(sample_frame["control_matched"].sum()),
        },
    )
    modules["m3"] = _module_record(
        "m3",
        official_weights["m3"],
        {"context_residual_pcc": float(context_summary["pcc"]["value"])},
        {"context_residual_pcc": float(context_summary["pcc"]["proxy_value"])},
        {key: float(value) for key, value in internal_weights["m3"].items()},
        normalization,
        {
            "target_samples": len(chem_frame),
            "matched_reference_samples": int(chem_frame["context_reference_matched"].sum()),
        },
    )
    modules["m4"] = _module_record(
        "m4",
        official_weights["m4"],
        {"drug_residual_pcc": float(drug_summary["pcc"]["value"])},
        {"drug_residual_pcc": float(drug_summary["pcc"]["proxy_value"])},
        {key: float(value) for key, value in internal_weights["m4"].items()},
        normalization,
        {
            "target_samples": len(strain_frame),
            "matched_reference_samples": int(strain_frame["drug_reference_matched"].sum()),
        },
    )

    m5_values: dict[str, float] = {}
    m5_proxy_values: dict[str, float] = {}
    m5_coverage: dict[str, Any] = {}
    for split, key in (("val_both", "both"), ("val_time", "time")):
        absolute = by_split[split]["absolute"]
        fc = by_split[split]["fc"]
        m5_values[f"{key}_sample_pcc"] = float(absolute["sample_pcc"]["value"])
        m5_values[f"{key}_sample_r2"] = float(absolute["sample_r2"]["value"])
        m5_values[f"{key}_fc_pcc"] = float(fc["pcc"]["value"])
        m5_proxy_values[f"{key}_sample_pcc"] = float(
            absolute["sample_pcc"]["proxy_value"]
        )
        m5_proxy_values[f"{key}_sample_r2"] = float(
            absolute["sample_r2"]["proxy_value"]
        )
        m5_proxy_values[f"{key}_fc_pcc"] = float(fc["pcc"]["proxy_value"])
        m5_coverage[split] = {
            "target_samples": int(absolute["samples"]),
            "fc_evaluated_samples": int(fc["pcc"]["evaluated_samples"]),
            "interpretation": "temporal_interpolation" if split == "val_time" else "dual_unseen",
        }
    modules["m5"] = _module_record(
        "m5",
        official_weights["m5"],
        m5_values,
        m5_proxy_values,
        {key: float(value) for key, value in internal_weights["m5"].items()},
        normalization,
        m5_coverage,
    )
    m6_values = {
        "direction_accuracy": float(dep["direction_accuracy"]),
        "high_effect_pcc": float(dep["high_effect_pcc_per_sample"]),
        "f1": float(dep["f1"]),
        "average_precision": float(dep["average_precision"]),
    }
    m6_proxy_values = {
        **m6_values,
        "high_effect_pcc": float(dep["high_effect_pcc_per_sample_proxy"]),
    }
    modules["m6"] = _module_record(
        "m6",
        official_weights["m6"],
        m6_values,
        m6_proxy_values,
        {key: float(value) for key, value in internal_weights["m6"].items()},
        normalization,
        {
            "control_matched_samples": int(sample_frame["control_matched"].sum()),
            "paired_values": int(dep["paired_values"]),
            "truth_high_effect_values": int(dep["truth_high_effect_values"]),
        },
    )

    weighted_proxy = float(
        sum(
            module["official_weight"] * module["normalized_score"]
            for module in modules.values()
        )
    )
    weighted_raw = float(
        sum(
            module["official_weight"]
            * (module["raw_score"] if np.isfinite(module["raw_score"]) else 0.0)
            for module in modules.values()
        )
    )
    control_match_rate = float(sample_frame["control_matched"].mean())
    context_rate = float(chem_frame["context_reference_matched"].mean()) if len(chem_frame) else 0.0
    drug_rate = float(strain_frame["drug_reference_matched"].mean()) if len(strain_frame) else 0.0
    mixed_controls = int(controls.details["mixed_control_labels"].sum())
    train_control_targets = int(controls.details["uses_train_control"].sum())
    validation_control_targets = int(controls.details["uses_validation_control"].sum())
    mixed_split_control_targets = int(
        (
            controls.details["uses_train_control"]
            & controls.details["uses_validation_control"]
        ).sum()
    )
    warnings = list(config.get("known_official_ambiguities", []))
    if control_match_rate < 1.0:
        warnings.append(f"Matched-control coverage is {control_match_rate:.3%}; unmatched samples score as undefined and zero in proxy components.")
    if context_rate < 1.0:
        warnings.append(f"Context-reference coverage in val_chem_only is {context_rate:.3%}.")
    if drug_rate < 1.0:
        warnings.append(f"Drug-reference coverage in val_strain_only is {drug_rate:.3%}.")
    if mixed_controls:
        warnings.append(f"{mixed_controls} targets matched both Water and DMSO; configured pooling policy was used.")

    result = {
        "schema_version": "1.1",
        "scorer_name": "goai-official-faithful-provisional",
        "scorer_version": __version__,
        "profile": config["profile"],
        "score_status": "provisional_due_to_unpublished_internal_aggregation",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "official_sources": config["official_sources"],
        "boundary": {
            "reference_fit_split": "train",
            "scored_splits": validation_splits,
            "scored_sample_role": "treatment_only",
            "validation_truth_used_for_scoring_only": True,
            "separate_evaluation_truth_loaded": False,
            "test_truth_loaded": False,
            "matched_control_truth_scope": "released_train_validation_controls_used_only_after_prediction_freeze",
            "prediction_scale": "log2",
        },
        "data_contract": {
            "metadata_file": data.metadata_path.name,
            "proteome_file": data.proteome_path.name,
            "prediction_file": data.prediction_path.name,
            "train_samples": int(data.metadata["split_final"].eq("train").sum()),
            "validation_treatment_samples": len(data.target_ids),
            "raw_proteins": len(data.train_missing_rate),
            "retained_proteins": len(data.retained_proteins),
            "removed_proteins": int(len(data.train_missing_rate) - len(data.retained_proteins)),
            "train_missing_rate_rule": f"< {config['protein_filter']['train_missing_rate_lt']}",
            **data.hashes,
            "train_reference_sha256": references.sha256,
        },
        "assumptions": {
            "control_match_semantic_fields": matching["semantic_fields"],
            "control_match_columns": [*matching["keys"], *matching.get("required_equal_fields", [])],
            "multi_control_aggregation": matching["aggregation"],
            "control_solvent_policy": matching["solvent_policy"],
            "reference_aggregation": config["references"]["aggregation"],
            "reference_aggregation_unit": config["references"]["aggregation_unit"],
            "per_sample_aggregation": aggregation,
            "score_normalization": normalization,
            "undefined_policy": config["aggregation"]["undefined_policy"],
            "scored_sample_scope": "treatment_only_local_assumption",
            "module5_internal_weights": "configured_equal_local_assumption",
            "module6_internal_weights": "configured_equal_local_assumption",
            "module6_detection_axis": "pooled_sample_protein_pairs",
            "module6_high_effect_pcc_axis": f"per_sample_{aggregation}",
        },
        "provisional_weighted_proxy": weighted_proxy,
        "provisional_weighted_proxy_100": float(100.0 * weighted_proxy),
        "provisional_weighted_raw_unclipped": weighted_raw,
        "modules": modules,
        "metrics": {
            "absolute_all": absolute_all,
            "matched_control_fc_all": fc_all,
            "context_residual_val_chem_only": context_summary,
            "drug_residual_val_strain_only": drug_summary,
            "dep": dep,
            "by_split": by_split,
        },
        "coverage": {
            "control_match_rate": control_match_rate,
            "context_reference_rate_val_chem_only": context_rate,
            "drug_reference_rate_val_strain_only": drug_rate,
            "mixed_solvent_target_count": mixed_controls,
            "targets_using_train_control": train_control_targets,
            "targets_using_validation_control": validation_control_targets,
            "targets_using_both_train_and_validation_controls": mixed_split_control_targets,
        },
        "input_audit": {
            "observed_split_counts": {
                str(key): int(value)
                for key, value in data.metadata["split_final"].astype(str).value_counts().items()
            },
            "scored_target_counts": {
                str(key): int(value)
                for key, value in sample_frame["split"].value_counts().items()
            },
            "target_truth_finite_values": int(
                np.isfinite(data.truth_log2.loc[data.target_ids].to_numpy(copy=False)).sum()
            ),
            "target_truth_missing_values": int(
                np.isnan(data.truth_log2.loc[data.target_ids].to_numpy(copy=False)).sum()
            ),
            "prediction_finite_values": int(data.prediction_log2.size),
            "prediction_min_log2": float(data.prediction_log2.to_numpy(copy=False).min()),
            "prediction_max_log2": float(data.prediction_log2.to_numpy(copy=False).max()),
            "prediction_scale_validation": "contract_and_provenance_only_numeric_scale_is_not_uniquely_inferable",
        },
        "train_reference": references.manifest(),
        "warnings": warnings,
    }
    return EvaluationArtifacts(
        result=result,
        sample_metrics=sample_frame,
        control_matches=controls.details,
        reference_manifest=references.manifest(),
    )
