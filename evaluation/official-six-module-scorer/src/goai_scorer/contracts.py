from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SAMPLE_ID = "sample_ID"
TRAIN_SPLIT = "train"
VALIDATION_SPLITS = ("val_chem_only", "val_strain_only", "val_both", "val_time")
EXPECTED_INTERNAL_COMPONENTS = {
    "m1": {"sample_pcc", "sample_r2"},
    "m2": {"fc_pcc"},
    "m3": {"context_residual_pcc"},
    "m4": {"drug_residual_pcc"},
    "m5": {
        "both_sample_pcc",
        "both_sample_r2",
        "both_fc_pcc",
        "time_sample_pcc",
        "time_sample_r2",
        "time_fc_pcc",
    },
    "m6": {"direction_accuracy", "high_effect_pcc", "f1", "average_precision"},
}


@dataclass(frozen=True)
class ScoringData:
    metadata: pd.DataFrame
    truth_log2: pd.DataFrame
    prediction_log2: pd.DataFrame
    train_missing_rate: pd.Series
    retained_proteins: pd.Index
    target_ids: pd.Index
    metadata_path: Path
    proteome_path: Path
    prediction_path: Path
    hashes: dict[str, str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_strings(values: pd.Index | list[str]) -> str:
    payload = "\n".join(map(str, values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "profile",
        "validation_splits",
        "input_contract",
        "control_labels",
        "quality_control",
        "protein_filter",
        "control_matching",
        "references",
        "aggregation",
        "module_weights",
        "module_internal_weights",
        "dep",
        "official_sources",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Scorer config is missing keys: {missing}")
    weights = config["module_weights"]
    if set(weights) != {"m1", "m2", "m3", "m4", "m5", "m6"}:
        raise ValueError("module_weights must contain exactly m1..m6")
    if not np.isclose(sum(float(x) for x in weights.values()), 1.0):
        raise ValueError("module_weights must sum to 1")
    official_weights = {"m1": 0.20, "m2": 0.25, "m3": 0.20, "m4": 0.20, "m5": 0.10, "m6": 0.05}
    if any(not np.isclose(float(weights[key]), value) for key, value in official_weights.items()):
        raise ValueError("Current handbook profile must use official weights 20/25/20/20/10/5")
    if config["protein_filter"].get("fit_split") != TRAIN_SPLIT:
        raise ValueError("Protein filtering must be fit on split_final=train")
    if config["references"].get("fit_split") != TRAIN_SPLIT:
        raise ValueError("Reference statistics must be fit on split_final=train")
    configured_splits = list(map(str, config["validation_splits"]))
    if len(configured_splits) != len(VALIDATION_SPLITS) or set(configured_splits) != set(
        VALIDATION_SPLITS
    ):
        raise ValueError(f"Current handbook profile must score each split once: {list(VALIDATION_SPLITS)}")

    input_contract = config["input_contract"]
    expected_split_counts = input_contract.get("expected_split_counts", {})
    expected_split_keys = {TRAIN_SPLIT, *VALIDATION_SPLITS}
    if set(map(str, expected_split_counts)) != expected_split_keys:
        raise ValueError(
            f"input_contract.expected_split_counts must contain exactly {sorted(expected_split_keys)}"
        )
    for value in expected_split_counts.values():
        numeric = float(value)
        if not np.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
            raise ValueError("Expected split counts must be positive integers")
    if int(input_contract.get("expected_raw_proteins", 0)) <= 0:
        raise ValueError("input_contract.expected_raw_proteins must be positive")
    for key in ("expected_metadata_sha256", "expected_proteome_sha256"):
        value = input_contract.get(key)
        if value is not None and (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value.lower())
        ):
            raise ValueError(f"{key} must be null or a 64-character SHA-256 hex digest")

    internal = config["module_internal_weights"]
    if set(internal) != set(EXPECTED_INTERNAL_COMPONENTS):
        raise ValueError("module_internal_weights must contain exactly m1..m6")
    for module_id, expected_components in EXPECTED_INTERNAL_COMPONENTS.items():
        component_weights = internal[module_id]
        if set(component_weights) != expected_components:
            raise ValueError(
                f"Internal components for {module_id} must be exactly {sorted(expected_components)}"
            )
        numeric_weights = np.asarray(list(component_weights.values()), dtype=np.float64)
        if not np.isfinite(numeric_weights).all() or np.any(numeric_weights < 0):
            raise ValueError(f"Internal weights for {module_id} must be finite and non-negative")
        if not np.isclose(float(numeric_weights.sum()), 1.0):
            raise ValueError(f"Internal weights for {module_id} must sum to 1")

    if config["control_matching"].get("aggregation") not in {"mean", "median"}:
        raise ValueError("control_matching.aggregation must be mean or median")
    if config["control_matching"].get("solvent_policy") != (
        "pool_all_exactly_matched_water_and_dmso_and_report_conflicts"
    ):
        raise ValueError("Unsupported control solvent policy")
    if config["references"].get("unmatched_policy") != (
        "undefined_zero_in_proxy_no_global_fallback"
    ):
        raise ValueError("Unsupported reference unmatched policy")
    if config["references"].get("aggregation") != "mean":
        raise ValueError("The handbook reference definitions require arithmetic means")
    if config["references"].get("aggregation_unit") != "matched_train_treatment_sample":
        raise ValueError("Unsupported reference aggregation unit")
    if config["aggregation"].get("per_sample") not in {"mean", "median"}:
        raise ValueError("aggregation.per_sample must be mean or median")
    if not isinstance(config["aggregation"].get("allow_prediction_column_superset"), bool):
        raise TypeError("aggregation.allow_prediction_column_superset must be boolean")
    if config["aggregation"].get("score_normalization") not in {
        "clip_0_1",
        "correlation_to_unit",
        "identity",
    }:
        raise ValueError("Unsupported score normalization policy")
    if config["aggregation"].get("undefined_policy") != (
        "raw_finite_only_and_zero_undefined_samples_in_proxy_without_weight_renormalization"
    ):
        raise ValueError("Unsupported undefined-value policy")
    dep = config["dep"]
    if not np.isfinite(float(dep.get("absolute_delta_gt", np.nan))) or float(
        dep["absolute_delta_gt"]
    ) <= 0:
        raise ValueError("dep.absolute_delta_gt must be finite and positive")
    raw_fixed_ks = dep.get("fixed_recall_ks", [])
    if any(not float(value).is_integer() for value in raw_fixed_ks):
        raise ValueError("dep.fixed_recall_ks must be unique positive integers")
    fixed_ks = [int(value) for value in raw_fixed_ks]
    if not fixed_ks or len(fixed_ks) != len(set(fixed_ks)) or any(value <= 0 for value in fixed_ks):
        raise ValueError("dep.fixed_recall_ks must be unique positive integers")
    if dep.get("module_auprc_metric") != "average_precision":
        raise ValueError("This scorer profile supports average_precision as the M6 AUPRC component")
    if dep.get("recall_not_used_alone") is not True:
        raise ValueError("Current handbook profile must not use Recall alone")
    return config


def _validate_unique_index(frame: pd.DataFrame, name: str) -> None:
    if frame.index.hasnans:
        raise ValueError(f"{name} has missing {SAMPLE_ID}")
    if not frame.index.is_unique:
        duplicate = frame.index[frame.index.duplicated()].unique()[:5].tolist()
        raise ValueError(f"{name} has duplicate {SAMPLE_ID}: {duplicate}")


def _csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        try:
            header = next(csv.reader(handle))
        except StopIteration as exc:
            raise ValueError(f"CSV file is empty: {path.name}") from exc
    duplicates = sorted(column for column, count in Counter(header).items() if count > 1)
    if duplicates:
        raise ValueError(f"{path.name} has duplicate columns: {duplicates[:5]}")
    return header


def _check_expected_hash(actual: str, expected: str | None, label: str) -> None:
    if expected is not None and actual.lower() != expected.lower():
        raise ValueError(f"{label} SHA-256 does not match the frozen input contract")


def _guard_validation_truth_inputs(metadata_path: Path, proteome_path: Path) -> None:
    """Reject obvious evaluation-set truth paths.

    This scorer intentionally supports only the released train/validation bundle.
    It has no mode that reads a separate evaluation-set proteome.
    """

    for path in (metadata_path, proteome_path):
        lowered = path.name.lower()
        if "test" in lowered and "train_val" not in lowered:
            raise ValueError(
                f"Validation-only scorer rejected a non-train_val truth input: {path.name}"
            )


def treatment_mask(metadata: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    labels = metadata["perturbation_no_concentration"].astype(str)
    controls = set(map(str, config["control_labels"]))
    qc = config["quality_control"]
    qc_label = str(qc["label"])
    qc_id = str(qc["pert_id"]).lstrip("#")
    pert_ids = metadata["pert_id"].astype(str).str.lstrip("#")
    return ~labels.isin(controls | {qc_label}) & pert_ids.ne(qc_id)


def load_scoring_data(
    metadata_path: Path,
    proteome_path: Path,
    prediction_path: Path,
    config: dict[str, Any],
) -> ScoringData:
    metadata_path = metadata_path.resolve()
    proteome_path = proteome_path.resolve()
    prediction_path = prediction_path.resolve()
    _guard_validation_truth_inputs(metadata_path, proteome_path)

    metadata_sha256 = sha256_file(metadata_path)
    proteome_sha256 = sha256_file(proteome_path)
    prediction_sha256 = sha256_file(prediction_path)
    input_contract = config["input_contract"]
    _check_expected_hash(
        metadata_sha256,
        input_contract.get("expected_metadata_sha256"),
        "Metadata",
    )
    _check_expected_hash(
        proteome_sha256,
        input_contract.get("expected_proteome_sha256"),
        "Proteome",
    )

    metadata = pd.read_csv(metadata_path, dtype={SAMPLE_ID: "string"}).set_index(
        SAMPLE_ID, drop=True
    )
    metadata.index = metadata.index.astype(str)
    _validate_unique_index(metadata, metadata_path.name)
    required_metadata = {
        "split_final",
        "pert_id",
        "perturbation_no_concentration",
        *config["control_matching"]["keys"],
        *config["control_matching"].get("required_equal_fields", []),
        *config["references"]["context_keys"],
        config["references"]["drug_key"],
    }
    missing_metadata = sorted(required_metadata.difference(metadata.columns))
    if missing_metadata:
        raise ValueError(f"Metadata is missing required columns: {missing_metadata}")
    missing_values = metadata.loc[:, sorted(required_metadata)].isna().sum()
    missing_values = missing_values[missing_values.gt(0)]
    if len(missing_values):
        raise ValueError(f"Metadata has missing required values: {missing_values.to_dict()}")
    if int(metadata["split_final"].eq(TRAIN_SPLIT).sum()) == 0:
        raise ValueError("No split_final=train rows")
    observed_split_counts = {
        str(key): int(value)
        for key, value in metadata["split_final"].astype(str).value_counts().items()
    }
    expected_split_counts = {
        str(key): int(value)
        for key, value in input_contract["expected_split_counts"].items()
    }
    if observed_split_counts != expected_split_counts:
        unexpected = sorted(set(observed_split_counts).difference(expected_split_counts))
        if unexpected:
            raise ValueError(f"Metadata contains forbidden or unknown split_final values: {unexpected}")
        raise ValueError(
            "Metadata split counts do not match the frozen input contract: "
            f"observed={observed_split_counts}, expected={expected_split_counts}"
        )

    raw_header = _csv_header(proteome_path)
    if SAMPLE_ID not in raw_header:
        raise ValueError(f"Proteome file lacks {SAMPLE_ID}")
    raw_protein_count = len(raw_header) - 1
    expected_raw_proteins = int(input_contract["expected_raw_proteins"])
    if raw_protein_count != expected_raw_proteins:
        raise ValueError(
            f"Proteome feature count mismatch: observed={raw_protein_count}, "
            f"expected={expected_raw_proteins}"
        )
    raw_dtypes = {column: np.float64 for column in raw_header if column != SAMPLE_ID}
    raw_dtypes[SAMPLE_ID] = "string"
    raw = pd.read_csv(
        proteome_path,
        index_col=SAMPLE_ID,
        dtype=raw_dtypes,
    )
    raw.index = raw.index.astype(str)
    _validate_unique_index(raw, proteome_path.name)
    if set(raw.index) != set(metadata.index):
        missing = metadata.index.difference(raw.index)[:5].tolist()
        extra = raw.index.difference(metadata.index)[:5].tolist()
        raise ValueError(f"Metadata/proteome sample mismatch: missing={missing}, extra={extra}")
    raw = raw.loc[metadata.index]

    observed = raw.to_numpy(copy=False)
    nonmissing = ~np.isnan(observed)
    if np.any(nonmissing & ~np.isfinite(observed)):
        raise ValueError("Observed raw protein intensities may contain NaN but not +/-inf")
    finite = observed[nonmissing]
    if finite.size == 0 or np.any(finite <= 0):
        raise ValueError("Observed raw protein intensities must be finite-positive where present")

    train_rows = metadata["split_final"].eq(TRAIN_SPLIT)
    train_missing_rate = raw.loc[train_rows].isna().mean(axis=0)
    threshold = float(config["protein_filter"]["train_missing_rate_lt"])
    retained = train_missing_rate.index[train_missing_rate.lt(threshold)]
    if retained.empty:
        raise ValueError("Train-only protein filter removed every protein")

    with np.errstate(divide="ignore", invalid="ignore"):
        truth_values = np.log2(raw.loc[:, retained].to_numpy(dtype=np.float64, copy=False))
    truth = pd.DataFrame(truth_values, index=raw.index, columns=retained)

    validation_splits = set(map(str, config["validation_splits"]))
    unknown_splits = validation_splits.difference(set(metadata["split_final"].astype(str)))
    if unknown_splits:
        raise ValueError(f"Configured validation splits are absent: {sorted(unknown_splits)}")
    target = metadata["split_final"].astype(str).isin(validation_splits) & treatment_mask(metadata, config)
    target_ids = metadata.index[target]
    if target_ids.empty:
        raise ValueError("No validation treatment samples to score")

    pred_header = _csv_header(prediction_path)
    if SAMPLE_ID not in pred_header:
        raise ValueError(f"Prediction file lacks {SAMPLE_ID}")
    pred_dtypes = {column: np.float64 for column in pred_header if column != SAMPLE_ID}
    pred_dtypes[SAMPLE_ID] = "string"
    prediction = pd.read_csv(
        prediction_path,
        index_col=SAMPLE_ID,
        dtype=pred_dtypes,
    )
    prediction.index = prediction.index.astype(str)
    _validate_unique_index(prediction, prediction_path.name)
    unknown_sample_ids = prediction.index.difference(metadata.index)
    if len(unknown_sample_ids):
        raise ValueError(f"Prediction has unknown sample IDs: {unknown_sample_ids[:5].tolist()}")
    missing_target_ids = target_ids.difference(prediction.index)
    if len(missing_target_ids):
        raise ValueError(f"Prediction is missing validation targets: {missing_target_ids[:5].tolist()}")
    missing_proteins = retained.difference(prediction.columns)
    if len(missing_proteins):
        raise ValueError(f"Prediction is missing retained proteins: {missing_proteins[:5].tolist()}")

    unknown_protein_columns = prediction.columns.difference(raw.columns)
    if len(unknown_protein_columns):
        raise ValueError(
            f"Prediction has unknown protein columns: {unknown_protein_columns[:5].tolist()}"
        )
    allowed_extra_columns = bool(config["aggregation"].get("allow_prediction_column_superset", True))
    non_scored_official_columns = prediction.columns.difference(retained)
    if len(non_scored_official_columns) and not allowed_extra_columns:
        raise ValueError(
            "Prediction contains official but non-scored protein columns while column superset "
            f"is disabled: {non_scored_official_columns[:5].tolist()}"
        )
    prediction = prediction.loc[target_ids, retained]
    prediction_values = prediction.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(prediction_values).all():
        raise ValueError("Predictions must be finite for every retained protein and target sample")

    hashes = {
        "metadata_sha256": metadata_sha256,
        "proteome_sha256": proteome_sha256,
        "prediction_sha256": prediction_sha256,
        "sample_axis_sha256": sha256_strings(target_ids),
        "protein_axis_sha256": sha256_strings(retained),
    }
    return ScoringData(
        metadata=metadata,
        truth_log2=truth,
        prediction_log2=prediction,
        train_missing_rate=train_missing_rate,
        retained_proteins=retained,
        target_ids=target_ids,
        metadata_path=metadata_path,
        proteome_path=proteome_path,
        prediction_path=prediction_path,
        hashes=hashes,
    )
