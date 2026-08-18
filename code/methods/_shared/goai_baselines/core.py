from __future__ import annotations

import csv
import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SAMPLE_ID = "sample_ID"
TRAIN_SPLIT = "train"
VALIDATION_SPLITS = ("val_chem_only", "val_strain_only", "val_both", "val_time")
CONTROL_LABELS = ("Water", "DMSO")
QC_LABEL = "Quality Control"
QC_PERT_ID = "48"
CONTROL_KEYS = (
    "data_source",
    "Strains",
    "Medium",
    "Temperature",
    "pert_time",
    "pert_time_unit",
    "instrument",
    "Yeast_cell_plate",
)


@dataclass(frozen=True)
class BaselineData:
    metadata: pd.DataFrame
    truth_log2: pd.DataFrame
    retained_proteins: pd.Index
    train_ids: pd.Index
    target_ids: pd.Index
    train_mean: pd.Series
    hashes: dict[str, str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_strings(values: pd.Index | list[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    if len(header) != len(set(header)):
        raise ValueError(f"Duplicate CSV columns in {path}")
    return header


def treatment_mask(metadata: pd.DataFrame) -> pd.Series:
    labels = metadata["perturbation_no_concentration"].astype(str)
    pert_ids = metadata["pert_id"].astype(str).str.lstrip("#")
    return ~labels.isin({*CONTROL_LABELS, QC_LABEL}) & pert_ids.ne(QC_PERT_ID)


def load_baseline_data(
    metadata_path: Path,
    proteome_path: Path,
    missing_rate_threshold: float,
) -> BaselineData:
    if "test" in proteome_path.name.lower() and "train_val" not in proteome_path.name.lower():
        raise ValueError("Method runner refuses test proteome truth")
    metadata = pd.read_csv(metadata_path, dtype={SAMPLE_ID: "string"}).set_index(SAMPLE_ID)
    metadata.index = metadata.index.astype(str)
    if not metadata.index.is_unique:
        raise ValueError("Metadata sample_ID must be unique")
    required = {
        "split_final",
        "pert_id",
        "perturbation_no_concentration",
        *CONTROL_KEYS,
    }
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}")
    if metadata.loc[:, sorted(required)].isna().any().any():
        raise ValueError("Required metadata fields may not be missing")
    observed_splits = set(metadata["split_final"].astype(str))
    allowed_splits = {TRAIN_SPLIT, *VALIDATION_SPLITS}
    if observed_splits != allowed_splits:
        raise ValueError(f"Unexpected split_final values: {sorted(observed_splits)}")

    header = _csv_header(proteome_path)
    if SAMPLE_ID not in header:
        raise ValueError("Proteome CSV lacks sample_ID")
    dtypes = {column: np.float64 for column in header if column != SAMPLE_ID}
    dtypes[SAMPLE_ID] = "string"
    raw = pd.read_csv(proteome_path, index_col=SAMPLE_ID, dtype=dtypes)
    raw.index = raw.index.astype(str)
    if not raw.index.is_unique or set(raw.index) != set(metadata.index):
        raise ValueError("Proteome and metadata sample axes differ")
    raw = raw.loc[metadata.index]
    values = raw.to_numpy(copy=False)
    observed = ~np.isnan(values)
    if np.any(observed & (~np.isfinite(values) | (values <= 0))):
        raise ValueError("Observed proteome values must be finite-positive")

    train_mask = metadata["split_final"].eq(TRAIN_SPLIT)
    missing_rate = raw.loc[train_mask].isna().mean(axis=0)
    retained = missing_rate.index[missing_rate.lt(float(missing_rate_threshold))]
    with np.errstate(divide="ignore", invalid="ignore"):
        log2 = pd.DataFrame(
            np.log2(raw.loc[:, retained].to_numpy(dtype=np.float64, copy=False)),
            index=raw.index,
            columns=retained,
        )
    train_ids = metadata.index[train_mask]
    target_mask = metadata["split_final"].isin(VALIDATION_SPLITS) & treatment_mask(metadata)
    target_ids = metadata.index[target_mask]
    train_mean = log2.loc[train_ids].mean(axis=0)
    if train_mean.isna().any():
        raise ValueError("A retained protein has no finite train mean")
    return BaselineData(
        metadata=metadata,
        truth_log2=log2,
        retained_proteins=retained,
        train_ids=train_ids,
        target_ids=target_ids,
        train_mean=train_mean,
        hashes={
            "metadata_sha256": sha256_file(metadata_path),
            "proteome_sha256": sha256_file(proteome_path),
            "sample_axis_sha256": sha256_strings(target_ids),
            "protein_axis_sha256": sha256_strings(retained),
        },
    )


def repeat_profile(profile: pd.Series, target_ids: pd.Index) -> pd.DataFrame:
    values = np.broadcast_to(profile.to_numpy(dtype=np.float64), (len(target_ids), len(profile)))
    return pd.DataFrame(values.copy(), index=target_ids, columns=profile.index)


def group_profile(
    metadata: pd.DataFrame,
    truth: pd.DataFrame,
    fit_ids: pd.Index,
    target_ids: pd.Index,
    keys: list[str],
) -> pd.DataFrame:
    joined = metadata.loc[fit_ids, keys].copy()
    joined["__sample_id"] = fit_ids
    group_map: dict[tuple[str, ...], list[str]] = {}
    for sample_id, row in joined.iterrows():
        key = tuple(str(row[column]) for column in keys)
        group_map.setdefault(key, []).append(str(sample_id))
    rows = []
    for _, row in metadata.loc[target_ids, keys].iterrows():
        ids = group_map.get(tuple(str(row[column]) for column in keys), [])
        if ids:
            with warnings.catch_warnings(), np.errstate(invalid="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                rows.append(np.nanmean(truth.loc[ids].to_numpy(dtype=np.float64), axis=0))
        else:
            rows.append(np.full(truth.shape[1], np.nan))
    return pd.DataFrame(rows, index=target_ids, columns=truth.columns)


def hierarchical_context_prediction(
    data: BaselineData,
    levels: list[list[str]],
    fit_ids: pd.Index | None = None,
) -> tuple[pd.DataFrame, list[int]]:
    if fit_ids is None:
        fit_ids = data.train_ids[
            data.metadata.loc[data.train_ids, "perturbation_no_concentration"].ne(QC_LABEL)
        ]
    prediction = pd.DataFrame(np.nan, index=data.target_ids, columns=data.retained_proteins)
    sample_level = np.full(len(data.target_ids), -1, dtype=int)
    for level_index, keys in enumerate(levels):
        if not keys:
            candidate = repeat_profile(data.train_mean, data.target_ids)
        else:
            candidate = group_profile(data.metadata, data.truth_log2, fit_ids, data.target_ids, keys)
        before = prediction.isna().all(axis=1).to_numpy()
        prediction = prediction.fillna(candidate)
        after = prediction.notna().any(axis=1).to_numpy()
        sample_level[(sample_level < 0) & before & after] = level_index
    prediction = prediction.fillna(data.train_mean)
    return prediction, sample_level.tolist()


def design_matrix(
    train_metadata: pd.DataFrame,
    target_metadata: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_parts = [np.ones((len(train_metadata), 1), dtype=np.float64)]
    target_parts = [np.ones((len(target_metadata), 1), dtype=np.float64)]
    feature_names = ["intercept"]
    for column in categorical:
        train_values = train_metadata[column].astype(str)
        target_values = target_metadata[column].astype(str)
        categories = sorted(train_values.unique().tolist())
        for category in categories:
            train_parts.append(train_values.eq(category).to_numpy(dtype=np.float64)[:, None])
            target_parts.append(target_values.eq(category).to_numpy(dtype=np.float64)[:, None])
            feature_names.append(f"{column}={category}")
    numeric_stats = {}
    for column in numeric:
        train_values = pd.to_numeric(train_metadata[column], errors="raise").to_numpy(np.float64)
        target_values = pd.to_numeric(target_metadata[column], errors="raise").to_numpy(np.float64)
        mean = float(train_values.mean())
        scale = float(train_values.std()) or 1.0
        train_parts.append(((train_values - mean) / scale)[:, None])
        target_parts.append(((target_values - mean) / scale)[:, None])
        feature_names.append(column)
        numeric_stats[column] = {"mean": mean, "scale": scale}
    return (
        np.hstack(train_parts),
        np.hstack(target_parts),
        {"feature_count": len(feature_names), "feature_names": feature_names, "numeric_stats": numeric_stats},
    )


def multioutput_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Ridge inputs must be finite")
    penalty = np.eye(x.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def impute_targets(y: pd.DataFrame, fallback: pd.Series) -> tuple[np.ndarray, pd.Series]:
    means = y.mean(axis=0).fillna(fallback)
    return y.fillna(means).to_numpy(dtype=np.float64), means


def exact_control_profiles(
    data: BaselineData,
    target_ids: pd.Index,
    allowed_control_ids: pd.Index,
) -> tuple[pd.DataFrame, pd.Series]:
    profiles = group_profile(
        data.metadata,
        data.truth_log2,
        allowed_control_ids,
        target_ids,
        list(CONTROL_KEYS),
    )
    matched = profiles.notna().any(axis=1)
    return profiles, matched
