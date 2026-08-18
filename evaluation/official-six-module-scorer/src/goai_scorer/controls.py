from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ControlMatches:
    profiles: pd.DataFrame
    details: pd.DataFrame


def _metadata_key(row: pd.Series, fields: list[str]) -> tuple[Any, ...]:
    return tuple(row[field] for field in fields)


def match_control_profiles(
    metadata: pd.DataFrame,
    truth_log2: pd.DataFrame,
    target_ids: pd.Index,
    control_ids: pd.Index,
    keys: list[str],
    required_equal_fields: list[str],
    aggregation: str,
    control_label_column: str = "perturbation_no_concentration",
) -> ControlMatches:
    fields = list(keys)
    for field in required_equal_fields:
        if field not in fields:
            fields.append(field)
    if metadata.loc[target_ids, fields].isna().any().any():
        raise ValueError("Target metadata has missing control-matching fields")
    if metadata.loc[control_ids, fields].isna().any().any():
        raise ValueError("Control metadata has missing control-matching fields")

    control_groups: dict[tuple[Any, ...], list[str]] = {}
    for sample_id, row in metadata.loc[control_ids].iterrows():
        control_groups.setdefault(_metadata_key(row, fields), []).append(str(sample_id))

    profile_rows: list[np.ndarray] = []
    detail_rows: list[dict[str, Any]] = []
    protein_columns = truth_log2.columns
    for target_id, row in metadata.loc[target_ids].iterrows():
        candidates = control_groups.get(_metadata_key(row, fields), [])
        if candidates:
            values = truth_log2.loc[candidates].to_numpy(dtype=np.float64, copy=False)
            with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if aggregation == "mean":
                    profile = np.nanmean(values, axis=0)
                elif aggregation == "median":
                    profile = np.nanmedian(values, axis=0)
                else:
                    raise ValueError(f"Unknown control aggregation: {aggregation}")
            labels = sorted(
                metadata.loc[candidates, control_label_column].astype(str).unique().tolist()
            )
            splits = sorted(metadata.loc[candidates, "split_final"].astype(str).unique().tolist())
        else:
            profile = np.full(len(protein_columns), np.nan, dtype=np.float64)
            labels = []
            splits = []
        profile_rows.append(profile)
        detail_rows.append(
            {
                "sample_ID": str(target_id),
                "matched": bool(candidates),
                "candidate_count": len(candidates),
                "control_sample_IDs": "|".join(candidates),
                "control_labels": "|".join(labels),
                "control_splits": "|".join(splits),
                "mixed_control_labels": len(labels) > 1,
                "uses_train_control": "train" in splits,
                "uses_validation_control": any(split != "train" for split in splits),
                "aggregation": aggregation,
                "finite_control_proteins": int(np.isfinite(profile).sum()),
            }
        )
    return ControlMatches(
        profiles=pd.DataFrame(profile_rows, index=target_ids, columns=protein_columns),
        details=pd.DataFrame(detail_rows).set_index("sample_ID", drop=False),
    )
