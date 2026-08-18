from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .contracts import TRAIN_SPLIT, treatment_mask
from .controls import match_control_profiles


def _group_key(row: pd.Series, fields: list[str]) -> tuple[Any, ...]:
    return tuple(row[field] for field in fields)


def _hash_reference_map(
    context: dict[tuple[Any, ...], np.ndarray],
    drug: dict[tuple[Any, ...], np.ndarray],
    source_treatment_ids: pd.Index,
    source_control_ids: pd.Index,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"source_treatments\0")
    digest.update("\n".join(map(str, source_treatment_ids)).encode("utf-8"))
    digest.update(b"source_controls\0")
    digest.update("\n".join(map(str, source_control_ids)).encode("utf-8"))
    for prefix, mapping in (("context", context), ("drug", drug)):
        for key in sorted(mapping, key=lambda item: json.dumps(item, default=str, ensure_ascii=False)):
            digest.update(prefix.encode("utf-8"))
            digest.update(json.dumps(key, default=str, ensure_ascii=False).encode("utf-8"))
            array = np.asarray(mapping[key], dtype="<f8")
            finite = np.isfinite(array)
            digest.update(finite.astype(np.uint8).tobytes())
            digest.update(np.where(finite, array, 0.0).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainReferences:
    context: dict[tuple[Any, ...], np.ndarray]
    drug: dict[tuple[Any, ...], np.ndarray]
    context_keys: list[str]
    drug_key: str
    source_sample_ids: pd.Index
    source_control_sample_ids: pd.Index
    source_delta_count: int
    unmatched_train_treatments: int
    sha256: str

    def context_for(self, row: pd.Series) -> np.ndarray | None:
        return self.context.get(_group_key(row, self.context_keys))

    def drug_for(self, row: pd.Series) -> np.ndarray | None:
        return self.drug.get((row[self.drug_key],))

    def manifest(self) -> dict[str, Any]:
        return {
            "fit_split": TRAIN_SPLIT,
            "source_treatment_samples": len(self.source_sample_ids),
            "source_control_samples": len(self.source_control_sample_ids),
            "source_control_sample_ids_sha256": hashlib.sha256(
                "\n".join(map(str, self.source_control_sample_ids)).encode("utf-8")
            ).hexdigest(),
            "matched_source_deltas": int(self.source_delta_count),
            "unmatched_train_treatments": int(self.unmatched_train_treatments),
            "context_keys": self.context_keys,
            "drug_key": self.drug_key,
            "context_reference_groups": len(self.context),
            "drug_reference_groups": len(self.drug),
            "reference_sha256": self.sha256,
        }


def _mean_by_group(
    metadata: pd.DataFrame,
    deltas: pd.DataFrame,
    fields: list[str],
) -> dict[tuple[Any, ...], np.ndarray]:
    mapping: dict[tuple[Any, ...], list[np.ndarray]] = {}
    for sample_id, row in metadata.loc[deltas.index].iterrows():
        mapping.setdefault(_group_key(row, fields), []).append(
            deltas.loc[sample_id].to_numpy(dtype=np.float64, copy=False)
        )
    result: dict[tuple[Any, ...], np.ndarray] = {}
    for key, arrays in mapping.items():
        stacked = np.vstack(arrays)
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result[key] = np.nanmean(stacked, axis=0)
    return result


def fit_train_references(
    metadata: pd.DataFrame,
    truth_log2: pd.DataFrame,
    config: dict[str, Any],
) -> TrainReferences:
    train_rows = metadata["split_final"].eq(TRAIN_SPLIT)
    train_treatment_ids = metadata.index[train_rows & treatment_mask(metadata, config)]
    control_labels = set(map(str, config["control_labels"]))
    train_control_ids = metadata.index[
        train_rows
        & metadata["perturbation_no_concentration"].astype(str).isin(control_labels)
    ]
    matching = config["control_matching"]
    matched = match_control_profiles(
        metadata=metadata,
        truth_log2=truth_log2,
        target_ids=train_treatment_ids,
        control_ids=train_control_ids,
        keys=list(matching["keys"]),
        required_equal_fields=list(matching.get("required_equal_fields", [])),
        aggregation=str(matching["aggregation"]),
    )
    matched_ids = matched.details.index[matched.details["matched"].astype(bool)]
    used_control_ids = pd.Index(
        sorted(
            {
                sample_id
                for value in matched.details.loc[matched_ids, "control_sample_IDs"]
                for sample_id in str(value).split("|")
                if sample_id
            }
        )
    )
    deltas = truth_log2.loc[matched_ids] - matched.profiles.loc[matched_ids]
    reference_cfg = config["references"]
    context_keys = list(reference_cfg["context_keys"])
    drug_key = str(reference_cfg["drug_key"])
    context = _mean_by_group(metadata, deltas, context_keys)
    drug = _mean_by_group(metadata, deltas, [drug_key])
    digest = _hash_reference_map(
        context,
        drug,
        pd.Index(matched_ids),
        used_control_ids,
    )
    return TrainReferences(
        context=context,
        drug=drug,
        context_keys=context_keys,
        drug_key=drug_key,
        source_sample_ids=pd.Index(matched_ids),
        source_control_sample_ids=used_control_ids,
        source_delta_count=len(deltas),
        unmatched_train_treatments=int(len(train_treatment_ids) - len(deltas)),
        sha256=digest,
    )
