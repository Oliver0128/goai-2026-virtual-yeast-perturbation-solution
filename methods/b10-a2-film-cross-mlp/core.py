from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass(frozen=True)
class FeatureBundle:
    fit: np.ndarray
    target: np.ndarray
    manifest: dict[str, Any]


@dataclass(frozen=True)
class StructureTable:
    competition_names: np.ndarray
    pubchem_cids: np.ndarray
    morgan_bits: np.ndarray
    descriptor_names: np.ndarray
    descriptors_raw: np.ndarray


@dataclass(frozen=True)
class DualInputBundle:
    fit_condition: np.ndarray
    target_condition: np.ndarray
    fit_drug: np.ndarray
    target_drug: np.ndarray
    manifest: dict[str, Any]


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def load_structure_table(npz_path: Path, contract_path: Path) -> StructureTable:
    expected_keys = {
        "competition_names",
        "pubchem_cids",
        "morgan_bits",
        "descriptor_names",
        "descriptors_raw",
    }
    with np.load(npz_path, allow_pickle=False) as artifact:
        if set(artifact.files) != expected_keys:
            raise ValueError(
                f"Unexpected structure artifact keys: {sorted(artifact.files)}"
            )
        table = StructureTable(**{key: artifact[key].copy() for key in expected_keys})
    contract = pd.read_csv(contract_path)
    if len(contract) != len(table.competition_names):
        raise ValueError("Structure contract and artifact entity counts differ")
    if not contract["competition_name"].is_unique:
        raise ValueError("Structure contract compound axis must be unique")
    if (
        table.morgan_bits.shape != (len(contract), 2048)
        or table.morgan_bits.dtype != np.uint8
    ):
        raise ValueError(
            "Morgan fingerprint contract requires uint8 shape (entities, 2048)"
        )
    if table.descriptors_raw.shape != (len(contract), 5):
        raise ValueError("Descriptor contract requires shape (entities, 5)")
    if not np.isfinite(table.descriptors_raw).all():
        raise ValueError("Raw descriptors must be finite")
    if set(np.unique(table.morgan_bits)) > {0, 1}:
        raise ValueError("Morgan fingerprints must be binary")
    if (
        table.competition_names.tolist()
        != contract["competition_name"].astype(str).tolist()
    ):
        raise ValueError("Structure artifact compound axis differs from contract.csv")
    if table.pubchem_cids.tolist() != contract["pubchem_cid"].astype(int).tolist():
        raise ValueError("Structure artifact CID axis differs from contract.csv")
    if len(set(table.competition_names.tolist())) != len(table.competition_names):
        raise ValueError("Structure artifact compound axis must be unique")
    return table


def encode_metadata(
    fit_metadata: pd.DataFrame,
    target_metadata: pd.DataFrame,
    categorical_columns: list[str],
    log2_numeric_columns: list[str],
) -> FeatureBundle:
    fit_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    feature_names: list[str] = []
    categorical_manifest: dict[str, Any] = {}
    numeric_manifest: dict[str, Any] = {}
    feature_slices: dict[str, list[int]] = {}

    for column in categorical_columns:
        start = len(feature_names)
        fit_values = fit_metadata[column].astype(str)
        target_values = target_metadata[column].astype(str)
        categories = sorted(fit_values.unique().tolist())
        fit_parts.append(
            np.column_stack(
                [
                    fit_values.eq(category).to_numpy(dtype=np.float32)
                    for category in categories
                ]
            )
        )
        target_parts.append(
            np.column_stack(
                [
                    target_values.eq(category).to_numpy(dtype=np.float32)
                    for category in categories
                ]
            )
        )
        feature_names.extend(f"{column}={category}" for category in categories)
        feature_slices[column] = [start, len(feature_names)]
        unseen = sorted(set(target_values.unique()).difference(categories))
        categorical_manifest[column] = {
            "categories": categories,
            "target_unseen_categories": unseen,
            "target_rows_with_unseen": int(target_values.isin(unseen).sum()),
        }

    for column in log2_numeric_columns:
        fit_raw = pd.to_numeric(fit_metadata[column], errors="raise").to_numpy(
            np.float64
        )
        target_raw = pd.to_numeric(target_metadata[column], errors="raise").to_numpy(
            np.float64
        )
        if np.any(fit_raw <= 0) or np.any(target_raw <= 0):
            raise ValueError(f"{column} must be positive before log2 transform")
        fit_values = np.log2(fit_raw)
        target_values = np.log2(target_raw)
        mean = float(fit_values.mean())
        scale = float(fit_values.std()) or 1.0
        fit_parts.append(((fit_values - mean) / scale).astype(np.float32)[:, None])
        target_parts.append(
            ((target_values - mean) / scale).astype(np.float32)[:, None]
        )
        feature_names.append(f"log2({column})")
        numeric_manifest[column] = {
            "transform": "log2_zscore",
            "mean": mean,
            "scale": scale,
        }

    fit = np.hstack(fit_parts).astype(np.float32, copy=False)
    target = np.hstack(target_parts).astype(np.float32, copy=False)
    if not np.isfinite(fit).all() or not np.isfinite(target).all():
        raise ValueError("Encoded metadata must be finite")
    return FeatureBundle(
        fit=fit,
        target=target,
        manifest={
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "feature_slices": feature_slices,
            "categorical": categorical_manifest,
            "numeric": numeric_manifest,
            "unseen_policy": "all-zero block for categories absent from split_final=train",
        },
    )


def encode_structure(
    fit_metadata: pd.DataFrame,
    target_metadata: pd.DataFrame,
    table: StructureTable,
    compound_column: str,
) -> FeatureBundle:
    names = table.competition_names.astype(str).tolist()
    name_to_index = {name: index for index, name in enumerate(names)}
    if len(name_to_index) != len(names):
        raise ValueError("Structure entity names must be unique")

    fit_labels = fit_metadata[compound_column].astype(str)
    target_labels = target_metadata[compound_column].astype(str)
    fit_available = fit_labels.isin(name_to_index).to_numpy()
    if not fit_available.any():
        raise ValueError("No fit compound has an available structure")

    descriptor_fit_rows = np.vstack(
        [
            table.descriptors_raw[name_to_index[label]]
            for label in fit_labels[fit_available]
        ]
    ).astype(np.float64)
    descriptor_mean = descriptor_fit_rows.mean(axis=0)
    descriptor_scale = descriptor_fit_rows.std(axis=0)
    descriptor_scale[descriptor_scale == 0] = 1.0
    fit_seen_compounds = set(fit_labels[fit_available])

    def transform(labels: pd.Series) -> tuple[np.ndarray, int, int]:
        rows = np.zeros(
            (
                len(labels),
                table.morgan_bits.shape[1] + table.descriptors_raw.shape[1] + 2,
            ),
            dtype=np.float32,
        )
        available_count = 0
        unseen_count = 0
        for row_index, label in enumerate(labels):
            artifact_index = name_to_index.get(label)
            if artifact_index is not None:
                rows[row_index, : table.morgan_bits.shape[1]] = table.morgan_bits[
                    artifact_index
                ]
                rows[
                    row_index,
                    table.morgan_bits.shape[1] : table.morgan_bits.shape[1]
                    + table.descriptors_raw.shape[1],
                ] = (
                    table.descriptors_raw[artifact_index].astype(np.float64)
                    - descriptor_mean
                ) / descriptor_scale
                rows[row_index, -2] = 1.0
                available_count += 1
            if label in fit_seen_compounds:
                rows[row_index, -1] = 1.0
            else:
                unseen_count += 1
        return rows, available_count, unseen_count

    fit, fit_coverage, fit_unseen = transform(fit_labels)
    target, target_coverage, target_unseen = transform(target_labels)
    if not np.isfinite(fit).all() or not np.isfinite(target).all():
        raise ValueError("Encoded structure features must be finite")
    return FeatureBundle(
        fit=fit,
        target=target,
        manifest={
            "compound_column": compound_column,
            "artifact_entity_count": len(names),
            "fingerprint_dimension": int(table.morgan_bits.shape[1]),
            "descriptor_names": table.descriptor_names.astype(str).tolist(),
            "descriptor_mean": descriptor_mean.tolist(),
            "descriptor_scale": descriptor_scale.tolist(),
            "descriptor_scaling_fit_scope": "fit rows only",
            "fit_rows": len(fit),
            "fit_structure_coverage_rows": fit_coverage,
            "fit_unseen_identity_rows": fit_unseen,
            "target_rows": len(target),
            "target_structure_coverage_rows": target_coverage,
            "target_unseen_identity_rows": target_unseen,
            "mask_columns": ["structure_available", "identity_seen_in_fit"],
            "missing_structure_policy": "zero fingerprint and descriptor plus structure_available=0",
            "unseen_identity_policy": "public structure retained plus identity_seen_in_fit=0",
        },
    )


def split_dual_inputs(
    metadata: FeatureBundle,
    structure: FeatureBundle,
    compound_column: str,
) -> DualInputBundle:
    compound_slice = metadata.manifest["feature_slices"][compound_column]
    start, end = int(compound_slice[0]), int(compound_slice[1])
    keep = np.r_[np.arange(start), np.arange(end, metadata.fit.shape[1])]
    fit_condition = metadata.fit[:, keep].astype(np.float32, copy=False)
    target_condition = metadata.target[:, keep].astype(np.float32, copy=False)
    fit_identity = metadata.fit[:, start:end]
    target_identity = metadata.target[:, start:end]
    fit_drug = np.hstack([fit_identity, structure.fit]).astype(np.float32, copy=False)
    target_drug = np.hstack([target_identity, structure.target]).astype(
        np.float32, copy=False
    )
    condition_names = (
        metadata.manifest["feature_names"][:start]
        + metadata.manifest["feature_names"][end:]
    )
    if any(name.startswith(f"{compound_column}=") for name in condition_names):
        raise ValueError(
            "Condition encoder input must exclude compound one-hot features"
        )
    for values in (fit_condition, target_condition, fit_drug, target_drug):
        if not np.isfinite(values).all():
            raise ValueError("Dual-encoder inputs must be finite")
    return DualInputBundle(
        fit_condition=fit_condition,
        target_condition=target_condition,
        fit_drug=fit_drug,
        target_drug=target_drug,
        manifest={
            "condition_feature_count": int(fit_condition.shape[1]),
            "condition_feature_names": condition_names,
            "condition_excludes_compound_identity": True,
            "drug_feature_count": int(fit_drug.shape[1]),
            "compound_identity_dimension": end - start,
            "drug_layout": [
                "train-vocabulary compound one-hot",
                "Morgan-2048",
                "five train-standardized RDKit descriptors",
                "structure_available",
                "identity_seen_in_fit",
            ],
            "drug_structure_offset": end - start,
            "drug_seen_mask_index": int(fit_drug.shape[1] - 1),
        },
    )


def masked_mse(
    prediction: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    observed = mask.sum()
    if observed.item() <= 0:
        raise ValueError("A batch must contain at least one observed target")
    return (((prediction - truth) ** 2) * mask).sum() / observed


def apply_compound_identity_dropout(
    drug: torch.Tensor,
    compound_identity_dim: int,
    seen_mask_index: int,
    probability: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("identity dropout probability must be in [0, 1]")
    rows = torch.rand(drug.shape[0], generator=generator).lt(probability)
    if not bool(rows.any()):
        return drug, rows
    dropped = drug.clone()
    dropped[rows, :compound_identity_dim] = 0.0
    dropped[rows, seen_mask_index] = 0.0
    return dropped, rows


def _encoder(
    input_dim: int, hidden_dim: int, latent_dim: int, dropout: float
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, latent_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class ResidualFusionBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.branch(x))


class FilmCrossMLP(nn.Module):
    def __init__(
        self,
        condition_dim: int,
        drug_dim: int,
        output_dim: int,
        encoder_hidden_dim: int,
        latent_dim: int,
        fusion_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.condition_encoder = _encoder(
            condition_dim, encoder_hidden_dim, latent_dim, dropout
        )
        self.drug_encoder = _encoder(drug_dim, encoder_hidden_dim, latent_dim, dropout)
        self.film_gamma = nn.Linear(latent_dim, latent_dim)
        self.film_beta = nn.Linear(latent_dim, latent_dim)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)
        self.fusion_stem = nn.Sequential(
            nn.Linear(3 * latent_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fusion_residual = ResidualFusionBlock(fusion_dim, dropout)
        self.output_head = nn.Linear(fusion_dim, output_dim)

    def forward(
        self,
        condition: torch.Tensor,
        drug: torch.Tensor,
        return_audit: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        condition_latent = self.condition_encoder(condition)
        drug_latent = self.drug_encoder(drug)
        gamma = self.film_gamma(drug_latent)
        beta = self.film_beta(drug_latent)
        condition_film = (1.0 + gamma) * condition_latent + beta
        fusion_input = torch.cat([condition_latent, drug_latent, condition_film], dim=1)
        fusion = self.fusion_residual(self.fusion_stem(fusion_input))
        residual = self.output_head(fusion)
        if return_audit:
            return residual, {
                "condition_latent": condition_latent,
                "drug_latent": drug_latent,
                "gamma": gamma,
                "beta": beta,
                "condition_film": condition_film,
                "fusion": fusion,
            }
        return residual


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
