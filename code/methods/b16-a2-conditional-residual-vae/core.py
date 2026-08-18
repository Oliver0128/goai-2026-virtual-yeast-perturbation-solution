from __future__ import annotations

import hashlib
import importlib.util
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
E7_CORE_PATH = HERE.parent / "b10-a2-film-cross-mlp" / "core.py"
SPEC = importlib.util.spec_from_file_location("b16_a2_frozen_e7_core", E7_CORE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load frozen E7 core: {E7_CORE_PATH}")
E7_CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = E7_CORE
SPEC.loader.exec_module(E7_CORE)

FeatureBundle = E7_CORE.FeatureBundle
StructureTable = E7_CORE.StructureTable
DualInputBundle = E7_CORE.DualInputBundle
FilmCrossMLP = E7_CORE.FilmCrossMLP
set_reproducible_seed = E7_CORE.set_reproducible_seed
load_structure_table = E7_CORE.load_structure_table
encode_metadata = E7_CORE.encode_metadata
encode_structure = E7_CORE.encode_structure
split_dual_inputs = E7_CORE.split_dual_inputs
masked_mse = E7_CORE.masked_mse
parameter_count = E7_CORE.parameter_count


@dataclass(frozen=True)
class PairedDeltaBundle:
    eligible: np.ndarray
    target_delta: np.ndarray
    target_mask: np.ndarray
    control_drug_options: np.ndarray
    control_weights: np.ndarray
    manifest: dict[str, Any]


def frozen_e7_core_sha256() -> str:
    return hashlib.sha256(E7_CORE_PATH.read_bytes()).hexdigest()


def build_train_only_paired_delta(
    fit_metadata: pd.DataFrame,
    fit_truth: pd.DataFrame,
    fit_drug: np.ndarray,
    control_keys: tuple[str, ...],
    control_labels: tuple[str, ...],
    compound_column: str,
) -> PairedDeltaBundle:
    if not fit_metadata.index.equals(fit_truth.index):
        raise ValueError("fit metadata and truth axes must match")
    if len(fit_metadata) != len(fit_drug):
        raise ValueError("fit drug rows must match fit metadata")
    labels = fit_metadata[compound_column].astype(str)
    control_set = set(control_labels)
    control_positions = np.flatnonzero(labels.isin(control_set).to_numpy())
    treatment_positions = np.flatnonzero(~labels.isin(control_set).to_numpy())
    groups: dict[tuple[str, ...], list[int]] = {}
    for position in control_positions:
        row = fit_metadata.iloc[position]
        key = tuple(str(row[column]) for column in control_keys)
        groups.setdefault(key, []).append(int(position))

    row_count = len(fit_metadata)
    protein_count = fit_truth.shape[1]
    eligible = np.zeros(row_count, dtype=bool)
    target_delta = np.zeros((row_count, protein_count), dtype=np.float32)
    target_mask = np.zeros((row_count, protein_count), dtype=np.float32)
    control_drug_options = np.zeros(
        (row_count, len(control_labels), fit_drug.shape[1]), dtype=np.float32
    )
    control_weights = np.zeros((row_count, len(control_labels)), dtype=np.float32)
    matched_control_rows = 0
    mixed_label_treatments = 0
    truth_values = fit_truth.to_numpy(np.float64, copy=False)

    for position in treatment_positions:
        row = fit_metadata.iloc[position]
        key = tuple(str(row[column]) for column in control_keys)
        matches = groups.get(key, [])
        if not matches:
            continue
        eligible[position] = True
        matched_control_rows += len(matches)
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            control_mean = np.nanmean(truth_values[matches], axis=0)
        treatment_truth = truth_values[position]
        observed = np.isfinite(treatment_truth) & np.isfinite(control_mean)
        target_delta[position, observed] = (
            treatment_truth[observed] - control_mean[observed]
        ).astype(np.float32)
        target_mask[position, observed] = 1.0
        present_labels = 0
        for label_index, control_label in enumerate(control_labels):
            label_matches = [
                match for match in matches if labels.iloc[match] == control_label
            ]
            if not label_matches:
                continue
            present_labels += 1
            control_drug_options[position, label_index] = fit_drug[label_matches[0]]
            control_weights[position, label_index] = len(label_matches) / len(matches)
        if present_labels > 1:
            mixed_label_treatments += 1
    if np.any(eligible & ~np.isclose(control_weights.sum(axis=1), 1.0)):
        raise ValueError("Eligible paired rows must have control weights summing to one")
    return PairedDeltaBundle(
        eligible=eligible,
        target_delta=target_delta,
        target_mask=target_mask,
        control_drug_options=control_drug_options,
        control_weights=control_weights,
        manifest={
            "fit_scope": "non-QC split_final=train rows only",
            "matching_keys": list(control_keys),
            "control_labels": list(control_labels),
            "treatment_rows": len(treatment_positions),
            "matched_treatment_rows": int(eligible.sum()),
            "unmatched_treatment_rows": int(len(treatment_positions) - eligible.sum()),
            "control_rows": len(control_positions),
            "matched_control_row_uses": int(matched_control_rows),
            "mixed_control_label_treatments": int(mixed_label_treatments),
            "target": "treatment log2 truth minus exact-key train-control nanmean",
            "validation_or_test_rows_accepted": False,
        },
    )


def diagonal_gaussian_kl(
    q_mu: torch.Tensor,
    q_logvar: torch.Tensor,
    p_mu: torch.Tensor,
    p_logvar: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        p_logvar
        - q_logvar
        + (q_logvar.exp() + (q_mu - p_mu).square()) / p_logvar.exp()
        - 1.0
    )


def free_bits_kl(kl_per_dimension: torch.Tensor, free_bits: float) -> torch.Tensor:
    if kl_per_dimension.ndim != 2:
        raise ValueError("KL tensor must have shape (batch, latent_dim)")
    if free_bits < 0:
        raise ValueError("free_bits must be non-negative")
    return kl_per_dimension.mean(dim=0).clamp_min(free_bits).sum()


class ConditionalResidualVAE(nn.Module):
    def __init__(
        self,
        context_dim: int,
        output_dim: int,
        latent_dim: int,
        residual_hidden_dim: int,
        distribution_hidden_dim: int,
        decoder_hidden_dim: int,
        dropout: float,
        correction_bound: float,
    ) -> None:
        super().__init__()
        if latent_dim != 64:
            raise ValueError("B16-A2 contract requires latent_dim=64")
        if correction_bound <= 0:
            raise ValueError("correction_bound must be positive")
        self.context_dim = int(context_dim)
        self.output_dim = int(output_dim)
        self.latent_dim = int(latent_dim)
        self.correction_bound = float(correction_bound)
        self.residual_encoder = nn.Sequential(
            nn.Linear(output_dim, residual_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(residual_hidden_dim),
            nn.Dropout(dropout),
        )
        self.prior_trunk = nn.Sequential(
            nn.Linear(context_dim, distribution_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(distribution_hidden_dim),
        )
        self.prior_mu = nn.Linear(distribution_hidden_dim, latent_dim)
        self.prior_logvar = nn.Linear(distribution_hidden_dim, latent_dim)
        self.posterior_trunk = nn.Sequential(
            nn.Linear(
                context_dim + residual_hidden_dim + 1, distribution_hidden_dim
            ),
            nn.GELU(),
            nn.LayerNorm(distribution_hidden_dim),
        )
        self.posterior_mu = nn.Linear(distribution_hidden_dim, latent_dim)
        self.posterior_logvar = nn.Linear(distribution_hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(context_dim + latent_dim, decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_head = nn.Linear(decoder_hidden_dim, output_dim)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    @staticmethod
    def _bounded_logvar(logvar: torch.Tensor) -> torch.Tensor:
        return logvar.clamp(min=-8.0, max=4.0)

    def prior_parameters(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.prior_trunk(context)
        return self.prior_mu(hidden), self._bounded_logvar(self.prior_logvar(hidden))

    def posterior_parameters(
        self,
        context: torch.Tensor,
        target_residual: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked_residual = target_residual * target_mask
        observed_fraction = target_mask.mean(dim=1, keepdim=True)
        residual_hidden = self.residual_encoder(masked_residual)
        hidden = self.posterior_trunk(
            torch.cat([context, residual_hidden, observed_fraction], dim=1)
        )
        return self.posterior_mu(hidden), self._bounded_logvar(
            self.posterior_logvar(hidden)
        )

    def decode(self, context: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        raw = self.output_head(self.decoder(torch.cat([context, latent], dim=1)))
        return self.correction_bound * torch.tanh(raw)

    def forward_prior_mean(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        p_mu, p_logvar = self.prior_parameters(context)
        correction = self.decode(context, p_mu)
        return correction, {"p_mu": p_mu, "p_logvar": p_logvar}

    def forward_posterior(
        self,
        context: torch.Tensor,
        target_residual: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        p_mu, p_logvar = self.prior_parameters(context)
        q_mu, q_logvar = self.posterior_parameters(
            context, target_residual, target_mask
        )
        epsilon = torch.randn_like(q_mu)
        latent = q_mu + (0.5 * q_logvar).exp() * epsilon
        correction = self.decode(context, latent)
        return correction, {
            "p_mu": p_mu,
            "p_logvar": p_logvar,
            "q_mu": q_mu,
            "q_logvar": q_logvar,
            "latent": latent,
        }


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
