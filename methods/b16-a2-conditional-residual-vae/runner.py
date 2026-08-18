from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from core import (
    ConditionalResidualVAE,
    FilmCrossMLP,
    build_train_only_paired_delta,
    diagonal_gaussian_kl,
    encode_metadata,
    encode_structure,
    free_bits_kl,
    frozen_e7_core_sha256,
    load_structure_table,
    masked_mse,
    parameter_count,
    set_reproducible_seed,
    split_dual_inputs,
    trainable_parameter_count,
)
from goai_baselines.core import (
    CONTROL_KEYS,
    CONTROL_LABELS,
    QC_LABEL,
    load_baseline_data,
    sha256_file,
    treatment_mask,
    write_json,
)

import wandb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--proteome", type=Path, required=True)
    parser.add_argument("--structure-npz", type=Path, required=True)
    parser.add_argument("--structure-contract", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--e7-model", type=Path, required=True)
    parser.add_argument("--e7-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--latent-audit", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default=None)
    parser.add_argument("--scorer-script", type=Path, required=True)
    parser.add_argument("--score-config", type=Path, required=True)
    parser.add_argument("--score-output", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-output", type=Path)
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    return device


def validate_config(config: dict[str, Any]) -> None:
    exact = {
        "method_id": "b16-a2-conditional-residual-vae",
        "seed": 42,
        "latent_dim": 64,
        "maximum_additional_parameters": 1_500_000,
    }
    for key, expected in exact.items():
        if config[key] != expected:
            raise ValueError(f"B16-A2 contract requires {key}={expected}")
    for key in (
        "correction_bound",
        "learning_rate",
        "lambda_prior_absolute",
        "lambda_delta",
        "kl_beta_max",
    ):
        if not np.isfinite(float(config[key])) or float(config[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if int(config["kl_warmup_epochs"]) < 1:
        raise ValueError("kl_warmup_epochs must be positive")
    if float(config["kl_free_bits_per_dim"]) < 0:
        raise ValueError("kl_free_bits_per_dim must be non-negative")


def shuffled_batches(
    indices: np.ndarray, batch_size: int, generator: torch.Generator
) -> list[np.ndarray]:
    order = torch.randperm(len(indices), generator=generator).numpy()
    return [
        indices[order[start : start + batch_size]]
        for start in range(0, len(indices), batch_size)
    ]


def load_frozen_e7(
    checkpoint_path: Path,
    condition_dim: int,
    drug_dim: int,
    output_dim: int,
    protein_axis: list[str],
    device: torch.device,
) -> tuple[FilmCrossMLP, torch.Tensor, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["method_id"] != "b10-a2-film-cross-mlp" or config["seed"] != 42:
        raise ValueError("Frozen bypass must be the seed-42 E7 checkpoint")
    if checkpoint["retained_proteins"] != protein_axis:
        raise ValueError("Frozen E7 checkpoint protein axis differs from current train-only axis")
    manifest = checkpoint["dual_input_manifest"]
    if int(manifest["condition_feature_count"]) != condition_dim:
        raise ValueError("Frozen E7 condition feature dimension mismatch")
    if int(manifest["drug_feature_count"]) != drug_dim:
        raise ValueError("Frozen E7 drug feature dimension mismatch")
    model = FilmCrossMLP(
        condition_dim=condition_dim,
        drug_dim=drug_dim,
        output_dim=output_dim,
        encoder_hidden_dim=int(config["encoder_hidden_dim"]),
        latent_dim=int(config["latent_dim"]),
        fusion_dim=int(config["fusion_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for value in model.parameters():
        value.requires_grad_(False)
    base = checkpoint["base_profile"].to(torch.float32).to(device)
    return model, base, checkpoint


def build_cvae(config: dict[str, Any], output_dim: int) -> ConditionalResidualVAE:
    return ConditionalResidualVAE(
        context_dim=256,
        output_dim=output_dim,
        latent_dim=int(config["latent_dim"]),
        residual_hidden_dim=int(config["residual_hidden_dim"]),
        distribution_hidden_dim=int(config["distribution_hidden_dim"]),
        decoder_hidden_dim=int(config["decoder_hidden_dim"]),
        dropout=float(config["dropout"]),
        correction_bound=float(config["correction_bound"]),
    )


def extract_e7_outputs(
    e7: FilmCrossMLP,
    condition: torch.Tensor,
    drug: torch.Tensor,
    base: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions: list[torch.Tensor] = []
    contexts: list[torch.Tensor] = []
    e7.eval()
    with torch.no_grad():
        for start in range(0, len(condition), batch_size):
            end = start + batch_size
            residual, audit = e7(
                condition[start:end].to(device),
                drug[start:end].to(device),
                return_audit=True,
            )
            predictions.append((base + residual).cpu())
            contexts.append(audit["fusion"].cpu())
    return torch.cat(predictions), torch.cat(contexts)


def extract_control_e7_outputs(
    e7: FilmCrossMLP,
    condition: torch.Tensor,
    control_drug_options: np.ndarray,
    control_weights: np.ndarray,
    base: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_count, option_count, _ = control_drug_options.shape
    output_dim = e7.output_head.out_features
    context_dim = e7.fusion_stem[0].out_features
    weighted_predictions = torch.zeros(row_count, output_dim, dtype=torch.float32)
    contexts = torch.zeros(row_count, option_count, context_dim, dtype=torch.float32)
    for option in range(option_count):
        active = np.flatnonzero(control_weights[:, option] > 0)
        for start in range(0, len(active), batch_size):
            positions = active[start : start + batch_size]
            with torch.no_grad():
                residual, audit = e7(
                    condition[positions].to(device),
                    torch.from_numpy(control_drug_options[positions, option]).to(device),
                    return_audit=True,
                )
                prediction = (base + residual).cpu()
            weights = torch.from_numpy(control_weights[positions, option : option + 1])
            weighted_predictions[positions] += weights * prediction
            contexts[positions, option] = audit["fusion"].cpu()
    return weighted_predictions, contexts


def evaluate_prior(
    model: ConditionalResidualVAE,
    context: torch.Tensor,
    e7_prediction: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    squared_error = 0.0
    observed = 0.0
    correction_abs_sum = 0.0
    correction_elements = 0
    correction_abs_max = 0.0
    saturation = 0
    with torch.no_grad():
        for start in range(0, len(context), batch_size):
            end = start + batch_size
            correction, _ = model.forward_prior_mean(context[start:end].to(device))
            prediction = e7_prediction[start:end].to(device) + correction
            truth_batch = truth[start:end].to(device)
            mask_batch = mask[start:end].to(device)
            squared_error += float(
                (((prediction - truth_batch).square()) * mask_batch).sum().item()
            )
            observed += float(mask_batch.sum().item())
            absolute = correction.abs()
            correction_abs_sum += float(absolute.sum().item())
            correction_elements += correction.numel()
            correction_abs_max = max(correction_abs_max, float(absolute.max().item()))
            saturation += int(
                (absolute >= 0.95 * model.correction_bound).sum().item()
            )
    mse = squared_error / observed
    return {
        "masked_mse": mse,
        "masked_rmse": mse**0.5,
        "correction_abs_mean": correction_abs_sum / correction_elements,
        "correction_abs_max": correction_abs_max,
        "correction_saturation_fraction": saturation / correction_elements,
    }


def weighted_control_correction(
    model: ConditionalResidualVAE,
    control_context: torch.Tensor,
    control_weights: torch.Tensor,
) -> torch.Tensor:
    correction = torch.zeros(
        (len(control_context), model.output_dim),
        dtype=control_context.dtype,
        device=control_context.device,
    )
    for option in range(control_context.shape[1]):
        option_correction, _ = model.forward_prior_mean(control_context[:, option])
        correction = correction + control_weights[:, option : option + 1] * option_correction
    return correction


def kl_beta(epoch: int, config: dict[str, Any]) -> float:
    fraction = min(epoch / int(config["kl_warmup_epochs"]), 1.0)
    return float(config["kl_beta_max"]) * fraction


def train_with_early_stopping(
    model: ConditionalResidualVAE,
    train_context: torch.Tensor,
    train_e7_prediction: torch.Tensor,
    train_truth: torch.Tensor,
    train_mask: torch.Tensor,
    paired_eligible: np.ndarray,
    paired_target_delta: torch.Tensor,
    paired_target_mask: torch.Tensor,
    paired_control_e7_prediction: torch.Tensor,
    paired_control_context: torch.Tensor,
    paired_control_weights: torch.Tensor,
    validation_context: torch.Tensor,
    validation_e7_prediction: torch.Tensor,
    validation_truth: torch.Tensor,
    validation_mask: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    wandb_run: Any,
) -> tuple[ConditionalResidualVAE, int, list[dict[str, float]]]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(int(config["seed"]))
    indices = np.arange(len(train_context))
    baseline_metrics = evaluate_prior(
        model,
        validation_context,
        validation_e7_prediction,
        validation_truth,
        validation_mask,
        int(config["evaluation_batch_size"]),
        device,
    )
    history: list[dict[str, float]] = [
        {
            "epoch": 0,
            "kl_beta": 0.0,
            "official_validation_masked_mse": baseline_metrics["masked_mse"],
            "official_validation_masked_rmse": baseline_metrics["masked_rmse"],
            "official_validation_correction_abs_mean": 0.0,
            "official_validation_correction_abs_max": 0.0,
            "official_validation_correction_saturation_fraction": 0.0,
            "selection_role": "exact frozen E7 dense bypass safety baseline",
        }
    ]
    wandb_run.log({f"selection/{key}": value for key, value in history[0].items() if key != "selection_role"})
    best_loss = baseline_metrics["masked_mse"]
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0

    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        aggregate = {
            "combined": 0.0,
            "absolute_posterior": 0.0,
            "absolute_prior": 0.0,
            "paired_delta_prior": 0.0,
            "kl_raw": 0.0,
            "kl_free_bits": 0.0,
            "batches": 0.0,
            "delta_batches": 0.0,
        }
        beta = kl_beta(epoch, config)
        for batch in shuffled_batches(indices, int(config["batch_size"]), generator):
            context = train_context[batch].to(device)
            e7_prediction = train_e7_prediction[batch].to(device)
            truth = train_truth[batch].to(device)
            mask = train_mask[batch].to(device)
            target_residual = (truth - e7_prediction) * mask
            optimizer.zero_grad(set_to_none=True)
            posterior_correction, posterior = model.forward_posterior(
                context, target_residual, mask
            )
            prior_correction, _ = model.forward_prior_mean(context)
            absolute_posterior = masked_mse(
                e7_prediction + posterior_correction, truth, mask
            )
            absolute_prior = masked_mse(e7_prediction + prior_correction, truth, mask)
            kl_dimensions = diagonal_gaussian_kl(
                posterior["q_mu"],
                posterior["q_logvar"],
                posterior["p_mu"],
                posterior["p_logvar"],
            )
            kl_raw = kl_dimensions.sum(dim=1).mean()
            kl_with_free_bits = free_bits_kl(
                kl_dimensions, float(config["kl_free_bits_per_dim"])
            )
            eligible_local = paired_eligible[batch]
            if bool(eligible_local.any()):
                selected = np.flatnonzero(eligible_local)
                global_selected = batch[selected]
                treatment_prediction = (
                    e7_prediction[selected] + prior_correction[selected]
                )
                control_context = paired_control_context[global_selected].to(device)
                control_weights = paired_control_weights[global_selected].to(device)
                control_prediction = paired_control_e7_prediction[
                    global_selected
                ].to(device) + weighted_control_correction(
                    model, control_context, control_weights
                )
                delta_loss = masked_mse(
                    treatment_prediction - control_prediction,
                    paired_target_delta[global_selected].to(device),
                    paired_target_mask[global_selected].to(device),
                )
                aggregate["paired_delta_prior"] += float(delta_loss.detach())
                aggregate["delta_batches"] += 1.0
            else:
                delta_loss = torch.zeros((), device=device)
            loss = (
                absolute_posterior
                + float(config["lambda_prior_absolute"]) * absolute_prior
                + float(config["lambda_delta"]) * delta_loss
                + beta * kl_with_free_bits
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite B16-A2 training loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip_norm"])
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("Non-finite B16-A2 gradient norm")
            optimizer.step()
            aggregate["combined"] += float(loss.detach())
            aggregate["absolute_posterior"] += float(absolute_posterior.detach())
            aggregate["absolute_prior"] += float(absolute_prior.detach())
            aggregate["kl_raw"] += float(kl_raw.detach())
            aggregate["kl_free_bits"] += float(kl_with_free_bits.detach())
            aggregate["batches"] += 1.0

        validation_metrics = evaluate_prior(
            model,
            validation_context,
            validation_e7_prediction,
            validation_truth,
            validation_mask,
            int(config["evaluation_batch_size"]),
            device,
        )
        batches = aggregate["batches"]
        row = {
            "epoch": epoch,
            "kl_beta": beta,
            "train_combined_loss_batch_mean": aggregate["combined"] / batches,
            "train_absolute_posterior_mse_batch_mean": aggregate[
                "absolute_posterior"
            ]
            / batches,
            "train_absolute_prior_mse_batch_mean": aggregate["absolute_prior"]
            / batches,
            "train_paired_delta_prior_mse_batch_mean": aggregate[
                "paired_delta_prior"
            ]
            / max(aggregate["delta_batches"], 1.0),
            "train_kl_raw_nats_per_sample_batch_mean": aggregate["kl_raw"] / batches,
            "train_kl_free_bits_objective_batch_mean": aggregate["kl_free_bits"]
            / batches,
            "official_validation_masked_mse": validation_metrics["masked_mse"],
            "official_validation_masked_rmse": validation_metrics["masked_rmse"],
            "official_validation_correction_abs_mean": validation_metrics[
                "correction_abs_mean"
            ],
            "official_validation_correction_abs_max": validation_metrics[
                "correction_abs_max"
            ],
            "official_validation_correction_saturation_fraction": validation_metrics[
                "correction_saturation_fraction"
            ],
        }
        history.append(row)
        wandb_run.log({f"selection/{key}": value for key, value in row.items()})
        print(
            f"early-stop epoch={epoch:03d} "
            f"post_mse={row['train_absolute_posterior_mse_batch_mean']:.6f} "
            f"prior_mse={row['train_absolute_prior_mse_batch_mean']:.6f} "
            f"delta_mse={row['train_paired_delta_prior_mse_batch_mean']:.6f} "
            f"kl={row['train_kl_raw_nats_per_sample_batch_mean']:.4f} "
            f"beta={beta:.6f} "
            f"val_rmse={validation_metrics['masked_rmse']:.6f} "
            f"corr={validation_metrics['correction_abs_mean']:.4f}",
            flush=True,
        )
        validation_loss = validation_metrics["masked_mse"]
        if validation_loss < best_loss - float(config["early_stopping_min_delta"]):
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["early_stopping_patience"]):
                break
    model.load_state_dict(best_state)
    return model, best_epoch, history


def latent_collapse_audit(
    model: ConditionalResidualVAE,
    train_context: torch.Tensor,
    train_e7_prediction: torch.Tensor,
    train_truth: torch.Tensor,
    train_mask: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    q_mu_rows: list[torch.Tensor] = []
    p_mu_rows: list[torch.Tensor] = []
    kl_rows: list[torch.Tensor] = []
    squared = {"e7": 0.0, "prior": 0.0, "posterior_mean": 0.0}
    observed = 0.0
    with torch.no_grad():
        for start in range(0, len(train_context), batch_size):
            end = start + batch_size
            context = train_context[start:end].to(device)
            e7_prediction = train_e7_prediction[start:end].to(device)
            truth = train_truth[start:end].to(device)
            mask = train_mask[start:end].to(device)
            residual = (truth - e7_prediction) * mask
            p_mu, p_logvar = model.prior_parameters(context)
            q_mu, q_logvar = model.posterior_parameters(context, residual, mask)
            prior_correction = model.decode(context, p_mu)
            posterior_correction = model.decode(context, q_mu)
            kl_dimensions = diagonal_gaussian_kl(q_mu, q_logvar, p_mu, p_logvar)
            q_mu_rows.append(q_mu.cpu())
            p_mu_rows.append(p_mu.cpu())
            kl_rows.append(kl_dimensions.cpu())
            squared["e7"] += float(((e7_prediction - truth).square() * mask).sum())
            squared["prior"] += float(
                ((e7_prediction + prior_correction - truth).square() * mask).sum()
            )
            squared["posterior_mean"] += float(
                ((e7_prediction + posterior_correction - truth).square() * mask).sum()
            )
            observed += float(mask.sum())
    q_mu_all = torch.cat(q_mu_rows)
    p_mu_all = torch.cat(p_mu_rows)
    kl_all = torch.cat(kl_rows)
    q_mu_variance = q_mu_all.var(dim=0, unbiased=False)
    active_units = int((q_mu_variance > 0.01).sum().item())
    raw_kl_per_sample = kl_all.sum(dim=1)
    raw_kl_mean = float(raw_kl_per_sample.mean().item())
    collapse_flag = bool(active_units < 5 or raw_kl_mean < 0.1)
    return {
        "schema_version": "1.0",
        "audit_scope": "non-QC split_final=train rows only",
        "validation_posterior_calls": 0,
        "test_posterior_calls": 0,
        "inference_distribution": "conditional prior mean only; no sampling",
        "latent_dim": model.latent_dim,
        "raw_kl_nats_per_sample_mean": raw_kl_mean,
        "raw_kl_nats_per_sample_median": float(raw_kl_per_sample.median().item()),
        "raw_kl_per_dimension_mean": float(kl_all.mean().item()),
        "active_units_q_mu_variance_gt_0p01": active_units,
        "q_mu_variance_mean": float(q_mu_variance.mean().item()),
        "q_mu_variance_max": float(q_mu_variance.max().item()),
        "prior_mu_variance_mean": float(
            p_mu_all.var(dim=0, unbiased=False).mean().item()
        ),
        "posterior_prior_mu_abs_gap_mean": float(
            (q_mu_all - p_mu_all).abs().mean().item()
        ),
        "train_masked_rmse": {
            name: (value / observed) ** 0.5 for name, value in squared.items()
        },
        "collapse_rule": "flag if active_units<5 or mean raw KL<0.1 nat/sample",
        "posterior_collapse_flag": collapse_flag,
    }


def predict_prior_mean(
    model: ConditionalResidualVAE,
    context: torch.Tensor,
    e7_prediction: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    model.eval()
    predictions: list[np.ndarray] = []
    correction_abs_sum = 0.0
    correction_count = 0
    correction_abs_max = 0.0
    saturated = 0
    with torch.no_grad():
        for start in range(0, len(context), batch_size):
            end = start + batch_size
            correction, _ = model.forward_prior_mean(context[start:end].to(device))
            prediction = e7_prediction[start:end].to(device) + correction
            predictions.append(prediction.cpu().numpy())
            absolute = correction.abs()
            correction_abs_sum += float(absolute.sum().item())
            correction_count += correction.numel()
            correction_abs_max = max(correction_abs_max, float(absolute.max().item()))
            saturated += int(
                (absolute >= 0.95 * model.correction_bound).sum().item()
            )
    return np.vstack(predictions).astype(np.float64), {
        "correction_abs_mean": correction_abs_sum / correction_count,
        "correction_abs_max": correction_abs_max,
        "correction_saturation_fraction": saturated / correction_count,
        "inference_distribution": "prior_mean",
        "posterior_calls": 0,
    }


def scenario_metrics(metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        split: {
            "log2_rmse": values["absolute"]["sample_rmse"]["value"],
            "delta_pcc": values["fc"]["pcc"]["value"],
        }
        for split, values in metrics["metrics"]["by_split"].items()
    }


def smoke_real_batch(
    model: ConditionalResidualVAE,
    train_context: torch.Tensor,
    train_e7_prediction: torch.Tensor,
    train_truth: torch.Tensor,
    train_mask: torch.Tensor,
    paired_eligible: np.ndarray,
    paired_target_delta: torch.Tensor,
    paired_target_mask: torch.Tensor,
    paired_control_e7_prediction: torch.Tensor,
    paired_control_context: torch.Tensor,
    paired_control_weights: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model = model.to(device)
    model.train()
    eligible_rows = np.flatnonzero(paired_eligible)
    rows = eligible_rows[: min(8, len(eligible_rows))]
    context = train_context[rows].to(device)
    e7_prediction = train_e7_prediction[rows].to(device)
    truth = train_truth[rows].to(device)
    mask = train_mask[rows].to(device)
    residual = (truth - e7_prediction) * mask
    correction, audit = model.forward_posterior(context, residual, mask)
    prior_correction, _ = model.forward_prior_mean(context)
    absolute = masked_mse(e7_prediction + correction, truth, mask)
    prior_absolute = masked_mse(e7_prediction + prior_correction, truth, mask)
    control_context = paired_control_context[rows].to(device)
    control_weights = paired_control_weights[rows].to(device)
    control_prediction = paired_control_e7_prediction[rows].to(
        device
    ) + weighted_control_correction(model, control_context, control_weights)
    delta_loss = masked_mse(
        e7_prediction + prior_correction - control_prediction,
        paired_target_delta[rows].to(device),
        paired_target_mask[rows].to(device),
    )
    kl_dimensions = diagonal_gaussian_kl(
        audit["q_mu"], audit["q_logvar"], audit["p_mu"], audit["p_logvar"]
    )
    kl = free_bits_kl(kl_dimensions, float(config["kl_free_bits_per_dim"]))
    loss = (
        absolute
        + float(config["lambda_prior_absolute"]) * prior_absolute
        + float(config["lambda_delta"]) * delta_loss
        + 0.001 * kl
    )
    loss.backward()
    gradient_finite = all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    return {
        "schema_version": "1.0",
        "real_rows": len(rows),
        "context_shape": list(context.shape),
        "protein_shape": list(truth.shape),
        "latent_shape": list(audit["q_mu"].shape),
        "absolute_mse": float(absolute.detach()),
        "prior_absolute_mse": float(prior_absolute.detach()),
        "paired_delta_prior_mse": float(delta_loss.detach()),
        "free_bits_kl": float(kl.detach()),
        "combined_loss": float(loss.detach()),
        "gradients_finite": gradient_finite,
        "additional_parameter_count": trainable_parameter_count(model),
        "additional_parameter_budget_pass": trainable_parameter_count(model)
        <= int(config["maximum_additional_parameters"]),
        "prior_mean_initial_correction_abs_max": float(
            prior_correction.detach().abs().max()
        ),
        "validation_posterior_calls": 0,
        "device": str(device),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    set_reproducible_seed(int(config["seed"]))
    device = resolve_device(args.device)
    output_dir = args.manifest.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.snapshot.json", config)

    data = load_baseline_data(args.metadata, args.proteome, config["missing_rate_threshold"])
    fit_ids = data.train_ids[
        data.metadata.loc[data.train_ids, "perturbation_no_concentration"].ne(QC_LABEL)
    ]
    fit_metadata = data.metadata.loc[fit_ids]
    target_metadata = data.metadata.loc[data.target_ids]
    metadata_bundle = encode_metadata(
        fit_metadata,
        target_metadata,
        config["categorical_columns"],
        config["log2_numeric_columns"],
    )
    structure_table = load_structure_table(args.structure_npz, args.structure_contract)
    structure_bundle = encode_structure(
        fit_metadata, target_metadata, structure_table, config["compound_column"]
    )
    inputs = split_dual_inputs(
        metadata_bundle, structure_bundle, config["compound_column"]
    )
    train_condition = torch.from_numpy(inputs.fit_condition)
    train_drug = torch.from_numpy(inputs.fit_drug)
    target_condition = torch.from_numpy(inputs.target_condition)
    target_drug = torch.from_numpy(inputs.target_drug)
    train_truth_frame = data.truth_log2.loc[fit_ids]
    train_truth_values = train_truth_frame.to_numpy(np.float32)
    train_truth = torch.from_numpy(np.nan_to_num(train_truth_values, nan=0.0))
    train_mask = torch.from_numpy(np.isfinite(train_truth_values).astype(np.float32))
    validation_rows = treatment_mask(target_metadata).to_numpy(copy=True)
    validation_selector = torch.from_numpy(validation_rows)
    validation_truth_frame = data.truth_log2.loc[data.target_ids[validation_rows]]
    validation_truth_values = validation_truth_frame.to_numpy(np.float32)
    validation_truth = torch.from_numpy(
        np.nan_to_num(validation_truth_values, nan=0.0)
    )
    validation_mask = torch.from_numpy(
        np.isfinite(validation_truth_values).astype(np.float32)
    )

    e7, e7_base, _e7_checkpoint = load_frozen_e7(
        args.e7_model,
        train_condition.shape[1],
        train_drug.shape[1],
        train_truth.shape[1],
        data.retained_proteins.astype(str).tolist(),
        device,
    )
    evaluation_batch_size = int(config["evaluation_batch_size"])
    train_e7_prediction, train_context = extract_e7_outputs(
        e7,
        train_condition,
        train_drug,
        e7_base,
        evaluation_batch_size,
        device,
    )
    target_e7_prediction, target_context = extract_e7_outputs(
        e7,
        target_condition,
        target_drug,
        e7_base,
        evaluation_batch_size,
        device,
    )
    validation_e7_prediction = target_e7_prediction[validation_selector]
    validation_context = target_context[validation_selector]
    paired = build_train_only_paired_delta(
        fit_metadata,
        train_truth_frame,
        inputs.fit_drug,
        CONTROL_KEYS,
        CONTROL_LABELS,
        config["compound_column"],
    )
    if paired.manifest["matched_treatment_rows"] != 5066 or paired.manifest[
        "unmatched_treatment_rows"
    ] != 12:
        raise ValueError(f"Frozen train matched-control coverage changed: {paired.manifest}")
    paired_control_e7_prediction, paired_control_context = extract_control_e7_outputs(
        e7,
        train_condition,
        paired.control_drug_options,
        paired.control_weights,
        e7_base,
        evaluation_batch_size,
        device,
    )
    cvae = build_cvae(config, train_truth.shape[1])
    additional_parameters = trainable_parameter_count(cvae)
    if additional_parameters > int(config["maximum_additional_parameters"]):
        raise ValueError(
            f"Additional parameter budget exceeded: {additional_parameters}"
        )

    if args.smoke_only:
        if args.smoke_output is None:
            raise ValueError("--smoke-only requires --smoke-output")
        smoke = smoke_real_batch(
            cvae,
            train_context,
            train_e7_prediction,
            train_truth,
            train_mask,
            paired.eligible,
            torch.from_numpy(paired.target_delta),
            torch.from_numpy(paired.target_mask),
            paired_control_e7_prediction,
            paired_control_context,
            torch.from_numpy(paired.control_weights),
            config,
            device,
        )
        smoke.update(
            {
                "metadata_sha256": data.hashes["metadata_sha256"],
                "proteome_sha256": data.hashes["proteome_sha256"],
                "e7_model_sha256": sha256_file(args.e7_model),
                "frozen_e7_core_sha256": frozen_e7_core_sha256(),
                "matched_train_treatment_rows": paired.manifest[
                    "matched_treatment_rows"
                ],
            }
        )
        write_json(args.smoke_output, smoke)
        print(json.dumps(smoke, ensure_ascii=False, indent=2))
        return 0

    wandb_config = config["wandb"]
    wandb_mode = args.wandb_mode or wandb_config["mode"]
    wandb_run = wandb.init(
        project=wandb_config["project"],
        entity=wandb_config.get("entity"),
        name=output_dir.name,
        job_type=wandb_config["job_type"],
        tags=wandb_config["tags"],
        config=config,
        dir=str(output_dir),
        mode=wandb_mode,
        save_code=False,
    )
    wandb_run.define_metric("selection/epoch")
    wandb_run.define_metric("selection/*", step_metric="selection/epoch")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    cvae, best_epoch, history = train_with_early_stopping(
        cvae,
        train_context,
        train_e7_prediction,
        train_truth,
        train_mask,
        paired.eligible,
        torch.from_numpy(paired.target_delta),
        torch.from_numpy(paired.target_mask),
        paired_control_e7_prediction,
        paired_control_context,
        torch.from_numpy(paired.control_weights),
        validation_context,
        validation_e7_prediction,
        validation_truth,
        validation_mask,
        config,
        device,
        wandb_run,
    )
    write_json(
        args.history,
        {
            "schema_version": "1.0",
            "best_epoch": best_epoch,
            "selection_split": "official validation treatment rows",
            "epoch_zero_policy": "exact frozen E7 dense bypass safety baseline",
            "selection_history": history,
        },
    )
    latent_audit = latent_collapse_audit(
        cvae,
        train_context,
        train_e7_prediction,
        train_truth,
        train_mask,
        evaluation_batch_size,
        device,
    )
    write_json(args.latent_audit, latent_audit)
    predictions, prediction_audit = predict_prior_mean(
        cvae,
        target_context,
        target_e7_prediction,
        evaluation_batch_size,
        device,
    )
    if predictions.shape != (len(data.target_ids), len(data.retained_proteins)):
        raise ValueError("Prediction shape mismatch")
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions must be finite")
    pd.DataFrame(
        predictions, index=data.target_ids, columns=data.retained_proteins
    ).rename_axis("sample_ID").reset_index().to_csv(args.prediction, index=False)

    torch.save(
        {
            "state_dict": cvae.state_dict(),
            "config": config,
            "frozen_e7_model_sha256": sha256_file(args.e7_model),
            "frozen_e7_core_sha256": frozen_e7_core_sha256(),
            "metadata_feature_manifest": metadata_bundle.manifest,
            "structure_feature_manifest": structure_bundle.manifest,
            "dual_input_manifest": inputs.manifest,
            "paired_delta_manifest": paired.manifest,
            "retained_proteins": data.retained_proteins.astype(str).tolist(),
            "inference_contract": "rebuild frozen E7, extract fusion context, add CVAE prior-mean bounded correction",
        },
        args.model,
    )

    args.score_output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(args.scorer_script),
            str(args.metadata),
            str(args.proteome),
            str(args.prediction),
            str(args.score_config),
            str(args.score_output),
        ],
        check=True,
    )
    metrics = json.loads(args.score_output.read_text(encoding="utf-8"))
    score_payload = {
        "validation/provisional_total_100": metrics["provisional_weighted_proxy_100"],
        "validation/provisional_total_01": metrics["provisional_weighted_proxy"],
    }
    for module, values in metrics["modules"].items():
        score_payload[f"validation/{module}/normalized_score"] = values[
            "normalized_score"
        ]
        score_payload[f"validation/{module}/raw_score"] = values["raw_score"]
        score_payload[f"validation/{module}/weighted_points"] = values[
            "weighted_points"
        ]
    wandb_run.log(score_payload)
    for key, value in score_payload.items():
        wandb_run.summary[key] = value
    wandb_run.summary["selection/best_epoch"] = best_epoch
    wandb_run.summary["model/additional_parameter_count"] = additional_parameters
    wandb_run.summary["latent/posterior_collapse_flag"] = latent_audit[
        "posterior_collapse_flag"
    ]
    wandb_run.summary["latent/raw_kl_nats_per_sample_mean"] = latent_audit[
        "raw_kl_nats_per_sample_mean"
    ]

    peak_gpu_bytes = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    mapped_gpu = visible_devices.split(",", maxsplit=1)[0] if visible_devices else None
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
        "cuda_visible_devices": visible_devices,
        "runtime_device_mapping": f"runtime cuda:0 maps to physical GPU {mapped_gpu}",
        "peak_gpu_memory_allocated_bytes": peak_gpu_bytes,
        "elapsed_seconds": time.perf_counter() - started,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "distributed_training": False,
        "gpu_allocation_note": "GPU allocation is controlled by the caller through CUDA_VISIBLE_DEVICES",
    }
    wandb_info = {
        "mode": wandb_mode,
        "project": wandb_run.project,
        "entity": wandb_run.entity,
        "run_id": wandb_run.id,
        "run_name": wandb_run.name,
        "run_path": wandb_run.path,
        "run_url": wandb_run.url,
        "cloud_synced": wandb_mode == "online",
        "model_artifact_uploaded": False,
        "prediction_uploaded": False,
    }
    data_contract = {
        **data.hashes,
        "train_rows": len(data.train_ids),
        "fit_rows_excluding_qc": len(fit_ids),
        "matched_train_treatment_rows": paired.manifest["matched_treatment_rows"],
        "unmatched_train_treatment_rows": paired.manifest["unmatched_treatment_rows"],
        "validation_target_rows": len(data.target_ids),
        "validation_treatment_rows_for_early_stopping": int(validation_rows.sum()),
        "retained_proteins": len(data.retained_proteins),
        "missing_rate_threshold": config["missing_rate_threshold"],
        "test_proteome_loaded": False,
        "official_validation_used_for_gradient_training": False,
        "official_validation_used_for_early_stopping": True,
        "validation_target_entered_posterior_encoder": False,
        "test_target_entered_posterior_encoder": False,
        "protein_filter_fit_scope": "split_final=train only",
        "metadata_categories_and_numeric_statistics_fit_scope": "non-QC split_final=train rows only",
        "descriptor_statistics_fit_scope": "non-QC split_final=train rows only",
        "validation_or_test_truth_used_for_features": False,
    }
    manifest = {
        "schema_version": "1.0",
        "method_id": config["method_id"],
        "variant_id": config["variant_id"],
        "deployable": config["deployable"],
        "diagnostic_only": config["diagnostic_only"],
        "generated_at": datetime.now().astimezone().isoformat(),
        "data_contract": data_contract,
        "input_hashes": {
            "config_sha256": sha256_file(args.config),
            "structure_npz_sha256": sha256_file(args.structure_npz),
            "structure_contract_sha256": sha256_file(args.structure_contract),
            "structure_manifest_sha256": sha256_file(args.structure_manifest),
            "score_config_sha256": sha256_file(args.score_config),
            "frozen_e7_model_sha256": sha256_file(args.e7_model),
            "frozen_e7_manifest_sha256": sha256_file(args.e7_manifest),
            "frozen_e7_core_sha256": frozen_e7_core_sha256(),
        },
        "output_hashes": {
            "prediction_sha256": sha256_file(args.prediction),
            "model_sha256": sha256_file(args.model),
            "history_sha256": sha256_file(args.history),
            "latent_audit_sha256": sha256_file(args.latent_audit),
            "metrics_sha256": sha256_file(args.score_output),
        },
        "wandb": wandb_info,
        "runtime": runtime,
        "method_details": {
            "architecture": "frozen E7 full-rank dense prediction plus zero-initialized bounded latent-64 conditional residual VAE",
            "frozen_e7_parameter_count": parameter_count(e7),
            "additional_trainable_parameter_count": additional_parameters,
            "additional_parameter_budget": config["maximum_additional_parameters"],
            "additional_parameter_budget_pass": additional_parameters
            <= int(config["maximum_additional_parameters"]),
            "posterior_input": "frozen E7 fusion context plus train-only masked E7 residual target",
            "prior_input": "frozen E7 metadata/RDKit fusion context only",
            "inference": "conditional prior mean only; deterministic; no target proteins and no latent sampling",
            "residual": {
                "zero_initialized": True,
                "bound": config["correction_bound"],
                **prediction_audit,
            },
            "loss": {
                "absolute_posterior": "protein-space elementwise masked MSE",
                "absolute_prior": "protein-space elementwise masked MSE",
                "lambda_prior_absolute": config["lambda_prior_absolute"],
                "paired_delta": "train-only exact-control protein-space masked MSE using prior-mean predictions",
                "lambda_delta": config["lambda_delta"],
                "kl_beta_max": config["kl_beta_max"],
                "kl_warmup_epochs": config["kl_warmup_epochs"],
                "kl_free_bits_per_dim": config["kl_free_bits_per_dim"],
            },
            "best_epoch_from_official_validation": best_epoch,
            "epoch_zero_is_exact_e7_bypass": True,
            "seed": config["seed"],
            "paired_delta_manifest": paired.manifest,
            "latent_collapse_audit": latent_audit,
            "metadata_feature_manifest": metadata_bundle.manifest,
            "structure_feature_manifest": structure_bundle.manifest,
            "dual_input_manifest": inputs.manifest,
        },
        "validation_scoring": {
            "prediction_frozen_before_truth_scoring": True,
            "posterior_calls": 0,
            "metrics_sha256": sha256_file(args.score_output),
            "provisional_total_100": metrics["provisional_weighted_proxy_100"],
        },
    }
    write_json(args.manifest, manifest)
    best_row = next(row for row in history if int(row["epoch"]) == best_epoch)
    result = {
        "schema_version": "1.0",
        "experiment_id": output_dir.name,
        "method_id": config["method_id"],
        "selection_policy": "official validation prior-mean masked RMSE; epoch 0 exact E7 fallback; patience 40",
        "completed_epochs": len(history) - 1,
        "best_epoch": best_epoch,
        "best_official_validation_masked_rmse": best_row[
            "official_validation_masked_rmse"
        ],
        "total_100": metrics["provisional_weighted_proxy_100"],
        "modules": {
            module: values["normalized_score"]
            for module, values in metrics["modules"].items()
        },
        "scenario_metrics": scenario_metrics(metrics),
        "wandb": wandb_info,
        "resource": runtime,
        "boundary_audit": data_contract,
        "model_audit": {
            "additional_parameter_count": additional_parameters,
            "frozen_e7_parameter_count": parameter_count(e7),
            "prediction": prediction_audit,
            "latent": latent_audit,
        },
        "metrics_sha256": sha256_file(args.score_output),
        "model_sha256": sha256_file(args.model),
        "prediction_sha256": sha256_file(args.prediction),
    }
    write_json(args.result, result)
    hashes = {
        "config.json": sha256_file(args.config),
        "core.py": sha256_file(Path(__file__).with_name("core.py")),
        "runner.py": sha256_file(Path(__file__)),
        "frozen-e7-model.pt": sha256_file(args.e7_model),
        "prediction.csv": sha256_file(args.prediction),
        "model.pt": sha256_file(args.model),
        "training-history.json": sha256_file(args.history),
        "latent-audit.json": sha256_file(args.latent_audit),
        "metrics/six-module.json": sha256_file(args.score_output),
        "manifest.json": sha256_file(args.manifest),
        "result.json": sha256_file(args.result),
    }
    write_json(output_dir / "artifact-hashes.json", hashes)
    wandb_run.finish()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
