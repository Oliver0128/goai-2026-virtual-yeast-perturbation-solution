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
    FilmCrossMLP,
    apply_compound_identity_dropout,
    encode_metadata,
    encode_structure,
    load_structure_table,
    masked_mse,
    parameter_count,
    set_reproducible_seed,
    split_dual_inputs,
)
from goai_baselines.core import (
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default=None)
    parser.add_argument("--scorer-script", type=Path, required=True)
    parser.add_argument("--score-config", type=Path, required=True)
    parser.add_argument("--score-output", type=Path, required=True)
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    return device


def shuffled_batches(
    indices: np.ndarray, batch_size: int, generator: torch.Generator
) -> list[np.ndarray]:
    order = torch.randperm(len(indices), generator=generator).numpy()
    return [
        indices[order[start : start + batch_size]]
        for start in range(0, len(indices), batch_size)
    ]


def build_model(
    condition_dim: int,
    drug_dim: int,
    output_dim: int,
    config: dict[str, Any],
) -> FilmCrossMLP:
    return FilmCrossMLP(
        condition_dim=condition_dim,
        drug_dim=drug_dim,
        output_dim=output_dim,
        encoder_hidden_dim=int(config["encoder_hidden_dim"]),
        latent_dim=int(config["latent_dim"]),
        fusion_dim=int(config["fusion_dim"]),
        dropout=float(config["dropout"]),
    )


def _audit_accumulator() -> dict[str, float]:
    return {
        "rows": 0.0,
        "gamma_abs_sum": 0.0,
        "gamma_elements": 0.0,
        "gamma_abs_max": 0.0,
        "beta_abs_sum": 0.0,
        "beta_elements": 0.0,
        "beta_abs_max": 0.0,
        "condition_film_abs_max": 0.0,
    }


def _update_audit(
    accumulator: dict[str, float], audit: dict[str, torch.Tensor]
) -> None:
    gamma = audit["gamma"]
    beta = audit["beta"]
    accumulator["rows"] += float(gamma.shape[0])
    accumulator["gamma_abs_sum"] += float(gamma.abs().sum().item())
    accumulator["gamma_elements"] += float(gamma.numel())
    accumulator["gamma_abs_max"] = max(
        accumulator["gamma_abs_max"], float(gamma.abs().max().item())
    )
    accumulator["beta_abs_sum"] += float(beta.abs().sum().item())
    accumulator["beta_elements"] += float(beta.numel())
    accumulator["beta_abs_max"] = max(
        accumulator["beta_abs_max"], float(beta.abs().max().item())
    )
    accumulator["condition_film_abs_max"] = max(
        accumulator["condition_film_abs_max"],
        float(audit["condition_film"].abs().max().item()),
    )


def _finalize_audit(accumulator: dict[str, float]) -> dict[str, float]:
    return {
        "gamma_abs_mean": accumulator["gamma_abs_sum"]
        / max(accumulator["gamma_elements"], 1.0),
        "gamma_abs_max": accumulator["gamma_abs_max"],
        "beta_abs_mean": accumulator["beta_abs_sum"]
        / max(accumulator["beta_elements"], 1.0),
        "beta_abs_max": accumulator["beta_abs_max"],
        "condition_film_abs_max": accumulator["condition_film_abs_max"],
    }


def evaluate(
    model: FilmCrossMLP,
    condition: torch.Tensor,
    drug: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
    base: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    squared_error = 0.0
    observed = 0.0
    audit_accumulator = _audit_accumulator()
    with torch.no_grad():
        for start in range(0, len(condition), batch_size):
            end = start + batch_size
            cb = condition[start:end].to(device)
            db = drug[start:end].to(device)
            yb = truth[start:end].to(device)
            mb = mask[start:end].to(device)
            residual, audit = model(cb, db, return_audit=True)
            prediction = base + residual
            squared_error += ((((prediction - yb) ** 2) * mb).sum()).item()
            observed += mb.sum().item()
            _update_audit(audit_accumulator, audit)
    metrics = {
        "masked_mse": squared_error / observed,
        "masked_rmse": (squared_error / observed) ** 0.5,
    }
    metrics.update(_finalize_audit(audit_accumulator))
    return metrics


def train_with_early_stopping(
    train_condition: torch.Tensor,
    train_drug: torch.Tensor,
    train_truth: torch.Tensor,
    train_mask: torch.Tensor,
    validation_condition: torch.Tensor,
    validation_drug: torch.Tensor,
    validation_truth: torch.Tensor,
    validation_mask: torch.Tensor,
    base: torch.Tensor,
    compound_identity_dim: int,
    seen_mask_index: int,
    config: dict[str, Any],
    device: torch.device,
    wandb_run: Any,
) -> tuple[FilmCrossMLP, int, list[dict[str, float]]]:
    seed = int(config["seed"])
    set_reproducible_seed(seed)
    model = build_model(
        train_condition.shape[1], train_drug.shape[1], train_truth.shape[1], config
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    shuffle_generator = torch.Generator().manual_seed(seed)
    dropout_generator = torch.Generator().manual_seed(seed + 1)
    indices = np.arange(len(train_condition))
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    base_device = base.to(device)

    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        dropped_rows = 0
        for batch in shuffled_batches(
            indices, int(config["batch_size"]), shuffle_generator
        ):
            dropped_drug, dropped = apply_compound_identity_dropout(
                train_drug[batch],
                compound_identity_dim=compound_identity_dim,
                seen_mask_index=seen_mask_index,
                probability=float(config["compound_identity_dropout"]),
                generator=dropout_generator,
            )
            dropped_rows += int(dropped.sum().item())
            cb = train_condition[batch].to(device)
            db = dropped_drug.to(device)
            yb = train_truth[batch].to(device)
            mb = train_mask[batch].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = base_device + model(cb, db)
            loss = masked_mse(prediction, yb, mb)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            optimizer.step()

        train_metrics = evaluate(
            model,
            train_condition,
            train_drug,
            train_truth,
            train_mask,
            base_device,
            int(config["evaluation_batch_size"]),
            device,
        )
        validation_metrics = evaluate(
            model,
            validation_condition,
            validation_drug,
            validation_truth,
            validation_mask,
            base_device,
            int(config["evaluation_batch_size"]),
            device,
        )
        row = {
            "epoch": epoch,
            "train_masked_mse": train_metrics["masked_mse"],
            "train_masked_rmse": train_metrics["masked_rmse"],
            "official_validation_masked_mse": validation_metrics["masked_mse"],
            "official_validation_masked_rmse": validation_metrics["masked_rmse"],
            "official_validation_gamma_abs_mean": validation_metrics["gamma_abs_mean"],
            "official_validation_gamma_abs_max": validation_metrics["gamma_abs_max"],
            "official_validation_beta_abs_mean": validation_metrics["beta_abs_mean"],
            "official_validation_beta_abs_max": validation_metrics["beta_abs_max"],
            "official_validation_condition_film_abs_max": validation_metrics[
                "condition_film_abs_max"
            ],
            "identity_dropped_rows": dropped_rows,
        }
        history.append(row)
        wandb_run.log({f"selection/{key}": value for key, value in row.items()})
        print(
            f"early-stop epoch={epoch:03d} train_rmse={train_metrics['masked_rmse']:.6f} "
            f"official_validation_rmse={validation_metrics['masked_rmse']:.6f} "
            f"gamma_abs_mean={validation_metrics['gamma_abs_mean']:.4f} "
            f"beta_abs_mean={validation_metrics['beta_abs_mean']:.4f} "
            f"dropped={dropped_rows}",
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
    if best_state is None or best_epoch < 1:
        raise RuntimeError("Early stopping failed to select an epoch")
    model.load_state_dict(best_state)
    return model, best_epoch, history


def predict(
    model: FilmCrossMLP,
    condition: torch.Tensor,
    drug: torch.Tensor,
    base: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    model.eval()
    rows: list[np.ndarray] = []
    audit_accumulator = _audit_accumulator()
    base_device = base.to(device)
    with torch.no_grad():
        for start in range(0, len(condition), batch_size):
            end = start + batch_size
            residual, audit = model(
                condition[start:end].to(device),
                drug[start:end].to(device),
                return_audit=True,
            )
            rows.append((base_device + residual).cpu().numpy())
            _update_audit(audit_accumulator, audit)
    predictions = np.vstack(rows).astype(np.float64)
    return predictions, _finalize_audit(audit_accumulator)


def scenario_metrics(metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for split, values in metrics["metrics"]["by_split"].items():
        result[split] = {
            "log2_rmse": values["absolute"]["sample_rmse"]["value"],
            "delta_pcc": values["fc"]["pcc"]["value"],
        }
    return result


def _validate_config(config: dict[str, Any]) -> None:
    if config["method_id"] != "b10-a2-film-cross-mlp":
        raise ValueError("Unexpected method_id")
    required_exact = {
        "seed": 42,
        "max_epochs": 1000,
        "early_stopping_patience": 50,
        "encoder_hidden_dim": 256,
        "latent_dim": 128,
        "fusion_dim": 256,
        "fusion_residual_blocks": 1,
    }
    for key, expected in required_exact.items():
        if int(config[key]) != expected:
            raise ValueError(
                f"E7 controlled-comparison contract requires {key}={expected}"
            )
    required_float = {
        "dropout": 0.2,
        "compound_identity_dropout": 0.25,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
    }
    for key, expected in required_float.items():
        if not np.isclose(float(config[key]), expected):
            raise ValueError(
                f"E7 controlled-comparison contract requires {key}={expected}"
            )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _validate_config(config)
    device = resolve_device(args.device)
    output_dir = args.manifest.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.snapshot.json", config)

    data = load_baseline_data(
        args.metadata, args.proteome, config["missing_rate_threshold"]
    )
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
        fit_metadata,
        target_metadata,
        structure_table,
        config["compound_column"],
    )
    inputs = split_dual_inputs(
        metadata_bundle, structure_bundle, config["compound_column"]
    )

    train_truth_frame = data.truth_log2.loc[fit_ids]
    train_truth_values = train_truth_frame.to_numpy(np.float32)
    train_truth = torch.from_numpy(np.nan_to_num(train_truth_values, nan=0.0))
    train_mask = torch.from_numpy(np.isfinite(train_truth_values).astype(np.float32))
    validation_row_mask = treatment_mask(target_metadata).to_numpy(copy=True)
    validation_ids = data.target_ids[validation_row_mask]
    validation_truth_frame = data.truth_log2.loc[validation_ids]
    validation_truth_values = validation_truth_frame.to_numpy(np.float32)
    validation_truth = torch.from_numpy(np.nan_to_num(validation_truth_values, nan=0.0))
    validation_mask = torch.from_numpy(
        np.isfinite(validation_truth_values).astype(np.float32)
    )

    train_condition = torch.from_numpy(inputs.fit_condition)
    target_condition = torch.from_numpy(inputs.target_condition)
    train_drug = torch.from_numpy(inputs.fit_drug)
    target_drug = torch.from_numpy(inputs.target_drug)
    validation_selector = torch.from_numpy(validation_row_mask)
    validation_condition = target_condition[validation_selector]
    validation_drug = target_drug[validation_selector]
    base_values = (
        train_truth_frame.mean(axis=0).fillna(data.train_mean).to_numpy(np.float32)
    )
    base = torch.from_numpy(base_values)[None, :]

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

    model, best_epoch, history = train_with_early_stopping(
        train_condition,
        train_drug,
        train_truth,
        train_mask,
        validation_condition,
        validation_drug,
        validation_truth,
        validation_mask,
        base,
        int(inputs.manifest["compound_identity_dimension"]),
        int(inputs.manifest["drug_seen_mask_index"]),
        config,
        device,
        wandb_run,
    )
    predictions, prediction_audit = predict(
        model,
        target_condition,
        target_drug,
        base,
        int(config["evaluation_batch_size"]),
        device,
    )
    if predictions.shape != (len(data.target_ids), len(data.retained_proteins)):
        raise ValueError("Prediction shape mismatch")
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions must be finite")
    if not all(np.isfinite(value) for value in prediction_audit.values()):
        raise ValueError("FiLM audit values must be finite")

    pd.DataFrame(
        predictions, index=data.target_ids, columns=data.retained_proteins
    ).rename_axis("sample_ID").reset_index().to_csv(args.prediction, index=False)
    write_json(
        args.history,
        {
            "schema_version": "1.0",
            "best_epoch": best_epoch,
            "selection_split": "official validation treatment rows",
            "selection_history": history,
        },
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "metadata_feature_manifest": metadata_bundle.manifest,
            "structure_feature_manifest": structure_bundle.manifest,
            "dual_input_manifest": inputs.manifest,
            "base_profile": base,
            "retained_proteins": data.retained_proteins.astype(str).tolist(),
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
        score_payload[f"validation/{module}/raw_score"] = values["raw_score"]
        score_payload[f"validation/{module}/normalized_score"] = values[
            "normalized_score"
        ]
        score_payload[f"validation/{module}/weighted_points"] = values[
            "weighted_points"
        ]
    wandb_run.log(score_payload)
    for key, value in score_payload.items():
        wandb_run.summary[key] = value
    wandb_run.summary["selection/best_epoch"] = best_epoch
    wandb_run.summary["model/parameter_count"] = parameter_count(model)

    peak_gpu_bytes = 0
    if device.type == "cuda":
        peak_gpu_bytes = int(torch.cuda.max_memory_allocated())
    elapsed_seconds = time.perf_counter() - started
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
        "validation_target_rows": len(data.target_ids),
        "validation_treatment_rows_for_early_stopping": len(validation_ids),
        "retained_proteins": len(data.retained_proteins),
        "missing_rate_threshold": config["missing_rate_threshold"],
        "test_proteome_loaded": False,
        "official_validation_used_for_gradient_training": False,
        "official_validation_used_for_early_stopping": True,
        "protein_filter_fit_scope": "split_final=train only",
        "base_profile_fit_scope": "non-QC split_final=train rows only",
        "metadata_categories_and_numeric_statistics_fit_scope": "non-QC split_final=train rows only",
        "descriptor_statistics_fit_scope": "non-QC split_final=train rows only",
        "validation_or_test_truth_used_for_features": False,
    }
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "peak_gpu_memory_allocated_bytes": peak_gpu_bytes,
        "elapsed_seconds": elapsed_seconds,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "distributed_training": False,
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
        },
        "output_hashes": {
            "prediction_sha256": sha256_file(args.prediction),
            "model_sha256": sha256_file(args.model),
            "history_sha256": sha256_file(args.history),
            "metrics_sha256": sha256_file(args.score_output),
        },
        "wandb": wandb_info,
        "runtime": runtime,
        "method_details": {
            "architecture": "condition encoder 128 + drug encoder 128 + drug-conditioned FiLM(c) + residual fusion 256 + single 256-to-protein residual head",
            "controlled_difference": "c_film=(1+gamma(d))*c+beta(d); gamma/beta output layers zero-initialized",
            "condition_encoder_input": "all encoded metadata except compound one-hot",
            "drug_encoder_input": "compound one-hot plus Morgan-2048 plus descriptors and masks",
            "parameter_count": parameter_count(model),
            "best_epoch_from_official_validation": best_epoch,
            "maximum_epochs": config["max_epochs"],
            "early_stopping_patience": config["early_stopping_patience"],
            "seed": config["seed"],
            "loss": "elementwise mask-aware MSE over observed log2 targets",
            "base_profile": "mean over non-QC split_final=train rows after train-only protein filter",
            "identity_dropout": {
                "probability": config["compound_identity_dropout"],
                "scope": "training only; compound one-hot and identity_seen_in_fit mask; public structure retained",
            },
            "film_prediction_audit": prediction_audit,
            "metadata_feature_manifest": metadata_bundle.manifest,
            "structure_feature_manifest": structure_bundle.manifest,
            "dual_input_manifest": inputs.manifest,
        },
        "validation_scoring": {
            "prediction_frozen_before_truth_scoring": True,
            "metrics_sha256": sha256_file(args.score_output),
            "provisional_total_100": metrics["provisional_weighted_proxy_100"],
        },
    }
    write_json(args.manifest, manifest)
    result = {
        "schema_version": "1.0",
        "experiment_id": output_dir.name,
        "method_id": config["method_id"],
        "selection_policy": "official validation treatment rows; maximum 1000 epochs; patience 50",
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "best_official_validation_masked_rmse": history[best_epoch - 1][
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
        "film_audit": prediction_audit,
        "metrics_sha256": sha256_file(args.score_output),
        "model_sha256": sha256_file(args.model),
    }
    write_json(args.result, result)
    wandb_run.finish()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
