from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    (parent for parent in HERE.parents if (parent / "code").is_dir() or (parent / "solution").is_dir()),
    HERE.parents[1],
)
_METHOD_ROOTS = (
    PROJECT_ROOT / "solution" / "methods",
    PROJECT_ROOT / "code" / "methods",
    PROJECT_ROOT / "methods",
)
E7_DIR = next(
    (root / "final" / "b10-a2-film-cross-mlp" for root in _METHOD_ROOTS if (root / "final" / "b10-a2-film-cross-mlp").exists()),
    PROJECT_ROOT / "code" / "methods" / "b10-a2-film-cross-mlp",
)
SHARED_DIR = next(
    (root / "_shared" for root in _METHOD_ROOTS if (root / "_shared").exists()),
    PROJECT_ROOT / "code" / "methods" / "_shared",
)
sys.path.insert(0, str(SHARED_DIR))

from goai_baselines.core import (
    QC_LABEL,
    load_baseline_data,
    sha256_file,
    sha256_strings,
)


def load_e7_module() -> Any:
    path = E7_DIR / "core.py"
    spec = importlib.util.spec_from_file_location("b16_oof_source_e7_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load E7 source core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


E7 = load_e7_module()
FilmCrossMLP = E7.FilmCrossMLP
apply_compound_identity_dropout = E7.apply_compound_identity_dropout
encode_metadata = E7.encode_metadata
encode_structure = E7.encode_structure
load_structure_table = E7.load_structure_table
masked_mse = E7.masked_mse
parameter_count = E7.parameter_count
set_reproducible_seed = E7.set_reproducible_seed
split_dual_inputs = E7.split_dual_inputs

BIOLOGICAL_GROUP_COLUMNS = (
    "Strains",
    "perturbation_no_concentration",
    "Medium",
    "Temperature",
    "pert_time",
    "pert_time_unit",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build genuine train-only five-fold E7 OOF predictions."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--proteome", type=Path, required=True)
    parser.add_argument("--structure-npz", type=Path, required=True)
    parser.add_argument("--structure-contract", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--e7-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    return parser


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return device


def biological_groups(metadata: pd.DataFrame) -> np.ndarray:
    missing = sorted(set(BIOLOGICAL_GROUP_COLUMNS).difference(metadata.columns))
    if missing:
        raise ValueError(f"Missing biological grouping columns: {missing}")
    return (
        metadata.loc[:, BIOLOGICAL_GROUP_COLUMNS]
        .astype(str)
        .agg("\x1f".join, axis=1)
        .to_numpy(dtype=str)
    )


def make_fold_ids(
    metadata: pd.DataFrame, n_splits: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    groups = biological_groups(metadata)
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_ids = np.full(len(metadata), -1, dtype=np.int16)
    for fold_id, (_, holdout_indices) in enumerate(
        splitter.split(np.zeros(len(metadata)), groups=groups)
    ):
        if np.any(fold_ids[holdout_indices] != -1):
            raise RuntimeError("A row was assigned to more than one OOF fold")
        fold_ids[holdout_indices] = fold_id
    if np.any(fold_ids < 0):
        raise RuntimeError("OOF fold assignment does not cover every row")
    group_to_fold: dict[str, int] = {}
    for group, fold_id in zip(groups, fold_ids, strict=True):
        previous = group_to_fold.setdefault(group, int(fold_id))
        if previous != int(fold_id):
            raise RuntimeError("A biological replicate group crosses OOF folds")
    return fold_ids, groups


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


def evaluate_mse(
    model: FilmCrossMLP,
    condition: torch.Tensor,
    drug: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
    base: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    squared_error = 0.0
    observed = 0.0
    with torch.no_grad():
        for start in range(0, len(condition), batch_size):
            end = start + batch_size
            prediction = base.to(device) + model(
                condition[start:end].to(device), drug[start:end].to(device)
            )
            batch_mask = mask[start:end].to(device)
            batch_truth = truth[start:end].to(device)
            squared_error += float(
                ((((prediction - batch_truth) ** 2) * batch_mask).sum()).item()
            )
            observed += float(batch_mask.sum().item())
    if observed <= 0:
        raise RuntimeError("OOF evaluation fold has no observed targets")
    return squared_error / observed


def train_fold(
    train_condition: torch.Tensor,
    train_drug: torch.Tensor,
    train_truth: torch.Tensor,
    train_mask: torch.Tensor,
    holdout_condition: torch.Tensor,
    holdout_drug: torch.Tensor,
    holdout_truth: torch.Tensor,
    holdout_mask: torch.Tensor,
    base: torch.Tensor,
    compound_identity_dim: int,
    seen_mask_index: int,
    config: dict[str, Any],
    device: torch.device,
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
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, float]] = []
    base_device = base.to(device)
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        for batch in shuffled_batches(
            indices, int(config["batch_size"]), shuffle_generator
        ):
            dropped_drug, _ = apply_compound_identity_dropout(
                train_drug[batch],
                compound_identity_dim=compound_identity_dim,
                seen_mask_index=seen_mask_index,
                probability=float(config["compound_identity_dropout"]),
                generator=dropout_generator,
            )
            prediction = base_device + model(
                train_condition[batch].to(device), dropped_drug.to(device)
            )
            loss = masked_mse(
                prediction,
                train_truth[batch].to(device),
                train_mask[batch].to(device),
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite OOF fold training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        train_mse = evaluate_mse(
            model,
            train_condition,
            train_drug,
            train_truth,
            train_mask,
            base,
            int(config["evaluation_batch_size"]),
            device,
        )
        holdout_mse = evaluate_mse(
            model,
            holdout_condition,
            holdout_drug,
            holdout_truth,
            holdout_mask,
            base,
            int(config["evaluation_batch_size"]),
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "fold_train_masked_mse": train_mse,
                "fold_holdout_masked_mse": holdout_mse,
            }
        )
        print(
            f"oof epoch={epoch:03d} train_rmse={train_mse**0.5:.6f} "
            f"holdout_rmse={holdout_mse**0.5:.6f}",
            flush=True,
        )
        if holdout_mse < best_loss - float(config["early_stopping_min_delta"]):
            best_loss = holdout_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["early_stopping_patience"]):
                break
    if best_state is None:
        raise RuntimeError("OOF fold early stopping selected no checkpoint")
    model.load_state_dict(best_state)
    return model, best_epoch, history


def predict(
    model: FilmCrossMLP,
    condition: torch.Tensor,
    drug: torch.Tensor,
    base: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(condition), batch_size):
            end = start + batch_size
            values = base.to(device) + model(
                condition[start:end].to(device), drug[start:end].to(device)
            )
            rows.append(values.cpu().numpy())
    return np.vstack(rows).astype(np.float32, copy=False)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    write_json(temporary, payload)
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if "test" in args.proteome.name.lower() and "train_val" not in args.proteome.name.lower():
        raise ValueError("OOF builder refuses test proteome truth")
    config = json.loads(args.e7_config.read_text(encoding="utf-8"))
    if config["method_id"] != "b10-a2-film-cross-mlp":
        raise ValueError("OOF source config must be E7 b10-a2-film-cross-mlp")
    config["seed"] = int(args.seed)
    config["max_epochs"] = int(args.max_epochs)
    config["early_stopping_patience"] = int(args.patience)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    data = load_baseline_data(
        args.metadata, args.proteome, float(config["missing_rate_threshold"])
    )
    fit_ids = data.train_ids[
        data.metadata.loc[data.train_ids, "perturbation_no_concentration"].ne(QC_LABEL)
    ]
    fit_metadata = data.metadata.loc[fit_ids]
    truth_frame = data.truth_log2.loc[fit_ids]
    truth_values = truth_frame.to_numpy(np.float32)
    observed_mask = np.isfinite(truth_values)
    fold_ids, groups = make_fold_ids(fit_metadata, int(args.n_splits), int(args.seed))
    structure_table = load_structure_table(args.structure_npz, args.structure_contract)
    predictions = np.full(truth_values.shape, np.nan, dtype=np.float32)

    staging_folds = output_dir / f".folds.tmp-{os.getpid()}"
    if staging_folds.exists():
        shutil.rmtree(staging_folds)
    staging_folds.mkdir(parents=True)
    fold_manifests: list[dict[str, Any]] = []
    for fold_id in range(int(args.n_splits)):
        print(f"starting OOF fold {fold_id}", flush=True)
        train_selector = fold_ids != fold_id
        holdout_selector = fold_ids == fold_id
        fold_train_ids = fit_ids[train_selector]
        fold_holdout_ids = fit_ids[holdout_selector]
        train_metadata = fit_metadata.loc[fold_train_ids]
        holdout_metadata = fit_metadata.loc[fold_holdout_ids]
        metadata_bundle = encode_metadata(
            train_metadata,
            holdout_metadata,
            config["categorical_columns"],
            config["log2_numeric_columns"],
        )
        structure_bundle = encode_structure(
            train_metadata,
            holdout_metadata,
            structure_table,
            config["compound_column"],
        )
        inputs = split_dual_inputs(
            metadata_bundle, structure_bundle, config["compound_column"]
        )
        train_values = truth_frame.loc[fold_train_ids].to_numpy(np.float32)
        holdout_values = truth_frame.loc[fold_holdout_ids].to_numpy(np.float32)
        train_truth = torch.from_numpy(np.nan_to_num(train_values, nan=0.0))
        holdout_truth = torch.from_numpy(np.nan_to_num(holdout_values, nan=0.0))
        train_mask = torch.from_numpy(np.isfinite(train_values).astype(np.float32))
        holdout_mask = torch.from_numpy(
            np.isfinite(holdout_values).astype(np.float32)
        )
        with np.errstate(invalid="ignore"):
            base_values = np.nanmean(train_values, axis=0).astype(np.float32)
        if not np.isfinite(base_values).all():
            raise RuntimeError("A retained protein lacks an observed fold-train value")
        base = torch.from_numpy(base_values)[None, :]
        model, best_epoch, history = train_fold(
            torch.from_numpy(inputs.fit_condition),
            torch.from_numpy(inputs.fit_drug),
            train_truth,
            train_mask,
            torch.from_numpy(inputs.target_condition),
            torch.from_numpy(inputs.target_drug),
            holdout_truth,
            holdout_mask,
            base,
            int(inputs.manifest["compound_identity_dimension"]),
            int(inputs.manifest["drug_seen_mask_index"]),
            config,
            device,
        )
        holdout_predictions = predict(
            model,
            torch.from_numpy(inputs.target_condition),
            torch.from_numpy(inputs.target_drug),
            base,
            int(config["evaluation_batch_size"]),
            device,
        )
        predictions[holdout_selector] = holdout_predictions
        fold_path = staging_folds / f"fold-{fold_id}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": config,
                "fold_id": fold_id,
                "fold_train_ids": fold_train_ids.astype(str).tolist(),
                "fold_holdout_ids": fold_holdout_ids.astype(str).tolist(),
                "base_profile": base,
                "retained_proteins": data.retained_proteins.astype(str).tolist(),
                "metadata_feature_manifest": metadata_bundle.manifest,
                "structure_feature_manifest": structure_bundle.manifest,
                "dual_input_manifest": inputs.manifest,
            },
            fold_path,
        )
        history_path = staging_folds / f"fold-{fold_id}-history.json"
        write_json(
            history_path,
            {
                "schema_version": "1.0",
                "selection_scope": "this fold holdout inside split_final=train only",
                "best_epoch": best_epoch,
                "history": history,
            },
        )
        fold_manifests.append(
            {
                "fold_id": fold_id,
                "train_rows": int(train_selector.sum()),
                "holdout_rows": int(holdout_selector.sum()),
                "train_unique_biological_groups": int(np.unique(groups[train_selector]).size),
                "holdout_unique_biological_groups": int(
                    np.unique(groups[holdout_selector]).size
                ),
                "biological_group_overlap": 0,
                "best_epoch": best_epoch,
                "best_holdout_masked_mse": float(
                    min(row["fold_holdout_masked_mse"] for row in history)
                ),
                "parameter_count": parameter_count(model),
                "checkpoint": f"folds/fold-{fold_id}.pt",
                "checkpoint_sha256": sha256_file(fold_path),
                "history": f"folds/fold-{fold_id}-history.json",
                "history_sha256": sha256_file(history_path),
                "statistics_scope": "fold training rows only",
                "base_profile_sha256": hashlib.sha256(base_values.tobytes()).hexdigest(),
                "metadata_feature_manifest": metadata_bundle.manifest,
                "structure_feature_manifest": structure_bundle.manifest,
                "dual_input_manifest": inputs.manifest,
            }
        )

    if not np.isfinite(predictions).all():
        raise RuntimeError("OOF predictions are not finite and fully covered")
    final_folds = output_dir / "folds"
    if final_folds.exists():
        raise FileExistsError(f"Refusing to overwrite existing {final_folds}")
    os.replace(staging_folds, final_folds)
    npz_path = output_dir / "e7-oof-predictions.npz"
    atomic_npz(
        npz_path,
        sample_ids=fit_ids.astype(str).to_numpy(dtype=str),
        protein_names=data.retained_proteins.astype(str).to_numpy(dtype=str),
        predictions=predictions,
        fold_ids=fold_ids.astype(np.int16, copy=False),
        observed_mask=observed_mask.astype(bool, copy=False),
    )
    residual = truth_values - predictions
    observed_residual = residual[observed_mask]
    peak_gpu_bytes = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
    manifest = {
        "schema_version": "1.0",
        "artifact_id": "b16-e7-oof-residual-v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "artifact_file": npz_path.name,
        "artifact_sha256": sha256_file(npz_path),
        "keys": {
            "sample_ids": {"dtype": "unicode", "shape": [len(fit_ids)]},
            "protein_names": {
                "dtype": "unicode",
                "shape": [len(data.retained_proteins)],
            },
            "predictions": {
                "dtype": "float32",
                "shape": list(predictions.shape),
                "semantics": "one prediction per non-QC split_final=train row from a model that excluded that row's biological group",
            },
            "fold_ids": {"dtype": "int16", "shape": [len(fit_ids)]},
            "observed_mask": {"dtype": "bool", "shape": list(observed_mask.shape)},
        },
        "axes": {
            "sample_axis_sha256": sha256_strings(fit_ids),
            "protein_axis_sha256": sha256_strings(data.retained_proteins),
            "sample_rows": len(fit_ids),
            "proteins": len(data.retained_proteins),
        },
        "source": {
            **data.hashes,
            "e7_config_sha256": sha256_file(args.e7_config),
            "structure_npz_sha256": sha256_file(args.structure_npz),
            "structure_contract_sha256": sha256_file(args.structure_contract),
            "structure_manifest_sha256": sha256_file(args.structure_manifest),
            "truth_reconstruction": "load only WAYB_WAYC_proteome_raw_train_val.csv, apply log2 and the manifest protein axis, then select sample_ids; missingness is also materialized as observed_mask",
        },
        "fold_contract": {
            "algorithm": "sklearn.model_selection.GroupKFold",
            "n_splits": int(args.n_splits),
            "shuffle": True,
            "random_state": int(args.seed),
            "group_columns": list(BIOLOGICAL_GROUP_COLUMNS),
            "group_semantics": "technical replicates sharing biological condition cannot cross folds",
            "all_rows_covered_once": bool(np.all(fold_ids >= 0)),
            "unique_biological_groups": int(np.unique(groups).size),
        },
        "training_contract": {
            "source_architecture": "E7 b10-a2-film-cross-mlp",
            "seed": int(args.seed),
            "max_epochs": int(args.max_epochs),
            "early_stopping_patience": int(args.patience),
            "selection": "fold holdout masked MSE; both train and holdout are split_final=train",
            "per_fold_metadata_categories_numeric_statistics_scope": "fold training rows only",
            "per_fold_rdkit_descriptor_statistics_scope": "fold training rows only",
            "per_fold_base_profile_scope": "fold training rows only",
            "fixed_protein_axis_scope": "official split_final=train missing-rate filter shared with E7",
            "official_validation_truth_loaded": False,
            "test_truth_loaded": False,
            "in_sample_prediction_used": False,
        },
        "folds": fold_manifests,
        "quality": {
            "finite_prediction_fraction": float(np.isfinite(predictions).mean()),
            "observed_fraction": float(observed_mask.mean()),
            "observed_oof_residual_mean": float(observed_residual.mean()),
            "observed_oof_residual_rmse": float(
                np.sqrt(np.mean(observed_residual.astype(np.float64) ** 2))
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sklearn": __import__("sklearn").__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor(),
            "peak_gpu_memory_allocated_bytes": peak_gpu_bytes,
            "elapsed_seconds": time.perf_counter() - started,
            "omp_num_threads": int(os.environ.get("OMP_NUM_THREADS", "0")),
            "mkl_num_threads": int(os.environ.get("MKL_NUM_THREADS", "0")),
            "distributed_training": False,
        },
    }
    atomic_json(output_dir / "manifest.json", manifest)
    print(f"artifact={npz_path}")
    print(f"artifact_sha256={manifest['artifact_sha256']}")
    print(f"oof_rmse={manifest['quality']['observed_oof_residual_rmse']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
