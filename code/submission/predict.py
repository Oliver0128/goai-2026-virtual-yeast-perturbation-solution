from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
E7_CORE_PATH = ROOT / "methods" / "b10-a2-film-cross-mlp" / "core.py"
CVAE_CORE_PATH = ROOT / "methods" / "b16-a2-conditional-residual-vae" / "core.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E7 = _load_module("solution_e7_core", E7_CORE_PATH)
CVAE = _load_module("solution_cvae_core", CVAE_CORE_PATH)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Truth-free inference for the frozen YeaFiLM + Conditional Residual VAE."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--structure-npz", type=Path, required=True)
    parser.add_argument("--structure-contract", type=Path, required=True)
    parser.add_argument("--e7-model", type=Path, required=True)
    parser.add_argument("--cvae-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--treatment-only",
        action="store_true",
        help="Keep only non-control, non-QC rows. Use only if required by the official submission contract.",
    )
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return device


def encode_metadata_from_manifest(
    metadata: pd.DataFrame, manifest: dict[str, Any]
) -> np.ndarray:
    parts: list[np.ndarray] = []
    names: list[str] = []
    for column, specification in manifest["categorical"].items():
        if column not in metadata:
            raise ValueError(f"Metadata is missing categorical column {column}")
        values = metadata[column].astype(str)
        categories = [str(value) for value in specification["categories"]]
        parts.append(
            np.column_stack(
                [values.eq(category).to_numpy(dtype=np.float32) for category in categories]
            )
        )
        names.extend(f"{column}={category}" for category in categories)
    for column, specification in manifest["numeric"].items():
        if column not in metadata:
            raise ValueError(f"Metadata is missing numeric column {column}")
        raw = pd.to_numeric(metadata[column], errors="raise").to_numpy(np.float64)
        if np.any(raw <= 0):
            raise ValueError(f"{column} must be positive before log2 transform")
        values = np.log2(raw)
        mean = float(specification["mean"])
        scale = float(specification["scale"])
        parts.append(((values - mean) / scale).astype(np.float32)[:, None])
        names.append(f"log2({column})")
    if names != manifest["feature_names"]:
        raise ValueError("Rebuilt metadata feature axis differs from checkpoint")
    encoded = np.hstack(parts).astype(np.float32, copy=False)
    if not np.isfinite(encoded).all():
        raise ValueError("Encoded metadata contains non-finite values")
    return encoded


def encode_structure_from_manifest(
    metadata: pd.DataFrame,
    table: Any,
    manifest: dict[str, Any],
    compound_column: str,
    seen_compounds: set[str],
) -> np.ndarray:
    name_to_index = {
        str(name): index for index, name in enumerate(table.competition_names.tolist())
    }
    descriptor_mean = np.asarray(manifest["descriptor_mean"], dtype=np.float64)
    descriptor_scale = np.asarray(manifest["descriptor_scale"], dtype=np.float64)
    labels = metadata[compound_column].astype(str)
    rows = np.zeros(
        (len(labels), table.morgan_bits.shape[1] + table.descriptors_raw.shape[1] + 2),
        dtype=np.float32,
    )
    for row_index, label in enumerate(labels):
        artifact_index = name_to_index.get(label)
        if artifact_index is not None:
            rows[row_index, : table.morgan_bits.shape[1]] = table.morgan_bits[
                artifact_index
            ]
            start = table.morgan_bits.shape[1]
            stop = start + table.descriptors_raw.shape[1]
            rows[row_index, start:stop] = (
                table.descriptors_raw[artifact_index].astype(np.float64)
                - descriptor_mean
            ) / descriptor_scale
            rows[row_index, -2] = 1.0
        if label in seen_compounds:
            rows[row_index, -1] = 1.0
    if not np.isfinite(rows).all():
        raise ValueError("Encoded structure features contain non-finite values")
    return rows


def build_inputs(
    metadata: pd.DataFrame,
    e7_checkpoint: dict[str, Any],
    structure_table: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    metadata_manifest = e7_checkpoint["metadata_feature_manifest"]
    structure_manifest = e7_checkpoint["structure_feature_manifest"]
    compound_column = str(e7_checkpoint["config"]["compound_column"])
    encoded_metadata = encode_metadata_from_manifest(metadata, metadata_manifest)
    compound_start, compound_stop = map(
        int, metadata_manifest["feature_slices"][compound_column]
    )
    categories = metadata_manifest["categorical"][compound_column]["categories"]
    structure = encode_structure_from_manifest(
        metadata,
        structure_table,
        structure_manifest,
        compound_column,
        set(map(str, categories)),
    )
    keep = np.r_[
        np.arange(compound_start),
        np.arange(compound_stop, encoded_metadata.shape[1]),
    ]
    condition = encoded_metadata[:, keep]
    identity = encoded_metadata[:, compound_start:compound_stop]
    drug = np.hstack([identity, structure]).astype(np.float32, copy=False)
    contract = e7_checkpoint["dual_input_manifest"]
    if condition.shape[1] != int(contract["condition_feature_count"]):
        raise ValueError("Condition feature dimension differs from checkpoint")
    if drug.shape[1] != int(contract["drug_feature_count"]):
        raise ValueError("Drug feature dimension differs from checkpoint")
    return torch.from_numpy(condition), torch.from_numpy(drug)


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    device = resolve_device(args.device)
    metadata = pd.read_csv(args.metadata, dtype={"sample_ID": "string"})
    if "sample_ID" not in metadata or not metadata["sample_ID"].is_unique:
        raise ValueError("metadata must contain a unique sample_ID column")
    if args.treatment_only:
        labels = metadata["perturbation_no_concentration"].astype(str)
        pert_ids = metadata["pert_id"].astype(str).str.lstrip("#")
        metadata = metadata.loc[
            ~labels.isin({"Water", "DMSO", "Quality Control"}) & pert_ids.ne("48")
        ].copy()
    sample_ids = metadata["sample_ID"].astype(str).tolist()

    e7_checkpoint = torch.load(args.e7_model, map_location="cpu", weights_only=False)
    cvae_checkpoint = torch.load(
        args.cvae_model, map_location="cpu", weights_only=False
    )
    if e7_checkpoint["retained_proteins"] != cvae_checkpoint["retained_proteins"]:
        raise ValueError("E7 and CVAE protein axes differ")
    if cvae_checkpoint["frozen_e7_model_sha256"] != sha256_file(args.e7_model):
        raise ValueError("CVAE checkpoint was not trained against this E7 checkpoint")
    if cvae_checkpoint["frozen_e7_core_sha256"] != sha256_file(E7_CORE_PATH):
        raise ValueError("E7 core source differs from the frozen CVAE contract")

    structure_table = E7.load_structure_table(
        args.structure_npz, args.structure_contract
    )
    condition, drug = build_inputs(metadata, e7_checkpoint, structure_table)
    e7_config = e7_checkpoint["config"]
    e7_model = E7.FilmCrossMLP(
        condition_dim=condition.shape[1],
        drug_dim=drug.shape[1],
        output_dim=len(e7_checkpoint["retained_proteins"]),
        encoder_hidden_dim=int(e7_config["encoder_hidden_dim"]),
        latent_dim=int(e7_config["latent_dim"]),
        fusion_dim=int(e7_config["fusion_dim"]),
        dropout=float(e7_config["dropout"]),
    ).to(device)
    e7_model.load_state_dict(e7_checkpoint["state_dict"], strict=True)
    e7_model.eval()

    cvae_config = cvae_checkpoint["config"]
    cvae_model = CVAE.ConditionalResidualVAE(
        context_dim=256,
        output_dim=len(cvae_checkpoint["retained_proteins"]),
        latent_dim=int(cvae_config["latent_dim"]),
        residual_hidden_dim=int(cvae_config["residual_hidden_dim"]),
        distribution_hidden_dim=int(cvae_config["distribution_hidden_dim"]),
        decoder_hidden_dim=int(cvae_config["decoder_hidden_dim"]),
        dropout=float(cvae_config["dropout"]),
        correction_bound=float(cvae_config["correction_bound"]),
    ).to(device)
    cvae_model.load_state_dict(cvae_checkpoint["state_dict"], strict=True)
    cvae_model.eval()
    base = e7_checkpoint["base_profile"].to(torch.float32).to(device)

    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(metadata), args.batch_size):
            stop = start + args.batch_size
            residual, model_audit = e7_model(
                condition[start:stop].to(device),
                drug[start:stop].to(device),
                return_audit=True,
            )
            correction, _ = cvae_model.forward_prior_mean(model_audit["fusion"])
            predictions.append((base + residual + correction).cpu().numpy())
    matrix = np.vstack(predictions)
    proteins = list(map(str, e7_checkpoint["retained_proteins"]))
    output = pd.DataFrame(matrix, columns=proteins)
    output.insert(0, "sample_ID", sample_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, lineterminator="\n")

    audit = {
        "schema_version": "1.0",
        "inference_mode": "frozen E7 plus deterministic CVAE conditional-prior mean",
        "truth_loaded": False,
        "posterior_called": False,
        "sample_count": len(sample_ids),
        "protein_count": len(proteins),
        "treatment_only": bool(args.treatment_only),
        "metadata_sha256": sha256_file(args.metadata),
        "structure_npz_sha256": sha256_file(args.structure_npz),
        "structure_contract_sha256": sha256_file(args.structure_contract),
        "e7_model_sha256": sha256_file(args.e7_model),
        "cvae_model_sha256": sha256_file(args.cvae_model),
        "output_sha256": sha256_file(args.output),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
