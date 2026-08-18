from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .core import load_baseline_data, sha256_file, write_json
from .methods import METHODS


def build_parser(default_method: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(METHODS), default=default_method, required=default_method is None)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--proteome", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(default_method: str | None = None, argv: list[str] | None = None) -> int:
    args = build_parser(default_method).parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    method = args.method or default_method
    if config["method_id"] != method:
        raise ValueError("Config method_id does not match runner")
    data = load_baseline_data(args.metadata, args.proteome, config["missing_rate_threshold"])
    prediction, details = METHODS[method](data, config)
    prediction = prediction.loc[data.target_ids, data.retained_proteins]
    if prediction.shape != (len(data.target_ids), len(data.retained_proteins)):
        raise ValueError("Prediction shape mismatch")
    if not np.isfinite(prediction.to_numpy()).all():
        raise ValueError("Predictions must be finite")
    args.prediction.parent.mkdir(parents=True, exist_ok=True)
    prediction.rename_axis("sample_ID").reset_index().to_csv(args.prediction, index=False)
    manifest = {
        "schema_version": "1.0",
        "method_id": method,
        "variant_id": config.get("variant_id", "default"),
        "deployable": bool(config["deployable"]),
        "diagnostic_only": bool(config["diagnostic_only"]),
        "generated_at": datetime.now().astimezone().isoformat(),
        "data_contract": {
            **data.hashes,
            "train_rows": len(data.train_ids),
            "validation_target_rows": len(data.target_ids),
            "retained_proteins": len(data.retained_proteins),
            "missing_rate_threshold": config["missing_rate_threshold"],
            "test_proteome_loaded": False,
        },
        "config_sha256": sha256_file(args.config),
        "prediction_sha256": sha256_file(args.prediction),
        "prediction_bytes": args.prediction.stat().st_size,
        "method_details": details,
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0
