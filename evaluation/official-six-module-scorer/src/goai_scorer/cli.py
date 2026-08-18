from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import load_config, load_scoring_data, sha256_file
from .evaluate import evaluate_validation


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score frozen validation predictions with the GOAI six-module provisional profile."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--proteome", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-details", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config.resolve())
    data = load_scoring_data(
        metadata_path=args.metadata,
        proteome_path=args.proteome,
        prediction_path=args.prediction,
        config=config,
    )
    artifacts = evaluate_validation(data, config)
    output = args.output.resolve()
    stem = output.name.removesuffix(".json")
    details_dir = output.parent / f"{stem}.details"
    result = dict(artifacts.result)
    result["data_contract"]["config_sha256"] = sha256_file(args.config.resolve())
    result["artifacts"] = {}
    if not args.no_details:
        details_dir.mkdir(parents=True, exist_ok=True)
        sample_path = details_dir / "sample_metrics.csv.gz"
        control_path = details_dir / "control_matches.csv.gz"
        reference_path = details_dir / "train_reference_manifest.json"
        summary_path = details_dir / "score_summary.csv"
        components_path = details_dir / "component_metrics.csv"
        artifacts.sample_metrics.to_csv(sample_path, index=False, compression="gzip")
        artifacts.control_matches.to_csv(control_path, index=False, compression="gzip")
        _write_json(reference_path, artifacts.reference_manifest)
        summary_rows = [
            {
                "row_type": "total",
                "module_id": "total",
                "name": "provisional_weighted_proxy",
                "official_weight": 1.0,
                "raw_score": result["provisional_weighted_raw_unclipped"],
                "proxy_score_before_normalization": np.nan,
                "normalized_score": result["provisional_weighted_proxy"],
                "weighted_points": result["provisional_weighted_proxy_100"],
            }
        ]
        component_rows = []
        for module_id, module in result["modules"].items():
            summary_rows.append(
                {
                    "row_type": "module",
                    "module_id": module_id,
                    "name": module["name"],
                    "official_weight": module["official_weight"],
                    "raw_score": module["raw_score"],
                    "proxy_score_before_normalization": module[
                        "proxy_score_before_normalization"
                    ],
                    "normalized_score": module["normalized_score"],
                    "weighted_points": module["weighted_points"],
                }
            )
            for component, raw_value in module["internal_values"].items():
                component_rows.append(
                    {
                        "module_id": module_id,
                        "module_name": module["name"],
                        "component": component,
                        "raw_value": raw_value,
                        "proxy_value": module["internal_proxy_values"][component],
                        "normalized_value": module["normalized_components"][component],
                        "internal_weight": module["internal_weights"][component],
                    }
                )
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        pd.DataFrame(component_rows).to_csv(components_path, index=False)
        result["artifacts"] = {
            "sample_metrics": str(sample_path),
            "sample_metrics_sha256": sha256_file(sample_path),
            "control_matches": str(control_path),
            "control_matches_sha256": sha256_file(control_path),
            "train_reference_manifest": str(reference_path),
            "train_reference_manifest_sha256": sha256_file(reference_path),
            "score_summary": str(summary_path),
            "score_summary_sha256": sha256_file(summary_path),
            "component_metrics": str(components_path),
            "component_metrics_sha256": sha256_file(components_path),
        }
    _write_json(output, result)
    # Hash the final result only after it has been written; store it in a sidecar to avoid self-reference.
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    print(f"profile={result['profile']}")
    print(f"provisional_weighted_proxy_100={result['provisional_weighted_proxy_100']:.6f}")
    for module_id, module in result["modules"].items():
        print(
            f"{module_id}={module['normalized_score']:.6f} "
            f"weighted_points={module['weighted_points']:.6f}"
        )
    print(f"result_json={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
