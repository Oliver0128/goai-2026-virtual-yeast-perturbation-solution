from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .core import sha256_file, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("experiment_dirs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    rows = []
    for directory in args.experiment_dirs:
        manifest_path = directory / "manifest.json"
        metrics_path = directory / "metrics/six-module.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {
            "experiment_id": directory.name,
            "method_id": manifest["method_id"],
            "deployable": manifest["deployable"],
            "diagnostic_only": manifest["diagnostic_only"],
            "total_100": metrics["provisional_weighted_proxy_100"],
            "modules": {
                key: value["normalized_score"] for key, value in metrics["modules"].items()
            },
            "weighted_points": {
                key: value["weighted_points"] for key, value in metrics["modules"].items()
            },
            "manifest_sha256": sha256_file(manifest_path),
            "metrics_sha256": sha256_file(metrics_path),
            "method_details": manifest["method_details"],
        }
        write_json(directory / "result.json", {"schema_version": "1.0", **row})
        rows.append(row)
    deployable = sorted((row for row in rows if row["deployable"]), key=lambda row: row["total_100"], reverse=True)
    diagnostic = sorted((row for row in rows if row["diagnostic_only"]), key=lambda row: row["total_100"], reverse=True)
    write_json(
        args.output,
        {
            "schema_version": "1.0",
            "generated_at": datetime.now().astimezone().isoformat(),
            "score_profile": "current-handbook-sample-mean-v1",
            "ranking_policy": "deployable methods ranked separately from diagnostic-only methods",
            "deployable_ranking": deployable,
            "diagnostic_only": diagnostic,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
