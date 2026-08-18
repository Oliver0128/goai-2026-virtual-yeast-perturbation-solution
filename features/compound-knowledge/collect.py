from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def run_wrapper(wrapper: Path, raw_dir: Path, endpoint: str, name: str, arguments: list[str]) -> Path:
    output = raw_dir / f"{name}.json"
    command = ["uv", "run", str(wrapper), endpoint, *arguments, "--output", str(output)]
    subprocess.run(command, check=True)
    return output


def collect_list_endpoint(
    wrapper: Path,
    raw_dir: Path,
    endpoint: str,
    prefix: str,
    filters: list[str],
    result_key: str,
    limit: int = 1000,
    normalize: bool = False,
) -> list[Path]:
    paths: list[Path] = []
    offset = 0
    while True:
        arguments: list[str] = ["--filter", *filters] if filters else []
        arguments.extend(["--limit", str(limit), "--offset", str(offset)])
        if normalize:
            arguments.append("--normalize")
        path = run_wrapper(wrapper, raw_dir, endpoint, f"{prefix}-offset-{offset:06d}", arguments)
        paths.append(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        count = len(payload.get(result_key, []))
        total = int(payload.get("page_meta", {}).get("total_count", count))
        offset += count
        if count == 0 or offset >= total:
            break
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure-contract", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    args = parser.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    contract = pd.read_csv(args.structure_contract, dtype=str).fillna("")

    molecule_paths: list[Path] = []
    for batch_number, keys in enumerate(chunks(contract["pubchem_inchikey"].tolist(), 20)):
        molecule_paths.extend(
            collect_list_endpoint(
                args.wrapper,
                args.raw_dir,
                "molecule",
                f"molecules-batch-{batch_number:02d}",
                [f"molecule_structures__standard_inchi_key__in={','.join(keys)}"],
                "molecules",
                limit=100,
            )
        )
    molecule_rows = []
    for path in molecule_paths:
        molecule_rows.extend(json.loads(path.read_text(encoding="utf-8")).get("molecules", []))
    key_to_ids: dict[str, list[str]] = {}
    for molecule in molecule_rows:
        key = (molecule.get("molecule_structures") or {}).get("standard_inchi_key", "")
        if key:
            key_to_ids.setdefault(key, []).append(molecule["molecule_chembl_id"])
    mapped_ids = sorted({identifier for values in key_to_ids.values() for identifier in values})

    mechanism_paths: list[Path] = []
    activity_paths: list[Path] = []
    for batch_number, ids in enumerate(chunks(mapped_ids, 20)):
        id_filter = f"molecule_chembl_id__in={','.join(ids)}"
        mechanism_paths.extend(
            collect_list_endpoint(
                args.wrapper, args.raw_dir, "mechanism", f"mechanisms-batch-{batch_number:02d}", [id_filter], "mechanisms"
            )
        )
        activity_paths.extend(
            collect_list_endpoint(
                args.wrapper,
                args.raw_dir,
                "activity",
                f"activities-all-batch-{batch_number:02d}",
                [id_filter],
                "activities",
                normalize=True,
            )
        )

    target_ids: set[str] = set()
    for path in activity_paths:
        for activity in json.loads(path.read_text(encoding="utf-8")).get("activities", []):
            target_organism = (activity.get("target_organism") or "").lower()
            assay_organism = (activity.get("assay_organism") or "").lower()
            if "saccharomyces cerevisiae" not in target_organism and "saccharomyces cerevisiae" not in assay_organism:
                continue
            if activity.get("target_chembl_id"):
                target_ids.add(activity["target_chembl_id"])
    target_paths: list[Path] = []
    for batch_number, ids in enumerate(chunks(sorted(target_ids), 40)):
        target_paths.append(
            run_wrapper(
                args.wrapper,
                args.raw_dir,
                "target",
                f"targets-batch-{batch_number:02d}",
                ["--ids", ";".join(ids), "--limit", "1000"],
            )
        )
    index = {
        "molecule_files": [path.name for path in molecule_paths],
        "mechanism_files": [path.name for path in mechanism_paths],
        "activity_files": [path.name for path in activity_paths],
        "target_files": [path.name for path in target_paths],
        "mapped_chembl_ids": mapped_ids,
        "exact_inchikey_coverage": sum(key in key_to_ids for key in contract["pubchem_inchikey"]),
        "competition_entity_count": len(contract),
    }
    (args.raw_dir / "collection-index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(index, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
