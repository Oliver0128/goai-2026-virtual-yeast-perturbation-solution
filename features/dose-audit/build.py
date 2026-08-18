from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def join_unique(values: pd.Series) -> str:
    return ";".join(sorted({str(value) for value in values.dropna() if str(value) != ""}))


def build(
    metadata_train_val: Path,
    metadata_test: Path,
    compound_contract: Path,
    compound_activities: Path,
    pride_search_dir: Path,
    pride_pxd023613: Path,
    output_dir: Path,
) -> dict:
    train_val = pd.read_csv(metadata_train_val, dtype=str).fillna("")
    test = pd.read_csv(metadata_test, dtype=str).fillna("")
    metadata = pd.concat([train_val.assign(_released_file="train_val"), test.assign(_released_file="test")], ignore_index=True)
    compounds = pd.read_csv(compound_contract, dtype=str).fillna("")
    activities = pd.read_csv(compound_activities, dtype=str).fillna("")
    actual_dose_columns = [
        column
        for column in metadata.columns
        if column != "perturbation_no_concentration"
        and any(token in column.lower() for token in ["dose", "dosage", "concentration", "ic50", "ec50"])
    ]
    if actual_dose_columns:
        raise ValueError(f"Unexpected dose-like metadata columns require manual review: {actual_dose_columns}")

    non_qc = metadata.loc[metadata["perturbation_no_concentration"].ne("Quality Control")].copy()
    activity_count = activities.groupby("competition_name").size().to_dict() if len(activities) else {}
    activity_value_count = (
        activities.loc[activities["standard_value"].ne("")].groupby("competition_name").size().to_dict()
        if len(activities)
        else {}
    )
    rows: list[dict] = []
    for name in compounds["competition_name"]:
        subset = non_qc.loc[non_qc["perturbation_no_concentration"].eq(name)]
        rows.append(
            {
                "competition_name": name,
                "metadata_row_count": len(subset),
                "data_sources": join_unique(subset["data_source"]),
                "pert_ids": join_unique(subset["pert_id"]),
                "pert_id_count": subset["pert_id"].nunique(),
                "perturbation_times": join_unique(subset["pert_time"] + " " + subset["pert_time_unit"]),
                "split_roles": join_unique(subset["split_final"]),
                "competition_dose_status": "not_released",
                "competition_dose_value": "",
                "competition_dose_unit": "",
                "recoverability": "not_recoverable_from_released_metadata_or_molecular_structure",
                "chembl_yeast_activity_count": int(activity_count.get(name, 0)),
                "chembl_yeast_activity_with_value_count": int(activity_value_count.get(name, 0)),
                "external_activity_policy": "mechanistic_prior_only; never substitute for competition treatment dose",
            }
        )
    contract = pd.DataFrame(rows)

    collision_rows: list[dict] = []
    for pert_id, group in metadata.groupby("pert_id"):
        names = sorted(group["perturbation_no_concentration"].unique())
        if len(names) > 1:
            collision_rows.append(
                {
                    "pert_id": pert_id,
                    "chemical_name_count": len(names),
                    "chemical_names": ";".join(names),
                    "data_sources": join_unique(group["data_source"]),
                }
            )
    collisions = pd.DataFrame(collision_rows).sort_values("pert_id")

    pride_search_rows: list[dict] = []
    for path in sorted(pride_search_dir.glob("pride-search-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result_count = len(payload) if isinstance(payload, list) else len(payload.get("content", []))
        pride_search_rows.append(
            {
                "source_kind": "PRIDE Archive exact keyword search",
                "source_id": path.stem,
                "query_or_scope": path.stem.removeprefix("pride-search-").replace("-", " "),
                "result_count": result_count,
                "dose_evidence": "",
                "adoption_status": "no_authoritative_competition_source_found",
                "reason": "No matching PRIDE project was returned by the current search endpoint"
                if result_count == 0
                else "Returned projects require sample-level identity matching before any adoption",
            }
        )
    pxd = json.loads(pride_pxd023613.read_text(encoding="utf-8"))
    protocol = str(pxd.get("sampleProcessingProtocol") or "")
    concentration_match = re.search(r"working concentration of\s+([0-9.]+)\s*([µuμ]M)", protocol, flags=re.IGNORECASE)
    organisms = ";".join(sorted({str(item.get("name") or "") for item in pxd.get("organisms", []) if item.get("name")}))
    instruments = ";".join(sorted({str(item.get("name") or "") for item in pxd.get("instruments", []) if item.get("name")}))
    pride_search_rows.append(
        {
            "source_kind": "PRIDE project manual audit",
            "source_id": pxd.get("accession", "PXD023613"),
            "query_or_scope": pxd.get("title", ""),
            "result_count": 1,
            "dose_evidence": f"{concentration_match.group(1)} {concentration_match.group(2)}" if concentration_match else "",
            "adoption_status": "excluded_not_competition_dose",
            "reason": (
                "Public external prior uses prototrophic BY4741, TripleTOF 6600 and an overnight 10 uM drug screen; "
                "these conditions do not identify any WAYB/WAYC competition sample or its treatment concentration"
            ),
        }
    )
    pride_search_rows.insert(
        0,
        {
            "source_kind": "released competition metadata",
            "source_id": "WAYB_WAYC metadata train_val + test",
            "query_or_scope": "13,412 released rows; all 15 columns audited",
            "result_count": len(metadata),
            "dose_evidence": "",
            "adoption_status": "authoritative_absence_contract",
            "reason": "No explicit treatment dose column; perturbation_no_concentration is a name field and pert_id is not globally unique",
        },
    )
    source_audit = pd.DataFrame(pride_search_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.csv"
    collision_path = output_dir / "pert-id-collisions.csv"
    source_audit_path = output_dir / "source-audit.csv"
    contract.to_csv(contract_path, index=False, lineterminator="\n")
    collisions.to_csv(collision_path, index=False, lineterminator="\n")
    source_audit.to_csv(source_audit_path, index=False, lineterminator="\n")
    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        "released_metadata_rows": len(metadata),
        "released_metadata_column_count": len(train_val.columns),
        "released_metadata_columns": train_val.columns.tolist(),
        "explicit_dose_columns": actual_dose_columns,
        "name_field": "perturbation_no_concentration",
        "competition_compound_count": len(contract),
        "pert_id_unique_count": metadata["pert_id"].nunique(),
        "pert_id_collision_count": len(collisions),
        "dose_conclusion": "No competition treatment concentration is present; it cannot be reconstructed from structure, pert_id, ChEMBL potency, IC50, or public assay concentration",
        "safe_model_policy": "Use compound identity/structure and separately audited mechanism priors; omit competition dose unless an authoritative competition source releases it",
        "source_provenance_audit": {
            "pride_exact_keyword_searches": len(pride_search_rows) - 2,
            "pride_exact_keyword_zero_hit_searches": sum(int(row["result_count"] == 0) for row in pride_search_rows if row["source_kind"].startswith("PRIDE Archive")),
            "pxd023613_external_dose": f"{concentration_match.group(1)} {concentration_match.group(2)}" if concentration_match else "not parsed",
            "pxd023613_organisms": organisms,
            "pxd023613_instruments": instruments,
            "pxd023613_adopted_as_competition_dose": False,
            "conclusion": "The only locally cited PRIDE prior is an external BY4741/TripleTOF experiment and cannot identify WAYB/WAYC sample doses",
        },
        "protein_values_loaded": False,
        "hashes": {
            "metadata_train_val_sha256": sha256_file(metadata_train_val),
            "metadata_test_sha256": sha256_file(metadata_test),
            "compound_contract_sha256": sha256_file(compound_contract),
            "compound_activities_sha256": sha256_file(compound_activities),
            "contract_sha256": sha256_file(contract_path),
            "pert_id_collisions_sha256": sha256_file(collision_path),
            "source_audit_sha256": sha256_file(source_audit_path),
            "pride_pxd023613_sha256": sha256_file(pride_pxd023613),
        },
    }
    manifest["hashes"]["pride_search_snapshot_sha256"] = hashlib.sha256(
        "\n".join(f"{path.name}:{sha256_file(path)}" for path in sorted(pride_search_dir.glob("pride-search-*.json"))).encode()
    ).hexdigest()
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-train-val", type=Path, required=True)
    parser.add_argument("--metadata-test", type=Path, required=True)
    parser.add_argument("--compound-contract", type=Path, required=True)
    parser.add_argument("--compound-activities", type=Path, required=True)
    parser.add_argument("--pride-search-dir", type=Path, required=True)
    parser.add_argument("--pride-pxd023613", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.metadata_train_val,
                args.metadata_test,
                args.compound_contract,
                args.compound_activities,
                args.pride_search_dir,
                args.pride_pxd023613,
                args.output_dir,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
