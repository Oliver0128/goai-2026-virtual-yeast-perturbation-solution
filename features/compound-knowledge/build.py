from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(raw_dir: Path, names: list[str], key: str) -> list[dict]:
    rows: list[dict] = []
    for name in names:
        rows.extend(json.loads((raw_dir / name).read_text(encoding="utf-8")).get(key, []))
    return rows


def build(structure_contract: Path, protein_contract: Path, raw_dir: Path, output_dir: Path) -> dict:
    structures = pd.read_csv(structure_contract, dtype=str).fillna("")
    proteins = pd.read_csv(protein_contract, dtype=str).fillna("")
    index = json.loads((raw_dir / "collection-index.json").read_text(encoding="utf-8"))
    molecules = load_rows(raw_dir, index["molecule_files"], "molecules")
    mechanisms = load_rows(raw_dir, index["mechanism_files"], "mechanisms")
    activities = load_rows(raw_dir, index["activity_files"], "activities")
    targets = load_rows(raw_dir, index["target_files"], "targets")

    molecules_by_key: dict[str, list[dict]] = {}
    for row in molecules:
        key = (row.get("molecule_structures") or {}).get("standard_inchi_key", "")
        if key:
            molecules_by_key.setdefault(key, []).append(row)
    compound_rows: list[dict] = []
    name_by_chembl: dict[str, str] = {}
    for structure in structures.to_dict("records"):
        matches = molecules_by_key.get(structure["pubchem_inchikey"], [])
        if len(matches) > 1:
            raise ValueError(f"Multiple exact ChEMBL structures for {structure['competition_name']}")
        molecule = matches[0] if matches else {}
        chembl_id = molecule.get("molecule_chembl_id", "")
        if chembl_id:
            name_by_chembl[chembl_id] = structure["competition_name"]
        compound_rows.append(
            {
                "competition_name": structure["competition_name"],
                "pubchem_cid": structure["pubchem_cid"],
                "pubchem_inchikey": structure["pubchem_inchikey"],
                "chembl_mapping_status": "exact_standard_inchikey" if chembl_id else "no_exact_chembl_match",
                "chembl_id": chembl_id,
                "chembl_pref_name": molecule.get("pref_name", "") or "",
                "molecule_type": molecule.get("molecule_type", "") or "",
                "max_phase": molecule.get("max_phase", "") or "",
                "first_approval": molecule.get("first_approval", "") or "",
                "atc_classifications": ";".join(molecule.get("atc_classifications") or []),
                "chemical_probe": molecule.get("chemical_probe", "") if molecule else "",
            }
        )
    compound_contract = pd.DataFrame(compound_rows)

    search_files = {
        "Cisplatin": "unmapped-search-cisplatin.json",
        "Doxycycline hyclate": "unmapped-search-doxycycline-hyclate.json",
        "Nystatin dihydrate": "unmapped-search-nystatin-dihydrate.json",
    }
    candidate_rows: list[dict] = []
    structure_key_by_name = structures.set_index("competition_name")["pubchem_inchikey"].to_dict()
    for competition_name, filename in search_files.items():
        path = raw_dir / filename
        if not path.exists():
            continue
        for rank, molecule in enumerate(json.loads(path.read_text(encoding="utf-8")).get("molecules", []), start=1):
            pref_name = molecule.get("pref_name", "") or ""
            candidate_key = (molecule.get("molecule_structures") or {}).get("standard_inchi_key", "") or ""
            exact_name = pref_name.casefold() == competition_name.casefold()
            candidate_rows.append(
                {
                    "competition_name": competition_name,
                    "search_rank": rank,
                    "candidate_chembl_id": molecule.get("molecule_chembl_id", ""),
                    "candidate_pref_name": pref_name,
                    "candidate_standard_inchikey": candidate_key,
                    "pubchem_standard_inchikey": structure_key_by_name[competition_name],
                    "candidate_relation": "exact_preferred_name" if exact_name else "search_result_only",
                    "standard_inchikey_match": bool(candidate_key and candidate_key == structure_key_by_name[competition_name]),
                    "adoption_status": "not_adopted_without_exact_standard_inchikey",
                }
            )
    candidate_frame = pd.DataFrame(candidate_rows)

    mechanism_rows: list[dict] = []
    for row in mechanisms:
        chembl_id = row.get("molecule_chembl_id", "")
        mechanism_rows.append(
            {
                "competition_name": name_by_chembl.get(chembl_id, ""),
                "chembl_id": chembl_id,
                "mechanism_of_action": row.get("mechanism_of_action", ""),
                "action_type": row.get("action_type", ""),
                "target_chembl_id": row.get("target_chembl_id", ""),
                "target_name": row.get("target_name", ""),
                "direct_interaction": row.get("direct_interaction", ""),
                "mechanism_comment": row.get("mechanism_comment", ""),
                "source": "ChEMBL curated mechanism; organism may be non-yeast",
            }
        )
    mechanism_frame = pd.DataFrame(mechanism_rows).drop_duplicates()

    activity_rows: list[dict] = []
    for row in activities:
        target_organism = (row.get("target_organism") or "").lower()
        assay_organism = (row.get("assay_organism") or "").lower()
        if "saccharomyces cerevisiae" not in target_organism and "saccharomyces cerevisiae" not in assay_organism:
            continue
        chembl_id = row.get("molecule_chembl_id", "")
        activity_rows.append(
            {
                "competition_name": name_by_chembl.get(chembl_id, ""),
                "chembl_id": chembl_id,
                "activity_id": row.get("activity_id", ""),
                "target_chembl_id": row.get("target_chembl_id", ""),
                "target_pref_name": row.get("target_pref_name", ""),
                "target_organism": row.get("target_organism", ""),
                "assay_chembl_id": row.get("assay_chembl_id", ""),
                "assay_organism": row.get("assay_organism", ""),
                "assay_type": row.get("assay_type", ""),
                "standard_type": row.get("standard_type", ""),
                "standard_relation": row.get("standard_relation", ""),
                "standard_value": row.get("standard_value", ""),
                "standard_units": row.get("standard_units", ""),
                "pchembl_value": row.get("pchembl_value", ""),
                "normalized_value_nM": row.get("normalized_value_nM", ""),
                "data_validity_comment": row.get("data_validity_comment", ""),
                "activity_comment": row.get("activity_comment", ""),
                "document_chembl_id": row.get("document_chembl_id", ""),
            }
        )
    activity_frame = pd.DataFrame(activity_rows).drop_duplicates(subset=["activity_id"])

    target_rows: list[dict] = []
    target_accessions: dict[str, list[str]] = {}
    for target in targets:
        target_id = target.get("target_chembl_id", "")
        components = target.get("target_components") or []
        accessions = sorted({component.get("accession", "") for component in components if component.get("accession")})
        target_accessions[target_id] = accessions
        target_rows.append(
            {
                "target_chembl_id": target_id,
                "pref_name": target.get("pref_name", ""),
                "organism": target.get("organism", ""),
                "target_type": target.get("target_type", ""),
                "tax_id": target.get("tax_id", ""),
                "uniprot_accessions": ";".join(accessions),
            }
        )
    target_frame = pd.DataFrame(target_rows).drop_duplicates(subset=["target_chembl_id"])
    protein_by_accession = proteins.set_index("uniprot_accession")["competition_label"].to_dict()
    edge_rows: list[dict] = []
    for activity in activity_frame.to_dict("records"):
        for accession in target_accessions.get(activity["target_chembl_id"], []):
            edge_rows.append(
                {
                    "competition_name": activity["competition_name"],
                    "protein_label": protein_by_accession.get(accession, ""),
                    "uniprot_accession": accession,
                    "target_chembl_id": activity["target_chembl_id"],
                    "activity_id": activity["activity_id"],
                    "standard_type": activity["standard_type"],
                    "standard_relation": activity["standard_relation"],
                    "standard_value": activity["standard_value"],
                    "standard_units": activity["standard_units"],
                    "pchembl_value": activity["pchembl_value"],
                    "direct_competition_axis_match": accession in protein_by_accession,
                }
            )
    edge_frame = pd.DataFrame(edge_rows)
    pathway_columns = [
        "competition_label",
        "reactome_ids",
        "kegg_ids",
        "go_ids",
        "uniprot_pathway",
    ]
    if len(edge_frame):
        pathway_frame = edge_frame.loc[edge_frame["direct_competition_axis_match"].astype(bool)].merge(
            proteins[pathway_columns], left_on="protein_label", right_on="competition_label", how="left"
        )
        pathway_frame = pathway_frame[
            [
                "competition_name",
                "protein_label",
                "uniprot_accession",
                "target_chembl_id",
                "activity_id",
                "reactome_ids",
                "kegg_ids",
                "go_ids",
                "uniprot_pathway",
            ]
        ].drop_duplicates()
    else:
        pathway_frame = pd.DataFrame(
            columns=[
                "competition_name",
                "protein_label",
                "uniprot_accession",
                "target_chembl_id",
                "activity_id",
                "reactome_ids",
                "kegg_ids",
                "go_ids",
                "uniprot_pathway",
            ]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "contract.csv": compound_contract,
        "mapping-candidates.csv": candidate_frame,
        "mechanisms.csv": mechanism_frame,
        "yeast-activities.csv.gz": activity_frame,
        "targets.csv": target_frame,
        "compound-protein-edges.csv.gz": edge_frame,
        "compound-pathways.csv.gz": pathway_frame,
    }
    for name, frame in outputs.items():
        frame.to_csv(
            output_dir / name,
            index=False,
            compression={"method": "gzip", "mtime": 0} if name.endswith(".gz") else None,
            lineterminator="\n",
        )
    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        "chembl_release": "ChEMBL_37",
        "chembl_release_date": "2026-05-01",
        "competition_entity_count": len(compound_contract),
        "exact_chembl_match_count": int(compound_contract["chembl_id"].ne("").sum()),
        "unmapped_compounds": compound_contract.loc[compound_contract["chembl_id"].eq(""), "competition_name"].tolist(),
        "non_adopted_mapping_candidate_rows": len(candidate_frame),
        "curated_mechanism_rows": len(mechanism_frame),
        "yeast_activity_rows": len(activity_frame),
        "yeast_target_rows": len(target_frame),
        "compound_protein_edge_rows": len(edge_frame),
        "direct_competition_axis_edge_rows": int(edge_frame.get("direct_competition_axis_match", pd.Series(dtype=bool)).sum()),
        "compound_pathway_rows": len(pathway_frame),
        "dose_policy": "ChEMBL activity concentrations are external assay measurements and must never be used as the missing competition treatment dose",
        "hashes": {
            "structure_contract_sha256": sha256_file(structure_contract),
            "protein_contract_sha256": sha256_file(protein_contract),
            **{name: sha256_file(output_dir / name) for name in outputs},
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure-contract", type=Path, required=True)
    parser.add_argument("--protein-contract", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.structure_contract, args.protein_contract, args.raw_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
