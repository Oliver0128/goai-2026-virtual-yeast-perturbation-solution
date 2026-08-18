from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build(features_root: Path, output_dir: Path) -> dict:
    strain = load(features_root / "strain-genome" / "manifest.json")
    protein = load(features_root / "protein-identity" / "manifest.json")
    string = load(features_root / "protein-identity" / "string-manifest.json")
    structure = load(features_root / "compound-structure" / "manifest.json")
    compound = load(features_root / "compound-knowledge" / "manifest.json")
    dose = load(features_root / "dose-audit" / "manifest.json")

    if protein["data_boundary"]["protein_values_loaded"]:
        raise ValueError("Protein identity build unexpectedly loaded protein values")
    if not protein["data_boundary"]["protein_header_only"]:
        raise ValueError("Protein output axis was not header-only")
    if structure["data_contract"]["proteome_truth_loaded"] or dose["protein_values_loaded"]:
        raise ValueError("External feature build touched competition proteome truth")
    if dose["explicit_dose_columns"]:
        raise ValueError("Dose audit requires manual review")

    use_policy = pd.DataFrame(
        [
            {
                "artifact": "compound-structure/fingerprints.npz",
                "recommended_tier": "primary_candidate",
                "coverage": "56/56 compounds",
                "required_guard": "scale using train rows only; preserve curated parent/salt caveats",
            },
            {
                "artifact": "strain-genome/population-distance-pca.npz",
                "recommended_tier": "primary_candidate_with_missing_mask",
                "coverage": "5/6 strains",
                "required_guard": "never impute DHY210 from validation/test labels",
            },
            {
                "artifact": "strain-genome/genomic-features.npz",
                "recommended_tier": "primary_candidate_with_ablation",
                "coverage": "5/6 strains",
                "required_guard": "public entity data only; handle missing DHY210 explicitly",
            },
            {
                "artifact": "strain-genome/strain-proteome-features.npz",
                "recommended_tier": "primary_candidate_with_missing_mask_and_ablation",
                "coverage": "BAH 5168; BAI 5168; CEK 5159; CGD 4921; CRD 5162; DHY210 0 of 5243",
                "required_guard": "Peter-2018 SNP-inferred source; preserve source caveats and never synthesize DHY210",
            },
            {
                "artifact": "strain-genome/phenotype-features-diagnostic.npz",
                "recommended_tier": "diagnostic_only",
                "coverage": "5/6 strains x 35 conditions",
                "required_guard": "do not use in primary model without separate leakage/compliance approval",
            },
            {
                "artifact": "protein-identity/sequence-features.npz",
                "recommended_tier": "primary_candidate",
                "coverage": "5243/5243 proteins",
                "required_guard": "downstream scaling fitted on train rows only",
            },
            {
                "artifact": "protein-identity/go-annotations.csv.gz",
                "recommended_tier": "primary_candidate_with_ablation",
                "coverage": f"{protein['cross_reference_coverage']['go_ids']}/5243 proteins",
                "required_guard": "record UniProt/QuickGO snapshot and ontology semantics",
            },
            {
                "artifact": "protein-identity/go-annotations-evidence.csv.gz",
                "recommended_tier": "primary_candidate_with_evidence_filtering_and_ablation",
                "coverage": f"{protein['go_evidence_protein_coverage']}/5243 proteins; {protein['go_evidence_annotation_rows']} evidence rows",
                "required_guard": "retain evidence code/reference/source; evaluate experimental-only and all-evidence variants separately",
            },
            {
                "artifact": "protein-identity/ppi-physical-edges.tsv.gz",
                "recommended_tier": "primary_candidate_with_ablation",
                "coverage": f"{string['covered_protein_count']}/5243 proteins; {string['edge_count']} edges",
                "required_guard": "STRING v12 physical network, score >=400; no competition values",
            },
            {
                "artifact": "compound-knowledge/mechanisms.csv",
                "recommended_tier": "mechanistic_prior_or_sensitivity",
                "coverage": f"{compound['curated_mechanism_rows']} rows",
                "required_guard": "organism may be non-yeast; do not claim direct yeast causality",
            },
            {
                "artifact": "compound-knowledge/compound-protein-edges.csv.gz",
                "recommended_tier": "mechanistic_prior_with_ablation",
                "coverage": f"{compound['direct_competition_axis_edge_rows']} yeast-axis evidence rows",
                "required_guard": "keep assay provenance; do not convert activity concentration into competition dose",
            },
            {
                "artifact": "dose-audit/contract.csv",
                "recommended_tier": "prohibition_contract",
                "coverage": "56/56 compounds",
                "required_guard": "competition dose remains missing; no IC50/potency substitution",
            },
            {
                "artifact": "dose-audit/source-audit.csv",
                "recommended_tier": "audit_only",
                "coverage": "released metadata + 3 exact PRIDE searches + PXD023613 manual exclusion",
                "required_guard": "PXD023613 10 uM is BY4741/TripleTOF external evidence and is never a WAYB/WAYC dose",
            },
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "use-policy.csv"
    use_policy.to_csv(policy_path, index=False, lineterminator="\n")
    summary = {
        "schema_version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        "verdict": "pass_with_documented_gaps",
        "data_boundary": {
            "validation_or_test_proteome_values_loaded": False,
            "protein_axis_access": "header_only",
            "released_metadata_used": True,
            "external_public_entity_data_used": True,
        },
        "strain": {
            "competition_count": len(strain["competition_strains"]),
            "exact_1011_mapping_count": len(strain["exact_1011_project_mappings"]),
            "population_embedding_coverage": strain["population_embedding"]["competition_coverage"],
            "exact_ncbi_assembly_count": 2,
            "dhy210_exact_genotype_resolved": False,
            "dhy210_proxy": "S288C R64 GCF_000146045.2",
            "strain_proteome_axis_coverage": strain["strain_proteome"]["per_strain_axis_coverage"],
            "strain_proteome_source": strain["strain_proteome"]["source"],
            "exact_ena_assembly_unresolved": strain["identity_search_audit"]["ena_exact_unresolved"],
        },
        "protein": {
            "axis_count": protein["entity_count"],
            "uniprot_mapping_count": protein["entity_count"],
            "sgd_mapping_count": protein["cross_reference_coverage"]["sgd_id"],
            "go_mapping_count": protein["cross_reference_coverage"]["go_ids"],
            "go_evidence_mapping_count": protein["go_evidence_protein_coverage"],
            "go_evidence_annotation_rows": protein["go_evidence_annotation_rows"],
            "sequence_count": protein["entity_count"],
            "explicit_aliases": protein["explicit_aliases"],
            "ppi_edge_count": string["edge_count"],
            "ppi_protein_coverage": string["covered_protein_count"],
        },
        "compound": {
            "entity_count": compound["competition_entity_count"],
            "exact_pubchem_structure_count": structure["data_contract"]["entity_count"],
            "exact_chembl_match_count": compound["exact_chembl_match_count"],
            "unmapped_chembl": compound["unmapped_compounds"],
            "curated_mechanism_rows": compound["curated_mechanism_rows"],
            "yeast_activity_rows": compound["yeast_activity_rows"],
            "direct_compound_protein_rows": compound["direct_competition_axis_edge_rows"],
        },
        "dose": {
            "explicit_column_count": len(dose["explicit_dose_columns"]),
            "pert_id_collision_count": dose["pert_id_collision_count"],
            "recoverable": False,
            "conclusion": dose["dose_conclusion"],
            "pride_provenance_audit": dose["source_provenance_audit"],
        },
        "remaining_gaps": [
            "DHY210 exact accession, private variants, and exact genome remain unavailable",
            "CEK/CGD/CRD exact NCBI and ENA assemblies were not found by isolate name; their Peter-2018 strain proteomes are available but are not genome assemblies",
            "Three salt/solvate/inorganic compounds lack exact ChEMBL standard-InChIKey matches",
            "Curated mechanism and direct yeast target coverage are sparse and must be ablated",
            "Competition treatment dose is not released and is not reconstructible",
            f"{protein['entity_count'] - protein['go_evidence_protein_coverage']} protein-axis entries still lack an evidence-bearing GO annotation after targeted QuickGO gap queries",
        ],
        "license_review_required": ["UniProt", "QuickGO", "STRING", "ChEMBL"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.features_root, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
