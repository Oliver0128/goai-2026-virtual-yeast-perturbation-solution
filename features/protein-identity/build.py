from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

EXPLICIT_ALIASES = {
    "1-Oct": {
        "canonical_gene_symbol": "OCT1",
        "reason": "probable spreadsheet date conversion; OCT1 is the sole evidence-backed repair",
    }
}
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
GAF_COLUMNS = [
    "db",
    "db_object_id",
    "db_object_symbol",
    "qualifier",
    "go_id",
    "reference",
    "evidence_code",
    "with_or_from",
    "aspect_code",
    "db_object_name",
    "synonyms",
    "db_object_type",
    "taxon",
    "date",
    "assigned_by",
    "annotation_extension",
    "gene_product_form_id",
]
ASPECT_NAMES = {"P": "biological_process", "F": "molecular_function", "C": "cellular_component"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def first_token(value: str) -> str:
    return str(value).strip().rstrip(";").split(";")[0]


def parse_go_field(value: str, aspect: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in str(value).split(";"):
        item = item.strip()
        if not item:
            continue
        match = re.fullmatch(r"(.+?) \[(GO:\d+)\]", item)
        if match:
            rows.append({"aspect": aspect, "go_id": match.group(2), "go_term": match.group(1)})
    return rows


def load_evidence_annotations(
    sgd_gaf: Path,
    quickgo_dir: Path,
    contract: pd.DataFrame,
    go_term_lookup: dict[str, str],
) -> tuple[pd.DataFrame, dict]:
    by_sgd = contract.loc[contract["sgd_id"].ne("")].set_index("sgd_id").to_dict("index")
    by_uniprot = contract.set_index("uniprot_accession").to_dict("index")
    rows: list[dict] = []
    with gzip.open(sgd_gaf, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("!"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < len(GAF_COLUMNS):
                fields += [""] * (len(GAF_COLUMNS) - len(fields))
            record = dict(zip(GAF_COLUMNS, fields, strict=True))
            protein = by_sgd.get(record["db_object_id"])
            if protein is None:
                continue
            rows.append(
                {
                    "competition_label": protein["competition_label"],
                    "canonical_gene_symbol": protein["canonical_gene_symbol"],
                    "uniprot_accession": protein["uniprot_accession"],
                    "sgd_id": record["db_object_id"],
                    "source_database": "GOA SGD GAF",
                    "qualifier": record["qualifier"],
                    "go_id": record["go_id"],
                    "go_term": go_term_lookup.get(record["go_id"], ""),
                    "aspect": ASPECT_NAMES.get(record["aspect_code"], record["aspect_code"]),
                    "evidence_code": record["evidence_code"],
                    "reference": record["reference"],
                    "with_or_from": record["with_or_from"],
                    "assigned_by": record["assigned_by"],
                    "annotation_date": record["date"],
                    "annotation_extension": record["annotation_extension"],
                }
            )

    quickgo_counts: dict[str, int] = {}
    for path in sorted(quickgo_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        quickgo_counts[path.name] = int(payload.get("numberOfHits", len(payload.get("results", []))))
        for record in payload.get("results", []):
            accession = str(record.get("geneProductId", "")).removeprefix("UniProtKB:")
            protein = by_uniprot.get(accession)
            if protein is None:
                continue
            with_from = record.get("withFrom") or []
            rows.append(
                {
                    "competition_label": protein["competition_label"],
                    "canonical_gene_symbol": protein["canonical_gene_symbol"],
                    "uniprot_accession": accession,
                    "sgd_id": protein["sgd_id"],
                    "source_database": "QuickGO API spot check",
                    "qualifier": record.get("qualifier") or "",
                    "go_id": record.get("goId") or "",
                    "go_term": record.get("goName") or go_term_lookup.get(record.get("goId") or "", ""),
                    "aspect": record.get("goAspect") or "",
                    "evidence_code": record.get("evidenceCode") or "",
                    "reference": record.get("reference") or "",
                    "with_or_from": json.dumps(with_from, ensure_ascii=False, separators=(",", ":")),
                    "assigned_by": record.get("assignedBy") or "",
                    "annotation_date": record.get("date") or "",
                    "annotation_extension": json.dumps(record.get("extensions") or [], ensure_ascii=False, separators=(",", ":")),
                }
            )
    evidence = pd.DataFrame(rows).drop_duplicates().sort_values(
        ["competition_label", "go_id", "evidence_code", "reference", "source_database"]
    )
    return evidence, quickgo_counts


def build(
    proteome_csv: Path,
    uniprot_tsv: Path,
    sgd_gaf: Path,
    quickgo_dir: Path,
    output_dir: Path,
) -> dict:
    with proteome_csv.open(newline="", encoding="utf-8-sig") as handle:
        protein_labels = next(csv.reader(handle))[1:]
    if len(protein_labels) != 5243 or len(set(protein_labels)) != len(protein_labels):
        raise ValueError("Official protein axis must contain 5,243 unique labels")

    uniprot = pd.read_csv(uniprot_tsv, sep="\t", dtype=str).fillna("")
    if len(uniprot) != 6733:
        raise ValueError(f"Unexpected reviewed S288C UniProt row count: {len(uniprot)}")
    primary: dict[str, list[int]] = {}
    aliases: dict[str, list[int]] = {}
    for index, row in uniprot.iterrows():
        gene_primary = row["Gene Names (primary)"].strip()
        if gene_primary:
            primary.setdefault(gene_primary, []).append(index)
        for alias in row["Gene Names"].split():
            aliases.setdefault(alias, []).append(index)

    contract_rows: list[dict] = []
    go_rows: list[dict] = []
    sequence_matrix: list[np.ndarray] = []
    sequences: list[tuple[str, str]] = []
    for position, competition_label in enumerate(protein_labels):
        query = competition_label
        resolution = "primary_exact"
        note = ""
        if competition_label in EXPLICIT_ALIASES:
            query = EXPLICIT_ALIASES[competition_label]["canonical_gene_symbol"]
            resolution = "explicit_alias"
            note = EXPLICIT_ALIASES[competition_label]["reason"]
        candidates = primary.get(query, [])
        if len(candidates) != 1:
            candidates = aliases.get(query, [])
            if len(candidates) == 1 and resolution != "explicit_alias":
                resolution = "unique_uniprot_alias"
        if len(candidates) != 1:
            raise ValueError(f"Protein {competition_label!r} maps to {len(candidates)} UniProt rows")
        row = uniprot.loc[candidates[0]]
        sequence = row["Sequence"].strip()
        if not sequence or any(amino_acid not in set(AA_ORDER) for amino_acid in sequence):
            raise ValueError(f"Invalid canonical sequence for {competition_label}")
        canonical = row["Gene Names (primary)"].strip() or query
        sequence_sha = hashlib.sha256(sequence.encode()).hexdigest()
        composition = np.asarray([sequence.count(aa) / len(sequence) for aa in AA_ORDER], dtype=np.float32)
        sequence_matrix.append(np.concatenate([composition, np.asarray([np.log1p(len(sequence))], dtype=np.float32)]))
        sequences.append((competition_label, sequence))
        contract_rows.append(
            {
                "protein_position": position,
                "competition_label": competition_label,
                "canonical_gene_symbol": canonical,
                "resolution": resolution,
                "resolution_note": note,
                "uniprot_accession": row["Entry"],
                "uniprot_entry_name": row["Entry Name"],
                "sgd_id": first_token(row["SGD"]),
                "systematic_name": next((name for name in row["Gene Names"].split() if re.fullmatch(r"Y[A-P][LR]\d{3}[CW](?:-[A-Z])?", name)), ""),
                "protein_name": row["Protein names"],
                "sequence_length": int(row["Length"]),
                "sequence_sha256": sequence_sha,
                "string_id": first_token(row["STRING"]),
                "biogrid_id": first_token(row["BioGRID"]),
                "reactome_ids": row["Reactome"].rstrip(";"),
                "kegg_ids": row["KEGG"].rstrip(";"),
                "go_ids": row["Gene Ontology IDs"],
                "uniprot_pathway": row["Pathway"],
            }
        )
        for source_field, aspect in [
            ("Gene Ontology (biological process)", "biological_process"),
            ("Gene Ontology (molecular function)", "molecular_function"),
            ("Gene Ontology (cellular component)", "cellular_component"),
        ]:
            for annotation in parse_go_field(row[source_field], aspect):
                go_rows.append(
                    {
                        "competition_label": competition_label,
                        "canonical_gene_symbol": canonical,
                        "uniprot_accession": row["Entry"],
                        **annotation,
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    contract = pd.DataFrame(contract_rows)
    contract_path = output_dir / "contract.csv"
    contract.to_csv(contract_path, index=False, lineterminator="\n")
    go_path = output_dir / "go-annotations.csv.gz"
    pd.DataFrame(go_rows).drop_duplicates().sort_values(
        ["protein_position"] if "protein_position" in pd.DataFrame(go_rows).columns else ["competition_label", "aspect", "go_id"]
    ).to_csv(go_path, index=False, compression={"method": "gzip", "mtime": 0}, lineterminator="\n")
    go_term_lookup = {
        row["go_id"]: row["go_term"] for row in pd.DataFrame(go_rows).drop_duplicates().to_dict("records")
    }
    evidence, quickgo_counts = load_evidence_annotations(sgd_gaf, quickgo_dir, contract, go_term_lookup)
    evidence_path = output_dir / "go-annotations-evidence.csv.gz"
    evidence.to_csv(
        evidence_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
        lineterminator="\n",
    )
    fasta_path = output_dir / "sequences.fasta.gz"
    with (
        fasta_path.open("wb") as raw_handle,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gzip_handle,
        io.TextIOWrapper(gzip_handle, encoding="utf-8", newline="\n") as handle,
    ):
        for label, sequence in sequences:
            handle.write(f">{label}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")
    feature_path = output_dir / "sequence-features.npz"
    np.savez_compressed(
        feature_path,
        competition_labels=np.asarray(protein_labels, dtype=str),
        feature_names=np.asarray([f"aa_fraction_{aa}" for aa in AA_ORDER] + ["log1p_sequence_length"], dtype=str),
        features=np.vstack(sequence_matrix).astype(np.float32),
    )
    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        "entity_count": len(contract),
        "organism": "Saccharomyces cerevisiae S288C",
        "ncbi_taxon_id": 559292,
        "uniprot_reviewed_source_rows": len(uniprot),
        "resolution_counts": contract["resolution"].value_counts().sort_index().to_dict(),
        "go_annotation_rows": len(pd.DataFrame(go_rows).drop_duplicates()),
        "go_evidence_annotation_rows": len(evidence),
        "go_evidence_protein_coverage": int(evidence["competition_label"].nunique()),
        "go_evidence_source_counts": evidence["source_database"].value_counts().sort_index().to_dict(),
        "go_evidence_code_counts": evidence["evidence_code"].value_counts().sort_index().to_dict(),
        "quickgo_spotcheck": {
            "purpose": "Independent API verification of OCT1 alias and the three UniProt-mapped transposon proteins lacking SGD cross-references, followed by targeted accession queries for every GOA-GAF coverage gap; not a claim of a full-axis QuickGO crawl",
            "result_counts": quickgo_counts,
            "targeted_gap_query_count": sum(path.name.endswith("-gap-audit.json") for path in quickgo_dir.glob("*.json")),
            "targeted_gap_zero_hit_files": sorted(
                name for name, count in quickgo_counts.items() if name.endswith("-gap-audit.json") and count == 0
            ),
        },
        "cross_reference_coverage": {
            column: int(contract[column].astype(str).str.strip().ne("").sum())
            for column in ["sgd_id", "string_id", "biogrid_id", "reactome_ids", "kegg_ids", "go_ids", "uniprot_pathway"]
        },
        "sequence_feature_contract": {"amino_acid_order": AA_ORDER, "feature_count": 21, "scaling": "none"},
        "explicit_aliases": EXPLICIT_ALIASES,
        "data_boundary": {"protein_values_loaded": False, "protein_header_only": True, "learned_statistics": False},
        "hashes": {
            "official_protein_axis_sha256": sha256_text("\n".join(protein_labels) + "\n"),
            "uniprot_raw_sha256": sha256_file(uniprot_tsv),
            "contract_sha256": sha256_file(contract_path),
            "go_annotations_sha256": sha256_file(go_path),
            "go_evidence_annotations_sha256": sha256_file(evidence_path),
            "sgd_gaf_sha256": sha256_file(sgd_gaf),
            "sequences_sha256": sha256_file(fasta_path),
            "sequence_features_sha256": sha256_file(feature_path),
        },
    }
    manifest["hashes"]["quickgo_spotcheck_sha256"] = sha256_text(
        "\n".join(f"{path.name}:{sha256_file(path)}" for path in sorted(quickgo_dir.glob("*.json"))) + "\n"
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proteome-csv", type=Path, required=True)
    parser.add_argument("--uniprot-tsv", type=Path, required=True)
    parser.add_argument("--sgd-gaf", type=Path, required=True)
    parser.add_argument("--quickgo-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.proteome_csv, args.uniprot_tsv, args.sgd_gaf, args.quickgo_dir, args.output_dir),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
