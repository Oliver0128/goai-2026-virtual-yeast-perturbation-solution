from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

PROJECT_STRAINS = ["BAH", "BAI", "CEK", "CGD", "CRD"]
ALL_COMPETITION_STRAINS = PROJECT_STRAINS + ["DHY210"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_collection_rows(supplement: Path) -> pd.DataFrame:
    table = pd.read_excel(supplement, sheet_name="Table S1", header=3, dtype=str).fillna("")
    table = table.loc[table["Standardized name"].isin(PROJECT_STRAINS)].copy()
    if set(table["Standardized name"]) != set(PROJECT_STRAINS):
        raise ValueError("Peter 2018 supplement does not cover all five 1011-project strains")
    return table.set_index("Standardized name")


def load_ncbi_reports(paths: list[Path]) -> list[dict]:
    reports: list[dict] = []
    for path in paths:
        reports.extend(json.loads(path.read_text(encoding="utf-8")).get("reports", []))
    return reports


def exact_isolate_assembly(reports: list[dict], isolate_name: str) -> dict:
    matches: list[dict] = []
    for report in reports:
        organism = report.get("organism") or {}
        assembly_info = report.get("assembly_info") or {}
        biosample = assembly_info.get("biosample") or {}
        strain_names = {
            str((organism.get("infraspecific_names") or {}).get("strain") or ""),
            str(biosample.get("strain") or ""),
        }
        if isolate_name in strain_names:
            matches.append(report)
    preferred = [
        report
        for report in matches
        if (report.get("assembly_info") or {}).get("bioproject_accession") == "PRJNA396809"
    ]
    return (preferred or matches or [{}])[0]


def load_1011_proteomes(archive_path: Path) -> dict[str, dict[str, str]]:
    proteomes: dict[str, dict[str, str]] = {strain: {} for strain in PROJECT_STRAINS}

    def add_record(header: str, sequence_chunks: list[str]) -> None:
        token = header.split()[0]
        matched = next((strain for strain in PROJECT_STRAINS if token.startswith(f"{strain}_")), None)
        if matched is None:
            return
        systematic_name = token[len(matched) + 1 :].split("_", 1)[0]
        sequence = "".join(sequence_chunks).strip().rstrip("*")
        if not systematic_name or not sequence:
            raise ValueError(f"Invalid 1011-proteome record: {header!r}")
        previous = proteomes[matched].get(systematic_name)
        if previous is not None and previous != sequence:
            raise ValueError(f"Non-unique 1011-proteome sequence for {matched}/{systematic_name}")
        proteomes[matched][systematic_name] = sequence

    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".fasta"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"Unable to read archive member {member.name}")
            header = ""
            sequence_chunks: list[str] = []
            for raw_line in extracted:
                line = raw_line.decode("utf-8").strip()
                if line.startswith(">"):
                    if header:
                        add_record(header, sequence_chunks)
                    header = line[1:]
                    sequence_chunks = []
                elif line:
                    sequence_chunks.append(line)
            if header:
                add_record(header, sequence_chunks)
    return proteomes


def build(
    raw_dir: Path,
    protein_contract_path: Path,
    ena_audit_dir: Path,
    dhy210_literature_search: Path,
    output_dir: Path,
) -> dict:
    supplement = raw_dir / "peter-2018-supplementary-tables.xls"
    snp_path = raw_dir / "1011DistanceMatrixBasedOnSNPs.tab.gz"
    orf_path = raw_dir / "1011DistanceMatrixBasedOnORFs.tab.gz"
    presence_path = raw_dir / "genesMatrix_PresenceAbsence.tab.gz"
    copy_path = raw_dir / "genesMatrix_CopyNumber.tab.gz"
    frameshift_path = raw_dir / "genesMatrix_Frameshift.tab.gz"
    phenotype_path = raw_dir / "phenoMatrix_35ConditionsNormalizedByYPD.tab.gz"
    proteome_archive_path = raw_dir / "1011proteome.tar.gz"
    ncbi_dir = raw_dir / "ncbi"
    ncbi_species_pages = [
        ncbi_dir / "scerevisiae-assembly-report-page1.json",
        ncbi_dir / "scerevisiae-assembly-report-page2.json",
    ]
    s288c_report_path = ncbi_dir / "s288c-assembly-report.json"
    sources = [
        supplement,
        snp_path,
        orf_path,
        presence_path,
        copy_path,
        frameshift_path,
        phenotype_path,
        proteome_archive_path,
        *ncbi_species_pages,
        s288c_report_path,
    ]
    for path in sources:
        if not path.exists():
            raise FileNotFoundError(path)

    collection = load_collection_rows(supplement)
    ncbi_reports = load_ncbi_reports(ncbi_species_pages)
    contract_rows: list[dict] = []
    for strain in PROJECT_STRAINS:
        row = collection.loc[strain]
        ncbi = exact_isolate_assembly(ncbi_reports, row["Isolate name"])
        assembly_info = ncbi.get("assembly_info") or {}
        biosample = assembly_info.get("biosample") or {}
        wgs_info = ncbi.get("wgs_info") or {}
        contract_rows.append(
            {
                "competition_strain": strain,
                "identity_status": "exact_1011_project_standardized_name",
                "isolate_name": row["Isolate name"],
                "species": "Saccharomyces cerevisiae",
                "ncbi_species_taxon_id": 4932,
                "ecological_origin": row["Ecological origins"],
                "isolation_material": row["Isolation"],
                "geographical_origin": row["Geographical origins"],
                "clade": row["Clades"],
                "ploidy": row["Ploidy"],
                "zygosity": row["Zygosity"],
                "total_snps": row["Total number of SNPs"],
                "ncbi_assembly_accession": ncbi.get("accession", ""),
                "ncbi_biosample_accession": biosample.get("accession", ""),
                "ncbi_wgs_accession": wgs_info.get("wgs_project_accession", ""),
                "sgd_accession": "",
                "proxy_reference": "",
                "model_policy": "public 1011-project genomic features permitted; NCBI assembly only when exact isolate evidence exists",
            }
        )
    contract_rows.append(
        {
            "competition_strain": "DHY210",
            "identity_status": "official_tutorial_s288c_derived_exact_dhy210_genotype_unresolved",
            "isolate_name": "",
            "species": "Saccharomyces cerevisiae",
            "ncbi_species_taxon_id": 4932,
            "ecological_origin": "",
            "isolation_material": "",
            "geographical_origin": "",
            "clade": "",
            "ploidy": "",
            "zygosity": "",
            "total_snps": "",
            "ncbi_assembly_accession": "GCF_000146045.2",
            "ncbi_biosample_accession": "",
            "ncbi_wgs_accession": "",
            "sgd_accession": "SGD R64-5-1 annotation",
            "proxy_reference": "S288C R64 reference genome (GCF_000146045.2; taxon 559292); proxy for a stated derivative, not DHY210 genotype",
            "model_policy": "official tutorial permits S288C proxy; keep proxy flag and never claim genotype identity",
        }
    )
    contract = pd.DataFrame(contract_rows)

    snp = pd.read_csv(snp_path, sep="\t", index_col=0)
    orf = pd.read_csv(orf_path, sep="\t", index_col=0)
    if snp.index.tolist() != snp.columns.tolist() or orf.index.tolist() != orf.columns.tolist():
        raise ValueError("1011 distance matrix axes are inconsistent")
    if snp.index.tolist() != orf.index.tolist():
        raise ValueError("SNP and ORF distance matrix axes differ")
    population_axis = snp.index.astype(str).tolist()
    for strain in PROJECT_STRAINS:
        if strain not in snp.index:
            raise ValueError(f"Missing strain {strain} from 1011 distance matrices")

    snp_values = snp.to_numpy(dtype=np.float64)
    orf_values = orf.to_numpy(dtype=np.float64)
    combined = np.hstack(
        [
            (snp_values - snp_values.mean(axis=0)) / np.where(snp_values.std(axis=0) == 0, 1, snp_values.std(axis=0)),
            (orf_values - orf_values.mean(axis=0)) / np.where(orf_values.std(axis=0) == 0, 1, orf_values.std(axis=0)),
        ]
    )
    pca = PCA(n_components=32, svd_solver="randomized", random_state=20260816)
    population_embedding = pca.fit_transform(combined).astype(np.float32)
    target_rows = [population_axis.index(strain) for strain in PROJECT_STRAINS]
    embedding = np.full((6, 32), np.nan, dtype=np.float32)
    embedding[:5] = population_embedding[target_rows]
    embedding_available = np.asarray([True] * 5 + [False], dtype=bool)

    presence = pd.read_csv(presence_path, sep="\t", index_col=0).loc[PROJECT_STRAINS]
    copy_number = pd.read_csv(copy_path, sep="\t", index_col=0).loc[PROJECT_STRAINS]
    if presence.columns.tolist() != copy_number.columns.tolist():
        raise ValueError("Presence/absence and copy-number gene axes differ")
    frameshift = pd.read_csv(frameshift_path, sep="\t")
    frame_matrix = frameshift[PROJECT_STRAINS].T.to_numpy(dtype=np.float32)
    frame_gene_names = frameshift["Gene"].astype(str).to_numpy()
    phenotype = pd.read_csv(phenotype_path, sep="\t", index_col=0).loc[PROJECT_STRAINS]

    protein_contract = pd.read_csv(protein_contract_path, dtype=str).fillna("")
    if len(protein_contract) != 5243 or not protein_contract["competition_label"].is_unique:
        raise ValueError("Protein identity contract must contain the ordered 5,243-protein axis")
    strain_proteomes = load_1011_proteomes(proteome_archive_path)
    protein_axis = protein_contract["competition_label"].tolist()
    systematic_axis = protein_contract["systematic_name"].tolist()
    s288c_sequences: dict[str, str] = {}
    reference_fasta = protein_contract_path.parent / "sequences.fasta.gz"
    if not reference_fasta.exists():
        raise FileNotFoundError(reference_fasta)
    current_label = ""
    chunks: list[str] = []
    with gzip.open(reference_fasta, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if current_label:
                    s288c_sequences[current_label] = "".join(chunks)
                current_label = line[1:].split()[0]
                chunks = []
            elif line:
                chunks.append(line)
        if current_label:
            s288c_sequences[current_label] = "".join(chunks)
    if set(s288c_sequences) != set(protein_axis):
        raise ValueError("S288C reference FASTA axis differs from protein identity contract")

    proteome_present = np.zeros((6, len(protein_axis)), dtype=bool)
    comparable_length = np.zeros((6, len(protein_axis)), dtype=bool)
    identity_to_s288c = np.full((6, len(protein_axis)), np.nan, dtype=np.float32)
    length_ratio_to_s288c = np.full((6, len(protein_axis)), np.nan, dtype=np.float32)
    selected_records: list[tuple[str, str, str, str]] = []
    for strain_index, strain in enumerate(PROJECT_STRAINS):
        for protein_index, (label, systematic_name) in enumerate(zip(protein_axis, systematic_axis, strict=True)):
            if not systematic_name or systematic_name not in strain_proteomes[strain]:
                continue
            sequence = strain_proteomes[strain][systematic_name]
            reference = s288c_sequences[label]
            proteome_present[strain_index, protein_index] = True
            length_ratio_to_s288c[strain_index, protein_index] = len(sequence) / len(reference)
            if len(sequence) == len(reference):
                comparable_length[strain_index, protein_index] = True
                identity_to_s288c[strain_index, protein_index] = sum(
                    amino_acid == reference[position] for position, amino_acid in enumerate(sequence)
                ) / len(reference)
            selected_records.append((strain, label, systematic_name, sequence))

    pairwise_rows: list[dict] = []
    for first in PROJECT_STRAINS:
        for second in PROJECT_STRAINS:
            pairwise_rows.append(
                {
                    "strain_a": first,
                    "strain_b": second,
                    "snp_nonidentical_percent": float(snp.loc[first, second]),
                    "orf_symmetric_difference_count": int(orf.loc[first, second]),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.csv"
    contract.to_csv(contract_path, index=False, lineterminator="\n")
    pairwise_path = output_dir / "pairwise-distances.csv"
    pd.DataFrame(pairwise_rows).to_csv(pairwise_path, index=False, lineterminator="\n")
    embedding_path = output_dir / "population-distance-pca.npz"
    np.savez_compressed(
        embedding_path,
        competition_strains=np.asarray(ALL_COMPETITION_STRAINS, dtype=str),
        embedding=embedding,
        embedding_available=embedding_available,
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        population_axis=np.asarray(population_axis, dtype=str),
    )
    genomic_path = output_dir / "genomic-features.npz"
    np.savez_compressed(
        genomic_path,
        competition_strains=np.asarray(PROJECT_STRAINS, dtype=str),
        pangenome_gene_names=np.asarray(presence.columns, dtype=str),
        presence=np.asarray(presence, dtype=np.float32),
        copy_number=np.asarray(copy_number, dtype=np.float32),
        frameshift_gene_names=frame_gene_names.astype(str),
        frameshift=frame_matrix,
    )
    diagnostic_path = output_dir / "phenotype-features-diagnostic.npz"
    np.savez_compressed(
        diagnostic_path,
        competition_strains=np.asarray(PROJECT_STRAINS, dtype=str),
        condition_names=np.asarray(phenotype.columns, dtype=str),
        growth_ratios=np.asarray(phenotype, dtype=np.float32),
    )
    strain_proteome_path = output_dir / "strain-proteomes.fasta.gz"
    with (
        strain_proteome_path.open("wb") as raw_handle,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gzip_handle,
        io.TextIOWrapper(gzip_handle, encoding="utf-8", newline="\n") as handle,
    ):
        for strain, label, systematic_name, sequence in selected_records:
            handle.write(f">{strain}|{label}|{systematic_name}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")
    strain_proteome_feature_path = output_dir / "strain-proteome-features.npz"
    np.savez_compressed(
        strain_proteome_feature_path,
        competition_strains=np.asarray(ALL_COMPETITION_STRAINS, dtype=str),
        competition_labels=np.asarray(protein_axis, dtype=str),
        systematic_names=np.asarray(systematic_axis, dtype=str),
        present=proteome_present,
        comparable_length=comparable_length,
        identity_to_s288c=identity_to_s288c,
        length_ratio_to_s288c=length_ratio_to_s288c,
    )
    strain_proteome_summary_path = output_dir / "strain-proteome-summary.csv"
    pd.DataFrame(
        [
            {
                "competition_strain": strain,
                "source": "Peter et al. 2018 1011proteome.tar.gz",
                "source_status": "exact_1011_standardized_name" if strain in PROJECT_STRAINS else "unavailable_exact_dhy210",
                "archive_sequence_count": len(strain_proteomes.get(strain, {})),
                "competition_axis_sequence_count": int(proteome_present[index].sum()),
                "competition_axis_equal_length_count": int(comparable_length[index].sum()),
                "competition_axis_missing_count": int((~proteome_present[index]).sum()),
                "model_policy": "public strain proteome candidate with explicit missing mask"
                if strain in PROJECT_STRAINS
                else "do not synthesize exact DHY210 proteome; S288C remains a separately flagged proxy",
            }
            for index, strain in enumerate(ALL_COMPETITION_STRAINS)
        ]
    ).to_csv(strain_proteome_summary_path, index=False, lineterminator="\n")
    ena_direct_files = {
        "CEK": ena_audit_dir / "jcm-2985-4b-assembly.json",
        "CGD": ena_audit_dir / "ucd-09-448-assembly.json",
        "CRD": ena_audit_dir / "fima-3-assembly.json",
    }
    identity_search_rows: list[dict] = []
    for strain, path in ena_direct_files.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        identity_search_rows.append(
            {
                "competition_strain": strain,
                "source": "ENA Portal assembly exact strain search",
                "query": collection.loc[strain, "Isolate name"],
                "result_count": len(payload),
                "candidate_accessions": ";".join(sorted({str(item.get("assembly_accession") or "") for item in payload if item.get("assembly_accession")})),
                "adoption_status": "unresolved_no_exact_ena_assembly" if not payload else "manual_review_required",
                "reason": "No exact ENA assembly record returned" if not payload else "Search results require identity-level review",
            }
        )
    scrap = json.loads((ena_audit_dir / "prjeb59869-assemblies.json").read_text(encoding="utf-8"))
    for strain, title in [("BAH", "BAH"), ("BAI", "BAI_1a")]:
        candidates = [
            item
            for item in scrap
            if item.get("sample_title") == title and str(item.get("assembly_name") or "").endswith(".nuclear_genome.ScRAP")
        ]
        reason = "NCBI Peter-2018 isolate assembly already provides exact evidence; the ENA hit is retained only as a secondary candidate"
        if strain == "BAI":
            reason = "NCBI Peter-2018 isolate assembly already provides exact evidence; BAI_1a carries a derivative suffix and is retained only as a secondary candidate"
        identity_search_rows.append(
            {
                "competition_strain": strain,
                "source": "ENA ScRAP assembly panel",
                "query": title,
                "result_count": len(candidates),
                "candidate_accessions": ";".join(sorted({str(item.get("assembly_accession") or "") for item in candidates})),
                "adoption_status": "secondary_candidate_not_needed_for_exact_identity",
                "reason": reason,
            }
        )
    literature = json.loads(dhy210_literature_search.read_text(encoding="utf-8"))
    exact_dhy_hits = [
        item
        for item in literature.get("papers", [])
        if "DHY210" in json.dumps(item, ensure_ascii=False).upper()
    ]
    identity_search_rows.append(
        {
            "competition_strain": "DHY210",
            "source": "Crossref/Semantic Scholar/Europe PMC/PubMed exact literature search",
            "query": '"DHY210" yeast',
            "result_count": len(exact_dhy_hits),
            "candidate_accessions": "",
            "adoption_status": "unresolved_no_exact_primary_source",
            "reason": "Search services returned no record containing the exact DHY210 token; S288C remains a proxy, not an identity claim",
        }
    )
    identity_search_path = output_dir / "identity-search-audit.csv"
    pd.DataFrame(identity_search_rows).to_csv(identity_search_path, index=False, lineterminator="\n")
    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        "species": "Saccharomyces cerevisiae",
        "ncbi_species_taxon_id": 4932,
        "competition_strains": ALL_COMPETITION_STRAINS,
        "exact_1011_project_mappings": PROJECT_STRAINS,
        "unresolved_exact_genotype_and_accession": ["DHY210"],
        "dh_y210_proxy_policy": "Official tutorial describes DHY210 as S288C-derived and permits S288C proxy; exact DHY210 accession and variants remain unresolved; no 1011-population embedding is imputed",
        "ncbi_assembly_audit": {
            "BAH": "GCA_003277085.1",
            "BAI": "GCA_003276965.1",
            "CEK": "not found by exact isolate name in 1,755 S. cerevisiae assembly records",
            "CGD": "not found by exact isolate name in 1,755 S. cerevisiae assembly records",
            "CRD": "not found by exact isolate name in 1,755 S. cerevisiae assembly records",
            "DHY210_proxy": "GCF_000146045.2",
        },
        "population_embedding": {
            "method": "PCA over concatenated column-standardized 1011 SNP-distance and ORF-distance matrices",
            "fit_population_count": len(population_axis),
            "dimension": 32,
            "competition_coverage": 5,
            "learned_from_competition_labels_or_proteomes": False,
        },
        "genomic_feature_shapes": {
            "presence": list(presence.shape),
            "copy_number": list(copy_number.shape),
            "frameshift": list(frame_matrix.shape),
        },
        "strain_proteome": {
            "source": "Peter et al. 2018 1011proteome.tar.gz",
            "archive_md5_verified": "1bb03292cc91a87fb724d70a33ea3c67",
            "archive_design": "SNP-inferred proteomes organized by reference gene; gene presence/absence respected; one heterozygous allele randomly selected by source project",
            "competition_strain_coverage": 5,
            "competition_axis_shape": list(proteome_present.shape),
            "per_strain_axis_coverage": {
                strain: int(proteome_present[index].sum()) for index, strain in enumerate(ALL_COMPETITION_STRAINS)
            },
            "identity_definition": "position-wise amino-acid identity only when source and S288C canonical sequences have equal length; unequal lengths remain NaN with a separate length ratio",
            "dhy210_policy": "exact strain proteome unavailable; row remains missing and is never fabricated from S288C",
        },
        "identity_search_audit": {
            "ena_exact_unresolved": ["CEK", "CGD", "CRD"],
            "ena_scrap_secondary_candidates": {"BAH": "GCA_949124635", "BAI": "GCA_949124515"},
            "dhy210_exact_literature_hits": len(exact_dhy_hits),
            "policy": "Candidates never replace exact isolate identity without direct source-level equivalence evidence",
        },
        "phenotype_feature_policy": "diagnostic_only_high_leakage_risk; overlaps perturbation conditions and is not approved as primary model input",
        "hashes": {path.name: sha256_file(path) for path in sources},
    }
    manifest["hashes"]["protein_identity_contract_sha256"] = sha256_file(protein_contract_path)
    for path in [
        contract_path,
        pairwise_path,
        embedding_path,
        genomic_path,
        diagnostic_path,
        strain_proteome_path,
        strain_proteome_feature_path,
        strain_proteome_summary_path,
        identity_search_path,
    ]:
        manifest["hashes"][path.name] = sha256_file(path)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--protein-contract", type=Path, required=True)
    parser.add_argument("--ena-audit-dir", type=Path, required=True)
    parser.add_argument("--dhy210-literature-search", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.raw_dir,
                args.protein_contract,
                args.ena_audit_dir,
                args.dhy210_literature_search,
                args.output_dir,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
