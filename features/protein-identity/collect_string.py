from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def collect(
    contract_path: Path,
    wrapper: Path,
    raw_dir: Path,
    output_dir: Path,
    bulk_physical_links: Path,
    bulk_protein_info: Path,
) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = pd.read_csv(contract_path, dtype=str).fillna("")
    accessions = contract.loc[contract["uniprot_accession"].ne(""), "uniprot_accession"].tolist()

    version_path = raw_dir / "version.tsv"
    mapping_path = raw_dir / "mapping.tsv"
    base = ["uv", "run", str(wrapper)]
    run([*base, "version", "--output", str(version_path)])
    run(
        [
            *base,
            "map",
            "--output",
            str(mapping_path),
            "--identifiers",
            *accessions,
            "--species",
            "4932",
        ]
    )

    mapping = pd.read_csv(mapping_path, sep="\t", dtype=str).fillna("")
    if not mapping.empty and set(mapping["ncbiTaxonId"]) != {"4932"}:
        raise ValueError("STRING mapping returned a non-Saccharomyces cerevisiae taxon")
    mapping = mapping.sort_values(["queryIndex", "stringId"]).drop_duplicates("queryItem", keep="first")
    string_ids = set(mapping["stringId"].drop_duplicates())
    network = pd.read_csv(bulk_physical_links, sep=r"\s+", compression="gzip", dtype=str).fillna("")
    network["combined_score"] = network["combined_score"].astype(int)
    network = network.loc[
        network["protein1"].isin(string_ids)
        & network["protein2"].isin(string_ids)
        & network["combined_score"].ge(400)
    ].copy()
    info = pd.read_csv(bulk_protein_info, sep="\t", compression="gzip", dtype=str).fillna("")
    info = info.rename(columns={"#string_protein_id": "string_id"})
    preferred_by_string = info.set_index("string_id")["preferred_name"].to_dict()
    label_by_string = contract.set_index("string_id")["competition_label"].to_dict()
    if network.empty:
        edges = pd.DataFrame(
            columns=[
                "protein_a",
                "protein_b",
                "string_id_a",
                "string_id_b",
                "preferred_name_a",
                "preferred_name_b",
                "combined_score",
            ]
        )
    else:
        edges = network.rename(
            columns={
                "protein1": "string_id_a",
                "protein2": "string_id_b",
            }
        )
        edges["preferred_name_a"] = edges["string_id_a"].map(preferred_by_string).fillna("")
        edges["preferred_name_b"] = edges["string_id_b"].map(preferred_by_string).fillna("")
        edges.insert(0, "protein_a", edges["string_id_a"].map(label_by_string).fillna(""))
        edges.insert(1, "protein_b", edges["string_id_b"].map(label_by_string).fillna(""))
        edges = edges.loc[edges["protein_a"].ne("") & edges["protein_b"].ne("")].copy()
        unordered = edges.apply(lambda row: "\t".join(sorted([row["string_id_a"], row["string_id_b"]])), axis=1)
        edges = edges.loc[~unordered.duplicated()].copy()
        edges = edges[
            [
                "protein_a",
                "protein_b",
                "string_id_a",
                "string_id_b",
                "preferred_name_a",
                "preferred_name_b",
                "combined_score",
            ]
        ].sort_values(["protein_a", "protein_b"])

    edge_path = output_dir / "ppi-physical-edges.tsv.gz"
    edges.to_csv(
        edge_path,
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
        lineterminator="\n",
    )
    expected = contract.loc[contract["string_id"].ne(""), ["uniprot_accession", "string_id"]]
    observed = mapping.set_index("queryItem")["stringId"].to_dict()
    concordant = sum(observed.get(row.uniprot_accession) == row.string_id for row in expected.itertuples())
    version = pd.read_csv(version_path, sep="\t", dtype=str).iloc[0].to_dict()
    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now().astimezone().isoformat(),
        "source": "STRING API",
        "string_version": version.get("string_version", ""),
        "species": "Saccharomyces cerevisiae",
        "ncbi_taxon_id": 4932,
        "network_type": "physical",
        "required_score": 400,
        "competition_protein_count": len(contract),
        "mapped_accession_count": len(mapping),
        "preexisting_string_id_count": len(expected),
        "preexisting_mapping_concordant_count": concordant,
        "edge_count": len(edges),
        "covered_protein_count": len(set(edges["protein_a"]) | set(edges["protein_b"])),
        "policy": "External public entity feature only; no competition proteome values or labels were submitted to STRING",
        "hashes": {
            "contract_sha256": sha256_file(contract_path),
            "mapping_sha256": sha256_file(mapping_path),
            "bulk_physical_links_sha256": sha256_file(bulk_physical_links),
            "bulk_protein_info_sha256": sha256_file(bulk_protein_info),
            "edge_sha256": sha256_file(edge_path),
        },
    }
    (output_dir / "string-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bulk-physical-links", type=Path, required=True)
    parser.add_argument("--bulk-protein-info", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            collect(
                args.contract,
                args.wrapper,
                args.raw_dir,
                args.output_dir,
                args.bulk_physical_links,
                args.bulk_protein_info,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
