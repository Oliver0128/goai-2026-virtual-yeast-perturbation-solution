import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def test_contract_complete_unique_and_ordered() -> None:
    contract = pd.read_csv(HERE / "contract.csv", dtype=str).fillna("")
    assert len(contract) == 5243
    assert contract["protein_position"].astype(int).tolist() == list(range(5243))
    assert contract["competition_label"].is_unique
    assert contract["uniprot_accession"].ne("").all()
    assert contract["canonical_gene_symbol"].ne("").all()
    assert int(contract["sgd_id"].ne("").sum()) == 5240


def test_only_explicit_axis_repair_is_oct1() -> None:
    contract = pd.read_csv(HERE / "contract.csv").set_index("competition_label")
    explicit = contract.loc[contract["resolution"].eq("explicit_alias")]
    assert explicit.index.tolist() == ["1-Oct"]
    assert explicit.loc["1-Oct", "canonical_gene_symbol"] == "OCT1"
    assert explicit.loc["1-Oct", "uniprot_accession"] == "P35999"


def test_sequence_features_are_finite_and_aligned() -> None:
    contract = pd.read_csv(HERE / "contract.csv")
    with np.load(HERE / "sequence-features.npz", allow_pickle=False) as artifact:
        assert artifact["features"].shape == (5243, 21)
        assert np.isfinite(artifact["features"]).all()
        assert artifact["competition_labels"].tolist() == contract["competition_label"].tolist()


def test_string_network_contract_if_collected() -> None:
    manifest_path = HERE / "string-manifest.json"
    edge_path = HERE / "ppi-physical-edges.tsv.gz"
    if not manifest_path.exists() or not edge_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    edges = pd.read_csv(edge_path, sep="\t")
    assert manifest["ncbi_taxon_id"] == 4932
    assert manifest["network_type"] == "physical"
    assert manifest["required_score"] == 400
    assert len(edges) == manifest["edge_count"]
    axis = set(pd.read_csv(HERE / "contract.csv")["competition_label"])
    assert set(edges["protein_a"]).issubset(axis)
    assert set(edges["protein_b"]).issubset(axis)


def test_go_evidence_has_provenance_and_closes_transposon_spotchecks() -> None:
    evidence = pd.read_csv(HERE / "go-annotations-evidence.csv.gz", dtype=str).fillna("")
    assert len(evidence) > 100000
    assert evidence["competition_label"].nunique() > 5100
    assert evidence["evidence_code"].ne("").all()
    assert evidence["reference"].ne("").all()
    for label in ["1-Oct", "TY1B-LR4", "TY2A-GR1", "TY2B-GR1"]:
        assert label in set(evidence["competition_label"])
    transposons = evidence.loc[evidence["competition_label"].isin(["TY1B-LR4", "TY2A-GR1", "TY2B-GR1"])]
    assert set(transposons["source_database"]) == {"QuickGO API spot check"}
