from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def test_identity_contract_is_explicit() -> None:
    contract = pd.read_csv(HERE / "contract.csv", dtype=str).fillna("").set_index("competition_strain")
    assert contract.index.tolist() == ["BAH", "BAI", "CEK", "CGD", "CRD", "DHY210"]
    assert contract.loc["BAH", "isolate_name"] == "SX3"
    assert contract.loc["BAI", "isolate_name"] == "BJ6"
    assert contract.loc["CEK", "isolate_name"] == "JCM_2985-4B"
    assert contract.loc["CGD", "isolate_name"] == "UCD_09-448"
    assert contract.loc["CRD", "isolate_name"] == "FIMA_3"
    assert contract.loc["BAH", "ncbi_assembly_accession"] == "GCA_003277085.1"
    assert contract.loc["BAI", "ncbi_assembly_accession"] == "GCA_003276965.1"
    assert contract.loc["CEK", "ncbi_assembly_accession"] == ""
    assert contract.loc["CGD", "ncbi_assembly_accession"] == ""
    assert contract.loc["CRD", "ncbi_assembly_accession"] == ""
    assert contract.loc["DHY210", "identity_status"].startswith("official_tutorial_s288c_derived")
    assert contract.loc["DHY210", "ncbi_assembly_accession"] == "GCF_000146045.2"
    assert "not DHY210 genotype" in contract.loc["DHY210", "proxy_reference"]


def test_population_embedding_never_imputes_dhy210() -> None:
    with np.load(HERE / "population-distance-pca.npz", allow_pickle=False) as artifact:
        assert artifact["embedding"].shape == (6, 32)
        assert artifact["embedding_available"].tolist() == [True, True, True, True, True, False]
        assert np.isfinite(artifact["embedding"][:5]).all()
        assert np.isnan(artifact["embedding"][5]).all()


def test_genomic_and_distance_axes() -> None:
    pairwise = pd.read_csv(HERE / "pairwise-distances.csv")
    assert len(pairwise) == 25
    with np.load(HERE / "genomic-features.npz", allow_pickle=False) as artifact:
        assert artifact["presence"].shape == (5, 7796)
        assert artifact["copy_number"].shape == (5, 7796)
        assert artifact["frameshift"].shape[0] == 5


def test_strain_proteomes_are_public_exact_records_with_dhy210_missing() -> None:
    summary = pd.read_csv(HERE / "strain-proteome-summary.csv", dtype=str).fillna("").set_index("competition_strain")
    assert summary.index.tolist() == ["BAH", "BAI", "CEK", "CGD", "CRD", "DHY210"]
    assert (summary.loc[["BAH", "BAI", "CEK", "CGD", "CRD"], "competition_axis_sequence_count"].astype(int) > 4800).all()
    assert int(summary.loc["DHY210", "competition_axis_sequence_count"]) == 0
    with np.load(HERE / "strain-proteome-features.npz", allow_pickle=False) as artifact:
        assert artifact["present"].shape == (6, 5243)
        assert not artifact["present"][5].any()
        assert np.isnan(artifact["identity_to_s288c"][5]).all()
        comparable = artifact["comparable_length"]
        identity = artifact["identity_to_s288c"]
        assert np.isfinite(identity[comparable]).all()
        assert ((identity[comparable] >= 0) & (identity[comparable] <= 1)).all()


def test_identity_search_keeps_unresolved_candidates_unadopted() -> None:
    audit = pd.read_csv(HERE / "identity-search-audit.csv", dtype=str).fillna("")
    unresolved = audit.loc[audit["competition_strain"].isin(["CEK", "CGD", "CRD"])]
    assert unresolved["result_count"].astype(int).eq(0).all()
    assert unresolved["adoption_status"].eq("unresolved_no_exact_ena_assembly").all()
    dhy210 = audit.loc[audit["competition_strain"].eq("DHY210")].iloc[0]
    assert int(dhy210["result_count"]) == 0
    assert dhy210["adoption_status"] == "unresolved_no_exact_primary_source"
