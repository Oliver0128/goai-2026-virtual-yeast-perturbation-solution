from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def test_compound_contract_covers_all_non_qc_entities() -> None:
    contract = pd.read_csv(HERE / "contract.csv", dtype=str).fillna("")
    assert len(contract) == 56
    assert contract["competition_name"].is_unique
    assert contract["pubchem_inchikey"].ne("").all()


def test_external_activity_never_claims_competition_dose() -> None:
    manifest = (HERE / "manifest.json").read_text(encoding="utf-8")
    assert "must never be used as the missing competition treatment dose" in manifest


def test_nonexact_chembl_candidates_are_never_silently_adopted() -> None:
    contract = pd.read_csv(HERE / "contract.csv", dtype=str).fillna("").set_index("competition_name")
    candidates = pd.read_csv(HERE / "mapping-candidates.csv", dtype=str).fillna("")
    expected = {"Cisplatin", "Doxycycline hyclate", "Nystatin dihydrate"}
    assert set(contract.index[contract["chembl_id"].eq("")]) == expected
    assert set(candidates["competition_name"]) == expected
    assert set(candidates["adoption_status"]) == {"not_adopted_without_exact_standard_inchikey"}
    assert not candidates["standard_inchikey_match"].astype(str).str.lower().eq("true").any()


def test_compound_protein_edges_are_yeast_audited() -> None:
    activities = pd.read_csv(HERE / "yeast-activities.csv.gz", dtype=str).fillna("")
    if len(activities):
        assert activities["target_organism"].str.contains("Saccharomyces", case=False).any() or activities[
            "assay_organism"
        ].str.contains("Saccharomyces", case=False).any()


def test_compound_pathways_only_use_competition_axis_targets() -> None:
    pathways = pd.read_csv(HERE / "compound-pathways.csv.gz", dtype=str).fillna("")
    axis = set(pd.read_csv(HERE.parent / "protein-identity" / "contract.csv", dtype=str)["competition_label"])
    assert set(pathways["protein_label"]).issubset(axis)
