from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("goai_compound_contract", HERE / "build.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_committed_contract_has_complete_unique_coverage() -> None:
    contract = pd.read_csv(HERE / "contract.csv")
    assert len(contract) == 56
    assert contract["competition_name"].is_unique
    assert contract["pubchem_cid"].is_unique
    assert "Quality Control" not in set(contract["competition_name"])
    assert contract["pubchem_inchikey"].notna().all()
    assert contract["seen_in_test_metadata"].all()
    assert int(contract["seen_in_train_metadata"].sum()) == 39
    assert int(contract["seen_in_validation_metadata"].sum()) == 45


def test_all_committed_feature_smiles_roundtrip() -> None:
    contract = pd.read_csv(HERE / "contract.csv")
    for smiles in contract["rdkit_feature_canonical_isomeric_smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        assert MODULE.canonical_isomeric_smiles(mol) == smiles


def test_committed_fingerprint_artifact_matches_contract() -> None:
    contract = pd.read_csv(HERE / "contract.csv")
    with np.load(HERE / "fingerprints.npz", allow_pickle=False) as artifact:
        assert artifact["morgan_bits"].shape == (56, 2048)
        assert artifact["morgan_bits"].dtype == np.uint8
        assert set(np.unique(artifact["morgan_bits"])) <= {0, 1}
        assert artifact["descriptors_raw"].shape == (56, 5)
        assert np.isfinite(artifact["descriptors_raw"]).all()
        assert artifact["competition_names"].tolist() == contract["competition_name"].tolist()
        assert artifact["pubchem_cids"].tolist() == contract["pubchem_cid"].tolist()


def test_identity_caveats_and_full_record_exceptions_are_explicit() -> None:
    contract = pd.read_csv(HERE / "contract.csv").set_index("competition_name")
    assert contract.loc["CHX", "pubchem_query"] == "Cycloheximide"
    assert contract.loc["Oligomycin", "pubchem_query"] == "Oligomycin A"
    assert contract.loc["Tunicamycin", "pubchem_query"] == "Tunicamycin A"
    assert contract.loc["Cisplatin", "feature_molecule_policy"] == "full-record"
    assert contract.loc["NaCl", "feature_molecule_policy"] == "full-record"
