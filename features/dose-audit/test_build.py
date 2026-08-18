import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def test_dose_is_explicitly_unavailable_for_every_compound() -> None:
    contract = pd.read_csv(HERE / "contract.csv", dtype=str).fillna("")
    assert len(contract) == 56
    assert contract["competition_name"].is_unique
    assert set(contract["competition_dose_status"]) == {"not_released"}
    assert contract["competition_dose_value"].eq("").all()
    assert contract["competition_dose_unit"].eq("").all()


def test_pert_id_is_not_misused_as_dose() -> None:
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    collisions = pd.read_csv(HERE / "pert-id-collisions.csv", dtype=str)
    assert manifest["explicit_dose_columns"] == []
    assert manifest["pert_id_collision_count"] == 15
    assert len(collisions) == 15
    assert "cannot be reconstructed" in manifest["dose_conclusion"]


def test_external_pride_dose_is_explicitly_rejected() -> None:
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    audit = pd.read_csv(HERE / "source-audit.csv", dtype=str).fillna("")
    pxd = audit.loc[audit["source_id"].eq("PXD023613")].iloc[0]
    assert pxd["dose_evidence"].replace("μ", "µ").replace("uM", "µM") == "10 µM"
    assert pxd["adoption_status"] == "excluded_not_competition_dose"
    assert not manifest["source_provenance_audit"]["pxd023613_adopted_as_competition_dose"]
    assert "BY4741" in manifest["source_provenance_audit"]["conclusion"]
