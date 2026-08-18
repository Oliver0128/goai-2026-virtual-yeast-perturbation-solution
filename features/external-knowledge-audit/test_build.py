import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def test_audit_passes_without_truth_leakage() -> None:
    summary = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "pass_with_documented_gaps"
    assert not summary["data_boundary"]["validation_or_test_proteome_values_loaded"]
    assert summary["data_boundary"]["protein_axis_access"] == "header_only"
    assert not summary["dose"]["recoverable"]


def test_every_artifact_has_an_explicit_use_tier_and_guard() -> None:
    policy = pd.read_csv(HERE / "use-policy.csv", dtype=str).fillna("")
    assert len(policy) == 13
    assert policy["artifact"].is_unique
    assert policy["recommended_tier"].ne("").all()
    assert policy["required_guard"].ne("").all()
    assert "diagnostic_only" in set(policy["recommended_tier"])
    assert "prohibition_contract" in set(policy["recommended_tier"])
    assert "audit_only" in set(policy["recommended_tier"])


def test_final_audit_reports_new_sequence_and_evidence_coverage() -> None:
    summary = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))
    assert summary["strain"]["strain_proteome_axis_coverage"]["BAH"] > 5000
    assert summary["strain"]["strain_proteome_axis_coverage"]["DHY210"] == 0
    assert summary["protein"]["go_evidence_annotation_rows"] > 100000
    assert not summary["dose"]["pride_provenance_audit"]["pxd023613_adopted_as_competition_dose"]
