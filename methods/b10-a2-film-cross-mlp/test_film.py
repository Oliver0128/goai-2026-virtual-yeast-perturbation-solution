from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("b10_a2_core", HERE / "core.py")
assert SPEC is not None and SPEC.loader is not None
CORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)


def make_model(output_dim: int = 7) -> torch.nn.Module:
    return CORE.FilmCrossMLP(
        condition_dim=6,
        drug_dim=10,
        output_dim=output_dim,
        encoder_hidden_dim=8,
        latent_dim=4,
        fusion_dim=9,
        dropout=0.0,
    )


def test_film_zero_initialization_degenerates_to_condition_latent() -> None:
    model = make_model()
    model.eval()
    _, audit = model(torch.randn(5, 6), torch.randn(5, 10), return_audit=True)
    torch.testing.assert_close(audit["gamma"], torch.zeros_like(audit["gamma"]))
    torch.testing.assert_close(audit["beta"], torch.zeros_like(audit["beta"]))
    torch.testing.assert_close(audit["condition_film"], audit["condition_latent"])
    assert torch.count_nonzero(model.film_gamma.weight) == 0
    assert torch.count_nonzero(model.film_beta.weight) == 0


def test_film_and_fusion_dimensions() -> None:
    model = make_model(output_dim=11)
    output, audit = model(torch.randn(3, 6), torch.randn(3, 10), return_audit=True)
    assert output.shape == (3, 11)
    assert audit["condition_latent"].shape == (3, 4)
    assert audit["drug_latent"].shape == (3, 4)
    assert audit["gamma"].shape == (3, 4)
    assert audit["beta"].shape == (3, 4)
    assert audit["condition_film"].shape == (3, 4)
    assert audit["fusion"].shape == (3, 9)
    assert model.fusion_stem[0].in_features == 12
    assert model.output_head.in_features == 9
    assert model.output_head.out_features == 11


def test_identity_dropout_zeros_only_identity_and_seen_mask() -> None:
    drug = torch.arange(60, dtype=torch.float32).reshape(5, 12) + 1.0
    dropped, rows = CORE.apply_compound_identity_dropout(
        drug,
        compound_identity_dim=3,
        seen_mask_index=11,
        probability=1.0,
        generator=torch.Generator().manual_seed(42),
    )
    assert rows.all()
    assert torch.all(dropped[:, :3] == 0)
    torch.testing.assert_close(dropped[:, 3:11], drug[:, 3:11])
    assert torch.all(dropped[:, 11] == 0)
    torch.testing.assert_close(
        drug, torch.arange(60, dtype=torch.float32).reshape(5, 12) + 1.0
    )


def test_train_only_metadata_and_structure_encoding_and_dual_split() -> None:
    fit = pd.DataFrame(
        {
            "Strains": ["S1", "S2", "S1"],
            "perturbation_no_concentration": ["A", "B", "A"],
            "Medium": ["M", "M", "M"],
            "Temperature": [30, 30, 30],
            "pert_time_unit": ["min", "min", "min"],
            "data_source": ["D", "D", "D"],
            "instrument": ["I", "I", "I"],
            "pert_time": [2.0, 4.0, 8.0],
        }
    )
    target = pd.DataFrame(
        {
            "Strains": ["S3"],
            "perturbation_no_concentration": ["C"],
            "Medium": ["M"],
            "Temperature": [30],
            "pert_time_unit": ["min"],
            "data_source": ["D"],
            "instrument": ["I"],
            "pert_time": [1024.0],
        }
    )
    categorical = [
        "Strains",
        "perturbation_no_concentration",
        "Medium",
        "Temperature",
        "pert_time_unit",
        "data_source",
        "instrument",
    ]
    metadata = CORE.encode_metadata(fit, target, categorical, ["pert_time"])
    assert metadata.manifest["categorical"]["Strains"]["categories"] == ["S1", "S2"]
    assert metadata.manifest["categorical"]["Strains"]["target_unseen_categories"] == [
        "S3"
    ]
    assert metadata.manifest["categorical"]["perturbation_no_concentration"][
        "target_unseen_categories"
    ] == ["C"]
    assert metadata.manifest["numeric"]["pert_time"]["mean"] == 2.0

    table = CORE.StructureTable(
        competition_names=np.asarray(["A", "B", "C"]),
        pubchem_cids=np.asarray([1, 2, 3]),
        morgan_bits=np.asarray(
            [[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.uint8
        ),
        descriptor_names=np.asarray(["d1", "d2"]),
        descriptors_raw=np.asarray(
            [[1.0, 10.0], [3.0, 14.0], [99.0, 999.0]], dtype=np.float32
        ),
    )
    structure = CORE.encode_structure(
        fit, target, table, "perturbation_no_concentration"
    )
    assert structure.manifest["descriptor_mean"] == [5.0 / 3.0, 34.0 / 3.0]
    assert structure.manifest["descriptor_scaling_fit_scope"] == "fit rows only"
    assert structure.manifest["target_unseen_identity_rows"] == 1

    dual = CORE.split_dual_inputs(metadata, structure, "perturbation_no_concentration")
    compound_slice = metadata.manifest["feature_slices"][
        "perturbation_no_concentration"
    ]
    identity_dim = compound_slice[1] - compound_slice[0]
    assert dual.manifest["condition_excludes_compound_identity"] is True
    assert not any(
        name.startswith("perturbation_no_concentration=")
        for name in dual.manifest["condition_feature_names"]
    )
    assert dual.fit_condition.shape[1] == metadata.fit.shape[1] - identity_dim
    assert dual.fit_drug.shape[1] == identity_dim + structure.fit.shape[1]
    np.testing.assert_array_equal(
        dual.target_drug[0, :identity_dim], np.zeros(identity_dim)
    )
    assert dual.target_drug[0, -2] == 1.0
    assert dual.target_drug[0, -1] == 0.0


def test_masked_mse_backward_through_film_model() -> None:
    CORE.set_reproducible_seed(42)
    model = make_model(output_dim=5)
    prediction = model(torch.randn(4, 6), torch.randn(4, 10))
    truth = torch.randn(4, 5)
    mask = torch.ones(4, 5)
    mask[0, 1] = 0.0
    loss = CORE.masked_mse(prediction, truth, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.output_head.weight.grad is not None
    assert model.condition_encoder[0].weight.grad is not None
    assert model.drug_encoder[0].weight.grad is not None
    assert model.film_gamma.weight.grad is not None
    assert model.film_beta.weight.grad is not None
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_committed_structure_artifact_matches_contract_axis() -> None:
    project = HERE.parents[1]
    if not (project / "features/compound-structure/fingerprints.npz").is_file():
        pytest.skip("Frozen public feature artifacts are intentionally not distributed")
    table = CORE.load_structure_table(
        project / "features/compound-structure/fingerprints.npz",
        project / "features/compound-structure/contract.csv",
    )
    assert table.competition_names.shape == (56,)
    assert table.morgan_bits.shape == (56, 2048)
    assert table.descriptors_raw.shape == (56, 5)
    assert table.descriptor_names.tolist() == [
        "mol_wt",
        "mol_logp",
        "tpsa",
        "hbd",
        "hba",
    ]
