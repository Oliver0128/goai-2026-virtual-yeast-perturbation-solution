from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from goai_scorer.contracts import ScoringData

PROTEINS = pd.Index([f"P{i}" for i in range(1, 7)])


def _row(
    split: str,
    strain: str,
    drug: str,
    plate: str,
    time: int = 15,
    pert_id: str = "#9",
) -> dict[str, object]:
    return {
        "data_source": "WAYB",
        "Strains": strain,
        "Medium": "M",
        "Temperature": 30,
        "pert_time": time,
        "pert_time_unit": "min",
        "pert_id": pert_id,
        "perturbation_no_concentration": drug,
        "instrument": "I",
        "Yeast_cell_plate": plate,
        "split_final": split,
    }


@pytest.fixture
def config() -> dict:
    path = Path(__file__).parents[1] / "configs/current-handbook-sample-mean-v1.json"
    value = copy.deepcopy(json.loads(path.read_text(encoding="utf-8")))
    value["input_contract"] = {
        "expected_metadata_sha256": None,
        "expected_proteome_sha256": None,
        "expected_raw_proteins": 6,
        "expected_split_counts": {
            "train": 4,
            "val_chem_only": 1,
            "val_strain_only": 2,
            "val_both": 1,
            "val_time": 3,
        },
    }
    return value


@pytest.fixture
def synthetic_frames():
    metadata = pd.DataFrame.from_dict(
        {
            "tr_ctrl_water": _row("train", "S1", "Water", "P1", pert_id="#1"),
            "tr_ctrl_dmso": _row("train", "S1", "DMSO", "P1", pert_id="#2"),
            "tr_d1": _row("train", "S1", "D1", "P1", pert_id="#9"),
            "tr_d2": _row("train", "S1", "D2", "P1", pert_id="#10"),
            "vc": _row("val_chem_only", "S1", "C_NEW", "P1", pert_id="#11"),
            "vs_ctrl": _row("val_strain_only", "S2", "Water", "P2", pert_id="#1"),
            "vs": _row("val_strain_only", "S2", "D1", "P2", pert_id="#9"),
            "vb": _row("val_both", "S2", "C_NEW", "P2", pert_id="#11"),
            "vt_ctrl": _row("val_time", "S1", "Water", "P3", time=30, pert_id="#1"),
            "vt": _row("val_time", "S1", "D1", "P3", time=30, pert_id="#9"),
            "qc": _row("val_time", "S1", "Quality Control", "P3", time=30, pert_id="#48"),
        },
        orient="index",
    )
    ctrl1 = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    ctrl2 = np.array([2, 3, 4, 5, 6, 7], dtype=float)
    ctrl_time = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], dtype=float)
    d1 = np.array([2.0, -2.0, 0.5, -0.2, 1.2, -1.5])
    d2 = np.array([-1.5, 2.0, -0.3, 1.5, -2.0, 0.2])
    new = np.array([1.8, -1.2, 0.4, -1.4, 1.6, -0.5])
    d1_s2 = np.array([2.2, -1.7, 0.8, -0.4, 1.5, -1.2])
    new_s2 = np.array([1.6, -1.4, 0.7, -1.6, 1.3, -0.8])
    d1_time = np.array([1.5, -1.8, 0.6, -0.5, 1.1, -1.3])
    truth = pd.DataFrame(
        [
            ctrl1,
            ctrl1 + 0.2,
            ctrl1 + d1,
            ctrl1 + d2,
            ctrl1 + new,
            ctrl2,
            ctrl2 + d1_s2,
            ctrl2 + new_s2,
            ctrl_time,
            ctrl_time + d1_time,
            ctrl_time + 0.1,
        ],
        index=metadata.index,
        columns=PROTEINS,
    )
    target_ids = pd.Index(["vc", "vs", "vb", "vt"])
    prediction = truth.loc[target_ids].copy()
    return metadata, truth, prediction, target_ids


@pytest.fixture
def scoring_data(synthetic_frames, tmp_path):
    metadata, truth, prediction, target_ids = synthetic_frames
    dummy = tmp_path / "dummy.csv"
    dummy.write_text("x\n", encoding="utf-8")
    return ScoringData(
        metadata=metadata,
        truth_log2=truth,
        prediction_log2=prediction,
        train_missing_rate=pd.Series(0.0, index=PROTEINS),
        retained_proteins=PROTEINS,
        target_ids=target_ids,
        metadata_path=dummy,
        proteome_path=dummy,
        prediction_path=dummy,
        hashes={"prediction_sha256": "x", "sample_axis_sha256": "y", "protein_axis_sha256": "z"},
    )


@pytest.fixture
def synthetic_files(synthetic_frames, tmp_path):
    metadata, truth, prediction, _ = synthetic_frames
    metadata_path = tmp_path / "metadata_train_val.csv"
    proteome_path = tmp_path / "proteome_raw_train_val.csv"
    prediction_path = tmp_path / "prediction.csv"
    metadata.rename_axis("sample_ID").reset_index().to_csv(metadata_path, index=False)
    np.exp2(truth).rename_axis("sample_ID").reset_index().to_csv(proteome_path, index=False)
    prediction.rename_axis("sample_ID").reset_index().to_csv(prediction_path, index=False)
    return metadata_path, proteome_path, prediction_path
