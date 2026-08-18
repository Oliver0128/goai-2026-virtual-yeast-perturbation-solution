from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("b16_oof_build", HERE / "build.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def synthetic_metadata() -> pd.DataFrame:
    rows = []
    for group in range(20):
        for replicate in range(2):
            rows.append(
                {
                    "Strains": f"s{group % 3}",
                    "perturbation_no_concentration": f"d{group % 5}",
                    "Medium": "m",
                    "Temperature": 30,
                    "pert_time": group + 1,
                    "pert_time_unit": "min",
                    "replicate": replicate,
                }
            )
    return pd.DataFrame(rows)


def test_group_fold_assignment_is_complete_and_nonoverlapping() -> None:
    metadata = synthetic_metadata()
    fold_ids, groups = MODULE.make_fold_ids(metadata, n_splits=5, seed=42)
    assert fold_ids.dtype == np.int16
    assert set(fold_ids.tolist()) == set(range(5))
    assert len(fold_ids) == len(metadata)
    for group in np.unique(groups):
        assert len(np.unique(fold_ids[groups == group])) == 1


def test_atomic_npz_schema(tmp_path: Path) -> None:
    path = tmp_path / "artifact.npz"
    MODULE.atomic_npz(
        path,
        sample_ids=np.array(["a", "b"]),
        protein_names=np.array(["p1", "p2", "p3"]),
        predictions=np.ones((2, 3), dtype=np.float32),
        fold_ids=np.array([0, 1], dtype=np.int16),
        observed_mask=np.ones((2, 3), dtype=bool),
    )
    with np.load(path, allow_pickle=False) as artifact:
        assert set(artifact.files) == {
            "sample_ids",
            "protein_names",
            "predictions",
            "fold_ids",
            "observed_mask",
        }
        assert artifact["predictions"].dtype == np.float32
        assert artifact["fold_ids"].dtype == np.int16
        assert artifact["observed_mask"].dtype == np.bool_
