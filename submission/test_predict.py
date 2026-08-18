from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("solution_predict_test", HERE / "predict.py")
assert SPEC is not None and SPEC.loader is not None
PREDICT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREDICT
SPEC.loader.exec_module(PREDICT)


def test_metadata_rebuild_uses_frozen_categories_and_statistics() -> None:
    metadata = pd.DataFrame({"strain": ["A", "unseen"], "time": [2.0, 8.0]})
    manifest = {
        "categorical": {"strain": {"categories": ["A", "B"]}},
        "numeric": {"time": {"mean": 2.0, "scale": 1.0}},
        "feature_names": ["strain=A", "strain=B", "log2(time)"],
    }
    encoded = PREDICT.encode_metadata_from_manifest(metadata, manifest)
    np.testing.assert_allclose(encoded[0], [1.0, 0.0, -1.0])
    np.testing.assert_allclose(encoded[1], [0.0, 0.0, 1.0])


def test_structure_rebuild_preserves_unseen_identity_structure() -> None:
    table = PREDICT.E7.StructureTable(
        competition_names=np.asarray(["A", "C"]),
        pubchem_cids=np.asarray([1, 3]),
        morgan_bits=np.asarray([[1, 0], [0, 1]], dtype=np.uint8),
        descriptor_names=np.asarray(["d1"]),
        descriptors_raw=np.asarray([[2.0], [6.0]], dtype=np.float32),
    )
    metadata = pd.DataFrame({"compound": ["A", "C", "missing"]})
    manifest = {"descriptor_mean": [2.0], "descriptor_scale": [2.0]}
    encoded = PREDICT.encode_structure_from_manifest(
        metadata, table, manifest, "compound", {"A"}
    )
    np.testing.assert_allclose(encoded[0], [1, 0, 0, 1, 1])
    np.testing.assert_allclose(encoded[1], [0, 1, 2, 1, 0])
    np.testing.assert_allclose(encoded[2], [0, 0, 0, 0, 0])
