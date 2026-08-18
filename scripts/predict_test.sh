#!/usr/bin/env bash
set -euo pipefail

bundle=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python}
mkdir -p "$bundle/outputs"

"$python_bin" "$bundle/code/submission/predict.py" \
  --metadata "$bundle/data/inference-metadata/WAYB_WAYC_metadata_test(1).csv" \
  --structure-npz "$bundle/data/compound-structure/fingerprints.npz" \
  --structure-contract "$bundle/data/compound-structure/contract.csv" \
  --e7-model "$bundle/models/e7/model.pt" \
  --cvae-model "$bundle/models/crvae/model.pt" \
  --output "$bundle/outputs/test-prediction.csv" \
  --audit "$bundle/outputs/inference-audit.json" \
  --device auto
