#!/usr/bin/env bash
set -euo pipefail

bundle=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mode=${1:-quick}
python_bin=${PYTHON_BIN:-python}

if [[ "$mode" != "quick" && "$mode" != "inference" ]]; then
  echo "Usage: $0 [quick|inference]" >&2
  exit 2
fi

(cd "$bundle" && sha256sum -c audit/SHA256SUMS)
"$python_bin" -m pytest -q "$bundle/code/submission/test_predict.py"
"$python_bin" -m pytest -q "$bundle/code/methods/b10-a2-film-cross-mlp"
"$python_bin" -m pytest -q "$bundle/code/methods/b16-a2-conditional-residual-vae"
PYTHONPATH="$bundle/code/evaluation/official-six-module-scorer/src" \
  "$python_bin" -m pytest -q "$bundle/code/evaluation/official-six-module-scorer/tests"

if [[ "$mode" == "inference" ]]; then
  bash "$bundle/scripts/predict_test.sh"
fi

echo "Bundle verification completed in mode: $mode"
