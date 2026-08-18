#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 OUTPUT_DIR E7_CHECKPOINT_DIR [DEVICE]" >&2
  exit 2
fi

bundle=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_dir=$1
e7_dir=$2
device=${3:-auto}
python_bin=${PYTHON_BIN:-python}

if [[ -e "$output_dir" ]]; then
  echo "Refusing to overwrite existing output: $output_dir" >&2
  exit 2
fi
if [[ ! -f "$e7_dir/model.pt" || ! -f "$e7_dir/manifest.json" ]]; then
  echo "E7_CHECKPOINT_DIR must contain model.pt and manifest.json: $e7_dir" >&2
  exit 2
fi
mkdir -p "$output_dir"

"$python_bin" "$bundle/code/methods/b16-a2-conditional-residual-vae/run.py" \
  --metadata "$bundle/data/official/WAYB_WAYC_metadata_train_val(1).csv" \
  --proteome "$bundle/data/official/WAYB_WAYC_proteome_raw_train_val.csv" \
  --structure-npz "$bundle/data/compound-structure/fingerprints.npz" \
  --structure-contract "$bundle/data/compound-structure/contract.csv" \
  --structure-manifest "$bundle/data/compound-structure/manifest.json" \
  --e7-model "$e7_dir/model.pt" \
  --e7-manifest "$e7_dir/manifest.json" \
  --config "$bundle/code/methods/b16-a2-conditional-residual-vae/config.json" \
  --prediction "$output_dir/validation-prediction.csv" \
  --manifest "$output_dir/manifest.json" \
  --history "$output_dir/training-history.json" \
  --latent-audit "$output_dir/latent-audit.json" \
  --model "$output_dir/model.pt" \
  --result "$output_dir/result.json" \
  --device "$device" \
  --wandb-mode offline \
  --scorer-script "$bundle/code/evaluation/official-six-module-scorer/scripts/score_validation.sh" \
  --score-config "$bundle/code/evaluation/official-six-module-scorer/configs/current-handbook-sample-mean-v1.json" \
  --score-output "$output_dir/six-module.json"

echo "CR-VAE retraining completed: $output_dir"
