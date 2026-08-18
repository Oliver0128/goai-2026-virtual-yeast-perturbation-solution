#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 OUTPUT_DIR [DEVICE]" >&2
  exit 2
fi

bundle=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_dir=$1
device=${2:-auto}
python_bin=${PYTHON_BIN:-python}

if [[ -e "$output_dir" ]]; then
  echo "Refusing to overwrite existing output: $output_dir" >&2
  exit 2
fi
mkdir -p "$output_dir"

"$python_bin" "$bundle/code/methods/b16-a2-conditional-residual-vae/run.py" \
  --metadata "$bundle/data/official/WAYB_WAYC_metadata_train_val(1).csv" \
  --proteome "$bundle/data/official/WAYB_WAYC_proteome_raw_train_val.csv" \
  --structure-npz "$bundle/data/compound-structure/fingerprints.npz" \
  --structure-contract "$bundle/data/compound-structure/contract.csv" \
  --structure-manifest "$bundle/data/compound-structure/manifest.json" \
  --e7-model "$bundle/models/e7/model.pt" \
  --e7-manifest "$bundle/models/e7/manifest.json" \
  --config "$bundle/code/methods/b16-a2-conditional-residual-vae/config.json" \
  --prediction "$output_dir/unused-prediction.csv" \
  --manifest "$output_dir/unused-manifest.json" \
  --history "$output_dir/unused-history.json" \
  --latent-audit "$output_dir/unused-latent-audit.json" \
  --model "$output_dir/unused-model.pt" \
  --result "$output_dir/unused-result.json" \
  --device "$device" \
  --wandb-mode offline \
  --scorer-script "$bundle/code/evaluation/official-six-module-scorer/scripts/score_validation.sh" \
  --score-config "$bundle/code/evaluation/official-six-module-scorer/configs/current-handbook-sample-mean-v1.json" \
  --score-output "$output_dir/unused-six-module.json" \
  --smoke-only \
  --smoke-output "$output_dir/smoke-training.json"

echo "CR-VAE real-data training smoke test completed: $output_dir/smoke-training.json"
