#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
data_dir=${DATA_DIR:-"$repo/data/official"}
feature_dir=${FEATURE_DIR:-"$repo/data/compound-structure"}
e7_dir=${E7_DIR:-"$repo/outputs/yeafilm-seed42"}
run_dir=${RUN_DIR:-"$repo/outputs/conditional-residual-vae-seed42"}
wandb_mode=${WANDB_MODE:-offline}
mkdir -p "$run_dir/metrics"

python "$repo/methods/b16-a2-conditional-residual-vae/run.py" \
  --metadata "$data_dir/WAYB_WAYC_metadata_train_val(1).csv" \
  --proteome "$data_dir/WAYB_WAYC_proteome_raw_train_val.csv" \
  --structure-npz "$feature_dir/fingerprints.npz" \
  --structure-contract "$feature_dir/contract.csv" \
  --structure-manifest "$feature_dir/manifest.json" \
  --e7-model "$e7_dir/model.pt" \
  --e7-manifest "$e7_dir/manifest.json" \
  --config "$repo/methods/b16-a2-conditional-residual-vae/config.json" \
  --prediction "$run_dir/prediction.csv" \
  --manifest "$run_dir/manifest.json" \
  --history "$run_dir/training-history.json" \
  --latent-audit "$run_dir/latent-audit.json" \
  --model "$run_dir/model.pt" \
  --result "$run_dir/result.json" \
  --device auto \
  --wandb-mode "$wandb_mode" \
  --scorer-script "$repo/evaluation/official-six-module-scorer/scripts/score_validation.sh" \
  --score-config "$repo/evaluation/official-six-module-scorer/configs/current-handbook-sample-mean-v1.json" \
  --score-output "$run_dir/metrics/six-module.json"
