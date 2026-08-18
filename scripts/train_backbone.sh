#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
data_dir=${DATA_DIR:-"$repo/data/official"}
feature_dir=${FEATURE_DIR:-"$repo/data/compound-structure"}
run_dir=${RUN_DIR:-"$repo/outputs/yeafilm-seed42"}
wandb_mode=${WANDB_MODE:-offline}
mkdir -p "$run_dir/metrics"

python "$repo/methods/b10-a2-film-cross-mlp/run.py" \
  --metadata "$data_dir/WAYB_WAYC_metadata_train_val(1).csv" \
  --proteome "$data_dir/WAYB_WAYC_proteome_raw_train_val.csv" \
  --structure-npz "$feature_dir/fingerprints.npz" \
  --structure-contract "$feature_dir/contract.csv" \
  --structure-manifest "$feature_dir/manifest.json" \
  --config "$repo/methods/b10-a2-film-cross-mlp/config.json" \
  --prediction "$run_dir/prediction.csv" \
  --manifest "$run_dir/manifest.json" \
  --history "$run_dir/training-history.json" \
  --model "$run_dir/model.pt" \
  --result "$run_dir/result.json" \
  --device auto \
  --wandb-mode "$wandb_mode" \
  --scorer-script "$repo/evaluation/official-six-module-scorer/scripts/score_validation.sh" \
  --score-config "$repo/evaluation/official-six-module-scorer/configs/current-handbook-sample-mean-v1.json" \
  --score-output "$run_dir/metrics/six-module.json"
