#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 METADATA_TRAIN_VAL PROTEOME_RAW_TRAIN_VAL PREDICTION_LOG2 CONFIG OUTPUT_JSON" >&2
  exit 2
fi

scorer_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHONPATH="$scorer_dir/src${PYTHONPATH:+:$PYTHONPATH}" \
  python -m goai_scorer.cli \
  --metadata "$1" \
  --proteome "$2" \
  --prediction "$3" \
  --config "$4" \
  --output "$5"
