#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 OUTPUT_ROOT [DEVICE]" >&2
  exit 2
fi

bundle=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_root=$1
device=${2:-auto}

if [[ -e "$output_root" ]]; then
  echo "Refusing to overwrite existing output: $output_root" >&2
  exit 2
fi

bash "$bundle/scripts/train_e7.sh" "$output_root/e7" "$device"
bash "$bundle/scripts/train_crvae.sh" "$output_root/crvae" "$output_root/e7" "$device"

echo "Full two-stage retraining completed: $output_root"
