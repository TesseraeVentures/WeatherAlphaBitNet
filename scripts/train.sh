#!/bin/bash
# WeatherAlphaBitNet — one-command training
# Usage: bash scripts/train.sh [--data-path /path/to/era5.zarr] [--epochs 100]
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_PATH="${DATA_PATH:-}"
EPOCHS="${EPOCHS:-100}"

echo "=== WeatherAlphaBitNet Training ==="
echo "Arch config:  configs/arch_config.json"
echo "Train config: configs/train_config.json"
echo "Stations:     configs/stations.json"
echo "Data path:    ${DATA_PATH:-'(synthetic fallback)'}"
echo ""

python3 -m weatheralpha.train \
  --arch-config  configs/arch_config.json \
  --train-config configs/train_config.json \
  --stations     configs/stations.json \
  ${DATA_PATH:+--data-path "$DATA_PATH"} \
  --checkpoint-dir checkpoints/ \
  "$@"

echo ""
echo "Done. Best checkpoint at: checkpoints/best.pt"
