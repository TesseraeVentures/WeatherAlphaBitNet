#!/bin/bash
# WeatherAlphaBitNet — export trained model to MLX format (Apple Silicon)
# Usage: bash scripts/export_mlx.sh [--checkpoint checkpoints/best.pt] [--output mlx_model/]
set -euo pipefail
cd "$(dirname "$0")/.."

CHECKPOINT="${CHECKPOINT:-checkpoints/best.pt}"
OUTPUT="${OUTPUT:-mlx_model}"

echo "=== WeatherAlphaBitNet → MLX Export ==="
echo "Source checkpoint: $CHECKPOINT"
echo "Output directory:  $OUTPUT"
echo ""

python3 -m weatheralpha.export_mlx \
  --checkpoint  "$CHECKPOINT" \
  --arch-config configs/arch_config.json \
  --output      "$OUTPUT" \
  "$@"

echo ""
echo "MLX model ready at: $OUTPUT/"
echo "Deploy to Apple M4 Pro: copy $OUTPUT/ and run mlx_infer.py"
