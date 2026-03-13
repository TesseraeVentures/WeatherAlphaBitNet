#!/bin/bash
# WeatherAlphaBitNet — evaluate against StationBench
# Usage: bash scripts/eval.sh [--checkpoint checkpoints/best.pt]
set -euo pipefail
cd "$(dirname "$0")/.."

CHECKPOINT="${CHECKPOINT:-checkpoints/best.pt}"
OUTPUT="${OUTPUT:-eval_results.json}"

echo "=== WeatherAlphaBitNet StationBench Evaluation ==="
echo "Checkpoint: $CHECKPOINT"
echo "Stations:   configs/stations.json"
echo ""

python3 -m weatheralpha.eval \
  --checkpoint "$CHECKPOINT" \
  --stations   configs/stations.json \
  --output     "$OUTPUT" \
  "$@"

echo ""
echo "Results written to: $OUTPUT"
