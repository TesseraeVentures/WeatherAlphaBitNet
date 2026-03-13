#!/bin/bash
# WeatherAlphaBitNet autoresearch benchmark
#
# Called by sxt-research autonomous loop.
# Reads configs/arch_config.json → trains for 30 minutes → outputs metric JSON to stdout.
#
# Contract:
#   stdin:  (nothing)
#   stdout: JSON with keys: val_loss, station_mae_24h, station_mae_72h, params_m
#   stderr: training logs (verbose OK)
#   exit 0: success; exit 1: failure (sxt-research will retry or flag)
#
# The agent ONLY modifies configs/arch_config.json between iterations.
# DO NOT modify benchmark.sh, train.py, or model.py — those are fixed.

set -euo pipefail
cd "$(dirname "$0")/.."

ARCH_CONFIG="configs/arch_config.json"
RESULT_FILE="/tmp/weatheralpha_bench_result.json"

# Validate arch config exists
if [[ ! -f "$ARCH_CONFIG" ]]; then
  echo '{"error": "arch_config_not_found"}' >&2
  exit 1
fi

# Validate params constraint before training (fast check)
PARAMS_EST=$(python3 -c "
import json, sys
cfg = json.load(open('$ARCH_CONFIG'))
d = cfg.get('d_model', 256)
L = cfg.get('n_layers', 6)
n_priv = cfg.get('n_private_experts', 8)
n_share = cfg.get('n_shared_experts', 2)
n_exp = n_priv + n_share
# Rough estimate: attention + MoE per layer
attn = 4 * d * d  # Q K V O
moe = n_exp * 4 * d * d * 2  # each expert expand-4x
total = L * (attn + moe) / 1e6
print(f'{total:.1f}')
" 2>/dev/null || echo "999")

if python3 -c "exit(0 if float('$PARAMS_EST') < 100 else 1)" 2>/dev/null; then
  echo "Estimated params: ${PARAMS_EST}M (within 100M budget)" >&2
else
  echo '{"error": "params_budget_exceeded", "params_m": '"$PARAMS_EST"'}' 
  exit 1
fi

echo "Starting 30-minute training run..." >&2
echo "Arch config: $(cat $ARCH_CONFIG)" >&2

timeout 1800 python3 -m weatheralpha.train \
  --arch-config  "$ARCH_CONFIG" \
  --train-config configs/train_config.json \
  --stations     configs/stations.json \
  --budget-minutes 30 \
  --output       "$RESULT_FILE" 2>&1 || {
  echo '{"error": "training_failed", "params_m": '"$PARAMS_EST"'}' 
  exit 1
}

# Validate output exists and has required keys
if [[ ! -f "$RESULT_FILE" ]]; then
  echo '{"error": "no_output_written"}'
  exit 1
fi

REQUIRED_KEYS="val_loss station_mae_24h station_mae_72h params_m"
for key in $REQUIRED_KEYS; do
  if ! python3 -c "import json; d=json.load(open('$RESULT_FILE')); assert '$key' in d" 2>/dev/null; then
    echo "{\"error\": \"missing_key_$key\", \"raw\": $(cat $RESULT_FILE)}"
    exit 1
  fi
done

cat "$RESULT_FILE"
