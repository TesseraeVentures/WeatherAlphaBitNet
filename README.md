# WeatherAlphaBitNet 🌦️

**Station-level weather forecasting with BitNet b1.58 MoE — runs on CPU, targets Apple M4 Pro via MLX.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

---

## What is WeatherAlphaBitNet?

WeatherAlphaBitNet is a weather forecasting model designed for **CPU-native inference** with no GPU dependency. It combines three research advances:

1. **BitNet b1.58** — weights constrained to {-1, 0, +1} during forward pass, enabling 32× memory reduction and CPU-native integer arithmetic (no fp32 GEMM required).
2. **ARROW Shared-Private MoE** — Shared experts capture global atmospheric patterns; private experts specialise per station cluster (coastal, continental, mountain, etc.).
3. **Adaptive rollout scheduler** — RL-based selector chooses 6h / 12h / 24h rollout step per forecast, reducing error accumulation on long leads.

**Target:** sub-1°C station temperature MAE at 24-hour lead time.  
**Deployment:** Apple M4 Pro via MLX (integer arithmetic, no GPU required).

---

## Why BitNet?

| Property | FP32 | BF16 | BitNet b1.58 |
|---|---|---|---|
| Weight bits | 32 | 16 | ~1.58 |
| Memory (256d, 6L) | ~48 MB | ~24 MB | ~1.5 MB |
| CPU arithmetic | FMUL | FMUL | INT8/popcount |
| M4 Pro inference | ✅ slow | ✅ medium | ✅ fast |
| Gradient training | Full precision maintained | — | Full precision maintained |

BitNet keeps full-precision weights for gradient updates but quantises to {-1, 0, +1} during the forward pass via a straight-through estimator. BitNet v2 adds a Hadamard transformation before attention projections to suppress activation outliers that would otherwise degrade 1-bit accuracy.

---

## Architecture

```
Input: (batch, time_steps, features)  [ERA5 variables: T2m, U10, V10, MSLP, ...]
  │
  ├─ Station Embedding  →  geographic (lat/lon) + climate cluster (one-hot)
  │
  ├─ Temporal Patch Encoder  (patch_size_hours = 6 or 12)
  │
  ├─ N × WeatherAlphaBlock
  │     ├─ BitNetAttention (BitLinear Q/K/V/O + Hadamard v2)
  │     └─ SharedPrivateRouter
  │           ├─ Shared Experts × n_shared  (always active, global patterns)
  │           └─ Private Experts × n_private  (top-k routing, station clusters)
  │
  ├─ Adaptive Rollout Scheduler (RL)  →  selects 6h / 12h / 24h step
  │
  └─ Output Head  →  (station, lead_time, temperature_delta)
```

**Key papers:**
- [ARROW (2510.09734)](https://arxiv.org/abs/2510.09734) — Shared-Private MoE + RL adaptive rollout
- [BitNet b1.58 (Ma et al., 2024)](https://arxiv.org/abs/2402.17764) — 1-bit LLM weights
- [BitNet v2 (2504.18415)](https://arxiv.org/abs/2504.18415) — Hadamard activation outlier suppression
- [ChaosNexus (2509.21802)](https://arxiv.org/abs/2509.21802) — ScaleFormer + MoE for chaotic systems
- [FengWu](https://arxiv.org/abs/2304.02948) — ERA5-based medium-range forecasting baseline

---

## Quick Start

### Install

```bash
pip install -e ".[train]"
# For MLX export (Apple Silicon):
pip install -e ".[train,mlx]"
```

### Train

```bash
# Uses configs/arch_config.json + configs/train_config.json
bash scripts/train.sh

# Or directly:
python -m weatheralpha.train \
  --arch-config configs/arch_config.json \
  --train-config configs/train_config.json \
  --stations configs/stations.json
```

### Evaluate (StationBench)

```bash
bash scripts/eval.sh

# Or directly:
python -m weatheralpha.eval \
  --checkpoint checkpoints/best.pt \
  --stations configs/stations.json
```

### Export to MLX (Apple M4 Pro)

```bash
bash scripts/export_mlx.sh

# Or directly:
python -m weatheralpha.export_mlx \
  --checkpoint checkpoints/best.pt \
  --output mlx_model/
```

---

## Station Registry

Key stations in `configs/stations.json` (Polymarket-relevant):

| ICAO | Location | Cluster |
|------|----------|---------|
| KORD | Chicago O'Hare | great_lakes |
| KDFW | Dallas/Fort Worth | southern_plains |
| KMIA | Miami | subtropical |
| KSEA | Seattle-Tacoma | pacific_coast |
| KMDW | Chicago Midway | great_lakes |
| KDAL | Dallas Love Field | southern_plains |

---

## Architecture Hyperparams (`configs/arch_config.json`)

This file is the **autoresearch target** — the sxt-research loop tunes it automatically:

```json
{
  "d_model": 256,
  "n_heads": 8,
  "n_layers": 6,
  "n_shared_experts": 2,
  "n_private_experts": 8,
  "n_active_private": 2,
  "patch_size_hours": 6,
  "rollout_type": "fixed",
  "bitnet_mode": "b1.58",
  "dropout": 0.1,
  "max_lead_time_hours": 120
}
```

---

## Autoresearch Integration

The `autoresearch/` directory integrates with the **sxt-research autonomous loop**:

1. `autoresearch/program.md` — Research spec: what to optimise, search space, hypotheses
2. `autoresearch/benchmark.sh` — Called by sxt-research: reads `arch_config.json` → 30-min train → outputs metric JSON
3. The agent modifies **only** `configs/arch_config.json` between iterations
4. Metric target: minimise `station_mae_24h` with `params_m < 100`

```bash
# Run one benchmark iteration manually:
bash autoresearch/benchmark.sh
# → outputs to /tmp/weatheralpha_bench_result.json
```

---

## Project Structure

```
WeatherAlphaBitNet/
├── weatheralpha/
│   ├── bitnet.py         BitNet b1.58 linear layer (STE quantisation)
│   ├── attention.py      Multi-head attention + Hadamard (BitNet v2)
│   ├── routing.py        Shared-Private MoE router (ARROW)
│   ├── rollout.py        Adaptive rollout scheduler (RL)
│   ├── station_embed.py  Geographic + cluster station embeddings
│   ├── model.py          Full WeatherAlpha transformer
│   ├── era5.py           ERA5 data loading + normalisation
│   ├── train.py          Training loop (BitNet-aware optimizer)
│   ├── eval.py           StationBench evaluation
│   └── export_mlx.py     MLX export for Apple Silicon
├── configs/
│   ├── arch_config.json  ← autoresearch target
│   ├── train_config.json
│   └── stations.json
├── scripts/
│   ├── train.sh
│   ├── eval.sh
│   └── export_mlx.sh
└── autoresearch/
    ├── benchmark.sh
    └── program.md
```

---

## License

MIT — see [LICENSE](LICENSE).
