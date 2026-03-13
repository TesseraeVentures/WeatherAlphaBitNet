# WeatherAlpha-MoE — Autoresearch Spec
## What we're optimising
Architecture hyperparameters for WeatherAlpha-MoE: a station-specialised weather model
built on Mixture-of-Experts, trained with BitNet b1.58 weight constraints for CPU inference.

**North star:** sub-1°C zero-shot station temperature error at 24hr lead time (ChaosNexus baseline).
**Deployment target:** M4 Pro CPU via MLX (no GPU dependency).

## Key papers to absorb before iterating
- ARROW (2510.09734): Shared-Private MoE + RL adaptive rollout — use this routing pattern
- ChaosNexus (2509.21802): ScaleFormer + MoE for chaotic systems, sub-1°C at 5-day
- PatchMoE (2509.22279): Temporal + channel routing with Recurrent Noisy Gating
- BitNet v2 (2504.18415): Hadamard transformation for 1-bit attention activation outliers

## The target file
`arch_config.json` — architecture hyperparameters only.
This is the ONLY file the agent may modify.

## The metric
Run `benchmark.sh` → 30-min training run on ERA5 subset, outputs:
`val_loss`, `station_mae_24h`, `station_mae_72h`, `params_m` (model size in M params).
Optimise: minimise `station_mae_24h`. Constraint: `params_m < 100` (must fit CPU RAM easily).

## Architecture search space
```json
{
  "n_experts": [4, 8, 16, 32],
  "n_active_experts": [1, 2, 4],
  "patch_size_hours": [6, 12, 24],
  "d_model": [128, 256, 512],
  "n_heads": [4, 8, 16],
  "n_layers": [4, 6, 8, 12],
  "routing_type": ["shared_private", "top_k", "recurrent_noisy"],
  "rollout_intervals": ["fixed_6h", "adaptive_rl", "multi_scale"],
  "bitnet_mode": ["b1.58", "v2_hadamard", "fp32_baseline"]
}
```

## Hypotheses to explore (priority order)
1. ARROW routing (shared_private) vs top-k: shared experts capture global patterns,
   private experts specialise per station cluster
2. Patch size 6h vs 12h: 6h matches FengWu ERA5 cadence but 12h may reduce noise
3. BitNet b1.58 baseline vs fp32: quantify accuracy cost early
4. Adaptive rollout (RL scheduler) vs fixed 6h: ARROW shows RL rollout reduces
   error accumulation on long forecasts
5. 8 experts (2 active) vs 16 experts (4 active): sparse activation for CPU efficiency

## Constraints
- Agent can ONLY modify `arch_config.json`
- Must use BitNet-compatible architecture (no norm-free layers that break 1-bit training)
- params_m must stay < 100M (CPU inference budget)
- Training must complete within 30-min budget on current hardware
- Must train on real ERA5 data subset (no synthetic)

## Build status
🔴 Not built yet. Needs:
1. ERA5 training subset (existing pipeline can provide this)
2. BitNet b1.58 PyTorch training loop (or MLX equivalent)
3. benchmark.sh that runs arch_config.json → 30-min train → metric JSON

## Starred repo intelligence (absorbed 2026-03-13)

### WeatherMesh-3 (windborne/WeatherMesh-3) — ICLR 2026
Architecture: ResConv encoder-decoder + latent_size=1024 transformer
Key: parallel encoders (GFS weight=0.1, HRES weight=0.9) → single ERA5 output mesh
Variables: 13 surface + 13 pressure level variables
Files: model_latlon/{config,encoder,decoder,top}.py
NOTE: No weights released. Architecture only. Use as reference for encoder-decoder design.

### Microsoft Aurora (microsoft/aurora) — Nature 2025
Foundation model approach: pre-train on everything, fine-tune for specific tasks
Handles: temperature, air pollution, ocean waves from same architecture
Run on ERA5 directly (good for our pipeline)
Fine-tuning API available → could fine-tune Aurora for station-level temperature

### NVIDIA Earth2Studio (NVIDIA/earth2studio)
Framework for running AI weather models with ZarrBackend
`from earth2studio.models.px import FCN3` → clean inference API
Supports: FourCastNet3, AIFS, FengWu, Pangu
Our benchmark.sh should use Earth2Studio as the evaluation harness

### maderix/ANE (Apple Neural Engine training)
Status: RESEARCH ONLY — 5-9% ANE utilization, many ops fall back to CPU
Verdict: Do NOT use ANE for WeatherAlphaBitNet. Use MLX instead.
MLX path (trevin-creator/autoresearch-mlx) is production-ready.

### trevin-creator/autoresearch-mlx
MLX port of Karpathy's autoresearch for M4 Pro
No PyTorch required — pure MLX training loop
Use this as the base for WeatherAlphaBitNet M4 Pro training script

## Updated architecture recommendation
1. Model architecture: WeatherMesh-3-style encoder-decoder + ARROW MoE routing
2. Quantization: BitNet b1.58 (microsoft/BitNet for inference, custom training)
3. Training framework: Earth2Studio for data loading + eval, custom BitNet training loop
4. M4 Pro deployment: MLX export (not ANE — too early)
5. Evaluation: StationBench (juaAI/stationbench) for station-level RMSE
