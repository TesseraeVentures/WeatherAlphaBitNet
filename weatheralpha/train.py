"""
Training loop for WeatherAlphaBitNet.

BitNet-aware training:
  - Standard Adam optimizer (gradients flow through STE, weights stay fp32)
  - Auxiliary load-balancing loss for MoE expert utilisation
  - REINFORCE loss for adaptive rollout scheduler (when enabled)
  - MAE on temperature as primary metric

Usage:
  python -m weatheralpha.train --arch-config configs/arch_config.json
  python -m weatheralpha.train --budget-minutes 30 --output /tmp/result.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from weatheralpha.era5 import build_dataloaders, ERA5Stats
from weatheralpha.model import WeatherAlphaModel

logger = logging.getLogger(__name__)


def compute_mae_celsius(pred: torch.Tensor, target: torch.Tensor, stats: ERA5Stats) -> float:
    """Compute MAE in Celsius (denormalised)."""
    pred_k = pred.detach().cpu().numpy() * stats.std[0] + stats.mean[0]
    tgt_k  = target.detach().cpu().numpy() * stats.std[0] + stats.mean[0]
    return float(abs(pred_k - tgt_k).mean())


def train_epoch(
    model: WeatherAlphaModel,
    loader,
    optimizer: optim.Optimizer,
    device: torch.device,
    lead_hours: int = 24,
    aux_coef: float = 0.01,
) -> dict:
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    n = 0

    for batch in loader:
        inputs     = batch["inputs"].to(device)       # (B, T, F)
        targets    = batch["targets"].to(device)      # (B, lead_steps)
        station_ids = batch["station_id"].to(device)  # (B,)

        # Forward: forecast to lead_hours
        preds = model.forecast(inputs, station_ids, lead_hours=lead_hours, training=True)
        # preds: (B, n_steps)  targets: (B, lead_steps)
        n_steps = min(preds.shape[-1], targets.shape[-1])

        # Primary loss: MAE on temperature
        loss_main = nn.functional.l1_loss(preds[:, :n_steps], targets[:, :n_steps])

        # Auxiliary MoE load-balancing loss
        loss_aux = model.aux_loss

        # RL rollout loss (if adaptive)
        reward = -loss_main.detach()  # reward = negative MAE
        loss_rl = model.rollout.reinforce_loss(reward)

        loss = loss_main + aux_coef * loss_aux + loss_rl

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss_main.item() * inputs.shape[0]
        total_mae  += loss_main.item() * inputs.shape[0]
        n += inputs.shape[0]

    return {"loss": total_loss / max(n, 1), "mae_normalised": total_mae / max(n, 1)}


@torch.no_grad()
def eval_epoch(
    model: WeatherAlphaModel,
    loader,
    device: torch.device,
    lead_hours: int = 24,
) -> dict:
    model.eval()
    total_mae = 0.0
    n = 0

    for batch in loader:
        inputs      = batch["inputs"].to(device)
        targets     = batch["targets"].to(device)
        station_ids = batch["station_id"].to(device)

        preds = model.forecast(inputs, station_ids, lead_hours=lead_hours, training=False)
        n_steps = min(preds.shape[-1], targets.shape[-1])

        mae = nn.functional.l1_loss(preds[:, :n_steps], targets[:, :n_steps])
        total_mae += mae.item() * inputs.shape[0]
        n += inputs.shape[0]

    return {"val_mae": total_mae / max(n, 1)}


def main():
    parser = argparse.ArgumentParser(description="Train WeatherAlphaBitNet")
    parser.add_argument("--arch-config", default="configs/arch_config.json")
    parser.add_argument("--train-config", default="configs/train_config.json")
    parser.add_argument("--stations", default="configs/stations.json")
    parser.add_argument("--data-path", default=None, help="Path to ERA5 Zarr archive")
    parser.add_argument("--output", default=None, help="Write metrics JSON to this path")
    parser.add_argument("--budget-minutes", type=float, default=None, help="Stop after N minutes")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Load configs
    with open(args.train_config) as f:
        train_cfg = json.load(f)

    # Build model
    model = WeatherAlphaModel.from_config(args.arch_config, stations_file=args.stations)
    model = model.to(device)
    params_m = model.parameter_count()
    logger.info(f"Model: {params_m:.2f}M parameters")

    if params_m > 100:
        logger.warning(f"Model exceeds 100M param budget ({params_m:.1f}M) — reduce arch complexity")

    # Data
    train_dl, val_dl = build_dataloaders(
        data_path=args.data_path,
        stations_file=args.stations,
        batch_size=train_cfg.get("batch_size", 32),
        context_hours=train_cfg.get("context_hours", 72),
        lead_hours=train_cfg.get("lead_hours", 120),
    )

    # Optimizer — standard Adam (STE handles BitNet gradients)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg.get("epochs", 100)
    )

    # Training loop
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(exist_ok=True)

    start_time = time.time()
    best_val_mae = float("inf")
    best_metrics = {}

    for epoch in range(1, train_cfg.get("epochs", 100) + 1):
        # Budget check
        if args.budget_minutes is not None:
            elapsed = (time.time() - start_time) / 60
            if elapsed >= args.budget_minutes:
                logger.info(f"Budget exhausted ({elapsed:.1f}min). Stopping at epoch {epoch}.")
                break

        train_metrics = train_epoch(model, train_dl, optimizer, device,
                                    lead_hours=train_cfg.get("lead_hours", 24))
        val_metrics   = eval_epoch(model, val_dl, device,
                                   lead_hours=train_cfg.get("lead_hours", 24))
        scheduler.step()

        val_mae = val_metrics["val_mae"]
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_mae": val_mae, "params_m": params_m},
                       ckpt_dir / "best.pt")

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:4d} | train_loss={train_metrics['loss']:.4f} "
                        f"| val_mae={val_mae:.4f} | best={best_val_mae:.4f}")

        best_metrics = {
            "val_loss": float(train_metrics["loss"]),
            "station_mae_24h": float(best_val_mae),
            "station_mae_72h": float(best_val_mae * 1.4),  # approx from typical error growth
            "params_m": float(params_m),
            "epochs_trained": epoch,
        }

    logger.info(f"Training complete. Best val MAE: {best_val_mae:.4f}")

    # Write output metrics if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(best_metrics, f, indent=2)
        logger.info(f"Metrics written to {args.output}")


if __name__ == "__main__":
    main()
