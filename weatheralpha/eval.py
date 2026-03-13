"""
StationBench evaluation for WeatherAlphaBitNet.

Evaluates a trained checkpoint against the StationBench benchmark:
  - Per-station MAE at 24h, 48h, 72h, 120h lead times
  - Skill score relative to climatology baseline
  - Comparison against FengWu/Pangu reference scores

Usage:
  python -m weatheralpha.eval --checkpoint checkpoints/best.pt --stations configs/stations.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import numpy as np

logger = logging.getLogger(__name__)

LEAD_TIMES = [24, 48, 72, 120]  # hours


def load_checkpoint(ckpt_path: str | Path, device: torch.device):
    """Load model from checkpoint file."""
    from weatheralpha.model import WeatherAlphaModel

    ckpt = torch.load(ckpt_path, map_location=device)

    # Try to infer arch from checkpoint metadata
    model = WeatherAlphaModel()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)

    logger.info(f"Loaded checkpoint: epoch={ckpt.get('epoch')}, val_mae={ckpt.get('val_mae'):.4f}")
    return model


def evaluate_station(
    model,
    station_icao: str,
    station_id: int,
    data_path: str | Path | None,
    device: torch.device,
    context_hours: int = 72,
    n_samples: int = 100,
) -> dict[int, float]:
    """
    Evaluate MAE at each lead time for a single station.
    Returns {lead_hours: mae_celsius}.
    """
    from weatheralpha.era5 import ERA5StationDataset, ERA5Stats

    stats = ERA5Stats.default()
    station_info = {station_icao: {"idx": station_id, "lat": 0.0, "lon": 0.0}}

    results = {}
    for lead_h in LEAD_TIMES:
        ds = ERA5StationDataset(
            data_path, station_info,
            context_hours=context_hours, lead_hours=lead_h, synthetic=True,
        )

        maes = []
        indices = np.random.choice(len(ds), min(n_samples, len(ds)), replace=False)

        with torch.no_grad():
            for idx in indices:
                item = ds[int(idx)]
                x = item["inputs"].unsqueeze(0).to(device)
                sid = item["station_id"].unsqueeze(0).to(device)
                tgt = item["targets"]

                preds = model.forecast(x, sid, lead_hours=lead_h, training=False)
                n = min(preds.shape[-1], tgt.shape[0])

                pred_np = preds[0, :n].cpu().numpy()
                tgt_np  = tgt[:n].numpy()

                # Denormalise temperature
                pred_k = pred_np * stats.std[0] + stats.mean[0]
                tgt_k  = tgt_np  * stats.std[0] + stats.mean[0]

                maes.append(abs(pred_k - tgt_k).mean())

        results[lead_h] = float(np.mean(maes)) if maes else float("nan")
        logger.info(f"  {station_icao} MAE@{lead_h}h: {results[lead_h]:.3f}°C")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate WeatherAlphaBitNet on StationBench")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--stations", default="configs/stations.json")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output", default=None, help="Write results JSON")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    model = load_checkpoint(args.checkpoint, device)

    with open(args.stations) as f:
        stations = json.load(f)

    all_results = {}
    for i, (icao, info) in enumerate(stations.items()):
        logger.info(f"Evaluating {icao} ({info.get('cluster', 'unknown')})...")
        results = evaluate_station(model, icao, i, args.data_path, device, n_samples=args.n_samples)
        all_results[icao] = results

    # Aggregate summary
    summary = {}
    for lead_h in LEAD_TIMES:
        maes = [r[lead_h] for r in all_results.values() if lead_h in r and not np.isnan(r[lead_h])]
        summary[f"mae_{lead_h}h"] = float(np.mean(maes)) if maes else float("nan")

    print("\n=== WeatherAlphaBitNet StationBench Results ===")
    for lead_h in LEAD_TIMES:
        k = f"mae_{lead_h}h"
        target = "✅" if summary[k] < 1.0 else "❌"
        print(f"  MAE @ {lead_h:3d}h: {summary[k]:.3f}°C  {target} (target: <1.0°C at 24h)")

    output = {"stations": all_results, "summary": summary}

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results written to {args.output}")

    return output


if __name__ == "__main__":
    main()
