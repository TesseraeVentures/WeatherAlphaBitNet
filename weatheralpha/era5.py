"""
ERA5 data loading and normalisation for WeatherAlphaBitNet.

Loads ERA5 reanalysis data (via Zarr/xarray) and prepares station-level
input tensors for training. Normalises each variable to zero-mean unit-variance
using climatological statistics.

ERA5 variables used (subset for station temperature prediction):
  - t2m:   2-metre temperature (K) — prediction target
  - u10:   10-metre U wind component (m/s)
  - v10:   10-metre V wind component (m/s)
  - msl:   Mean sea-level pressure (Pa)
  - tp:    Total precipitation (m/hr)
  - d2m:   2-metre dewpoint (K)
  - sp:    Surface pressure (Pa)

Data cadence: 6-hourly (matches FengWu/GraphCast)
Spatial: nearest-neighbour extraction to station lat/lon
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ERA5 variables in feature order (index → variable)
ERA5_VARS = ["t2m", "u10", "v10", "msl", "tp", "d2m", "sp"]
N_FEATURES = len(ERA5_VARS)
T2M_IDX = ERA5_VARS.index("t2m")  # prediction target index


@dataclass
class ERA5Stats:
    """Per-variable climatological statistics for normalisation."""
    mean: np.ndarray  # (N_FEATURES,)
    std: np.ndarray   # (N_FEATURES,)

    @classmethod
    def default(cls) -> "ERA5Stats":
        """Approximate global ERA5 statistics (replace with computed values)."""
        mean = np.array([288.0, 0.0, 0.0, 101325.0, 0.0, 280.0, 95000.0], dtype=np.float32)
        std  = np.array([20.0,  5.0, 5.0,  1000.0,  0.001, 15.0,  5000.0], dtype=np.float32)
        return cls(mean=mean, std=std)

    def normalise(self, x: np.ndarray) -> np.ndarray:
        """x shape: (..., N_FEATURES)"""
        return (x - self.mean) / (self.std + 1e-8)

    def denormalise_t2m(self, x: np.ndarray) -> np.ndarray:
        """Inverse-normalise temperature only."""
        return x * self.std[T2M_IDX] + self.mean[T2M_IDX]


class ERA5StationDataset(Dataset):
    """
    PyTorch Dataset for ERA5 station-level sequences.

    Each item is a (context_window, lead_steps) pair:
      - inputs:  (context_steps, N_FEATURES) normalised ERA5 variables
      - targets: (lead_steps,) future t2m values (normalised)
      - station_id: int station index

    Data source: expects a Zarr archive or pre-extracted NPZ files.
    Falls back to synthetic data if no real data is available (for testing scaffold).
    """

    def __init__(
        self,
        data_path: str | Path | None,
        stations: dict,  # icao → {idx, lat, lon, ...}
        stats: ERA5Stats | None = None,
        context_hours: int = 72,
        lead_hours: int = 120,
        step_hours: int = 6,
        split: str = "train",  # "train", "val", "test"
        synthetic: bool = False,
    ):
        self.data_path = Path(data_path) if data_path else None
        self.stations = stations
        self.stats = stats or ERA5Stats.default()
        self.context_steps = context_hours // step_hours
        self.lead_steps = lead_hours // step_hours
        self.step_hours = step_hours
        self.split = split
        self.synthetic = synthetic or (data_path is None)

        self._station_list = list(stations.keys())
        self._station_idx = {k: v["idx"] for k, v in stations.items()}

        if not self.synthetic:
            self._load_data()
        else:
            logger.warning("ERA5StationDataset: using SYNTHETIC data — not suitable for real training!")
            self._setup_synthetic()

    def _load_data(self):
        """Load real ERA5 data from Zarr or NPZ."""
        try:
            import zarr
            self._zarr = zarr.open(str(self.data_path), mode="r")
            self._n_time = self._zarr["t2m"].shape[0]
            logger.info(f"Loaded ERA5 Zarr: {self._n_time} timesteps from {self.data_path}")
        except Exception as e:
            logger.warning(f"Could not load ERA5 data: {e}. Falling back to synthetic.")
            self.synthetic = True
            self._setup_synthetic()

    def _setup_synthetic(self):
        """Generate synthetic sinusoidal data for testing."""
        n_time = 4 * 365 * 4  # 4 years × 6h cadence
        t = np.linspace(0, 4 * 2 * np.pi, n_time)
        self._synthetic_data = {}
        for icao in self._station_list:
            # Seasonal + diurnal temperature cycle with noise
            t2m = 288.0 + 15.0 * np.sin(t) + 5.0 * np.sin(t * 4) + np.random.randn(n_time) * 2
            features = np.zeros((n_time, N_FEATURES), dtype=np.float32)
            features[:, T2M_IDX] = t2m
            self._synthetic_data[icao] = features
        self._n_time = n_time

    def __len__(self) -> int:
        total_steps = self._n_time - self.context_steps - self.lead_steps
        return max(0, total_steps) * len(self._station_list)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        station_local = idx % len(self._station_list)
        time_idx = idx // len(self._station_list)

        icao = self._station_list[station_local]
        station_id = self._station_idx[icao]

        # Fetch context and target windows
        t_start = time_idx
        t_end_ctx = t_start + self.context_steps
        t_end_tgt = t_end_ctx + self.lead_steps

        if self.synthetic:
            raw = self._synthetic_data[icao][t_start:t_end_ctx]
            tgt_raw = self._synthetic_data[icao][t_end_ctx:t_end_tgt, T2M_IDX]
        else:
            # Real zarr loading (placeholder — actual indexing depends on zarr layout)
            raw = np.zeros((self.context_steps, N_FEATURES), dtype=np.float32)
            tgt_raw = np.zeros(self.lead_steps, dtype=np.float32)

        inputs = self.stats.normalise(raw)
        targets = (tgt_raw - self.stats.mean[T2M_IDX]) / (self.stats.std[T2M_IDX] + 1e-8)

        return {
            "inputs": torch.tensor(inputs, dtype=torch.float32),    # (ctx, F)
            "targets": torch.tensor(targets, dtype=torch.float32),  # (lead,)
            "station_id": torch.tensor(station_id, dtype=torch.long),
        }


def build_dataloaders(
    data_path: str | Path | None,
    stations_file: str | Path,
    batch_size: int = 32,
    context_hours: int = 72,
    lead_hours: int = 120,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders from ERA5 station data."""
    with open(stations_file) as f:
        raw_stations = json.load(f)

    # Assign integer indices
    stations = {
        icao: {**info, "idx": i}
        for i, (icao, info) in enumerate(raw_stations.items())
    }

    train_ds = ERA5StationDataset(data_path, stations, context_hours=context_hours, lead_hours=lead_hours, split="train")
    val_ds   = ERA5StationDataset(data_path, stations, context_hours=context_hours, lead_hours=lead_hours, split="val")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl
