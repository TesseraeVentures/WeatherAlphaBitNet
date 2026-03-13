"""
Station ID embeddings — geographic + climate zone.

Each weather station gets a learnable embedding that combines:
  1. Continuous geographic features (lat, lon, elevation) — processed via small MLP
  2. Climate cluster one-hot (great_lakes, subtropical, pacific_coast, etc.)
  3. Learnable station-specific bias (fine-grained station personality)

This embedding is added to the temporal input at each time step, giving the model
a persistent "where am I" signal that biases the private MoE experts appropriately.
"""

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# Climate clusters — must match stations.json
CLUSTERS = [
    "great_lakes",
    "southern_plains",
    "subtropical",
    "pacific_coast",
    "continental",
    "mountain",
    "northeast",
    "unknown",
]
CLUSTER_TO_IDX = {c: i for i, c in enumerate(CLUSTERS)}


class StationEmbedding(nn.Module):
    """
    Learnable station embedding: geo-MLP + cluster one-hot + station bias.

    Produces a (d_model,) vector per station, broadcast over time dimension.

    Args:
        d_model:     Output embedding dimension
        n_stations:  Maximum number of registered stations
        station_file: Path to stations.json (optional — initialises geo features)
    """

    def __init__(self, d_model: int, n_stations: int = 64, station_file: str | Path | None = None):
        super().__init__()
        self.d_model = d_model
        self.n_clusters = len(CLUSTERS)

        # Geographic encoder: (lat, lon_sin, lon_cos, elev_norm) → d_model//2
        geo_dim = 4
        self.geo_encoder = nn.Sequential(
            nn.Linear(geo_dim, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 2),
        )

        # Cluster embedding
        self.cluster_embed = nn.Embedding(self.n_clusters, d_model // 4)

        # Station-specific learnable bias
        self.station_bias = nn.Embedding(n_stations, d_model // 4)

        # Final projection to d_model
        proj_in = d_model // 2 + d_model // 4 + d_model // 4
        self.proj = nn.Linear(proj_in, d_model)

        # Registry: maps station ICAO → (station_idx, cluster_idx, geo_tensor)
        self._station_registry: dict[str, dict] = {}
        self._next_idx = 0

        if station_file is not None:
            self._load_station_file(station_file)

    def _load_station_file(self, path: str | Path):
        """Load station metadata from stations.json."""
        with open(path) as f:
            stations = json.load(f)
        for icao, info in stations.items():
            self.register_station(
                icao=icao,
                lat=info["lat"],
                lon=info["lon"],
                elevation=info.get("elevation_m", 0.0),
                cluster=info.get("cluster", "unknown"),
            )

    def register_station(
        self,
        icao: str,
        lat: float,
        lon: float,
        elevation: float = 0.0,
        cluster: str = "unknown",
    ) -> int:
        """Register a new station and return its index."""
        if icao in self._station_registry:
            return self._station_registry[icao]["idx"]

        idx = self._next_idx
        self._next_idx += 1

        cluster_idx = CLUSTER_TO_IDX.get(cluster, CLUSTER_TO_IDX["unknown"])

        # Normalised geo features
        lat_norm = lat / 90.0
        lon_sin = math.sin(math.radians(lon))
        lon_cos = math.cos(math.radians(lon))
        elev_norm = elevation / 4000.0  # normalise by ~Kilimanjaro

        geo = [lat_norm, lon_sin, lon_cos, elev_norm]

        self._station_registry[icao] = {
            "idx": idx,
            "cluster_idx": cluster_idx,
            "geo": geo,
        }
        return idx

    def station_index(self, icao: str) -> int:
        """Get station index by ICAO code."""
        return self._station_registry[icao]["idx"]

    def forward(self, station_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            station_ids: (B,) integer station indices
        Returns:
            (B, d_model) station embeddings, ready to broadcast over time
        """
        # We need geo features and cluster indices for each station in the batch.
        # These are stored in the registry — we look them up and build tensors.
        B = station_ids.shape[0]
        device = station_ids.device

        geo_list, cluster_list = [], []
        for sid in station_ids.tolist():
            # Find registry entry by index (reverse lookup)
            found = False
            for info in self._station_registry.values():
                if info["idx"] == sid:
                    geo_list.append(info["geo"])
                    cluster_list.append(info["cluster_idx"])
                    found = True
                    break
            if not found:
                # Unknown station: zero geo, "unknown" cluster
                geo_list.append([0.0, 0.0, 1.0, 0.0])
                cluster_list.append(CLUSTER_TO_IDX["unknown"])

        geo = torch.tensor(geo_list, dtype=torch.float32, device=device)           # (B, 4)
        cluster_idx = torch.tensor(cluster_list, dtype=torch.long, device=device)  # (B,)

        geo_emb = self.geo_encoder(geo)                   # (B, d_model//2)
        cluster_emb = self.cluster_embed(cluster_idx)     # (B, d_model//4)
        station_emb = self.station_bias(station_ids)      # (B, d_model//4)

        combined = torch.cat([geo_emb, cluster_emb, station_emb], dim=-1)  # (B, d_model)
        return self.proj(combined)  # (B, d_model)
