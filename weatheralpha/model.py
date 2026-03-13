"""
WeatherAlpha core model — BitNet b1.58 MoE transformer.

Architecture:
  Input (B, T, F)  →  patch encoder  →  station embed  →  N × WeatherAlphaBlock
                   →  rollout head   →  output (B, lead, 1)

WeatherAlphaBlock:
  x → BitNetAttention → residual
  x → SharedPrivateRouter (ARROW MoE) → residual

Autoregressive rollout is handled externally by AdaptiveRolloutScheduler,
which calls model.step() with the appropriate step size.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from weatheralpha.attention import BitNetAttention
from weatheralpha.bitnet import BitLinear
from weatheralpha.rollout import AdaptiveRolloutScheduler
from weatheralpha.routing import SharedPrivateRouter
from weatheralpha.station_embed import StationEmbedding


class WeatherAlphaBlock(nn.Module):
    """
    Single transformer block: BitNetAttention + SharedPrivateMoE, both with pre-norm.
    """

    def __init__(self, d_model: int, n_heads: int, n_shared: int, n_private: int,
                 n_active: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = BitNetAttention(d_model, n_heads, dropout=dropout)
        self.router = SharedPrivateRouter(
            d_model, n_shared=n_shared, n_private=n_private,
            n_active_private=n_active, dropout=dropout,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None,
                station_ids: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm attention + residual
        x = x + self.drop(self.attn(self.norm1(x), mask=mask))
        # Pre-norm MoE + residual
        x = x + self.drop(self.router(self.norm2(x), station_ids=station_ids))
        return x

    @property
    def aux_loss(self) -> torch.Tensor:
        return self.router.aux_loss


class PatchEncoder(nn.Module):
    """
    Temporal patch encoder: groups time steps into patches, projects to d_model.
    ChaosNexus-inspired hierarchical patching for multi-scale temporal features.
    """

    def __init__(self, n_features: int, d_model: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        # Linear projection of flattened patch → d_model
        self.proj = nn.Linear(n_features * patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F) raw features
        Returns:
            (B, T//patch_size, d_model) patch embeddings
        """
        B, T, F = x.shape
        P = self.patch_size
        T_p = T // P
        # Reshape: (B, T_p, P*F) → (B, T_p, d_model)
        x_patch = x[:, :T_p * P, :].view(B, T_p, P * F)
        return self.norm(self.proj(x_patch))


class WeatherAlphaModel(nn.Module):
    """
    Full WeatherAlpha model: BitNet b1.58 MoE transformer for station temperature.

    Args (from arch_config.json):
        d_model:           Model dimension (default 256)
        n_heads:           Attention heads (default 8)
        n_layers:          Transformer blocks (default 6)
        n_shared_experts:  Shared experts per block (default 2)
        n_private_experts: Private experts per block (default 8)
        n_active_private:  Active private experts per token (default 2)
        patch_size_hours:  Temporal patch size in hours (default 6)
        rollout_type:      "fixed" or "adaptive_rl" (default "fixed")
        max_lead_time_hours: Maximum forecast lead time (default 120)
        dropout:           Dropout rate (default 0.1)
        n_features:        ERA5 input features (default 7)
        n_stations:        Maximum stations (default 64)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        n_shared_experts: int = 2,
        n_private_experts: int = 8,
        n_active_private: int = 2,
        patch_size_hours: int = 6,
        rollout_type: str = "fixed",
        max_lead_time_hours: int = 120,
        dropout: float = 0.1,
        n_features: int = 7,
        n_stations: int = 64,
        stations_file: str | Path | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size_hours // 6  # assuming 6h ERA5 cadence
        self.max_lead = max_lead_time_hours

        # Input encoding
        self.patch_encoder = PatchEncoder(n_features, d_model, max(1, self.patch_size))
        self.station_embed = StationEmbedding(d_model, n_stations=n_stations, station_file=stations_file)

        # Positional encoding (learnable)
        max_patches = max_lead_time_hours // 6 + 24  # generous upper bound
        self.pos_embed = nn.Embedding(max_patches, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            WeatherAlphaBlock(d_model, n_heads, n_shared_experts, n_private_experts,
                              n_active_private, dropout)
            for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)

        # Prediction head: d_model → 1 (temperature delta)
        self.head = nn.Linear(d_model, 1)

        # Adaptive rollout scheduler
        self.rollout = AdaptiveRolloutScheduler(d_model, max_lead_time_hours, rollout_type)

    @classmethod
    def from_config(cls, config_path: str | Path, stations_file: str | Path | None = None) -> "WeatherAlphaModel":
        """Load model from arch_config.json."""
        with open(config_path) as f:
            cfg = json.load(f)
        return cls(
            d_model=cfg.get("d_model", 256),
            n_heads=cfg.get("n_heads", 8),
            n_layers=cfg.get("n_layers", 6),
            n_shared_experts=cfg.get("n_shared_experts", 2),
            n_private_experts=cfg.get("n_private_experts", 8),
            n_active_private=cfg.get("n_active_private", 2),
            patch_size_hours=cfg.get("patch_size_hours", 6),
            rollout_type=cfg.get("rollout_type", "fixed"),
            max_lead_time_hours=cfg.get("max_lead_time_hours", 120),
            dropout=cfg.get("dropout", 0.1),
            stations_file=stations_file,
        )

    def encode(self, x: torch.Tensor, station_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence to latent representation.

        Args:
            x:           (B, T, F) ERA5 features
            station_ids: (B,) station indices
        Returns:
            (B, T', d_model) latent tokens
        """
        B, T, F = x.shape

        # Patch encoding
        h = self.patch_encoder(x)  # (B, T', d_model)
        T_p = h.shape[1]

        # Add positional embeddings
        pos = torch.arange(T_p, device=x.device)
        h = h + self.pos_embed(pos).unsqueeze(0)  # broadcast over B

        # Add station embedding (broadcast over T)
        station_emb = self.station_embed(station_ids)  # (B, d_model)
        h = h + station_emb.unsqueeze(1)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, station_ids=station_ids)

        return self.norm_out(h)

    def step(self, x: torch.Tensor, station_ids: torch.Tensor) -> torch.Tensor:
        """
        Single autoregressive step: encode context → predict next temperature.

        Returns:
            (B, 1) predicted temperature delta (normalised)
        """
        h = self.encode(x, station_ids)
        return self.head(h[:, -1, :])  # use last token

    def forecast(
        self, x: torch.Tensor, station_ids: torch.Tensor,
        lead_hours: int = 24, training: bool = False,
    ) -> torch.Tensor:
        """
        Autoregressive forecast to lead_hours.

        Args:
            x:           (B, T, F) context window
            station_ids: (B,) station indices
            lead_hours:  desired forecast lead in hours
            training:    passed to rollout scheduler for RL exploration
        Returns:
            (B, n_steps) predicted temperature (normalised)
        """
        state = self.encode(x, station_ids)  # (B, T', d_model)
        state_summary = state.mean(dim=1)    # (B, d_model)

        steps = self.rollout.plan_rollout(state_summary, lead_hours, training=training)

        predictions = []
        current_x = x.clone()

        for step_h in steps:
            pred = self.step(current_x, station_ids)  # (B, 1)
            predictions.append(pred)
            # Roll forward: drop oldest step_h/6 steps, append pred as new feature
            step_pts = max(1, step_h // 6)
            new_feat = torch.zeros(current_x.shape[0], step_pts, current_x.shape[-1], device=x.device)
            from weatheralpha.era5 import T2M_IDX
            new_feat[:, :, T2M_IDX] = pred.expand(-1, step_pts)
            current_x = torch.cat([current_x[:, step_pts:, :], new_feat], dim=1)

        return torch.cat(predictions, dim=-1)  # (B, n_steps)

    def parameter_count(self) -> float:
        """Return model size in millions of parameters."""
        return sum(p.numel() for p in self.parameters()) / 1e6

    @property
    def aux_loss(self) -> torch.Tensor:
        """Sum of load-balancing losses across all MoE blocks."""
        return sum(block.aux_loss for block in self.blocks)
