"""
WeatherAlphaBitNet — station-level weather forecasting with BitNet b1.58 MoE.

Architecture:
  - BitNet b1.58: weights ∈ {-1, 0, +1} for CPU-native inference
  - ARROW Shared-Private MoE: global + station-cluster specialisation
  - Hadamard attention (BitNet v2): outlier suppression
  - Adaptive rollout: RL-based 6h/12h/24h step selection

Target: sub-1°C station MAE at 24hr lead time on Apple M4 Pro (MLX).
"""

__version__ = "0.1.0"
__all__ = [
    "BitLinear",
    "BitNetAttention",
    "SharedPrivateRouter",
    "AdaptiveRolloutScheduler",
    "StationEmbedding",
    "WeatherAlphaModel",
]

from weatheralpha.bitnet import BitLinear
from weatheralpha.attention import BitNetAttention
from weatheralpha.routing import SharedPrivateRouter
from weatheralpha.rollout import AdaptiveRolloutScheduler
from weatheralpha.station_embed import StationEmbedding
from weatheralpha.model import WeatherAlphaModel
