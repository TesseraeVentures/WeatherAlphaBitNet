"""
BitNet b1.58 linear layer.

Weights quantized to {-1, 0, +1} during forward pass via Straight-Through Estimator.
Full-precision weights are maintained for gradient updates (SGD/Adam act on fp32).

Reference: "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
           Ma et al., 2024 — https://arxiv.org/abs/2402.17764
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BitLinear(nn.Linear):
    """
    BitNet b1.58 Linear layer.

    Weights quantized to {-1, 0, +1} during forward pass.
    Full-precision weights maintained for gradient updates.

    Key STE trick:
        w_quant = w + (round(w / scale).clamp(-1,1) * scale - w).detach()
    The .detach() means gradients flow through w as if no quantisation happened,
    but the forward pass uses the quantised weights.

    Activations are quantised to 8-bit (absmax) to avoid fp32 accumulation.
    """

    def weight_quant(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        AbsMax quantization: maps weights to {-1, 0, +1}.

        Scale = mean(|W|) so that most weights land in [-1, 1] before rounding.
        Returns (quantized_weight, scale) — both needed to reconstruct magnitude.
        """
        scale = weight.abs().mean().clamp(min=1e-8)
        quantized = (weight / scale).round().clamp(-1, 1)
        return quantized, scale

    def activation_quant(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-token absmax 8-bit quantization for activations.

        Scale per token (last dim) so outlier tokens don't pollute the batch.
        Returns dequantized tensor (same shape, but values snapped to grid).
        """
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        return (x * scale).round().clamp(-128, 127) / scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight

        # LayerNorm activations before quantisation (stabilises training)
        x_norm = F.layer_norm(x, [x.shape[-1]])

        # STE: forward uses quantised activations, backward uses x_norm
        x_quant = x_norm + (self.activation_quant(x_norm) - x_norm).detach()

        # STE: forward uses quantised weights, backward uses w
        w_quant, scale = self.weight_quant(w)
        w_quant_ste = w + (w_quant * scale - w).detach()

        return F.linear(x_quant, w_quant_ste, self.bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, mode=b1.58"


class BitFFN(nn.Module):
    """
    Feed-forward network with BitLinear layers.
    Standard expand-4x-contract FFN used inside MoE experts.
    """

    def __init__(self, d_model: int, expand: int = 4, dropout: float = 0.0):
        super().__init__()
        d_ff = d_model * expand
        self.fc1 = BitLinear(d_model, d_ff, bias=True)
        self.fc2 = BitLinear(d_ff, d_model, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x))))
