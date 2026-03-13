"""
BitNet-compatible multi-head attention with Hadamard outlier suppression.

BitNet v2 innovation: apply a normalised Hadamard matrix before Q/K/V projections.
Hadamard spreads activation outliers across all dimensions, making 1-bit quantisation
far more accurate (no single dimension dominates the absmax scale).

Reference: "BitNet v2: Native 4-bit Activations for 1-bit LLMs"
           Wang et al., 2025 — https://arxiv.org/abs/2504.18415
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from weatheralpha.bitnet import BitLinear


class BitNetAttention(nn.Module):
    """
    Multi-head attention with BitNet b1.58 projections and Hadamard pre-transform.

    Architecture:
      x  →  Hadamard(x)  →  Q/K/V (BitLinear)  →  scaled dot-product  →  Out (BitLinear)

    The Hadamard transform is its own inverse (H @ H = I * n), so it's lossless
    and free to invert. We don't invert — we just transform before projection so
    that the BitLinear activation quantiser sees a flatter distribution.

    Requires d_model to be a power of 2 for the Walsh-Hadamard matrix.
    If d_model is not a power of 2, the Hadamard is padded and sliced.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        # BitLinear projections (1-bit weights)
        self.q_proj = BitLinear(d_model, d_model, bias=False)
        self.k_proj = BitLinear(d_model, d_model, bias=False)
        self.v_proj = BitLinear(d_model, d_model, bias=False)
        self.out_proj = BitLinear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)

        # Hadamard matrix for outlier suppression (BitNet v2)
        # Registered as buffer so it moves to the right device automatically
        self.register_buffer("hadamard", self._hadamard_matrix(d_model))

    @staticmethod
    def _hadamard_matrix(n: int) -> torch.Tensor:
        """
        Generate normalised Hadamard matrix of size n×n.

        Uses scipy.linalg.hadamard which requires n to be a power of 2.
        If n is not a power of 2, we build a larger H and slice the top-left n×n block,
        then re-normalise. (Minor approximation; still greatly reduces outliers.)
        """
        import scipy.linalg

        # Find next power of 2
        p = 1
        while p < n:
            p *= 2

        h_full = torch.tensor(scipy.linalg.hadamard(p), dtype=torch.float32)
        h = h_full[:n, :n]
        # Normalise so H @ H.T ≈ I (exact when n == p)
        h = h / math.sqrt(p)
        return h  # (n, n)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:    (B, T, D) input tensor
            mask: (B, T) or (B, 1, T, T) boolean mask — True = keep, False = mask out
        Returns:
            (B, T, D) attended output
        """
        B, T, D = x.shape

        # Apply Hadamard before projections to suppress outliers
        x_had = x @ self.hadamard  # (B, T, D)

        # Q, K, V projections via BitLinear
        def _proj_split(proj, z):
            return proj(z).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q = _proj_split(self.q_proj, x_had)  # (B, H, T, d_head)
        k = _proj_split(self.k_proj, x_had)
        v = _proj_split(self.v_proj, x_had)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        if mask is not None:
            if mask.dim() == 2:
                # (B, T) → (B, 1, 1, T)
                mask = mask[:, None, None, :]
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = self.attn_drop(F.softmax(scores, dim=-1))

        # Aggregate values and project out
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
        return self.out_proj(out)

    def extra_repr(self) -> str:
        return f"n_heads={self.n_heads}, d_head={self.d_head}, hadamard=True"
