"""
Shared-Private MoE routing — ARROW pattern.

ARROW (Adaptive Rollout With Routing, 2510.09734) uses:
  - Shared experts: always active, capture global atmospheric patterns (e.g. pressure gradients)
  - Private experts: top-k routing, specialise per station cluster (coastal vs continental vs mountain)

This is superior to standard top-k MoE for weather because some patterns (Rossby waves,
jet stream dynamics) are universal while others (sea-breeze, orographic lift) are local.

Reference: "ARROW: Adaptive Rollout With Routing for Weather Forecasting"
           https://arxiv.org/abs/2510.09734
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from weatheralpha.bitnet import BitFFN, BitLinear


class SharedPrivateRouter(nn.Module):
    """
    ARROW-style Shared-Private MoE routing.

    Forward pass:
      output = mean(shared_experts(x)) + weighted_sum(top_k_private_experts(x))

    Load-balancing loss should be added to the training objective to prevent
    private expert collapse. Access via self.aux_loss after each forward pass.

    Args:
        d_model:          Model dimension
        n_shared:         Number of shared experts (always active)
        n_private:        Total number of private experts
        n_active_private: Top-k experts selected per token
        dropout:          Dropout inside experts
        load_balance_coef: Weight for auxiliary load-balancing loss
    """

    def __init__(
        self,
        d_model: int,
        n_shared: int = 2,
        n_private: int = 8,
        n_active_private: int = 2,
        dropout: float = 0.0,
        load_balance_coef: float = 0.01,
    ):
        super().__init__()
        self.n_shared = n_shared
        self.n_private = n_private
        self.n_active = n_active_private
        self.load_balance_coef = load_balance_coef

        # Shared experts (always active — global atmospheric patterns)
        self.shared_experts = nn.ModuleList([BitFFN(d_model, dropout=dropout) for _ in range(n_shared)])

        # Private experts (top-k routing — station cluster specialisation)
        self.private_experts = nn.ModuleList([BitFFN(d_model, dropout=dropout) for _ in range(n_private)])

        # Router: maps token → logits over private experts
        self.router = BitLinear(d_model, n_private, bias=False)

        # Auxiliary load-balancing loss (written during forward, read by training loop)
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor, station_ids: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:           (B, T, D) input tokens
            station_ids: (B,) optional station indices — reserved for future
                         station-conditioned routing bias
        Returns:
            (B, T, D) routed output
        """
        B, T, D = x.shape

        # ── Shared experts (always run, average their outputs) ──────────────
        shared_out = torch.stack([expert(x) for expert in self.shared_experts], dim=0).mean(0)
        # → (B, T, D)

        # ── Private experts (top-k token routing) ───────────────────────────
        router_logits = self.router(x)  # (B, T, n_private)
        router_probs = F.softmax(router_logits, dim=-1)  # normalised weights

        topk_scores, topk_idx = router_probs.topk(self.n_active, dim=-1)
        # topk_scores: (B, T, n_active) — re-normalise selected weights
        topk_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        private_out = torch.zeros_like(x)  # (B, T, D)
        x_flat = x.view(B * T, D)

        for i, expert in enumerate(self.private_experts):
            # Boolean mask: which (b, t) tokens select expert i?
            is_selected = (topk_idx == i)           # (B, T, n_active)
            token_mask = is_selected.any(dim=-1)    # (B, T)

            if not token_mask.any():
                continue

            flat_mask = token_mask.view(B * T)  # (B*T,)
            selected_x = x_flat[flat_mask]       # (n_sel, D)

            expert_out = expert(selected_x)      # (n_sel, D)

            # Weight: sum of weights for this expert across active slots
            # topk_weights has shape (B, T, n_active); is_selected same shape
            w = (topk_weights * is_selected.float()).sum(-1)  # (B, T)
            w_flat = w.view(B * T)[flat_mask].unsqueeze(-1)   # (n_sel, 1)

            private_out.view(B * T, D)[flat_mask] += expert_out * w_flat

        # ── Auxiliary load-balancing loss ────────────────────────────────────
        # Encourages uniform expert utilisation.
        # Loss = n_private * sum(f_i * P_i) where f_i = fraction of tokens to expert i
        # and P_i = mean router probability for expert i.
        token_counts = (topk_idx.view(B * T, self.n_active) == torch.arange(self.n_private, device=x.device).unsqueeze(0)).float().sum(0)
        f = token_counts / (B * T)  # fraction of tokens → each expert
        p = router_probs.view(B * T, self.n_private).mean(0)  # mean routing prob
        self.aux_loss = self.load_balance_coef * self.n_private * (f * p).sum()

        return shared_out + private_out

    def extra_repr(self) -> str:
        return (
            f"n_shared={self.n_shared}, n_private={self.n_private}, "
            f"n_active={self.n_active}, lb_coef={self.load_balance_coef}"
        )
