"""
Adaptive Rollout Scheduler — ARROW RL-based step selection.

Instead of always rolling out with a fixed 6h step, the scheduler learns to choose
among {6h, 12h, 24h} steps per forecast. Longer steps accumulate less error on
slow-varying patterns (e.g. synoptic scale), while 6h is needed for fast convection.

The scheduler is trained with REINFORCE: reward = −MAE at target lead time.
At inference it acts greedily (argmax over action logits).

Reference: ARROW (2510.09734) Section 3.3 "Adaptive Rollout with RL"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Available rollout step sizes (hours)
STEP_CHOICES = [6, 12, 24]


class RolloutPolicy(nn.Module):
    """
    Small policy network: hidden state → action logits over step sizes.

    Input: current forecast state summary (mean-pooled over spatial dims) + lead time embedding.
    Output: logits over STEP_CHOICES.

    Deliberately kept small (fp32) since it controls the rollout, not the forecast itself.
    """

    def __init__(self, d_model: int, n_actions: int = len(STEP_CHOICES)):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 2),  # +1 for normalised lead time
            nn.GELU(),
            nn.Linear(d_model // 2, n_actions),
        )

    def forward(self, state: torch.Tensor, lead_hours: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state:      (B, D) — mean-pooled forecast state
            lead_hours: (B, 1) — current accumulated lead time (normalised by max_lead)
        Returns:
            logits (B, n_actions)
        """
        x = torch.cat([state, lead_hours], dim=-1)
        return self.net(x)


class AdaptiveRolloutScheduler(nn.Module):
    """
    RL-based adaptive rollout scheduler.

    During training:
      - Sample actions from policy (or use argmax for eval)
      - Accumulate log-probs for REINFORCE update
      - Reward = −MAE at final lead time

    During inference:
      - Greedy action selection
      - Returns sequence of step sizes that sums to desired lead time

    Fallback: if rollout_type == "fixed", always uses 6h step (no RL).
    """

    def __init__(self, d_model: int, max_lead_hours: int = 120, rollout_type: str = "fixed"):
        super().__init__()
        self.max_lead = max_lead_hours
        self.rollout_type = rollout_type
        self.step_choices = STEP_CHOICES

        if rollout_type == "adaptive_rl":
            self.policy = RolloutPolicy(d_model, n_actions=len(STEP_CHOICES))
        else:
            self.policy = None

        # Storage for REINFORCE training
        self._log_probs: list[torch.Tensor] = []

    def plan_rollout(
        self,
        state: torch.Tensor,
        target_lead: int,
        training: bool = False,
    ) -> list[int]:
        """
        Plan a sequence of step sizes that reaches target_lead hours.

        Args:
            state:       (B, D) current model state (used by RL policy)
            target_lead: desired total lead time in hours
            training:    if True, sample actions (exploration); else greedy

        Returns:
            List of step sizes in hours, e.g. [6, 12, 6] for 24h lead
        """
        if self.rollout_type == "fixed" or self.policy is None:
            # Default: maximum-stride steps that divide evenly
            step = 6  # smallest safe step
            return [step] * (target_lead // step)

        steps = []
        accumulated = 0
        self._log_probs = []

        while accumulated < target_lead:
            remaining = target_lead - accumulated
            # Only offer steps that don't overshoot
            valid_mask = torch.tensor(
                [s <= remaining for s in self.step_choices],
                dtype=torch.bool,
                device=state.device,
            )

            lead_norm = torch.tensor([[accumulated / self.max_lead]], dtype=torch.float32, device=state.device)
            logits = self.policy(state[:1], lead_norm)  # use first batch item for planning
            logits = logits.masked_fill(~valid_mask, float("-inf"))

            if training:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                self._log_probs.append(dist.log_prob(action))
            else:
                action = logits.argmax(dim=-1)

            chosen_step = self.step_choices[action.item()]
            steps.append(chosen_step)
            accumulated += chosen_step

        return steps

    def reinforce_loss(self, reward: torch.Tensor) -> torch.Tensor:
        """
        REINFORCE loss: −reward * sum(log_probs).

        Args:
            reward: scalar tensor, typically −MAE (higher = better)
        Returns:
            scalar loss to add to total training loss
        """
        if not self._log_probs:
            return torch.tensor(0.0, device=reward.device)
        log_prob_sum = torch.stack(self._log_probs).sum()
        return -reward * log_prob_sum

    def extra_repr(self) -> str:
        return f"rollout_type={self.rollout_type}, max_lead={self.max_lead}h, steps={self.step_choices}"
