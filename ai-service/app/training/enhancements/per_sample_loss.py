"""
Per-Sample Loss Tracking for Training.

Extracted from training_enhancements.py (December 2025).

This module provides per-sample loss computation and tracking,
useful for hard example mining and curriculum learning.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


def compute_per_sample_loss(
    policy_logits: torch.Tensor,
    policy_targets: torch.Tensor,
    value_pred: torch.Tensor,
    value_targets: torch.Tensor,
    policy_weight: float = 1.0,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Compute per-sample combined loss for policy and value heads.

    This function computes individual losses for each sample in a batch,
    which is useful for hard example mining and curriculum learning.

    Args:
        policy_logits: Model policy output (B, num_actions) - logits
        policy_targets: Policy labels (B, num_actions) - probabilities
        value_pred: Model value output (B,) or (B, 1)
        value_targets: Value labels (B,) or (B, 1)
        policy_weight: Weight for policy loss relative to value (default 1.0)
        reduction: "none" for per-sample, "mean"/"sum" for aggregated

    Returns:
        Per-sample losses of shape (B,) if reduction="none", else scalar
    """
    batch_size = policy_logits.shape[0]

    # Policy loss: Cross-entropy (for probability targets)
    # Use log_softmax + element-wise multiply + sum
    log_probs = torch.nn.functional.log_softmax(policy_logits, dim=-1)
    # Handle sparse vs dense targets
    if policy_targets.dim() == 1:
        # Sparse targets (class indices)
        policy_loss = torch.nn.functional.cross_entropy(
            policy_logits, policy_targets, reduction="none"
        )
    else:
        # Dense targets (probability distribution)
        # Cross-entropy: -sum(target * log_prob)
        policy_loss = -torch.sum(policy_targets * log_probs, dim=-1)

    # Value loss: MSE per sample
    value_pred_flat = value_pred.view(batch_size)
    value_targets_flat = value_targets.view(batch_size)
    value_loss = (value_pred_flat - value_targets_flat).pow(2)

    # Combined loss
    combined = value_loss + policy_weight * policy_loss

    if reduction == "mean":
        return combined.mean()
    elif reduction == "sum":
        return combined.sum()
    return combined


@dataclass
class PerSampleLossRecord:
    """Record of per-sample loss for tracking across epochs."""
    sample_idx: int
    loss: float
    policy_loss: float
    value_loss: float
    epoch: int
    batch_idx: int


class PerSampleLossTracker:
    """
    Tracks per-sample losses across training for analysis and debugging.

    Useful for:
    - Identifying consistently hard examples
    - Detecting potential data quality issues
    - Curriculum learning adjustments
    - Training stability monitoring

    Usage:
        tracker = PerSampleLossTracker(max_samples=10000)

        for batch_idx, batch in enumerate(dataloader):
            outputs = model(batch)
            per_sample_losses = compute_per_sample_loss(...)

            tracker.record_batch(
                batch_indices=batch['indices'],
                losses=per_sample_losses,
                epoch=current_epoch,
                batch_idx=batch_idx,
            )

        # Get statistics
        stats = tracker.get_statistics()
        hard_samples = tracker.get_hardest_samples(n=100)
    """

    def __init__(
        self,
        max_samples: int = 10000,
        history_epochs: int = 3,
        percentile_hard: float = 90.0,
    ):
        """
        Args:
            max_samples: Maximum samples to track (uses LRU eviction)
            history_epochs: Number of past epochs to keep in history
            percentile_hard: Percentile threshold for "hard" samples
        """
        self.max_samples = max_samples
        self.history_epochs = history_epochs
        self.percentile_hard = percentile_hard

        # Current epoch tracking: sample_idx -> loss
        self._current_losses: dict[int, float] = {}

        # History: list of (epoch, sample_idx, loss) tuples
        self._history: deque = deque(maxlen=max_samples * history_epochs)

        # Running statistics
        self._total_samples = 0
        self._total_loss = 0.0
        self._loss_squared_sum = 0.0
        self._current_epoch = 0

    def record_batch(
        self,
        batch_indices: list[int] | torch.Tensor,
        losses: torch.Tensor,
        epoch: int,
        batch_idx: int = 0,
    ) -> None:
        """Record per-sample losses from a batch."""
        if isinstance(batch_indices, torch.Tensor):
            batch_indices = batch_indices.cpu().tolist()

        losses_np = losses.detach().cpu().numpy()

        for idx, loss in zip(batch_indices, losses_np, strict=False):
            loss_val = float(loss)

            # Update current epoch tracking
            self._current_losses[idx] = loss_val

            # Add to history
            self._history.append((epoch, idx, loss_val))

            # Update running stats
            self._total_samples += 1
            self._total_loss += loss_val
            self._loss_squared_sum += loss_val ** 2

        self._current_epoch = epoch

        # LRU eviction if needed
        if len(self._current_losses) > self.max_samples:
            # Remove oldest entries (crude LRU)
            oldest_keys = list(self._current_losses.keys())[:len(self._current_losses) - self.max_samples]
            for key in oldest_keys:
                del self._current_losses[key]

    def get_statistics(self) -> dict[str, float]:
        """Get aggregate statistics over tracked samples."""
        if self._total_samples == 0:
            return {
                "mean_loss": 0.0,
                "std_loss": 0.0,
                "min_loss": 0.0,
                "max_loss": 0.0,
                "total_samples": 0,
                "tracked_samples": 0,
            }

        mean = self._total_loss / self._total_samples
        variance = (self._loss_squared_sum / self._total_samples) - (mean ** 2)
        std = math.sqrt(max(0, variance))

        losses = list(self._current_losses.values())

        return {
            "mean_loss": mean,
            "std_loss": std,
            "min_loss": min(losses) if losses else 0.0,
            "max_loss": max(losses) if losses else 0.0,
            "total_samples": self._total_samples,
            "tracked_samples": len(self._current_losses),
            "percentile_threshold": np.percentile(losses, self.percentile_hard) if losses else 0.0,
        }

    def get_hardest_samples(self, n: int = 100) -> list[tuple[int, float]]:
        """Get the n samples with highest loss in current epoch."""
        sorted_samples = sorted(
            self._current_losses.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_samples[:n]

    def get_sample_history(self, sample_idx: int) -> list[tuple[int, float]]:
        """Get loss history for a specific sample (epoch, loss)."""
        return [
            (epoch, loss)
            for epoch, idx, loss in self._history
            if idx == sample_idx
        ]

    def reset_epoch(self) -> None:
        """Reset current epoch tracking (call between epochs)."""
        self._current_losses.clear()


__all__ = [
    "compute_per_sample_loss",
    "PerSampleLossRecord",
    "PerSampleLossTracker",
]
