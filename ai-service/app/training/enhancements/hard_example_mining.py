"""
Hard Example Mining for Training.

Extracted from training_enhancements.py (December 2025).

This module provides hard example identification and prioritization
for curriculum learning during neural network training.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class HardExample:
    """A hard example identified during training."""
    index: int
    loss: float
    uncertainty: float
    times_sampled: int = 1
    last_seen_step: int = 0


class HardExampleMiner:
    """
    Identifies and prioritizes hard examples for training.

    Hard examples are samples where the model:
    - Has high loss (prediction error)
    - Has high uncertainty (low confidence)
    - Consistently performs poorly

    This implements curriculum learning by focusing on difficult cases
    while maintaining diversity to prevent overfitting.

    Usage:
        miner = HardExampleMiner(buffer_size=10000, hard_fraction=0.3)

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            outputs = model(inputs)
            losses = compute_per_sample_loss(outputs, targets)

            # Record losses for mining
            batch_indices = get_batch_indices(batch_idx, batch_size)
            miner.record_batch(batch_indices, losses)

            # Get indices of hard examples to emphasize
            hard_indices = miner.get_hard_indices(num_samples=batch_size)

            # Optionally create a hard example batch
            if step % hard_batch_interval == 0:
                hard_batch = dataset[hard_indices]
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        hard_fraction: float = 0.3,
        loss_threshold_percentile: float = 80.0,
        uncertainty_weight: float = 0.3,
        decay_rate: float = 0.99,
        min_samples_before_mining: int = 1000,
        max_times_sampled: int = 10,
    ):
        """
        Args:
            buffer_size: Maximum number of examples to track
            hard_fraction: Fraction of hard examples in sampled batches
            loss_threshold_percentile: Percentile above which examples are "hard"
            uncertainty_weight: Weight of uncertainty vs loss in hardness score
            decay_rate: Decay factor for old hardness scores (per step)
            min_samples_before_mining: Minimum samples seen before mining starts
            max_times_sampled: Cap on how many times a hard example can be sampled
        """
        self.buffer_size = buffer_size
        self.hard_fraction = hard_fraction
        self.loss_threshold_percentile = loss_threshold_percentile
        self.uncertainty_weight = uncertainty_weight
        self.decay_rate = decay_rate
        self.min_samples_before_mining = min_samples_before_mining
        self.max_times_sampled = max_times_sampled

        # Track examples: index -> HardExample
        self._examples: dict[int, HardExample] = {}
        self._total_samples_seen = 0
        self._current_step = 0
        self._loss_history: deque = deque(maxlen=10000)

    def record_batch(
        self,
        indices: list[int] | np.ndarray | torch.Tensor,
        losses: list[float] | np.ndarray | torch.Tensor,
        uncertainties: list[float] | np.ndarray | torch.Tensor | None = None,
    ) -> None:
        """
        Record losses and uncertainties for a batch of examples.

        Args:
            indices: Dataset indices for the batch
            losses: Per-sample losses
            uncertainties: Per-sample uncertainties (e.g., entropy of policy)
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()
        if isinstance(losses, torch.Tensor):
            losses = losses.detach().cpu().numpy()
        if uncertainties is not None and isinstance(uncertainties, torch.Tensor):
            uncertainties = uncertainties.detach().cpu().numpy()

        indices = np.asarray(indices).flatten()
        losses = np.asarray(losses).flatten()

        if uncertainties is None:
            uncertainties = np.zeros_like(losses)
        else:
            uncertainties = np.asarray(uncertainties).flatten()

        self._current_step += 1

        for idx, loss, unc in zip(indices, losses, uncertainties, strict=False):
            idx = int(idx)
            self._loss_history.append(loss)

            if idx in self._examples:
                # Update existing example with exponential moving average
                ex = self._examples[idx]
                ex.loss = 0.7 * ex.loss + 0.3 * loss
                ex.uncertainty = 0.7 * ex.uncertainty + 0.3 * unc
                ex.times_sampled += 1
                ex.last_seen_step = self._current_step
            else:
                # Add new example
                self._examples[idx] = HardExample(
                    index=idx,
                    loss=loss,
                    uncertainty=unc,
                    times_sampled=1,
                    last_seen_step=self._current_step,
                )

        self._total_samples_seen += len(indices)

        # Prune buffer if too large
        if len(self._examples) > self.buffer_size:
            self._prune_buffer()

    def _prune_buffer(self) -> None:
        """Remove least hard examples to maintain buffer size."""
        if len(self._examples) <= self.buffer_size:
            return

        # Sort by hardness score and keep top buffer_size
        examples = list(self._examples.values())
        examples.sort(key=lambda e: self._compute_hardness(e), reverse=True)

        # Keep hardest examples
        keep_indices = {e.index for e in examples[:self.buffer_size]}
        self._examples = {
            idx: ex for idx, ex in self._examples.items()
            if idx in keep_indices
        }

    def _compute_hardness(self, example: HardExample) -> float:
        """Compute hardness score for an example."""
        # Decay based on staleness
        staleness = self._current_step - example.last_seen_step
        decay = self.decay_rate ** staleness

        # Penalize over-sampled examples
        sample_penalty = 1.0 / (1.0 + example.times_sampled / self.max_times_sampled)

        # Combine loss and uncertainty
        hardness = (
            (1 - self.uncertainty_weight) * example.loss +
            self.uncertainty_weight * example.uncertainty
        )

        return hardness * decay * sample_penalty

    def get_hard_indices(
        self,
        num_samples: int,
        return_scores: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Get indices of hard examples for focused training.

        Args:
            num_samples: Number of indices to return
            return_scores: Also return hardness scores

        Returns:
            Array of indices (and optionally scores)
        """
        if self._total_samples_seen < self.min_samples_before_mining:
            # Not enough data yet - return empty
            if return_scores:
                return np.array([], dtype=np.int64), np.array([])
            return np.array([], dtype=np.int64)

        if not self._examples:
            if return_scores:
                return np.array([], dtype=np.int64), np.array([])
            return np.array([], dtype=np.int64)

        # Compute hardness for all examples
        examples = list(self._examples.values())
        hardness_scores = np.array([self._compute_hardness(e) for e in examples])
        indices = np.array([e.index for e in examples])

        # Determine threshold
        threshold = np.percentile(hardness_scores, self.loss_threshold_percentile)
        hard_mask = hardness_scores >= threshold

        hard_indices = indices[hard_mask]
        hard_scores = hardness_scores[hard_mask]

        # Sample from hard examples (weighted by score)
        num_to_sample = min(num_samples, len(hard_indices))
        if num_to_sample == 0:
            if return_scores:
                return np.array([], dtype=np.int64), np.array([])
            return np.array([], dtype=np.int64)

        # Weighted sampling
        probs = hard_scores / hard_scores.sum()
        sampled_positions = np.random.choice(
            len(hard_indices),
            size=num_to_sample,
            replace=False,
            p=probs,
        )

        result_indices = hard_indices[sampled_positions]
        result_scores = hard_scores[sampled_positions]

        if return_scores:
            return result_indices, result_scores
        return result_indices

    def get_sample_weights(
        self,
        indices: list[int] | np.ndarray,
        base_weight: float = 1.0,
        hard_weight: float = 2.0,
    ) -> np.ndarray:
        """
        Get sampling weights for a batch, upweighting hard examples.

        Args:
            indices: Dataset indices to get weights for
            base_weight: Weight for normal examples
            hard_weight: Weight for hard examples

        Returns:
            Array of weights for each index
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu().numpy()
        indices = np.asarray(indices)

        weights = np.full(len(indices), base_weight)

        if self._total_samples_seen < self.min_samples_before_mining:
            return weights

        # Get hardness threshold
        if len(self._loss_history) > 100:
            threshold = np.percentile(list(self._loss_history), self.loss_threshold_percentile)
        else:
            return weights

        # Upweight hard examples
        for i, idx in enumerate(indices):
            if idx in self._examples and self._examples[idx].loss >= threshold:
                weights[i] = hard_weight

        return weights

    def create_mixed_batch_indices(
        self,
        batch_size: int,
        all_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Create a batch mixing random and hard examples.

        Args:
            batch_size: Total batch size
            all_indices: All available dataset indices

        Returns:
            Mixed batch of indices
        """
        num_hard = int(batch_size * self.hard_fraction)
        num_random = batch_size - num_hard

        # Get hard examples
        hard_indices = self.get_hard_indices(num_hard)

        # Get random examples (excluding already selected hard ones)
        hard_set = set(hard_indices)
        available = np.array([i for i in all_indices if i not in hard_set])

        if len(available) >= num_random:
            random_indices = np.random.choice(available, size=num_random, replace=False)
        else:
            random_indices = available

        # Combine and shuffle
        batch_indices = np.concatenate([hard_indices, random_indices])
        np.random.shuffle(batch_indices)

        return batch_indices

    def get_statistics(self) -> dict[str, Any]:
        """Get mining statistics."""
        if not self._examples:
            return {
                'total_samples_seen': self._total_samples_seen,
                'tracked_examples': 0,
                'mining_active': False,
            }

        losses = [e.loss for e in self._examples.values()]
        times_sampled = [e.times_sampled for e in self._examples.values()]

        return {
            'total_samples_seen': self._total_samples_seen,
            'tracked_examples': len(self._examples),
            'mining_active': self._total_samples_seen >= self.min_samples_before_mining,
            'mean_loss': np.mean(losses),
            'max_loss': np.max(losses),
            'loss_p90': np.percentile(losses, 90),
            'mean_times_sampled': np.mean(times_sampled),
            'max_times_sampled': np.max(times_sampled),
        }

    def reset(self) -> None:
        """Reset miner state."""
        self._examples.clear()
        self._total_samples_seen = 0
        self._current_step = 0
        self._loss_history.clear()

    # =========================================================================
    # Backwards Compatibility Methods (for drop-in replacement of train_nnue.py version)
    # =========================================================================

    def update_errors(
        self,
        indices: list[int] | np.ndarray,
        errors: list[float] | np.ndarray,
    ) -> None:
        """
        Update error history for given samples (backwards compatible).

        This is an alias for record_batch() to maintain compatibility with
        the train_nnue.py HardExampleMiner implementation.

        Args:
            indices: Dataset indices for the batch
            errors: Per-sample errors (treated as losses)
        """
        self.record_batch(indices, errors, uncertainties=None)

    def get_all_sample_weights(
        self,
        dataset_size: int,
        min_weight: float = 0.5,
        max_weight: float = 3.0,
    ) -> np.ndarray:
        """
        Compute sample weights for the entire dataset (backwards compatible).

        This method returns weights for all samples in the dataset, compatible
        with the train_nnue.py HardExampleMiner.get_sample_weights() method.

        Args:
            dataset_size: Total size of the dataset
            min_weight: Minimum weight for easy samples
            max_weight: Maximum weight for hard samples

        Returns:
            Array of weights for all samples
        """
        weights = np.full(dataset_size, min_weight, dtype=np.float32)

        if self._total_samples_seen < self.min_samples_before_mining:
            return np.ones(dataset_size, dtype=np.float32)

        if len(self._loss_history) < 100:
            return np.ones(dataset_size, dtype=np.float32)

        # Get hardness threshold
        threshold = np.percentile(list(self._loss_history), self.loss_threshold_percentile)

        # Update weights for tracked examples
        for idx, example in self._examples.items():
            if idx < dataset_size:
                # Scale weight based on hardness
                if example.loss >= threshold:
                    # Hard example - higher weight
                    hardness = min(1.0, (example.loss - threshold) / threshold if threshold > 0 else 0)
                    weights[idx] = min_weight + hardness * (max_weight - min_weight)
                else:
                    weights[idx] = min_weight

        return weights

    def get_stats(self) -> dict[str, Any]:
        """
        Get mining statistics (backwards compatible alias for get_statistics).
        """
        stats = self.get_statistics()
        # Map to train_nnue.py expected format
        return {
            'seen_samples': stats.get('total_samples_seen', 0),
            'seen_ratio': stats.get('tracked_examples', 0) / max(1, self.buffer_size),
            'mean_error': stats.get('mean_loss', 0),
            'max_error': stats.get('max_loss', 0),
            'mining_active': stats.get('mining_active', False),
        }


__all__ = [
    "HardExample",
    "HardExampleMiner",
]
