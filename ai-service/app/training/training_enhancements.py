"""
Training Enhancements for RingRift AI.

This module provides advanced training optimizations:
1. Checkpoint averaging for improved final model
2. Gradient accumulation for larger effective batch sizes
3. Data quality scoring for sample prioritization (with freshness weighting)
4. Adaptive learning rate based on Elo progress
5. Early stopping with patience
6. EWC (Elastic Weight Consolidation) for continual learning
7. Model ensemble support for self-play
8. Value head calibration automation
9. Training anomaly detection (NaN/Inf, loss spikes, gradient explosions)
10. Configurable validation intervals (step/epoch-based, adaptive)

Usage:
    from app.training.training_enhancements import (
        TrainingConfig,
        CheckpointAverager,
        GradientAccumulator,
        DataQualityScorer,
        HardExampleMiner,
        AdaptiveLRScheduler,
        WarmRestartsScheduler,
        AdaptiveGradientClipper,
        EWCRegularizer,
        ModelEnsemble,
        EnhancedEarlyStopping,
        TrainingAnomalyDetector,
        ValidationIntervalManager,
        SeedManager,
        create_training_enhancements,
    )
"""

from __future__ import annotations

import copy
import logging
import math
import time
import warnings
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Sampler

from app.utils.torch_utils import safe_load_checkpoint

# Import from modularized subpackage (December 2025)
from app.training.enhancements import (
    AdaptiveGradientClipper,
    AdaptiveLRScheduler,
    CalibrationAutomation,
    CheckpointAverager,
    GradientAccumulator,
    SeedManager,
    TrainingConfig,
    WarmRestartsScheduler,
    average_checkpoints,
    set_reproducible_seed,
)

# Anomaly detection extracted to dedicated module (December 2025)
from app.training.anomaly_detection import (
    AnomalyEvent,
    TrainingAnomalyDetector,
    TrainingLossAnomalyHandler,
    wire_training_loss_anomaly_handler,
)

# Validation scheduling extracted to dedicated module (December 2025)
from app.training.validation_scheduling import (
    EnhancedEarlyStopping,
    ValidationIntervalManager,
    ValidationResult,
)
from app.training.validation_scheduling import EarlyStopping  # Backwards compatible alias

logger = logging.getLogger(__name__)

# Distillation integration (December 2025)
# Re-export distillation classes for unified training enhancements API
try:
    from app.training.distillation import (
        DistillationConfig,
        DistillationTrainer,
        EnsembleTeacher,
        SoftTargetLoss,
        create_distillation_trainer,
        distill_checkpoint_ensemble,
    )
    HAS_DISTILLATION = True
except ImportError:
    HAS_DISTILLATION = False
    DistillationConfig = None
    DistillationTrainer = None
    EnsembleTeacher = None
    SoftTargetLoss = None
    create_distillation_trainer = None
    distill_checkpoint_ensemble = None

__all__ = [
    # Gradient management
    "AdaptiveGradientClipper",
    # Learning rate schedulers
    "AdaptiveLRScheduler",
    # Anomaly detection (extracted to anomaly_detection.py December 2025)
    "AnomalyEvent",
    # Core utilities
    "CheckpointAverager",
    "DataQualityScorer",
    # Regularization
    "EWCRegularizer",
    "EarlyStopping",  # Backwards compatible alias
    # Training control
    "EnhancedEarlyStopping",
    "GradientAccumulator",
    "HardExampleMiner",
    # Ensemble
    "ModelEnsemble",
    "PerSampleLossTracker",
    # Reproducibility
    "SeedManager",
    "TrainingAnomalyDetector",
    "TrainingLossAnomalyHandler",
    # Configuration
    "TrainingConfig",
    # Validation scheduling (extracted to validation_scheduling.py December 2025)
    "ValidationIntervalManager",
    "ValidationResult",
    "WarmRestartsScheduler",
    # Per-sample loss tracking (2025-12)
    "compute_per_sample_loss",
    # Distillation (2025-12)
    "DistillationConfig",
    "DistillationTrainer",
    "EnsembleTeacher",
    "SoftTargetLoss",
    "create_distillation_trainer",
    "distill_checkpoint_ensemble",
    # Factory function
    "create_training_enhancements",
    # Anomaly handler wiring (December 2025)
    "wire_training_loss_anomaly_handler",
]


# =============================================================================
# 0. Consolidated Training Configuration (Phase 7)
# =============================================================================

# MOVED: TrainingConfig is now imported from app.training.enhancements.training_config
# The class definition has been extracted to the enhancements subpackage.
# Import statement at top of file maintains backward compatibility.

# =============================================================================
# 1. Checkpoint Averaging
# =============================================================================


# MOVED: CheckpointAverager and average_checkpoints() are now imported from
# app.training.enhancements.checkpoint_averaging
# The class definitions have been extracted to the enhancements subpackage.

# =============================================================================
# 2. Gradient Accumulation
# =============================================================================


# MOVED: GradientAccumulator is now imported from
# app.training.enhancements.gradient_management
# The class definition has been extracted to the enhancements subpackage.

# =============================================================================
# 2b. Adaptive Gradient Clipping
# =============================================================================


# MOVED: AdaptiveGradientClipper is now imported from
# app.training.enhancements.gradient_management
# The class definition has been extracted to the enhancements subpackage.


# =============================================================================
# 3. Data Quality Scoring
# =============================================================================


@dataclass
class GameQualityScore:
    """Quality score for a training game."""
    game_id: str
    total_score: float
    length_score: float
    elo_score: float
    diversity_score: float
    decisive_score: float
    freshness_score: float = 1.0  # Phase 7: Time-based freshness (1.0 = newest)

    def to_dict(self) -> dict[str, Any]:
        return {
            'game_id': self.game_id,
            'total_score': self.total_score,
            'length_score': self.length_score,
            'elo_score': self.elo_score,
            'diversity_score': self.diversity_score,
            'decisive_score': self.decisive_score,
            'freshness_score': self.freshness_score,
        }


class DataQualityScorer:
    """Scores training data quality for sample prioritization.

    .. deprecated:: December 2025
        Use ``UnifiedQualityScorer`` from ``app.quality.unified_quality``
        instead. This class is retained for backwards compatibility but
        delegates to the unified scorer for consistency.

        Migration:
            # Old
            from app.training.training_enhancements import DataQualityScorer
            scorer = DataQualityScorer()
            freshness = scorer.compute_freshness_score(timestamp)

            # New
            from app.quality.unified_quality import get_quality_scorer
            scorer = get_quality_scorer()
            freshness = scorer.compute_freshness_score(timestamp)

    Higher quality games get higher sampling weights during training.

    Quality factors:
    - Game length (avoid very short/long games)
    - Elo differential between players
    - Move diversity/entropy
    - Decisive vs drawn games
    - Freshness (Phase 7): Exponential decay based on game age
    """

    def __init__(
        self,
        min_game_length: int = 20,
        max_game_length: int = 500,
        optimal_game_length: int = 100,
        max_elo_diff: float = 400.0,
        decisive_bonus: float = 1.2,
        draw_penalty: float = 0.8,
        freshness_decay_hours: float = 24.0,
        freshness_weight: float = 0.2,
    ):
        """
        Args:
            min_game_length: Minimum acceptable game length
            max_game_length: Maximum acceptable game length
            optimal_game_length: Optimal game length for highest score
            max_elo_diff: Maximum Elo differential for scoring
            decisive_bonus: Bonus multiplier for decisive games
            draw_penalty: Penalty multiplier for drawn games
            freshness_decay_hours: Half-life for freshness decay (default 24h)
            freshness_weight: Weight of freshness in total score (default 0.2)
        """
        warnings.warn(
            "DataQualityScorer is deprecated. Use UnifiedQualityScorer from "
            "app.quality.unified_quality instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.min_game_length = min_game_length
        self.max_game_length = max_game_length
        self.optimal_game_length = optimal_game_length
        self.max_elo_diff = max_elo_diff
        self.decisive_bonus = decisive_bonus
        self.draw_penalty = draw_penalty
        self.freshness_decay_hours = freshness_decay_hours
        self.freshness_weight = freshness_weight

    def compute_freshness_score(
        self,
        game_timestamp: float | None = None,
        current_time: float | None = None,
    ) -> float:
        """
        Compute freshness score using exponential decay.

        Recent games get higher scores, with decay based on freshness_decay_hours.

        Args:
            game_timestamp: Unix timestamp when game was played
            current_time: Current time (default: time.time())

        Returns:
            Freshness score (0-1, where 1 = newest)
        """
        if game_timestamp is None:
            return 0.5  # Neutral if unknown

        if current_time is None:
            current_time = time.time()

        age_hours = (current_time - game_timestamp) / 3600
        if age_hours < 0:
            return 1.0  # Future timestamp = max freshness

        # Exponential decay: score = exp(-age / decay_hours)
        freshness_score = math.exp(-age_hours / self.freshness_decay_hours)
        return max(0.0, min(1.0, freshness_score))

    def score_game(
        self,
        game_id: str,
        game_length: int,
        winner: int | None = None,
        elo_p1: float | None = None,
        elo_p2: float | None = None,
        move_entropy: float | None = None,
        game_timestamp: float | None = None,
        pre_computed_quality: float | None = None,
    ) -> GameQualityScore:
        """
        Score a game's quality for training.

        Args:
            game_id: Unique game identifier
            game_length: Number of moves in the game
            winner: Winner (1, 2, ..., or None for draw)
            elo_p1: Elo rating of player 1
            elo_p2: Elo rating of player 2
            move_entropy: Average move entropy (policy diversity)
            game_timestamp: Unix timestamp when game was played (Phase 7: freshness)
            pre_computed_quality: Pre-computed quality score from game_quality_scorer.py
                                  If provided, used as base quality with freshness applied.

        Returns:
            GameQualityScore with component scores
        """
        # If pre-computed quality available (from game_quality_scorer.py at recording time),
        # use it directly with freshness weighting applied
        if pre_computed_quality is not None:
            freshness_score = self.compute_freshness_score(game_timestamp)
            # Blend pre-computed quality with freshness
            remaining_weight = 1.0 - self.freshness_weight
            total_score = (remaining_weight * pre_computed_quality +
                           self.freshness_weight * freshness_score)
            return GameQualityScore(
                game_id=game_id,
                total_score=total_score,
                length_score=pre_computed_quality,  # Use as proxy
                elo_score=0.5,
                diversity_score=pre_computed_quality,  # Use as proxy
                decisive_score=1.0 if winner else 0.8,
                freshness_score=freshness_score,
            )
        # Length score: Gaussian around optimal length
        if game_length < self.min_game_length:
            length_score = 0.5 * (game_length / self.min_game_length)
        elif game_length > self.max_game_length:
            length_score = 0.5 * (self.max_game_length / game_length)
        else:
            # Gaussian centered at optimal
            sigma = (self.max_game_length - self.min_game_length) / 4
            diff = game_length - self.optimal_game_length
            length_score = math.exp(-(diff ** 2) / (2 * sigma ** 2))

        # Elo score: Prefer balanced games
        if elo_p1 is not None and elo_p2 is not None:
            elo_diff = abs(elo_p1 - elo_p2)
            elo_score = max(0, 1 - (elo_diff / self.max_elo_diff))
        else:
            elo_score = 0.5  # Neutral if unknown

        # Diversity score: Higher entropy = more diverse moves
        if move_entropy is not None:
            # Normalize entropy (typical range 0-4)
            diversity_score = min(1.0, move_entropy / 3.0)
        else:
            diversity_score = 0.5  # Neutral if unknown

        # Decisive score: Bonus for wins, penalty for draws
        if winner is not None and winner > 0:
            decisive_score = self.decisive_bonus
        else:
            decisive_score = self.draw_penalty

        # Phase 7: Freshness score (exponential decay)
        freshness_score = self.compute_freshness_score(game_timestamp)

        # Total score (weighted combination, redistributed with freshness)
        # Original weights: 0.3 length, 0.2 elo, 0.2 diversity, 0.3 decisive
        # New weights: scaled down to make room for freshness
        remaining_weight = 1.0 - self.freshness_weight
        total_score = (
            (0.3 * remaining_weight) * length_score +
            (0.2 * remaining_weight) * elo_score +
            (0.2 * remaining_weight) * diversity_score +
            (0.3 * remaining_weight) * decisive_score +
            self.freshness_weight * freshness_score
        )

        return GameQualityScore(
            game_id=game_id,
            total_score=total_score,
            length_score=length_score,
            elo_score=elo_score,
            diversity_score=diversity_score,
            decisive_score=decisive_score,
            freshness_score=freshness_score,
        )

    def compute_sample_weights(
        self,
        scores: list[GameQualityScore],
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Compute sampling weights from quality scores.

        Args:
            scores: List of quality scores
            temperature: Temperature for softmax (higher = more uniform)

        Returns:
            Normalized sampling weights
        """
        raw_scores = np.array([s.total_score for s in scores])

        # Apply temperature and normalize
        if temperature > 0:
            scaled = raw_scores / temperature
            weights = np.exp(scaled - np.max(scaled))  # Numerically stable softmax
            weights /= weights.sum()
        else:
            weights = np.ones(len(scores)) / len(scores)

        return weights


class QualityWeightedSampler(Sampler):
    """
    PyTorch sampler that weights samples by quality scores.

    Usage:
        scorer = DataQualityScorer()
        scores = [scorer.score_game(...) for game in games]
        sampler = QualityWeightedSampler(scores)
        dataloader = DataLoader(dataset, sampler=sampler)
    """

    def __init__(
        self,
        scores: list[GameQualityScore],
        num_samples: int | None = None,
        replacement: bool = True,
        temperature: float = 1.0,
    ):
        self.scores = scores
        self.num_samples = num_samples or len(scores)
        self.replacement = replacement

        scorer = DataQualityScorer()
        self.weights = torch.from_numpy(
            scorer.compute_sample_weights(scores, temperature)
        ).double()

    def __iter__(self):
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=self.replacement,
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


# =============================================================================
# 3b. Per-Sample Loss Tracking (2025-12)
# =============================================================================


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


# =============================================================================
# 3c. Hard Example Mining (Phase 7)
# =============================================================================


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


# =============================================================================
# 4. Adaptive Learning Rate (MOVED to enhancements/learning_rate_scheduling.py)
# =============================================================================
# AdaptiveLRScheduler - moved to app.training.enhancements.learning_rate_scheduling
# WarmRestartsScheduler - moved to app.training.enhancements.learning_rate_scheduling


# =============================================================================
# 4b. Training Anomaly Detection (MOVED December 2025)
# =============================================================================
# AnomalyEvent, TrainingAnomalyDetector moved to app.training.anomaly_detection
# Import at top of file maintains backward compatibility.


# =============================================================================
# 4c. Configurable Validation Intervals (MOVED December 2025)
# =============================================================================
# ValidationResult, ValidationIntervalManager moved to app.training.validation_scheduling
# Import at top of file maintains backward compatibility.


# =============================================================================
# 5. Enhanced Early Stopping (MOVED December 2025)
# =============================================================================
# EnhancedEarlyStopping, EarlyStopping moved to app.training.validation_scheduling
# Import at top of file maintains backward compatibility.


# =============================================================================
# 6. EWC (Elastic Weight Consolidation) for Continual Learning
# =============================================================================


class EWCRegularizer:
    """
    Elastic Weight Consolidation for continual learning.

    Prevents catastrophic forgetting when training on new data by
    penalizing changes to important parameters.

    Usage:
        ewc = EWCRegularizer(model)

        # After training on task 1
        ewc.compute_fisher(dataloader_task1)

        # When training on task 2
        loss = task_loss + ewc.penalty(model)
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 1000.0,
        normalize_fisher: bool = True,
    ):
        """
        Args:
            model: Model to apply EWC to
            lambda_ewc: Importance weight for EWC penalty
            normalize_fisher: Normalize Fisher information matrix
        """
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.normalize_fisher = normalize_fisher

        self._fisher: dict[str, torch.Tensor] = {}
        self._optimal_params: dict[str, torch.Tensor] = {}
        self._computed = False

    def compute_fisher(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module | None = None,
        num_samples: int = 1000,
        device: torch.device | None = None,
    ) -> None:
        """
        Compute Fisher information matrix from dataloader.

        Args:
            dataloader: DataLoader for computing Fisher
            criterion: Loss function (default: cross entropy)
            num_samples: Number of samples to use
            device: Device for computation
        """
        if device is None:
            device = next(self.model.parameters()).device

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # Store optimal parameters
        self._optimal_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Initialize Fisher to zero
        self._fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self.model.eval()
        samples_seen = 0

        for inputs, targets in dataloader:
            if samples_seen >= num_samples:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            self.model.zero_grad()
            outputs = self.model(inputs)

            # Use log-softmax for computing Fisher
            log_probs = torch.log_softmax(outputs, dim=-1)

            # Sample from output distribution
            labels = torch.distributions.Categorical(logits=outputs).sample()
            loss = -log_probs[range(len(labels)), labels].mean()
            loss.backward()

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self._fisher[name] += param.grad.pow(2)

            samples_seen += len(inputs)

        # Normalize Fisher
        for name in self._fisher:
            self._fisher[name] /= samples_seen

            if self.normalize_fisher:
                max_val = self._fisher[name].max()
                if max_val > 0:
                    self._fisher[name] /= max_val

        self._computed = True
        logger.info(f"Computed Fisher information from {samples_seen} samples")

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty for current model parameters.

        Args:
            model: Model to compute penalty for

        Returns:
            EWC penalty term
        """
        if not self._computed:
            return torch.tensor(0.0)

        penalty = torch.tensor(0.0, device=next(model.parameters()).device)

        for name, param in model.named_parameters():
            if name in self._fisher and param.requires_grad:
                diff = param - self._optimal_params[name]
                penalty += (self._fisher[name] * diff.pow(2)).sum()

        return 0.5 * self.lambda_ewc * penalty

    def save_state(self, path: str | Path) -> None:
        """Save EWC state to file."""
        state = {
            'fisher': self._fisher,
            'optimal_params': self._optimal_params,
            'lambda_ewc': self.lambda_ewc,
            'computed': self._computed,
        }
        torch.save(state, path)

    def load_state(self, path: str | Path) -> None:
        """Load EWC state from file."""
        state = safe_load_checkpoint(path, warn_on_unsafe=False)
        self._fisher = state['fisher']
        self._optimal_params = state['optimal_params']
        self.lambda_ewc = state['lambda_ewc']
        self._computed = state['computed']


# =============================================================================
# 7. Model Ensemble for Self-Play
# =============================================================================


class ModelEnsemble:
    """
    Ensemble of models for diverse self-play opponents.

    Using an ensemble for self-play provides more diverse training data
    and prevents overfitting to a single opponent's weaknesses.

    Usage:
        ensemble = ModelEnsemble(model_class=RingRiftCNN_v2)

        # Add models
        ensemble.add_model(best_model, weight=0.5)
        ensemble.add_model(previous_model, weight=0.3)
        ensemble.add_model(random_model, weight=0.2)

        # Sample opponent for self-play game
        opponent = ensemble.sample_model()
    """

    def __init__(
        self,
        model_class: type,
        model_kwargs: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ):
        """
        Args:
            model_class: Class to instantiate models from
            model_kwargs: Arguments for model constructor
            device: Device for models
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.device = device or torch.device('cpu')

        self._models: list[nn.Module] = []
        self._weights: list[float] = []
        self._names: list[str] = []

    def add_model(
        self,
        model_or_path: nn.Module | str | Path,
        weight: float = 1.0,
        name: str | None = None,
    ) -> None:
        """
        Add a model to the ensemble.

        Args:
            model_or_path: Model instance or path to checkpoint
            weight: Sampling weight (higher = more likely to be chosen)
            name: Optional name for the model
        """
        if isinstance(model_or_path, (str, Path)):
            # Load from checkpoint
            model = self.model_class(**self.model_kwargs).to(self.device)
            ckpt = safe_load_checkpoint(model_or_path, map_location=self.device, warn_on_unsafe=False)
            state_dict = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(state_dict)
            model.eval()
            model_name = name or Path(model_or_path).stem
        else:
            model = model_or_path.to(self.device)
            model.eval()
            model_name = name or f"model_{len(self._models)}"

        self._models.append(model)
        self._weights.append(weight)
        self._names.append(model_name)

        logger.info(f"Added model '{model_name}' to ensemble with weight {weight}")

    def sample_model(self) -> tuple[nn.Module, str]:
        """
        Sample a model from the ensemble based on weights.

        Returns:
            Tuple of (model, model_name)
        """
        if not self._models:
            raise ValueError("No models in ensemble")

        # Normalize weights
        total = sum(self._weights)
        probs = [w / total for w in self._weights]

        idx = np.random.choice(len(self._models), p=probs)
        return self._models[idx], self._names[idx]

    def get_model(self, name: str) -> nn.Module | None:
        """Get a specific model by name."""
        for i, n in enumerate(self._names):
            if n == name:
                return self._models[i]
        return None

    def update_weight(self, name: str, weight: float) -> None:
        """Update the weight of a specific model."""
        for i, n in enumerate(self._names):
            if n == name:
                self._weights[i] = weight
                return

    @property
    def num_models(self) -> int:
        """Number of models in ensemble."""
        return len(self._models)

    @property
    def model_names(self) -> list[str]:
        """Names of all models in ensemble."""
        return list(self._names)


# Note: Value Head Calibration Automation moved to app.training.enhancements.calibration


# =============================================================================
# 8. Calibration & Seed Management (MOVED to enhancements/ subpackage)
# =============================================================================
# CalibrationAutomation - moved to app.training.enhancements.calibration
# SeedManager - moved to app.training.enhancements.seed_management
# set_reproducible_seed() - moved to app.training.enhancements.seed_management


# =============================================================================
# Convenience Functions
# =============================================================================


def create_training_enhancements(
    model: nn.Module,
    optimizer: optim.Optimizer,
    config: dict[str, Any] | None = None,
    validation_fn: Callable[[nn.Module], tuple[float, dict[str, float]]] | None = None,
) -> dict[str, Any]:
    """
    Create a suite of training enhancements with default configuration.

    Args:
        model: Model to enhance training for
        optimizer: Optimizer to use
        config: Optional configuration overrides
        validation_fn: Optional validation function for ValidationIntervalManager

    Returns:
        Dictionary of enhancement objects
    """
    config = config or {}

    enhancements = {
        'checkpoint_averager': CheckpointAverager(
            num_checkpoints=config.get('avg_checkpoints', 5),
        ),
        'gradient_accumulator': GradientAccumulator(
            accumulation_steps=config.get('accumulation_steps', 1),
            max_grad_norm=config.get('max_grad_norm', 1.0),
        ),
        'quality_scorer': DataQualityScorer(
            freshness_decay_hours=config.get('freshness_decay_hours', 24.0),
            freshness_weight=config.get('freshness_weight', 0.2),
        ),
        'adaptive_lr': AdaptiveLRScheduler(
            optimizer=optimizer,
            base_lr=config.get('base_lr', 0.001),
        ),
        'early_stopping': EnhancedEarlyStopping(
            patience=config.get('patience', 10),
        ),
        'ewc': EWCRegularizer(
            model=model,
            lambda_ewc=config.get('lambda_ewc', 1000.0),
        ),
        'calibration': CalibrationAutomation(
            deviation_threshold=config.get('calibration_threshold', 0.05),
        ),
        'anomaly_detector': TrainingAnomalyDetector(
            loss_spike_threshold=config.get('loss_spike_threshold', 3.0),
            gradient_norm_threshold=config.get('gradient_norm_threshold', 100.0),
            halt_on_nan=config.get('halt_on_nan', True),
        ),
        'validation_manager': ValidationIntervalManager(
            validation_fn=validation_fn,
            interval_steps=config.get('validation_interval_steps', 1000),
            interval_epochs=config.get('validation_interval_epochs', None),
            subset_size=config.get('validation_subset_size', 1.0),
            adaptive_interval=config.get('adaptive_validation_interval', False),
        ),
        'hard_example_miner': HardExampleMiner(
            buffer_size=config.get('hard_example_buffer_size', 10000),
            hard_fraction=config.get('hard_example_fraction', 0.3),
            loss_threshold_percentile=config.get('hard_example_percentile', 80.0),
            min_samples_before_mining=config.get('min_samples_before_mining', 1000),
        ),
        'seed_manager': SeedManager(
            seed=config.get('seed'),
            deterministic=config.get('deterministic', False),
            benchmark=config.get('benchmark', True),
        ),
    }

    # Optionally add warm restarts scheduler
    if config.get('lr_scheduler') == 'warm_restarts':
        enhancements['warm_restarts_scheduler'] = WarmRestartsScheduler(
            optimizer=optimizer,
            T_0=config.get('warm_restart_t0', 10),
            T_mult=config.get('warm_restart_t_mult', 2),
            eta_min=config.get('warm_restart_eta_min', 1e-6),
            warmup_steps=config.get('warmup_steps', 0),
        )

    return enhancements


# =============================================================================
# Evaluation Feedback Handler (Phase 2 - December 2025)
# =============================================================================

class EvaluationFeedbackHandler:
    """Adjusts training hyperparameters based on evaluation feedback.

    Subscribes to EVALUATION_COMPLETED events and dynamically adjusts
    learning rate based on Elo trend:
    - Rising Elo → keep or slightly increase LR
    - Flat Elo → reduce LR to fine-tune
    - Falling Elo → significantly reduce LR (potential overtraining)

    Usage:
        handler = EvaluationFeedbackHandler(optimizer, config_key="hex8_2p")
        handler.subscribe()

        # During training loop:
        if handler.should_adjust_lr():
            handler.apply_lr_adjustment()
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        config_key: str,
        min_lr: float = 1e-6,
        max_lr: float = 1e-3,
        elo_history_window: int = 5,
        lr_increase_factor: float = 1.1,
        lr_decrease_factor: float = 0.8,
        lr_plateau_factor: float = 0.95,
        elo_rising_threshold: float = 10.0,
        elo_falling_threshold: float = -10.0,
    ):
        """Initialize the evaluation feedback handler.

        Args:
            optimizer: The PyTorch optimizer to adjust
            config_key: Board configuration key (e.g., "hex8_2p")
            min_lr: Minimum learning rate floor
            max_lr: Maximum learning rate ceiling
            elo_history_window: Number of Elo readings to track
            lr_increase_factor: LR multiplier when Elo is rising
            lr_decrease_factor: LR multiplier when Elo is falling
            lr_plateau_factor: LR multiplier when Elo is flat
            elo_rising_threshold: Elo change to consider "rising"
            elo_falling_threshold: Elo change to consider "falling"
        """
        self.optimizer = optimizer
        self.config_key = config_key
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.elo_history_window = elo_history_window
        self.lr_increase_factor = lr_increase_factor
        self.lr_decrease_factor = lr_decrease_factor
        self.lr_plateau_factor = lr_plateau_factor
        self.elo_rising_threshold = elo_rising_threshold
        self.elo_falling_threshold = elo_falling_threshold

        # State
        self._elo_history: list[float] = []
        self._pending_adjustment: float | None = None
        self._last_adjustment_epoch: int = -1
        self._subscribed = False

        logger.debug(
            f"[EvaluationFeedbackHandler] Initialized for {config_key} "
            f"(LR range: {min_lr:.1e} - {max_lr:.1e})"
        )

    def subscribe(self) -> bool:
        """Subscribe to EVALUATION_COMPLETED events.

        Returns:
            True if subscription succeeded, False otherwise.
        """
        if self._subscribed:
            return True

        try:
            from app.coordination.event_router import get_router, subscribe
            from app.distributed.data_events import DataEventType

            router = get_router()
            if router is None:
                logger.debug("[EvaluationFeedbackHandler] Event router not available")
                return False

            def on_evaluation_completed(event):
                """Handle EVALUATION_COMPLETED events."""
                payload = event.payload if hasattr(event, "payload") else event
                event_config = payload.get("config", "")

                # Only respond to our config's events
                if event_config != self.config_key:
                    return

                elo = payload.get("elo", 0.0)
                self._record_elo(elo)

            subscribe(DataEventType.EVALUATION_COMPLETED, on_evaluation_completed)

            # Also subscribe to HYPERPARAMETER_UPDATED for runtime LR updates (December 2025)
            # This closes the feedback loop: GauntletFeedbackController -> HYPERPARAMETER_UPDATED -> runtime LR change
            def on_hyperparameter_updated(event):
                """Handle HYPERPARAMETER_UPDATED for runtime hyperparameter changes."""
                payload = event.payload if hasattr(event, "payload") else event
                event_config = payload.get("config", "")

                # Only respond to our config's events
                if event_config != self.config_key:
                    return

                parameter = payload.get("parameter", "")
                new_value = payload.get("new_value", None)
                reason = payload.get("reason", "unknown")

                if parameter == "learning_rate" and new_value is not None:
                    try:
                        new_lr = float(new_value)
                        # Clamp to valid range
                        new_lr = max(self.min_lr, min(self.max_lr, new_lr))
                        old_lr = self.optimizer.param_groups[0]["lr"]

                        # Apply immediately to optimizer
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = new_lr

                        logger.info(
                            f"[EvaluationFeedbackHandler] Runtime LR update for {self.config_key}: "
                            f"{old_lr:.2e} -> {new_lr:.2e} (reason: {reason})"
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(f"[EvaluationFeedbackHandler] Invalid LR value: {new_value} ({e})")

                # P0.2 (Dec 2025): Also handle lr_multiplier for relative LR adjustments
                elif parameter == "lr_multiplier" and new_value is not None:
                    try:
                        multiplier = float(new_value)
                        old_lr = self.optimizer.param_groups[0]["lr"]
                        new_lr = old_lr * multiplier
                        # Clamp to valid range
                        new_lr = max(self.min_lr, min(self.max_lr, new_lr))

                        # Apply immediately to optimizer
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = new_lr

                        logger.info(
                            f"[EvaluationFeedbackHandler] Runtime LR multiplier for {self.config_key}: "
                            f"{old_lr:.2e} * {multiplier:.2f} = {new_lr:.2e} (reason: {reason})"
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(f"[EvaluationFeedbackHandler] Invalid LR multiplier: {new_value} ({e})")

            subscribe(DataEventType.HYPERPARAMETER_UPDATED, on_hyperparameter_updated)
            self._subscribed = True

            logger.info(
                f"[EvaluationFeedbackHandler] Subscribed to EVALUATION_COMPLETED + HYPERPARAMETER_UPDATED "
                f"for {self.config_key}"
            )
            return True

        except ImportError:
            logger.debug("[EvaluationFeedbackHandler] Event system not available")
            return False
        except Exception as e:
            logger.debug(f"[EvaluationFeedbackHandler] Failed to subscribe: {e}")
            return False

    def _record_elo(self, elo: float) -> None:
        """Record a new Elo rating and compute LR adjustment."""
        self._elo_history.append(elo)

        # Keep only the last N readings
        if len(self._elo_history) > self.elo_history_window:
            self._elo_history = self._elo_history[-self.elo_history_window:]

        # Need at least 2 readings to compute trend
        if len(self._elo_history) < 2:
            return

        # Compute Elo trend (simple moving average of differences)
        elo_changes = [
            self._elo_history[i] - self._elo_history[i - 1]
            for i in range(1, len(self._elo_history))
        ]
        avg_elo_change = sum(elo_changes) / len(elo_changes)

        # Determine adjustment based on trend
        if avg_elo_change >= self.elo_rising_threshold:
            # Elo is rising - keep or slightly increase LR
            factor = self.lr_increase_factor
            trend = "rising"
        elif avg_elo_change <= self.elo_falling_threshold:
            # Elo is falling - reduce LR significantly
            factor = self.lr_decrease_factor
            trend = "falling"
        else:
            # Elo is flat - slight reduction to fine-tune
            factor = self.lr_plateau_factor
            trend = "plateau"

        self._pending_adjustment = factor

        logger.info(
            f"[EvaluationFeedbackHandler] Elo trend for {self.config_key}: "
            f"{trend} (Δ={avg_elo_change:+.1f}), "
            f"pending LR adjustment: ×{factor:.2f}"
        )

    def should_adjust_lr(self) -> bool:
        """Check if there's a pending LR adjustment."""
        return self._pending_adjustment is not None

    def apply_lr_adjustment(self, current_epoch: int = 0) -> float | None:
        """Apply the pending LR adjustment.

        Args:
            current_epoch: Current training epoch (for debouncing)

        Returns:
            New learning rate if adjusted, None otherwise.
        """
        if self._pending_adjustment is None:
            return None

        # Debounce: don't adjust more than once per epoch
        if current_epoch <= self._last_adjustment_epoch:
            return None

        factor = self._pending_adjustment
        self._pending_adjustment = None
        self._last_adjustment_epoch = current_epoch

        # Get current LR and compute new LR
        current_lr = self.optimizer.param_groups[0]["lr"]
        new_lr = current_lr * factor

        # Clamp to valid range
        new_lr = max(self.min_lr, min(self.max_lr, new_lr))

        # Apply adjustment
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        logger.info(
            f"[EvaluationFeedbackHandler] Adjusted LR for {self.config_key}: "
            f"{current_lr:.2e} → {new_lr:.2e} (×{factor:.2f})"
        )

        return new_lr

    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def get_elo_history(self) -> list[float]:
        """Get recorded Elo history."""
        return list(self._elo_history)

    def get_status(self) -> dict[str, Any]:
        """Get handler status."""
        return {
            "config_key": self.config_key,
            "subscribed": self._subscribed,
            "current_lr": self.get_current_lr(),
            "elo_history": self._elo_history,
            "pending_adjustment": self._pending_adjustment,
            "last_adjustment_epoch": self._last_adjustment_epoch,
        }


def create_evaluation_feedback_handler(
    optimizer: optim.Optimizer,
    config_key: str,
    **kwargs: Any,
) -> EvaluationFeedbackHandler:
    """Factory function to create and subscribe an EvaluationFeedbackHandler.

    Args:
        optimizer: PyTorch optimizer
        config_key: Board configuration (e.g., "hex8_2p")
        **kwargs: Additional handler configuration

    Returns:
        Configured and subscribed handler
    """
    handler = EvaluationFeedbackHandler(optimizer, config_key, **kwargs)
    handler.subscribe()
    return handler


# =============================================================================
# TRAINING_LOSS_ANOMALY → QUALITY_CHECK Handler (MOVED December 2025)
# =============================================================================
# TrainingLossAnomalyHandler, wire_training_loss_anomaly_handler moved to
# app.training.anomaly_detection
# Import at top of file maintains backward compatibility.
