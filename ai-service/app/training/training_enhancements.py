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
    EWCRegularizer,
    EvaluationFeedbackHandler,
    GradientAccumulator,
    ModelEnsemble,
    SeedManager,
    TrainingConfig,
    WarmRestartsScheduler,
    average_checkpoints,
    create_evaluation_feedback_handler,
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
    # Regularization (extracted to enhancements/ewc_regularization.py December 2025)
    "EWCRegularizer",
    "EarlyStopping",  # Backwards compatible alias
    # Training control
    "EnhancedEarlyStopping",
    # Evaluation feedback (extracted to enhancements/evaluation_feedback.py December 2025)
    "EvaluationFeedbackHandler",
    "create_evaluation_feedback_handler",
    "GradientAccumulator",
    "HardExampleMiner",
    # Ensemble (extracted to enhancements/model_ensemble.py December 2025)
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

# MOVED: EWCRegularizer is now imported from
# app.training.enhancements.ewc_regularization
# The class definition has been extracted to the enhancements subpackage.


# =============================================================================
# 7. Model Ensemble for Self-Play
# =============================================================================

# MOVED: ModelEnsemble is now imported from
# app.training.enhancements.model_ensemble
# The class definition has been extracted to the enhancements subpackage.


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

# MOVED: EvaluationFeedbackHandler and create_evaluation_feedback_handler are now imported from
# app.training.enhancements.evaluation_feedback
# The class and factory function have been extracted to the enhancements subpackage.


# =============================================================================
# TRAINING_LOSS_ANOMALY → QUALITY_CHECK Handler (MOVED December 2025)
# =============================================================================
# TrainingLossAnomalyHandler, wire_training_loss_anomaly_handler moved to
# app.training.anomaly_detection
# Import at top of file maintains backward compatibility.
