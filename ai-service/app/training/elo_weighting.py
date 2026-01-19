"""ELO-Weighted Data Sampling for RingRift AI Training.

Weights training samples by opponent strength - games against stronger
opponents get higher weight, avoiding overfitting on weak-opponent games.

Architecture Note:
    The core Elo sigmoid logic is defined in app.quality.unified_quality.
    This module provides numpy-vectorized implementations for training
    performance. Both modules use the same sigmoid formula:

        sigmoid = 1 / (1 + exp(-elo_diff / scale))
        weight = min_weight + sigmoid * (max_weight - min_weight)

    For single-sample computation, use:
        from app.quality.unified_quality import compute_elo_weights_batch

    For training loops with numpy arrays, use this module's functions
    for better vectorization performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Import unified Elo computation for reference
try:
    from app.quality.unified_quality import compute_elo_weights_batch as _unified_elo_weights
    HAS_UNIFIED_QUALITY = True
except ImportError:
    HAS_UNIFIED_QUALITY = False
    _unified_elo_weights = None

try:
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class EloWeightConfig:
    """Configuration for ELO-based sample weighting."""
    base_elo: float = 1500.0  # Reference Elo
    elo_scale: float = 400.0  # Elo difference scaling
    min_weight: float = 0.2  # Minimum sample weight
    max_weight: float = 3.0  # Maximum sample weight
    normalize_weights: bool = True  # Normalize to mean=1


# =============================================================================
# Unified Sampler Base Class (2025-12)
# =============================================================================


class WeightedSamplerBase:
    """Base class for weighted training samplers.

    Provides a common interface for samplers that weight training samples
    based on various criteria (Elo, uncertainty, quality, etc.).

    Subclasses should:
    1. Initialize `self.weights` as a numpy array of sampling probabilities
    2. Optionally override `_compute_weights()` for custom weight computation

    Interface:
        - `weights`: np.ndarray - Normalized sampling probabilities
        - `sample(n_samples)`: Draw weighted random samples
        - `get_weight(idx)`: Get weight for a specific sample
        - `update_weights(new_weights)`: Update weights from new scores
    """

    weights: np.ndarray

    def sample(self, n_samples: int) -> np.ndarray:
        """Sample indices with replacement according to weights.

        Args:
            n_samples: Number of samples to draw

        Returns:
            Array of sampled indices
        """
        return np.random.choice(
            len(self.weights),
            size=n_samples,
            replace=True,
            p=self.weights / self.weights.sum(),  # Ensure normalized
        )

    def get_weight(self, idx: int) -> float:
        """Get weight for a specific sample.

        Args:
            idx: Sample index

        Returns:
            Weight value for the sample
        """
        return float(self.weights[idx])

    def update_weights(self, new_weights: np.ndarray) -> None:
        """Update sampling weights.

        Args:
            new_weights: New weight array (will be normalized internally)
        """
        self.weights = new_weights.astype(np.float64)


class EloWeightedSampler(WeightedSamplerBase):
    """Weights training samples based on opponent Elo rating."""

    def __init__(
        self,
        sample_elos: np.ndarray,
        model_elo: float = 1500.0,
        config: EloWeightConfig | None = None,
    ):
        """Initialize the ELO-weighted sampler.

        Args:
            sample_elos: Array of opponent Elo ratings for each sample
            model_elo: Current model's Elo rating
            config: Weighting configuration
        """
        self.sample_elos = sample_elos
        self.model_elo = model_elo
        self.config = config or EloWeightConfig()
        self.weights = self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        """Compute sample weights from Elo ratings."""
        config = self.config

        # Higher Elo opponents = higher weight
        elo_diff = self.sample_elos - self.model_elo

        # Sigmoid-like transformation: harder games get more weight
        # Games vs stronger opponents (positive diff) get higher weight
        raw_weights = 1.0 / (1.0 + np.exp(-elo_diff / config.elo_scale))

        # Scale to [0, 1] and then to [min_weight, max_weight]
        weights = config.min_weight + raw_weights * (config.max_weight - config.min_weight)

        # Normalize to mean=1 if requested
        if config.normalize_weights:
            weights = weights / weights.mean()

        return weights

    def update_model_elo(self, new_elo: float):
        """Update model Elo and recompute weights."""
        self.model_elo = new_elo
        self.weights = self._compute_weights()

    # sample() and get_weight() inherited from WeightedSamplerBase


class EloWeightedDataset:
    """Dataset wrapper that applies Elo-based weights during training."""

    def __init__(
        self,
        base_dataset: Dataset,
        sample_elos: np.ndarray,
        model_elo: float = 1500.0,
    ):
        self.base_dataset = base_dataset
        self.sampler = EloWeightedSampler(sample_elos, model_elo)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        item = self.base_dataset[idx]
        weight = self.sampler.get_weight(idx)

        if isinstance(item, tuple):
            return (*item, weight)
        return item, weight


def compute_elo_weights(
    opponent_elos: np.ndarray,
    model_elo: float = 1500.0,
    elo_scale: float = 400.0,
    min_weight: float = 0.2,
    max_weight: float = 3.0,
) -> np.ndarray:
    """Compute sample weights from opponent Elo ratings.

    This is a numpy-optimized version of the canonical implementation in
    app.quality.unified_quality.compute_elo_weights_batch(). Both use
    the same sigmoid formula for consistency.

    Args:
        opponent_elos: Opponent Elo for each sample (numpy array)
        model_elo: Current model Elo
        elo_scale: Scaling factor for Elo difference
        min_weight: Minimum sample weight
        max_weight: Maximum sample weight

    Returns:
        Normalized sample weights (numpy array)
    """
    elo_diff = opponent_elos - model_elo
    raw_weights = 1.0 / (1.0 + np.exp(-elo_diff / elo_scale))
    weights = min_weight + raw_weights * (max_weight - min_weight)
    return weights / weights.mean()


def compute_generator_elo_weights(
    generator_elos: np.ndarray,
    baseline_elo: float = 1000.0,
    elo_scale: float = 200.0,
    min_weight: float = 0.3,
    max_weight: float = 3.0,
) -> np.ndarray:
    """Compute sample weights based on generator model Elo.

    January 2026 - Elo-gated training enhancement.

    Samples from stronger generating models get higher weight, because they
    represent higher quality play. This implements the quality-weighted
    sampling strategy for iterative strength improvement.

    Formula: weight = sigmoid((elo - baseline) / scale)
    Default params give:
        - 800 Elo generator: weight ~0.5x
        - 1000 Elo generator: weight ~1.0x
        - 1200 Elo generator: weight ~2.7x
        - 1500 Elo generator: weight ~2.9x

    Args:
        generator_elos: Generator model Elo for each sample (numpy array)
        baseline_elo: Center point for sigmoid (default: 1000)
        elo_scale: Steepness of sigmoid (lower = steeper, default: 200)
        min_weight: Minimum weight for weakest generators (default: 0.3)
        max_weight: Maximum weight for strongest generators (default: 3.0)

    Returns:
        Normalized sample weights (numpy array, mean=1.0)
    """
    # Sigmoid transformation: higher Elo → higher weight
    elo_diff = generator_elos - baseline_elo
    raw_weights = 1.0 / (1.0 + np.exp(-elo_diff / elo_scale))

    # Scale to [min_weight, max_weight]
    weights = min_weight + raw_weights * (max_weight - min_weight)

    # Normalize to mean=1.0 for use with WeightedRandomSampler
    return weights / weights.mean()


def compute_human_game_weights(
    human_won: np.ndarray,
    ai_difficulty: np.ndarray,
    game_lengths: np.ndarray,
    ai_elo: np.ndarray | None = None,
    win_multiplier: float = 3.0,
    min_weight: float = 0.5,
    max_weight: float = 6.0,
) -> np.ndarray:
    """Compute sample weights for human vs AI games.

    January 2026 - Human game training enhancement.

    Human wins against AI expose blind spots and provide high-quality
    training signal. This function weights samples to prioritize:
    1. Human wins (3x base weight by default)
    2. Games against stronger AI (higher difficulty = higher weight)
    3. Longer games (more learning signal)

    Formula:
        base = 1.0 (loss) or win_multiplier (win)
        difficulty_mult = 1.0 + (difficulty / 10)  # 1.1 to 2.0
        elo_mult = 1.0 + max(0, (ai_elo - 1200) / 400)  # 1.0 to ~2.5
        length_mult = min(2.0, 1.0 + game_length / 50)  # 1.0 to 2.0
        weight = base * difficulty_mult * elo_mult * length_mult

    Example weights:
        - Human loss vs easy AI (diff=3, 20 moves): ~1.3
        - Human win vs easy AI (diff=3, 20 moves): ~3.9
        - Human win vs hard AI (diff=8, 40 moves): ~10.8
        - Human win vs very hard AI (diff=10, 60 moves, 1600 Elo): ~18.0

    Args:
        human_won: Boolean array indicating human wins
        ai_difficulty: AI difficulty level for each game (1-10)
        game_lengths: Number of moves in each game
        ai_elo: Optional AI Elo ratings (uses difficulty if not provided)
        win_multiplier: Weight multiplier for human wins (default: 3.0)
        min_weight: Minimum weight (default: 0.5)
        max_weight: Maximum weight (default: 6.0)

    Returns:
        Normalized sample weights (numpy array, mean=1.0)
    """
    # Base weight: higher for human wins
    base_weights = np.where(human_won, win_multiplier, 1.0)

    # Difficulty scaling: harder AI = more valuable game
    difficulty_mult = 1.0 + (ai_difficulty / 10.0)  # 1.1 to 2.0

    # Elo scaling (if provided): stronger AI = more valuable game
    if ai_elo is not None:
        elo_mult = 1.0 + np.maximum(0, (ai_elo - 1200) / 400)  # 1.0 to ~2.5
    else:
        elo_mult = np.ones_like(base_weights)

    # Length bonus: longer games have more training signal
    length_mult = np.minimum(2.0, 1.0 + game_lengths / 50)  # 1.0 to 2.0

    # Combine all multipliers
    weights = base_weights * difficulty_mult * elo_mult * length_mult

    # Clamp to [min_weight, max_weight]
    weights = np.clip(weights, min_weight, max_weight)

    # Normalize to mean=1.0
    return weights / weights.mean()
