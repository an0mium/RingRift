"""Source-based sample weighting for training data.

Weights samples by data source quality - NN+MCTS games get higher weight
than heuristic games, improving sample efficiency from limited high-quality data.

This module is part of the self-play loop improvement plan (Phase 1).

Usage:
    from app.training.source_weighting import SourceWeightedSampler

    # Create sampler with engine modes
    engine_modes = np.array(['gumbel_mcts', 'heuristic', 'mcts', ...])
    sampler = SourceWeightedSampler(engine_modes)

    # Get weighted indices for batch
    indices = sampler.sample(batch_size=64)

    # Or get weights for WeightedRandomSampler
    weights = sampler.get_sample_weights()
    torch_sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from app.utils.numpy_utils import safe_load_npz

if TYPE_CHECKING:
    from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Engine modes considered high-quality (NN-guided search)
HIGH_QUALITY_MODES = frozenset({
    'gumbel_mcts',
    'gumbel',
    'mcts',
    'nn_mcts',
    'improved_mcts',
    'nn_descent',
    'nnue_guided',
    # Human gameplay is high quality (Jan 2026)
    'human',
    'sandbox_human',  # Sandbox with human player
    'lobby_human',    # Lobby game with human player
})

# Engine modes considered medium quality (some neural guidance)
MEDIUM_QUALITY_MODES = frozenset({
    'policy_only',
    'descent',
    'descent_only',
    'nn_minimax',
})

# Engine modes considered base quality (heuristic-only)
BASE_QUALITY_MODES = frozenset({
    'heuristic',
    'heuristic_only',
    'random',
    'random_only',
})


@dataclass
class SourceWeightConfig:
    """Configuration for source-based sample weighting."""
    high_quality_weight: float = 3.0  # Weight for NN+MCTS games
    medium_quality_weight: float = 1.5  # Weight for policy-only/descent games
    base_quality_weight: float = 1.0  # Weight for heuristic games
    unknown_weight: float = 1.0  # Weight for samples with unknown source
    normalize: bool = True  # Normalize weights to mean=1


@dataclass
class FreshnessWeightConfig:
    """Configuration for data freshness-based weighting.

    December 2025: Added for quality-weighted sampling improvement.
    Fresh data from recent selfplay is weighted higher than older data
    since it reflects current model capabilities better.
    """
    half_life_days: float = 3.0  # Days until weight decays to 50%
    min_weight: float = 0.1  # Minimum weight for very old data
    max_weight: float = 1.0  # Maximum weight for fresh data
    enabled: bool = True  # Enable freshness weighting


def compute_freshness_weight(
    age_seconds: float,
    config: FreshnessWeightConfig | None = None,
) -> float:
    """Compute sample weight based on data age (freshness).

    Uses exponential decay: weight = max_weight * exp(-age / half_life)

    December 2025: Part of quality-weighted sampling improvement.

    Args:
        age_seconds: Age of the data in seconds (time since game was played).
        config: Freshness weighting configuration.

    Returns:
        Weight value between min_weight and max_weight.
    """
    import math

    config = config or FreshnessWeightConfig()

    if not config.enabled or age_seconds <= 0:
        return config.max_weight

    # Convert half_life from days to seconds
    half_life_seconds = config.half_life_days * 24 * 3600

    # Exponential decay formula
    decay = math.exp(-age_seconds * math.log(2) / half_life_seconds)
    weight = config.max_weight * decay

    # Clamp to min_weight
    return max(config.min_weight, weight)


def compute_freshness_weights(
    timestamps: np.ndarray | list[float] | None,
    reference_time: float | None = None,
    config: FreshnessWeightConfig | None = None,
) -> np.ndarray:
    """Compute sample weights based on data freshness for batch of samples.

    December 2025: Part of quality-weighted sampling improvement.

    Args:
        timestamps: Array of Unix timestamps (seconds since epoch) for each sample.
        reference_time: Reference time for computing age (default: current time).
        config: Freshness weighting configuration.

    Returns:
        Array of freshness weights.
    """
    import time

    if timestamps is None or len(timestamps) == 0:
        return np.ones(1)

    config = config or FreshnessWeightConfig()
    reference_time = reference_time or time.time()

    n_samples = len(timestamps)
    weights = np.ones(n_samples, dtype=np.float64)

    for i, ts in enumerate(timestamps):
        age = reference_time - float(ts)
        weights[i] = compute_freshness_weight(age, config)

    return weights


def compute_combined_weights(
    engine_modes: np.ndarray | list[str] | None = None,
    timestamps: np.ndarray | list[float] | None = None,
    source_config: SourceWeightConfig | None = None,
    freshness_config: FreshnessWeightConfig | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Compute combined weights from source quality and data freshness.

    December 2025: Part of quality-weighted sampling improvement.
    Combines engine-based quality weights with time-based freshness weights
    for more effective training sample selection.

    Final weight = source_weight * freshness_weight

    Args:
        engine_modes: Array of engine mode strings for each sample.
        timestamps: Array of Unix timestamps for each sample.
        source_config: Source quality weighting configuration.
        freshness_config: Freshness weighting configuration.
        normalize: Normalize final weights to mean=1.

    Returns:
        Array of combined sample weights.
    """
    # Get source-based weights (or uniform if not provided)
    if engine_modes is not None and len(engine_modes) > 0:
        source_weights = compute_source_weights(engine_modes, source_config)
    else:
        n = len(timestamps) if timestamps is not None else 1
        source_weights = np.ones(n, dtype=np.float64)

    # Get freshness-based weights
    if timestamps is not None and len(timestamps) > 0:
        freshness_weights = compute_freshness_weights(timestamps, config=freshness_config)
    else:
        freshness_weights = np.ones(len(source_weights), dtype=np.float64)

    # Ensure same length
    if len(source_weights) != len(freshness_weights):
        logger.warning(
            f"Weight length mismatch: source={len(source_weights)}, "
            f"freshness={len(freshness_weights)}. Using source length."
        )
        freshness_weights = np.ones(len(source_weights), dtype=np.float64)

    # Combine weights multiplicatively
    combined = source_weights * freshness_weights

    # Normalize if requested
    if normalize and combined.mean() > 0:
        combined = combined / combined.mean()

    return combined


def get_quality_tier(engine_mode: str | None) -> str:
    """Classify engine mode into quality tier.

    Args:
        engine_mode: Engine mode string or None.

    Returns:
        Quality tier: 'high', 'medium', 'base', or 'unknown'.
    """
    if engine_mode is None:
        return 'unknown'

    mode = engine_mode.lower().strip()

    if mode in HIGH_QUALITY_MODES:
        return 'high'
    elif mode in MEDIUM_QUALITY_MODES:
        return 'medium'
    elif mode in BASE_QUALITY_MODES:
        return 'base'
    else:
        return 'unknown'


def compute_source_weights(
    engine_modes: np.ndarray | list[str] | None,
    config: SourceWeightConfig | None = None,
) -> np.ndarray:
    """Compute sample weights based on data source quality.

    Args:
        engine_modes: Array of engine mode strings for each sample.
            If None, returns uniform weights.
        config: Weighting configuration.

    Returns:
        Normalized sample weights as numpy array.
    """
    if engine_modes is None:
        logger.debug("No engine_modes provided, returning uniform weights")
        return np.ones(1)

    config = config or SourceWeightConfig()
    n_samples = len(engine_modes)

    if n_samples == 0:
        return np.array([])

    weights = np.ones(n_samples, dtype=np.float64)

    # Count samples by quality tier for logging
    tier_counts = {'high': 0, 'medium': 0, 'base': 0, 'unknown': 0}

    for i, mode in enumerate(engine_modes):
        tier = get_quality_tier(mode if isinstance(mode, str) else str(mode))
        tier_counts[tier] += 1

        if tier == 'high':
            weights[i] = config.high_quality_weight
        elif tier == 'medium':
            weights[i] = config.medium_quality_weight
        elif tier == 'base':
            weights[i] = config.base_quality_weight
        else:
            weights[i] = config.unknown_weight

    # Normalize to mean=1 if requested
    if config.normalize and weights.mean() > 0:
        weights = weights / weights.mean()

    # Log distribution
    total = sum(tier_counts.values())
    if total > 0:
        logger.info(
            f"Source weighting: high={tier_counts['high']} ({100*tier_counts['high']/total:.1f}%), "
            f"medium={tier_counts['medium']} ({100*tier_counts['medium']/total:.1f}%), "
            f"base={tier_counts['base']} ({100*tier_counts['base']/total:.1f}%), "
            f"unknown={tier_counts['unknown']} ({100*tier_counts['unknown']/total:.1f}%)"
        )

    return weights


class SourceWeightedSampler:
    """Sampler that weights samples by data source quality."""

    def __init__(
        self,
        engine_modes: np.ndarray | list[str] | None,
        config: SourceWeightConfig | None = None,
    ):
        """Initialize the source-weighted sampler.

        Args:
            engine_modes: Array of engine mode strings for each sample.
            config: Weighting configuration.
        """
        self.config = config or SourceWeightConfig()
        self.engine_modes = engine_modes

        if engine_modes is None or len(engine_modes) == 0:
            self.weights = np.array([1.0])
            self._n_samples = 0
        else:
            self.weights = compute_source_weights(engine_modes, self.config)
            self._n_samples = len(engine_modes)

    def __len__(self) -> int:
        return self._n_samples

    def sample(self, n_samples: int) -> np.ndarray:
        """Sample indices with replacement according to weights.

        Args:
            n_samples: Number of samples to draw.

        Returns:
            Array of sampled indices.
        """
        if self._n_samples == 0:
            return np.array([], dtype=np.int64)

        probs = self.weights / self.weights.sum()
        return np.random.choice(
            self._n_samples,
            size=n_samples,
            replace=True,
            p=probs,
        )

    def get_sample_weights(self) -> np.ndarray:
        """Get weights for use with PyTorch WeightedRandomSampler.

        Returns:
            Array of sample weights (not normalized to probabilities).
        """
        return self.weights

    def get_weight(self, idx: int) -> float:
        """Get weight for a specific sample.

        Args:
            idx: Sample index.

        Returns:
            Weight value for the sample.
        """
        if self._n_samples == 0:
            return 1.0
        return float(self.weights[idx])


def create_weighted_sampler_for_npz(npz_path: str, config: SourceWeightConfig | None = None):
    """Create a source-weighted sampler for an NPZ file.

    Looks for 'engine_modes' or 'engine_mode' key in the NPZ file.
    If not found, returns a uniform sampler.

    Args:
        npz_path: Path to NPZ file.
        config: Weighting configuration.

    Returns:
        Tuple of (SourceWeightedSampler, num_samples).
    """
    import numpy as np

    data = safe_load_npz(npz_path)

    # Try to find engine mode data
    engine_modes = None
    for key in ['engine_modes', 'engine_mode', 'source_type', 'data_source']:
        if key in data:
            engine_modes = data[key]
            logger.info(f"Found source metadata '{key}' in {npz_path}")
            break

    if engine_modes is None:
        # No source metadata - use first data array to get sample count
        for key in data.keys():
            arr = data[key]
            if hasattr(arr, 'shape') and len(arr.shape) > 0:
                n_samples = arr.shape[0]
                logger.warning(
                    f"No engine_mode metadata in {npz_path}, "
                    f"using uniform weights for {n_samples} samples"
                )
                return SourceWeightedSampler(None, config), n_samples

    sampler = SourceWeightedSampler(engine_modes, config)
    return sampler, len(sampler)


def add_engine_mode_to_npz(
    input_path: str,
    output_path: str,
    default_mode: str = "heuristic",
) -> None:
    """Add engine_mode metadata to an existing NPZ file.

    Useful for retrofitting existing training data with source information.

    Args:
        input_path: Path to input NPZ file.
        output_path: Path to output NPZ file.
        default_mode: Default engine mode to assign to all samples.
    """
    import numpy as np

    data = dict(safe_load_npz(input_path))

    # Get sample count from first array
    n_samples = None
    for key, arr in data.items():
        if hasattr(arr, 'shape') and len(arr.shape) > 0:
            n_samples = arr.shape[0]
            break

    if n_samples is None:
        raise ValueError(f"Could not determine sample count from {input_path}")

    # Add engine_modes array
    data['engine_modes'] = np.array([default_mode] * n_samples, dtype=object)

    logger.info(f"Adding engine_mode='{default_mode}' to {n_samples} samples")
    np.savez_compressed(output_path, **data)
    logger.info(f"Saved to {output_path}")
