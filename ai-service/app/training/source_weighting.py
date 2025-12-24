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

    data = np.load(npz_path, allow_pickle=True)

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

    data = dict(np.load(input_path, allow_pickle=True))

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
