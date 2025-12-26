"""
Data Quality Scoring for Training.

Extracted from training_enhancements.py (December 2025).

This module provides data quality scoring for sample prioritization,
including freshness weighting and quality-weighted sampling.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Sampler


@dataclass
class GameQualityScore:
    """Quality score for a training game."""
    game_id: str
    total_score: float
    length_score: float
    elo_score: float
    diversity_score: float
    decisive_score: float
    freshness_score: float = 1.0  # Time-based freshness (1.0 = newest)

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
    - Freshness: Exponential decay based on game age
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
            game_timestamp: Unix timestamp when game was played (freshness)
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

        # Freshness score (exponential decay)
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

        # Suppress deprecation warning during sampler init
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
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


__all__ = [
    "GameQualityScore",
    "DataQualityScorer",
    "QualityWeightedSampler",
]
