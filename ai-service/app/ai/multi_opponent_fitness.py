"""Multi-opponent fitness evaluation for robust CMA-ES training.

Evaluates candidates against ALL baseline personas to prevent overfitting
to any single opponent style. This addresses the core issue where training
against a single opponent produced weights that won 81.25% vs that opponent
but only 47.7% in round-robin tournaments.

Key insight: Optimizing against one opponent learns exploits, not generalizable
strategy. Multi-opponent evaluation ensures robust performance.

Fitness aggregation: 0.4 * min(win_rates) + 0.6 * mean(win_rates)
- Pure average allows exploiting weak opponents
- Pure minimum is too conservative
- 40/60 weighting ensures no catastrophic losses while rewarding overall strength
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from app.ai.gpu_parallel_games import evaluate_candidate_fitness_gpu
from app.ai.heuristic_weights import (
    HEURISTIC_V1_BALANCED,
    HEURISTIC_V1_AGGRESSIVE,
    HEURISTIC_V1_TERRITORIAL,
    HEURISTIC_V1_DEFENSIVE,
)

logger = logging.getLogger(__name__)


@dataclass
class MultiOpponentResult:
    """Result from multi-opponent fitness evaluation."""

    per_opponent: dict[str, float]  # Win rates per opponent persona
    aggregate: float  # Combined fitness score
    self_play: float  # Self-play win rate (should be ~0.5)
    games_played: int = 0  # Total games evaluated
    behavioral_vector: np.ndarray | None = None  # For novelty search (optional)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "per_opponent": self.per_opponent,
            "aggregate": self.aggregate,
            "self_play": self.self_play,
            "games_played": self.games_played,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultiOpponentResult":
        """Reconstruct from dict."""
        return cls(
            per_opponent=data["per_opponent"],
            aggregate=data["aggregate"],
            self_play=data["self_play"],
            games_played=data.get("games_played", 0),
        )


# Baseline opponent profiles - these are the 4 core personas
BASELINE_OPPONENTS = {
    "balanced": dict(HEURISTIC_V1_BALANCED),
    "aggressive": dict(HEURISTIC_V1_AGGRESSIVE),
    "territorial": dict(HEURISTIC_V1_TERRITORIAL),
    "defensive": dict(HEURISTIC_V1_DEFENSIVE),
}

# Default evaluation parameters
DEFAULT_GAMES_PER_OPPONENT = 32
DEFAULT_SELF_PLAY_GAMES = 24
DEFAULT_MIN_WEIGHT = 0.4  # Weight for min component in aggregation


def evaluate_multi_opponent(
    candidate_weights: dict[str, float],
    games_per_opponent: int = DEFAULT_GAMES_PER_OPPONENT,
    self_play_games: int = DEFAULT_SELF_PLAY_GAMES,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    board_size: int = 8,
    num_players: int = 2,
    max_moves: int = 200,
    device: torch.device | None = None,
    opponents: dict[str, dict[str, float]] | None = None,
    include_self_play: bool = True,
) -> MultiOpponentResult:
    """Evaluate candidate against all baseline personas.

    Args:
        candidate_weights: Heuristic weights to evaluate
        games_per_opponent: Number of games per opponent (default 32)
        self_play_games: Number of self-play games (default 24)
        min_weight: Weight for min component in aggregation (default 0.4)
        board_size: Board size (default 8 for square8)
        num_players: Number of players (default 2)
        max_moves: Max moves per game (default 200)
        device: Torch device for GPU evaluation
        opponents: Override default opponents (optional)
        include_self_play: Whether to include self-play evaluation

    Returns:
        MultiOpponentResult with per-opponent scores and aggregate fitness
    """
    if opponents is None:
        opponents = BASELINE_OPPONENTS

    results: dict[str, float] = {}
    total_games = 0

    # Evaluate against each opponent
    for opponent_name, opponent_weights in opponents.items():
        logger.debug(f"Evaluating vs {opponent_name} ({games_per_opponent} games)")

        win_rate = evaluate_candidate_fitness_gpu(
            candidate_weights=candidate_weights,
            opponent_weights=opponent_weights,
            num_games=games_per_opponent,
            board_size=board_size,
            num_players=num_players,
            max_moves=max_moves,
            device=device,
        )
        results[opponent_name] = win_rate
        total_games += games_per_opponent

        logger.debug(f"  vs {opponent_name}: {win_rate:.1%}")

    # Self-play evaluation
    self_play_rate = 0.5  # Default if not evaluated
    if include_self_play and self_play_games > 0:
        logger.debug(f"Evaluating self-play ({self_play_games} games)")

        self_play_rate = evaluate_candidate_fitness_gpu(
            candidate_weights=candidate_weights,
            opponent_weights=candidate_weights,
            num_games=self_play_games,
            board_size=board_size,
            num_players=num_players,
            max_moves=max_moves,
            device=device,
        )
        total_games += self_play_games
        logger.debug(f"  self-play: {self_play_rate:.1%}")

    # Compute aggregate fitness
    win_rates = list(results.values())
    aggregate = compute_aggregate_fitness(win_rates, min_weight)

    logger.info(
        f"Multi-opponent fitness: {aggregate:.3f} "
        f"(min={min(win_rates):.1%}, mean={np.mean(win_rates):.1%})"
    )

    return MultiOpponentResult(
        per_opponent=results,
        aggregate=aggregate,
        self_play=self_play_rate,
        games_played=total_games,
    )


def compute_aggregate_fitness(
    win_rates: list[float],
    min_weight: float = DEFAULT_MIN_WEIGHT,
) -> float:
    """Compute aggregate fitness from per-opponent win rates.

    Formula: min_weight * min(win_rates) + (1 - min_weight) * mean(win_rates)

    Args:
        win_rates: List of win rates against each opponent
        min_weight: Weight for minimum component (default 0.4)

    Returns:
        Aggregate fitness score (0.0 to 1.0)
    """
    if not win_rates:
        return 0.0

    min_rate = min(win_rates)
    mean_rate = float(np.mean(win_rates))

    return min_weight * min_rate + (1 - min_weight) * mean_rate


def evaluate_multi_opponent_parallel(
    candidates: list[dict[str, float]],
    games_per_opponent: int = DEFAULT_GAMES_PER_OPPONENT,
    self_play_games: int = DEFAULT_SELF_PLAY_GAMES,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    board_size: int = 8,
    num_players: int = 2,
    max_moves: int = 200,
    device: torch.device | None = None,
    max_workers: int = 4,
) -> list[MultiOpponentResult]:
    """Evaluate multiple candidates in parallel using thread pool.

    This is useful for local multi-GPU evaluation or CPU parallelism.
    For cluster-wide distribution, use the distributed coordinator.

    Args:
        candidates: List of weight dictionaries to evaluate
        games_per_opponent: Games per opponent per candidate
        self_play_games: Self-play games per candidate
        min_weight: Aggregation min weight
        board_size: Board size
        num_players: Number of players
        max_moves: Max moves per game
        device: Torch device (if None, distributes across available GPUs)
        max_workers: Maximum parallel workers

    Returns:
        List of MultiOpponentResult, one per candidate
    """
    from concurrent.futures import ThreadPoolExecutor

    def evaluate_single(idx_weights: tuple[int, dict[str, float]]) -> tuple[int, MultiOpponentResult]:
        idx, weights = idx_weights
        # Assign to GPU round-robin if multiple available
        if device is None and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            assigned_device = torch.device(f"cuda:{idx % gpu_count}")
        else:
            assigned_device = device

        result = evaluate_multi_opponent(
            candidate_weights=weights,
            games_per_opponent=games_per_opponent,
            self_play_games=self_play_games,
            min_weight=min_weight,
            board_size=board_size,
            num_players=num_players,
            max_moves=max_moves,
            device=assigned_device,
        )
        return idx, result

    # Parallel evaluation
    results: list[MultiOpponentResult | None] = [None] * len(candidates)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, result in executor.map(evaluate_single, enumerate(candidates)):
            results[idx] = result

    return [r for r in results if r is not None]


def get_opponent_names() -> list[str]:
    """Get list of opponent persona names."""
    return list(BASELINE_OPPONENTS.keys())


def get_opponent_weights(name: str) -> dict[str, float] | None:
    """Get weights for a specific opponent persona."""
    return BASELINE_OPPONENTS.get(name)


# Convenience function for testing
def quick_evaluate(
    candidate_weights: dict[str, float],
    games: int = 8,
) -> MultiOpponentResult:
    """Quick evaluation with fewer games (for testing)."""
    return evaluate_multi_opponent(
        candidate_weights=candidate_weights,
        games_per_opponent=games,
        self_play_games=games // 2,
    )
