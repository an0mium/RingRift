"""Prioritized Fictitious Self-Play (PFSP) Opponent Pool Management.

Extracted from advanced_training.py (December 2025).

Implements opponent selection strategies from AlphaStar:
- Prioritize opponents the current model struggles against
- Maintain diverse opponent pool
- Balance exploitation (hard opponents) and exploration (new opponents)

Usage:
    from app.training.pfsp_pool import PFSPOpponentPool, OpponentStats

    pool = PFSPOpponentPool()
    pool.add_opponent("models/gen1.pth", elo=1500)
    pool.add_opponent("models/gen2.pth", elo=1600)

    # Get opponent for next game
    opponent = pool.sample_opponent(current_elo=1550)

    # Update after game
    pool.update_stats(opponent.model_path, won=False)
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OpponentStats:
    """Statistics for an opponent in the pool."""
    model_path: str
    model_name: str
    elo: float = 1500.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    last_played: datetime | None = None
    generation: int = 0
    priority_score: float = 1.0


class PFSPOpponentPool:
    """
    Prioritized Fictitious Self-Play (PFSP) opponent pool management.

    Implements opponent selection strategies from AlphaStar:
    - Prioritize opponents the current model struggles against
    - Maintain diverse opponent pool
    - Balance exploitation (hard opponents) and exploration (new opponents)

    Usage:
        pool = PFSPOpponentPool()
        pool.add_opponent("models/gen1.pth", elo=1500)
        pool.add_opponent("models/gen2.pth", elo=1600)

        # Get opponent for next game
        opponent = pool.sample_opponent(current_elo=1550)

        # Update after game
        pool.update_stats(opponent.model_path, won=False)
    """

    def __init__(
        self,
        max_pool_size: int = 20,
        hard_opponent_weight: float = 0.7,
        diversity_weight: float = 0.2,
        recency_weight: float = 0.1,
        min_games_for_priority: int = 5,
    ):
        """
        Args:
            max_pool_size: Maximum opponents to keep
            hard_opponent_weight: Weight for prioritizing hard opponents
            diversity_weight: Weight for opponent diversity
            recency_weight: Weight for recently played opponents
            min_games_for_priority: Min games before using win rate for priority
        """
        self.max_pool_size = max_pool_size
        self.hard_opponent_weight = hard_opponent_weight
        self.diversity_weight = diversity_weight
        self.recency_weight = recency_weight
        self.min_games_for_priority = min_games_for_priority

        self._opponents: dict[str, OpponentStats] = {}
        self._game_history: deque = deque(maxlen=1000)

    def add_opponent(
        self,
        model_path: str,
        elo: float = 1500.0,
        generation: int = 0,
        name: str | None = None,
    ) -> None:
        """Add an opponent to the pool."""
        if model_path in self._opponents:
            logger.debug(f"Opponent {model_path} already in pool")
            return

        # Evict oldest if at capacity
        if len(self._opponents) >= self.max_pool_size:
            self._evict_oldest()

        name = name or Path(model_path).stem
        self._opponents[model_path] = OpponentStats(
            model_path=model_path,
            model_name=name,
            elo=elo,
            generation=generation,
        )

        logger.info(f"Added opponent '{name}' to pool (elo={elo}, gen={generation})")

    def remove_opponent(self, model_path: str) -> None:
        """Remove an opponent from the pool."""
        if model_path in self._opponents:
            del self._opponents[model_path]
            logger.info(f"Removed opponent {model_path} from pool")

    def sample_opponent(
        self,
        current_elo: float = 1500.0,
        exclude: list[str] | None = None,
        strategy: str = "pfsp",
    ) -> OpponentStats | None:
        """
        Sample an opponent from the pool.

        Args:
            current_elo: Current model's Elo rating
            exclude: Paths to exclude from sampling
            strategy: Sampling strategy ("pfsp", "uniform", "elo_based")

        Returns:
            Selected opponent or None if pool is empty
        """
        candidates = [
            opp for path, opp in self._opponents.items()
            if exclude is None or path not in exclude
        ]

        if not candidates:
            return None

        if strategy == "uniform":
            return np.random.choice(candidates)

        elif strategy == "elo_based":
            # Prefer opponents near current Elo
            weights = []
            for opp in candidates:
                elo_diff = abs(opp.elo - current_elo)
                # Gaussian weighting around current Elo
                weight = math.exp(-(elo_diff ** 2) / (2 * 200 ** 2))
                weights.append(weight)

            weights = np.array(weights)
            weights /= weights.sum()
            return np.random.choice(candidates, p=weights)

        else:  # PFSP
            return self._pfsp_sample(candidates, current_elo)

    def _pfsp_sample(
        self,
        candidates: list[OpponentStats],
        current_elo: float,
    ) -> OpponentStats:
        """PFSP sampling: prioritize hard opponents."""
        scores = []

        for opp in candidates:
            # Hard opponent score: lower win rate = higher priority
            if opp.games_played >= self.min_games_for_priority:
                win_rate = opp.wins / opp.games_played if opp.games_played > 0 else 0.5
                # Prioritize opponents we lose to
                hard_score = 1.0 - win_rate
            else:
                # Unknown difficulty, moderate priority
                hard_score = 0.5

            # Diversity score: less played = higher priority
            max_games = max(o.games_played for o in candidates) or 1
            diversity_score = 1.0 - (opp.games_played / max_games)

            # Recency score: recently played = lower priority
            if opp.last_played:
                time_since = (datetime.now() - opp.last_played).total_seconds()
                # Decay over 1 hour
                recency_score = min(1.0, time_since / 3600)
            else:
                recency_score = 1.0

            # Combined score
            score = (
                self.hard_opponent_weight * hard_score +
                self.diversity_weight * diversity_score +
                self.recency_weight * recency_score
            )
            scores.append(score)

        # Sample proportionally to scores
        scores = np.array(scores)
        scores = np.maximum(scores, 0.01)  # Ensure all positive
        probs = scores / scores.sum()

        return np.random.choice(candidates, p=probs)

    def update_stats(
        self,
        model_path: str,
        won: bool,
        drew: bool = False,
        elo_change: float = 0.0,
    ) -> None:
        """Update opponent statistics after a game."""
        if model_path not in self._opponents:
            return

        opp = self._opponents[model_path]
        opp.games_played += 1
        opp.last_played = datetime.now()

        if drew:
            opp.draws += 1
        elif won:
            opp.wins += 1
        else:
            opp.losses += 1

        # Update Elo if provided
        if elo_change != 0:
            opp.elo += elo_change

        # Update priority score
        self._update_priority(opp)

        # Record history
        self._game_history.append({
            'opponent': model_path,
            'won': won,
            'drew': drew,
            'timestamp': datetime.now(),
        })

    def _update_priority(self, opp: OpponentStats) -> None:
        """Update priority score for an opponent."""
        if opp.games_played < self.min_games_for_priority:
            opp.priority_score = 1.0
            return

        # Priority based on win rate against this opponent
        win_rate = opp.wins / opp.games_played
        # Higher priority for lower win rates (harder opponents)
        opp.priority_score = 1.0 - win_rate

    def _evict_oldest(self) -> None:
        """Evict the oldest/least useful opponent."""
        if not self._opponents:
            return

        # Score opponents for eviction
        eviction_scores = {}
        for path, opp in self._opponents.items():
            # Prefer to evict:
            # - High win rate (easy opponents)
            # - Long since played
            # - Low generation (old models)

            win_rate = opp.wins / max(1, opp.games_played)
            age_score = opp.generation / max(o.generation for o in self._opponents.values()) if self._opponents else 0

            eviction_scores[path] = win_rate - 0.3 * age_score

        # Evict highest scoring (easiest/oldest)
        to_evict = max(eviction_scores, key=eviction_scores.get)
        logger.info(f"Evicting opponent {to_evict} from pool")
        del self._opponents[to_evict]

    def get_pool_stats(self) -> dict[str, Any]:
        """Get statistics about the opponent pool."""
        if not self._opponents:
            return {'size': 0}

        elos = [o.elo for o in self._opponents.values()]
        games = [o.games_played for o in self._opponents.values()]

        return {
            'size': len(self._opponents),
            'avg_elo': np.mean(elos),
            'min_elo': min(elos),
            'max_elo': max(elos),
            'total_games': sum(games),
            'avg_games_per_opponent': np.mean(games),
        }

    def get_opponents(self) -> list[OpponentStats]:
        """Get all opponents in the pool."""
        return list(self._opponents.values())

    def save_pool(self, path: str | Path) -> None:
        """Save pool state to file."""
        import json

        data = {
            'opponents': {
                p: {
                    'model_path': o.model_path,
                    'model_name': o.model_name,
                    'elo': o.elo,
                    'games_played': o.games_played,
                    'wins': o.wins,
                    'losses': o.losses,
                    'draws': o.draws,
                    'generation': o.generation,
                    'priority_score': o.priority_score,
                }
                for p, o in self._opponents.items()
            },
            'config': {
                'max_pool_size': self.max_pool_size,
                'hard_opponent_weight': self.hard_opponent_weight,
                'diversity_weight': self.diversity_weight,
                'recency_weight': self.recency_weight,
            },
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_pool(self, path: str | Path) -> None:
        """Load pool state from file."""
        import json

        with open(path) as f:
            data = json.load(f)

        for p, o in data['opponents'].items():
            self._opponents[p] = OpponentStats(**o)

        config = data.get('config', {})
        self.max_pool_size = config.get('max_pool_size', self.max_pool_size)
        self.hard_opponent_weight = config.get('hard_opponent_weight', self.hard_opponent_weight)


__all__ = [
    "OpponentStats",
    "PFSPOpponentPool",
]
