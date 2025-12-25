"""Elo rating calculation utilities for tournament system.

This module provides pure Elo calculation functions and dataclasses.
Supports both 2-player and multiplayer games. For multiplayer, games are
decomposed into virtual pairwise matchups based on final rankings.

Note: For service-level Elo operations (persistence, cross-component coordination),
use `app.training.elo_service` which provides:
- Singleton EloService with SQLite persistence
- Thread-safe operations with unified_elo.db
- Training feedback hooks for parameter adaptation
- Integration with model lifecycle management

This module (`app.tournament.elo`) focuses on:
- Core Elo calculation algorithms
- Glicko-style confidence intervals
- Multiplayer Elo decomposition
- Tournament rating utilities

See: app.config.thresholds for canonical Elo constants.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

# Import canonical Elo constants
try:
    from app.config.thresholds import INITIAL_ELO_RATING
except ImportError:
    INITIAL_ELO_RATING = 1500.0


@dataclass
class EloVelocity:
    """Elo improvement velocity over a time period.

    This dataclass tracks the rate of Elo change, useful for:
    - Detecting rapid improvement (increase training intensity)
    - Detecting plateaus (trigger hyperparameter search)
    - Monitoring training health

    Part of the strength-driven training system (December 2025).
    """

    elo_per_hour: float  # Elo change per hour
    elo_per_game: float  # Elo change per game
    trend: str  # "improving", "stable", "declining"
    lookback_hours: float  # Hours considered for calculation
    games_in_period: int  # Games played in the lookback period
    elo_start: float  # Elo at start of period
    elo_end: float  # Elo at end of period

    @property
    def is_improving(self) -> bool:
        """Check if model is actively improving."""
        return self.trend == "improving"

    @property
    def is_plateau(self) -> bool:
        """Check if model has plateaued (stable or declining)."""
        return self.trend in ("stable", "declining")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "elo_per_hour": round(self.elo_per_hour, 2),
            "elo_per_game": round(self.elo_per_game, 3),
            "trend": self.trend,
            "lookback_hours": self.lookback_hours,
            "games_in_period": self.games_in_period,
            "elo_start": round(self.elo_start, 1),
            "elo_end": round(self.elo_end, 1),
        }


@dataclass
class EloRating:
    """Elo rating for an agent.

    Includes confidence interval estimation based on Glicko-style rating deviation.
    The rating deviation (RD) starts high for new players and decreases as more
    games are played, reflecting increased confidence in the rating.

    Note: Uses INITIAL_ELO_RATING from app.config.thresholds as default.
    """

    agent_id: str
    rating: float = INITIAL_ELO_RATING
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    rating_history: list[tuple[datetime, float]] = field(default_factory=list)

    # Glicko-style rating deviation parameters
    # Initial RD for new players (high uncertainty)
    INITIAL_RD: float = 350.0
    # Minimum RD after many games (baseline uncertainty)
    MIN_RD: float = 50.0
    # Games needed to reach near-minimum RD
    RD_DECAY_GAMES: int = 100

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played

    @property
    def expected_score(self) -> float:
        """Expected score based on games played."""
        if self.games_played == 0:
            return 0.5
        return (self.wins + 0.5 * self.draws) / self.games_played

    @property
    def rating_deviation(self) -> float:
        """Estimate rating deviation (uncertainty) based on games played.

        Uses exponential decay from INITIAL_RD toward MIN_RD as games increase.
        This approximates Glicko's RD without the time-based component.

        Returns:
            Rating deviation (standard deviation of rating estimate).
        """
        if self.games_played == 0:
            return self.INITIAL_RD

        # Exponential decay: RD = MIN_RD + (INITIAL_RD - MIN_RD) * e^(-games/decay)
        decay_factor = math.exp(-self.games_played / self.RD_DECAY_GAMES)
        return self.MIN_RD + (self.INITIAL_RD - self.MIN_RD) * decay_factor

    def confidence_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        """Calculate confidence interval for the rating.

        Args:
            confidence: Confidence level (default 0.95 for 95% CI).

        Returns:
            Tuple of (lower_bound, upper_bound) for the rating.

        Example:
            >>> rating = EloRating("agent_1", rating=1600, games_played=50)
            >>> lower, upper = rating.confidence_interval(0.95)
            >>> print(f"Rating: {rating.rating} [{lower:.0f}, {upper:.0f}]")
            Rating: 1600 [1412, 1788]
        """
        from statistics import NormalDist

        # Calculate z-score for the confidence level
        if confidence <= 0.0 or confidence >= 1.0:
            confidence = 0.95
        p = 0.5 + confidence / 2.0
        z = NormalDist().inv_cdf(p)

        rd = self.rating_deviation
        margin = z * rd

        return (round(self.rating - margin, 1), round(self.rating + margin, 1))

    @property
    def ci_95(self) -> tuple[float, float]:
        """Convenience property for 95% confidence interval."""
        return self.confidence_interval(0.95)

    @property
    def uncertainty_str(self) -> str:
        """Human-readable rating with uncertainty.

        Returns:
            String like "1600 ± 94" showing rating and 95% CI half-width.
        """
        rd = self.rating_deviation
        # 95% CI uses z ≈ 1.96
        margin = round(1.96 * rd)
        return f"{round(self.rating)} ± {margin}"

    def to_dict(self) -> dict:
        ci_lower, ci_upper = self.ci_95
        return {
            "agent_id": self.agent_id,
            "rating": self.rating,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": round(self.win_rate, 3),
            "rating_deviation": round(self.rating_deviation, 1),
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
        }

    def compute_velocity(
        self,
        lookback_hours: float = 24.0,
        improving_threshold_per_hour: float = 10.0,
        declining_threshold_per_hour: float = -5.0,
    ) -> EloVelocity:
        """Compute Elo improvement velocity over a lookback period.

        This method analyzes the rating history to determine the rate of
        Elo change, useful for:
        - Detecting rapid improvement (increase training intensity)
        - Detecting plateaus (trigger hyperparameter search)
        - Monitoring training health

        Part of the strength-driven training system (December 2025).

        Args:
            lookback_hours: Hours to look back for velocity calculation.
            improving_threshold_per_hour: Elo/hour above which trend is "improving".
            declining_threshold_per_hour: Elo/hour below which trend is "declining".

        Returns:
            EloVelocity with rate metrics and trend classification.

        Example:
            >>> rating = EloRating("model_v1", rating=1600, games_played=100)
            >>> # After adding some history entries...
            >>> velocity = rating.compute_velocity(lookback_hours=24)
            >>> print(f"Elo/hour: {velocity.elo_per_hour:.1f}, trend: {velocity.trend}")
            Elo/hour: 15.2, trend: improving
        """
        from datetime import timedelta

        if len(self.rating_history) < 2:
            return EloVelocity(
                elo_per_hour=0.0,
                elo_per_game=0.0,
                trend="stable",
                lookback_hours=lookback_hours,
                games_in_period=len(self.rating_history),
                elo_start=self.rating,
                elo_end=self.rating,
            )

        now = datetime.now()
        cutoff = now - timedelta(hours=lookback_hours)

        # Filter history to lookback period
        recent_history = [
            (ts, elo) for ts, elo in self.rating_history
            if ts >= cutoff
        ]

        if len(recent_history) < 2:
            # Not enough data in lookback period, use all history
            recent_history = self.rating_history

        if len(recent_history) < 2:
            return EloVelocity(
                elo_per_hour=0.0,
                elo_per_game=0.0,
                trend="stable",
                lookback_hours=lookback_hours,
                games_in_period=len(recent_history),
                elo_start=self.rating,
                elo_end=self.rating,
            )

        first_ts, first_elo = recent_history[0]
        last_ts, last_elo = recent_history[-1]

        elo_delta = last_elo - first_elo
        time_delta_seconds = (last_ts - first_ts).total_seconds()
        time_delta_hours = max(time_delta_seconds / 3600, 0.001)  # Avoid division by zero
        games_delta = len(recent_history) - 1  # Number of games between first and last

        elo_per_hour = elo_delta / time_delta_hours
        elo_per_game = elo_delta / max(games_delta, 1)

        # Classify trend
        if elo_per_hour > improving_threshold_per_hour:
            trend = "improving"
        elif elo_per_hour < declining_threshold_per_hour:
            trend = "declining"
        else:
            trend = "stable"

        return EloVelocity(
            elo_per_hour=elo_per_hour,
            elo_per_game=elo_per_game,
            trend=trend,
            lookback_hours=lookback_hours,
            games_in_period=len(recent_history),
            elo_start=first_elo,
            elo_end=last_elo,
        )


class EloCalculator:
    """Standard Elo rating calculator with configurable K-factor."""

    def __init__(
        self,
        initial_rating: float = 1500.0,
        k_factor: float = 32.0,
        k_factor_high_rated: float = 16.0,
        high_rated_threshold: float = 2400.0,
        k_factor_provisional: float = 40.0,
        provisional_games: int = 30,
    ):
        """Initialize Elo calculator.

        Args:
            initial_rating: Starting rating for new players.
            k_factor: K-factor for normal players.
            k_factor_high_rated: K-factor for high-rated players.
            high_rated_threshold: Rating threshold for high-rated K-factor.
            k_factor_provisional: K-factor for provisional (new) players.
            provisional_games: Number of games before player is established.
        """
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.k_factor_high_rated = k_factor_high_rated
        self.high_rated_threshold = high_rated_threshold
        self.k_factor_provisional = k_factor_provisional
        self.provisional_games = provisional_games

        self._ratings: dict[str, EloRating] = {}

    def get_rating(self, agent_id: str) -> EloRating:
        """Get or create rating for an agent."""
        if agent_id not in self._ratings:
            self._ratings[agent_id] = EloRating(
                agent_id=agent_id,
                rating=self.initial_rating,
            )
        return self._ratings[agent_id]

    def get_k_factor(self, rating: EloRating) -> float:
        """Get K-factor for a player based on rating and games played."""
        if rating.games_played < self.provisional_games:
            return self.k_factor_provisional
        elif rating.rating >= self.high_rated_threshold:
            return self.k_factor_high_rated
        return self.k_factor

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A vs player B.

        Returns probability of player A winning (0.0 to 1.0).
        """
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))

    def update_ratings(
        self,
        agent_a_id: str,
        agent_b_id: str,
        result: float,
        timestamp: datetime | None = None,
    ) -> tuple[float, float]:
        """Update ratings after a match.

        Args:
            agent_a_id: First player's agent ID.
            agent_b_id: Second player's agent ID.
            result: Result from A's perspective (1.0=win, 0.5=draw, 0.0=loss).
            timestamp: Optional timestamp for rating history.

        Returns:
            Tuple of (new_rating_a, new_rating_b).
        """
        timestamp = timestamp or datetime.now()

        rating_a = self.get_rating(agent_a_id)
        rating_b = self.get_rating(agent_b_id)

        expected_a = self.expected_score(rating_a.rating, rating_b.rating)
        expected_b = 1.0 - expected_a

        k_a = self.get_k_factor(rating_a)
        k_b = self.get_k_factor(rating_b)

        result_b = 1.0 - result

        new_rating_a = rating_a.rating + k_a * (result - expected_a)
        new_rating_b = rating_b.rating + k_b * (result_b - expected_b)

        rating_a.rating = new_rating_a
        rating_b.rating = new_rating_b

        rating_a.games_played += 1
        rating_b.games_played += 1

        if result == 1.0:
            rating_a.wins += 1
            rating_b.losses += 1
        elif result == 0.0:
            rating_a.losses += 1
            rating_b.wins += 1
        else:
            rating_a.draws += 1
            rating_b.draws += 1

        rating_a.rating_history.append((timestamp, new_rating_a))
        rating_b.rating_history.append((timestamp, new_rating_b))

        return new_rating_a, new_rating_b

    def get_leaderboard(self) -> list[EloRating]:
        """Get ratings sorted by rating (descending)."""
        return sorted(
            self._ratings.values(),
            key=lambda r: r.rating,
            reverse=True,
        )

    def get_all_ratings(self) -> dict[str, EloRating]:
        """Get all ratings."""
        return self._ratings.copy()

    def reset(self) -> None:
        """Reset all ratings."""
        self._ratings.clear()

    def update_multiplayer_ratings(
        self,
        rankings: Sequence[str],
        timestamp: datetime | None = None,
    ) -> dict[str, float]:
        """Update ratings after a multiplayer game based on final rankings.

        Decomposes the multiplayer result into virtual pairwise matchups.
        If player A finishes ahead of player B, A "beat" B in a virtual head-to-head.

        Args:
            rankings: Ordered list of agent IDs by finish position.
                      rankings[0] = 1st place, rankings[1] = 2nd place, etc.
            timestamp: Optional timestamp for rating history.

        Returns:
            Dict mapping agent_id to new rating.

        Example:
            # Player A wins, B second, C third, D fourth
            update_multiplayer_ratings(["agent_a", "agent_b", "agent_c", "agent_d"])
            # Results in virtual matchups:
            # A beats B, A beats C, A beats D
            # B beats C, B beats D
            # C beats D
        """
        if len(rankings) < 2:
            raise ValueError("Need at least 2 players for multiplayer rating update")

        timestamp = timestamp or datetime.now()
        n_players = len(rankings)

        # Get all ratings and calculate rating changes
        ratings = {agent_id: self.get_rating(agent_id) for agent_id in rankings}
        rating_deltas: dict[str, float] = dict.fromkeys(rankings, 0.0)

        # Process all pairwise matchups
        # For each pair (i, j) where i < j: player at rank i beat player at rank j
        for i in range(n_players):
            for j in range(i + 1, n_players):
                winner_id = rankings[i]  # Higher rank (lower index) = winner
                loser_id = rankings[j]

                winner_rating = ratings[winner_id]
                loser_rating = ratings[loser_id]

                # Calculate expected scores
                expected_winner = self.expected_score(
                    winner_rating.rating, loser_rating.rating
                )
                expected_loser = 1.0 - expected_winner

                # Get K-factors (scaled down by number of opponents for stability)
                k_winner = self.get_k_factor(winner_rating) / (n_players - 1)
                k_loser = self.get_k_factor(loser_rating) / (n_players - 1)

                # Winner gets result=1.0, loser gets result=0.0
                rating_deltas[winner_id] += k_winner * (1.0 - expected_winner)
                rating_deltas[loser_id] += k_loser * (0.0 - expected_loser)

        # Apply all rating changes and update stats
        new_ratings = {}
        for idx, agent_id in enumerate(rankings):
            rating = ratings[agent_id]
            rating.rating += rating_deltas[agent_id]
            rating.games_played += 1
            rating.rating_history.append((timestamp, rating.rating))

            # Update win/loss based on position
            if idx == 0:
                rating.wins += 1  # 1st place = win
            elif idx == n_players - 1:
                rating.losses += 1  # Last place = loss
            # Middle positions are neither win nor loss (could track separately)

            new_ratings[agent_id] = rating.rating

        return new_ratings

    def update_multiplayer_with_ties(
        self,
        rankings: Sequence[tuple[str, int]],
        timestamp: datetime | None = None,
    ) -> dict[str, float]:
        """Update ratings for multiplayer with potential ties.

        Args:
            rankings: List of (agent_id, rank) tuples. Rank 1 = first place.
                      Multiple agents can share the same rank (ties).
            timestamp: Optional timestamp for rating history.

        Returns:
            Dict mapping agent_id to new rating.

        Example:
            # A and B tie for 1st, C gets 3rd
            update_multiplayer_with_ties([
                ("agent_a", 1), ("agent_b", 1), ("agent_c", 3)
            ])
        """
        if len(rankings) < 2:
            raise ValueError("Need at least 2 players")

        timestamp = timestamp or datetime.now()
        n_players = len(rankings)

        # Get all ratings and calculate rating changes
        agents = [r[0] for r in rankings]
        ranks = {r[0]: r[1] for r in rankings}
        ratings = {agent_id: self.get_rating(agent_id) for agent_id in agents}
        rating_deltas: dict[str, float] = dict.fromkeys(agents, 0.0)

        # Process all pairwise matchups
        for i in range(n_players):
            for j in range(i + 1, n_players):
                agent_i, agent_j = agents[i], agents[j]
                rank_i, rank_j = ranks[agent_i], ranks[agent_j]

                rating_i = ratings[agent_i]
                rating_j = ratings[agent_j]

                expected_i = self.expected_score(rating_i.rating, rating_j.rating)
                expected_j = 1.0 - expected_i

                k_i = self.get_k_factor(rating_i) / (n_players - 1)
                k_j = self.get_k_factor(rating_j) / (n_players - 1)

                # Determine actual result based on ranks
                if rank_i < rank_j:
                    # i finished higher (better rank = lower number)
                    result_i, result_j = 1.0, 0.0
                elif rank_i > rank_j:
                    # j finished higher
                    result_i, result_j = 0.0, 1.0
                else:
                    # Tie
                    result_i, result_j = 0.5, 0.5

                rating_deltas[agent_i] += k_i * (result_i - expected_i)
                rating_deltas[agent_j] += k_j * (result_j - expected_j)

        # Apply all rating changes and update stats
        new_ratings = {}
        min_rank = min(ranks.values())
        max_rank = max(ranks.values())

        for agent_id in agents:
            rating = ratings[agent_id]
            rating.rating += rating_deltas[agent_id]
            rating.games_played += 1
            rating.rating_history.append((timestamp, rating.rating))

            rank = ranks[agent_id]
            if rank == min_rank:
                rating.wins += 1
            elif rank == max_rank:
                rating.losses += 1
            # Ties in non-extreme positions don't count as win/loss

            new_ratings[agent_id] = rating.rating

        return new_ratings

    def to_dict(self) -> dict:
        """Serialize calculator state."""
        return {
            "ratings": {
                agent_id: rating.to_dict()
                for agent_id, rating in self._ratings.items()
            },
            "config": {
                "initial_rating": self.initial_rating,
                "k_factor": self.k_factor,
                "k_factor_high_rated": self.k_factor_high_rated,
                "high_rated_threshold": self.high_rated_threshold,
                "k_factor_provisional": self.k_factor_provisional,
                "provisional_games": self.provisional_games,
            },
        }
