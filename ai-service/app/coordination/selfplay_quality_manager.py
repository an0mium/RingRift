"""Quality and diversity management for selfplay scheduling.

January 2026 Sprint 17.9: Extracted from selfplay_scheduler.py to reduce module size.

This module provides:
1. OpponentDiversityTracker - Tracks opponent types seen per config for diversity
2. QualityManager - Unified interface for quality scoring and diversity tracking

The goal is to maximize training robustness by ensuring configs play against diverse
opponents and have high-quality training data.

Usage:
    from app.coordination.selfplay_quality_manager import (
        QualityManager,
        get_quality_manager,
    )

    # Get singleton instance
    manager = get_quality_manager()

    # Record opponent for diversity tracking
    manager.record_opponent("hex8_2p", "gumbel")

    # Get quality score (cached with TTL)
    quality = manager.get_config_quality("hex8_2p")

    # Get diversity score (0.0 = low variety, 1.0 = high variety)
    diversity = manager.get_diversity_score("hex8_2p")
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from app.coordination.config_state_cache import ConfigStateCache, create_quality_cache

logger = logging.getLogger(__name__)

# Default opponent types for diversity calculation
# These are the known opponent types that can be used in selfplay
KNOWN_OPPONENT_TYPES = frozenset([
    "heuristic",
    "policy",
    "gumbel",
    "mcts",
    "minimax",
    "random",
    "nnue",
    "mixed",
])

# Maximum opponent types for diversity score normalization
DEFAULT_MAX_OPPONENT_TYPES = 8


@dataclass
class DiversityStats:
    """Statistics for diversity tracking.

    Attributes:
        config_key: Configuration key
        opponent_types_seen: Set of opponent types played against
        diversity_score: Normalized score (0.0-1.0)
        games_by_opponent: Count of games per opponent type (if tracked)
    """
    config_key: str
    opponent_types_seen: frozenset[str] = field(default_factory=frozenset)
    diversity_score: float = 0.0
    games_by_opponent: dict[str, int] = field(default_factory=dict)

    @property
    def opponent_count(self) -> int:
        """Number of distinct opponent types seen."""
        return len(self.opponent_types_seen)


class OpponentDiversityTracker:
    """Tracks opponent types seen per config for diversity maximization.

    January 2026 Sprint 10: Introduced to maximize training robustness.
    Configs that play against more diverse opponents get higher priority
    to ensure models learn to handle varied playing styles.

    Thread Safety:
        Uses a threading.Lock for thread-safe updates.

    Example:
        >>> tracker = OpponentDiversityTracker(max_opponent_types=8)
        >>> tracker.record_opponent("hex8_2p", "gumbel")
        >>> tracker.record_opponent("hex8_2p", "heuristic")
        >>> tracker.get_diversity_score("hex8_2p")
        0.25  # 2 types seen / 8 max types
    """

    def __init__(
        self,
        max_opponent_types: int = DEFAULT_MAX_OPPONENT_TYPES,
        track_game_counts: bool = False,
    ):
        """Initialize diversity tracker.

        Args:
            max_opponent_types: Maximum opponent types for score normalization
            track_game_counts: If True, also track game counts per opponent type
        """
        self._max_opponent_types = max_opponent_types
        self._track_game_counts = track_game_counts

        # Opponent types seen per config
        self._opponent_types_by_config: dict[str, set[str]] = {}

        # Cached diversity scores
        self._diversity_scores: dict[str, float] = {}

        # Optional: game counts per opponent type
        self._games_by_opponent: dict[str, dict[str, int]] = {}

        # Thread safety
        self._lock = threading.Lock()

        # Logging prefix
        self._log_prefix = "[DiversityTracker]"

    @property
    def max_opponent_types(self) -> int:
        """Maximum opponent types for normalization."""
        return self._max_opponent_types

    def record_opponent(
        self,
        config_key: str,
        opponent_type: str,
        game_count: int = 1,
    ) -> float:
        """Record that a config played against an opponent type.

        Args:
            config_key: Config like "hex8_2p"
            opponent_type: Type of opponent (e.g., "heuristic", "gumbel")
            game_count: Number of games played (for weighted tracking)

        Returns:
            Updated diversity score for the config
        """
        with self._lock:
            # Initialize if needed
            if config_key not in self._opponent_types_by_config:
                self._opponent_types_by_config[config_key] = set()

            # Record opponent type
            self._opponent_types_by_config[config_key].add(opponent_type)

            # Track game counts if enabled
            if self._track_game_counts:
                if config_key not in self._games_by_opponent:
                    self._games_by_opponent[config_key] = {}
                self._games_by_opponent[config_key][opponent_type] = (
                    self._games_by_opponent[config_key].get(opponent_type, 0) + game_count
                )

            # Recompute diversity score
            score = self._compute_diversity_score(config_key)
            self._diversity_scores[config_key] = score

            logger.debug(
                f"{self._log_prefix} {config_key} played vs {opponent_type}, "
                f"diversity: {score:.2f} ({len(self._opponent_types_by_config[config_key])}/{self._max_opponent_types})"
            )

            return score

    def _compute_diversity_score(self, config_key: str) -> float:
        """Compute diversity score for a config.

        Args:
            config_key: Config like "hex8_2p"

        Returns:
            Diversity score between 0.0 (low diversity) and 1.0 (high diversity)
        """
        opponents_seen = self._opponent_types_by_config.get(config_key, set())
        if not opponents_seen:
            return 0.0  # No opponents seen = low diversity
        return min(1.0, len(opponents_seen) / self._max_opponent_types)

    def get_diversity_score(self, config_key: str) -> float:
        """Get diversity score for a config.

        Args:
            config_key: Config like "hex8_2p"

        Returns:
            Diversity score between 0.0 (low diversity) and 1.0 (high diversity)
        """
        with self._lock:
            return self._diversity_scores.get(config_key, 0.0)

    def get_opponent_types_seen(self, config_key: str) -> int:
        """Get number of distinct opponent types played by a config.

        Args:
            config_key: Config like "hex8_2p"

        Returns:
            Number of distinct opponent types
        """
        with self._lock:
            return len(self._opponent_types_by_config.get(config_key, set()))

    def get_opponent_types(self, config_key: str) -> frozenset[str]:
        """Get the set of opponent types played by a config.

        Args:
            config_key: Config like "hex8_2p"

        Returns:
            Frozen set of opponent type names
        """
        with self._lock:
            return frozenset(self._opponent_types_by_config.get(config_key, set()))

    def get_stats(self, config_key: str) -> DiversityStats:
        """Get full diversity statistics for a config.

        Args:
            config_key: Config like "hex8_2p"

        Returns:
            DiversityStats with full tracking information
        """
        with self._lock:
            return DiversityStats(
                config_key=config_key,
                opponent_types_seen=frozenset(
                    self._opponent_types_by_config.get(config_key, set())
                ),
                diversity_score=self._diversity_scores.get(config_key, 0.0),
                games_by_opponent=dict(
                    self._games_by_opponent.get(config_key, {})
                ) if self._track_game_counts else {},
            )

    def get_all_diversity_scores(self) -> dict[str, float]:
        """Get diversity scores for all tracked configs.

        Returns:
            Dict mapping config_key to diversity score
        """
        with self._lock:
            return dict(self._diversity_scores)

    def get_low_diversity_configs(
        self,
        threshold: float = 0.5,
        config_keys: list[str] | None = None,
    ) -> list[str]:
        """Get configs with diversity score below threshold.

        Args:
            threshold: Diversity score threshold (default 0.5)
            config_keys: Optional list to filter by

        Returns:
            List of config keys with low diversity
        """
        with self._lock:
            if config_keys is None:
                config_keys = list(self._diversity_scores.keys())

            return [
                config for config in config_keys
                if self._diversity_scores.get(config, 0.0) < threshold
            ]

    def reset(self, config_key: str | None = None) -> int:
        """Reset diversity tracking.

        Args:
            config_key: Specific config to reset, or None to reset all

        Returns:
            Number of configs reset
        """
        with self._lock:
            if config_key is None:
                count = len(self._opponent_types_by_config)
                self._opponent_types_by_config.clear()
                self._diversity_scores.clear()
                self._games_by_opponent.clear()
                logger.debug(f"{self._log_prefix} Reset all diversity tracking ({count} configs)")
                return count
            elif config_key in self._opponent_types_by_config:
                del self._opponent_types_by_config[config_key]
                self._diversity_scores.pop(config_key, None)
                self._games_by_opponent.pop(config_key, None)
                logger.debug(f"{self._log_prefix} Reset diversity tracking for {config_key}")
                return 1
            return 0

    def get_status(self) -> dict[str, Any]:
        """Get tracker status for health checks.

        Returns:
            Dict with tracking statistics
        """
        with self._lock:
            return {
                "max_opponent_types": self._max_opponent_types,
                "configs_tracked": len(self._opponent_types_by_config),
                "track_game_counts": self._track_game_counts,
                "diversity_scores": dict(self._diversity_scores),
            }


class QualityManager:
    """Unified quality and diversity management for selfplay scheduling.

    Combines ConfigStateCache (quality scoring) with OpponentDiversityTracker
    (diversity tracking) to provide a single interface for the scheduler.

    Thread Safety:
        Uses internal locks for thread-safe operations.

    Example:
        >>> manager = QualityManager()
        >>> manager.record_opponent("hex8_2p", "gumbel")
        >>> quality = manager.get_config_quality("hex8_2p")
        >>> diversity = manager.get_diversity_score("hex8_2p")
    """

    def __init__(
        self,
        quality_cache_ttl: float = 30.0,
        max_opponent_types: int = DEFAULT_MAX_OPPONENT_TYPES,
        quality_provider: Optional[Callable[[str], Optional[float]]] = None,
    ):
        """Initialize quality manager.

        Args:
            quality_cache_ttl: TTL for quality score cache (seconds)
            max_opponent_types: Max opponent types for diversity normalization
            quality_provider: Optional callback to fetch quality on cache miss
        """
        # Quality caching
        self._quality_cache = ConfigStateCache(
            ttl_seconds=quality_cache_ttl,
            quality_provider=quality_provider,
        )

        # Diversity tracking
        self._diversity_tracker = OpponentDiversityTracker(
            max_opponent_types=max_opponent_types,
            track_game_counts=True,
        )

        # Logging prefix
        self._log_prefix = "[QualityManager]"

    @property
    def quality_cache(self) -> ConfigStateCache:
        """Access underlying quality cache."""
        return self._quality_cache

    @property
    def diversity_tracker(self) -> OpponentDiversityTracker:
        """Access underlying diversity tracker."""
        return self._diversity_tracker

    # =========================================================================
    # Quality Methods (delegated to ConfigStateCache)
    # =========================================================================

    def get_config_quality(self, config_key: str) -> float:
        """Get data quality score for a config.

        Higher quality score = better training data (Gumbel MCTS, passed parity).
        Lower quality = heuristic-only games, parity failures.

        Args:
            config_key: Config key like "hex8_2p"

        Returns:
            Quality score 0.0-1.0 (default 0.7 if unavailable)
        """
        return self._quality_cache.get_quality_or_fetch(config_key)

    def get_all_config_qualities(self, config_keys: list[str]) -> dict[str, float]:
        """Get data quality scores for multiple configs.

        Args:
            config_keys: List of config keys to look up

        Returns:
            Dict mapping config_key to quality score (0.0-1.0)
        """
        return self._quality_cache.get_all_qualities(config_keys)

    def invalidate_quality_cache(self, config_key: str | None = None) -> int:
        """Invalidate quality cache for a config or all configs.

        Call this when quality data changes externally.

        Args:
            config_key: Specific config to invalidate, or None to clear all

        Returns:
            Number of entries invalidated
        """
        return self._quality_cache.invalidate(config_key)

    def set_quality(self, config_key: str, quality: float) -> None:
        """Manually set quality score in cache.

        Args:
            config_key: Config key like "hex8_2p"
            quality: Quality score (0.0-1.0)
        """
        self._quality_cache.set_quality(config_key, quality)

    # =========================================================================
    # Diversity Methods (delegated to OpponentDiversityTracker)
    # =========================================================================

    def record_opponent(
        self,
        config_key: str,
        opponent_type: str,
        game_count: int = 1,
    ) -> float:
        """Record that a config played against an opponent type.

        Args:
            config_key: Config like "hex8_2p"
            opponent_type: Type of opponent (e.g., "heuristic", "gumbel")
            game_count: Number of games played

        Returns:
            Updated diversity score for the config
        """
        return self._diversity_tracker.record_opponent(
            config_key, opponent_type, game_count
        )

    def get_diversity_score(self, config_key: str) -> float:
        """Get diversity score for a config.

        Args:
            config_key: Config like "hex8_2p"

        Returns:
            Diversity score between 0.0 (low) and 1.0 (high)
        """
        return self._diversity_tracker.get_diversity_score(config_key)

    def get_opponent_types_seen(self, config_key: str) -> int:
        """Get number of distinct opponent types played by a config.

        Args:
            config_key: Config like "hex8_2p"

        Returns:
            Number of distinct opponent types
        """
        return self._diversity_tracker.get_opponent_types_seen(config_key)

    def get_diversity_stats(self, config_key: str) -> DiversityStats:
        """Get full diversity statistics for a config.

        Args:
            config_key: Config like "hex8_2p"

        Returns:
            DiversityStats with full tracking information
        """
        return self._diversity_tracker.get_stats(config_key)

    # =========================================================================
    # Combined Methods
    # =========================================================================

    def get_config_scores(self, config_key: str) -> tuple[float, float]:
        """Get both quality and diversity scores for a config.

        Args:
            config_key: Config like "hex8_2p"

        Returns:
            Tuple of (quality_score, diversity_score)
        """
        return (
            self.get_config_quality(config_key),
            self.get_diversity_score(config_key),
        )

    def get_underserved_configs(
        self,
        config_keys: list[str],
        quality_threshold: float = 0.6,
        diversity_threshold: float = 0.5,
    ) -> list[str]:
        """Get configs that need attention (low quality or low diversity).

        Args:
            config_keys: List of config keys to check
            quality_threshold: Quality score threshold
            diversity_threshold: Diversity score threshold

        Returns:
            List of config keys that are underserved
        """
        underserved = []
        for config in config_keys:
            quality = self.get_config_quality(config)
            diversity = self.get_diversity_score(config)
            if quality < quality_threshold or diversity < diversity_threshold:
                underserved.append(config)
        return underserved

    def reset(self, config_key: str | None = None) -> int:
        """Reset all tracking for a config or all configs.

        Args:
            config_key: Specific config to reset, or None to reset all

        Returns:
            Number of configs reset
        """
        quality_count = self._quality_cache.invalidate(config_key)
        diversity_count = self._diversity_tracker.reset(config_key)
        return max(quality_count, diversity_count)

    def get_status(self) -> dict[str, Any]:
        """Get manager status for health checks.

        Returns:
            Dict with quality and diversity statistics
        """
        return {
            "quality_cache": self._quality_cache.get_status(),
            "diversity_tracker": self._diversity_tracker.get_status(),
        }


# =============================================================================
# Singleton Management
# =============================================================================

_quality_manager_instance: QualityManager | None = None
_quality_manager_lock = threading.Lock()


def get_quality_manager() -> QualityManager:
    """Get the singleton QualityManager instance.

    Creates the manager with default configuration if not exists.

    Returns:
        QualityManager singleton instance
    """
    global _quality_manager_instance

    if _quality_manager_instance is None:
        with _quality_manager_lock:
            if _quality_manager_instance is None:
                # Create with QualityMonitorDaemon integration
                def quality_provider(config_key: str) -> Optional[float]:
                    """Fetch quality from QualityMonitorDaemon."""
                    try:
                        from app.coordination.quality_monitor_daemon import get_quality_daemon
                        daemon = get_quality_daemon()
                        if daemon:
                            return daemon.get_config_quality(config_key)
                    except ImportError:
                        logger.debug("[QualityManager] quality_monitor_daemon not available")
                    except (AttributeError, KeyError) as e:
                        logger.debug(f"[QualityManager] Error fetching quality: {e}")
                    return None

                _quality_manager_instance = QualityManager(
                    quality_cache_ttl=30.0,
                    max_opponent_types=DEFAULT_MAX_OPPONENT_TYPES,
                    quality_provider=quality_provider,
                )
                logger.debug("[QualityManager] Created singleton instance")

    return _quality_manager_instance


def reset_quality_manager() -> None:
    """Reset the QualityManager singleton (for testing)."""
    global _quality_manager_instance
    with _quality_manager_lock:
        _quality_manager_instance = None
        logger.debug("[QualityManager] Reset singleton instance")
