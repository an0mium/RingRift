"""Tests for selfplay_quality_manager module.

January 2026 Sprint 17.9: Tests for quality and diversity management.
"""

from __future__ import annotations

import pytest
import threading
import time

from app.coordination.selfplay_quality_manager import (
    OpponentDiversityTracker,
    QualityManager,
    DiversityStats,
    get_quality_manager,
    reset_quality_manager,
    KNOWN_OPPONENT_TYPES,
    DEFAULT_MAX_OPPONENT_TYPES,
)


class TestOpponentDiversityTracker:
    """Tests for OpponentDiversityTracker."""

    def test_init_defaults(self):
        """Test tracker initialization with defaults."""
        tracker = OpponentDiversityTracker()
        assert tracker.max_opponent_types == DEFAULT_MAX_OPPONENT_TYPES
        assert tracker.get_diversity_score("hex8_2p") == 0.0
        assert tracker.get_opponent_types_seen("hex8_2p") == 0

    def test_init_custom_max(self):
        """Test tracker with custom max opponent types."""
        tracker = OpponentDiversityTracker(max_opponent_types=4)
        assert tracker.max_opponent_types == 4

    def test_record_opponent_single(self):
        """Test recording a single opponent."""
        tracker = OpponentDiversityTracker(max_opponent_types=8)

        score = tracker.record_opponent("hex8_2p", "gumbel")

        assert score == 0.125  # 1/8
        assert tracker.get_opponent_types_seen("hex8_2p") == 1
        assert tracker.get_diversity_score("hex8_2p") == 0.125

    def test_record_opponent_multiple_unique(self):
        """Test recording multiple unique opponents."""
        tracker = OpponentDiversityTracker(max_opponent_types=8)

        tracker.record_opponent("hex8_2p", "gumbel")
        tracker.record_opponent("hex8_2p", "heuristic")
        tracker.record_opponent("hex8_2p", "policy")
        score = tracker.record_opponent("hex8_2p", "mcts")

        assert score == 0.5  # 4/8
        assert tracker.get_opponent_types_seen("hex8_2p") == 4

    def test_record_opponent_duplicate(self):
        """Test recording same opponent twice doesn't increase diversity."""
        tracker = OpponentDiversityTracker(max_opponent_types=8)

        tracker.record_opponent("hex8_2p", "gumbel")
        score = tracker.record_opponent("hex8_2p", "gumbel")

        assert score == 0.125  # Still 1/8
        assert tracker.get_opponent_types_seen("hex8_2p") == 1

    def test_record_opponent_max_diversity(self):
        """Test diversity score caps at 1.0."""
        tracker = OpponentDiversityTracker(max_opponent_types=4)

        for opponent in ["gumbel", "heuristic", "policy", "mcts", "random", "nnue"]:
            tracker.record_opponent("hex8_2p", opponent)

        assert tracker.get_diversity_score("hex8_2p") == 1.0

    def test_multiple_configs_independent(self):
        """Test that different configs have independent tracking."""
        tracker = OpponentDiversityTracker(max_opponent_types=8)

        tracker.record_opponent("hex8_2p", "gumbel")
        tracker.record_opponent("hex8_2p", "heuristic")
        tracker.record_opponent("square8_2p", "gumbel")

        assert tracker.get_opponent_types_seen("hex8_2p") == 2
        assert tracker.get_opponent_types_seen("square8_2p") == 1
        assert tracker.get_diversity_score("hex8_2p") == 0.25  # 2/8
        assert tracker.get_diversity_score("square8_2p") == 0.125  # 1/8

    def test_get_opponent_types(self):
        """Test getting the set of opponent types."""
        tracker = OpponentDiversityTracker()

        tracker.record_opponent("hex8_2p", "gumbel")
        tracker.record_opponent("hex8_2p", "heuristic")

        types = tracker.get_opponent_types("hex8_2p")

        assert isinstance(types, frozenset)
        assert types == frozenset(["gumbel", "heuristic"])

    def test_get_opponent_types_unknown_config(self):
        """Test getting types for unknown config returns empty set."""
        tracker = OpponentDiversityTracker()

        types = tracker.get_opponent_types("unknown_config")

        assert types == frozenset()

    def test_get_stats(self):
        """Test getting full diversity stats."""
        tracker = OpponentDiversityTracker(track_game_counts=True)

        tracker.record_opponent("hex8_2p", "gumbel", game_count=5)
        tracker.record_opponent("hex8_2p", "heuristic", game_count=3)

        stats = tracker.get_stats("hex8_2p")

        assert isinstance(stats, DiversityStats)
        assert stats.config_key == "hex8_2p"
        assert stats.opponent_count == 2
        assert stats.diversity_score == 0.25  # 2/8
        assert stats.games_by_opponent == {"gumbel": 5, "heuristic": 3}

    def test_get_stats_no_game_counts(self):
        """Test stats without game count tracking."""
        tracker = OpponentDiversityTracker(track_game_counts=False)

        tracker.record_opponent("hex8_2p", "gumbel", game_count=5)

        stats = tracker.get_stats("hex8_2p")

        assert stats.games_by_opponent == {}

    def test_get_all_diversity_scores(self):
        """Test getting all diversity scores."""
        tracker = OpponentDiversityTracker()

        tracker.record_opponent("hex8_2p", "gumbel")
        tracker.record_opponent("square8_2p", "gumbel")
        tracker.record_opponent("square8_2p", "heuristic")

        scores = tracker.get_all_diversity_scores()

        assert len(scores) == 2
        assert scores["hex8_2p"] == 0.125
        assert scores["square8_2p"] == 0.25

    def test_get_low_diversity_configs(self):
        """Test finding configs with low diversity."""
        tracker = OpponentDiversityTracker(max_opponent_types=4)

        # hex8_2p: 1/4 = 0.25 (below threshold)
        tracker.record_opponent("hex8_2p", "gumbel")

        # square8_2p: 3/4 = 0.75 (above threshold)
        tracker.record_opponent("square8_2p", "gumbel")
        tracker.record_opponent("square8_2p", "heuristic")
        tracker.record_opponent("square8_2p", "policy")

        low_div = tracker.get_low_diversity_configs(threshold=0.5)

        assert "hex8_2p" in low_div
        assert "square8_2p" not in low_div

    def test_get_low_diversity_configs_with_filter(self):
        """Test filtering low diversity by config list."""
        tracker = OpponentDiversityTracker()

        tracker.record_opponent("hex8_2p", "gumbel")
        tracker.record_opponent("square8_2p", "gumbel")

        # Only check hex8_2p
        low_div = tracker.get_low_diversity_configs(
            threshold=0.5,
            config_keys=["hex8_2p"]
        )

        assert low_div == ["hex8_2p"]

    def test_reset_single_config(self):
        """Test resetting a single config."""
        tracker = OpponentDiversityTracker()

        tracker.record_opponent("hex8_2p", "gumbel")
        tracker.record_opponent("square8_2p", "gumbel")

        count = tracker.reset("hex8_2p")

        assert count == 1
        assert tracker.get_diversity_score("hex8_2p") == 0.0
        assert tracker.get_diversity_score("square8_2p") == 0.125

    def test_reset_all(self):
        """Test resetting all configs."""
        tracker = OpponentDiversityTracker()

        tracker.record_opponent("hex8_2p", "gumbel")
        tracker.record_opponent("square8_2p", "gumbel")

        count = tracker.reset()

        assert count == 2
        assert tracker.get_diversity_score("hex8_2p") == 0.0
        assert tracker.get_diversity_score("square8_2p") == 0.0

    def test_reset_unknown_config(self):
        """Test resetting unknown config returns 0."""
        tracker = OpponentDiversityTracker()

        count = tracker.reset("unknown")

        assert count == 0

    def test_get_status(self):
        """Test getting tracker status."""
        tracker = OpponentDiversityTracker(max_opponent_types=8, track_game_counts=True)

        tracker.record_opponent("hex8_2p", "gumbel")

        status = tracker.get_status()

        assert status["max_opponent_types"] == 8
        assert status["configs_tracked"] == 1
        assert status["track_game_counts"] is True
        assert "hex8_2p" in status["diversity_scores"]

    def test_thread_safety(self):
        """Test thread safety of record_opponent."""
        tracker = OpponentDiversityTracker()
        opponents = ["gumbel", "heuristic", "policy", "mcts", "random"]
        errors = []

        def record_opponents():
            try:
                for _ in range(100):
                    for opponent in opponents:
                        tracker.record_opponent("hex8_2p", opponent)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_opponents) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert tracker.get_opponent_types_seen("hex8_2p") == 5


class TestQualityManager:
    """Tests for QualityManager."""

    def test_init_defaults(self):
        """Test manager initialization with defaults."""
        manager = QualityManager()

        assert manager.quality_cache is not None
        assert manager.diversity_tracker is not None

    def test_init_custom_ttl(self):
        """Test manager with custom TTL."""
        manager = QualityManager(quality_cache_ttl=60.0)

        assert manager.quality_cache.ttl_seconds == 60.0

    def test_get_config_quality_default(self):
        """Test quality returns default when no provider."""
        manager = QualityManager()

        quality = manager.get_config_quality("hex8_2p")

        assert quality == 0.7  # Default

    def test_get_config_quality_with_provider(self):
        """Test quality with custom provider."""
        def quality_provider(config_key: str) -> float:
            if config_key == "hex8_2p":
                return 0.9
            return 0.5

        manager = QualityManager(quality_provider=quality_provider)

        assert manager.get_config_quality("hex8_2p") == 0.9
        assert manager.get_config_quality("square8_2p") == 0.5

    def test_get_all_config_qualities(self):
        """Test getting quality for multiple configs."""
        manager = QualityManager()

        qualities = manager.get_all_config_qualities(["hex8_2p", "square8_2p"])

        assert len(qualities) == 2
        assert "hex8_2p" in qualities
        assert "square8_2p" in qualities

    def test_set_and_get_quality(self):
        """Test manually setting quality."""
        manager = QualityManager()

        manager.set_quality("hex8_2p", 0.95)
        quality = manager.get_config_quality("hex8_2p")

        assert quality == 0.95

    def test_invalidate_quality_single(self):
        """Test invalidating single config."""
        manager = QualityManager()
        manager.set_quality("hex8_2p", 0.95)
        manager.set_quality("square8_2p", 0.85)

        count = manager.invalidate_quality_cache("hex8_2p")

        assert count == 1
        # After invalidation, should return default
        assert manager.get_config_quality("hex8_2p") == 0.7
        # Other config should still be cached
        assert manager.get_config_quality("square8_2p") == 0.85

    def test_invalidate_quality_all(self):
        """Test invalidating all configs."""
        manager = QualityManager()
        manager.set_quality("hex8_2p", 0.95)
        manager.set_quality("square8_2p", 0.85)

        count = manager.invalidate_quality_cache()

        assert count == 2

    def test_diversity_delegation(self):
        """Test diversity methods delegate to tracker."""
        manager = QualityManager()

        score = manager.record_opponent("hex8_2p", "gumbel")

        assert score == 0.125
        assert manager.get_diversity_score("hex8_2p") == 0.125
        assert manager.get_opponent_types_seen("hex8_2p") == 1

    def test_get_diversity_stats(self):
        """Test getting full diversity stats."""
        manager = QualityManager()
        manager.record_opponent("hex8_2p", "gumbel", game_count=5)
        manager.record_opponent("hex8_2p", "heuristic", game_count=3)

        stats = manager.get_diversity_stats("hex8_2p")

        assert stats.opponent_count == 2
        assert stats.games_by_opponent == {"gumbel": 5, "heuristic": 3}

    def test_get_config_scores(self):
        """Test getting both scores at once."""
        manager = QualityManager()
        manager.set_quality("hex8_2p", 0.85)
        manager.record_opponent("hex8_2p", "gumbel")
        manager.record_opponent("hex8_2p", "heuristic")

        quality, diversity = manager.get_config_scores("hex8_2p")

        assert quality == 0.85
        assert diversity == 0.25  # 2/8

    def test_get_underserved_configs(self):
        """Test finding underserved configs."""
        manager = QualityManager(max_opponent_types=4)

        # Good quality, low diversity
        manager.set_quality("hex8_2p", 0.8)
        manager.record_opponent("hex8_2p", "gumbel")  # 0.25 diversity

        # Low quality, high diversity
        manager.set_quality("square8_2p", 0.4)
        manager.record_opponent("square8_2p", "gumbel")
        manager.record_opponent("square8_2p", "heuristic")
        manager.record_opponent("square8_2p", "policy")  # 0.75 diversity

        # Good quality, high diversity
        manager.set_quality("hex8_4p", 0.9)
        manager.record_opponent("hex8_4p", "gumbel")
        manager.record_opponent("hex8_4p", "heuristic")
        manager.record_opponent("hex8_4p", "policy")
        manager.record_opponent("hex8_4p", "mcts")  # 1.0 diversity

        underserved = manager.get_underserved_configs(
            config_keys=["hex8_2p", "square8_2p", "hex8_4p"],
            quality_threshold=0.6,
            diversity_threshold=0.5,
        )

        # hex8_2p: low diversity (0.25 < 0.5)
        # square8_2p: low quality (0.4 < 0.6)
        assert "hex8_2p" in underserved
        assert "square8_2p" in underserved
        assert "hex8_4p" not in underserved

    def test_reset_single_config(self):
        """Test resetting single config."""
        manager = QualityManager()
        manager.set_quality("hex8_2p", 0.9)
        manager.record_opponent("hex8_2p", "gumbel")

        count = manager.reset("hex8_2p")

        assert count >= 1
        assert manager.get_config_quality("hex8_2p") == 0.7  # Default
        assert manager.get_diversity_score("hex8_2p") == 0.0

    def test_reset_all(self):
        """Test resetting all configs."""
        manager = QualityManager()
        manager.set_quality("hex8_2p", 0.9)
        manager.set_quality("square8_2p", 0.8)
        manager.record_opponent("hex8_2p", "gumbel")

        count = manager.reset()

        assert count >= 2

    def test_get_status(self):
        """Test getting manager status."""
        manager = QualityManager()
        manager.set_quality("hex8_2p", 0.9)
        manager.record_opponent("hex8_2p", "gumbel")

        status = manager.get_status()

        assert "quality_cache" in status
        assert "diversity_tracker" in status
        assert status["quality_cache"]["entries_count"] >= 1
        assert status["diversity_tracker"]["configs_tracked"] >= 1


class TestSingleton:
    """Tests for singleton management."""

    def test_get_quality_manager(self):
        """Test getting singleton instance."""
        reset_quality_manager()  # Ensure clean state

        manager1 = get_quality_manager()
        manager2 = get_quality_manager()

        assert manager1 is manager2

    def test_reset_quality_manager(self):
        """Test resetting singleton."""
        manager1 = get_quality_manager()
        reset_quality_manager()
        manager2 = get_quality_manager()

        assert manager1 is not manager2

    def test_singleton_preserves_state(self):
        """Test singleton preserves state across calls."""
        reset_quality_manager()

        manager1 = get_quality_manager()
        manager1.record_opponent("hex8_2p", "gumbel")

        manager2 = get_quality_manager()

        assert manager2.get_diversity_score("hex8_2p") == 0.125


class TestKnownOpponentTypes:
    """Tests for known opponent types constant."""

    def test_known_types_exist(self):
        """Test known opponent types are defined."""
        assert len(KNOWN_OPPONENT_TYPES) > 0
        assert "gumbel" in KNOWN_OPPONENT_TYPES
        assert "heuristic" in KNOWN_OPPONENT_TYPES
        assert "mcts" in KNOWN_OPPONENT_TYPES

    def test_known_types_is_frozenset(self):
        """Test known types is immutable."""
        assert isinstance(KNOWN_OPPONENT_TYPES, frozenset)
