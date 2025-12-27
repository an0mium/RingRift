"""Tests for app.quality.unified_quality - Unified Quality Scoring System.

This module tests the UnifiedQualityScorer which is the single source of truth
for all quality scoring operations across the RingRift AI system.
"""

from __future__ import annotations

import math
import time
from unittest.mock import MagicMock, patch

import pytest

from app.quality.unified_quality import (
    GameQuality,
    QualityCategory,
    QualityWeights,
    UnifiedQualityScorer,
    compute_game_quality,
    compute_sample_weight,
    compute_sync_priority,
    get_quality_category,
    get_quality_scorer,
)


# =============================================================================
# QualityCategory Tests
# =============================================================================


class TestQualityCategory:
    """Tests for QualityCategory enum."""

    def test_category_values(self):
        """All categories should have correct values."""
        assert QualityCategory.EXCELLENT.value == "excellent"
        assert QualityCategory.GOOD.value == "good"
        assert QualityCategory.ADEQUATE.value == "adequate"
        assert QualityCategory.POOR.value == "poor"
        assert QualityCategory.UNUSABLE.value == "unusable"

    def test_from_score_excellent(self):
        """Score 0.85+ should be EXCELLENT."""
        assert QualityCategory.from_score(0.85) == QualityCategory.EXCELLENT
        assert QualityCategory.from_score(0.90) == QualityCategory.EXCELLENT
        assert QualityCategory.from_score(1.0) == QualityCategory.EXCELLENT

    def test_from_score_good(self):
        """Score 0.70-0.85 should be GOOD."""
        assert QualityCategory.from_score(0.70) == QualityCategory.GOOD
        assert QualityCategory.from_score(0.75) == QualityCategory.GOOD
        assert QualityCategory.from_score(0.84) == QualityCategory.GOOD

    def test_from_score_adequate(self):
        """Score 0.50-0.70 should be ADEQUATE."""
        assert QualityCategory.from_score(0.50) == QualityCategory.ADEQUATE
        assert QualityCategory.from_score(0.60) == QualityCategory.ADEQUATE
        assert QualityCategory.from_score(0.69) == QualityCategory.ADEQUATE

    def test_from_score_poor(self):
        """Score 0.30-0.50 should be POOR."""
        assert QualityCategory.from_score(0.30) == QualityCategory.POOR
        assert QualityCategory.from_score(0.40) == QualityCategory.POOR
        assert QualityCategory.from_score(0.49) == QualityCategory.POOR

    def test_from_score_unusable(self):
        """Score <0.30 should be UNUSABLE."""
        assert QualityCategory.from_score(0.29) == QualityCategory.UNUSABLE
        assert QualityCategory.from_score(0.10) == QualityCategory.UNUSABLE
        assert QualityCategory.from_score(0.0) == QualityCategory.UNUSABLE

    def test_from_score_negative(self):
        """Negative scores should be UNUSABLE."""
        assert QualityCategory.from_score(-0.1) == QualityCategory.UNUSABLE

    def test_from_score_above_one(self):
        """Scores above 1.0 should still be EXCELLENT."""
        assert QualityCategory.from_score(1.5) == QualityCategory.EXCELLENT


# =============================================================================
# GameQuality Tests
# =============================================================================


class TestGameQuality:
    """Tests for GameQuality dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        quality = GameQuality(game_id="test-game")

        assert quality.game_id == "test-game"
        assert quality.outcome_score == 0.0
        assert quality.length_score == 0.0
        assert quality.phase_balance_score == 0.0
        assert quality.diversity_score == 0.0
        assert quality.source_reputation_score == 0.0
        assert quality.avg_player_elo == 1500.0
        assert quality.quality_score == 0.0
        assert quality.training_weight == 1.0
        assert quality.is_decisive is False

    def test_category_property(self):
        """category property should return correct category."""
        quality = GameQuality(game_id="test", quality_score=0.80)
        assert quality.category == QualityCategory.GOOD

        quality2 = GameQuality(game_id="test", quality_score=0.90)
        assert quality2.category == QualityCategory.EXCELLENT

    def test_to_dict(self):
        """to_dict should return serializable dict."""
        quality = GameQuality(
            game_id="test-123",
            quality_score=0.75,
            training_weight=1.2,
            game_length=50,
            is_decisive=True,
        )
        d = quality.to_dict()

        assert d["game_id"] == "test-123"
        assert d["quality_score"] == 0.75
        assert d["training_weight"] == 1.2
        assert d["game_length"] == 50
        assert d["is_decisive"] is True

    def test_to_dict_contains_all_scores(self):
        """to_dict should include all component scores."""
        quality = GameQuality(
            game_id="test",
            outcome_score=0.8,
            length_score=0.7,
            phase_balance_score=0.6,
            diversity_score=0.5,
        )
        d = quality.to_dict()

        assert "outcome_score" in d
        assert "length_score" in d
        assert "phase_balance_score" in d
        assert "diversity_score" in d


# =============================================================================
# QualityWeights Tests
# =============================================================================


class TestQualityWeights:
    """Tests for QualityWeights dataclass."""

    def test_default_weights(self):
        """Default weights should sum to 1.0 for game quality."""
        weights = QualityWeights()

        game_sum = (
            weights.outcome_weight +
            weights.length_weight +
            weights.phase_balance_weight +
            weights.diversity_weight +
            weights.source_reputation_weight
        )
        assert abs(game_sum - 1.0) < 0.01

    def test_default_sync_weights(self):
        """Default sync weights should sum to 1.0."""
        weights = QualityWeights()

        sync_sum = (
            weights.sync_elo_weight +
            weights.sync_length_weight +
            weights.sync_decisive_weight
        )
        assert abs(sync_sum - 1.0) < 0.01

    def test_default_sampling_weights(self):
        """Default sampling weights should sum to 1.0."""
        weights = QualityWeights()

        sampling_sum = (
            weights.quality_weight +
            weights.recency_weight +
            weights.priority_weight
        )
        assert abs(sampling_sum - 1.0) < 0.01

    def test_elo_range(self):
        """Elo range should be reasonable."""
        weights = QualityWeights()
        assert weights.min_elo < weights.default_elo < weights.max_elo

    def test_game_length_range(self):
        """Game length range should be reasonable."""
        weights = QualityWeights()
        assert weights.min_game_length < weights.optimal_game_length < weights.max_game_length

    def test_from_config_no_config(self):
        """from_config should return defaults without config."""
        weights = QualityWeights.from_config(None)
        assert weights.outcome_weight == 0.25

    def test_from_config_with_mock(self):
        """from_config should use config values when available."""
        mock_config = MagicMock()
        mock_config.outcome_weight = 0.5
        mock_config.length_weight = 0.5

        weights = QualityWeights.from_config(mock_config)
        assert weights.outcome_weight == 0.5
        assert weights.length_weight == 0.5


# =============================================================================
# UnifiedQualityScorer Tests
# =============================================================================


class TestUnifiedQualityScorerInit:
    """Tests for UnifiedQualityScorer initialization."""

    def setup_method(self):
        """Reset singleton before each test."""
        UnifiedQualityScorer.reset_instance()

    def test_default_init(self):
        """Should initialize with default weights."""
        scorer = UnifiedQualityScorer()
        assert scorer.weights is not None
        assert scorer.elo_lookup is None

    def test_custom_weights(self):
        """Should accept custom weights."""
        custom_weights = QualityWeights(outcome_weight=0.5)
        scorer = UnifiedQualityScorer(weights=custom_weights)
        assert scorer.weights.outcome_weight == 0.5

    def test_elo_lookup(self):
        """Should accept elo_lookup function."""
        def lookup(model_id: str) -> float:
            return 1600.0

        scorer = UnifiedQualityScorer(elo_lookup=lookup)
        assert scorer.elo_lookup is not None
        assert scorer.elo_lookup("test") == 1600.0

    def test_set_elo_lookup(self):
        """set_elo_lookup should update the lookup function."""
        scorer = UnifiedQualityScorer()
        assert scorer.elo_lookup is None

        scorer.set_elo_lookup(lambda x: 1700.0)
        assert scorer.elo_lookup("any") == 1700.0


class TestUnifiedQualityScorerSingleton:
    """Tests for singleton pattern."""

    def setup_method(self):
        """Reset singleton before each test."""
        UnifiedQualityScorer.reset_instance()

    def test_get_instance_creates_singleton(self):
        """get_instance should create singleton."""
        scorer1 = UnifiedQualityScorer.get_instance()
        scorer2 = UnifiedQualityScorer.get_instance()
        assert scorer1 is scorer2

    def test_reset_instance_clears_singleton(self):
        """reset_instance should clear singleton."""
        scorer1 = UnifiedQualityScorer.get_instance()
        UnifiedQualityScorer.reset_instance()
        scorer2 = UnifiedQualityScorer.get_instance()
        assert scorer1 is not scorer2


class TestComputeGameQuality:
    """Tests for compute_game_quality method."""

    def setup_method(self):
        """Reset singleton before each test."""
        UnifiedQualityScorer.reset_instance()

    def test_basic_quality_computation(self):
        """Should compute quality for basic game data."""
        scorer = UnifiedQualityScorer()
        game_data = {
            "game_id": "test-game",
            "total_moves": 80,
            "winner": 0,
        }
        quality = scorer.compute_game_quality(game_data)

        assert quality.game_id == "test-game"
        assert quality.game_length == 80
        assert quality.is_decisive is True
        assert 0.0 <= quality.quality_score <= 1.0

    def test_decisive_vs_draw(self):
        """Decisive games should have higher outcome score."""
        scorer = UnifiedQualityScorer()

        decisive_game = {"game_id": "decisive", "total_moves": 50, "winner": 0}
        draw_game = {"game_id": "draw", "total_moves": 50, "winner": -1}

        decisive_quality = scorer.compute_game_quality(decisive_game)
        draw_quality = scorer.compute_game_quality(draw_game)

        assert decisive_quality.outcome_score > draw_quality.outcome_score

    def test_optimal_length_score(self):
        """Game at optimal length should have high length score."""
        scorer = UnifiedQualityScorer()
        weights = scorer.weights

        game_data = {
            "game_id": "optimal",
            "total_moves": weights.optimal_game_length,
            "winner": 0,
        }
        quality = scorer.compute_game_quality(game_data)

        # At optimal length, score should be 1.0
        assert quality.length_score == 1.0

    def test_short_game_length_score(self):
        """Very short games should have low length score."""
        scorer = UnifiedQualityScorer()

        game_data = {
            "game_id": "short",
            "total_moves": 5,  # Below min_game_length
            "winner": 0,
        }
        quality = scorer.compute_game_quality(game_data)

        assert quality.length_score == 0.0

    def test_very_long_game_length_score(self):
        """Very long games should have slight penalty."""
        scorer = UnifiedQualityScorer()
        weights = scorer.weights

        game_data = {
            "game_id": "long",
            "total_moves": weights.max_game_length + 50,
            "winner": 0,
        }
        quality = scorer.compute_game_quality(game_data)

        assert quality.length_score == 0.8

    def test_elo_lookup_used(self):
        """Elo lookup should be called for model_version."""
        scorer = UnifiedQualityScorer()

        def mock_lookup(model_id: str) -> float:
            return 1800.0

        game_data = {
            "game_id": "elo-test",
            "total_moves": 50,
            "winner": 0,
            "model_version": "v2.0",
        }
        quality = scorer.compute_game_quality(game_data, elo_lookup=mock_lookup)

        assert quality.avg_player_elo == 1800.0

    def test_quality_score_bounded(self):
        """Quality score should be bounded to [0, 1]."""
        scorer = UnifiedQualityScorer()

        # Create game with all high scores
        game_data = {
            "game_id": "high-quality",
            "total_moves": 80,
            "winner": 0,
            "phase_balance_score": 1.0,
            "diversity_score": 1.0,
            "source": "test-source",
        }
        quality = scorer.compute_game_quality(game_data)

        assert 0.0 <= quality.quality_score <= 1.0

    def test_move_count_fallback(self):
        """Should use move_count if total_moves not provided."""
        scorer = UnifiedQualityScorer()
        game_data = {
            "game_id": "test",
            "move_count": 60,
            "winner": 0,
        }
        quality = scorer.compute_game_quality(game_data)
        assert quality.game_length == 60


class TestComputeSampleWeight:
    """Tests for compute_sample_weight method."""

    def setup_method(self):
        """Reset singleton before each test."""
        UnifiedQualityScorer.reset_instance()

    def test_basic_sample_weight(self):
        """Should compute sample weight from quality."""
        scorer = UnifiedQualityScorer()
        quality = GameQuality(game_id="test", quality_score=0.8)
        weight = scorer.compute_sample_weight(quality)
        assert weight >= 0.0

    def test_higher_quality_higher_weight(self):
        """Higher quality should generally lead to higher weight."""
        scorer = UnifiedQualityScorer()

        high_quality = GameQuality(game_id="high", quality_score=0.9)
        low_quality = GameQuality(game_id="low", quality_score=0.3)

        high_weight = scorer.compute_sample_weight(high_quality, recency_hours=0)
        low_weight = scorer.compute_sample_weight(low_quality, recency_hours=0)

        assert high_weight > low_weight

    def test_recency_decay(self):
        """Older games should have lower weight due to recency decay."""
        scorer = UnifiedQualityScorer()
        quality = GameQuality(game_id="test", quality_score=0.8)

        fresh_weight = scorer.compute_sample_weight(quality, recency_hours=0)
        old_weight = scorer.compute_sample_weight(quality, recency_hours=100)

        assert fresh_weight > old_weight

    def test_float_input(self):
        """Should accept float quality score directly."""
        scorer = UnifiedQualityScorer()
        weight = scorer.compute_sample_weight(0.7, recency_hours=0)
        assert weight >= 0.0


class TestComputeSyncPriority:
    """Tests for compute_sync_priority method."""

    def setup_method(self):
        """Reset singleton before each test."""
        UnifiedQualityScorer.reset_instance()

    def test_basic_sync_priority(self):
        """Should compute sync priority."""
        scorer = UnifiedQualityScorer()
        quality = GameQuality(
            game_id="test",
            avg_player_elo=1800,
            game_length=100,
            is_decisive=True,
        )
        priority = scorer.compute_sync_priority(quality)
        assert 0.0 <= priority <= 1.0

    def test_decisive_higher_priority(self):
        """Decisive games should have higher priority."""
        scorer = UnifiedQualityScorer()

        decisive = GameQuality(game_id="d", is_decisive=True, game_length=50, avg_player_elo=1500)
        draw = GameQuality(game_id="n", is_decisive=False, game_length=50, avg_player_elo=1500)

        assert scorer.compute_sync_priority(decisive) > scorer.compute_sync_priority(draw)

    def test_higher_elo_higher_priority(self):
        """Higher Elo games should have higher priority."""
        scorer = UnifiedQualityScorer()

        high_elo = GameQuality(game_id="h", avg_player_elo=2000, game_length=50, is_decisive=True)
        low_elo = GameQuality(game_id="l", avg_player_elo=1300, game_length=50, is_decisive=True)

        assert scorer.compute_sync_priority(high_elo) > scorer.compute_sync_priority(low_elo)

    def test_urgency_bonus(self):
        """Older unsynced games should get urgency bonus."""
        scorer = UnifiedQualityScorer()
        quality = GameQuality(game_id="test", game_length=50, is_decisive=True)

        fresh_priority = scorer.compute_sync_priority(quality, urgency_hours=0)
        urgent_priority = scorer.compute_sync_priority(quality, urgency_hours=48)

        assert urgent_priority > fresh_priority


class TestComputeEloWeight:
    """Tests for Elo-based weighting."""

    def setup_method(self):
        """Reset singleton before each test."""
        UnifiedQualityScorer.reset_instance()

    def test_equal_elo_weight(self):
        """Equal Elo should give ~1.25 weight (midpoint)."""
        scorer = UnifiedQualityScorer()
        weight = scorer.compute_elo_weight(1500, 1500)
        # Sigmoid of 0 = 0.5, so weight = 0.5 + 0.5 * 1.5 = 1.25
        assert abs(weight - 1.25) < 0.1

    def test_stronger_opponent_higher_weight(self):
        """Stronger opponent should give higher weight."""
        scorer = UnifiedQualityScorer()

        weaker = scorer.compute_elo_weight(1500, 1300)
        stronger = scorer.compute_elo_weight(1500, 1700)

        assert stronger > weaker

    def test_batch_elo_weights(self):
        """compute_elo_weights_batch should process multiple samples."""
        scorer = UnifiedQualityScorer()

        opponent_elos = [1400, 1500, 1600, 1700]
        weights = scorer.compute_elo_weights_batch(opponent_elos, model_elo=1500)

        assert len(weights) == 4
        # Should be normalized to mean=1
        mean_weight = sum(weights) / len(weights)
        assert abs(mean_weight - 1.0) < 0.01

    def test_batch_empty_list(self):
        """compute_elo_weights_batch should handle empty list."""
        scorer = UnifiedQualityScorer()
        weights = scorer.compute_elo_weights_batch([], model_elo=1500)
        assert weights == []

    def test_batch_without_normalization(self):
        """compute_elo_weights_batch should skip normalization when requested."""
        scorer = UnifiedQualityScorer()

        opponent_elos = [1500, 1500, 1500]
        weights = scorer.compute_elo_weights_batch(
            opponent_elos, model_elo=1500, normalize=False
        )

        # Without normalization, equal Elo gives midpoint weight
        assert all(abs(w - 1.6) < 0.2 for w in weights)


class TestQualityThresholds:
    """Tests for quality threshold checks."""

    def setup_method(self):
        """Reset singleton before each test."""
        UnifiedQualityScorer.reset_instance()

    def test_is_high_quality(self):
        """is_high_quality should check threshold correctly."""
        scorer = UnifiedQualityScorer()

        high = GameQuality(game_id="h", quality_score=0.8)
        low = GameQuality(game_id="l", quality_score=0.5)

        assert scorer.is_high_quality(high) is True
        assert scorer.is_high_quality(low) is False

    def test_is_training_worthy(self):
        """is_training_worthy should check minimum quality."""
        scorer = UnifiedQualityScorer()

        good = GameQuality(game_id="g", quality_score=0.5)
        bad = GameQuality(game_id="b", quality_score=0.2)

        assert scorer.is_training_worthy(good) is True
        assert scorer.is_training_worthy(bad) is False

    def test_is_priority_sync_worthy(self):
        """is_priority_sync_worthy should check sync threshold."""
        scorer = UnifiedQualityScorer()

        priority = GameQuality(game_id="p", quality_score=0.6)
        normal = GameQuality(game_id="n", quality_score=0.4)

        assert scorer.is_priority_sync_worthy(priority) is True
        assert scorer.is_priority_sync_worthy(normal) is False


class TestFreshnessScore:
    """Tests for freshness score computation."""

    def setup_method(self):
        """Reset singleton before each test."""
        UnifiedQualityScorer.reset_instance()

    def test_fresh_game_high_score(self):
        """Recent game should have high freshness."""
        scorer = UnifiedQualityScorer()
        current = time.time()
        freshness = scorer.compute_freshness_score(current - 60, current)  # 1 minute old
        assert freshness > 0.95

    def test_old_game_low_score(self):
        """Old game should have low freshness."""
        scorer = UnifiedQualityScorer()
        current = time.time()
        week_ago = current - (7 * 24 * 3600)  # 1 week old
        freshness = scorer.compute_freshness_score(week_ago, current)
        assert freshness < 0.1

    def test_none_timestamp(self):
        """None timestamp should return neutral 0.5."""
        scorer = UnifiedQualityScorer()
        freshness = scorer.compute_freshness_score(None)
        assert freshness == 0.5

    def test_future_timestamp(self):
        """Future timestamp should return max freshness."""
        scorer = UnifiedQualityScorer()
        current = time.time()
        future = current + 3600  # 1 hour in future
        freshness = scorer.compute_freshness_score(future, current)
        assert freshness == 1.0


# =============================================================================
# Module-level Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def setup_method(self):
        """Reset singleton before each test."""
        UnifiedQualityScorer.reset_instance()

    def test_get_quality_scorer(self):
        """get_quality_scorer should return singleton."""
        scorer1 = get_quality_scorer()
        scorer2 = get_quality_scorer()
        assert scorer1 is scorer2

    def test_compute_game_quality_function(self):
        """compute_game_quality should work as convenience function."""
        game_data = {
            "game_id": "test",
            "total_moves": 50,
            "winner": 0,
        }
        quality = compute_game_quality(game_data)
        assert quality.game_id == "test"

    def test_compute_sample_weight_function(self):
        """compute_sample_weight should work as convenience function."""
        quality = GameQuality(game_id="test", quality_score=0.7)
        weight = compute_sample_weight(quality)
        assert weight >= 0.0

    def test_compute_sync_priority_function(self):
        """compute_sync_priority should work as convenience function."""
        quality = GameQuality(game_id="test", game_length=50, is_decisive=True)
        priority = compute_sync_priority(quality)
        assert 0.0 <= priority <= 1.0

    def test_get_quality_category(self):
        """get_quality_category should return correct category."""
        assert get_quality_category(0.90) == QualityCategory.EXCELLENT
        assert get_quality_category(0.75) == QualityCategory.GOOD
        assert get_quality_category(0.55) == QualityCategory.ADEQUATE


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def setup_method(self):
        """Reset singleton before each test."""
        UnifiedQualityScorer.reset_instance()

    def test_empty_game_data(self):
        """Should handle minimal game data."""
        scorer = UnifiedQualityScorer()
        quality = scorer.compute_game_quality({"game_id": "empty"})
        assert quality.game_id == "empty"
        assert quality.game_length == 0

    def test_negative_move_count(self):
        """Should handle negative move count gracefully."""
        scorer = UnifiedQualityScorer()
        quality = scorer.compute_game_quality({
            "game_id": "negative",
            "total_moves": -5,
            "winner": 0,
        })
        # Negative should be treated as 0 or handled gracefully
        assert quality.length_score == 0.0

    def test_very_high_elo(self):
        """Should handle Elo above max range."""
        scorer = UnifiedQualityScorer()

        def high_elo_lookup(model_id: str) -> float:
            return 3000.0

        quality = scorer.compute_game_quality(
            {"game_id": "high-elo", "total_moves": 50, "winner": 0, "model_version": "v1"},
            elo_lookup=high_elo_lookup,
        )
        # Elo score should be clamped to 1.0
        assert quality.elo_score == 1.0

    def test_elo_lookup_failure(self):
        """Should handle Elo lookup failures gracefully."""
        scorer = UnifiedQualityScorer()

        def failing_lookup(model_id: str) -> float:
            raise ValueError("Lookup failed")

        # Should not raise
        quality = scorer.compute_game_quality(
            {"game_id": "fail", "total_moves": 50, "winner": 0, "model_version": "v1"},
            elo_lookup=failing_lookup,
        )
        # Should use default Elo
        assert quality.avg_player_elo == 1500.0
