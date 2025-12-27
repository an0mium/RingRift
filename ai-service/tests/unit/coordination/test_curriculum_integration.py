"""Tests for curriculum_integration.py module.

December 2025: Added as part of test coverage initiative.
"""

from __future__ import annotations

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from app.coordination.curriculum_integration import (
    MomentumToCurriculumBridge,
    PFSPWeaknessWatcher,
    QualityPenaltyToCurriculumWatcher,
    QualityToTemperatureWatcher,
    wire_all_feedback_loops,
    unwire_all_feedback_loops,
    get_integration_status,
    get_exploration_boost,
    get_mastered_opponents,
    force_momentum_sync,
    get_quality_penalty_weights,
    reset_quality_penalty,
    get_promotion_failure_counts,
    reset_promotion_failure_count,
)


# =============================================================================
# MomentumToCurriculumBridge Tests
# =============================================================================


class TestMomentumToCurriculumBridge:
    """Tests for MomentumToCurriculumBridge class."""

    def test_init_defaults(self):
        """Bridge initializes with default values."""
        bridge = MomentumToCurriculumBridge()
        assert bridge.poll_interval_seconds == 10.0
        assert bridge.momentum_weight_boost == 0.3
        assert bridge._running is False
        assert bridge._event_subscribed is False

    def test_init_custom(self):
        """Bridge accepts custom parameters."""
        bridge = MomentumToCurriculumBridge(
            poll_interval_seconds=30.0,
            momentum_weight_boost=0.5,
        )
        assert bridge.poll_interval_seconds == 30.0
        assert bridge.momentum_weight_boost == 0.5

    def test_stop_when_not_running(self):
        """Stop is safe when not running."""
        bridge = MomentumToCurriculumBridge()
        bridge.stop()  # Should not raise

    def test_start_sets_running_flag(self):
        """Start sets the running flag."""
        bridge = MomentumToCurriculumBridge()

        # Mock event subscription to avoid actual event bus
        with patch.object(bridge, '_subscribe_to_events', return_value=True):
            bridge.start()
            assert bridge._running is True
            bridge.stop()

    def test_force_sync_returns_dict(self):
        """force_sync returns a dict (may contain weights from accelerator)."""
        bridge = MomentumToCurriculumBridge()
        result = bridge.force_sync()
        assert isinstance(result, dict)
        # Values should be floats if present
        for key, value in result.items():
            assert isinstance(key, str)
            assert isinstance(value, float)


# =============================================================================
# PFSPWeaknessWatcher Tests
# =============================================================================


class TestPFSPWeaknessWatcher:
    """Tests for PFSPWeaknessWatcher class."""

    def test_init(self):
        """Watcher initializes with correct defaults."""
        watcher = PFSPWeaknessWatcher()
        assert watcher.MASTERY_THRESHOLD == 0.85
        assert watcher.MIN_GAMES_FOR_MASTERY == 20
        assert watcher.CHECK_INTERVAL == 120.0
        assert watcher._running is False
        assert len(watcher._mastered_matchups) == 0

    def test_get_mastered_matchups_empty(self):
        """get_mastered_matchups returns empty list initially."""
        watcher = PFSPWeaknessWatcher()
        assert watcher.get_mastered_matchups() == []

    def test_stop_when_not_running(self):
        """Stop is safe when not running."""
        watcher = PFSPWeaknessWatcher()
        watcher.stop()  # Should not raise

    def test_extract_config_standard(self):
        """_extract_config parses standard model IDs."""
        watcher = PFSPWeaknessWatcher()

        assert watcher._extract_config("hex8_2p_v123") == "hex8_2p"
        assert watcher._extract_config("square8_4p_v1") == "square8_4p"
        assert watcher._extract_config("canonical_hex8_2p") == "hex8_2p"

    def test_extract_config_fallback(self):
        """_extract_config falls back to first two parts."""
        watcher = PFSPWeaknessWatcher()

        assert watcher._extract_config("model_unknown") == "model_unknown"
        assert watcher._extract_config("simple") == "simple"


# =============================================================================
# QualityPenaltyToCurriculumWatcher Tests
# =============================================================================


class TestQualityPenaltyToCurriculumWatcher:
    """Tests for QualityPenaltyToCurriculumWatcher class."""

    def test_init(self):
        """Watcher initializes with correct defaults."""
        watcher = QualityPenaltyToCurriculumWatcher()
        assert watcher.WEIGHT_REDUCTION_PER_PENALTY == 0.15
        assert watcher._subscribed is False
        assert len(watcher._penalty_weights) == 0

    def test_get_penalty_weights_empty(self):
        """get_penalty_weights returns empty dict initially."""
        watcher = QualityPenaltyToCurriculumWatcher()
        assert watcher.get_penalty_weights() == {}

    def test_reset_penalty(self):
        """reset_penalty removes weight for config."""
        watcher = QualityPenaltyToCurriculumWatcher()
        watcher._penalty_weights["hex8_2p"] = 0.7

        watcher.reset_penalty("hex8_2p")
        assert "hex8_2p" not in watcher._penalty_weights

    def test_reset_penalty_nonexistent(self):
        """reset_penalty is safe for nonexistent config."""
        watcher = QualityPenaltyToCurriculumWatcher()
        watcher.reset_penalty("nonexistent")  # Should not raise


# =============================================================================
# QualityToTemperatureWatcher Tests
# =============================================================================


class TestQualityToTemperatureWatcher:
    """Tests for QualityToTemperatureWatcher class."""

    def test_init(self):
        """Watcher initializes with correct defaults."""
        watcher = QualityToTemperatureWatcher()
        assert watcher.EXPLORATION_BOOST_FACTOR == 1.3
        assert watcher._subscribed is False
        assert len(watcher._quality_boosts) == 0

    def test_low_quality_threshold_property(self):
        """LOW_QUALITY_THRESHOLD is a property with fallback."""
        watcher = QualityToTemperatureWatcher()
        threshold = watcher.LOW_QUALITY_THRESHOLD
        # Should be a float between 0 and 1
        assert isinstance(threshold, float)
        assert 0 < threshold < 1

    def test_get_exploration_boost_default(self):
        """get_exploration_boost returns 1.0 for unknown config."""
        watcher = QualityToTemperatureWatcher()
        assert watcher.get_exploration_boost("unknown_config") == 1.0

    def test_get_exploration_boost_tracked(self):
        """get_exploration_boost returns tracked value."""
        watcher = QualityToTemperatureWatcher()
        watcher._quality_boosts["hex8_2p"] = 1.5

        assert watcher.get_exploration_boost("hex8_2p") == 1.5

    def test_get_all_boosts_empty(self):
        """get_all_boosts returns empty dict initially."""
        watcher = QualityToTemperatureWatcher()
        assert watcher.get_all_boosts() == {}

    def test_get_all_boosts_copy(self):
        """get_all_boosts returns a copy."""
        watcher = QualityToTemperatureWatcher()
        watcher._quality_boosts["hex8_2p"] = 1.3

        boosts = watcher.get_all_boosts()
        boosts["hex8_2p"] = 2.0  # Modify the copy

        # Original should be unchanged
        assert watcher._quality_boosts["hex8_2p"] == 1.3


# =============================================================================
# Wiring Function Tests
# =============================================================================


class TestWiringFunctions:
    """Tests for wire/unwire functions."""

    def test_wire_already_active(self):
        """wire_all_feedback_loops returns early if already active."""
        # First call
        with patch('app.coordination.curriculum_integration._integration_active', True):
            result = wire_all_feedback_loops()
            assert result["status"] == "already_active"

    def test_unwire_clears_state(self):
        """unwire_all_feedback_loops clears integration state."""
        # Ensure clean state after unwire
        unwire_all_feedback_loops()

        status = get_integration_status()
        assert status["active"] is False
        assert status["watchers"] == []


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_exploration_boost_no_watcher(self):
        """get_exploration_boost returns 1.0 when watcher not active."""
        unwire_all_feedback_loops()  # Ensure clean state
        assert get_exploration_boost("any_config") == 1.0

    def test_get_mastered_opponents_empty(self):
        """get_mastered_opponents returns empty list when watcher not active."""
        unwire_all_feedback_loops()  # Ensure clean state
        assert get_mastered_opponents() == []

    def test_force_momentum_sync_empty(self):
        """force_momentum_sync returns empty dict when bridge not active."""
        unwire_all_feedback_loops()  # Ensure clean state
        assert force_momentum_sync() == {}

    def test_get_quality_penalty_weights_empty(self):
        """get_quality_penalty_weights returns empty dict when watcher not active."""
        unwire_all_feedback_loops()  # Ensure clean state
        assert get_quality_penalty_weights() == {}

    def test_reset_quality_penalty_no_watcher(self):
        """reset_quality_penalty is safe when watcher not active."""
        unwire_all_feedback_loops()  # Ensure clean state
        reset_quality_penalty("any_config")  # Should not raise

    def test_get_promotion_failure_counts_empty(self):
        """get_promotion_failure_counts returns empty dict when watcher not active."""
        unwire_all_feedback_loops()  # Ensure clean state
        assert get_promotion_failure_counts() == {}

    def test_reset_promotion_failure_count_no_watcher(self):
        """reset_promotion_failure_count is safe when watcher not active."""
        unwire_all_feedback_loops()  # Ensure clean state
        reset_promotion_failure_count("any_config")  # Should not raise


# =============================================================================
# Integration Status Tests
# =============================================================================


class TestIntegrationStatus:
    """Tests for get_integration_status function."""

    def test_status_inactive(self):
        """Status shows inactive when not wired."""
        unwire_all_feedback_loops()  # Ensure clean state

        status = get_integration_status()
        assert status["active"] is False
        assert isinstance(status["watchers"], list)

    def test_status_structure(self):
        """Status has expected structure."""
        status = get_integration_status()

        assert "active" in status
        assert "watchers" in status
        assert isinstance(status["active"], bool)
        assert isinstance(status["watchers"], list)
