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


# =============================================================================
# Event Handler Tests (December 29, 2025)
# =============================================================================


class TestCurriculumAdvancementHandler:
    """Tests for _on_curriculum_advancement_needed handler.

    December 29, 2025: Tests for the handler that closes the curriculum feedback loop.
    When a config stagnates (3+ evaluations with minimal Elo improvement),
    TrainingTriggerDaemon emits CURRICULUM_ADVANCEMENT_NEEDED.
    """

    def test_handler_with_valid_event(self):
        """Handler processes valid event with config_key."""
        bridge = MomentumToCurriculumBridge()

        # Create mock event with proper structure
        mock_event = MagicMock()
        mock_event.payload = {
            "config_key": "hex8_2p",
            "reason": "elo_plateau",
            "timestamp": time.time(),
        }

        # Mock dependencies to avoid side effects
        with patch.object(bridge, '_sync_weights'):
            bridge._on_curriculum_advancement_needed(mock_event)
            # Should not raise

    def test_handler_with_dict_event(self):
        """Handler accepts dict directly (without .payload attribute)."""
        bridge = MomentumToCurriculumBridge()

        # Event can be a dict directly
        event = {
            "config_key": "square8_4p",
            "reason": "elo_plateau",
            "timestamp": time.time(),
        }

        with patch.object(bridge, '_sync_weights'):
            bridge._on_curriculum_advancement_needed(event)
            # Should not raise

    def test_handler_skips_empty_config_key(self):
        """Handler returns early when config_key is missing."""
        bridge = MomentumToCurriculumBridge()

        mock_event = MagicMock()
        mock_event.payload = {"reason": "elo_plateau"}

        # Should not call _sync_weights for empty config
        with patch.object(bridge, '_sync_weights') as mock_sync:
            bridge._on_curriculum_advancement_needed(mock_event)
            mock_sync.assert_not_called()

    def test_handler_updates_last_sync_time(self):
        """Handler updates _last_sync_time on successful processing."""
        bridge = MomentumToCurriculumBridge()
        initial_time = bridge._last_sync_time

        mock_event = MagicMock()
        mock_event.payload = {
            "config_key": "hexagonal_3p",
            "reason": "elo_plateau",
            "timestamp": time.time(),
        }

        with patch.object(bridge, '_sync_weights'):
            bridge._on_curriculum_advancement_needed(mock_event)
            assert bridge._last_sync_time > initial_time

    def test_handler_calls_sync_weights(self):
        """Handler calls _sync_weights to propagate curriculum changes."""
        bridge = MomentumToCurriculumBridge()

        mock_event = MagicMock()
        mock_event.payload = {
            "config_key": "square19_2p",
            "reason": "elo_plateau",
            "timestamp": time.time(),
        }

        with patch.object(bridge, '_sync_weights') as mock_sync:
            bridge._on_curriculum_advancement_needed(mock_event)
            mock_sync.assert_called_once()

    def test_handler_with_config_alias(self):
        """Handler accepts 'config' as alias for 'config_key'."""
        bridge = MomentumToCurriculumBridge()

        mock_event = MagicMock()
        mock_event.payload = {
            "config": "hex8_4p",  # Uses 'config' instead of 'config_key'
            "reason": "stagnation",
        }

        with patch.object(bridge, '_sync_weights'):
            bridge._on_curriculum_advancement_needed(mock_event)
            # Should not raise

    def test_handler_graceful_failure_on_curriculum_import_error(self):
        """Handler handles ImportError when curriculum_feedback not available."""
        bridge = MomentumToCurriculumBridge()

        mock_event = MagicMock()
        mock_event.payload = {
            "config_key": "hex8_2p",
            "reason": "elo_plateau",
        }

        # Patch the curriculum import to raise ImportError
        # The import happens inside the handler, so we patch the module path
        with patch.object(bridge, '_sync_weights'):
            with patch.dict('sys.modules', {'app.training.curriculum_feedback': None}):
                # Should handle gracefully without raising
                bridge._on_curriculum_advancement_needed(mock_event)

    def test_handler_graceful_failure_on_attribute_error(self):
        """Handler handles AttributeError from malformed event."""
        bridge = MomentumToCurriculumBridge()

        # Create event that will cause AttributeError
        mock_event = None  # This will cause .payload access to fail

        # Should not raise, just log warning
        bridge._on_curriculum_advancement_needed(mock_event)

    def test_curriculum_advancement_needed_event_exists(self):
        """CURRICULUM_ADVANCEMENT_NEEDED event type exists in DataEventType."""
        from app.distributed.data_events import DataEventType

        assert hasattr(DataEventType, 'CURRICULUM_ADVANCEMENT_NEEDED')
        assert DataEventType.CURRICULUM_ADVANCEMENT_NEEDED.value == "curriculum_advancement_needed"

    def test_curriculum_advanced_event_exists(self):
        """CURRICULUM_ADVANCED event type exists in DataEventType."""
        from app.distributed.data_events import DataEventType

        assert hasattr(DataEventType, 'CURRICULUM_ADVANCED')
        assert DataEventType.CURRICULUM_ADVANCED.value == "curriculum_advanced"
