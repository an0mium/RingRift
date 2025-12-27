"""Integration tests for PROMOTION_FAILED → curriculum weight increase.

Tests the PromotionFailedToCurriculumWatcher class that increases curriculum
weights when model promotion fails, ensuring more diverse training data is
generated for the next training cycle.

Event flow:
1. Promotion fails (emits PROMOTION_FAILED)
2. PromotionFailedToCurriculumWatcher increases curriculum weight by 20%
3. Weight caps at 2.5x after multiple consecutive failures
4. Emits CURRICULUM_REBALANCED to notify downstream systems

December 2025 - Phase 3 integration tests
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.curriculum_integration import PromotionFailedToCurriculumWatcher


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def watcher():
    """Create a PromotionFailedToCurriculumWatcher instance."""
    return PromotionFailedToCurriculumWatcher()


@pytest.fixture
def mock_event_router():
    """Mock event router for subscription testing."""
    router = MagicMock()
    router.subscribe = MagicMock()
    router.unsubscribe = MagicMock()
    router.publish_sync = MagicMock()
    return router


@pytest.fixture
def mock_curriculum_feedback():
    """Mock curriculum feedback for weight updates."""
    feedback = MagicMock()
    feedback._current_weights = {}
    feedback.weight_min = 0.5
    feedback.weight_max = 2.5
    return feedback


@pytest.fixture
def mock_event():
    """Create a mock PROMOTION_FAILED event."""
    event = MagicMock()
    event.payload = {
        "config_key": "hex8_2p",
        "error": "gauntlet_failed",
        "model_id": "hex8_2p_v123",
    }
    return event


# =============================================================================
# Subscription Tests
# =============================================================================


class TestSubscription:
    """Tests for event subscription."""

    def test_subscribe_success(self, watcher, mock_event_router):
        """Test successful subscription to PROMOTION_FAILED."""
        # Mock the imports that happen inside the subscribe method
        with patch("app.coordination.event_router.get_router", return_value=mock_event_router):
            with patch("app.distributed.data_events.DataEventType") as mock_event_type:
                mock_event_type.PROMOTION_FAILED.value = "PROMOTION_FAILED"
                result = watcher.subscribe()

                assert result is True
                assert watcher._subscribed is True
                mock_event_router.subscribe.assert_called_once()

    def test_subscribe_already_subscribed(self, watcher, mock_event_router):
        """Test subscribing when already subscribed returns True."""
        watcher._subscribed = True
        result = watcher.subscribe()

        assert result is True
        # Should not call subscribe again
        mock_event_router.subscribe.assert_not_called()

    def test_subscribe_router_not_available(self, watcher):
        """Test subscription fails gracefully when router not available."""
        with patch("app.coordination.event_router.get_router", return_value=None):
            result = watcher.subscribe()

            assert result is False
            assert watcher._subscribed is False

    def test_subscribe_import_error(self, watcher):
        """Test subscription handles import errors."""
        with patch("app.coordination.event_router.get_router", side_effect=ImportError):
            result = watcher.subscribe()

            assert result is False
            assert watcher._subscribed is False

    def test_unsubscribe_success(self, watcher, mock_event_router):
        """Test successful unsubscription."""
        watcher._subscribed = True

        with patch("app.coordination.event_router.get_router", return_value=mock_event_router):
            with patch("app.distributed.data_events.DataEventType") as mock_event_type:
                mock_event_type.PROMOTION_FAILED.value = "PROMOTION_FAILED"
                watcher.unsubscribe()

                assert watcher._subscribed is False
                mock_event_router.unsubscribe.assert_called_once()


# =============================================================================
# Event Handling Tests
# =============================================================================


class TestEventHandling:
    """Tests for PROMOTION_FAILED event handling."""

    def test_handles_promotion_failed_first_failure(self, watcher, mock_event, mock_curriculum_feedback):
        """Test handling first promotion failure increases weight by 20%."""
        # Mock at the module where it's imported
        with patch("app.training.curriculum_feedback.get_curriculum_feedback", return_value=mock_curriculum_feedback):
            mock_curriculum_feedback._current_weights["hex8_2p"] = 1.0

            watcher._on_promotion_failed(mock_event)

            # Check weight increased by 20%
            assert "hex8_2p" in mock_curriculum_feedback._current_weights
            assert mock_curriculum_feedback._current_weights["hex8_2p"] == 1.2

            # Check failure count tracked
            assert watcher._failure_counts["hex8_2p"] == 1

    def test_handles_consecutive_failures(self, watcher, mock_event, mock_curriculum_feedback):
        """Test consecutive failures increase weight based on total failure count."""
        with patch("app.training.curriculum_feedback.get_curriculum_feedback", return_value=mock_curriculum_feedback):
            mock_curriculum_feedback._current_weights = {"hex8_2p": 1.0}
            mock_curriculum_feedback.weight_min = 0.5
            mock_curriculum_feedback.weight_max = 2.5

            # First failure: 1.0 * (1.0 + 1*0.2) = 1.0 * 1.2 = 1.2
            watcher._on_promotion_failed(mock_event)
            weight1 = mock_curriculum_feedback._current_weights["hex8_2p"]
            assert abs(weight1 - 1.2) < 0.01

            # Second failure: 1.2 * (1.0 + 2*0.2) = 1.2 * 1.4 = 1.68
            watcher._on_promotion_failed(mock_event)
            weight2 = mock_curriculum_feedback._current_weights["hex8_2p"]
            assert abs(weight2 - 1.68) < 0.01

            # Third failure: 1.68 * (1.0 + 3*0.2) = 1.68 * 1.6 = 2.688, capped at 2.5
            watcher._on_promotion_failed(mock_event)
            weight3 = mock_curriculum_feedback._current_weights["hex8_2p"]
            assert abs(weight3 - 2.5) < 0.01  # Should be capped at max

    def test_weight_caps_at_max(self, watcher, mock_event, mock_curriculum_feedback):
        """Test weight increase caps at 2.5x."""
        with patch("app.training.curriculum_feedback.get_curriculum_feedback", return_value=mock_curriculum_feedback):
            # Start near the cap
            mock_curriculum_feedback._current_weights["hex8_2p"] = 2.3

            # Should cap at 2.5
            watcher._on_promotion_failed(mock_event)
            assert mock_curriculum_feedback._current_weights["hex8_2p"] == 2.5

            # Further failures should not exceed cap
            watcher._on_promotion_failed(mock_event)
            assert mock_curriculum_feedback._current_weights["hex8_2p"] == 2.5

    def test_handles_missing_config_key(self, watcher):
        """Test gracefully handles event with missing config_key."""
        event = MagicMock()
        event.payload = {"error": "test"}

        # Should not raise
        watcher._on_promotion_failed(event)

        # No failures tracked
        assert len(watcher._failure_counts) == 0

    def test_tracks_failure_count_per_config(self, watcher, mock_curriculum_feedback):
        """Test failure counts tracked separately per config."""
        with patch("app.training.curriculum_feedback.get_curriculum_feedback", return_value=mock_curriculum_feedback):
            mock_curriculum_feedback._current_weights["hex8_2p"] = 1.0
            mock_curriculum_feedback._current_weights["square8_4p"] = 1.0

            # Fail hex8_2p twice
            event1 = MagicMock()
            event1.payload = {"config_key": "hex8_2p", "error": "test"}
            watcher._on_promotion_failed(event1)
            watcher._on_promotion_failed(event1)

            # Fail square8_4p once
            event2 = MagicMock()
            event2.payload = {"config_key": "square8_4p", "error": "test"}
            watcher._on_promotion_failed(event2)

            # Check separate counts
            assert watcher._failure_counts["hex8_2p"] == 2
            assert watcher._failure_counts["square8_4p"] == 1


# =============================================================================
# Curriculum Weight Update Tests
# =============================================================================


class TestCurriculumWeightUpdate:
    """Tests for curriculum weight update logic."""

    def test_increase_from_default_weight(self, watcher, mock_curriculum_feedback):
        """Test weight increase from default 1.0."""
        with patch("app.training.curriculum_feedback.get_curriculum_feedback", return_value=mock_curriculum_feedback):
            # Default weight
            mock_curriculum_feedback._current_weights["hex8_2p"] = 1.0

            watcher._increase_curriculum_weight("hex8_2p", 1, "test_error")

            # Should be 1.2x (20% increase)
            assert mock_curriculum_feedback._current_weights["hex8_2p"] == 1.2

    def test_respects_weight_min(self, watcher, mock_curriculum_feedback):
        """Test respects weight_min boundary."""
        with patch("app.training.curriculum_feedback.get_curriculum_feedback", return_value=mock_curriculum_feedback):
            # Start at minimum
            mock_curriculum_feedback._current_weights["hex8_2p"] = 0.5

            watcher._increase_curriculum_weight("hex8_2p", 1, "test_error")

            # Should increase from min
            assert mock_curriculum_feedback._current_weights["hex8_2p"] == 0.6

    def test_respects_weight_max(self, watcher, mock_curriculum_feedback):
        """Test respects weight_max boundary."""
        with patch("app.training.curriculum_feedback.get_curriculum_feedback", return_value=mock_curriculum_feedback):
            # Start very high
            mock_curriculum_feedback._current_weights["hex8_2p"] = 2.4

            # 5 failures would normally be 2.4 * 1.2^5 = 5.97
            watcher._increase_curriculum_weight("hex8_2p", 5, "test_error")

            # Should cap at 2.5
            assert mock_curriculum_feedback._current_weights["hex8_2p"] == 2.5

    def test_handles_missing_weight(self, watcher, mock_curriculum_feedback):
        """Test handles config with no existing weight."""
        with patch("app.training.curriculum_feedback.get_curriculum_feedback", return_value=mock_curriculum_feedback):
            # No existing weight
            watcher._increase_curriculum_weight("new_config", 1, "test_error")

            # Should create new weight at 1.2
            assert mock_curriculum_feedback._current_weights["new_config"] == 1.2


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEventEmission:
    """Tests for CURRICULUM_REBALANCED event emission."""

    def test_emits_rebalance_event(self, watcher, mock_event_router, mock_curriculum_feedback):
        """Test emits CURRICULUM_REBALANCED after weight increase."""
        with patch("app.coordination.event_router.get_router", return_value=mock_event_router):
            with patch("app.training.curriculum_feedback.get_curriculum_feedback", return_value=mock_curriculum_feedback):
                mock_curriculum_feedback._current_weights["hex8_2p"] = 1.2

                watcher._increase_curriculum_weight("hex8_2p", 1, "test_error")

                # Check event emitted
                mock_event_router.publish_sync.assert_called_once()
                call_args = mock_event_router.publish_sync.call_args

                # Verify event type
                assert call_args[0][0] == "CURRICULUM_REBALANCED"

                # Verify payload
                payload = call_args[0][1]
                assert payload["trigger"] == "promotion_failed"
                assert payload["changed_configs"] == ["hex8_2p"]
                assert "hex8_2p" in payload["new_weights"]
                assert payload["failure_count"] == 1

    def test_event_emission_failure_handled(self, watcher, mock_curriculum_feedback):
        """Test gracefully handles event emission failures."""
        with patch("app.coordination.event_router.get_router", side_effect=ImportError):
            with patch("app.training.curriculum_feedback.get_curriculum_feedback", return_value=mock_curriculum_feedback):
                mock_curriculum_feedback._current_weights["hex8_2p"] = 1.0

                # Should not raise despite emission failure
                watcher._increase_curriculum_weight("hex8_2p", 1, "test_error")

                # Weight still updated
                assert mock_curriculum_feedback._current_weights["hex8_2p"] == 1.2


# =============================================================================
# Failure Count Management Tests
# =============================================================================


class TestFailureCountManagement:
    """Tests for failure count tracking and reset."""

    def test_get_failure_counts(self, watcher):
        """Test getting current failure counts."""
        watcher._failure_counts = {"hex8_2p": 3, "square8_4p": 1}

        counts = watcher.get_failure_counts()

        assert counts == {"hex8_2p": 3, "square8_4p": 1}
        # Should return a copy
        assert counts is not watcher._failure_counts

    def test_reset_failure_count(self, watcher):
        """Test resetting failure count for a config."""
        watcher._failure_counts = {"hex8_2p": 5, "square8_4p": 2}

        watcher.reset_failure_count("hex8_2p")

        assert "hex8_2p" not in watcher._failure_counts
        assert watcher._failure_counts["square8_4p"] == 2

    def test_reset_nonexistent_count(self, watcher):
        """Test resetting count for config without failures."""
        # Should not raise
        watcher.reset_failure_count("nonexistent_config")


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_malformed_event(self, watcher):
        """Test handles event with malformed payload."""
        event = MagicMock()
        event.payload = None

        # Should not raise
        watcher._on_promotion_failed(event)

    def test_handles_curriculum_import_error(self, watcher, mock_event):
        """Test handles curriculum_feedback import failure."""
        with patch("app.training.curriculum_feedback.get_curriculum_feedback", side_effect=ImportError):
            # Should not raise
            watcher._on_promotion_failed(mock_event)

    def test_handles_attribute_error(self, watcher):
        """Test handles event missing payload attribute."""
        event = MagicMock(spec=[])  # No payload attribute

        # Should not raise
        watcher._on_promotion_failed(event)

    def test_handles_type_error(self, watcher, mock_curriculum_feedback):
        """Test handles type errors in weight calculation."""
        with patch("app.training.curriculum_feedback.get_curriculum_feedback", return_value=mock_curriculum_feedback):
            # Set weight to invalid type
            mock_curriculum_feedback._current_weights["hex8_2p"] = "invalid"

            # Should handle gracefully
            watcher._increase_curriculum_weight("hex8_2p", 1, "test")
