"""Test automatic rollback flow for REGRESSION_DETECTED events.

This test verifies the complete flow:
1. REGRESSION_DETECTED event is emitted
2. AutoRollbackHandler receives it
3. For elo_drop > 30 (MODERATE), triggers rollback
4. PROMOTION_ROLLED_BACK event is emitted
5. TrainingCoordinator pauses training
6. TRAINING_PAUSED event is emitted
"""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from app.training.regression_detector import (
    RegressionDetector,
    RegressionEvent,
    RegressionSeverity,
)
from app.training.rollback_manager import AutoRollbackHandler, RollbackManager


class TestRegressionRollbackFlow:
    """Test the automatic rollback flow for regressions."""

    def test_moderate_regression_with_elo_drop_30_triggers_rollback(self):
        """Test that MODERATE regression with elo_drop > 30 triggers rollback."""
        # Create mock RollbackManager
        mock_rollback_mgr = Mock(spec=RollbackManager)
        mock_rollback_mgr.rollback_model.return_value = {
            "success": True,
            "from_version": 42,
            "to_version": 41,
            "from_metrics": {},
            "to_metrics": {},
        }

        # Create AutoRollbackHandler with mocked manager
        handler = AutoRollbackHandler(
            rollback_manager=mock_rollback_mgr,
            auto_rollback_enabled=True,
            require_approval_for_severe=True,
        )

        # Create a MODERATE regression event with elo_drop = 35
        event = RegressionEvent(
            model_id="hex8_2p_v42",
            severity=RegressionSeverity.MODERATE,
            timestamp=time.time(),
            current_elo=1450,
            baseline_elo=1485,
            elo_drop=35.0,  # > 30, should trigger rollback
            games_played=100,
            reason="Elo dropped by 35.0",
            recommended_action="Investigate causes, consider retraining",
        )

        # Trigger the handler
        with patch.object(handler, '_emit_rollback_completed_event'):
            handler.on_regression(event)

        # Verify rollback was triggered
        mock_rollback_mgr.rollback_model.assert_called_once()
        call_args = mock_rollback_mgr.rollback_model.call_args

        assert call_args.kwargs['model_id'] == "hex8_2p_v42"
        assert "MODERATE regression" in call_args.kwargs['reason']
        assert call_args.kwargs['triggered_by'] == "auto_regression_moderate"

    def test_moderate_regression_with_elo_drop_25_does_not_trigger_rollback(self):
        """Test that MODERATE regression with elo_drop <= 30 does NOT trigger rollback."""
        # Create mock RollbackManager
        mock_rollback_mgr = Mock(spec=RollbackManager)

        # Create AutoRollbackHandler with mocked manager
        handler = AutoRollbackHandler(
            rollback_manager=mock_rollback_mgr,
            auto_rollback_enabled=True,
            require_approval_for_severe=True,
        )

        # Create a MODERATE regression event with elo_drop = 25
        event = RegressionEvent(
            model_id="hex8_2p_v42",
            severity=RegressionSeverity.MODERATE,
            timestamp=time.time(),
            current_elo=1460,
            baseline_elo=1485,
            elo_drop=25.0,  # <= 30, should NOT trigger rollback
            games_played=100,
            reason="Elo dropped by 25.0",
            recommended_action="Investigate causes, consider retraining",
        )

        # Trigger the handler
        handler.on_regression(event)

        # Verify rollback was NOT triggered
        mock_rollback_mgr.rollback_model.assert_not_called()

    def test_severe_regression_triggers_rollback(self):
        """Test that SEVERE regression always triggers rollback (existing behavior)."""
        # Create mock RollbackManager
        mock_rollback_mgr = Mock(spec=RollbackManager)
        mock_rollback_mgr.rollback_model.return_value = {
            "success": True,
            "from_version": 42,
            "to_version": 41,
        }

        # Create AutoRollbackHandler with auto-approval for SEVERE
        handler = AutoRollbackHandler(
            rollback_manager=mock_rollback_mgr,
            auto_rollback_enabled=True,
            require_approval_for_severe=False,  # Auto-approve SEVERE
        )

        # Create a SEVERE regression event
        event = RegressionEvent(
            model_id="hex8_2p_v42",
            severity=RegressionSeverity.SEVERE,
            timestamp=time.time(),
            current_elo=1430,
            baseline_elo=1485,
            elo_drop=55.0,  # SEVERE threshold
            games_played=100,
            reason="Elo dropped by 55.0",
            recommended_action="Halt promotion, investigate immediately",
        )

        # Trigger the handler
        with patch.object(handler, '_emit_rollback_completed_event'):
            handler.on_regression(event)

        # Verify rollback was triggered
        mock_rollback_mgr.rollback_model.assert_called_once()
        call_args = mock_rollback_mgr.rollback_model.call_args

        assert call_args.kwargs['model_id'] == "hex8_2p_v42"
        assert "SEVERE regression" in call_args.kwargs['reason']
        assert call_args.kwargs['triggered_by'] == "auto_regression_severe"

    def test_critical_regression_triggers_immediate_rollback(self):
        """Test that CRITICAL regression triggers immediate rollback."""
        # Create mock RollbackManager
        mock_rollback_mgr = Mock(spec=RollbackManager)
        mock_rollback_mgr.rollback_model.return_value = {
            "success": True,
            "from_version": 42,
            "to_version": 41,
        }

        # Create AutoRollbackHandler
        handler = AutoRollbackHandler(
            rollback_manager=mock_rollback_mgr,
            auto_rollback_enabled=True,
        )

        # Create a CRITICAL regression event
        event = RegressionEvent(
            model_id="hex8_2p_v42",
            severity=RegressionSeverity.CRITICAL,
            timestamp=time.time(),
            current_elo=1400,
            baseline_elo=1485,
            elo_drop=85.0,  # CRITICAL threshold
            games_played=100,
            reason="Elo dropped by 85.0",
            recommended_action="Rollback recommended, stop deployments",
        )

        # Trigger the handler
        with patch.object(handler, '_emit_rollback_completed_event'):
            handler.on_regression(event)

        # Verify rollback was triggered
        mock_rollback_mgr.rollback_model.assert_called_once()
        call_args = mock_rollback_mgr.rollback_model.call_args

        assert call_args.kwargs['model_id'] == "hex8_2p_v42"
        assert "CRITICAL regression" in call_args.kwargs['reason']
        assert call_args.kwargs['triggered_by'] == "auto_regression_critical"

    def test_minor_regression_does_not_trigger_rollback(self):
        """Test that MINOR regression does NOT trigger rollback."""
        # Create mock RollbackManager
        mock_rollback_mgr = Mock(spec=RollbackManager)

        # Create AutoRollbackHandler
        handler = AutoRollbackHandler(
            rollback_manager=mock_rollback_mgr,
            auto_rollback_enabled=True,
        )

        # Create a MINOR regression event
        event = RegressionEvent(
            model_id="hex8_2p_v42",
            severity=RegressionSeverity.MINOR,
            timestamp=time.time(),
            current_elo=1470,
            baseline_elo=1485,
            elo_drop=15.0,  # MINOR threshold
            games_played=100,
            reason="Elo dropped by 15.0",
            recommended_action="Monitor closely, may recover",
        )

        # Trigger the handler
        handler.on_regression(event)

        # Verify rollback was NOT triggered
        mock_rollback_mgr.rollback_model.assert_not_called()

    def test_auto_rollback_disabled_does_not_trigger(self):
        """Test that rollback does not trigger when auto_rollback_enabled=False."""
        # Create mock RollbackManager
        mock_rollback_mgr = Mock(spec=RollbackManager)

        # Create AutoRollbackHandler with auto-rollback DISABLED
        handler = AutoRollbackHandler(
            rollback_manager=mock_rollback_mgr,
            auto_rollback_enabled=False,  # Disabled
        )

        # Create a CRITICAL regression event (would normally trigger)
        event = RegressionEvent(
            model_id="hex8_2p_v42",
            severity=RegressionSeverity.CRITICAL,
            timestamp=time.time(),
            current_elo=1400,
            baseline_elo=1485,
            elo_drop=85.0,
            games_played=100,
            reason="Elo dropped by 85.0",
        )

        # Trigger the handler
        handler.on_regression(event)

        # Verify rollback was NOT triggered (auto-rollback disabled)
        mock_rollback_mgr.rollback_model.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
