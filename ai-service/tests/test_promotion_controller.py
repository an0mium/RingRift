"""Unit tests for the Promotion Controller.

Tests cover:
- PromotionType enum values
- PromotionCriteria defaults and customization
- PromotionDecision creation and properties
- PromotionController instantiation and singleton pattern
- GraduatedResponseAction logic
- RollbackMonitor daily count tracking
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


class TestPromotionTypeEnum:
    """Test PromotionType enum values."""

    def test_promotion_type_values(self):
        """Test that PromotionType has expected values."""
        from app.training.promotion_controller import PromotionType

        assert PromotionType.STAGING.value == "staging"
        assert PromotionType.PRODUCTION.value == "production"
        assert PromotionType.TIER.value == "tier"
        assert PromotionType.CHAMPION.value == "champion"
        assert PromotionType.ROLLBACK.value == "rollback"

    def test_promotion_type_iteration(self):
        """Test that all PromotionType values can be iterated."""
        from app.training.promotion_controller import PromotionType

        types = list(PromotionType)
        assert len(types) == 5


class TestPromotionCriteria:
    """Test PromotionCriteria dataclass."""

    def test_default_criteria(self):
        """Test default promotion criteria values."""
        from app.training.promotion_controller import PromotionCriteria

        criteria = PromotionCriteria()

        assert criteria.min_elo_improvement == 25.0
        assert criteria.min_games_played == 50
        assert criteria.min_win_rate == 0.52
        assert criteria.max_value_mse_degradation == 0.05
        assert criteria.confidence_threshold == 0.95
        assert criteria.tier_elo_threshold is None

    def test_custom_criteria(self):
        """Test custom promotion criteria."""
        from app.training.promotion_controller import PromotionCriteria

        criteria = PromotionCriteria(
            min_elo_improvement=50.0,
            min_games_played=100,
            min_win_rate=0.55,
            tier_elo_threshold=1800.0,
        )

        assert criteria.min_elo_improvement == 50.0
        assert criteria.min_games_played == 100
        assert criteria.min_win_rate == 0.55
        assert criteria.tier_elo_threshold == 1800.0


class TestPromotionDecision:
    """Test PromotionDecision dataclass."""

    def test_decision_creation(self):
        """Test creating a promotion decision."""
        from app.training.promotion_controller import (
            PromotionDecision,
            PromotionType,
        )

        decision = PromotionDecision(
            model_id="test_model_v1",
            promotion_type=PromotionType.PRODUCTION,
            should_promote=True,
            confidence=0.98,
            reason="Elo improved by 50 points over 100 games",
        )

        assert decision.model_id == "test_model_v1"
        assert decision.promotion_type == PromotionType.PRODUCTION
        assert decision.should_promote is True
        assert decision.confidence == 0.98
        assert "Elo improved" in decision.reason

    def test_decision_rejection(self):
        """Test creating a rejection decision."""
        from app.training.promotion_controller import (
            PromotionDecision,
            PromotionType,
        )

        decision = PromotionDecision(
            model_id="test_model_v2",
            promotion_type=PromotionType.STAGING,
            should_promote=False,
            confidence=0.45,
            reason="Insufficient games played (25 < 50)",
        )

        assert decision.should_promote is False
        assert decision.confidence == 0.45


class TestPromotionController:
    """Test PromotionController class."""

    def test_controller_instantiation(self):
        """Test that PromotionController can be instantiated."""
        from app.training.promotion_controller import PromotionController

        controller = PromotionController()
        assert controller is not None

    def test_singleton_pattern(self):
        """Test that get_promotion_controller returns singleton."""
        from app.training.promotion_controller import get_promotion_controller

        controller1 = get_promotion_controller()
        controller2 = get_promotion_controller()

        # Both calls should return the same instance
        assert controller1 is controller2

    def test_pending_promotion_checks(self):
        """Test getting pending promotion checks."""
        from app.training.promotion_controller import PromotionController

        controller = PromotionController()
        pending = controller.get_pending_promotion_checks()

        # Should return a dict (may be empty)
        assert isinstance(pending, dict)


class TestGraduatedResponseAction:
    """Test GraduatedResponseAction enum and logic."""

    def test_action_values(self):
        """Test GraduatedResponseAction enum values."""
        from app.training.promotion_controller import GraduatedResponseAction

        assert GraduatedResponseAction.NOTIFY.value == "notify"
        assert GraduatedResponseAction.SLOW_DOWN.value == "slow_down"
        assert GraduatedResponseAction.INVESTIGATE.value == "investigate"
        assert GraduatedResponseAction.PAUSE_TRAINING.value == "pause_training"
        assert GraduatedResponseAction.ESCALATE_HUMAN.value == "escalate_human"

    def test_action_iteration(self):
        """Test that all actions can be iterated."""
        from app.training.promotion_controller import GraduatedResponseAction

        actions = list(GraduatedResponseAction)
        assert len(actions) == 5


class TestRollbackEvent:
    """Test RollbackEvent dataclass."""

    def test_rollback_event_creation(self):
        """Test creating a rollback event."""
        from app.training.promotion_controller import RollbackEvent

        event = RollbackEvent(
            triggered_at=datetime.now().isoformat(),
            current_model_id="model_v5",
            rollback_model_id="model_v4",
            reason="Elo dropped by 100 points",
            elo_regression=-50.0,
            games_played=30,
        )

        assert event.current_model_id == "model_v5"
        assert event.rollback_model_id == "model_v4"
        assert "Elo dropped" in event.reason
        assert event.elo_regression == -50.0

    def test_rollback_event_to_dict(self):
        """Test RollbackEvent serialization."""
        from app.training.promotion_controller import RollbackEvent

        event = RollbackEvent(
            triggered_at="2025-12-19T12:00:00",
            current_model_id="model_v5",
            rollback_model_id="model_v4",
            reason="Regression detected",
        )

        data = event.to_dict()
        assert data["current_model_id"] == "model_v5"
        assert data["rollback_model_id"] == "model_v4"


class TestRollbackMonitor:
    """Test RollbackMonitor class."""

    def test_monitor_instantiation(self):
        """Test that RollbackMonitor can be instantiated."""
        from app.training.promotion_controller import RollbackMonitor

        monitor = RollbackMonitor()
        assert monitor is not None

    def test_daily_rollback_count_initial(self):
        """Test initial daily rollback count is zero."""
        from app.training.promotion_controller import RollbackMonitor

        monitor = RollbackMonitor()
        count = monitor.get_daily_rollback_count()

        assert count == 0

    def test_rollback_history_initial(self):
        """Test initial rollback history is empty."""
        from app.training.promotion_controller import RollbackMonitor

        monitor = RollbackMonitor()
        history = monitor.get_rollback_history()

        assert isinstance(history, list)
        assert len(history) == 0

    def test_get_rollback_monitor_factory(self):
        """Test that get_rollback_monitor is a factory function."""
        from app.training.promotion_controller import get_rollback_monitor

        monitor1 = get_rollback_monitor()
        monitor2 = get_rollback_monitor()

        # Factory returns new instances
        assert monitor1 is not monitor2
        # But both are valid RollbackMonitor instances
        assert type(monitor1).__name__ == "RollbackMonitor"
        assert type(monitor2).__name__ == "RollbackMonitor"


class TestABTestManager:
    """Test ABTestManager class."""

    def test_manager_instantiation(self):
        """Test that ABTestManager can be instantiated."""
        from app.training.promotion_controller import ABTestManager

        manager = ABTestManager()
        assert manager is not None

    def test_get_ab_test_manager_factory(self):
        """Test that get_ab_test_manager is a factory function."""
        from app.training.promotion_controller import get_ab_test_manager

        manager1 = get_ab_test_manager()
        manager2 = get_ab_test_manager()

        # Factory returns new instances
        assert manager1 is not manager2
        # But both are valid ABTestManager instances
        assert type(manager1).__name__ == "ABTestManager"
        assert type(manager2).__name__ == "ABTestManager"

    def test_get_test_status_nonexistent(self):
        """Test getting status of nonexistent test."""
        from app.training.promotion_controller import ABTestManager

        manager = ABTestManager()
        status = manager.get_test_status("nonexistent_test")

        assert status is None


class TestModuleImports:
    """Test that all public exports are importable."""

    def test_main_exports(self):
        """Test importing main exports from promotion_controller."""
        from app.training.promotion_controller import (
            PromotionController,
            PromotionType,
            PromotionCriteria,
            PromotionDecision,
            get_promotion_controller,
        )

        assert PromotionController is not None
        assert PromotionType is not None
        assert PromotionCriteria is not None
        assert PromotionDecision is not None
        assert callable(get_promotion_controller)

    def test_rollback_exports(self):
        """Test importing rollback-related exports."""
        from app.training.promotion_controller import (
            RollbackMonitor,
            RollbackEvent,
            GraduatedResponseAction,
            get_rollback_monitor,
        )

        assert RollbackMonitor is not None
        assert RollbackEvent is not None
        assert GraduatedResponseAction is not None
        assert callable(get_rollback_monitor)

    def test_ab_test_exports(self):
        """Test importing A/B test exports."""
        from app.training.promotion_controller import (
            ABTestManager,
            get_ab_test_manager,
        )

        assert ABTestManager is not None
        assert callable(get_ab_test_manager)
