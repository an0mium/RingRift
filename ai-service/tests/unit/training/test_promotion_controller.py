"""Tests for Promotion Controller.

Tests the unified promotion controller, rollback monitor, and A/B test manager.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from app.training.promotion_controller import (
    ABTestConfig,
    ABTestManager,
    ABTestResult,
    LoggingNotificationHook,
    NotificationHook,
    PromotionController,
    PromotionCriteria,
    PromotionDecision,
    PromotionType,
    RollbackCriteria,
    RollbackEvent,
    RollbackMonitor,
    get_ab_test_manager,
    get_promotion_controller,
    get_rollback_monitor,
)


class TestPromotionCriteria:
    """Tests for PromotionCriteria dataclass."""

    def test_default_values(self):
        """Should have sensible default values."""
        criteria = PromotionCriteria()
        assert criteria.min_elo_improvement == 25.0
        assert criteria.min_games_played == 100
        assert criteria.min_win_rate == 0.52
        assert criteria.max_value_mse_degradation == 0.05
        assert criteria.confidence_threshold == 0.95

    def test_custom_values(self):
        """Should accept custom values."""
        criteria = PromotionCriteria(
            min_elo_improvement=50.0,
            min_games_played=100,
        )
        assert criteria.min_elo_improvement == 50.0
        assert criteria.min_games_played == 100


class TestPromotionDecision:
    """Tests for PromotionDecision dataclass."""

    def test_creation(self):
        """Should create decision with required fields."""
        decision = PromotionDecision(
            model_id="model_v1",
            promotion_type=PromotionType.PRODUCTION,
            should_promote=True,
            reason="Passed all criteria",
        )
        assert decision.model_id == "model_v1"
        assert decision.promotion_type == PromotionType.PRODUCTION
        assert decision.should_promote is True

    def test_to_dict(self):
        """Should convert to dictionary."""
        decision = PromotionDecision(
            model_id="model_v1",
            promotion_type=PromotionType.STAGING,
            should_promote=False,
            reason="Insufficient games",
            games_played=30,
        )
        d = decision.to_dict()
        assert d["model_id"] == "model_v1"
        assert d["promotion_type"] == "staging"
        assert d["should_promote"] is False
        assert d["games_played"] == 30

    def test_evaluated_at_defaults(self):
        """Should have evaluated_at timestamp."""
        decision = PromotionDecision(
            model_id="test",
            promotion_type=PromotionType.PRODUCTION,
            should_promote=True,
            reason="test",
        )
        assert decision.evaluated_at is not None


class TestPromotionController:
    """Tests for PromotionController class."""

    def test_initialization(self):
        """Should initialize with defaults."""
        controller = PromotionController()
        assert controller.criteria is not None
        assert controller.criteria.min_elo_improvement == 25.0

    def test_initialization_with_custom_criteria(self):
        """Should accept custom criteria."""
        criteria = PromotionCriteria(min_elo_improvement=50.0)
        controller = PromotionController(criteria=criteria)
        assert controller.criteria.min_elo_improvement == 50.0

    def test_evaluate_promotion_insufficient_games(self):
        """Should reject when insufficient games."""
        controller = PromotionController()
        # Mock elo service to return low game count
        mock_elo = MagicMock()
        mock_rating = MagicMock()
        mock_rating.rating = 1500
        mock_rating.games_played = 10  # Below min_games_played
        mock_rating.win_rate = 0.6
        mock_elo.get_rating.return_value = mock_rating
        controller._elo_service = mock_elo

        decision = controller.evaluate_promotion(
            model_id="test_model",
            promotion_type=PromotionType.PRODUCTION,
        )

        assert decision.should_promote is False
        assert "Insufficient games" in decision.reason

    def test_evaluate_rollback(self):
        """Should evaluate rollback condition."""
        controller = PromotionController()

        # No elo service should result in no rollback
        decision = controller.evaluate_promotion(
            model_id="test_model",
            promotion_type=PromotionType.ROLLBACK,
        )

        assert decision.promotion_type == PromotionType.ROLLBACK
        assert decision.should_promote is False

    def test_execute_promotion_dry_run(self):
        """Should support dry run mode."""
        controller = PromotionController()
        decision = PromotionDecision(
            model_id="test",
            promotion_type=PromotionType.PRODUCTION,
            should_promote=True,
            reason="test",
        )

        result = controller.execute_promotion(decision, dry_run=True)
        assert result is True

    def test_execute_promotion_should_not_promote(self):
        """Should skip when should_promote is False."""
        controller = PromotionController()
        decision = PromotionDecision(
            model_id="test",
            promotion_type=PromotionType.PRODUCTION,
            should_promote=False,
            reason="Did not meet criteria",
        )

        result = controller.execute_promotion(decision)
        assert result is False


class TestRollbackCriteria:
    """Tests for RollbackCriteria dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        criteria = RollbackCriteria()
        assert criteria.elo_regression_threshold == -30.0
        assert criteria.min_games_for_regression == 20
        assert criteria.consecutive_checks_required == 3
        assert criteria.min_win_rate == 0.40
        assert criteria.cooldown_seconds == 3600
        assert criteria.max_rollbacks_per_day == 3


class TestRollbackEvent:
    """Tests for RollbackEvent dataclass."""

    def test_creation(self):
        """Should create rollback event."""
        event = RollbackEvent(
            triggered_at="2025-01-01T00:00:00",
            current_model_id="model_v2",
            rollback_model_id="model_v1",
            reason="Elo regression",
        )
        assert event.current_model_id == "model_v2"
        assert event.rollback_model_id == "model_v1"

    def test_to_dict(self):
        """Should convert to dictionary."""
        event = RollbackEvent(
            triggered_at="2025-01-01T00:00:00",
            current_model_id="model_v2",
            rollback_model_id="model_v1",
            reason="Test",
            elo_regression=-40.0,
        )
        d = event.to_dict()
        assert d["current_model_id"] == "model_v2"
        assert d["elo_regression"] == -40.0


class TestRollbackMonitor:
    """Tests for RollbackMonitor class."""

    def test_initialization(self):
        """Should initialize with defaults."""
        monitor = RollbackMonitor()
        assert monitor.criteria is not None
        assert monitor.criteria.elo_regression_threshold == -30.0

    def test_initialization_with_custom_criteria(self):
        """Should accept custom criteria."""
        criteria = RollbackCriteria(elo_regression_threshold=-50.0)
        monitor = RollbackMonitor(criteria=criteria)
        assert monitor.criteria.elo_regression_threshold == -50.0

    def test_cooldown_initially_inactive(self):
        """Should have no cooldown initially."""
        monitor = RollbackMonitor()
        active, remaining = monitor.is_cooldown_active("square8", 2)
        assert active is False
        assert remaining is None

    def test_set_cooldown_bypass(self):
        """Should support cooldown bypass."""
        monitor = RollbackMonitor()
        monitor.set_cooldown_bypass(True)
        # After bypass, cooldown should not be active even if recorded
        monitor._last_rollback_time["square8_2p"] = datetime.now()
        active, _ = monitor.is_cooldown_active("square8", 2)
        assert active is False

    def test_daily_rollback_count_empty(self):
        """Should return 0 when no rollbacks."""
        monitor = RollbackMonitor()
        count = monitor.get_daily_rollback_count()
        assert count == 0

    def test_max_rollbacks_not_reached(self):
        """Should not be reached with no rollbacks."""
        monitor = RollbackMonitor()
        reached, count = monitor.is_max_daily_rollbacks_reached()
        assert reached is False
        assert count == 0

    def test_add_notification_hook(self):
        """Should add notification hooks."""
        monitor = RollbackMonitor()
        hook = LoggingNotificationHook()
        monitor.add_notification_hook(hook)
        assert len(monitor._hooks) == 1

    def test_get_regression_status_no_history(self):
        """Should return empty status for unknown model."""
        monitor = RollbackMonitor()
        status = monitor.get_regression_status("unknown_model")
        assert status["checks"] == 0
        assert status["at_risk"] is False

    def test_check_for_regression_no_elo_service(self):
        """Should return False when no Elo service."""
        monitor = RollbackMonitor()
        should_rollback, event = monitor.check_for_regression(
            model_id="test",
            previous_model_id="prev",
        )
        assert should_rollback is False
        assert event is None

    def test_execute_rollback_dry_run(self):
        """Should support dry run rollback."""
        monitor = RollbackMonitor()
        event = RollbackEvent(
            triggered_at="2025-01-01T00:00:00",
            current_model_id="v2",
            rollback_model_id="v1",
            reason="Test",
        )
        result = monitor.execute_rollback(event, dry_run=True)
        assert result is True

    def test_get_rollback_history_empty(self):
        """Should return empty history initially."""
        monitor = RollbackMonitor()
        history = monitor.get_rollback_history()
        assert history == []


class TestNotificationHooks:
    """Tests for notification hook classes."""

    def test_base_hook_methods_exist(self):
        """Base hook should have all methods."""
        hook = NotificationHook()
        # These should not raise
        hook.on_regression_detected("test", {})
        hook.on_at_risk("test", {})
        hook.on_rollback_triggered(MagicMock())
        hook.on_rollback_completed(MagicMock(), True)

    def test_logging_hook_initialization(self):
        """Should initialize logging hook."""
        hook = LoggingNotificationHook()
        assert hook.logger is not None

    def test_logging_hook_methods(self):
        """Logging hook methods should not raise."""
        hook = LoggingNotificationHook()
        hook.on_regression_detected("model", {"consecutive_regressions": 1})
        hook.on_at_risk("model", {"consecutive_regressions": 2})

        event = RollbackEvent(
            triggered_at="2025-01-01",
            current_model_id="v2",
            rollback_model_id="v1",
            reason="test",
        )
        hook.on_rollback_triggered(event)
        hook.on_rollback_completed(event, True)
        hook.on_rollback_completed(event, False)


class TestABTestConfig:
    """Tests for ABTestConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = ABTestConfig(
            test_id="test_1",
            control_model_id="v1",
            treatment_model_id="v2",
        )
        assert config.traffic_split == 0.5
        assert config.min_games_per_variant == 100
        assert config.significance_threshold == 0.95
        assert config.auto_promote is False

    def test_to_dict(self):
        """Should convert to dictionary."""
        config = ABTestConfig(
            test_id="test_1",
            control_model_id="v1",
            treatment_model_id="v2",
            board_type="hex8",
        )
        d = config.to_dict()
        assert d["test_id"] == "test_1"
        assert d["board_type"] == "hex8"


class TestABTestResult:
    """Tests for ABTestResult dataclass."""

    def test_creation(self):
        """Should create result with required fields."""
        result = ABTestResult(
            test_id="test_1",
            control_elo=1500,
            treatment_elo=1550,
            elo_difference=50,
            control_games=100,
            treatment_games=100,
            control_win_rate=0.5,
            treatment_win_rate=0.55,
            is_significant=True,
            winner="treatment",
            confidence=0.98,
            recommendation="Promote treatment",
        )
        assert result.winner == "treatment"
        assert result.is_significant is True

    def test_to_dict(self):
        """Should convert to dictionary."""
        result = ABTestResult(
            test_id="test_1",
            control_elo=1500,
            treatment_elo=1520,
            elo_difference=20,
            control_games=50,
            treatment_games=50,
            control_win_rate=0.5,
            treatment_win_rate=0.52,
            is_significant=False,
            winner=None,
            confidence=0.8,
            recommendation="Need more games",
        )
        d = result.to_dict()
        assert d["elo_difference"] == 20
        assert d["winner"] is None


class TestABTestManager:
    """Tests for ABTestManager class."""

    def test_initialization(self):
        """Should initialize correctly."""
        manager = ABTestManager()
        assert len(manager._active_tests) == 0
        assert len(manager._completed_tests) == 0

    def test_start_test(self):
        """Should start a new test."""
        manager = ABTestManager()
        config = ABTestConfig(
            test_id="test_1",
            control_model_id="v1",
            treatment_model_id="v2",
        )
        result = manager.start_test(config)
        assert result is True
        assert "test_1" in manager._active_tests

    def test_start_duplicate_test(self):
        """Should reject duplicate test."""
        manager = ABTestManager()
        config = ABTestConfig(
            test_id="test_1",
            control_model_id="v1",
            treatment_model_id="v2",
        )
        manager.start_test(config)
        result = manager.start_test(config)
        assert result is False

    def test_start_test_invalid_split(self):
        """Should reject invalid traffic split."""
        manager = ABTestManager()
        config = ABTestConfig(
            test_id="test_1",
            control_model_id="v1",
            treatment_model_id="v2",
            traffic_split=1.5,  # Invalid
        )
        result = manager.start_test(config)
        assert result is False

    def test_get_model_for_game(self):
        """Should return model based on traffic split."""
        manager = ABTestManager()
        config = ABTestConfig(
            test_id="test_1",
            control_model_id="v1",
            treatment_model_id="v2",
            traffic_split=0.5,
        )
        manager.start_test(config)

        # Run multiple times to check randomness
        results = set()
        for _ in range(100):
            model = manager.get_model_for_game("test_1")
            results.add(model)

        # Should have both models with 50/50 split
        assert "v1" in results
        assert "v2" in results

    def test_get_model_for_unknown_test(self):
        """Should return None for unknown test."""
        manager = ABTestManager()
        model = manager.get_model_for_game("unknown")
        assert model is None

    def test_get_all_test_models(self):
        """Should return both models."""
        manager = ABTestManager()
        config = ABTestConfig(
            test_id="test_1",
            control_model_id="v1",
            treatment_model_id="v2",
        )
        manager.start_test(config)

        control, treatment = manager.get_all_test_models("test_1")
        assert control == "v1"
        assert treatment == "v2"

    def test_list_active_tests(self):
        """Should list active tests."""
        manager = ABTestManager()
        config = ABTestConfig(
            test_id="test_1",
            control_model_id="v1",
            treatment_model_id="v2",
        )
        manager.start_test(config)

        active = manager.list_active_tests()
        assert len(active) == 1
        assert active[0].test_id == "test_1"

    def test_stop_test(self):
        """Should stop and remove test."""
        manager = ABTestManager()
        config = ABTestConfig(
            test_id="test_1",
            control_model_id="v1",
            treatment_model_id="v2",
        )
        manager.start_test(config)
        manager.stop_test("test_1", analyze=False)

        assert "test_1" not in manager._active_tests

    def test_analyze_test_no_elo_service(self):
        """Should handle missing Elo service gracefully."""
        manager = ABTestManager()
        config = ABTestConfig(
            test_id="test_1",
            control_model_id="v1",
            treatment_model_id="v2",
        )
        manager.start_test(config)

        # Patch the elo_service property to return None
        with patch.object(type(manager), 'elo_service', new_callable=lambda: property(lambda self: None)):
            result = manager.analyze_test("test_1")
            assert result is None

    def test_get_test_status_completed(self):
        """Should show completed test status."""
        manager = ABTestManager()
        # Add a completed test directly
        result = ABTestResult(
            test_id="test_1",
            control_elo=1500,
            treatment_elo=1500,
            elo_difference=0,
            control_games=0,
            treatment_games=0,
            control_win_rate=0.5,
            treatment_win_rate=0.5,
            is_significant=False,
            winner=None,
            confidence=0,
            recommendation="test",
        )
        manager._completed_tests["test_1"] = result

        status = manager.get_test_status("test_1")
        assert status["status"] == "completed"


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_promotion_controller(self):
        """Should return controller instance."""
        controller = get_promotion_controller()
        assert isinstance(controller, PromotionController)

    def test_get_promotion_controller_with_criteria(self):
        """Should accept criteria."""
        criteria = PromotionCriteria(min_elo_improvement=100)
        # Use use_singleton=False to get a fresh instance with custom criteria
        controller = get_promotion_controller(criteria=criteria, use_singleton=False)
        assert controller.criteria.min_elo_improvement == 100

    def test_get_rollback_monitor(self):
        """Should return monitor instance."""
        monitor = get_rollback_monitor()
        assert isinstance(monitor, RollbackMonitor)

    def test_get_ab_test_manager(self):
        """Should return manager instance."""
        manager = get_ab_test_manager()
        assert isinstance(manager, ABTestManager)
