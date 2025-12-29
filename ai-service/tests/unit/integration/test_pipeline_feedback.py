"""Tests for app/integration/pipeline_feedback.py module.

Covers:
- FeedbackAction enum
- FeedbackSignal dataclass
- FeedbackState dataclass
- EvaluationAnalyzer class
- DataQualityMonitor class
- TrainingMonitor class
- PipelineFeedbackController class
- FeedbackSignalRouter class
- OpponentWinRateTracker class
"""

import asyncio
import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.integration.pipeline_feedback import (
    DataQualityMonitor,
    EvaluationAnalyzer,
    FeedbackAction,
    FeedbackSignal,
    FeedbackSignalRouter,
    FeedbackState,
    OpponentWinRateTracker,
    PipelineFeedbackController,
    TrainingMonitor,
    create_feedback_controller,
    create_feedback_router,
    create_opponent_tracker,
)


class TestFeedbackAction:
    """Tests for FeedbackAction enum."""

    def test_data_collection_actions(self):
        """Test data collection feedback actions exist."""
        assert FeedbackAction.INCREASE_DATA_COLLECTION.value == "increase_data"
        assert FeedbackAction.DECREASE_DATA_COLLECTION.value == "decrease_data"

    def test_curriculum_actions(self):
        """Test curriculum weight feedback actions exist."""
        assert FeedbackAction.INCREASE_CURRICULUM_WEIGHT.value == "increase_weight"
        assert FeedbackAction.DECREASE_CURRICULUM_WEIGHT.value == "decrease_weight"

    def test_optimization_actions(self):
        """Test optimization trigger actions exist."""
        assert FeedbackAction.TRIGGER_CMAES.value == "trigger_cmaes"
        assert FeedbackAction.TRIGGER_NAS.value == "trigger_nas"

    def test_training_actions(self):
        """Test training adjustment actions exist."""
        assert FeedbackAction.EXTEND_TRAINING.value == "extend_training"
        assert FeedbackAction.REDUCE_TRAINING.value == "reduce_training"
        assert FeedbackAction.ADJUST_TEMPERATURE.value == "adjust_temperature"

    def test_data_quality_actions(self):
        """Test data quality actions exist."""
        assert FeedbackAction.QUARANTINE_DATA.value == "quarantine_data"

    def test_promotion_actions(self):
        """Test promotion-related actions exist."""
        assert FeedbackAction.PROMOTION_FAILED.value == "promotion_failed"
        assert FeedbackAction.PROMOTION_SUCCEEDED.value == "promotion_succeeded"
        assert FeedbackAction.URGENT_RETRAINING.value == "urgent_retraining"

    def test_scaling_actions(self):
        """Test utilization-related scaling actions exist."""
        assert FeedbackAction.SCALE_UP_SELFPLAY.value == "scale_up_selfplay"
        assert FeedbackAction.SCALE_DOWN_SELFPLAY.value == "scale_down_selfplay"

    def test_no_action(self):
        """Test NO_ACTION exists."""
        assert FeedbackAction.NO_ACTION.value == "no_action"


class TestFeedbackSignal:
    """Tests for FeedbackSignal dataclass."""

    def test_basic_creation(self):
        """Test creating a basic FeedbackSignal."""
        signal = FeedbackSignal(
            source_stage="evaluation",
            target_stage="training",
            action=FeedbackAction.TRIGGER_CMAES,
            magnitude=0.8,
            reason="Plateau detected",
        )
        assert signal.source_stage == "evaluation"
        assert signal.target_stage == "training"
        assert signal.action == FeedbackAction.TRIGGER_CMAES
        assert signal.magnitude == 0.8
        assert signal.reason == "Plateau detected"

    def test_default_timestamp(self):
        """Test that timestamp defaults to current time."""
        before = time.time()
        signal = FeedbackSignal(
            source_stage="a",
            target_stage="b",
            action=FeedbackAction.NO_ACTION,
            magnitude=0.0,
            reason="test",
        )
        after = time.time()
        assert before <= signal.timestamp <= after

    def test_custom_timestamp(self):
        """Test custom timestamp."""
        ts = 1234567890.0
        signal = FeedbackSignal(
            source_stage="a",
            target_stage="b",
            action=FeedbackAction.NO_ACTION,
            magnitude=0.0,
            reason="test",
            timestamp=ts,
        )
        assert signal.timestamp == ts

    def test_default_metadata(self):
        """Test that metadata defaults to empty dict."""
        signal = FeedbackSignal(
            source_stage="a",
            target_stage="b",
            action=FeedbackAction.NO_ACTION,
            magnitude=0.0,
            reason="test",
        )
        assert signal.metadata == {}

    def test_custom_metadata(self):
        """Test custom metadata."""
        metadata = {"config": "hex8_2p", "elo": 1500}
        signal = FeedbackSignal(
            source_stage="a",
            target_stage="b",
            action=FeedbackAction.NO_ACTION,
            magnitude=0.0,
            reason="test",
            metadata=metadata,
        )
        assert signal.metadata == metadata


class TestFeedbackState:
    """Tests for FeedbackState dataclass."""

    def test_default_values(self):
        """Test default FeedbackState values."""
        state = FeedbackState()
        assert isinstance(state.curriculum_weights, dict)
        assert state.temperature_multiplier == 1.0
        assert state.epochs_multiplier == 1.0
        assert state.batch_size_multiplier == 1.0
        assert state.games_per_worker_multiplier == 1.0
        assert state.elo_history == []
        assert state.plateau_count == 0
        assert state.parity_failure_rate == 0.0
        assert state.consecutive_parity_failures == 0
        assert state.quarantined_games == 0
        assert state.data_quality_score == 1.0
        assert state.last_cmaes_trigger == 0.0
        assert state.last_nas_trigger == 0.0
        assert state.cmaes_cooldown_hours == 6.0
        assert state.nas_cooldown_hours == 24.0

    def test_promotion_tracking_defaults(self):
        """Test promotion tracking defaults."""
        state = FeedbackState()
        assert state.consecutive_promotion_failures == 0
        assert state.last_promotion_success == 0.0
        assert state.promotion_failure_configs == {}

    def test_curriculum_weights_default_factory(self):
        """Test curriculum_weights uses defaultdict."""
        state = FeedbackState()
        # Should return 1.0 for unknown config
        assert state.curriculum_weights.get("unknown_config", 1.0) == 1.0


class TestEvaluationAnalyzer:
    """Tests for EvaluationAnalyzer class."""

    def test_initialization(self):
        """Test EvaluationAnalyzer initialization."""
        analyzer = EvaluationAnalyzer()
        assert analyzer.win_rates == {}
        assert analyzer.elo_trends == {}

    def test_add_result(self):
        """Test adding evaluation results."""
        analyzer = EvaluationAnalyzer()
        analyzer.add_result("hex8_2p", 0.55, 1520)
        assert "hex8_2p" in analyzer.win_rates
        assert "hex8_2p" in analyzer.elo_trends
        assert analyzer.win_rates["hex8_2p"] == [0.55]
        assert analyzer.elo_trends["hex8_2p"] == [1520]

    def test_add_result_bounds_lists(self):
        """Test that lists are bounded to 20 entries."""
        analyzer = EvaluationAnalyzer()
        for i in range(25):
            analyzer.add_result("hex8_2p", 0.5, 1500 + i)
        assert len(analyzer.win_rates["hex8_2p"]) == 20
        assert len(analyzer.elo_trends["hex8_2p"]) == 20

    def test_get_weak_configs_insufficient_samples(self):
        """Test get_weak_configs with insufficient samples."""
        analyzer = EvaluationAnalyzer()
        analyzer.add_result("hex8_2p", 0.3, 1400)  # 1 sample
        analyzer.add_result("hex8_2p", 0.3, 1400)  # 2 samples
        # Need 3 samples
        assert analyzer.get_weak_configs() == []

    def test_get_weak_configs(self):
        """Test identifying weak configurations."""
        analyzer = EvaluationAnalyzer()
        # Add 5 weak results
        for _ in range(5):
            analyzer.add_result("hex8_2p", 0.35, 1400)  # Below 0.45 threshold
        # Add 5 strong results
        for _ in range(5):
            analyzer.add_result("square8_2p", 0.6, 1600)
        weak = analyzer.get_weak_configs()
        assert "hex8_2p" in weak
        assert "square8_2p" not in weak

    def test_get_elo_trend(self):
        """Test Elo trend calculation."""
        analyzer = EvaluationAnalyzer()
        # Add increasing Elo
        for i in range(10):
            analyzer.add_result("hex8_2p", 0.5, 1500 + i * 10)
        trend = analyzer.get_elo_trend("hex8_2p", lookback=5)
        assert trend == 40  # 1590 - 1550

    def test_get_elo_trend_insufficient_data(self):
        """Test Elo trend with insufficient data."""
        analyzer = EvaluationAnalyzer()
        analyzer.add_result("hex8_2p", 0.5, 1500)
        analyzer.add_result("hex8_2p", 0.5, 1510)
        trend = analyzer.get_elo_trend("hex8_2p", lookback=5)
        assert trend == 0.0

    def test_is_plateau(self):
        """Test plateau detection."""
        analyzer = EvaluationAnalyzer()
        # Add flat Elo
        for i in range(10):
            analyzer.add_result("hex8_2p", 0.5, 1500 + i)  # Only +1 per sample
        assert analyzer.is_plateau("hex8_2p", min_improvement=15.0, lookback=5)

    def test_is_not_plateau(self):
        """Test non-plateau detection."""
        analyzer = EvaluationAnalyzer()
        # Add increasing Elo
        for i in range(10):
            analyzer.add_result("hex8_2p", 0.5, 1500 + i * 10)
        assert not analyzer.is_plateau("hex8_2p", min_improvement=15.0, lookback=5)


class TestDataQualityMonitor:
    """Tests for DataQualityMonitor class."""

    def test_initialization(self):
        """Test DataQualityMonitor initialization."""
        monitor = DataQualityMonitor()
        assert monitor.parity_results == []
        assert monitor.game_lengths == []
        assert monitor.outlier_games == []

    def test_add_parity_result_passed(self):
        """Test adding passed parity result."""
        monitor = DataQualityMonitor()
        monitor.add_parity_result(True)
        assert len(monitor.parity_results) == 1
        assert monitor.parity_results[0] is True

    def test_add_parity_result_failed_with_game_id(self):
        """Test adding failed parity result with game ID."""
        monitor = DataQualityMonitor()
        monitor.add_parity_result(False, "game-123")
        assert len(monitor.parity_results) == 1
        assert monitor.parity_results[0] is False
        assert "game-123" in monitor.outlier_games

    def test_parity_results_bounded(self):
        """Test parity results are bounded to 1000."""
        monitor = DataQualityMonitor()
        for i in range(1100):
            monitor.add_parity_result(True)
        assert len(monitor.parity_results) == 1000

    def test_get_parity_failure_rate_no_results(self):
        """Test failure rate with no results."""
        monitor = DataQualityMonitor()
        assert monitor.get_parity_failure_rate() == 0.0

    def test_get_parity_failure_rate(self):
        """Test parity failure rate calculation."""
        monitor = DataQualityMonitor()
        # Add 80 passed, 20 failed
        for _ in range(80):
            monitor.add_parity_result(True)
        for _ in range(20):
            monitor.add_parity_result(False)
        rate = monitor.get_parity_failure_rate()
        assert abs(rate - 0.2) < 0.001  # 20% failure (approximate)

    def test_should_quarantine_below_threshold(self):
        """Test should_quarantine below threshold."""
        monitor = DataQualityMonitor()
        for _ in range(95):
            monitor.add_parity_result(True)
        for _ in range(5):
            monitor.add_parity_result(False)
        assert not monitor.should_quarantine(failure_threshold=0.1)

    def test_should_quarantine_above_threshold(self):
        """Test should_quarantine above threshold."""
        monitor = DataQualityMonitor()
        for _ in range(85):
            monitor.add_parity_result(True)
        for _ in range(15):
            monitor.add_parity_result(False)
        assert monitor.should_quarantine(failure_threshold=0.1)


class TestTrainingMonitor:
    """Tests for TrainingMonitor class."""

    def test_initialization(self):
        """Test TrainingMonitor initialization."""
        monitor = TrainingMonitor()
        assert monitor.loss_history == []
        assert monitor.val_loss_history == []
        assert monitor.learning_rate_history == []

    def test_add_training_metrics(self):
        """Test adding training metrics."""
        monitor = TrainingMonitor()
        monitor.add_training_metrics(0.5, val_loss=0.6, learning_rate=0.001)
        assert monitor.loss_history == [0.5]
        assert monitor.val_loss_history == [0.6]
        assert monitor.learning_rate_history == [0.001]

    def test_add_training_metrics_partial(self):
        """Test adding partial training metrics."""
        monitor = TrainingMonitor()
        monitor.add_training_metrics(0.5)
        assert monitor.loss_history == [0.5]
        assert monitor.val_loss_history == []
        assert monitor.learning_rate_history == []

    def test_is_loss_plateau_insufficient_data(self):
        """Test loss plateau with insufficient data."""
        monitor = TrainingMonitor()
        monitor.add_training_metrics(0.5)
        assert not monitor.is_loss_plateau(lookback=10)

    def test_is_loss_plateau(self):
        """Test loss plateau detection."""
        monitor = TrainingMonitor()
        # Add flat loss
        for i in range(15):
            monitor.add_training_metrics(0.5 - i * 0.001)  # Very small improvement
        assert monitor.is_loss_plateau(lookback=10, threshold=0.01)

    def test_is_not_loss_plateau(self):
        """Test non-plateau detection."""
        monitor = TrainingMonitor()
        # Add decreasing loss
        for i in range(15):
            monitor.add_training_metrics(0.5 - i * 0.01)
        assert not monitor.is_loss_plateau(lookback=10, threshold=0.01)

    def test_is_overfitting_insufficient_data(self):
        """Test overfitting detection with insufficient data."""
        monitor = TrainingMonitor()
        monitor.add_training_metrics(0.5, val_loss=0.5)
        assert not monitor.is_overfitting()

    def test_is_overfitting(self):
        """Test overfitting detection."""
        monitor = TrainingMonitor()
        # Train loss decreasing, val loss increasing
        for i in range(10):
            monitor.add_training_metrics(0.5 - i * 0.02, val_loss=0.5 + i * 0.02)
        assert monitor.is_overfitting()

    def test_is_not_overfitting(self):
        """Test non-overfitting detection."""
        monitor = TrainingMonitor()
        # Both losses decreasing
        for i in range(10):
            monitor.add_training_metrics(0.5 - i * 0.02, val_loss=0.5 - i * 0.01)
        assert not monitor.is_overfitting()


class TestPipelineFeedbackController:
    """Tests for PipelineFeedbackController class."""

    def test_initialization_no_state_path(self):
        """Test initialization without state path."""
        controller = PipelineFeedbackController()
        assert controller.state_path is None
        assert isinstance(controller.state, FeedbackState)
        assert isinstance(controller.eval_analyzer, EvaluationAnalyzer)
        assert isinstance(controller.data_monitor, DataQualityMonitor)
        assert isinstance(controller.training_monitor, TrainingMonitor)
        assert controller.signals == []

    def test_initialization_with_state_path(self):
        """Test initialization with state path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "feedback_state.json"
            controller = PipelineFeedbackController(state_path=state_path)
            assert controller.state_path == state_path

    def test_load_state_from_file(self):
        """Test loading state from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "feedback_state.json"
            # Create state file
            state_data = {
                "temperature_multiplier": 1.5,
                "epochs_multiplier": 1.2,
                "plateau_count": 3,
            }
            state_path.write_text(json.dumps(state_data))
            controller = PipelineFeedbackController(state_path=state_path)
            assert controller.state.temperature_multiplier == 1.5
            assert controller.state.epochs_multiplier == 1.2
            assert controller.state.plateau_count == 3

    def test_save_state(self):
        """Test saving state to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "subdir" / "feedback_state.json"
            controller = PipelineFeedbackController(state_path=state_path)
            controller.state.temperature_multiplier = 1.3
            controller._save_state()
            assert state_path.exists()
            saved = json.loads(state_path.read_text())
            assert saved["temperature_multiplier"] == 1.3

    def test_emit_signal(self):
        """Test signal emission."""
        controller = PipelineFeedbackController()
        signal = FeedbackSignal(
            source_stage="test",
            target_stage="target",
            action=FeedbackAction.NO_ACTION,
            magnitude=0.5,
            reason="test signal",
        )
        controller._emit_signal(signal)
        assert len(controller.signals) == 1
        assert controller.signals[0] == signal

    def test_signal_history_bounded(self):
        """Test signal history is bounded."""
        controller = PipelineFeedbackController()
        controller._max_signals = 10
        for i in range(20):
            signal = FeedbackSignal(
                source_stage="test",
                target_stage="target",
                action=FeedbackAction.NO_ACTION,
                magnitude=0.5,
                reason=f"test signal {i}",
            )
            controller._emit_signal(signal)
        assert len(controller.signals) == 10

    def test_get_curriculum_weight_default(self):
        """Test default curriculum weight."""
        controller = PipelineFeedbackController()
        assert controller.get_curriculum_weight("unknown_config") == 1.0

    def test_get_curriculum_weight_custom(self):
        """Test custom curriculum weight."""
        controller = PipelineFeedbackController()
        controller.state.curriculum_weights["hex8_2p"] = 1.5
        assert controller.get_curriculum_weight("hex8_2p") == 1.5

    def test_get_temperature_multiplier(self):
        """Test temperature multiplier getter."""
        controller = PipelineFeedbackController()
        controller.state.temperature_multiplier = 1.3
        assert controller.get_temperature_multiplier() == 1.3

    def test_get_epochs_multiplier(self):
        """Test epochs multiplier getter."""
        controller = PipelineFeedbackController()
        controller.state.epochs_multiplier = 1.2
        assert controller.get_epochs_multiplier() == 1.2

    def test_get_games_per_worker_multiplier(self):
        """Test games per worker multiplier getter."""
        controller = PipelineFeedbackController()
        controller.state.games_per_worker_multiplier = 0.8
        assert controller.get_games_per_worker_multiplier() == 0.8

    def test_should_quarantine_data(self):
        """Test quarantine check."""
        controller = PipelineFeedbackController()
        # Add many failures
        for _ in range(20):
            controller.data_monitor.add_parity_result(False)
        for _ in range(80):
            controller.data_monitor.add_parity_result(True)
        assert controller.should_quarantine_data()

    def test_update_data_quality(self):
        """Test data quality update."""
        controller = PipelineFeedbackController()
        score = controller.update_data_quality(
            parity_results=[True, True, True, False],  # 25% failure
            draw_rate=0.15,  # Below 20% threshold
        )
        assert 0 < score <= 1.0

    def test_get_pending_actions(self):
        """Test getting pending actions."""
        controller = PipelineFeedbackController()
        # Emit a trigger signal
        signal = FeedbackSignal(
            source_stage="test",
            target_stage="cmaes",
            action=FeedbackAction.TRIGGER_CMAES,
            magnitude=1.0,
            reason="plateau",
        )
        controller._emit_signal(signal)
        pending = controller.get_pending_actions()
        assert len(pending) == 1
        assert pending[0].action == FeedbackAction.TRIGGER_CMAES

    def test_get_state_summary(self):
        """Test state summary."""
        controller = PipelineFeedbackController()
        controller.state.plateau_count = 2
        summary = controller.get_state_summary()
        assert "plateau_count" in summary
        assert summary["plateau_count"] == 2
        assert "curriculum_weights" in summary
        assert "weak_configs" in summary

    @pytest.mark.asyncio
    async def test_on_stage_complete_unknown_stage(self):
        """Test on_stage_complete with unknown stage."""
        controller = PipelineFeedbackController()
        await controller.on_stage_complete("unknown_stage", {})
        # Should not raise

    @pytest.mark.asyncio
    async def test_on_evaluation_complete(self):
        """Test evaluation completion handler."""
        controller = PipelineFeedbackController()
        await controller.on_stage_complete("evaluation", {
            "config_key": "hex8_2p",
            "win_rate": 0.55,
            "elo": 1520,
        })
        assert "hex8_2p" in controller.eval_analyzer.win_rates

    @pytest.mark.asyncio
    async def test_on_training_complete(self):
        """Test training completion handler."""
        controller = PipelineFeedbackController()
        await controller.on_stage_complete("training", {
            "final_loss": 0.5,
            "val_loss": 0.6,
        })
        assert len(controller.training_monitor.loss_history) == 1

    @pytest.mark.asyncio
    async def test_on_parity_validation_complete(self):
        """Test parity validation completion handler."""
        controller = PipelineFeedbackController()
        await controller.on_stage_complete("parity-validation", {
            "passed": 90,
            "failed": 10,
        })
        assert abs(controller.state.parity_failure_rate - 0.1) < 0.001

    @pytest.mark.asyncio
    async def test_on_promotion_complete_success(self):
        """Test promotion success handler."""
        controller = PipelineFeedbackController()
        await controller.on_stage_complete("promotion", {
            "config_key": "hex8_2p",
            "success": True,
            "elo_gain": 25,
            "model_id": "model_v1",
        })
        assert controller.state.consecutive_promotion_failures == 0
        assert len(controller.signals) >= 1

    @pytest.mark.asyncio
    async def test_on_promotion_complete_failure(self):
        """Test promotion failure handler."""
        controller = PipelineFeedbackController()
        await controller.on_stage_complete("promotion", {
            "config_key": "hex8_2p",
            "success": False,
            "model_id": "model_v1",
            "reason": "low Elo",
        })
        assert controller.state.consecutive_promotion_failures == 1
        assert "hex8_2p" in controller.state.promotion_failure_configs

    @pytest.mark.asyncio
    async def test_on_stage_failed(self):
        """Test stage failure handler."""
        controller = PipelineFeedbackController()
        await controller.on_stage_failed("training", {
            "config_key": "hex8_2p",
            "error": "OOM error",
        })
        # Should reduce batch size for memory errors
        assert controller.state.batch_size_multiplier < 1.0

    def test_reset_consecutive_failures(self):
        """Test resetting consecutive failure count."""
        controller = PipelineFeedbackController()
        controller.state.consecutive_failures = {"training": 5}
        controller.reset_consecutive_failures("training")
        assert controller.state.consecutive_failures["training"] == 0


class TestFeedbackSignalRouter:
    """Tests for FeedbackSignalRouter class."""

    def test_initialization(self):
        """Test router initialization."""
        router = FeedbackSignalRouter()
        assert router._handlers == {}
        assert router._signal_history == []

    def test_register_handler(self):
        """Test handler registration."""
        router = FeedbackSignalRouter()

        def handler(signal):
            return True

        router.register_handler(FeedbackAction.TRIGGER_CMAES, handler, name="test_handler")
        assert len(router._handlers[FeedbackAction.TRIGGER_CMAES]) == 1

    def test_unregister_handler(self):
        """Test handler unregistration."""
        router = FeedbackSignalRouter()

        def handler(signal):
            return True

        router.register_handler(FeedbackAction.TRIGGER_CMAES, handler)
        router.unregister_handler(FeedbackAction.TRIGGER_CMAES, handler)
        assert len(router._handlers[FeedbackAction.TRIGGER_CMAES]) == 0

    @pytest.mark.asyncio
    async def test_route_no_handlers(self):
        """Test routing with no handlers."""
        router = FeedbackSignalRouter()
        signal = FeedbackSignal(
            source_stage="test",
            target_stage="cmaes",
            action=FeedbackAction.TRIGGER_CMAES,
            magnitude=1.0,
            reason="test",
        )
        results = await router.route(signal)
        assert results == []

    @pytest.mark.asyncio
    async def test_route_sync_handler(self):
        """Test routing to sync handler."""
        router = FeedbackSignalRouter()

        def handler(signal):
            return True

        router.register_handler(FeedbackAction.TRIGGER_CMAES, handler, name="sync_handler")
        signal = FeedbackSignal(
            source_stage="test",
            target_stage="cmaes",
            action=FeedbackAction.TRIGGER_CMAES,
            magnitude=1.0,
            reason="test",
        )
        results = await router.route(signal)
        assert len(results) == 1
        assert results[0] == ("sync_handler", True)

    @pytest.mark.asyncio
    async def test_route_async_handler(self):
        """Test routing to async handler."""
        router = FeedbackSignalRouter()

        async def handler(signal):
            return True

        router.register_handler(FeedbackAction.TRIGGER_CMAES, handler, name="async_handler")
        signal = FeedbackSignal(
            source_stage="test",
            target_stage="cmaes",
            action=FeedbackAction.TRIGGER_CMAES,
            magnitude=1.0,
            reason="test",
        )
        results = await router.route(signal)
        assert len(results) == 1
        assert results[0] == ("async_handler", True)

    @pytest.mark.asyncio
    async def test_route_handler_exception(self):
        """Test routing with handler exception."""
        router = FeedbackSignalRouter()

        def handler(signal):
            raise RuntimeError("Test error")

        router.register_handler(FeedbackAction.TRIGGER_CMAES, handler, name="bad_handler")
        signal = FeedbackSignal(
            source_stage="test",
            target_stage="cmaes",
            action=FeedbackAction.TRIGGER_CMAES,
            magnitude=1.0,
            reason="test",
        )
        results = await router.route(signal)
        assert len(results) == 1
        assert results[0] == ("bad_handler", False)

    @pytest.mark.asyncio
    async def test_route_records_history(self):
        """Test routing records history."""
        router = FeedbackSignalRouter()

        def handler(signal):
            return True

        router.register_handler(FeedbackAction.TRIGGER_CMAES, handler)
        signal = FeedbackSignal(
            source_stage="test",
            target_stage="cmaes",
            action=FeedbackAction.TRIGGER_CMAES,
            magnitude=1.0,
            reason="test",
        )
        await router.route(signal)
        assert len(router._signal_history) == 1

    def test_get_history(self):
        """Test getting history."""
        router = FeedbackSignalRouter()
        # Manually add to history
        signal = FeedbackSignal(
            source_stage="test",
            target_stage="cmaes",
            action=FeedbackAction.TRIGGER_CMAES,
            magnitude=1.0,
            reason="test",
        )
        router._record_history(signal, "ok")
        history = router.get_history()
        assert len(history) == 1
        assert history[0]["action"] == "trigger_cmaes"

    def test_get_history_filtered_by_action(self):
        """Test filtering history by action."""
        router = FeedbackSignalRouter()
        signal1 = FeedbackSignal(
            source_stage="test",
            target_stage="cmaes",
            action=FeedbackAction.TRIGGER_CMAES,
            magnitude=1.0,
            reason="test1",
        )
        signal2 = FeedbackSignal(
            source_stage="test",
            target_stage="nas",
            action=FeedbackAction.TRIGGER_NAS,
            magnitude=1.0,
            reason="test2",
        )
        router._record_history(signal1, "ok")
        router._record_history(signal2, "ok")
        history = router.get_history(action=FeedbackAction.TRIGGER_CMAES)
        assert len(history) == 1
        assert history[0]["action"] == "trigger_cmaes"

    def test_get_stats(self):
        """Test getting stats."""
        router = FeedbackSignalRouter()

        def handler(signal):
            return True

        router.register_handler(FeedbackAction.TRIGGER_CMAES, handler)
        signal = FeedbackSignal(
            source_stage="test",
            target_stage="cmaes",
            action=FeedbackAction.TRIGGER_CMAES,
            magnitude=1.0,
            reason="test",
        )
        router._record_history(signal, "ok")
        stats = router.get_stats()
        assert stats["total_signals_routed"] == 1
        assert "trigger_cmaes" in stats["handlers_registered"]


class TestOpponentWinRateTracker:
    """Tests for OpponentWinRateTracker class."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = OpponentWinRateTracker()
        assert tracker.min_games == 10
        assert tracker.weak_threshold == 0.45

    def test_initialization_custom(self):
        """Test tracker initialization with custom values."""
        tracker = OpponentWinRateTracker(min_games=5, weak_threshold=0.4)
        assert tracker.min_games == 5
        assert tracker.weak_threshold == 0.4

    def test_record_game_win(self):
        """Test recording a win."""
        tracker = OpponentWinRateTracker()
        tracker.record_game("model_v1", "mcts_100", won=True)
        assert tracker._records["model_v1"]["mcts_100"]["wins"] == 1

    def test_record_game_loss(self):
        """Test recording a loss."""
        tracker = OpponentWinRateTracker()
        tracker.record_game("model_v1", "mcts_100", won=False)
        assert tracker._records["model_v1"]["mcts_100"]["losses"] == 1

    def test_record_game_draw(self):
        """Test recording a draw."""
        tracker = OpponentWinRateTracker()
        tracker.record_game("model_v1", "mcts_100", draw=True)
        assert tracker._records["model_v1"]["mcts_100"]["draws"] == 1

    def test_get_win_rate_insufficient_games(self):
        """Test win rate with insufficient games."""
        tracker = OpponentWinRateTracker(min_games=10)
        for i in range(5):
            tracker.record_game("model_v1", "mcts_100", won=True)
        assert tracker.get_win_rate("model_v1", "mcts_100") is None

    def test_get_win_rate(self):
        """Test win rate calculation."""
        tracker = OpponentWinRateTracker(min_games=10)
        for _ in range(8):
            tracker.record_game("model_v1", "mcts_100", won=True)
        for _ in range(2):
            tracker.record_game("model_v1", "mcts_100", won=False)
        assert tracker.get_win_rate("model_v1", "mcts_100") == 0.8

    def test_get_win_rate_with_draws(self):
        """Test win rate with draws."""
        tracker = OpponentWinRateTracker(min_games=10)
        for _ in range(5):
            tracker.record_game("model_v1", "mcts_100", won=True)
        for _ in range(3):
            tracker.record_game("model_v1", "mcts_100", won=False)
        for _ in range(2):
            tracker.record_game("model_v1", "mcts_100", draw=True)
        # (5 + 0.5*2) / 10 = 0.6
        assert tracker.get_win_rate("model_v1", "mcts_100") == 0.6

    def test_get_weak_opponents(self):
        """Test getting weak opponents."""
        tracker = OpponentWinRateTracker(min_games=10, weak_threshold=0.45)
        # Strong opponent
        for _ in range(10):
            tracker.record_game("model_v1", "random", won=True)
        # Weak opponent
        for _ in range(4):
            tracker.record_game("model_v1", "mcts_1000", won=True)
        for _ in range(6):
            tracker.record_game("model_v1", "mcts_1000", won=False)
        weak = tracker.get_weak_opponents("model_v1")
        assert len(weak) == 1
        assert weak[0][0] == "mcts_1000"
        assert weak[0][1] == 0.4

    def test_get_strong_opponents(self):
        """Test getting strong opponents."""
        tracker = OpponentWinRateTracker(min_games=10)
        # Strong opponent
        for _ in range(7):
            tracker.record_game("model_v1", "random", won=True)
        for _ in range(3):
            tracker.record_game("model_v1", "random", won=False)
        # Weak opponent
        for _ in range(4):
            tracker.record_game("model_v1", "mcts_1000", won=True)
        for _ in range(6):
            tracker.record_game("model_v1", "mcts_1000", won=False)
        strong = tracker.get_strong_opponents("model_v1", threshold=0.60)
        assert len(strong) == 1
        assert strong[0][0] == "random"
        assert strong[0][1] == 0.7

    def test_get_summary(self):
        """Test getting summary."""
        tracker = OpponentWinRateTracker(min_games=10)
        for _ in range(10):
            tracker.record_game("model_v1", "random", won=True)
        summary = tracker.get_summary("model_v1")
        assert summary["total_opponents"] == 1
        assert "random" in summary["opponents"]
        assert summary["opponents"]["random"]["wins"] == 10

    def test_history_bounded(self):
        """Test history is bounded to 10000."""
        tracker = OpponentWinRateTracker()
        for i in range(11000):
            tracker.record_game("model", f"opp_{i % 10}", won=True)
        assert len(tracker._history) == 10000


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_feedback_controller(self):
        """Test create_feedback_controller function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = create_feedback_controller(Path(tmpdir))
            assert isinstance(controller, PipelineFeedbackController)
            assert controller.state_path == Path(tmpdir) / "logs" / "feedback" / "feedback_state.json"

    def test_create_feedback_controller_with_string_path(self):
        """Test create_feedback_controller with string path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = create_feedback_controller(tmpdir)
            assert isinstance(controller, PipelineFeedbackController)

    def test_create_feedback_router(self):
        """Test create_feedback_router function."""
        router = create_feedback_router()
        assert isinstance(router, FeedbackSignalRouter)

    def test_create_opponent_tracker(self):
        """Test create_opponent_tracker function."""
        tracker = create_opponent_tracker(min_games=5, weak_threshold=0.4)
        assert isinstance(tracker, OpponentWinRateTracker)
        assert tracker.min_games == 5
        assert tracker.weak_threshold == 0.4
