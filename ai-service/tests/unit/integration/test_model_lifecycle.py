"""
Tests for app.integration.model_lifecycle module.

Tests the model lifecycle management infrastructure:
- LifecycleConfig configuration
- LifecycleStage and PromotionDecision enums
- EvaluationResult dataclass
- PromotionGate promotion decisions
- TrainingConditions and TrainingTrigger
- ModelSyncCoordinator cluster sync
- ModelLifecycleManager orchestration
"""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.integration.model_lifecycle import (
    EvaluationResult,
    LifecycleConfig,
    LifecycleStage,
    ModelLifecycleManager,
    ModelSyncCoordinator,
    PromotionDecision,
    PromotionGate,
    TrainingConditions,
    TrainingTrigger,
)


class TestLifecycleConfig:
    """Tests for LifecycleConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LifecycleConfig()
        assert config.registry_dir == "data/model_registry"
        assert config.model_storage_dir == "data/models"
        assert config.min_elo_improvement > 0
        assert config.min_games_for_staging > 0
        assert config.min_games_for_production > 0
        assert 0 < config.min_win_rate_vs_production <= 1.0
        assert config.auto_train_on_data_threshold is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LifecycleConfig(
            registry_dir="/custom/registry",
            model_storage_dir="/custom/models",
            min_elo_improvement=50.0,
            min_games_for_staging=100,
            sync_interval_seconds=600.0,
        )
        assert config.registry_dir == "/custom/registry"
        assert config.model_storage_dir == "/custom/models"
        assert config.min_elo_improvement == 50.0
        assert config.min_games_for_staging == 100
        assert config.sync_interval_seconds == 600.0

    def test_p2p_api_base_default(self):
        """Test P2P API base has sensible default."""
        config = LifecycleConfig()
        # Should contain localhost and port
        assert "localhost" in config.p2p_api_base or "127.0.0.1" in config.p2p_api_base

    def test_rollback_settings(self):
        """Test rollback configuration settings."""
        config = LifecycleConfig(
            auto_rollback_on_regression=False,
            regression_threshold_elo=-100.0,
        )
        assert config.auto_rollback_on_regression is False
        assert config.regression_threshold_elo == -100.0

    def test_webhook_settings(self):
        """Test webhook configuration."""
        config = LifecycleConfig(
            on_promotion_webhook="http://example.com/promote",
            on_training_webhook="http://example.com/train",
        )
        assert config.on_promotion_webhook == "http://example.com/promote"
        assert config.on_training_webhook == "http://example.com/train"


class TestLifecycleStage:
    """Tests for LifecycleStage enum."""

    def test_all_stages_defined(self):
        """Test all lifecycle stages are defined."""
        assert LifecycleStage.TRAINING.value == "training"
        assert LifecycleStage.CALIBRATING.value == "calibrating"
        assert LifecycleStage.EVALUATING.value == "evaluating"
        assert LifecycleStage.STAGING.value == "staging"
        assert LifecycleStage.PRODUCTION.value == "production"
        assert LifecycleStage.ROLLBACK_CANDIDATE.value == "rollback"
        assert LifecycleStage.ARCHIVED.value == "archived"
        assert LifecycleStage.REJECTED.value == "rejected"

    def test_stage_count(self):
        """Test expected number of stages."""
        # Ensure all 8 stages are present
        assert len(LifecycleStage) == 8


class TestPromotionDecision:
    """Tests for PromotionDecision enum."""

    def test_all_decisions_defined(self):
        """Test all promotion decisions are defined."""
        assert PromotionDecision.PROMOTE.value == "promote"
        assert PromotionDecision.HOLD.value == "hold"
        assert PromotionDecision.REJECT.value == "reject"
        assert PromotionDecision.ROLLBACK.value == "rollback"

    def test_decision_count(self):
        """Test expected number of decisions."""
        assert len(PromotionDecision) == 4


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_minimal_creation(self):
        """Test minimal creation with required fields."""
        result = EvaluationResult(
            model_id="test_model",
            version=1,
        )
        assert result.model_id == "test_model"
        assert result.version == 1
        assert result.elo is None
        assert result.games_played == 0
        assert result.win_rate is None

    def test_full_creation(self):
        """Test creation with all fields."""
        result = EvaluationResult(
            model_id="test_model",
            version=2,
            elo=1650.0,
            elo_uncertainty=50.0,
            games_played=200,
            win_rate=0.65,
            draw_rate=0.10,
            value_mse=0.05,
            policy_accuracy=0.75,
            elo_vs_production=25.0,
            win_rate_vs_production=0.55,
            games_vs_production=100,
            avg_game_length=45.5,
            calibration_error=0.02,
            inference_time_ms=5.5,
        )
        assert result.elo == 1650.0
        assert result.games_played == 200
        assert result.win_rate == 0.65
        assert result.elo_vs_production == 25.0
        assert result.inference_time_ms == 5.5


class TestPromotionGate:
    """Tests for PromotionGate class."""

    def test_initialization(self):
        """Test PromotionGate initialization."""
        config = LifecycleConfig()
        gate = PromotionGate(config)
        assert gate.config is config
        assert gate._evaluation_history == {}

    def test_min_elo_improvement_property(self):
        """Test min_elo_improvement property."""
        config = LifecycleConfig(min_elo_improvement=30.0)
        gate = PromotionGate(config)
        assert gate.min_elo_improvement == 30.0

    def test_min_games_for_production_property(self):
        """Test min_games_for_production property."""
        config = LifecycleConfig(min_games_for_production=250)
        gate = PromotionGate(config)
        assert gate.min_games_for_production == 250

    def test_evaluate_for_staging_insufficient_games(self):
        """Test staging evaluation with insufficient games."""
        config = LifecycleConfig(min_games_for_staging=50)
        gate = PromotionGate(config)
        result = EvaluationResult(
            model_id="test",
            version=1,
            games_played=30,
            elo=1500.0,
        )
        decision, reason = gate.evaluate_for_staging(result)
        assert decision == PromotionDecision.HOLD
        assert "Insufficient games" in reason

    def test_evaluate_for_staging_no_elo(self):
        """Test staging evaluation without Elo rating."""
        config = LifecycleConfig(min_games_for_staging=50)
        gate = PromotionGate(config)
        result = EvaluationResult(
            model_id="test",
            version=1,
            games_played=100,
            elo=None,
        )
        decision, reason = gate.evaluate_for_staging(result)
        assert decision == PromotionDecision.HOLD
        assert "No Elo" in reason

    def test_evaluate_for_staging_low_elo(self):
        """Test staging evaluation with too low Elo."""
        config = LifecycleConfig(min_games_for_staging=50)
        gate = PromotionGate(config)
        result = EvaluationResult(
            model_id="test",
            version=1,
            games_played=100,
            elo=1300.0,  # Below 1400 threshold
        )
        decision, reason = gate.evaluate_for_staging(result)
        assert decision == PromotionDecision.REJECT
        assert "Elo too low" in reason

    def test_evaluate_for_staging_low_policy_accuracy(self):
        """Test staging evaluation with low policy accuracy."""
        config = LifecycleConfig(min_games_for_staging=50)
        gate = PromotionGate(config)
        result = EvaluationResult(
            model_id="test",
            version=1,
            games_played=100,
            elo=1600.0,
            policy_accuracy=0.2,  # Below 0.3 threshold
        )
        decision, reason = gate.evaluate_for_staging(result)
        assert decision == PromotionDecision.REJECT
        assert "Policy accuracy" in reason

    def test_evaluate_for_staging_success(self):
        """Test successful staging evaluation."""
        config = LifecycleConfig(min_games_for_staging=50)
        gate = PromotionGate(config)
        result = EvaluationResult(
            model_id="test",
            version=1,
            games_played=100,
            elo=1600.0,
            policy_accuracy=0.5,
        )
        decision, reason = gate.evaluate_for_staging(result)
        assert decision == PromotionDecision.PROMOTE
        assert "staging criteria" in reason.lower()

    def test_evaluate_for_production_no_production_model(self):
        """Test production evaluation without existing production model."""
        config = LifecycleConfig(min_games_for_production=200)
        gate = PromotionGate(config)
        result = EvaluationResult(
            model_id="test",
            version=1,
            games_played=250,
            elo=1600.0,
        )
        decision, reason = gate.evaluate_for_production(result, production_result=None)
        assert decision == PromotionDecision.PROMOTE
        assert "No current production" in reason

    def test_evaluate_for_production_insufficient_games(self):
        """Test production evaluation with insufficient games."""
        config = LifecycleConfig(min_games_for_production=200)
        gate = PromotionGate(config)
        result = EvaluationResult(
            model_id="test",
            version=1,
            games_played=100,  # Below threshold
            elo=1600.0,
        )
        prod = EvaluationResult(model_id="prod", version=0, games_played=500)
        decision, reason = gate.evaluate_for_production(result, prod)
        assert decision == PromotionDecision.HOLD
        assert "Insufficient games" in reason

    def test_evaluate_for_production_insufficient_h2h(self):
        """Test production evaluation with insufficient head-to-head games."""
        config = LifecycleConfig(min_games_for_production=200)
        gate = PromotionGate(config)
        result = EvaluationResult(
            model_id="test",
            version=1,
            games_played=250,
            elo=1600.0,
            games_vs_production=30,  # Below 50 threshold
        )
        prod = EvaluationResult(model_id="prod", version=0, games_played=500)
        decision, reason = gate.evaluate_for_production(result, prod)
        assert decision == PromotionDecision.HOLD
        assert "head-to-head" in reason.lower()

    def test_evaluate_for_production_regression(self):
        """Test production evaluation with significant regression."""
        config = LifecycleConfig(
            min_games_for_production=200,
            regression_threshold_elo=-50.0,
        )
        gate = PromotionGate(config)
        result = EvaluationResult(
            model_id="test",
            version=1,
            games_played=250,
            elo=1500.0,
            games_vs_production=100,
            elo_vs_production=-60.0,  # Significant regression
        )
        prod = EvaluationResult(model_id="prod", version=0, games_played=500, elo=1560.0)
        decision, reason = gate.evaluate_for_production(result, prod)
        assert decision == PromotionDecision.REJECT
        assert "regression" in reason.lower()

    def test_evaluate_for_production_success(self):
        """Test successful production evaluation."""
        config = LifecycleConfig(
            min_games_for_production=200,
            min_elo_improvement=20.0,
            min_win_rate_vs_production=0.45,
        )
        gate = PromotionGate(config)
        result = EvaluationResult(
            model_id="test",
            version=1,
            games_played=250,
            elo=1650.0,
            games_vs_production=100,
            elo_vs_production=30.0,
            win_rate_vs_production=0.55,
        )
        prod = EvaluationResult(model_id="prod", version=0, games_played=500, elo=1620.0)
        decision, reason = gate.evaluate_for_production(result, prod)
        assert decision == PromotionDecision.PROMOTE
        assert "improved" in reason.lower()

    def test_check_rollback_disabled(self):
        """Test rollback check when disabled."""
        config = LifecycleConfig(auto_rollback_on_regression=False)
        gate = PromotionGate(config)
        current = EvaluationResult(model_id="c", version=1, elo=1400.0)
        previous = EvaluationResult(model_id="p", version=0, elo=1600.0)
        should_rollback, reason = gate.check_rollback(current, previous)
        assert should_rollback is False
        assert "disabled" in reason.lower()

    def test_check_rollback_elo_regression(self):
        """Test rollback check on Elo regression."""
        config = LifecycleConfig(
            auto_rollback_on_regression=True,
            regression_threshold_elo=-50.0,
        )
        gate = PromotionGate(config)
        current = EvaluationResult(model_id="c", version=1, elo=1450.0)
        previous = EvaluationResult(model_id="p", version=0, elo=1600.0)
        should_rollback, reason = gate.check_rollback(current, previous)
        assert should_rollback is True
        assert "Elo regression" in reason

    def test_check_rollback_win_rate_collapse(self):
        """Test rollback check on win rate collapse."""
        config = LifecycleConfig(auto_rollback_on_regression=True)
        gate = PromotionGate(config)
        current = EvaluationResult(model_id="c", version=1, win_rate=0.35)
        previous = EvaluationResult(model_id="p", version=0, win_rate=0.55)
        should_rollback, reason = gate.check_rollback(current, previous)
        assert should_rollback is True
        assert "Win rate" in reason

    def test_check_rollback_no_issue(self):
        """Test rollback check with no issues."""
        config = LifecycleConfig(auto_rollback_on_regression=True)
        gate = PromotionGate(config)
        current = EvaluationResult(model_id="c", version=1, elo=1620.0, win_rate=0.55)
        previous = EvaluationResult(model_id="p", version=0, elo=1600.0, win_rate=0.50)
        should_rollback, reason = gate.check_rollback(current, previous)
        assert should_rollback is False
        assert "No rollback" in reason


class TestTrainingConditions:
    """Tests for TrainingConditions dataclass."""

    def test_default_values(self):
        """Test default values."""
        conditions = TrainingConditions()
        assert conditions.new_games_count == 0
        assert conditions.hours_since_last_training == 0.0
        assert conditions.data_quality_score == 1.0
        assert conditions.elo_plateau_detected is False
        assert conditions.curriculum_stage_ready is False
        assert conditions.manual_trigger is False

    def test_custom_values(self):
        """Test custom values."""
        conditions = TrainingConditions(
            new_games_count=1000,
            hours_since_last_training=24.0,
            data_quality_score=0.95,
            elo_plateau_detected=True,
            curriculum_stage_ready=True,
            manual_trigger=True,
        )
        assert conditions.new_games_count == 1000
        assert conditions.hours_since_last_training == 24.0
        assert conditions.elo_plateau_detected is True


class TestTrainingTrigger:
    """Tests for TrainingTrigger class."""

    def test_initialization(self):
        """Test TrainingTrigger initialization."""
        config = LifecycleConfig()
        trigger = TrainingTrigger(config)
        assert trigger.config is config
        assert trigger._last_training_time is None
        assert trigger._last_games_count == 0

    def test_manual_trigger(self):
        """Test manual training trigger."""
        config = LifecycleConfig()
        trigger = TrainingTrigger(config)
        conditions = TrainingConditions(manual_trigger=True)
        should_train, reason = trigger.should_trigger_training(conditions)
        assert should_train is True
        assert "Manual" in reason

    def test_auto_training_disabled(self):
        """Test trigger when auto-training is disabled."""
        config = LifecycleConfig(auto_train_on_data_threshold=False)
        trigger = TrainingTrigger(config)
        conditions = TrainingConditions(new_games_count=10000)
        should_train, reason = trigger.should_trigger_training(conditions)
        assert should_train is False
        assert "disabled" in reason.lower()

    def test_low_data_quality(self):
        """Test trigger with low data quality."""
        config = LifecycleConfig()
        trigger = TrainingTrigger(config)
        conditions = TrainingConditions(
            new_games_count=10000,
            data_quality_score=0.3,  # Below 0.5 threshold
        )
        should_train, reason = trigger.should_trigger_training(conditions)
        assert should_train is False
        assert "quality" in reason.lower()

    def test_curriculum_ready_trigger(self):
        """Test trigger when curriculum stage ready."""
        config = LifecycleConfig()
        trigger = TrainingTrigger(config)
        conditions = TrainingConditions(curriculum_stage_ready=True)
        should_train, reason = trigger.should_trigger_training(conditions)
        assert should_train is True
        assert "Curriculum" in reason

    def test_sufficient_games_trigger(self):
        """Test trigger with sufficient new games (fallback logic)."""
        config = LifecycleConfig(min_games_for_training=500)
        # Create trigger and mock signal_computer to None for fallback
        trigger = TrainingTrigger(config)
        trigger._signal_computer = None
        conditions = TrainingConditions(new_games_count=600)
        should_train, reason = trigger.should_trigger_training(conditions)
        assert should_train is True
        assert "games" in reason.lower()

    def test_insufficient_games(self):
        """Test trigger with insufficient games."""
        config = LifecycleConfig(min_games_for_training=500)
        trigger = TrainingTrigger(config)
        trigger._signal_computer = None
        conditions = TrainingConditions(new_games_count=100)
        should_train, reason = trigger.should_trigger_training(conditions)
        assert should_train is False
        assert "Waiting" in reason

    def test_stale_data_trigger(self):
        """Test trigger with stale data."""
        config = LifecycleConfig(
            min_games_for_training=500,
            training_data_staleness_hours=6.0,
        )
        trigger = TrainingTrigger(config)
        trigger._signal_computer = None
        conditions = TrainingConditions(
            new_games_count=300,  # Half of min threshold
            hours_since_last_training=8.0,  # Stale
        )
        should_train, reason = trigger.should_trigger_training(conditions)
        assert should_train is True
        assert "staleness" in reason.lower()

    def test_elo_plateau_trigger(self):
        """Test trigger on Elo plateau with games below threshold.

        When Elo plateau is detected with 100+ games but below min_games_for_training,
        training should be triggered with plateau reason.
        """
        config = LifecycleConfig(min_games_for_training=500)
        trigger = TrainingTrigger(config)
        trigger._signal_computer = None
        # Set games to 150 - above 100 threshold for plateau but below 250 (half of 500)
        conditions = TrainingConditions(
            new_games_count=150,
            elo_plateau_detected=True,
        )
        should_train, reason = trigger.should_trigger_training(conditions)
        assert should_train is True
        assert "plateau" in reason.lower()

    def test_record_training_started(self):
        """Test recording training start."""
        config = LifecycleConfig()
        trigger = TrainingTrigger(config)
        trigger.record_training_started(500)
        assert trigger._last_training_time is not None
        assert trigger._last_games_count == 500

    def test_record_training(self):
        """Test recording training."""
        config = LifecycleConfig()
        trigger = TrainingTrigger(config)
        trigger.record_training(1000)
        assert trigger._last_training_time is not None
        assert trigger._last_games_count == 1000

    def test_get_conditions(self):
        """Test getting current conditions."""
        config = LifecycleConfig()
        trigger = TrainingTrigger(config)
        trigger.record_training(500)

        # Get conditions with 700 current games
        conditions = trigger.get_conditions(
            current_games=700,
            data_quality=0.9,
            elo_plateau=True,
            curriculum_ready=False,
        )
        assert conditions.new_games_count == 200  # 700 - 500
        assert conditions.data_quality_score == 0.9
        assert conditions.elo_plateau_detected is True
        assert conditions.curriculum_stage_ready is False


class TestModelSyncCoordinator:
    """Tests for ModelSyncCoordinator class."""

    def test_initialization(self):
        """Test ModelSyncCoordinator initialization."""
        config = LifecycleConfig()
        sync = ModelSyncCoordinator(config)
        assert sync.config is config
        assert sync._sync_state == {}
        assert sync._last_sync == {}
        assert sync._distributed_coordinator is None

    def test_get_distributed_coordinator_lazy(self):
        """Test lazy loading of distributed coordinator."""
        config = LifecycleConfig()
        sync = ModelSyncCoordinator(config)
        # First access should try to load
        coord = sync._get_distributed_coordinator()
        # May be None if module not available
        assert sync._distributed_coordinator is not None or coord is None

    @pytest.mark.asyncio
    async def test_get_cluster_status_no_aiohttp(self):
        """Test get_cluster_status without aiohttp."""
        config = LifecycleConfig()
        sync = ModelSyncCoordinator(config)

        # Mock HAS_AIOHTTP to False
        with patch('app.integration.model_lifecycle.HAS_AIOHTTP', False):
            status = await sync.get_cluster_status()
            assert "error" in status

    @pytest.mark.asyncio
    async def test_push_model_no_targets(self):
        """Test push_model with no target nodes."""
        config = LifecycleConfig()
        sync = ModelSyncCoordinator(config)
        sync._distributed_coordinator = None  # Disable distributed sync

        with patch('app.integration.model_lifecycle.HAS_AIOHTTP', True):
            with patch.object(sync, 'get_cluster_status', return_value={"nodes": []}):
                results = await sync.push_model(
                    model_path=Path("/fake/model.pth"),
                    model_id="test",
                    version=1,
                )
                assert results == {}

    @pytest.mark.asyncio
    async def test_pull_production_model_no_aiohttp(self):
        """Test pull_production_model without aiohttp."""
        config = LifecycleConfig()
        sync = ModelSyncCoordinator(config)

        with patch('app.integration.model_lifecycle.HAS_AIOHTTP', False):
            result = await sync.pull_production_model(Path("/tmp/model.pth"))
            assert result is None

    @pytest.mark.asyncio
    async def test_broadcast_promotion_no_aiohttp(self):
        """Test broadcast_promotion without aiohttp."""
        config = LifecycleConfig()
        sync = ModelSyncCoordinator(config)

        with patch('app.integration.model_lifecycle.HAS_AIOHTTP', False):
            # Should not raise
            await sync.broadcast_promotion(
                model_id="test",
                version=1,
                stage="production",
                reason="Test promotion",
            )


class TestModelLifecycleManager:
    """Tests for ModelLifecycleManager class."""

    def test_initialization_default_config(self):
        """Test initialization with default config."""
        manager = ModelLifecycleManager()
        assert manager.config is not None
        assert manager.promotion_gate is not None
        assert manager.training_trigger is not None
        assert manager.sync_coordinator is not None

    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        config = LifecycleConfig(min_elo_improvement=50.0)
        manager = ModelLifecycleManager(config)
        assert manager.config.min_elo_improvement == 50.0

    def test_components_initialized(self):
        """Test all components are properly initialized."""
        manager = ModelLifecycleManager()
        assert isinstance(manager.promotion_gate, PromotionGate)
        assert isinstance(manager.training_trigger, TrainingTrigger)
        assert isinstance(manager.sync_coordinator, ModelSyncCoordinator)

    def test_lifecycle_state_tracking(self):
        """Test lifecycle state tracking."""
        manager = ModelLifecycleManager()
        assert manager._lifecycle_state == {}
        assert manager._evaluation_queue == []
        assert manager._pending_promotions == []


class TestIntegration:
    """Integration tests for model lifecycle components."""

    def test_full_promotion_flow(self):
        """Test full promotion flow through staging to production."""
        config = LifecycleConfig(
            min_games_for_staging=50,
            min_games_for_production=100,
            min_elo_improvement=20.0,
        )
        gate = PromotionGate(config)

        # Initial evaluation - should pass staging
        result = EvaluationResult(
            model_id="candidate",
            version=1,
            games_played=100,
            elo=1600.0,
            policy_accuracy=0.6,
        )
        staging_decision, _ = gate.evaluate_for_staging(result)
        assert staging_decision == PromotionDecision.PROMOTE

        # Production evaluation - should pass
        result.games_played = 150
        result.games_vs_production = 100
        result.elo_vs_production = 30.0
        result.win_rate_vs_production = 0.55

        prod = EvaluationResult(model_id="prod", version=0, games_played=500, elo=1570.0)
        prod_decision, _ = gate.evaluate_for_production(result, prod)
        assert prod_decision == PromotionDecision.PROMOTE

    def test_training_to_evaluation_flow(self):
        """Test training trigger to evaluation flow."""
        config = LifecycleConfig(min_games_for_training=500)
        trigger = TrainingTrigger(config)
        trigger._signal_computer = None  # Use fallback logic

        # Simulate accumulating games
        trigger.record_training(1000)

        # Check conditions after more games
        conditions = trigger.get_conditions(
            current_games=1600,
            data_quality=0.95,
        )
        should_train, reason = trigger.should_trigger_training(conditions)
        assert should_train is True
        assert "games" in reason.lower()

        # Record new training
        trigger.record_training_started(1600)

        # Check conditions right after
        conditions = trigger.get_conditions(current_games=1600)
        should_train, _ = trigger.should_trigger_training(conditions)
        assert should_train is False  # No new games

    def test_rollback_scenario(self):
        """Test rollback scenario after regression."""
        config = LifecycleConfig(
            auto_rollback_on_regression=True,
            regression_threshold_elo=-50.0,
        )
        gate = PromotionGate(config)

        current = EvaluationResult(
            model_id="new",
            version=2,
            elo=1500.0,
            win_rate=0.45,
        )
        previous = EvaluationResult(
            model_id="prev",
            version=1,
            elo=1600.0,
            win_rate=0.55,
        )

        should_rollback, reason = gate.check_rollback(current, previous)
        assert should_rollback is True
        assert "Elo regression" in reason
