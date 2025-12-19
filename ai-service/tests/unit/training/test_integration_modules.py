"""Tests for training integration modules.

Tests:
- lifecycle_integration.py
- model_state_machine.py
- thread_integration.py
- event_integration.py
- task_lifecycle_integration.py
- metrics_integration.py
- locking_integration.py
- exception_integration.py
"""

import asyncio
import pytest
import threading
import time
from unittest.mock import Mock, MagicMock, patch, AsyncMock


# =============================================================================
# Model State Machine Tests
# =============================================================================

class TestModelStateMachine:
    """Tests for app.training.model_state_machine."""

    def test_model_state_enum(self):
        """Test ModelState enum values."""
        from app.training.model_state_machine import ModelState

        assert ModelState.TRAINING.value == "training"
        assert ModelState.PRODUCTION.value == "production"
        assert ModelState.ROLLED_BACK.value == "rolled_back"

    def test_lifecycle_singleton(self):
        """Test get_model_lifecycle returns singleton."""
        from app.training.model_state_machine import (
            get_model_lifecycle,
            reset_model_lifecycle,
        )

        reset_model_lifecycle()

        lifecycle1 = get_model_lifecycle()
        lifecycle2 = get_model_lifecycle()

        assert lifecycle1 is lifecycle2

        reset_model_lifecycle()

    def test_register_model(self):
        """Test registering a model."""
        from app.training.model_state_machine import (
            ModelLifecycleStateMachine,
            ModelState,
        )

        lifecycle = ModelLifecycleStateMachine()

        lifecycle.register_model("test_model_1", ModelState.TRAINING)
        record = lifecycle.get_record("test_model_1")

        assert record is not None
        assert record.model_id == "test_model_1"
        assert record.current_state == ModelState.TRAINING

    def test_valid_transition(self):
        """Test valid state transition."""
        from app.training.model_state_machine import (
            ModelLifecycleStateMachine,
            ModelState,
        )

        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("test_model_2", ModelState.TRAINING)

        # TRAINING -> TRAINED is valid
        result = lifecycle.transition("test_model_2", ModelState.TRAINED)
        assert result is True

        record = lifecycle.get_record("test_model_2")
        assert record.current_state == ModelState.TRAINED

    def test_invalid_transition(self):
        """Test invalid state transition raises exception."""
        from app.training.model_state_machine import (
            ModelLifecycleStateMachine,
            ModelState,
            InvalidTransitionError,
        )

        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("test_model_3", ModelState.TRAINING)

        # TRAINING -> PRODUCTION is NOT valid (skips steps)
        with pytest.raises(InvalidTransitionError):
            lifecycle.transition("test_model_3", ModelState.PRODUCTION)

        # State should remain TRAINING
        record = lifecycle.get_record("test_model_3")
        assert record.current_state == ModelState.TRAINING

    def test_transition_history(self):
        """Test transition history is recorded."""
        from app.training.model_state_machine import (
            ModelLifecycleStateMachine,
            ModelState,
        )

        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("test_model_4", ModelState.TRAINING)
        lifecycle.transition("test_model_4", ModelState.TRAINED, reason="Training complete")

        record = lifecycle.get_record("test_model_4")
        assert len(record.history) >= 1

    def test_get_valid_transitions(self):
        """Test getting valid transitions from a state."""
        from app.training.model_state_machine import (
            ModelLifecycleStateMachine,
            ModelState,
        )

        lifecycle = ModelLifecycleStateMachine()
        lifecycle.register_model("test_model_5", ModelState.EVALUATED)

        valid = lifecycle.get_valid_transitions("test_model_5")
        # From EVALUATED, should be able to go to STAGING
        assert ModelState.STAGING in valid


# =============================================================================
# Lifecycle Integration Tests
# =============================================================================

class TestLifecycleIntegration:
    """Tests for app.training.lifecycle_integration."""

    def test_background_eval_service_init(self):
        """Test BackgroundEvalService initialization."""
        from app.training.lifecycle_integration import BackgroundEvalService

        service = BackgroundEvalService(
            model_getter=lambda: {"state_dict": {}},
            eval_interval=1000,
            games_per_eval=10,
        )

        assert service.name == "background_eval"
        assert service.dependencies == []  # No dependencies without real games

    def test_background_selfplay_service_init(self):
        """Test BackgroundSelfplayService initialization."""
        from app.training.lifecycle_integration import BackgroundSelfplayService

        service = BackgroundSelfplayService(
            config={"board": "square8", "players": 2},
        )

        assert service.name == "background_selfplay"
        assert service.dependencies == []

    def test_training_lifecycle_manager(self):
        """Test TrainingLifecycleManager registration."""
        from app.training.lifecycle_integration import TrainingLifecycleManager

        manager = TrainingLifecycleManager()

        eval_service = manager.register_eval_service(
            model_getter=lambda: {},
            eval_interval=500,
        )

        assert eval_service is not None
        assert "background_eval" in manager._services

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check on service."""
        from app.training.lifecycle_integration import BackgroundEvalService
        from app.core.health import HealthStatus

        service = BackgroundEvalService(
            model_getter=lambda: {},
        )

        # Before start, should be unhealthy
        status = await service.check_health()
        assert status.state.value != "healthy"


# =============================================================================
# Thread Integration Tests
# =============================================================================

class TestThreadIntegration:
    """Tests for app.training.thread_integration."""

    def test_spawn_eval_thread(self):
        """Test spawning evaluation thread."""
        from app.training.thread_integration import (
            spawn_eval_thread,
            get_training_thread_spawner,
            reset_training_thread_spawner,
        )

        reset_training_thread_spawner()

        completed = threading.Event()

        def eval_loop():
            completed.set()

        thread = spawn_eval_thread(
            target=eval_loop,
            name="test_eval_thread",
        )

        # Wait for completion
        completed.wait(timeout=2.0)
        assert completed.is_set()

        reset_training_thread_spawner()

    def test_training_thread_group(self):
        """Test TrainingThreadGroup constants."""
        from app.training.thread_integration import TrainingThreadGroup

        assert TrainingThreadGroup.EVALUATION == "evaluation"
        assert TrainingThreadGroup.DATA_LOADING == "data_loading"
        assert TrainingThreadGroup.CHECKPOINTING == "checkpointing"

    def test_training_thread_spawner_stats(self):
        """Test spawner statistics."""
        from app.training.thread_integration import (
            get_training_thread_spawner,
            reset_training_thread_spawner,
        )

        reset_training_thread_spawner()
        spawner = get_training_thread_spawner()

        stats = spawner.get_stats()
        assert "threads_spawned" in stats
        assert "threads_running" in stats

        reset_training_thread_spawner()


# =============================================================================
# Event Integration Tests
# =============================================================================

class TestEventIntegration:
    """Tests for app.training.event_integration."""

    def test_training_topics(self):
        """Test TrainingTopics constants."""
        from app.training.event_integration import TrainingTopics

        assert TrainingTopics.TRAINING_STARTED == "training.started"
        assert TrainingTopics.TRAINING_COMPLETED == "training.completed"
        assert TrainingTopics.EVAL_COMPLETED == "training.eval.completed"
        assert TrainingTopics.CHECKPOINT_SAVED == "training.checkpoint.saved"

    def test_training_event_dataclass(self):
        """Test TrainingEvent dataclass."""
        from app.training.event_integration import TrainingStartedEvent

        event = TrainingStartedEvent(
            topic="training.started",
            config_key="square8_2p",
            job_id="job-123",
            total_epochs=100,
            batch_size=256,
        )

        assert event.config_key == "square8_2p"
        assert event.total_epochs == 100

    def test_evaluation_event_dataclass(self):
        """Test EvaluationCompletedEvent dataclass."""
        from app.training.event_integration import EvaluationCompletedEvent

        event = EvaluationCompletedEvent(
            topic="training.eval.completed",
            config_key="square8_2p",
            elo=1650,
            win_rate=0.65,
            games_played=100,
            passes_gating=True,
        )

        assert event.elo == 1650
        assert event.passes_gating is True

    @pytest.mark.asyncio
    async def test_publish_training_started(self):
        """Test publishing training started event."""
        from app.training.event_integration import publish_training_started
        from app.core.event_bus import reset_event_bus

        reset_event_bus()

        count = await publish_training_started(
            config_key="test_config",
            job_id="test_job",
            total_epochs=50,
        )

        # Should return number of handlers notified (may be 0)
        assert count >= 0

        reset_event_bus()


# =============================================================================
# Task Lifecycle Integration Tests
# =============================================================================

class TestTaskLifecycleIntegration:
    """Tests for app.training.task_lifecycle_integration."""

    def test_training_task_type_constants(self):
        """Test TrainingTaskType constants."""
        from app.training.task_lifecycle_integration import TrainingTaskType

        assert TrainingTaskType.TRAINING_JOB == "training_job"
        assert TrainingTaskType.EVALUATION == "evaluation"
        assert TrainingTaskType.SELFPLAY == "selfplay"

    def test_register_training_job(self):
        """Test registering a training job."""
        from app.training.task_lifecycle_integration import (
            TrainingTaskTracker,
            TrainingTaskType,
        )

        tracker = TrainingTaskTracker(node_id="test_node")

        info = tracker.register_job(
            job_id="test_job_1",
            config_key="square8_2p",
            auto_heartbeat=False,  # Don't start heartbeat thread
        )

        assert info.task_id == "training:test_job_1"
        assert info.config_key == "square8_2p"
        assert info.task_type == TrainingTaskType.TRAINING_JOB

        tracker.shutdown()

    def test_heartbeat(self):
        """Test sending heartbeat."""
        from app.training.task_lifecycle_integration import TrainingTaskTracker

        tracker = TrainingTaskTracker(node_id="test_node")

        info = tracker.register_job(
            job_id="test_job_2",
            config_key="square8_2p",
            auto_heartbeat=False,
        )

        result = tracker.heartbeat(info.task_id, step=100)
        assert result is True

        task_info = tracker.get_task(info.task_id)
        assert task_info.step == 100

        tracker.shutdown()

    def test_complete_task(self):
        """Test completing a task."""
        from app.training.task_lifecycle_integration import TrainingTaskTracker

        tracker = TrainingTaskTracker(node_id="test_node")

        info = tracker.register_job(
            job_id="test_job_3",
            config_key="square8_2p",
            auto_heartbeat=False,
        )

        tracker.complete(info.task_id, success=True, result={"loss": 0.01})

        # Task should be removed from tracker
        assert tracker.get_task(info.task_id) is None

        tracker.shutdown()


# =============================================================================
# Metrics Integration Tests
# =============================================================================

class TestMetricsIntegration:
    """Tests for app.training.metrics_integration."""

    def test_training_metric_names(self):
        """Test TrainingMetricNames constants."""
        from app.training.metrics_integration import TrainingMetricNames

        assert TrainingMetricNames.STEPS_TOTAL == "training_steps_total"
        assert TrainingMetricNames.CURRENT_LOSS == "training_loss"
        assert TrainingMetricNames.CURRENT_ELO == "model_elo"

    def test_training_metrics_step(self):
        """Test TrainingMetrics.step() method."""
        from app.training.metrics_integration import TrainingMetrics
        from app.metrics.unified_publisher import reset_metrics_publisher

        reset_metrics_publisher()

        # Should not raise
        TrainingMetrics.step(
            config_key="square8_2p",
            step=1000,
            loss=0.01,
            learning_rate=0.001,
        )

        reset_metrics_publisher()

    def test_training_metrics_evaluation(self):
        """Test TrainingMetrics.evaluation() method."""
        from app.training.metrics_integration import TrainingMetrics
        from app.metrics.unified_publisher import reset_metrics_publisher

        reset_metrics_publisher()

        # Should not raise
        TrainingMetrics.evaluation(
            config_key="square8_2p",
            elo=1650,
            win_rate=0.65,
            games_played=100,
        )

        reset_metrics_publisher()

    def test_epoch_timer_context_manager(self):
        """Test TrainingMetrics.epoch_timer() context manager."""
        from app.training.metrics_integration import TrainingMetrics
        from app.metrics.unified_publisher import reset_metrics_publisher

        reset_metrics_publisher()

        with TrainingMetrics.epoch_timer("square8_2p", epoch=5):
            time.sleep(0.01)  # Small delay

        reset_metrics_publisher()


# =============================================================================
# Locking Integration Tests
# =============================================================================

class TestLockingIntegration:
    """Tests for app.training.locking_integration."""

    def test_training_lock_type_constants(self):
        """Test TrainingLockType constants."""
        from app.training.locking_integration import TrainingLockType

        assert TrainingLockType.CHECKPOINT == "checkpoint"
        assert TrainingLockType.PROMOTION == "promotion"
        assert TrainingLockType.SELFPLAY == "selfplay"

    def test_checkpoint_lock_context_manager(self):
        """Test checkpoint_lock context manager."""
        from app.training.locking_integration import checkpoint_lock

        with checkpoint_lock("test_config", timeout=1) as lock:
            # Should acquire lock
            assert lock is not None
            assert lock.is_held()

        # Lock should be released
        assert not lock.is_held()

    def test_training_locks_class(self):
        """Test TrainingLocks static methods."""
        from app.training.locking_integration import TrainingLocks

        with TrainingLocks.checkpoint("test_config_2", timeout=1) as lock:
            assert lock is not None

    def test_is_training_locked(self):
        """Test is_training_locked function."""
        from app.training.locking_integration import (
            TrainingLocks,
            TrainingLockType,
            is_training_locked,
        )

        # Should not be locked initially
        assert not is_training_locked(TrainingLockType.CHECKPOINT, "test_config_3")

        # Lock and check
        with TrainingLocks.checkpoint("test_config_3", timeout=1):
            assert is_training_locked(TrainingLockType.CHECKPOINT, "test_config_3")


# =============================================================================
# Exception Integration Tests
# =============================================================================

class TestExceptionIntegration:
    """Tests for app.training.exception_integration."""

    def test_training_exception_types(self):
        """Test training exception types."""
        from app.training.exception_integration import (
            TrainingError,
            CheckpointError,
            EvaluationError,
            SelfplayError,
            DataLoadError,
        )

        assert issubclass(TrainingError, Exception)
        assert issubclass(CheckpointError, TrainingError)
        assert issubclass(EvaluationError, TrainingError)

    def test_training_retry_policies(self):
        """Test TrainingRetryPolicies."""
        from app.training.exception_integration import TrainingRetryPolicies
        from app.core.error_handler import RetryStrategy

        assert TrainingRetryPolicies.CHECKPOINT_SAVE.max_attempts == 3
        assert TrainingRetryPolicies.DATA_LOAD.strategy == RetryStrategy.EXPONENTIAL_JITTER

    def test_retry_checkpoint_save_decorator(self):
        """Test retry_checkpoint_save decorator."""
        from app.training.exception_integration import retry_checkpoint_save

        call_count = 0

        @retry_checkpoint_save
        def flaky_save():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise IOError("Transient error")
            return True

        result = flaky_save()
        assert result is True
        assert call_count == 2  # One failure, one success

    def test_safe_training_step(self):
        """Test safe_training_step wrapper."""
        from app.training.exception_integration import safe_training_step

        def failing_step():
            raise RuntimeError("GPU error")

        result = safe_training_step(failing_step, default=-1, log_errors=False)
        assert result == -1

    def test_training_error_aggregator(self):
        """Test TrainingErrorAggregator."""
        from app.training.exception_integration import TrainingErrorAggregator

        errors = TrainingErrorAggregator("test operation", max_errors_before_abort=3)

        errors.add(ValueError("Error 1"))
        errors.add(ValueError("Error 2"))

        assert errors.count == 2
        assert not errors.should_abort()

        errors.add(ValueError("Error 3"))
        assert errors.should_abort()

    def test_training_error_context(self):
        """Test training_error_context manager."""
        from app.training.exception_integration import (
            training_error_context,
            TrainingError,
        )

        with pytest.raises(TrainingError):
            with training_error_context("test operation"):
                raise ValueError("Something went wrong")


# =============================================================================
# Thread Spawner Core Tests
# =============================================================================

class TestThreadSpawnerCore:
    """Tests for app.core.thread_spawner."""

    def test_spawn_basic_thread(self):
        """Test spawning a basic thread."""
        from app.core.thread_spawner import ThreadSpawner, ThreadState

        spawner = ThreadSpawner()
        completed = threading.Event()

        def simple_task():
            completed.set()

        thread = spawner.spawn(target=simple_task, name="test_thread")

        completed.wait(timeout=2.0)
        assert completed.is_set()

        # Wait for thread to finish
        thread.join(timeout=1.0)
        assert thread.state in (ThreadState.COMPLETED, ThreadState.RUNNING)

        spawner.shutdown(timeout=2.0)

    def test_thread_restart_on_failure(self):
        """Test thread restarts on failure."""
        from app.core.thread_spawner import ThreadSpawner, RestartPolicy

        spawner = ThreadSpawner()
        call_count = 0
        success = threading.Event()

        def flaky_task():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Transient error")
            success.set()

        thread = spawner.spawn(
            target=flaky_task,
            name="flaky_thread",
            restart_policy=RestartPolicy.ON_FAILURE,
            max_restarts=3,
            restart_delay=0.1,
        )

        success.wait(timeout=5.0)
        assert success.is_set()
        assert call_count == 2

        spawner.shutdown(timeout=2.0)

    def test_spawner_health_check(self):
        """Test spawner health check."""
        from app.core.thread_spawner import ThreadSpawner

        spawner = ThreadSpawner()
        health = spawner.health_check()

        assert "healthy" in health
        assert "running" in health

        spawner.shutdown(timeout=1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
