"""
Tests for app.coordination.event_emitters module.

Tests the centralized event emission functions for all event types.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.event_emitters import (
    _emit_via_router,
    _emit_via_router_sync,
    # Helper functions
    _get_timestamp,
    # Backpressure events
    emit_backpressure_activated,
    emit_backpressure_released,
    # Cache events
    emit_cache_invalidated,
    emit_coordinator_health_degraded,
    emit_coordinator_heartbeat,
    emit_coordinator_shutdown,
    emit_curriculum_rebalanced,
    # Evaluation events
    emit_evaluation_complete,
    emit_handler_failed,
    emit_handler_timeout,
    emit_host_offline,
    # Host/Node events
    emit_host_online,
    emit_model_corrupted,
    emit_node_recovered,
    # Optimization events
    emit_optimization_triggered,
    # Metrics events
    emit_plateau_detected,
    # Promotion events
    emit_promotion_complete,
    # Quality events
    emit_quality_updated,
    emit_regression_detected,
    # Selfplay events
    emit_selfplay_complete,
    # Sync events
    emit_sync_complete,
    emit_task_abandoned,
    # Task events
    emit_task_complete,
    emit_task_orphaned,
    emit_training_complete,
    emit_training_complete_sync,
    emit_training_rollback_completed,
    # Error Recovery events
    emit_training_rollback_needed,
    # Training events
    emit_training_started,
    emit_training_triggered,
)

# ============================================
# Test Fixtures
# ============================================

@pytest.fixture
def mock_stage_bus():
    """Create a mock stage event bus."""
    bus = AsyncMock()
    bus.emit = AsyncMock(return_value=None)
    return bus


@pytest.fixture
def mock_data_bus():
    """Create a mock data event bus."""
    bus = AsyncMock()
    bus.publish = AsyncMock(return_value=None)
    return bus


@pytest.fixture
def mock_router():
    """Create a mock event router."""
    router = AsyncMock()
    router.publish = AsyncMock(return_value=None)
    router.publish_sync = MagicMock(return_value=None)
    return router


# ============================================
# Test Helper Functions
# ============================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_timestamp_format(self):
        """Test timestamp returns ISO format."""
        ts = _get_timestamp()
        # Should be parseable as ISO format
        dt = datetime.fromisoformat(ts)
        assert dt is not None

    def test_get_timestamp_is_recent(self):
        """Test timestamp is recent."""
        ts = _get_timestamp()
        dt = datetime.fromisoformat(ts)
        diff = datetime.now() - dt
        assert diff.total_seconds() < 1.0

    @pytest.mark.asyncio
    async def test_emit_via_router_no_router(self):
        """Test emit_via_router returns False when router unavailable."""
        with patch("app.coordination.event_emitters.HAS_EVENT_ROUTER", False):
            result = await _emit_via_router("test_event", {"key": "value"})
            assert result is False

    @pytest.mark.asyncio
    async def test_emit_via_router_disabled(self):
        """Test emit_via_router returns False when USE_UNIFIED_ROUTER is False."""
        with patch("app.coordination.event_emitters.USE_UNIFIED_ROUTER", False):
            result = await _emit_via_router("test_event", {"key": "value"})
            assert result is False

    @pytest.mark.asyncio
    async def test_emit_via_router_success(self, mock_router):
        """Test emit_via_router success path."""
        with patch("app.coordination.event_emitters.HAS_EVENT_ROUTER", True), \
             patch("app.coordination.event_emitters.USE_UNIFIED_ROUTER", True), \
             patch("app.coordination.event_emitters.get_event_router", return_value=mock_router):
            result = await _emit_via_router("test_event", {"key": "value"}, "test_source")
            assert result is True
            mock_router.publish.assert_called_once()

    def test_emit_via_router_sync_no_router(self):
        """Test emit_via_router_sync returns False when router unavailable."""
        with patch("app.coordination.event_emitters.HAS_EVENT_ROUTER", False):
            result = _emit_via_router_sync("test_event", {"key": "value"})
            assert result is False

    def test_emit_via_router_sync_success(self, mock_router):
        """Test emit_via_router_sync success path."""
        with patch("app.coordination.event_emitters.HAS_EVENT_ROUTER", True), \
             patch("app.coordination.event_emitters.USE_UNIFIED_ROUTER", True), \
             patch("app.coordination.event_emitters.get_event_router", return_value=mock_router):
            result = _emit_via_router_sync("test_event", {"key": "value"}, "test_source")
            assert result is True
            mock_router.publish_sync.assert_called_once()


# ============================================
# Test Training Events
# ============================================

class TestTrainingEvents:
    """Tests for training event emission."""

    @pytest.mark.asyncio
    async def test_emit_training_started_no_stage_events(self):
        """Test emit_training_started returns False without stage events."""
        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", False):
            result = await emit_training_started(
                job_id="job-1",
                board_type="square8",
                num_players=2,
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_emit_training_started_success(self, mock_stage_bus):
        """Test emit_training_started success."""
        mock_result_class = MagicMock()
        mock_stage_event = MagicMock()
        mock_stage_event.TRAINING_STARTED = MagicMock(value="training_started")

        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", True), \
             patch("app.coordination.event_emitters.StageCompletionResult", mock_result_class), \
             patch("app.coordination.event_emitters.StageEvent", mock_stage_event), \
             patch("app.coordination.event_emitters.get_stage_bus", return_value=mock_stage_bus), \
             patch("app.coordination.event_emitters.USE_UNIFIED_ROUTER", False):
            result = await emit_training_started(
                job_id="job-1",
                board_type="square8",
                num_players=2,
                model_version="v1.0",
                node_name="node-1",
            )
            assert result is True
            mock_stage_bus.emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_training_complete_success(self, mock_stage_bus):
        """Test emit_training_complete with success=True."""
        mock_result_class = MagicMock()
        mock_stage_event = MagicMock()
        mock_stage_event.TRAINING_COMPLETE = MagicMock(value="training_complete")
        mock_stage_event.TRAINING_FAILED = MagicMock(value="training_failed")

        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", True), \
             patch("app.coordination.event_emitters.StageCompletionResult", mock_result_class), \
             patch("app.coordination.event_emitters.StageEvent", mock_stage_event), \
             patch("app.coordination.event_emitters.get_stage_bus", return_value=mock_stage_bus), \
             patch("app.coordination.event_emitters.USE_UNIFIED_ROUTER", False):
            result = await emit_training_complete(
                job_id="job-1",
                board_type="square8",
                num_players=2,
                success=True,
                final_loss=0.05,
                final_elo=1650.0,
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_emit_training_complete_failure(self, mock_stage_bus):
        """Test emit_training_complete with success=False."""
        mock_result_class = MagicMock()
        mock_stage_event = MagicMock()
        mock_stage_event.TRAINING_COMPLETE = MagicMock(value="training_complete")
        mock_stage_event.TRAINING_FAILED = MagicMock(value="training_failed")

        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", True), \
             patch("app.coordination.event_emitters.StageCompletionResult", mock_result_class), \
             patch("app.coordination.event_emitters.StageEvent", mock_stage_event), \
             patch("app.coordination.event_emitters.get_stage_bus", return_value=mock_stage_bus), \
             patch("app.coordination.event_emitters.USE_UNIFIED_ROUTER", False):
            result = await emit_training_complete(
                job_id="job-1",
                board_type="square8",
                num_players=2,
                success=False,
            )
            assert result is True

    def test_emit_training_complete_sync_no_stage_events(self):
        """Test emit_training_complete_sync returns False without stage events."""
        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", False):
            result = emit_training_complete_sync(
                job_id="job-1",
                board_type="square8",
                num_players=2,
            )
            assert result is False


# ============================================
# Test Selfplay Events
# ============================================

class TestSelfplayEvents:
    """Tests for selfplay event emission."""

    @pytest.mark.asyncio
    async def test_emit_selfplay_complete_no_stage_events(self):
        """Test emit_selfplay_complete returns False without stage events."""
        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", False):
            result = await emit_selfplay_complete(
                task_id="task-1",
                board_type="square8",
                num_players=2,
                games_generated=1000,
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_emit_selfplay_complete_standard(self, mock_stage_bus):
        """Test emit_selfplay_complete with standard type."""
        mock_result_class = MagicMock()
        mock_stage_event = MagicMock()
        mock_stage_event.SELFPLAY_COMPLETE = MagicMock(value="selfplay_complete")
        mock_stage_event.GPU_SELFPLAY_COMPLETE = MagicMock(value="gpu_selfplay_complete")
        mock_stage_event.CANONICAL_SELFPLAY_COMPLETE = MagicMock(value="canonical_selfplay_complete")

        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", True), \
             patch("app.coordination.event_emitters.StageCompletionResult", mock_result_class), \
             patch("app.coordination.event_emitters.StageEvent", mock_stage_event), \
             patch("app.coordination.event_emitters.get_stage_bus", return_value=mock_stage_bus), \
             patch("app.coordination.event_emitters.USE_UNIFIED_ROUTER", False):
            result = await emit_selfplay_complete(
                task_id="task-1",
                board_type="square8",
                num_players=2,
                games_generated=1000,
                selfplay_type="standard",
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_emit_selfplay_complete_gpu_accelerated(self, mock_stage_bus):
        """Test emit_selfplay_complete with gpu_accelerated type."""
        mock_result_class = MagicMock()
        mock_stage_event = MagicMock()
        mock_stage_event.SELFPLAY_COMPLETE = MagicMock(value="selfplay_complete")
        mock_stage_event.GPU_SELFPLAY_COMPLETE = MagicMock(value="gpu_selfplay_complete")
        mock_stage_event.CANONICAL_SELFPLAY_COMPLETE = MagicMock(value="canonical_selfplay_complete")

        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", True), \
             patch("app.coordination.event_emitters.StageCompletionResult", mock_result_class), \
             patch("app.coordination.event_emitters.StageEvent", mock_stage_event), \
             patch("app.coordination.event_emitters.get_stage_bus", return_value=mock_stage_bus), \
             patch("app.coordination.event_emitters.USE_UNIFIED_ROUTER", False):
            result = await emit_selfplay_complete(
                task_id="task-1",
                board_type="square8",
                num_players=2,
                games_generated=1000,
                selfplay_type="gpu_accelerated",
            )
            assert result is True


# ============================================
# Test Evaluation Events
# ============================================

class TestEvaluationEvents:
    """Tests for evaluation event emission."""

    @pytest.mark.asyncio
    async def test_emit_evaluation_complete_no_stage_events(self):
        """Test emit_evaluation_complete returns False without stage events."""
        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", False):
            result = await emit_evaluation_complete(
                model_id="model-1",
                board_type="square8",
                num_players=2,
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_emit_evaluation_complete_success(self, mock_stage_bus):
        """Test emit_evaluation_complete success."""
        mock_result_class = MagicMock()
        mock_stage_event = MagicMock()
        mock_stage_event.EVALUATION_COMPLETE = MagicMock(value="evaluation_complete")

        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", True), \
             patch("app.coordination.event_emitters.StageCompletionResult", mock_result_class), \
             patch("app.coordination.event_emitters.StageEvent", mock_stage_event), \
             patch("app.coordination.event_emitters.get_stage_bus", return_value=mock_stage_bus), \
             patch("app.coordination.event_emitters.USE_UNIFIED_ROUTER", False):
            result = await emit_evaluation_complete(
                model_id="model-1",
                board_type="square8",
                num_players=2,
                win_rate=0.55,
                elo_delta=25.0,
                games_played=100,
            )
            assert result is True


# ============================================
# Test Promotion Events
# ============================================

class TestPromotionEvents:
    """Tests for promotion event emission."""

    @pytest.mark.asyncio
    async def test_emit_promotion_complete_no_stage_events(self):
        """Test emit_promotion_complete returns False without stage events."""
        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", False):
            result = await emit_promotion_complete(
                model_id="model-1",
                board_type="square8",
                num_players=2,
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_emit_promotion_complete_success(self, mock_stage_bus):
        """Test emit_promotion_complete success."""
        mock_result_class = MagicMock()
        mock_stage_event = MagicMock()
        mock_stage_event.PROMOTION_COMPLETE = MagicMock(value="promotion_complete")

        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", True), \
             patch("app.coordination.event_emitters.StageCompletionResult", mock_result_class), \
             patch("app.coordination.event_emitters.StageEvent", mock_stage_event), \
             patch("app.coordination.event_emitters.get_stage_bus", return_value=mock_stage_bus), \
             patch("app.coordination.event_emitters.USE_UNIFIED_ROUTER", False):
            result = await emit_promotion_complete(
                model_id="model-1",
                board_type="square8",
                num_players=2,
                promotion_type="production",
                elo_improvement=50.0,
            )
            assert result is True


# ============================================
# Test Sync Events
# ============================================

class TestSyncEvents:
    """Tests for sync event emission."""

    @pytest.mark.asyncio
    async def test_emit_sync_complete_no_stage_events(self):
        """Test emit_sync_complete returns False without stage events."""
        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", False):
            result = await emit_sync_complete(
                sync_type="data",
                items_synced=100,
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_emit_sync_complete_success(self, mock_stage_bus):
        """Test emit_sync_complete success."""
        mock_result_class = MagicMock()
        mock_stage_event = MagicMock()
        mock_stage_event.SYNC_COMPLETE = MagicMock(value="sync_complete")

        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", True), \
             patch("app.coordination.event_emitters.StageCompletionResult", mock_result_class), \
             patch("app.coordination.event_emitters.StageEvent", mock_stage_event), \
             patch("app.coordination.event_emitters.get_stage_bus", return_value=mock_stage_bus), \
             patch("app.coordination.event_emitters.USE_UNIFIED_ROUTER", False):
            result = await emit_sync_complete(
                sync_type="data",
                items_synced=100,
                duration_seconds=5.0,
                components=["games", "models"],
            )
            assert result is True


# ============================================
# Test Quality Events
# ============================================

class TestQualityEvents:
    """Tests for quality event emission."""

    @pytest.mark.asyncio
    async def test_emit_quality_updated_no_data_events(self):
        """Test emit_quality_updated returns False without data events."""
        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", False):
            result = await emit_quality_updated(
                board_type="square8",
                num_players=2,
                avg_quality=0.85,
                total_games=10000,
                high_quality_games=8500,
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_emit_quality_updated_success(self, mock_data_bus):
        """Test emit_quality_updated success."""
        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.QUALITY_SCORE_UPDATED = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=mock_data_bus):
            result = await emit_quality_updated(
                board_type="square8",
                num_players=2,
                avg_quality=0.85,
                total_games=10000,
                high_quality_games=8500,
            )
            assert result is True
            mock_data_bus.publish.assert_called_once()


# ============================================
# Test Task Events
# ============================================

class TestTaskEvents:
    """Tests for generic task event emission."""

    @pytest.mark.asyncio
    async def test_emit_task_complete_no_stage_events(self):
        """Test emit_task_complete returns False without stage events."""
        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", False):
            result = await emit_task_complete(
                task_id="task-1",
                task_type="selfplay",
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_emit_task_complete_unknown_type(self, mock_stage_bus):
        """Test emit_task_complete with unknown task type."""
        mock_result_class = MagicMock()
        mock_stage_event = MagicMock()
        mock_stage_event.SELFPLAY_COMPLETE = MagicMock()

        with patch("app.coordination.event_emitters.HAS_STAGE_EVENTS", True), \
             patch("app.coordination.event_emitters.StageCompletionResult", mock_result_class), \
             patch("app.coordination.event_emitters.StageEvent", mock_stage_event), \
             patch("app.coordination.event_emitters.get_stage_bus", return_value=mock_stage_bus), \
             patch("app.coordination.event_emitters.USE_UNIFIED_ROUTER", False):
            result = await emit_task_complete(
                task_id="task-1",
                task_type="unknown_type",
            )
            # Unknown type returns False
            assert result is False


# ============================================
# Test Optimization Events
# ============================================

class TestOptimizationEvents:
    """Tests for optimization event emission."""

    @pytest.mark.asyncio
    async def test_emit_optimization_triggered_no_data_events(self):
        """Test emit_optimization_triggered returns False without data events."""
        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", False):
            result = await emit_optimization_triggered(
                optimization_type="cmaes",
                run_id="run-1",
                reason="plateau_detected",
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_emit_optimization_triggered_cmaes(self, mock_data_bus):
        """Test emit_optimization_triggered with cmaes type."""
        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.CMAES_TRIGGERED = MagicMock()
        mock_event_type.NAS_TRIGGERED = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=mock_data_bus):
            result = await emit_optimization_triggered(
                optimization_type="cmaes",
                run_id="run-1",
                reason="plateau_detected",
                generations=50,
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_emit_optimization_triggered_nas(self, mock_data_bus):
        """Test emit_optimization_triggered with nas type."""
        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.CMAES_TRIGGERED = MagicMock()
        mock_event_type.NAS_TRIGGERED = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=mock_data_bus):
            result = await emit_optimization_triggered(
                optimization_type="nas",
                run_id="run-1",
                reason="architecture_search",
            )
            assert result is True


# ============================================
# Test Metrics Events
# ============================================

class TestMetricsEvents:
    """Tests for metrics event emission."""

    @pytest.mark.asyncio
    async def test_emit_plateau_detected_no_data_events(self):
        """Test emit_plateau_detected returns False without data events."""
        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", False):
            result = await emit_plateau_detected(
                metric_name="val_loss",
                current_value=0.05,
                best_value=0.04,
                epochs_since_improvement=10,
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_emit_plateau_detected_success(self, mock_data_bus):
        """Test emit_plateau_detected success."""
        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.PLATEAU_DETECTED = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=mock_data_bus):
            result = await emit_plateau_detected(
                metric_name="val_loss",
                current_value=0.05,
                best_value=0.04,
                epochs_since_improvement=10,
                plateau_type="loss",
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_emit_regression_detected_success(self, mock_data_bus):
        """Test emit_regression_detected success."""
        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.REGRESSION_DETECTED = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=mock_data_bus):
            result = await emit_regression_detected(
                metric_name="elo",
                current_value=1600.0,
                previous_value=1650.0,
                severity="moderate",
            )
            assert result is True


# ============================================
# Test Backpressure Events
# ============================================

class TestBackpressureEvents:
    """Tests for backpressure event emission."""

    @pytest.mark.asyncio
    async def test_emit_backpressure_activated_no_data_events(self):
        """Test emit_backpressure_activated returns False without data events."""
        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", False):
            result = await emit_backpressure_activated(
                node_id="node-1",
                level="high",
                reason="gpu_memory_full",
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_emit_backpressure_activated_success(self, mock_data_bus):
        """Test emit_backpressure_activated success."""
        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.BACKPRESSURE_ACTIVATED = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=mock_data_bus):
            result = await emit_backpressure_activated(
                node_id="node-1",
                level="high",
                reason="gpu_memory_full",
                resource_type="gpu",
                utilization=0.95,
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_emit_backpressure_released_success(self, mock_data_bus):
        """Test emit_backpressure_released success."""
        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.BACKPRESSURE_RELEASED = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=mock_data_bus):
            result = await emit_backpressure_released(
                node_id="node-1",
                previous_level="high",
                duration_seconds=120.0,
            )
            assert result is True


# ============================================
# Test Cache Events
# ============================================

class TestCacheEvents:
    """Tests for cache event emission."""

    @pytest.mark.asyncio
    async def test_emit_cache_invalidated_no_data_events(self):
        """Test emit_cache_invalidated returns False without data events."""
        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", False):
            result = await emit_cache_invalidated(
                invalidation_type="model",
                target_id="model-1",
                count=5,
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_emit_cache_invalidated_success(self, mock_data_bus):
        """Test emit_cache_invalidated success."""
        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.CACHE_INVALIDATED = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=mock_data_bus):
            result = await emit_cache_invalidated(
                invalidation_type="model",
                target_id="model-1",
                count=5,
                affected_nodes=["node-1", "node-2"],
            )
            assert result is True


# ============================================
# Test Host/Node Events
# ============================================

class TestHostNodeEvents:
    """Tests for host/node event emission."""

    @pytest.mark.asyncio
    async def test_emit_host_online_no_data_events(self):
        """Test emit_host_online returns False without data events."""
        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", False):
            result = await emit_host_online(node_id="node-1")
            assert result is False

    @pytest.mark.asyncio
    async def test_emit_host_online_success(self, mock_data_bus):
        """Test emit_host_online success."""
        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.HOST_ONLINE = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=mock_data_bus):
            result = await emit_host_online(
                node_id="node-1",
                host_type="gh200",
                capabilities={"gpu_count": 1, "memory_gb": 96},
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_emit_host_offline_success(self, mock_data_bus):
        """Test emit_host_offline success."""
        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.HOST_OFFLINE = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=mock_data_bus):
            result = await emit_host_offline(
                node_id="node-1",
                reason="maintenance",
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_emit_node_recovered_success(self, mock_data_bus):
        """Test emit_node_recovered success."""
        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.NODE_RECOVERED = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=mock_data_bus):
            result = await emit_node_recovered(
                node_id="node-1",
                recovery_type="automatic",
                offline_duration_seconds=300.0,
            )
            assert result is True


# ============================================
# Test Error Recovery Events
# ============================================

class TestErrorRecoveryEvents:
    """Tests for error recovery event emission."""

    @pytest.mark.asyncio
    async def test_emit_training_rollback_needed_success(self, mock_data_bus):
        """Test emit_training_rollback_needed success."""
        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.TRAINING_ROLLBACK_NEEDED = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=mock_data_bus):
            result = await emit_training_rollback_needed(
                model_id="model-1",
                reason="regression_detected",
                checkpoint_path="/checkpoints/epoch_10.pt",
                severity="moderate",
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_emit_handler_failed_success(self, mock_data_bus):
        """Test emit_handler_failed success."""
        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.HANDLER_FAILED = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=mock_data_bus):
            result = await emit_handler_failed(
                handler_name="on_training_complete",
                event_type="TRAINING_COMPLETE",
                error="Connection timeout",
                coordinator="training_coordinator",
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_emit_handler_timeout_success(self, mock_data_bus):
        """Test emit_handler_timeout success."""
        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.HANDLER_TIMEOUT = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=mock_data_bus):
            result = await emit_handler_timeout(
                handler_name="on_sync_complete",
                event_type="SYNC_COMPLETE",
                timeout_seconds=30.0,
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_emit_coordinator_health_degraded_success(self, mock_data_bus):
        """Test emit_coordinator_health_degraded success."""
        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.COORDINATOR_HEALTH_DEGRADED = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=mock_data_bus):
            result = await emit_coordinator_health_degraded(
                coordinator_name="training_coordinator",
                reason="high_error_rate",
                health_score=0.5,
                issues=["handler_timeout", "connection_errors"],
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_emit_coordinator_shutdown_success(self, mock_data_bus):
        """Test emit_coordinator_shutdown success."""
        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.COORDINATOR_SHUTDOWN = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=mock_data_bus):
            result = await emit_coordinator_shutdown(
                coordinator_name="selfplay_coordinator",
                reason="graceful",
                remaining_tasks=0,
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_emit_coordinator_heartbeat_success(self, mock_data_bus):
        """Test emit_coordinator_heartbeat success."""
        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.COORDINATOR_HEARTBEAT = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=mock_data_bus):
            result = await emit_coordinator_heartbeat(
                coordinator_name="training_coordinator",
                health_score=1.0,
                active_handlers=2,
                events_processed=1000,
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_emit_task_abandoned_success(self, mock_data_bus):
        """Test emit_task_abandoned success."""
        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.TASK_ABANDONED = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=mock_data_bus):
            result = await emit_task_abandoned(
                task_id="task-1",
                task_type="selfplay",
                node_id="node-1",
                reason="node_shutdown",
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_emit_task_orphaned_success(self, mock_data_bus):
        """Test emit_task_orphaned success."""
        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.TASK_ORPHANED = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=mock_data_bus):
            result = await emit_task_orphaned(
                task_id="task-1",
                task_type="training",
                node_id="node-1",
                last_heartbeat=1234567890.0,
                reason="heartbeat_timeout",
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_emit_model_corrupted_success(self, mock_data_bus):
        """Test emit_model_corrupted success."""
        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.MODEL_CORRUPTED = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=mock_data_bus):
            result = await emit_model_corrupted(
                model_id="model-1",
                model_path="/models/model-1.pt",
                corruption_type="checksum_mismatch",
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_emit_training_rollback_completed_success(self, mock_data_bus):
        """Test emit_training_rollback_completed success."""
        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.TRAINING_ROLLBACK_COMPLETED = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=mock_data_bus):
            result = await emit_training_rollback_completed(
                model_id="model-1",
                checkpoint_path="/checkpoints/epoch_10.pt",
                rollback_from="v1.5",
                reason="regression_detected",
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_emit_curriculum_rebalanced_success(self, mock_data_bus):
        """Test emit_curriculum_rebalanced success."""
        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.CURRICULUM_REBALANCED = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=mock_data_bus):
            result = await emit_curriculum_rebalanced(
                config="square8_2p",
                old_weights={"easy": 0.5, "hard": 0.5},
                new_weights={"easy": 0.3, "hard": 0.7},
                reason="elo_improvement",
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_emit_training_triggered_success(self, mock_data_bus):
        """Test emit_training_triggered success."""
        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.TRAINING_THRESHOLD_REACHED = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=mock_data_bus):
            result = await emit_training_triggered(
                config="square8_2p",
                job_id="job-1",
                trigger_reason="threshold",
                game_count=10000,
                threshold=5000,
            )
            assert result is True


# ============================================
# Test Error Handling
# ============================================

class TestErrorHandling:
    """Tests for error handling in event emission."""

    @pytest.mark.asyncio
    async def test_emit_handles_bus_exception(self):
        """Test that emission handles bus exceptions gracefully."""
        mock_bus = AsyncMock()
        mock_bus.publish = AsyncMock(side_effect=Exception("Bus error"))

        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.QUALITY_SCORE_UPDATED = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=mock_bus):
            result = await emit_quality_updated(
                board_type="square8",
                num_players=2,
                avg_quality=0.85,
                total_games=10000,
                high_quality_games=8500,
            )
            # Should return False on exception, not raise
            assert result is False

    @pytest.mark.asyncio
    async def test_emit_handles_none_bus(self):
        """Test that emission handles None bus gracefully."""
        mock_event_class = MagicMock()
        mock_event_type = MagicMock()
        mock_event_type.QUALITY_SCORE_UPDATED = MagicMock()

        with patch("app.coordination.event_emitters.HAS_DATA_EVENTS", True), \
             patch("app.coordination.event_emitters.DataEvent", mock_event_class), \
             patch("app.coordination.event_emitters.DataEventType", mock_event_type), \
             patch("app.coordination.event_emitters.get_data_bus", return_value=None):
            result = await emit_quality_updated(
                board_type="square8",
                num_players=2,
                avg_quality=0.85,
                total_games=10000,
                high_quality_games=8500,
            )
            assert result is False
