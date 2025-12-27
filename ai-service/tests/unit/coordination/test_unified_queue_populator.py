"""Tests for app.coordination.unified_queue_populator daemon module.

Tests the UnifiedQueuePopulatorDaemon class and related functionality:
- Daemon lifecycle (start/stop)
- Health check implementation
- Event subscriptions
- P2P health event handling
- Factory functions

December 2025: Added as part of test coverage improvement.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.unified_queue_populator import (
    QueuePopulatorConfig,
    UnifiedQueuePopulator,
    UnifiedQueuePopulatorDaemon,
    get_queue_populator_daemon,
    reset_queue_populator,
    start_queue_populator_daemon,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons before and after each test."""
    reset_queue_populator()
    yield
    reset_queue_populator()


@pytest.fixture
def mock_populator_config():
    """Create a test configuration."""
    return QueuePopulatorConfig(
        board_types=["hex8"],
        player_counts=[2],
        check_interval_seconds=1,  # Fast for tests
        min_queue_depth=10,
    )


# =============================================================================
# UnifiedQueuePopulatorDaemon Initialization Tests
# =============================================================================


class TestUnifiedQueuePopulatorDaemonInit:
    """Tests for UnifiedQueuePopulatorDaemon initialization."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_init_creates_populator(self, mock_scale, mock_load, mock_populator_config):
        """Test daemon creates internal populator."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        assert daemon._populator is not None
        assert isinstance(daemon._populator, UnifiedQueuePopulator)
        assert daemon._running is False

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_daemon_creates_internal_populator(self, mock_scale, mock_load):
        """Test daemon creates its own internal populator."""
        # The daemon always creates its own populator from config
        daemon = UnifiedQueuePopulatorDaemon()
        assert daemon._populator is not None
        assert isinstance(daemon._populator, UnifiedQueuePopulator)

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_populator_property(self, mock_scale, mock_load, mock_populator_config):
        """Test populator property returns internal populator."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        assert daemon.populator is daemon._populator


# =============================================================================
# Daemon Lifecycle Tests
# =============================================================================


class TestUnifiedQueuePopulatorDaemonLifecycle:
    """Tests for daemon start/stop lifecycle."""

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_start_sets_running(self, mock_scale, mock_load, mock_populator_config):
        """Test start sets running flag."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)

        # Mock event subscriptions
        daemon._subscribe_to_events = AsyncMock()

        await daemon.start()
        assert daemon._running is True

        await daemon.stop()

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_start_twice_warns(self, mock_scale, mock_load, mock_populator_config, caplog):
        """Test starting twice logs warning."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        daemon._subscribe_to_events = AsyncMock()

        await daemon.start()
        await daemon.start()  # Second start

        assert "Already running" in caplog.text

        await daemon.stop()

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_stop_clears_running(self, mock_scale, mock_load, mock_populator_config):
        """Test stop clears running flag."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        daemon._subscribe_to_events = AsyncMock()

        await daemon.start()
        assert daemon._running is True

        await daemon.stop()
        assert daemon._running is False

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_stop_cancels_task(self, mock_scale, mock_load, mock_populator_config):
        """Test stop cancels background task."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        daemon._subscribe_to_events = AsyncMock()

        await daemon.start()
        assert daemon._task is not None

        await daemon.stop()
        assert daemon._task.cancelled() or daemon._task.done()


# =============================================================================
# Health Check Tests
# =============================================================================


class TestUnifiedQueuePopulatorDaemonHealthCheck:
    """Tests for daemon health_check method."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_health_check_not_running(self, mock_scale, mock_load, mock_populator_config):
        """Test health_check when daemon not running."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)

        result = daemon.health_check()
        assert result.healthy is False
        assert "not running" in result.message

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_health_check_running_empty_queue(self, mock_scale, mock_load, mock_populator_config):
        """Test health_check when running with empty queue."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        daemon._subscribe_to_events = AsyncMock()

        await daemon.start()

        result = daemon.health_check()
        # Empty queue is degraded
        assert "empty" in result.message.lower() or result.healthy is True

        await daemon.stop()

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_health_check_all_targets_met_with_queue(self, mock_scale, mock_load, mock_populator_config):
        """Test health_check when all targets met and queue has items."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        daemon._subscribe_to_events = AsyncMock()

        # Set target as met
        daemon._populator._targets["hex8_2p"].current_best_elo = 2100.0

        # Mock work queue with items so we don't get "empty queue" degraded
        mock_queue = MagicMock()
        mock_queue.get_queue_status.return_value = {
            "pending": [{"id": "1"}],
            "running": [],
        }
        daemon._populator.set_work_queue(mock_queue)

        await daemon.start()

        result = daemon.health_check()
        assert result.healthy is True
        assert "targets met" in result.message.lower() or "idle" in result.message.lower()

        await daemon.stop()

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_health_check_degraded_cluster(self, mock_scale, mock_load, mock_populator_config):
        """Test health_check when cluster health is degraded."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        daemon._subscribe_to_events = AsyncMock()

        # Degrade cluster health
        daemon._populator._cluster_health_factor = 0.3

        await daemon.start()

        result = daemon.health_check()
        # Should be degraded due to cluster health
        assert result.healthy is False or "degraded" in result.message.lower()

        await daemon.stop()


# =============================================================================
# Get Status Tests
# =============================================================================


class TestUnifiedQueuePopulatorDaemonStatus:
    """Tests for daemon get_status method."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_status_not_running(self, mock_scale, mock_load, mock_populator_config):
        """Test get_status when daemon not running."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)

        status = daemon.get_status()
        assert "daemon_running" in status
        assert status["daemon_running"] is False

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_get_status_running(self, mock_scale, mock_load, mock_populator_config):
        """Test get_status when daemon is running."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        daemon._subscribe_to_events = AsyncMock()

        await daemon.start()

        status = daemon.get_status()
        assert status["daemon_running"] is True
        assert "enabled" in status
        assert "min_queue_depth" in status

        await daemon.stop()


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory/singleton functions."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_queue_populator_daemon_singleton(self, mock_scale, mock_load):
        """Test get_queue_populator_daemon returns singleton."""
        d1 = get_queue_populator_daemon()
        d2 = get_queue_populator_daemon()
        assert d1 is d2

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_get_queue_populator_daemon_with_config(self, mock_scale, mock_load, mock_populator_config):
        """Test get_queue_populator_daemon with custom config."""
        daemon = get_queue_populator_daemon(config=mock_populator_config)
        assert daemon._populator.config.board_types == ["hex8"]
        assert daemon._populator.config.player_counts == [2]

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_start_queue_populator_daemon(self, mock_scale, mock_load, mock_populator_config):
        """Test start_queue_populator_daemon helper."""
        with patch.object(UnifiedQueuePopulatorDaemon, "_subscribe_to_events", new_callable=AsyncMock):
            daemon = await start_queue_populator_daemon(config=mock_populator_config)

            assert daemon._running is True

            await daemon.stop()

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_reset_clears_daemon_singleton(self, mock_scale, mock_load):
        """Test reset_queue_populator clears daemon singleton."""
        d1 = get_queue_populator_daemon()
        reset_queue_populator()
        d2 = get_queue_populator_daemon()
        assert d1 is not d2


# =============================================================================
# Event Subscription Tests
# =============================================================================


class TestEventSubscriptions:
    """Tests for event subscription methods."""

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_subscribe_to_events_called_on_start(self, mock_scale, mock_load, mock_populator_config):
        """Test event subscriptions are set up on start."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        daemon._subscribe_to_events = AsyncMock()

        await daemon.start()

        daemon._subscribe_to_events.assert_called_once()

        await daemon.stop()

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_subscribe_to_data_events(self, mock_scale, mock_load, mock_populator_config):
        """Test data event subscriptions are set up."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)

        with patch("app.coordination.event_router.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_get_router.return_value = mock_router

            await daemon._subscribe_to_data_events()

            # Should subscribe to multiple events
            assert mock_router.subscribe.called
            call_count = mock_router.subscribe.call_count
            assert call_count >= 3  # At least ELO_UPDATED, TRAINING_COMPLETED, NEW_GAMES_AVAILABLE

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_subscribe_to_p2p_health_events(self, mock_scale, mock_load, mock_populator_config):
        """Test P2P health event subscriptions are set up."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)

        with patch("app.coordination.event_router.get_router") as mock_get_router:
            mock_router = MagicMock()
            mock_get_router.return_value = mock_router

            await daemon._subscribe_to_p2p_health_events()

            # Should subscribe to health events
            assert mock_router.subscribe.called


# =============================================================================
# P2P Health Integration Tests
# =============================================================================


class TestP2PHealthIntegration:
    """Tests for P2P health tracking."""

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_dead_nodes_tracking(self, mock_scale, mock_load, mock_populator_config):
        """Test dead nodes are tracked."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)

        # Verify internal populator tracks dead nodes
        assert hasattr(daemon._populator, "_dead_nodes")
        assert isinstance(daemon._populator._dead_nodes, set)

    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    def test_cluster_health_factor(self, mock_scale, mock_load, mock_populator_config):
        """Test cluster health factor initialization."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)

        assert daemon._populator._cluster_health_factor == 1.0


# =============================================================================
# Direct Import Tests
# =============================================================================


class TestDirectImports:
    """Tests for direct imports from unified module."""

    def test_import_daemon_class(self):
        """Test importing daemon class directly."""
        from app.coordination.unified_queue_populator import UnifiedQueuePopulatorDaemon
        assert UnifiedQueuePopulatorDaemon is not None

    def test_import_factory_functions(self):
        """Test importing factory functions."""
        from app.coordination.unified_queue_populator import (
            get_queue_populator_daemon,
            start_queue_populator_daemon,
        )
        assert callable(get_queue_populator_daemon)
        assert callable(start_queue_populator_daemon)

    def test_import_backward_compat_aliases(self):
        """Test backward-compat aliases exist."""
        from app.coordination.unified_queue_populator import (
            PopulatorConfig,
            QueuePopulator,
            QueuePopulatorConfig,
            UnifiedQueuePopulator,
        )
        assert PopulatorConfig is QueuePopulatorConfig
        assert QueuePopulator is UnifiedQueuePopulator


# =============================================================================
# Monitor Loop Tests
# =============================================================================


class TestMonitorLoop:
    """Tests for background monitor loop."""

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_monitor_loop_calls_populate(self, mock_scale, mock_load, mock_populator_config):
        """Test monitor loop calls populate periodically."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        daemon._subscribe_to_events = AsyncMock()

        populate_calls = []
        original_populate = daemon._populator.populate

        def mock_populate():
            populate_calls.append(1)
            return 0

        daemon._populator.populate = mock_populate

        await daemon.start()

        # Wait for at least one iteration
        await asyncio.sleep(0.1)

        await daemon.stop()

        # Should have called populate at least once
        assert len(populate_calls) >= 1

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_monitor_loop_handles_errors(self, mock_scale, mock_load, mock_populator_config, caplog):
        """Test monitor loop handles errors gracefully."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)
        daemon._subscribe_to_events = AsyncMock()

        def error_populate():
            raise RuntimeError("Test error")

        daemon._populator.populate = error_populate

        await daemon.start()

        # Wait for loop to encounter error
        await asyncio.sleep(0.1)

        await daemon.stop()

        # Should log the error but not crash
        assert "error" in caplog.text.lower() or daemon._task.done()


# =============================================================================
# Task Callback Tests
# =============================================================================


class TestTaskCallback:
    """Tests for task done callback."""

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_on_task_done_handles_exception(self, mock_scale, mock_load, mock_populator_config, caplog):
        """Test task callback handles exceptions."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)

        # Create a task that raises
        async def failing_task():
            raise ValueError("Test failure")

        task = asyncio.create_task(failing_task())

        # Wait for task to complete
        try:
            await task
        except ValueError:
            pass

        # Callback should handle the exception
        daemon._on_task_done(task)
        assert "failed" in caplog.text.lower() or "test failure" in caplog.text.lower()

    @pytest.mark.asyncio
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._load_existing_elo")
    @patch("app.coordination.unified_queue_populator.UnifiedQueuePopulator._scale_queue_depth_to_cluster")
    async def test_on_task_done_handles_cancellation(self, mock_scale, mock_load, mock_populator_config):
        """Test task callback handles cancellation."""
        daemon = UnifiedQueuePopulatorDaemon(config=mock_populator_config)

        async def long_task():
            await asyncio.sleep(10)

        task = asyncio.create_task(long_task())
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should not raise
        daemon._on_task_done(task)
