"""Tests for unified_data_plane_daemon.py - Consolidated Data Synchronization.

This module tests the UnifiedDataPlaneDaemon and related classes that
coordinate all data movement across the RingRift cluster.

Test coverage: DataPlaneConfig, DataPlaneStats, EventBridge,
UnifiedDataPlaneDaemon, singleton functions.
"""

import asyncio
import os
import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.unified_data_plane_daemon import (
    DataPlaneConfig,
    DataPlaneStats,
    EventBridge,
    UnifiedDataPlaneDaemon,
    get_data_plane_daemon,
    reset_data_plane_daemon,
)
from app.coordination.data_catalog import DataEntry, DataType
from app.coordination.sync_planner_v2 import SyncPlan, SyncPriority
from app.coordination.transport_manager import Transport
from app.coordination.protocols import CoordinatorStatus


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_catalog():
    """Create a mock DataCatalog."""
    catalog = MagicMock()
    catalog.health_check.return_value = MagicMock(healthy=True)
    catalog.get_total_entries.return_value = 10
    catalog.get_by_config.return_value = []
    catalog.get_manifest.return_value = {}
    return catalog


@pytest.fixture
def mock_planner():
    """Create a mock SyncPlanner."""
    planner = MagicMock()
    planner.start = AsyncMock()
    planner.stop = AsyncMock()
    planner.health_check.return_value = MagicMock(healthy=True)
    planner.plan_for_event.return_value = []
    planner.execute_plan = AsyncMock(return_value=True)
    planner.submit_plan = AsyncMock()
    planner.plan_replication.return_value = []
    planner.get_status.return_value = {"pending_plans": 0}
    return planner


@pytest.fixture
def mock_transport():
    """Create a mock TransportManager."""
    transport = MagicMock()
    transport.health_check.return_value = MagicMock(healthy=True)
    return transport


@pytest.fixture
def config():
    """Create test configuration."""
    return DataPlaneConfig(
        catalog_refresh_interval=1.0,  # Fast for testing
        replication_check_interval=1.0,
        s3_backup_interval=1.0,
        manifest_broadcast_interval=1.0,
        s3_enabled=False,
        owc_enabled=False,
    )


@pytest.fixture
def daemon(config, mock_catalog, mock_planner, mock_transport):
    """Create a test daemon instance with mocked dependencies."""
    return UnifiedDataPlaneDaemon(
        config=config,
        catalog=mock_catalog,
        planner=mock_planner,
        transport=mock_transport,
    )


@pytest.fixture
def sample_entries():
    """Create sample DataEntry objects."""
    return [
        DataEntry(
            path="games/hex8_2p.db",
            data_type=DataType.GAMES,
            config_key="hex8_2p",
            size_bytes=1000000,
            checksum="abc123",
            mtime=time.time(),
            locations={"node-1"},
            primary_location="node-1",
        ),
        DataEntry(
            path="games/hex8_4p.db",
            data_type=DataType.GAMES,
            config_key="hex8_4p",
            size_bytes=2000000,
            checksum="def456",
            mtime=time.time(),
            locations={"node-1", "node-2"},
            primary_location="node-1",
        ),
    ]


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before and after each test."""
    # We don't use the async reset to avoid issues
    import app.coordination.unified_data_plane_daemon as module
    module._data_plane_daemon = None
    yield
    module._data_plane_daemon = None


# =============================================================================
# Test DataPlaneConfig
# =============================================================================


class TestDataPlaneConfig:
    """Tests for DataPlaneConfig dataclass."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = DataPlaneConfig()

        assert config.catalog_refresh_interval == 60.0
        assert config.replication_check_interval == 300.0
        assert config.s3_backup_interval == 3600.0
        assert config.min_replication_factor == 3
        assert config.target_replication_factor == 5
        assert config.max_concurrent_syncs == 5
        assert config.s3_enabled is False
        assert config.owc_enabled is False

    def test_custom_values(self):
        """Config should accept custom values."""
        config = DataPlaneConfig(
            catalog_refresh_interval=30.0,
            min_replication_factor=5,
            s3_enabled=True,
            s3_bucket="my-bucket",
        )

        assert config.catalog_refresh_interval == 30.0
        assert config.min_replication_factor == 5
        assert config.s3_enabled is True
        assert config.s3_bucket == "my-bucket"

    def test_from_env(self, monkeypatch):
        """from_env should read environment variables."""
        monkeypatch.setenv("RINGRIFT_CATALOG_REFRESH_INTERVAL", "120")
        monkeypatch.setenv("RINGRIFT_MIN_REPLICATION", "7")
        monkeypatch.setenv("RINGRIFT_S3_BACKUP_ENABLED", "true")
        monkeypatch.setenv("RINGRIFT_S3_BUCKET", "env-bucket")

        config = DataPlaneConfig.from_env()

        assert config.catalog_refresh_interval == 120.0
        assert config.min_replication_factor == 7
        assert config.s3_enabled is True
        assert config.s3_bucket == "env-bucket"

    def test_from_env_defaults(self):
        """from_env should use defaults for missing env vars."""
        config = DataPlaneConfig.from_env()

        # Should have default values
        assert config.catalog_refresh_interval == 60.0
        assert config.min_replication_factor == 3


# =============================================================================
# Test DataPlaneStats
# =============================================================================


class TestDataPlaneStats:
    """Tests for DataPlaneStats dataclass."""

    def test_default_values(self):
        """Stats should initialize to zero."""
        stats = DataPlaneStats()

        assert stats.events_received == 0
        assert stats.events_processed == 0
        assert stats.syncs_initiated == 0
        assert stats.syncs_completed == 0
        assert stats.bytes_synced == 0

    def test_to_dict(self):
        """to_dict should include all stats."""
        stats = DataPlaneStats(
            events_received=100,
            events_processed=95,
            events_failed=5,
            syncs_initiated=50,
            syncs_completed=48,
            syncs_failed=2,
            bytes_synced=1000000,
            start_time=time.time() - 3600,  # 1 hour ago
        )

        data = stats.to_dict()

        assert data["events"]["received"] == 100
        assert data["events"]["processed"] == 95
        assert data["events"]["failed"] == 5
        assert data["syncs"]["initiated"] == 50
        assert data["syncs"]["completed"] == 48
        assert data["syncs"]["bytes"] == 1000000
        assert data["uptime_seconds"] >= 3599  # ~1 hour

    def test_to_dict_no_start_time(self):
        """to_dict should handle zero start_time."""
        stats = DataPlaneStats()
        data = stats.to_dict()

        assert data["uptime_seconds"] == 0


# =============================================================================
# Test EventBridge
# =============================================================================


class TestEventBridge:
    """Tests for EventBridge class."""

    def test_init(self):
        """EventBridge should initialize with callback."""
        callback = MagicMock()
        bridge = EventBridge(callback)

        assert bridge._on_event is callback
        assert bridge._running is False
        assert len(bridge._subscribed_events) > 0

    def test_subscribed_events(self):
        """EventBridge should have correct event subscriptions."""
        bridge = EventBridge(MagicMock())

        assert "SELFPLAY_COMPLETE" in bridge._subscribed_events
        assert "TRAINING_COMPLETED" in bridge._subscribed_events
        assert "MODEL_PROMOTED" in bridge._subscribed_events
        assert "ORPHAN_GAMES_DETECTED" in bridge._subscribed_events

    @pytest.mark.asyncio
    async def test_start_no_event_bus(self):
        """start should handle missing event bus gracefully."""
        bridge = EventBridge(MagicMock())

        with patch(
            "app.coordination.event_router.get_event_bus",
            side_effect=ImportError,
        ):
            await bridge.start()

        assert bridge._running is True

    @pytest.mark.asyncio
    async def test_stop(self):
        """stop should clear subscriptions."""
        bridge = EventBridge(MagicMock())
        bridge._running = True
        bridge._subscriptions = ["EVENT_1", "EVENT_2"]

        await bridge.stop()

        assert bridge._running is False
        assert len(bridge._subscriptions) == 0

    @pytest.mark.asyncio
    async def test_handle_event_dict(self):
        """_handle_event should process dict events."""
        callback = MagicMock()
        bridge = EventBridge(callback)

        event = {
            "event_type": "TEST_EVENT",
            "payload": {"key": "value"},
        }

        await bridge._handle_event(event)

        callback.assert_called_once_with("TEST_EVENT", {"key": "value"})

    @pytest.mark.asyncio
    async def test_handle_event_object(self):
        """_handle_event should process object events."""
        callback = MagicMock()
        bridge = EventBridge(callback)

        @dataclass
        class MockEvent:
            event_type: str
            payload: dict

        event = MockEvent(event_type="OBJECT_EVENT", payload={"data": 123})

        await bridge._handle_event(event)

        callback.assert_called_once_with("OBJECT_EVENT", {"data": 123})

    def test_emit_success(self):
        """emit should call safe_emit_event."""
        bridge = EventBridge(MagicMock())

        with patch(
            "app.coordination.event_router.safe_emit_event"
        ) as mock_emit:
            bridge.emit("TEST_EVENT", {"key": "value"})

            mock_emit.assert_called_once_with(
                "TEST_EVENT",
                {"key": "value"},
                source="UnifiedDataPlane",
            )

    def test_emit_no_router(self):
        """emit should handle missing event router."""
        bridge = EventBridge(MagicMock())

        with patch(
            "app.coordination.event_router.safe_emit_event",
            side_effect=ImportError,
        ):
            # Should not raise
            bridge.emit("TEST_EVENT", {"key": "value"})


# =============================================================================
# Test UnifiedDataPlaneDaemon - Initialization
# =============================================================================


class TestUnifiedDataPlaneDaemonInit:
    """Tests for daemon initialization."""

    def test_init_defaults(self, mock_catalog, mock_planner, mock_transport):
        """Daemon should initialize with provided dependencies."""
        daemon = UnifiedDataPlaneDaemon(
            catalog=mock_catalog,
            planner=mock_planner,
            transport=mock_transport,
        )

        assert daemon._catalog is mock_catalog
        assert daemon._planner is mock_planner
        assert daemon._transport is mock_transport
        assert daemon._running is False
        assert daemon._status == CoordinatorStatus.INITIALIZING

    def test_init_custom_config(self, mock_catalog, mock_planner, mock_transport):
        """Daemon should use custom config."""
        config = DataPlaneConfig(min_replication_factor=7)

        daemon = UnifiedDataPlaneDaemon(
            config=config,
            catalog=mock_catalog,
            planner=mock_planner,
            transport=mock_transport,
        )

        assert daemon._config.min_replication_factor == 7

    def test_init_stats(self, daemon):
        """Daemon should initialize stats."""
        assert daemon._stats.events_received == 0
        assert daemon._stats.syncs_initiated == 0


# =============================================================================
# Test UnifiedDataPlaneDaemon - Lifecycle
# =============================================================================


class TestUnifiedDataPlaneDaemonLifecycle:
    """Tests for daemon lifecycle methods."""

    @pytest.mark.asyncio
    async def test_start(self, daemon, mock_planner):
        """start should initialize all components."""
        with patch(
            "app.coordination.unified_data_plane_daemon.register_coordinator"
        ):
            await daemon.start()

        assert daemon._running is True
        assert daemon._status == CoordinatorStatus.RUNNING
        assert daemon._start_time > 0
        mock_planner.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_idempotent(self, daemon):
        """start should be idempotent."""
        with patch(
            "app.coordination.unified_data_plane_daemon.register_coordinator"
        ):
            await daemon.start()
            first_start_time = daemon._start_time

            await daemon.start()

            assert daemon._start_time == first_start_time

    @pytest.mark.asyncio
    async def test_stop(self, daemon, mock_planner):
        """stop should clean up all components."""
        with patch(
            "app.coordination.unified_data_plane_daemon.register_coordinator"
        ):
            await daemon.start()

        with patch(
            "app.coordination.unified_data_plane_daemon.unregister_coordinator"
        ):
            await daemon.stop()

        assert daemon._running is False
        assert daemon._status == CoordinatorStatus.STOPPED
        mock_planner.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, daemon):
        """stop should be idempotent."""
        # Stop without starting
        await daemon.stop()

        assert daemon._running is False


# =============================================================================
# Test UnifiedDataPlaneDaemon - Health Check
# =============================================================================


class TestUnifiedDataPlaneDaemonHealth:
    """Tests for health check."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, daemon):
        """health_check should return healthy when all components healthy."""
        with patch(
            "app.coordination.unified_data_plane_daemon.register_coordinator"
        ):
            await daemon.start()

        result = daemon.health_check()

        assert result.healthy is True
        assert result.status == CoordinatorStatus.RUNNING
        assert "running" in result.details
        assert result.details["running"] is True

    def test_health_check_unhealthy_component(self, daemon, mock_catalog):
        """health_check should detect unhealthy components."""
        mock_catalog.health_check.return_value = MagicMock(healthy=False)
        daemon._status = CoordinatorStatus.RUNNING

        result = daemon.health_check()

        assert result.healthy is False
        assert result.status == CoordinatorStatus.DEGRADED
        assert "catalog" in result.message.lower()

    @pytest.mark.asyncio
    async def test_health_check_includes_stats(self, daemon):
        """health_check should include stats in details."""
        with patch(
            "app.coordination.unified_data_plane_daemon.register_coordinator"
        ):
            await daemon.start()

        daemon._stats.events_received = 50
        daemon._stats.syncs_completed = 10

        result = daemon.health_check()

        assert "stats" in result.details
        assert result.details["stats"]["events"]["received"] == 50
        assert result.details["stats"]["syncs"]["completed"] == 10


# =============================================================================
# Test UnifiedDataPlaneDaemon - Event Handling
# =============================================================================


class TestUnifiedDataPlaneDaemonEvents:
    """Tests for event handling."""

    def test_on_event_updates_stats(self, daemon, mock_planner):
        """_on_event should update event stats."""
        mock_planner.plan_for_event.return_value = []

        daemon._on_event("TEST_EVENT", {"key": "value"})

        assert daemon._stats.events_received == 1
        assert daemon._stats.events_processed == 1

    def test_on_event_creates_plans(self, daemon, mock_planner, sample_entries):
        """_on_event should create plans from planner."""
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=sample_entries,
        )
        mock_planner.plan_for_event.return_value = [plan]

        daemon._on_event("SELFPLAY_COMPLETE", {"config_key": "hex8_2p"})

        mock_planner.plan_for_event.assert_called_once_with(
            "SELFPLAY_COMPLETE", {"config_key": "hex8_2p"}
        )

    def test_on_event_error_handling(self, daemon, mock_planner):
        """_on_event should handle errors gracefully."""
        mock_planner.plan_for_event.side_effect = Exception("Planner error")

        daemon._on_event("TEST_EVENT", {})

        assert daemon._stats.events_received == 1
        assert daemon._stats.events_failed == 1

    @pytest.mark.asyncio
    async def test_execute_plan_success(self, daemon, mock_planner, sample_entries):
        """_execute_plan should update stats on success."""
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=sample_entries,
        )
        mock_planner.execute_plan = AsyncMock(return_value=True)

        with patch.object(daemon, "_emit_sync_completed") as mock_emit:
            await daemon._execute_plan(plan)

        assert daemon._stats.syncs_initiated == 1
        assert daemon._stats.syncs_completed == 1
        assert daemon._stats.last_sync_time > 0
        mock_emit.assert_called_once_with(plan)

    @pytest.mark.asyncio
    async def test_execute_plan_failure(self, daemon, mock_planner, sample_entries):
        """_execute_plan should handle failures."""
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=sample_entries,
        )
        mock_planner.execute_plan = AsyncMock(return_value=False)

        with patch.object(daemon, "_emit_sync_failed") as mock_emit:
            await daemon._execute_plan(plan)

        assert daemon._stats.syncs_initiated == 1
        assert daemon._stats.syncs_failed == 1
        mock_emit.assert_called_once_with(plan)


# =============================================================================
# Test UnifiedDataPlaneDaemon - Public API
# =============================================================================


class TestUnifiedDataPlaneDaemonPublicAPI:
    """Tests for public API methods."""

    @pytest.mark.asyncio
    async def test_trigger_priority_sync_success(
        self, daemon, mock_catalog, mock_planner, sample_entries
    ):
        """trigger_priority_sync should create and submit plan."""
        mock_catalog.get_by_config.return_value = sample_entries

        with patch(
            "app.config.cluster_config.get_cluster_nodes"
        ) as mock_nodes:
            mock_node = MagicMock()
            mock_node.is_gpu_node = True
            mock_nodes.return_value = {
                "node-1": mock_node,
                "node-2": mock_node,
                "node-3": mock_node,
            }

            result = await daemon.trigger_priority_sync("hex8_2p")

        assert result is True
        mock_planner.submit_plan.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_priority_sync_no_entries(self, daemon, mock_catalog):
        """trigger_priority_sync should return False with no entries."""
        mock_catalog.get_by_config.return_value = []

        result = await daemon.trigger_priority_sync("unknown_config")

        assert result is False

    @pytest.mark.asyncio
    async def test_trigger_priority_sync_with_source(
        self, daemon, mock_catalog, mock_planner, sample_entries
    ):
        """trigger_priority_sync should use provided source."""
        mock_catalog.get_by_config.return_value = sample_entries

        with patch(
            "app.config.cluster_config.get_cluster_nodes"
        ) as mock_nodes:
            mock_node = MagicMock()
            mock_node.is_gpu_node = True
            mock_nodes.return_value = {"node-2": mock_node}

            await daemon.trigger_priority_sync(
                "hex8_2p",
                source_node="specific-source",
            )

        # Check the plan was created with correct source
        call_args = mock_planner.submit_plan.call_args
        plan = call_args[0][0]
        assert plan.source_node == "specific-source"

    @pytest.mark.asyncio
    async def test_get_status(self, daemon):
        """get_status should return comprehensive status."""
        with patch(
            "app.coordination.unified_data_plane_daemon.register_coordinator"
        ):
            await daemon.start()

        status = daemon.get_status()

        assert "node_id" in status
        assert "running" in status
        assert status["running"] is True
        assert "status" in status
        assert "health" in status
        assert "stats" in status
        assert "config" in status
        assert "catalog" in status

    def test_receive_manifest(self, daemon, mock_catalog):
        """receive_manifest should register entries."""
        mock_catalog.register_from_manifest.return_value = 10

        result = daemon.receive_manifest(
            "peer-1",
            {"file.db": {"size": 100, "mtime": time.time()}},
        )

        assert result == 10
        mock_catalog.register_from_manifest.assert_called_once()


# =============================================================================
# Test UnifiedDataPlaneDaemon - Event Emission
# =============================================================================


class TestUnifiedDataPlaneDaemonEmission:
    """Tests for event emission methods."""

    def test_emit_sync_completed(self, daemon, sample_entries):
        """_emit_sync_completed should emit correct event."""
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=sample_entries,
            reason="selfplay_sync",
            config_key="hex8_2p",
        )

        with patch.object(daemon._event_bridge, "emit") as mock_emit:
            daemon._emit_sync_completed(plan)

        # Should emit DATA_SYNC_COMPLETED
        calls = mock_emit.call_args_list
        assert len(calls) >= 1
        assert calls[0][0][0] == "DATA_SYNC_COMPLETED"
        assert calls[0][0][1]["source_node"] == "node-1"

    def test_emit_sync_completed_selfplay_emits_new_games(
        self, daemon, sample_entries
    ):
        """_emit_sync_completed should emit NEW_GAMES_AVAILABLE for selfplay."""
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=sample_entries,
            reason="selfplay_complete",
            config_key="hex8_2p",
        )

        with patch.object(daemon._event_bridge, "emit") as mock_emit:
            daemon._emit_sync_completed(plan)

        # Should emit both DATA_SYNC_COMPLETED and NEW_GAMES_AVAILABLE
        calls = mock_emit.call_args_list
        event_types = [call[0][0] for call in calls]

        assert "DATA_SYNC_COMPLETED" in event_types
        assert "NEW_GAMES_AVAILABLE" in event_types

    def test_emit_sync_failed(self, daemon, sample_entries):
        """_emit_sync_failed should emit DATA_SYNC_FAILED."""
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=sample_entries,
            error="Connection failed",
        )

        with patch.object(daemon._event_bridge, "emit") as mock_emit:
            daemon._emit_sync_failed(plan)

        mock_emit.assert_called_once()
        call_args = mock_emit.call_args
        assert call_args[0][0] == "DATA_SYNC_FAILED"
        assert call_args[0][1]["error"] == "Connection failed"


# =============================================================================
# Test Singleton Functions
# =============================================================================


class TestSingletonFunctions:
    """Tests for module-level singleton functions."""

    def test_get_data_plane_daemon_returns_singleton(self):
        """get_data_plane_daemon should return same instance."""
        with patch(
            "app.coordination.unified_data_plane_daemon.get_data_catalog"
        ) as mock_cat, patch(
            "app.coordination.unified_data_plane_daemon.get_sync_planner"
        ) as mock_plan, patch(
            "app.coordination.unified_data_plane_daemon.get_transport_manager"
        ) as mock_trans:
            mock_cat.return_value = MagicMock()
            mock_plan.return_value = MagicMock()
            mock_trans.return_value = MagicMock()

            daemon1 = get_data_plane_daemon()
            daemon2 = get_data_plane_daemon()

            assert daemon1 is daemon2


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_concurrent_syncs_limited(self, daemon, mock_planner, sample_entries):
        """Concurrent syncs should be limited by semaphore."""
        # Configure to allow only 2 concurrent syncs
        daemon._sync_semaphore = asyncio.Semaphore(2)

        # Create 5 plans
        plans = [
            SyncPlan(
                source_node="node-1",
                target_nodes=["node-2"],
                entries=sample_entries,
            )
            for _ in range(5)
        ]

        # Track concurrent executions
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        original_execute = mock_planner.execute_plan

        async def tracked_execute(plan):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.1)  # Simulate work
            async with lock:
                current_concurrent -= 1
            return True

        mock_planner.execute_plan = tracked_execute

        # Execute all plans concurrently
        await asyncio.gather(*[daemon._execute_plan(plan) for plan in plans])

        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_catalog_refresh_handles_missing_method(
        self, daemon, mock_catalog
    ):
        """Catalog refresh should handle missing scan_local_directory."""
        # Remove scan_local_directory to simulate older catalog
        del mock_catalog.scan_local_directory

        with patch(
            "app.coordination.unified_data_plane_daemon.register_coordinator"
        ):
            await daemon.start()

        # Wait briefly for refresh loop to run
        await asyncio.sleep(0.1)

        # Should not crash
        await daemon.stop()

    def test_large_manifest_handling(self, daemon, mock_catalog):
        """Should handle large manifests efficiently."""
        # Create a large manifest
        manifest = {
            f"games/game_{i}.db": {"size": 1000, "mtime": time.time()}
            for i in range(1000)
        }

        mock_catalog.register_from_manifest.return_value = 1000

        result = daemon.receive_manifest("peer-1", manifest)

        assert result == 1000

    @pytest.mark.asyncio
    async def test_plan_with_empty_entries(self, daemon, mock_planner):
        """Should handle plans with empty entries."""
        plan = SyncPlan(
            source_node="node-1",
            target_nodes=["node-2"],
            entries=[],  # Empty
        )

        mock_planner.execute_plan = AsyncMock(return_value=True)

        with patch.object(daemon, "_emit_sync_completed"):
            await daemon._execute_plan(plan)

        # Should complete without error
        assert daemon._stats.syncs_completed == 1


# =============================================================================
# Test Background Tasks
# =============================================================================


class TestBackgroundTasks:
    """Tests for background task loops."""

    @pytest.mark.asyncio
    async def test_background_tasks_created(self, daemon):
        """start should create background tasks."""
        with patch(
            "app.coordination.unified_data_plane_daemon.register_coordinator"
        ):
            await daemon.start()

        # Should have at least 3 background tasks
        assert len(daemon._tasks) >= 3

        await daemon.stop()

    @pytest.mark.asyncio
    async def test_s3_task_only_when_enabled(self, daemon):
        """S3 backup task should only be created when enabled."""
        daemon._config.s3_enabled = False

        with patch(
            "app.coordination.unified_data_plane_daemon.register_coordinator"
        ):
            await daemon.start()

        task_names = [t.get_name() for t in daemon._tasks]
        assert "data_plane_s3_backup" not in task_names

        await daemon.stop()

    @pytest.mark.asyncio
    async def test_s3_task_when_enabled(self, config, mock_catalog, mock_planner, mock_transport):
        """S3 backup task should be created when enabled."""
        config.s3_enabled = True
        daemon = UnifiedDataPlaneDaemon(
            config=config,
            catalog=mock_catalog,
            planner=mock_planner,
            transport=mock_transport,
        )

        with patch(
            "app.coordination.unified_data_plane_daemon.register_coordinator"
        ):
            await daemon.start()

        task_names = [t.get_name() for t in daemon._tasks]
        assert "data_plane_s3_backup" in task_names

        await daemon.stop()

    @pytest.mark.asyncio
    async def test_tasks_cancelled_on_stop(self, daemon):
        """stop should cancel all background tasks."""
        with patch(
            "app.coordination.unified_data_plane_daemon.register_coordinator"
        ):
            await daemon.start()

        tasks = daemon._tasks.copy()

        with patch(
            "app.coordination.unified_data_plane_daemon.unregister_coordinator"
        ):
            await daemon.stop()

        # All tasks should be cancelled
        for task in tasks:
            assert task.cancelled() or task.done()
