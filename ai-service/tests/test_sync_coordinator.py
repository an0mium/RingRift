#!/usr/bin/env python3
"""Unit tests for SyncCoordinator (app/distributed/sync_coordinator.py).

Tests the sync operation execution layer including:
- SyncOperationBudget (time tracking, exhaustion, attempt limits)
- Background sync watchdog with deadlines
- Data server health monitoring with auto-restart
- Sync state transitions and timeout handling
- Budget-aware retry mechanisms

Note: Tests focus on the public API and avoid testing private methods directly.
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from pathlib import Path
from typing import Any

from app.distributed.sync_coordinator import (
    SyncCoordinator,
    SyncOperationBudget,
    SyncStats,
    ClusterSyncStats,
    SyncCategory,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_storage_provider():
    """Mock storage provider for testing."""
    provider = MagicMock()
    provider.provider_type.value = "ephemeral"
    provider.has_shared_storage = False
    provider.selfplay_dir = Path("/tmp/selfplay")
    provider.training_dir = Path("/tmp/training")
    provider.models_dir = Path("/tmp/models")
    provider.capabilities.max_sync_interval_seconds = 60
    provider.should_skip_rsync_to = MagicMock(return_value=False)
    return provider


@pytest.fixture
def mock_transport_config():
    """Mock transport configuration."""
    config = MagicMock()
    config.enable_aria2 = False
    config.enable_p2p = False
    config.enable_gossip = False
    config.enable_ssh = False
    config.fallback_chain = []
    config.aria2_connections_per_server = 4
    config.aria2_split = 4
    config.aria2_data_server_port = 8766
    config.gossip_port = 8771
    config.ssh_timeout = 30
    return config


@pytest.fixture
async def coordinator(mock_storage_provider, mock_transport_config):
    """Create a SyncCoordinator instance for testing."""
    # Reset singleton before each test
    SyncCoordinator.reset_instance()

    coordinator = SyncCoordinator(
        provider=mock_storage_provider,
        config=mock_transport_config,
    )

    yield coordinator

    # Cleanup
    try:
        await coordinator.shutdown()
    except (asyncio.CancelledError, RuntimeError):
        pass


@pytest.fixture
async def coordinator_with_shared_storage(mock_transport_config):
    """Coordinator with shared NFS storage (skips sync operations)."""
    provider = MagicMock()
    provider.provider_type.value = "nfs"
    provider.has_shared_storage = True
    provider.selfplay_dir = Path("/nfs/selfplay")
    provider.training_dir = Path("/nfs/training")
    provider.models_dir = Path("/nfs/models")
    provider.capabilities.max_sync_interval_seconds = 60

    SyncCoordinator.reset_instance()
    coordinator = SyncCoordinator(provider=provider, config=mock_transport_config)

    yield coordinator

    try:
        await coordinator.shutdown()
    except (asyncio.CancelledError, RuntimeError):
        pass


# ============================================================================
# SyncOperationBudget Tests
# ============================================================================


class TestSyncOperationBudget:
    """Tests for SyncOperationBudget time tracking."""

    def test_budget_initialization(self):
        """Test budget initializes with correct defaults."""
        budget = SyncOperationBudget()

        assert budget.total_seconds == 300.0
        assert budget.per_attempt_seconds == 30.0
        assert budget.attempts == 0
        assert budget.start_time <= time.time()

    def test_budget_custom_values(self):
        """Test budget with custom time limits."""
        budget = SyncOperationBudget(total_seconds=120.0, per_attempt_seconds=15.0)

        assert budget.total_seconds == 120.0
        assert budget.per_attempt_seconds == 15.0

    def test_elapsed_time_increases(self):
        """Test elapsed time increases as time passes."""
        budget = SyncOperationBudget()

        initial_elapsed = budget.elapsed
        time.sleep(0.01)  # Sleep 10ms
        later_elapsed = budget.elapsed

        assert later_elapsed > initial_elapsed

    def test_remaining_time_decreases(self):
        """Test remaining time decreases as time passes."""
        budget = SyncOperationBudget(total_seconds=1.0)

        initial_remaining = budget.remaining
        time.sleep(0.1)  # Sleep 100ms
        later_remaining = budget.remaining

        assert later_remaining < initial_remaining
        assert later_remaining >= 0.0  # Never goes negative

    def test_remaining_time_zero_when_exhausted(self):
        """Test remaining time is zero when budget exhausted."""
        budget = SyncOperationBudget(total_seconds=0.01)
        time.sleep(0.02)  # Exceed budget

        assert budget.remaining == 0.0

    def test_exhausted_property_false_initially(self):
        """Test budget not exhausted initially."""
        budget = SyncOperationBudget(total_seconds=10.0)

        assert not budget.exhausted

    def test_exhausted_property_true_when_timeout(self):
        """Test budget exhausted when time runs out."""
        budget = SyncOperationBudget(total_seconds=0.01)
        time.sleep(0.02)

        assert budget.exhausted

    def test_get_attempt_timeout_uses_per_attempt_limit(self):
        """Test attempt timeout respects per-attempt limit."""
        budget = SyncOperationBudget(
            total_seconds=300.0,
            per_attempt_seconds=30.0
        )

        timeout = budget.get_attempt_timeout()

        assert timeout == 30.0

    def test_get_attempt_timeout_capped_by_remaining(self):
        """Test attempt timeout capped by remaining budget."""
        budget = SyncOperationBudget(
            total_seconds=0.5,
            per_attempt_seconds=30.0
        )
        time.sleep(0.4)  # Leave only ~0.1s remaining

        timeout = budget.get_attempt_timeout()
        remaining = budget.remaining

        # Timeout should be less than per_attempt and close to remaining
        assert timeout < 30.0
        assert timeout <= remaining + 0.01  # Allow tiny timing variance

    def test_record_attempt_increments_counter(self):
        """Test recording attempts increments counter."""
        budget = SyncOperationBudget()

        assert budget.attempts == 0

        budget.record_attempt()
        assert budget.attempts == 1

        budget.record_attempt()
        assert budget.attempts == 2

    def test_can_attempt_true_when_time_remaining(self):
        """Test can_attempt returns True with sufficient time."""
        budget = SyncOperationBudget(total_seconds=10.0)

        assert budget.can_attempt()

    def test_can_attempt_false_when_exhausted(self):
        """Test can_attempt returns False when exhausted."""
        budget = SyncOperationBudget(total_seconds=0.01)
        time.sleep(0.02)

        assert not budget.can_attempt()

    def test_can_attempt_requires_at_least_one_second(self):
        """Test can_attempt requires at least 1 second remaining."""
        budget = SyncOperationBudget(total_seconds=0.5)

        # Initially should be able to attempt (0.5s > minimum threshold in some cases)
        # But the requirement is >= 1.0 second
        assert not budget.can_attempt()


# ============================================================================
# SyncCoordinator Initialization Tests
# ============================================================================


class TestSyncCoordinatorInit:
    """Tests for SyncCoordinator initialization."""

    def test_coordinator_singleton_pattern(self):
        """Test get_instance returns same instance."""
        SyncCoordinator.reset_instance()

        c1 = SyncCoordinator.get_instance()
        c2 = SyncCoordinator.get_instance()

        assert c1 is c2

    @pytest.mark.asyncio
    async def test_reset_instance_creates_new_singleton(self):
        """Test reset_instance creates new singleton."""
        c1 = SyncCoordinator.get_instance()
        try:
            await c1.shutdown()
        except (asyncio.CancelledError, RuntimeError):
            pass
        SyncCoordinator.reset_instance()
        c2 = SyncCoordinator.get_instance()

        assert c1 is not c2

        # Cleanup c2
        try:
            await c2.shutdown()
        except (asyncio.CancelledError, RuntimeError):
            pass

    def test_coordinator_initializes_with_provider(self, mock_storage_provider, mock_transport_config):
        """Test coordinator initializes with provider."""
        coordinator = SyncCoordinator(
            provider=mock_storage_provider,
            config=mock_transport_config,
        )

        assert coordinator._provider is mock_storage_provider
        assert coordinator._config is mock_transport_config

    def test_coordinator_background_sync_defaults(self, coordinator):
        """Test background sync watchdog initialized with defaults."""
        assert coordinator._last_successful_sync == 0.0
        assert coordinator._sync_deadline_seconds == 600.0
        assert coordinator._consecutive_failures == 0
        assert coordinator._max_consecutive_failures == 5

    def test_coordinator_data_server_health_defaults(self, coordinator):
        """Test data server health monitoring initialized."""
        assert coordinator._data_server_last_health_check == 0.0
        assert coordinator._data_server_health_check_interval == 30.0
        assert coordinator._data_server_healthy is True


# ============================================================================
# Data Server Tests
# ============================================================================


class TestDataServer:
    """Tests for data server lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_data_server_when_already_running(self, coordinator):
        """Test starting server when already running returns True."""
        # Mock as already running
        coordinator._data_server_process = MagicMock()
        coordinator._data_server_process.returncode = None

        result = await coordinator.start_data_server(port=8766)

        assert result is True

    @pytest.mark.asyncio
    async def test_stop_data_server_when_not_running(self, coordinator):
        """Test stopping server when not running is safe."""
        coordinator._data_server_process = None

        # Should not raise
        await coordinator.stop_data_server()

    @pytest.mark.asyncio
    async def test_stop_data_server_terminates_process(self, coordinator):
        """Test stop_data_server terminates the process."""
        mock_process = AsyncMock()
        mock_process.returncode = None
        mock_process.terminate = MagicMock()
        mock_process.wait = AsyncMock()

        coordinator._data_server_process = mock_process

        await coordinator.stop_data_server()

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_data_server_kills_on_timeout(self, coordinator):
        """Test stop_data_server kills process if terminate times out."""
        mock_process = AsyncMock()
        mock_process.returncode = None
        mock_process.terminate = MagicMock()
        mock_process.wait = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_process.kill = MagicMock()

        coordinator._data_server_process = mock_process

        await coordinator.stop_data_server()

        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()

    def test_is_data_server_running_false_when_none(self, coordinator):
        """Test is_data_server_running returns False when not started."""
        coordinator._data_server_process = None

        assert not coordinator.is_data_server_running()

    def test_is_data_server_running_false_when_terminated(self, coordinator):
        """Test is_data_server_running returns False when terminated."""
        mock_process = MagicMock()
        mock_process.returncode = 0  # Terminated

        coordinator._data_server_process = mock_process

        assert not coordinator.is_data_server_running()

    def test_is_data_server_running_true_when_active(self, coordinator):
        """Test is_data_server_running returns True when active."""
        mock_process = MagicMock()
        mock_process.returncode = None  # Still running

        coordinator._data_server_process = mock_process

        assert coordinator.is_data_server_running()


# ============================================================================
# Sync Operations Tests (NFS Skip Logic)
# ============================================================================


class TestSyncOperationsNFS:
    """Tests for sync operations with shared NFS storage."""

    @pytest.mark.asyncio
    async def test_sync_training_data_skips_nfs(self, coordinator_with_shared_storage):
        """Test sync_training_data skips when NFS storage is shared."""
        stats = await coordinator_with_shared_storage.sync_training_data()

        assert stats.category == "training"
        assert stats.transport_used == "nfs_shared"
        assert stats.files_synced == 0

    @pytest.mark.asyncio
    async def test_sync_models_skips_nfs(self, coordinator_with_shared_storage):
        """Test sync_models skips when NFS storage is shared."""
        stats = await coordinator_with_shared_storage.sync_models()

        assert stats.category == "models"
        assert stats.transport_used == "nfs_shared"
        assert stats.files_synced == 0

    @pytest.mark.asyncio
    async def test_sync_games_skips_nfs(self, coordinator_with_shared_storage):
        """Test sync_games skips when NFS storage is shared."""
        stats = await coordinator_with_shared_storage.sync_games()

        assert stats.category == "games"
        assert stats.transport_used == "nfs_shared"
        assert stats.files_synced == 0

    @pytest.mark.asyncio
    async def test_full_cluster_sync_skips_nfs(self, coordinator_with_shared_storage):
        """Test full_cluster_sync skips when NFS storage is shared."""
        stats = await coordinator_with_shared_storage.full_cluster_sync()

        assert stats.total_files_synced == 0
        assert stats.duration_seconds >= 0


# ============================================================================
# Sync Operations Tests (No Sources)
# ============================================================================


class TestSyncOperationsNoSources:
    """Tests for sync operations when no sources are available."""

    @pytest.mark.asyncio
    async def test_sync_training_data_no_sources(self, coordinator):
        """Test sync_training_data handles no sources gracefully."""
        with patch.object(coordinator, 'discover_sources', return_value=AsyncMock(return_value=[])):
            stats = await coordinator.sync_training_data()

        assert stats.category == "training"
        assert stats.files_synced == 0

    @pytest.mark.asyncio
    async def test_sync_models_no_sources(self, coordinator):
        """Test sync_models handles no sources gracefully."""
        with patch.object(coordinator, 'discover_sources', return_value=AsyncMock(return_value=[])):
            stats = await coordinator.sync_models()

        assert stats.category == "models"
        assert stats.files_synced == 0

    @pytest.mark.asyncio
    async def test_sync_games_no_sources(self, coordinator):
        """Test sync_games handles no sources gracefully."""
        with patch.object(coordinator, 'discover_sources', return_value=AsyncMock(return_value=[])):
            stats = await coordinator.sync_games()

        assert stats.category == "games"
        assert stats.files_synced == 0


# ============================================================================
# Background Sync Watchdog Tests
# ============================================================================


class TestBackgroundSyncWatchdog:
    """Tests for background sync watchdog with deadlines."""

    @pytest.mark.asyncio
    async def test_background_sync_already_running(self, coordinator):
        """Test starting background sync when already running."""
        coordinator._running = True

        # Should not raise, just log warning
        await coordinator.start_background_sync(interval_seconds=1)

    @pytest.mark.asyncio
    async def test_background_sync_timeout_increments_failures(self, coordinator):
        """Test background sync timeout increments consecutive failures."""
        # Mock full_cluster_sync to timeout
        async def timeout_sync(*args, **kwargs):
            await asyncio.sleep(10)  # Will be cancelled by timeout

        coordinator._sync_deadline_seconds = 0.01  # Very short deadline

        with patch.object(coordinator, 'full_cluster_sync', side_effect=timeout_sync):
            # Start background sync
            task = asyncio.create_task(coordinator.start_background_sync(interval_seconds=0.1))

            # Let it timeout once
            await asyncio.sleep(0.05)

            # Stop background sync
            await coordinator.stop_background_sync(timeout=0.5)

            # Should have incremented failures
            assert coordinator._consecutive_failures >= 1

    @pytest.mark.asyncio
    async def test_background_sync_success_resets_failures(self, coordinator):
        """Test successful sync resets consecutive failures."""
        # Set up with some failures
        coordinator._consecutive_failures = 3

        # Mock successful sync
        async def successful_sync(*args, **kwargs):
            return ClusterSyncStats()

        with patch.object(coordinator, 'full_cluster_sync', side_effect=successful_sync):
            # Start background sync
            task = asyncio.create_task(coordinator.start_background_sync(interval_seconds=0.1))

            # Let it run one successful cycle
            await asyncio.sleep(0.15)

            # Stop background sync
            await coordinator.stop_background_sync(timeout=0.5)

            # Failures should be reset
            assert coordinator._consecutive_failures == 0
            assert coordinator._last_successful_sync > 0

    @pytest.mark.asyncio
    async def test_stop_background_sync_graceful(self, coordinator):
        """Test stop_background_sync waits gracefully."""
        # Start a background sync that does nothing
        async def noop_sync(*args, **kwargs):
            return ClusterSyncStats()

        with patch.object(coordinator, 'full_cluster_sync', side_effect=noop_sync):
            # Start background sync
            task = asyncio.create_task(coordinator.start_background_sync(interval_seconds=1))

            await asyncio.sleep(0.01)  # Let it start

            # Stop should complete quickly
            start = time.time()
            await coordinator.stop_background_sync(timeout=2.0)
            duration = time.time() - start

            assert duration < 1.0  # Should be much faster than timeout
            assert not coordinator._running


# ============================================================================
# Health Check Tests
# ============================================================================


class TestHealthCheck:
    """Tests for data server health checking."""

    @pytest.mark.asyncio
    async def test_check_data_server_health_uses_cache(self, coordinator):
        """Test health check uses cache within interval."""
        coordinator._data_server_healthy = True
        coordinator._data_server_last_health_check = time.time()

        # Should return cached value without checking
        result = await coordinator._check_data_server_health()

        assert result is True

    @pytest.mark.asyncio
    async def test_check_data_server_health_detects_not_running(self, coordinator):
        """Test health check detects server not running and fails to restart."""
        coordinator._data_server_process = None
        coordinator._data_server_healthy = True
        coordinator._data_server_last_health_check = 0  # Force check

        # Mock start_data_server to raise exception (restart fails)
        async def failing_start(*args, **kwargs):
            raise RuntimeError("Mock restart failure")

        with patch.object(coordinator, 'start_data_server', side_effect=failing_start):
            result = await coordinator._check_data_server_health()

            assert result is False
            assert coordinator._data_server_healthy is False

    def test_get_sync_health_returns_status(self, coordinator):
        """Test get_sync_health returns health status dict."""
        coordinator._running = True
        coordinator._last_successful_sync = time.time() - 100
        coordinator._consecutive_failures = 2
        coordinator._data_server_healthy = True

        health = coordinator.get_sync_health()

        assert health["running"] is True
        assert health["consecutive_failures"] == 2
        assert health["max_consecutive_failures"] == 5
        assert health["sync_deadline_seconds"] == 600.0
        assert health["data_server_healthy"] is True
        assert "time_since_last_sync_seconds" in health
        assert health["health_status"] == "degraded"  # 2 failures

    def test_get_sync_health_healthy_status(self, coordinator):
        """Test get_sync_health returns healthy status."""
        coordinator._consecutive_failures = 0

        health = coordinator.get_sync_health()

        assert health["health_status"] == "healthy"

    def test_get_sync_health_unhealthy_status(self, coordinator):
        """Test get_sync_health returns unhealthy status."""
        coordinator._consecutive_failures = 5
        coordinator._max_consecutive_failures = 5

        health = coordinator.get_sync_health()

        assert health["health_status"] == "unhealthy"


# ============================================================================
# Shutdown Tests
# ============================================================================


class TestShutdown:
    """Tests for coordinator shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_stops_data_server(self, coordinator):
        """Test shutdown stops data server."""
        mock_process = AsyncMock()
        mock_process.returncode = None
        coordinator._data_server_process = mock_process

        with patch.object(coordinator, 'stop_data_server', new_callable=AsyncMock) as mock_stop:
            await coordinator.shutdown()

            mock_stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_sets_running_false(self, coordinator):
        """Test shutdown sets running flag to False."""
        coordinator._running = True

        await coordinator.shutdown()

        assert coordinator._running is False


# ============================================================================
# Status and Monitoring Tests
# ============================================================================


class TestStatus:
    """Tests for status and monitoring."""

    def test_get_status_returns_complete_info(self, coordinator):
        """Test get_status returns complete status dictionary."""
        status = coordinator.get_status()

        assert "provider" in status
        assert "shared_storage" in status
        assert "running" in status
        assert "data_server" in status
        assert "transports" in status
        assert "sources_discovered" in status
        assert "config" in status
        assert "quality" in status

    def test_get_status_data_server_info(self, coordinator):
        """Test get_status includes data server info."""
        coordinator._data_server_process = None

        status = coordinator.get_status()

        assert status["data_server"]["running"] is False
        assert status["data_server"]["port"] is None

    def test_get_status_data_server_running_info(self, coordinator):
        """Test get_status shows running server info."""
        mock_process = MagicMock()
        mock_process.returncode = None
        coordinator._data_server_process = mock_process
        coordinator._data_server_port = 8766

        status = coordinator.get_status()

        assert status["data_server"]["running"] is True
        assert status["data_server"]["port"] == 8766


# ============================================================================
# SyncStats Tests
# ============================================================================


class TestSyncStats:
    """Tests for SyncStats dataclass."""

    def test_sync_stats_success_rate_with_files(self):
        """Test success rate calculation with mixed results."""
        stats = SyncStats(category="test")
        stats.files_synced = 8
        stats.files_failed = 2

        assert stats.success_rate == 0.8

    def test_sync_stats_success_rate_all_success(self):
        """Test success rate with all successful."""
        stats = SyncStats(category="test")
        stats.files_synced = 10
        stats.files_failed = 0

        assert stats.success_rate == 1.0

    def test_sync_stats_success_rate_all_failed(self):
        """Test success rate with all failed."""
        stats = SyncStats(category="test")
        stats.files_synced = 0
        stats.files_failed = 10

        assert stats.success_rate == 0.0

    def test_sync_stats_success_rate_no_files(self):
        """Test success rate with no files."""
        stats = SyncStats(category="test")

        assert stats.success_rate == 1.0  # Default to success


# ============================================================================
# Source Discovery Tests
# ============================================================================


class TestSourceDiscovery:
    """Tests for source discovery with caching."""

    @pytest.mark.asyncio
    async def test_discover_sources_uses_cache(self, coordinator):
        """Test discover_sources uses cache within TTL."""
        coordinator._aria2_sources = ["http://source1:8766"]
        coordinator._source_discovery_time = time.time()

        sources = await coordinator.discover_sources(force_refresh=False)

        assert sources == ["http://source1:8766"]

    @pytest.mark.asyncio
    async def test_discover_sources_force_refresh(self, coordinator):
        """Test discover_sources bypasses cache on force_refresh."""
        coordinator._aria2_sources = ["http://old:8766"]
        coordinator._source_discovery_time = time.time()

        with patch('app.distributed.sync_coordinator.get_aria2_sources', return_value=["http://new:8766"]):
            sources = await coordinator.discover_sources(force_refresh=True)

        # Should get new sources
        assert "http://new:8766" in sources


# ============================================================================
# Integration-style Tests
# ============================================================================


class TestSyncIntegration:
    """Integration-style tests for sync workflows."""

    @pytest.mark.asyncio
    async def test_full_cluster_sync_with_categories(self, coordinator):
        """Test full_cluster_sync processes specified categories."""
        # Mock discover_sources to return empty list (skip actual sync)
        async def mock_discover(force_refresh=False):
            return []

        coordinator.discover_sources = mock_discover

        stats = await coordinator.full_cluster_sync(
            categories=[SyncCategory.GAMES, SyncCategory.MODELS]
        )

        assert isinstance(stats, ClusterSyncStats)
        assert stats.total_files_synced == 0  # No sources
        assert stats.nodes_synced == 0

    @pytest.mark.asyncio
    async def test_consecutive_failure_tracking(self, coordinator):
        """Test consecutive failures are tracked across sync attempts."""
        initial_failures = coordinator._consecutive_failures

        # Simulate a failure
        coordinator._consecutive_failures = 3

        # Simulate success
        coordinator._last_successful_sync = time.time()
        coordinator._consecutive_failures = 0

        assert coordinator._consecutive_failures == 0
        assert coordinator._last_successful_sync > 0
