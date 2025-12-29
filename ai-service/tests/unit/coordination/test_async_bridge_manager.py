"""Unit tests for AsyncBridgeManager - centralized async bridge management.

December 2025: Comprehensive test coverage for shared executor pool and
lifecycle management.
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.async_bridge_manager import (
    AsyncBridgeManager,
    BridgeConfig,
    BridgeStats,
    RegisteredBridge,
    get_bridge_manager,
    get_shared_executor,
    reset_bridge_manager,
    run_in_bridge_pool,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the global bridge manager before and after each test."""
    reset_bridge_manager()
    yield
    reset_bridge_manager()


@pytest.fixture
def manager() -> AsyncBridgeManager:
    """Create a fresh manager instance for testing."""
    return AsyncBridgeManager()


@pytest.fixture
def initialized_manager() -> AsyncBridgeManager:
    """Create and initialize a manager instance."""
    mgr = AsyncBridgeManager()
    mgr.initialize()
    return mgr


@pytest.fixture
def custom_config() -> BridgeConfig:
    """Create a custom configuration."""
    return BridgeConfig(
        max_workers=4,
        thread_name_prefix="test_bridge",
        shutdown_timeout_seconds=10.0,
        health_check_interval_seconds=30.0,
        queue_size_warning_threshold=25,
    )


# =============================================================================
# BridgeConfig Tests
# =============================================================================


class TestBridgeConfig:
    """Tests for BridgeConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BridgeConfig()
        assert config.max_workers == 8
        assert config.thread_name_prefix == "ringrift_bridge"
        assert config.shutdown_timeout_seconds == 30.0
        assert config.health_check_interval_seconds == 60.0
        assert config.queue_size_warning_threshold == 50

    def test_custom_values(self, custom_config: BridgeConfig):
        """Test custom configuration values."""
        assert custom_config.max_workers == 4
        assert custom_config.thread_name_prefix == "test_bridge"
        assert custom_config.shutdown_timeout_seconds == 10.0
        assert custom_config.health_check_interval_seconds == 30.0
        assert custom_config.queue_size_warning_threshold == 25


# =============================================================================
# BridgeStats Tests
# =============================================================================


class TestBridgeStats:
    """Tests for BridgeStats dataclass."""

    def test_default_values(self):
        """Test default statistics values."""
        stats = BridgeStats()
        assert stats.total_tasks_submitted == 0
        assert stats.total_tasks_completed == 0
        assert stats.total_tasks_failed == 0
        assert stats.active_tasks == 0
        assert stats.peak_active_tasks == 0
        assert stats.avg_task_duration_ms == 0.0
        assert stats.last_task_time == 0.0
        assert stats.bridges_registered == 0

    def test_custom_values(self):
        """Test custom statistics values."""
        stats = BridgeStats(
            total_tasks_submitted=100,
            total_tasks_completed=95,
            total_tasks_failed=5,
            active_tasks=3,
            peak_active_tasks=10,
            avg_task_duration_ms=15.5,
            last_task_time=1234567890.0,
            bridges_registered=2,
        )
        assert stats.total_tasks_submitted == 100
        assert stats.total_tasks_completed == 95
        assert stats.total_tasks_failed == 5
        assert stats.active_tasks == 3


# =============================================================================
# RegisteredBridge Tests
# =============================================================================


class TestRegisteredBridge:
    """Tests for RegisteredBridge dataclass."""

    def test_required_fields(self):
        """Test required fields."""
        bridge = RegisteredBridge(
            name="test_bridge",
            bridge=object(),
            registered_at=time.time(),
        )
        assert bridge.name == "test_bridge"
        assert bridge.shutdown_callback is None

    def test_with_shutdown_callback(self):
        """Test with shutdown callback."""
        callback = MagicMock()
        bridge = RegisteredBridge(
            name="test",
            bridge=object(),
            registered_at=time.time(),
            shutdown_callback=callback,
        )
        assert bridge.shutdown_callback is callback


# =============================================================================
# AsyncBridgeManager Initialization Tests
# =============================================================================


class TestAsyncBridgeManagerInit:
    """Tests for AsyncBridgeManager initialization."""

    def test_init_with_default_config(self, manager: AsyncBridgeManager):
        """Test initialization with default config."""
        assert manager.config.max_workers == 8
        assert manager._executor is None
        assert manager._initialized is False
        assert manager._shutting_down is False

    def test_init_with_custom_config(self, custom_config: BridgeConfig):
        """Test initialization with custom config."""
        manager = AsyncBridgeManager(config=custom_config)
        assert manager.config.max_workers == 4
        assert manager.config.thread_name_prefix == "test_bridge"

    def test_initialize_creates_executor(self, manager: AsyncBridgeManager):
        """Test that initialize creates the executor."""
        assert manager._executor is None
        manager.initialize()
        assert manager._executor is not None
        assert isinstance(manager._executor, ThreadPoolExecutor)
        assert manager._initialized is True

    def test_initialize_idempotent(self, manager: AsyncBridgeManager):
        """Test that initialize is idempotent."""
        manager.initialize()
        executor1 = manager._executor
        manager.initialize()  # Should not create new executor
        assert manager._executor is executor1

    @pytest.mark.asyncio
    async def test_initialize_async(self, manager: AsyncBridgeManager):
        """Test async initialization."""
        await manager.initialize_async()
        assert manager._initialized is True
        assert manager._executor is not None


# =============================================================================
# Executor Operations Tests
# =============================================================================


class TestExecutorOperations:
    """Tests for executor operations."""

    def test_get_executor_initializes_if_needed(self, manager: AsyncBridgeManager):
        """Test that get_executor auto-initializes."""
        assert manager._initialized is False
        executor = manager.get_executor()
        assert executor is not None
        assert manager._initialized is True

    def test_get_executor_returns_same_instance(
        self, initialized_manager: AsyncBridgeManager
    ):
        """Test that get_executor returns the same instance."""
        executor1 = initialized_manager.get_executor()
        executor2 = initialized_manager.get_executor()
        assert executor1 is executor2

    @pytest.mark.asyncio
    async def test_run_sync_basic(self, initialized_manager: AsyncBridgeManager):
        """Test running a sync function."""

        def add(a: int, b: int) -> int:
            return a + b

        result = await initialized_manager.run_sync(add, 2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_run_sync_with_kwargs(self, initialized_manager: AsyncBridgeManager):
        """Test running a sync function with kwargs."""

        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        result = await initialized_manager.run_sync(greet, "World", greeting="Hi")
        assert result == "Hi, World!"

    @pytest.mark.asyncio
    async def test_run_sync_updates_stats(self, initialized_manager: AsyncBridgeManager):
        """Test that run_sync updates statistics."""

        def slow_func() -> int:
            time.sleep(0.01)
            return 42

        await initialized_manager.run_sync(slow_func)

        stats = initialized_manager.get_stats()
        assert stats["total_tasks_submitted"] == 1
        assert stats["total_tasks_completed"] == 1
        assert stats["total_tasks_failed"] == 0
        assert stats["avg_task_duration_ms"] > 0

    @pytest.mark.asyncio
    async def test_run_sync_tracks_peak_active(
        self, initialized_manager: AsyncBridgeManager
    ):
        """Test that peak active tasks are tracked."""

        def slow_func() -> int:
            time.sleep(0.05)
            return 1

        # Run multiple tasks concurrently
        tasks = [initialized_manager.run_sync(slow_func) for _ in range(3)]
        await asyncio.gather(*tasks)

        stats = initialized_manager.get_stats()
        assert stats["peak_active_tasks"] >= 1

    @pytest.mark.asyncio
    async def test_run_sync_handles_exception(
        self, initialized_manager: AsyncBridgeManager
    ):
        """Test that exceptions are properly propagated."""

        def failing_func() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await initialized_manager.run_sync(failing_func)

        stats = initialized_manager.get_stats()
        assert stats["total_tasks_failed"] == 1

    @pytest.mark.asyncio
    async def test_run_sync_rejects_when_shutting_down(
        self, initialized_manager: AsyncBridgeManager
    ):
        """Test that run_sync rejects tasks during shutdown."""
        initialized_manager._shutting_down = True

        with pytest.raises(RuntimeError, match="shutting down"):
            await initialized_manager.run_sync(lambda: None)

    @pytest.mark.asyncio
    async def test_run_sync_auto_initializes(self, manager: AsyncBridgeManager):
        """Test that run_sync auto-initializes if needed."""
        assert manager._initialized is False
        result = await manager.run_sync(lambda: 42)
        assert result == 42
        assert manager._initialized is True


# =============================================================================
# Bridge Registration Tests
# =============================================================================


class TestBridgeRegistration:
    """Tests for bridge registration."""

    def test_register_bridge(self, initialized_manager: AsyncBridgeManager):
        """Test registering a bridge."""
        bridge = MagicMock()
        initialized_manager.register_bridge("test", bridge)

        assert "test" in initialized_manager._bridges
        assert initialized_manager._bridges["test"].bridge is bridge
        assert initialized_manager.get_stats()["bridges_registered"] == 1

    def test_register_bridge_with_callback(
        self, initialized_manager: AsyncBridgeManager
    ):
        """Test registering a bridge with shutdown callback."""
        bridge = MagicMock()
        callback = MagicMock()
        initialized_manager.register_bridge("test", bridge, shutdown_callback=callback)

        reg = initialized_manager._bridges["test"]
        assert reg.shutdown_callback is callback

    def test_get_bridge(self, initialized_manager: AsyncBridgeManager):
        """Test retrieving a registered bridge."""
        bridge = MagicMock()
        initialized_manager.register_bridge("test", bridge)

        retrieved = initialized_manager.get_bridge("test")
        assert retrieved is bridge

    def test_get_bridge_not_found(self, initialized_manager: AsyncBridgeManager):
        """Test retrieving a non-existent bridge."""
        result = initialized_manager.get_bridge("nonexistent")
        assert result is None

    def test_unregister_bridge(self, initialized_manager: AsyncBridgeManager):
        """Test unregistering a bridge."""
        bridge = MagicMock()
        initialized_manager.register_bridge("test", bridge)
        initialized_manager.unregister_bridge("test")

        assert "test" not in initialized_manager._bridges
        assert initialized_manager.get_stats()["bridges_registered"] == 0

    def test_unregister_nonexistent_bridge(
        self, initialized_manager: AsyncBridgeManager
    ):
        """Test unregistering a non-existent bridge (should not raise)."""
        initialized_manager.unregister_bridge("nonexistent")  # Should not raise


# =============================================================================
# Shutdown Tests
# =============================================================================


class TestShutdown:
    """Tests for shutdown functionality."""

    @pytest.mark.asyncio
    async def test_shutdown_basic(self, initialized_manager: AsyncBridgeManager):
        """Test basic shutdown."""
        await initialized_manager.shutdown(wait=False)

        assert initialized_manager._shutting_down is True
        assert initialized_manager._initialized is False
        assert initialized_manager._executor is None

    @pytest.mark.asyncio
    async def test_shutdown_calls_bridge_callbacks(
        self, initialized_manager: AsyncBridgeManager
    ):
        """Test that shutdown calls bridge callbacks."""
        callback = MagicMock()
        initialized_manager.register_bridge("test", MagicMock(), shutdown_callback=callback)

        await initialized_manager.shutdown(wait=False)

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_handles_async_callbacks(
        self, initialized_manager: AsyncBridgeManager
    ):
        """Test that shutdown handles async callbacks."""
        callback = AsyncMock()
        initialized_manager.register_bridge("test", MagicMock(), shutdown_callback=callback)

        await initialized_manager.shutdown(wait=False)

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self, initialized_manager: AsyncBridgeManager):
        """Test that shutdown is idempotent."""
        await initialized_manager.shutdown(wait=False)
        await initialized_manager.shutdown(wait=False)  # Should not raise

    @pytest.mark.asyncio
    async def test_shutdown_handles_callback_errors(
        self, initialized_manager: AsyncBridgeManager
    ):
        """Test that shutdown handles callback errors gracefully."""

        def bad_callback() -> None:
            raise RuntimeError("Callback error")

        initialized_manager.register_bridge(
            "test", MagicMock(), shutdown_callback=bad_callback
        )

        # Should not raise
        await initialized_manager.shutdown(wait=False)
        assert initialized_manager._shutting_down is True


# =============================================================================
# Statistics and Health Tests
# =============================================================================


class TestStatsAndHealth:
    """Tests for statistics and health reporting."""

    def test_get_stats(self, initialized_manager: AsyncBridgeManager):
        """Test getting statistics."""
        stats = initialized_manager.get_stats()

        assert "initialized" in stats
        assert "shutting_down" in stats
        assert "max_workers" in stats
        assert "total_tasks_submitted" in stats
        assert "bridge_names" in stats

    def test_get_health_healthy(self, initialized_manager: AsyncBridgeManager):
        """Test health check when healthy."""
        health = initialized_manager.get_health()

        assert health["healthy"] is True
        assert len(health["warnings"]) == 0
        assert "stats" in health

    def test_get_health_not_initialized(self, manager: AsyncBridgeManager):
        """Test health check when not initialized."""
        health = manager.get_health()

        assert health["healthy"] is False
        assert "Not initialized" in health["warnings"]

    def test_get_health_shutting_down(self, initialized_manager: AsyncBridgeManager):
        """Test health check when shutting down."""
        initialized_manager._shutting_down = True
        health = initialized_manager.get_health()

        assert health["healthy"] is False
        assert "Shutting down" in health["warnings"]

    def test_get_health_high_queue_depth(self, initialized_manager: AsyncBridgeManager):
        """Test health check with high queue depth."""
        initialized_manager._stats.active_tasks = 100  # Above threshold
        health = initialized_manager.get_health()

        assert any("High queue depth" in w for w in health["warnings"])

    def test_get_health_high_failure_rate(
        self, initialized_manager: AsyncBridgeManager
    ):
        """Test health check with high failure rate."""
        initialized_manager._stats.total_tasks_submitted = 100
        initialized_manager._stats.total_tasks_failed = 20  # 20% failure rate
        health = initialized_manager.get_health()

        assert any("High failure rate" in w for w in health["warnings"])

    def test_health_check_protocol(self, initialized_manager: AsyncBridgeManager):
        """Test health_check() for CoordinatorProtocol compliance."""
        result = initialized_manager.health_check()

        assert hasattr(result, "healthy")
        assert hasattr(result, "status")
        assert hasattr(result, "message")
        assert hasattr(result, "details")

    def test_health_check_status_running(self, initialized_manager: AsyncBridgeManager):
        """Test health_check returns RUNNING when healthy."""
        from app.coordination.protocols import CoordinatorStatus

        result = initialized_manager.health_check()
        assert result.status == CoordinatorStatus.RUNNING

    def test_health_check_status_initializing(self, manager: AsyncBridgeManager):
        """Test health_check returns INITIALIZING when not initialized."""
        from app.coordination.protocols import CoordinatorStatus

        result = manager.health_check()
        assert result.status == CoordinatorStatus.INITIALIZING

    def test_health_check_status_stopped(self, initialized_manager: AsyncBridgeManager):
        """Test health_check returns STOPPED when shutting down."""
        from app.coordination.protocols import CoordinatorStatus

        initialized_manager._shutting_down = True
        result = initialized_manager.health_check()
        assert result.status == CoordinatorStatus.STOPPED


# =============================================================================
# Singleton Management Tests
# =============================================================================


class TestSingletonManagement:
    """Tests for singleton management."""

    def test_get_bridge_manager_returns_singleton(self):
        """Test that get_bridge_manager returns a singleton."""
        manager1 = get_bridge_manager()
        manager2 = get_bridge_manager()
        assert manager1 is manager2

    def test_get_bridge_manager_accepts_config_on_first_call(self):
        """Test that config is only used on first call."""
        config = BridgeConfig(max_workers=2)
        manager1 = get_bridge_manager(config)
        manager2 = get_bridge_manager(BridgeConfig(max_workers=16))

        # Both should reference the same manager with original config
        assert manager1 is manager2
        assert manager1.config.max_workers == 2

    def test_get_shared_executor(self):
        """Test get_shared_executor convenience function."""
        executor = get_shared_executor()
        assert executor is not None
        assert isinstance(executor, ThreadPoolExecutor)

    def test_reset_bridge_manager(self):
        """Test reset_bridge_manager clears the singleton."""
        manager1 = get_bridge_manager()
        manager1.initialize()

        reset_bridge_manager()

        manager2 = get_bridge_manager()
        assert manager2 is not manager1
        assert manager2._initialized is False

    @pytest.mark.asyncio
    async def test_run_in_bridge_pool(self):
        """Test run_in_bridge_pool convenience function."""

        def compute(x: int) -> int:
            return x * 2

        result = await run_in_bridge_pool(compute, 21)
        assert result == 42


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_initialization(self):
        """Test that concurrent initialization is safe."""
        manager = AsyncBridgeManager()
        results = []

        def init_and_get():
            manager.initialize()
            results.append(manager.get_executor())

        threads = [threading.Thread(target=init_and_get) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same executor
        assert len(set(id(r) for r in results)) == 1

    def test_concurrent_bridge_registration(
        self, initialized_manager: AsyncBridgeManager
    ):
        """Test that concurrent bridge registration is safe."""

        def register_bridge(i: int):
            initialized_manager.register_bridge(f"bridge_{i}", object())

        threads = [threading.Thread(target=register_bridge, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(initialized_manager._bridges) == 10


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_run_sync_with_no_args(self, initialized_manager: AsyncBridgeManager):
        """Test running a function with no arguments."""
        result = await initialized_manager.run_sync(lambda: "no args")
        assert result == "no args"

    @pytest.mark.asyncio
    async def test_run_sync_returns_none(self, initialized_manager: AsyncBridgeManager):
        """Test running a function that returns None."""
        result = await initialized_manager.run_sync(lambda: None)
        assert result is None

    def test_register_bridge_overwrites(self, initialized_manager: AsyncBridgeManager):
        """Test that registering with same name overwrites."""
        bridge1 = MagicMock()
        bridge2 = MagicMock()

        initialized_manager.register_bridge("test", bridge1)
        initialized_manager.register_bridge("test", bridge2)

        assert initialized_manager.get_bridge("test") is bridge2

    def test_stats_bridge_names(self, initialized_manager: AsyncBridgeManager):
        """Test that stats includes bridge names."""
        initialized_manager.register_bridge("alpha", MagicMock())
        initialized_manager.register_bridge("beta", MagicMock())

        stats = initialized_manager.get_stats()
        assert "alpha" in stats["bridge_names"]
        assert "beta" in stats["bridge_names"]

    @pytest.mark.asyncio
    async def test_active_tasks_decremented_on_exception(
        self, initialized_manager: AsyncBridgeManager
    ):
        """Test that active tasks are decremented even on exception."""

        def failing() -> None:
            raise ValueError("oops")

        try:
            await initialized_manager.run_sync(failing)
        except ValueError:
            pass

        stats = initialized_manager.get_stats()
        assert stats["active_tasks"] == 0

    @pytest.mark.asyncio
    async def test_shutdown_with_pending_tasks(
        self, initialized_manager: AsyncBridgeManager
    ):
        """Test shutdown with tasks still running."""

        async def long_task():
            await initialized_manager.run_sync(lambda: time.sleep(0.5))

        # Start a task but don't await it
        task = asyncio.create_task(long_task())

        # Give it a moment to start
        await asyncio.sleep(0.05)

        # Shutdown without waiting
        await initialized_manager.shutdown(wait=False)

        # Cancel the pending task to clean up
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
