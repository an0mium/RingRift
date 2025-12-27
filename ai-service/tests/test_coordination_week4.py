#!/usr/bin/env python3
"""Week 4 Tests: Integration & Hardening for Coordination Infrastructure.

Tests the December 2025 coordination consolidation:
- pipeline_actions: Stage action invokers with subprocess handling
- CircuitBreaker: Failover and recovery tests
- sync_bandwidth: Bandwidth allocation and coordination
- daemon_adapters: Daemon lifecycle management
- event_router: Backwards compatibility with unified_event_coordinator
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip if coordination modules not available
pytest.importorskip("app.coordination")


# =============================================================================
# Pipeline Actions Tests
# =============================================================================


class TestPipelineActions:
    """Tests for pipeline action triggers."""

    def test_stage_completion_result_creation(self):
        """Test StageCompletionResult dataclass."""
        from app.coordination.pipeline_actions import StageCompletionResult

        result = StageCompletionResult(
            success=True,
            stage="training",
            iteration=1,
            duration_seconds=123.5,
            output_path="/models/test.pth",
            metadata={"epochs": 50},
        )

        assert result.success is True
        assert result.stage == "training"
        assert result.iteration == 1
        assert result.duration_seconds == 123.5
        assert result.output_path == "/models/test.pth"
        assert result.metadata["epochs"] == 50

    def test_stage_completion_result_to_dict(self):
        """Test StageCompletionResult.to_dict() method."""
        from app.coordination.pipeline_actions import StageCompletionResult

        result = StageCompletionResult(
            success=False,
            stage="evaluation",
            iteration=2,
            error="Test error",
        )

        d = result.to_dict()
        assert d["success"] is False
        assert d["stage"] == "evaluation"
        assert d["iteration"] == 2
        assert d["error"] == "Test error"

    def test_action_config_defaults(self):
        """Test ActionConfig has sensible defaults."""
        from app.coordination.pipeline_actions import ActionConfig

        config = ActionConfig()

        assert config.sync_timeout > 0
        assert config.export_timeout > 0
        assert config.training_timeout > 0
        assert config.evaluation_timeout > 0
        assert config.promotion_timeout > 0
        assert config.python_executable == "python3"

    def test_action_priority_enum(self):
        """Test ActionPriority enum values."""
        from app.coordination.pipeline_actions import ActionPriority

        assert ActionPriority.LOW.value == "low"
        assert ActionPriority.NORMAL.value == "normal"
        assert ActionPriority.HIGH.value == "high"
        assert ActionPriority.CRITICAL.value == "critical"

    @pytest.mark.asyncio
    async def test_trigger_data_sync_handles_exception(self):
        """Test trigger_data_sync handles exceptions gracefully."""
        from app.coordination.pipeline_actions import (
            ActionConfig,
            trigger_data_sync,
        )

        # Use a config with a non-existent script
        config = ActionConfig(sync_script="non_existent_script.py")

        result = await trigger_data_sync(
            board_type="hex8",
            num_players=2,
            iteration=1,
            config=config,
        )

        # Should fail but not raise exception
        assert result.success is False
        assert result.stage == "data_sync"
        assert result.iteration == 1

    @pytest.mark.asyncio
    async def test_trigger_npz_export_handles_exception(self):
        """Test trigger_npz_export handles exceptions gracefully."""
        from app.coordination.pipeline_actions import (
            ActionConfig,
            trigger_npz_export,
        )

        config = ActionConfig(export_script="non_existent_script.py")

        result = await trigger_npz_export(
            board_type="hex8",
            num_players=2,
            iteration=1,
            config=config,
        )

        assert result.success is False
        assert result.stage == "npz_export"


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for CircuitBreaker fault tolerance."""

    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker starts closed."""
        from app.coordination.data_pipeline_orchestrator import (
            CircuitBreaker,
            CircuitBreakerState,
        )

        cb = CircuitBreaker()

        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.is_closed is True
        assert cb.is_open is False
        assert cb.can_execute() is True

    def test_circuit_breaker_opens_after_threshold_failures(self):
        """Test circuit opens after failure threshold."""
        from app.coordination.data_pipeline_orchestrator import (
            CircuitBreaker,
            CircuitBreakerState,
        )

        cb = CircuitBreaker(failure_threshold=3)

        # Record failures up to threshold
        cb.record_failure("stage1", "error1")
        assert cb.state == CircuitBreakerState.CLOSED

        cb.record_failure("stage2", "error2")
        assert cb.state == CircuitBreakerState.CLOSED

        cb.record_failure("stage3", "error3")
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.is_open is True
        assert cb.can_execute() is False

    def test_circuit_breaker_success_resets_failure_count(self):
        """Test success in half-open resets to closed."""
        from app.coordination.data_pipeline_orchestrator import (
            CircuitBreaker,
            CircuitBreakerState,
        )

        cb = CircuitBreaker(failure_threshold=2, reset_timeout_seconds=0.1)

        # Open the circuit
        cb.record_failure("stage1", "error1")
        cb.record_failure("stage2", "error2")
        assert cb.state == CircuitBreakerState.OPEN

        # Wait for reset timeout
        time.sleep(0.15)

        # Should transition to half-open
        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert cb.can_execute() is True

        # Record success to reset
        cb.record_success("stage1")
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.is_closed is True

    def test_circuit_breaker_failure_in_half_open_reopens(self):
        """Test failure in half-open immediately reopens circuit."""
        from app.coordination.data_pipeline_orchestrator import (
            CircuitBreaker,
            CircuitBreakerState,
        )

        cb = CircuitBreaker(failure_threshold=2, reset_timeout_seconds=0.1)

        # Open the circuit
        cb.record_failure("stage1", "error1")
        cb.record_failure("stage2", "error2")

        # Wait for reset timeout
        time.sleep(0.15)
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Record failure - should reopen
        cb.record_failure("stage3", "error3")
        assert cb.state == CircuitBreakerState.OPEN

    def test_circuit_breaker_tracks_failures_by_stage(self):
        """Test circuit breaker tracks failures per stage."""
        from app.coordination.data_pipeline_orchestrator import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=10)  # High threshold

        cb.record_failure("training", "error1")
        cb.record_failure("training", "error2")
        cb.record_failure("evaluation", "error3")

        assert cb._failures_by_stage["training"] == 2
        assert cb._failures_by_stage["evaluation"] == 1

    def test_circuit_breaker_half_open_max_requests(self):
        """Test half-open state limits concurrent requests."""
        from app.coordination.data_pipeline_orchestrator import (
            CircuitBreaker,
            CircuitBreakerState,
        )

        cb = CircuitBreaker(
            failure_threshold=1,
            reset_timeout_seconds=0.1,
            half_open_max_requests=1,
        )

        # Open the circuit
        cb.record_failure("stage1", "error1")
        time.sleep(0.15)

        # Should be half-open
        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert cb.can_execute() is True

        # Simulate request in progress
        cb._half_open_requests = 1
        assert cb.can_execute() is False


# =============================================================================
# Bandwidth Coordination Tests
# =============================================================================


class TestBandwidthCoordination:
    """Tests for bandwidth-coordinated sync."""

    def test_transfer_priority_enum(self):
        """Test TransferPriority enum values."""
        from app.coordination.sync_bandwidth import TransferPriority

        assert TransferPriority.LOW.value == "low"
        assert TransferPriority.NORMAL.value == "normal"
        assert TransferPriority.HIGH.value == "high"
        assert TransferPriority.CRITICAL.value == "critical"

    def test_bandwidth_allocation_expiry(self):
        """Test BandwidthAllocation expiry detection."""
        from app.coordination.sync_bandwidth import (
            BandwidthAllocation,
            TransferPriority,
        )

        # Non-expired allocation
        alloc = BandwidthAllocation(
            host="test-host",
            priority=TransferPriority.NORMAL,
            bwlimit_kbps=10000,
            expires_at=time.time() + 3600,
        )
        assert alloc.is_expired is False

        # Expired allocation
        alloc_expired = BandwidthAllocation(
            host="test-host",
            priority=TransferPriority.NORMAL,
            bwlimit_kbps=10000,
            expires_at=time.time() - 1,
        )
        assert alloc_expired.is_expired is True

    def test_bandwidth_config_defaults(self):
        """Test BandwidthConfig has sensible defaults."""
        from app.coordination.sync_bandwidth import BandwidthConfig

        config = BandwidthConfig()

        assert config.default_bwlimit_kbps > 0
        assert config.max_bwlimit_kbps >= config.default_bwlimit_kbps
        assert config.min_bwlimit_kbps > 0
        assert config.max_concurrent_per_host >= 1
        assert config.max_concurrent_total >= config.max_concurrent_per_host

    def test_bandwidth_manager_singleton(self):
        """Test BandwidthManager singleton pattern."""
        from app.coordination.sync_bandwidth import BandwidthManager

        BandwidthManager.reset_instance()

        manager1 = BandwidthManager.get_instance()
        manager2 = BandwidthManager.get_instance()

        assert manager1 is manager2

        BandwidthManager.reset_instance()

    @pytest.mark.asyncio
    async def test_bandwidth_manager_allocation(self):
        """Test basic bandwidth allocation."""
        from app.coordination.sync_bandwidth import (
            BandwidthManager,
            TransferPriority,
        )

        BandwidthManager.reset_instance()
        manager = BandwidthManager.get_instance()

        allocation = await manager.request_allocation(
            host="test-host",
            priority=TransferPriority.NORMAL,
            timeout=5.0,
        )

        assert allocation is not None
        assert allocation.host == "test-host"
        assert allocation.bwlimit_kbps > 0

        # Release allocation
        await manager.release_allocation(allocation)

        BandwidthManager.reset_instance()

    @pytest.mark.asyncio
    async def test_bandwidth_manager_concurrent_limit(self):
        """Test bandwidth manager respects concurrent limits."""
        from app.coordination.sync_bandwidth import (
            BandwidthConfig,
            BandwidthManager,
            TransferPriority,
        )

        config = BandwidthConfig(max_concurrent_per_host=2)
        BandwidthManager.reset_instance()
        manager = BandwidthManager(config)

        # Allocate up to limit
        alloc1 = await manager.request_allocation("host1", TransferPriority.NORMAL)
        alloc2 = await manager.request_allocation("host1", TransferPriority.NORMAL)

        assert alloc1 is not None
        assert alloc2 is not None

        # Third should fail with short timeout (host limit reached)
        alloc3 = await manager.request_allocation("host1", TransferPriority.NORMAL, timeout=0.5)
        assert alloc3 is None

        # Different host should work
        alloc4 = await manager.request_allocation("host2", TransferPriority.NORMAL)
        assert alloc4 is not None

        # Cleanup
        await manager.release_allocation(alloc1)
        await manager.release_allocation(alloc2)
        await manager.release_allocation(alloc4)

        BandwidthManager.reset_instance()

    @pytest.mark.asyncio
    async def test_bandwidth_manager_priority_multiplier(self):
        """Test high priority gets more bandwidth."""
        from app.coordination.sync_bandwidth import (
            BandwidthConfig,
            BandwidthManager,
            TransferPriority,
        )

        config = BandwidthConfig(
            per_host_limit_kbps=10000,
            priority_multipliers={
                TransferPriority.LOW: 0.5,
                TransferPriority.NORMAL: 1.0,
                TransferPriority.HIGH: 1.5,
                TransferPriority.CRITICAL: 2.0,
            },
        )
        BandwidthManager.reset_instance()
        manager = BandwidthManager(config)

        low_alloc = await manager.request_allocation("host1", TransferPriority.LOW)
        await manager.release_allocation(low_alloc)

        high_alloc = await manager.request_allocation("host1", TransferPriority.HIGH)

        # High priority should get more bandwidth
        assert high_alloc.bwlimit_kbps >= low_alloc.bwlimit_kbps

        await manager.release_allocation(high_alloc)
        BandwidthManager.reset_instance()

    def test_bandwidth_manager_status(self):
        """Test bandwidth manager status reporting."""
        from app.coordination.sync_bandwidth import BandwidthManager

        BandwidthManager.reset_instance()
        manager = BandwidthManager.get_instance()

        status = manager.get_status()

        assert "total_usage_kbps" in status
        assert "total_limit_kbps" in status
        assert "active_transfers" in status
        assert "max_concurrent" in status
        assert "per_host" in status

        BandwidthManager.reset_instance()

    def test_sync_result_dataclass(self):
        """Test SyncResult dataclass."""
        from app.coordination.sync_bandwidth import SyncResult

        result = SyncResult(
            success=True,
            source="/local/path",
            dest="user@remote:/path",
            host="remote-host",
            bytes_transferred=1024000,
            duration_seconds=10.5,
            bwlimit_kbps=10000,
            effective_rate_kbps=9500.0,
        )

        assert result.success is True
        assert result.bytes_transferred == 1024000
        assert result.effective_rate_kbps == 9500.0


# =============================================================================
# Daemon Adapters Tests
# =============================================================================


class TestDaemonAdapters:
    """Tests for daemon adapter infrastructure."""

    def test_daemon_adapter_config_defaults(self):
        """Test DaemonAdapterConfig has sensible defaults."""
        from app.coordination.daemon_adapters import DaemonAdapterConfig

        config = DaemonAdapterConfig()

        assert config.acquire_role is True
        assert config.role_timeout_seconds > 0
        assert config.health_check_interval > 0
        assert config.unhealthy_threshold > 0
        assert config.auto_restart is True
        assert config.max_restarts > 0

    def test_get_available_adapters(self):
        """Test get_available_adapters returns daemon types."""
        from app.coordination.daemon_adapters import get_available_adapters
        from app.coordination.daemon_manager import DaemonType

        adapters = get_available_adapters()

        assert isinstance(adapters, list)
        assert len(adapters) > 0
        assert all(isinstance(a, DaemonType) for a in adapters)

    def test_get_daemon_adapter_returns_adapter(self):
        """Test get_daemon_adapter returns correct adapter type."""
        from app.coordination.daemon_adapters import (
            DistillationDaemonAdapter,
            get_daemon_adapter,
        )
        from app.coordination.daemon_manager import DaemonType

        adapter = get_daemon_adapter(DaemonType.DISTILLATION)

        assert adapter is not None
        assert isinstance(adapter, DistillationDaemonAdapter)
        assert adapter.daemon_type == DaemonType.DISTILLATION

    def test_distillation_adapter_role(self):
        """Test DistillationDaemonAdapter has correct role."""
        from app.coordination.daemon_adapters import DistillationDaemonAdapter
        from app.coordination.orchestrator_registry import OrchestratorRole

        adapter = DistillationDaemonAdapter()

        assert adapter.role == OrchestratorRole.DISTILLATION_LEADER

    def test_promotion_adapter_role(self):
        """Test PromotionDaemonAdapter has correct role."""
        from app.coordination.daemon_adapters import PromotionDaemonAdapter
        from app.coordination.orchestrator_registry import OrchestratorRole

        adapter = PromotionDaemonAdapter()

        assert adapter.role == OrchestratorRole.PROMOTION_LEADER

    def test_daemon_adapter_status(self):
        """Test daemon adapter status reporting."""
        from app.coordination.daemon_adapters import DistillationDaemonAdapter

        adapter = DistillationDaemonAdapter()
        status = adapter.get_status()

        assert "daemon_type" in status
        assert "role" in status
        assert "running" in status
        assert "healthy" in status
        assert status["running"] is False  # Not started yet
        assert status["healthy"] is True  # Default healthy

    def test_register_adapter_class(self):
        """Test registering custom adapter class."""
        from app.coordination.daemon_adapters import (
            DaemonAdapter,
            DaemonAdapterConfig,
            get_daemon_adapter,
            register_adapter_class,
        )
        from app.coordination.daemon_manager import DaemonType

        class CustomAdapter(DaemonAdapter):
            @property
            def daemon_type(self) -> DaemonType:
                return DaemonType.DISTILLATION

            async def _create_daemon(self):
                return MagicMock()

            async def _run_daemon(self, daemon):
                pass

        # Register custom adapter
        original = type(get_daemon_adapter(DaemonType.DISTILLATION))
        register_adapter_class(DaemonType.DISTILLATION, CustomAdapter)

        adapter = get_daemon_adapter(DaemonType.DISTILLATION)
        assert isinstance(adapter, CustomAdapter)

        # Restore original
        register_adapter_class(DaemonType.DISTILLATION, original)


# =============================================================================
# Event Router Backwards Compatibility Tests
# =============================================================================


class TestEventRouterBackwardsCompat:
    """Tests for event_router backwards compatibility with unified_event_coordinator."""

    def test_coordinator_stats_alias(self):
        """Test CoordinatorStats is available from event_router."""
        from app.coordination.event_router import CoordinatorStats

        stats = CoordinatorStats()
        assert hasattr(stats, "events_bridged_data_to_cross")
        assert hasattr(stats, "events_bridged_stage_to_cross")

    def test_get_event_coordinator_alias(self):
        """Test get_event_coordinator returns router."""
        from app.coordination.event_router import (
            UnifiedEventRouter,
            get_event_coordinator,
        )

        coordinator = get_event_coordinator()
        assert isinstance(coordinator, UnifiedEventRouter)

    def test_unified_event_coordinator_alias(self):
        """Test UnifiedEventCoordinator alias exists."""
        from app.coordination.event_router import (
            UnifiedEventCoordinator,
            UnifiedEventRouter,
        )

        assert UnifiedEventCoordinator is UnifiedEventRouter

    def test_register_handler_method(self):
        """Test register_handler method exists (backwards compat)."""
        from app.coordination.event_router import get_router

        router = get_router()
        assert hasattr(router, "register_handler")

        # Test it works
        handler_called = []

        def test_handler(event):
            handler_called.append(event)

        router.register_handler("TEST_EVENT", test_handler)
        # Handler should be registered

    @pytest.mark.asyncio
    async def test_emit_training_started(self):
        """Test emit_training_started function."""
        from app.coordination.event_router import emit_training_started

        # Should not raise
        await emit_training_started(
            config_key="hex8_2p",
            node_name="test-node",
            iteration=1,
        )

    @pytest.mark.asyncio
    async def test_emit_training_completed(self):
        """Test emit_training_completed function."""
        from app.coordination.event_router import emit_training_completed

        await emit_training_completed(
            config_key="hex8_2p",
            model_id="hex8_2p_v1",
            val_loss=0.05,
            epochs=50,
        )

    @pytest.mark.asyncio
    async def test_emit_evaluation_completed(self):
        """Test emit_evaluation_completed function."""
        from app.coordination.event_router import emit_evaluation_completed

        await emit_evaluation_completed(
            model_id="hex8_2p_v1",
            elo=1500.0,
            win_rate=0.75,
            games_played=100,
        )


# =============================================================================
# Integration Tests
# =============================================================================


class TestCoordinationIntegration:
    """Integration tests combining multiple coordination components."""

    @pytest.mark.asyncio
    async def test_pipeline_with_circuit_breaker(self):
        """Test pipeline actions respect circuit breaker state."""
        from app.coordination.data_pipeline_orchestrator import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=2)

        # Simulate failures opening circuit
        cb.record_failure("sync", "error1")
        cb.record_failure("sync", "error2")

        assert cb.is_open is True
        assert cb.can_execute() is False

        # Pipeline should check can_execute before running
        if cb.can_execute():
            pytest.fail("Should not execute when circuit is open")

    @pytest.mark.asyncio
    async def test_bandwidth_allocation_with_timeout(self):
        """Test bandwidth allocation timeout behavior."""
        from app.coordination.sync_bandwidth import (
            BandwidthConfig,
            BandwidthManager,
            TransferPriority,
        )

        # Config with very low concurrency
        config = BandwidthConfig(max_concurrent_total=1)
        BandwidthManager.reset_instance()
        manager = BandwidthManager(config)

        # Get first allocation
        alloc1 = await manager.request_allocation("host1", TransferPriority.NORMAL)
        assert alloc1 is not None

        # Second should timeout quickly
        start = time.time()
        alloc2 = await manager.request_allocation(
            "host2",
            TransferPriority.NORMAL,
            timeout=0.5,
        )
        elapsed = time.time() - start

        assert alloc2 is None
        assert elapsed >= 0.4  # Should have waited near timeout

        await manager.release_allocation(alloc1)
        BandwidthManager.reset_instance()

    def test_coordination_module_exports(self):
        """Test all expected exports are available from coordination module."""
        from app.coordination import (
            DataPipelineOrchestrator,
            PipelineConfig,
            get_pipeline_orchestrator,
        )
        from app.coordination.daemon_adapters import (
            DaemonAdapter,
            get_daemon_adapter,
        )
        from app.coordination.event_router import (
            UnifiedEventRouter,
            get_router,
            publish,
            subscribe,
        )
        from app.coordination.pipeline_actions import (
            StageCompletionResult,
            trigger_data_sync,
            trigger_evaluation,
            trigger_npz_export,
            trigger_promotion,
            trigger_training,
        )
        from app.coordination.sync_bandwidth import (
            BandwidthCoordinatedRsync,
            BandwidthManager,
            get_bandwidth_manager,
        )

        # All should be callable/accessible
        assert callable(get_router)
        assert callable(get_bandwidth_manager)
        assert callable(get_daemon_adapter)
        assert callable(get_pipeline_orchestrator)
        assert callable(trigger_data_sync)
        assert callable(trigger_training)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
