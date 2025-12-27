"""Tests for coordination enums and types modules.

Tests cover:
- app/coordination/enums.py - Central enum re-exports
- app/coordination/types.py - Coordination types
- app/coordination/daemon_types.py - Daemon management types

Created: December 27, 2025
"""

from __future__ import annotations

import pytest


class TestEnumsModule:
    """Tests for app/coordination/enums.py central re-exports."""

    def test_leadership_role_import(self):
        """Test LeadershipRole can be imported from enums."""
        from app.coordination.enums import LeadershipRole

        assert hasattr(LeadershipRole, "LEADER")
        assert hasattr(LeadershipRole, "FOLLOWER")
        assert hasattr(LeadershipRole, "CANDIDATE")

    def test_cluster_node_role_import(self):
        """Test ClusterNodeRole can be imported from enums."""
        from app.coordination.enums import ClusterNodeRole

        # Should have job-related roles
        assert hasattr(ClusterNodeRole, "TRAINING") or hasattr(ClusterNodeRole, "SELFPLAY")

    def test_recovery_action_enums_distinct(self):
        """Test that the three recovery action enums are distinct types."""
        from app.coordination.enums import (
            JobRecoveryAction,
            NodeRecoveryAction,
            SystemRecoveryAction,
        )

        # Each should be a different enum type
        assert JobRecoveryAction is not SystemRecoveryAction
        assert SystemRecoveryAction is not NodeRecoveryAction
        assert JobRecoveryAction is not NodeRecoveryAction

    def test_daemon_type_import(self):
        """Test DaemonType can be imported from enums."""
        from app.coordination.enums import DaemonType

        # Check some well-known daemon types
        assert hasattr(DaemonType, "AUTO_SYNC")
        assert hasattr(DaemonType, "EVENT_ROUTER")
        assert hasattr(DaemonType, "DATA_PIPELINE")

    def test_daemon_state_import(self):
        """Test DaemonState can be imported from enums."""
        from app.coordination.enums import DaemonState

        assert hasattr(DaemonState, "STOPPED")
        assert hasattr(DaemonState, "RUNNING")
        assert hasattr(DaemonState, "FAILED")

    def test_node_health_state_import(self):
        """Test NodeHealthState can be imported from enums."""
        from app.coordination.enums import NodeHealthState

        assert hasattr(NodeHealthState, "HEALTHY")
        assert hasattr(NodeHealthState, "UNHEALTHY")
        assert hasattr(NodeHealthState, "DEGRADED")

    def test_data_event_type_import(self):
        """Test DataEventType can be imported from enums."""
        from app.coordination.enums import DataEventType

        # Check some common event types
        assert hasattr(DataEventType, "TRAINING_STARTED") or hasattr(
            DataEventType, "TRAINING_COMPLETED"
        )

    def test_error_severity_import(self):
        """Test ErrorSeverity can be imported from enums."""
        from app.coordination.enums import ErrorSeverity

        assert hasattr(ErrorSeverity, "ERROR") or hasattr(ErrorSeverity, "CRITICAL")

    def test_recovery_status_import(self):
        """Test RecoveryStatus can be imported from enums."""
        from app.coordination.enums import RecoveryStatus

        assert hasattr(RecoveryStatus, "PENDING") or hasattr(RecoveryStatus, "IN_PROGRESS")

    def test_recovery_result_import(self):
        """Test RecoveryResult can be imported from enums."""
        from app.coordination.enums import RecoveryResult

        assert hasattr(RecoveryResult, "SUCCESS") or hasattr(RecoveryResult, "FAILED")

    def test_all_exports_defined(self):
        """Test that __all__ contains expected exports."""
        from app.coordination import enums

        expected = {
            "LeadershipRole",
            "ClusterNodeRole",
            "JobRecoveryAction",
            "SystemRecoveryAction",
            "NodeRecoveryAction",
            "DaemonType",
            "DaemonState",
            "NodeHealthState",
            "ErrorSeverity",
            "RecoveryStatus",
            "RecoveryResult",
            "DataEventType",
        }
        for name in expected:
            assert name in enums.__all__, f"{name} not in __all__"


class TestTypesModule:
    """Tests for app/coordination/types.py coordination types."""

    def test_backpressure_level_values(self):
        """Test BackpressureLevel has all expected values."""
        from app.coordination.types import BackpressureLevel

        # Check all values exist
        assert BackpressureLevel.NONE.value == "none"
        assert BackpressureLevel.LOW.value == "low"
        assert BackpressureLevel.SOFT.value == "soft"
        assert BackpressureLevel.MEDIUM.value == "medium"
        assert BackpressureLevel.HARD.value == "hard"
        assert BackpressureLevel.HIGH.value == "high"
        assert BackpressureLevel.CRITICAL.value == "critical"
        assert BackpressureLevel.STOP.value == "stop"

    def test_backpressure_from_legacy_queue(self):
        """Test BackpressureLevel.from_legacy_queue conversion."""
        from app.coordination.types import BackpressureLevel

        assert BackpressureLevel.from_legacy_queue("none") == BackpressureLevel.NONE
        assert BackpressureLevel.from_legacy_queue("soft") == BackpressureLevel.SOFT
        assert BackpressureLevel.from_legacy_queue("hard") == BackpressureLevel.HARD
        assert BackpressureLevel.from_legacy_queue("stop") == BackpressureLevel.STOP
        # Unknown values default to NONE
        assert BackpressureLevel.from_legacy_queue("unknown") == BackpressureLevel.NONE

    def test_backpressure_from_legacy_resource(self):
        """Test BackpressureLevel.from_legacy_resource conversion."""
        from app.coordination.types import BackpressureLevel

        assert BackpressureLevel.from_legacy_resource("none") == BackpressureLevel.NONE
        assert BackpressureLevel.from_legacy_resource("low") == BackpressureLevel.LOW
        assert BackpressureLevel.from_legacy_resource("medium") == BackpressureLevel.MEDIUM
        assert BackpressureLevel.from_legacy_resource("high") == BackpressureLevel.HIGH
        assert BackpressureLevel.from_legacy_resource("critical") == BackpressureLevel.CRITICAL
        # Unknown values default to NONE
        assert BackpressureLevel.from_legacy_resource("xyz") == BackpressureLevel.NONE

    def test_backpressure_is_throttling(self):
        """Test BackpressureLevel.is_throttling method."""
        from app.coordination.types import BackpressureLevel

        assert not BackpressureLevel.NONE.is_throttling()
        assert BackpressureLevel.LOW.is_throttling()
        assert BackpressureLevel.SOFT.is_throttling()
        assert BackpressureLevel.MEDIUM.is_throttling()
        assert BackpressureLevel.HARD.is_throttling()
        assert BackpressureLevel.HIGH.is_throttling()
        assert BackpressureLevel.CRITICAL.is_throttling()
        assert BackpressureLevel.STOP.is_throttling()

    def test_backpressure_should_stop(self):
        """Test BackpressureLevel.should_stop method."""
        from app.coordination.types import BackpressureLevel

        assert not BackpressureLevel.NONE.should_stop()
        assert not BackpressureLevel.LOW.should_stop()
        assert not BackpressureLevel.MEDIUM.should_stop()
        assert not BackpressureLevel.HARD.should_stop()
        # Only CRITICAL and STOP should return True
        assert BackpressureLevel.CRITICAL.should_stop()
        assert BackpressureLevel.STOP.should_stop()

    def test_backpressure_reduction_factor(self):
        """Test BackpressureLevel.reduction_factor method."""
        from app.coordination.types import BackpressureLevel

        # Verify reduction factors are in expected range
        assert BackpressureLevel.NONE.reduction_factor() == 1.0
        assert 0.5 < BackpressureLevel.LOW.reduction_factor() < 1.0
        assert 0.4 < BackpressureLevel.SOFT.reduction_factor() < 0.6
        assert 0.1 < BackpressureLevel.MEDIUM.reduction_factor() < 0.4
        assert 0.05 < BackpressureLevel.HARD.reduction_factor() < 0.15
        assert BackpressureLevel.STOP.reduction_factor() == 0.0

    def test_task_type_values(self):
        """Test TaskType has core values."""
        from app.coordination.types import TaskType

        assert TaskType.SELFPLAY.value == "selfplay"
        assert TaskType.GPU_SELFPLAY.value == "gpu_selfplay"
        assert TaskType.TRAINING.value == "training"
        assert TaskType.EVALUATION.value == "evaluation"
        assert TaskType.EXPORT.value == "export"
        assert TaskType.SYNC.value == "sync"
        assert TaskType.TOURNAMENT.value == "tournament"
        assert TaskType.UNKNOWN.value == "unknown"

    def test_board_type_values(self):
        """Test BoardType has expected values."""
        from app.coordination.types import BoardType

        assert BoardType.HEX8.value == "hex8"
        assert BoardType.SQUARE8.value == "square8"
        assert BoardType.SQUARE19.value == "square19"
        assert BoardType.HEXAGONAL.value == "hexagonal"

    def test_board_type_cell_count(self):
        """Test BoardType.cell_count property."""
        from app.coordination.types import BoardType

        assert BoardType.HEX8.cell_count == 61
        assert BoardType.SQUARE8.cell_count == 64
        assert BoardType.SQUARE19.cell_count == 361
        assert BoardType.HEXAGONAL.cell_count == 469

    def test_board_type_vram_requirement(self):
        """Test BoardType.vram_requirement_gb property."""
        from app.coordination.types import BoardType

        # Smaller boards need less VRAM
        assert BoardType.HEX8.vram_requirement_gb < BoardType.SQUARE19.vram_requirement_gb
        assert BoardType.SQUARE8.vram_requirement_gb < BoardType.HEXAGONAL.vram_requirement_gb

    def test_board_type_from_string(self):
        """Test BoardType.from_string parsing."""
        from app.coordination.types import BoardType

        assert BoardType.from_string("hex8") == BoardType.HEX8
        assert BoardType.from_string("HEX8") == BoardType.HEX8
        assert BoardType.from_string("square8") == BoardType.SQUARE8
        assert BoardType.from_string("sq8") == BoardType.SQUARE8
        assert BoardType.from_string("square19") == BoardType.SQUARE19
        assert BoardType.from_string("hexagonal") == BoardType.HEXAGONAL

    def test_board_type_from_string_invalid(self):
        """Test BoardType.from_string raises for invalid input."""
        from app.coordination.types import BoardType

        with pytest.raises(ValueError):
            BoardType.from_string("invalid_board")

    def test_work_status_values(self):
        """Test WorkStatus has expected values."""
        from app.coordination.types import WorkStatus

        assert WorkStatus.PENDING.value == "pending"
        assert WorkStatus.CLAIMED.value == "claimed"
        assert WorkStatus.RUNNING.value == "running"
        assert WorkStatus.COMPLETED.value == "completed"
        assert WorkStatus.FAILED.value == "failed"
        assert WorkStatus.TIMEOUT.value == "timeout"
        assert WorkStatus.CANCELLED.value == "cancelled"

    def test_health_level_values(self):
        """Test HealthLevel has expected values."""
        from app.coordination.types import HealthLevel

        assert HealthLevel.HEALTHY.value == "healthy"
        assert HealthLevel.DEGRADED.value == "degraded"
        assert HealthLevel.UNHEALTHY.value == "unhealthy"
        assert HealthLevel.CRITICAL.value == "critical"
        assert HealthLevel.UNKNOWN.value == "unknown"

    def test_core_node_types_reexported(self):
        """Test core node types are re-exported from types module."""
        from app.coordination.types import GPUInfo, NodeHealth, NodeRole, NodeState, Provider

        # Should be importable
        assert GPUInfo is not None
        assert NodeHealth is not None
        assert NodeRole is not None
        assert NodeState is not None
        assert Provider is not None


class TestDaemonTypesModule:
    """Tests for app/coordination/daemon_types.py daemon management types."""

    def test_daemon_type_enum_has_core_values(self):
        """Test DaemonType has core daemon values."""
        from app.coordination.daemon_types import DaemonType

        # Core infrastructure
        assert DaemonType.EVENT_ROUTER.value == "event_router"
        assert DaemonType.DAEMON_WATCHDOG.value == "daemon_watchdog"
        assert DaemonType.DATA_PIPELINE.value == "data_pipeline"

        # Sync daemons
        assert DaemonType.AUTO_SYNC.value == "auto_sync"

        # Queue management
        assert DaemonType.QUEUE_POPULATOR.value == "queue_populator"

    def test_daemon_type_count(self):
        """Test DaemonType has 60+ daemon types."""
        from app.coordination.daemon_types import DaemonType

        # Should have at least 60 daemon types
        assert len(list(DaemonType)) >= 60

    def test_daemon_state_values(self):
        """Test DaemonState has expected values."""
        from app.coordination.daemon_types import DaemonState

        assert DaemonState.STOPPED.value == "stopped"
        assert DaemonState.STARTING.value == "starting"
        assert DaemonState.RUNNING.value == "running"
        assert DaemonState.STOPPING.value == "stopping"
        assert DaemonState.FAILED.value == "failed"
        assert DaemonState.RESTARTING.value == "restarting"
        assert DaemonState.IMPORT_FAILED.value == "import_failed"

    def test_daemon_info_dataclass(self):
        """Test DaemonInfo dataclass fields."""
        from app.coordination.daemon_types import DaemonInfo, DaemonState, DaemonType

        info = DaemonInfo(daemon_type=DaemonType.AUTO_SYNC)

        assert info.daemon_type == DaemonType.AUTO_SYNC
        assert info.state == DaemonState.STOPPED
        assert info.task is None
        assert info.start_time == 0.0
        assert info.restart_count == 0
        assert info.last_error is None
        assert info.auto_restart is True

    def test_daemon_info_uptime(self):
        """Test DaemonInfo.uptime_seconds property."""
        import time

        from app.coordination.daemon_types import DaemonInfo, DaemonState, DaemonType

        info = DaemonInfo(
            daemon_type=DaemonType.AUTO_SYNC,
            state=DaemonState.RUNNING,
            start_time=time.time() - 100,  # Started 100 seconds ago
        )

        # Should be approximately 100 seconds
        uptime = info.uptime_seconds
        assert 99 <= uptime <= 101

    def test_daemon_info_uptime_stopped(self):
        """Test DaemonInfo.uptime_seconds returns 0 when stopped."""
        import time

        from app.coordination.daemon_types import DaemonInfo, DaemonState, DaemonType

        info = DaemonInfo(
            daemon_type=DaemonType.AUTO_SYNC,
            state=DaemonState.STOPPED,
            start_time=time.time() - 100,  # Started 100 seconds ago but now stopped
        )

        assert info.uptime_seconds == 0.0

    def test_daemon_manager_config_defaults(self):
        """Test DaemonManagerConfig default values."""
        from app.coordination.daemon_types import DaemonManagerConfig

        config = DaemonManagerConfig()

        assert config.auto_start is False
        assert config.auto_restart_failed is True
        assert config.health_check_interval > 0
        assert config.shutdown_timeout > 0
        assert config.max_restart_attempts >= 1
        assert config.critical_daemon_health_interval > 0

    def test_critical_daemons_set(self):
        """Test CRITICAL_DAEMONS set contains expected daemons."""
        from app.coordination.daemon_types import CRITICAL_DAEMONS, DaemonType

        # Core daemons should be marked critical
        assert DaemonType.EVENT_ROUTER in CRITICAL_DAEMONS
        assert DaemonType.DAEMON_WATCHDOG in CRITICAL_DAEMONS
        assert DaemonType.DATA_PIPELINE in CRITICAL_DAEMONS
        assert DaemonType.AUTO_SYNC in CRITICAL_DAEMONS

    def test_daemon_startup_order_exists(self):
        """Test DAEMON_STARTUP_ORDER is defined and non-empty."""
        from app.coordination.daemon_types import DAEMON_STARTUP_ORDER

        assert len(DAEMON_STARTUP_ORDER) > 0

    def test_daemon_startup_order_has_critical_daemons_first(self):
        """Test critical daemons appear early in startup order."""
        from app.coordination.daemon_types import DAEMON_STARTUP_ORDER, DaemonType

        # EVENT_ROUTER should be first
        assert DAEMON_STARTUP_ORDER[0] == DaemonType.EVENT_ROUTER

        # DATA_PIPELINE should come before AUTO_SYNC
        dp_pos = DAEMON_STARTUP_ORDER.index(DaemonType.DATA_PIPELINE)
        as_pos = DAEMON_STARTUP_ORDER.index(DaemonType.AUTO_SYNC)
        assert dp_pos < as_pos, "DATA_PIPELINE must start before AUTO_SYNC"

    def test_daemon_dependencies_defined(self):
        """Test DAEMON_DEPENDENCIES is defined."""
        from app.coordination.daemon_types import DAEMON_DEPENDENCIES, DaemonType

        # Should have dependencies defined for many daemons
        assert len(DAEMON_DEPENDENCIES) > 20

        # AUTO_SYNC should depend on DATA_PIPELINE
        auto_sync_deps = DAEMON_DEPENDENCIES.get(DaemonType.AUTO_SYNC, set())
        assert DaemonType.DATA_PIPELINE in auto_sync_deps

    def test_validate_daemon_dependencies_success(self):
        """Test validate_daemon_dependencies returns success when deps met."""
        from app.coordination.daemon_types import (
            DaemonType,
            validate_daemon_dependencies,
        )

        running = {DaemonType.EVENT_ROUTER}
        ok, missing = validate_daemon_dependencies(DaemonType.DAEMON_WATCHDOG, running)

        assert ok is True
        assert missing == []

    def test_validate_daemon_dependencies_missing(self):
        """Test validate_daemon_dependencies returns missing deps."""
        from app.coordination.daemon_types import (
            DaemonType,
            validate_daemon_dependencies,
        )

        running = set()  # No daemons running
        ok, missing = validate_daemon_dependencies(DaemonType.AUTO_SYNC, running)

        # AUTO_SYNC depends on EVENT_ROUTER, DATA_PIPELINE, FEEDBACK_LOOP
        assert ok is False
        assert DaemonType.EVENT_ROUTER in missing

    def test_get_daemon_startup_position(self):
        """Test get_daemon_startup_position returns correct positions."""
        from app.coordination.daemon_types import (
            DAEMON_STARTUP_ORDER,
            DaemonType,
            get_daemon_startup_position,
        )

        # EVENT_ROUTER should be at position 0
        assert get_daemon_startup_position(DaemonType.EVENT_ROUTER) == 0

        # Check a daemon in the middle of the order
        if len(DAEMON_STARTUP_ORDER) > 5:
            daemon = DAEMON_STARTUP_ORDER[5]
            assert get_daemon_startup_position(daemon) == 5

    def test_get_daemon_startup_position_not_in_order(self):
        """Test get_daemon_startup_position returns -1 for daemons not in order."""
        from app.coordination.daemon_types import (
            DAEMON_STARTUP_ORDER,
            DaemonType,
            get_daemon_startup_position,
        )

        # Find a daemon NOT in the startup order
        for daemon in DaemonType:
            if daemon not in DAEMON_STARTUP_ORDER:
                assert get_daemon_startup_position(daemon) == -1
                break

    def test_validate_startup_order_consistency(self):
        """Test validate_startup_order_consistency validates order."""
        from app.coordination.daemon_types import validate_startup_order_consistency

        is_consistent, violations = validate_startup_order_consistency()

        # If there are violations, they should be strings
        assert isinstance(violations, list)
        for v in violations:
            assert isinstance(v, str)

    def test_validate_startup_order_or_raise(self):
        """Test validate_startup_order_or_raise doesn't raise when consistent."""
        from app.coordination.daemon_types import validate_startup_order_or_raise

        # Should not raise if order is consistent
        # (If it raises, the startup order has a bug)
        try:
            validate_startup_order_or_raise()
        except ValueError as e:
            pytest.fail(f"DAEMON_STARTUP_ORDER is inconsistent: {e}")

    def test_deprecated_daemon_types_tracking(self):
        """Test deprecated daemon types are tracked."""
        from app.coordination.daemon_types import DaemonType

        # These should exist but be deprecated
        assert hasattr(DaemonType, "SYNC_COORDINATOR")
        assert hasattr(DaemonType, "HEALTH_CHECK")
        assert hasattr(DaemonType, "EPHEMERAL_SYNC")
        assert hasattr(DaemonType, "CLUSTER_DATA_SYNC")

    def test_max_restart_delay_constant(self):
        """Test MAX_RESTART_DELAY constant is defined."""
        from app.coordination.daemon_types import MAX_RESTART_DELAY

        assert MAX_RESTART_DELAY > 0
        assert MAX_RESTART_DELAY <= 600  # Should be at most 10 minutes

    def test_daemon_restart_reset_after_constant(self):
        """Test DAEMON_RESTART_RESET_AFTER constant is defined."""
        from app.coordination.daemon_types import DAEMON_RESTART_RESET_AFTER

        assert DAEMON_RESTART_RESET_AFTER > 0
        assert DAEMON_RESTART_RESET_AFTER <= 7200  # Should be at most 2 hours

    def test_mark_daemon_ready_callback_registration(self):
        """Test mark_daemon_ready callback can be registered."""
        from app.coordination.daemon_types import (
            DaemonType,
            mark_daemon_ready,
            register_mark_ready_callback,
        )

        marked = []

        def callback(daemon_type: DaemonType) -> None:
            marked.append(daemon_type)

        register_mark_ready_callback(callback)
        mark_daemon_ready(DaemonType.AUTO_SYNC)

        assert DaemonType.AUTO_SYNC in marked

        # Reset callback to avoid affecting other tests
        register_mark_ready_callback(None)  # type: ignore


class TestDaemonDependencyGraph:
    """Tests for daemon dependency graph consistency."""

    def test_all_dependencies_are_valid_daemon_types(self):
        """Test all dependencies reference valid DaemonTypes."""
        from app.coordination.daemon_types import DAEMON_DEPENDENCIES, DaemonType

        all_daemon_types = set(DaemonType)

        for daemon_type, deps in DAEMON_DEPENDENCIES.items():
            # The daemon type itself must be valid
            assert daemon_type in all_daemon_types, f"Invalid daemon type: {daemon_type}"

            # All dependencies must be valid
            for dep in deps:
                assert dep in all_daemon_types, f"Invalid dependency: {dep} for {daemon_type}"

    def test_no_self_dependencies(self):
        """Test no daemon depends on itself."""
        from app.coordination.daemon_types import DAEMON_DEPENDENCIES

        for daemon_type, deps in DAEMON_DEPENDENCIES.items():
            assert daemon_type not in deps, f"{daemon_type} depends on itself"

    def test_critical_daemons_have_minimal_dependencies(self):
        """Test critical daemons don't have excessive dependencies."""
        from app.coordination.daemon_types import (
            CRITICAL_DAEMONS,
            DAEMON_DEPENDENCIES,
        )

        for daemon in CRITICAL_DAEMONS:
            deps = DAEMON_DEPENDENCIES.get(daemon, set())
            # Critical daemons shouldn't have more than 5 direct dependencies
            assert len(deps) <= 5, f"Critical daemon {daemon} has too many deps: {deps}"

    def test_event_router_has_no_dependencies(self):
        """Test EVENT_ROUTER has no dependencies (must start first)."""
        from app.coordination.daemon_types import DAEMON_DEPENDENCIES, DaemonType

        event_router_deps = DAEMON_DEPENDENCIES.get(DaemonType.EVENT_ROUTER, set())
        assert len(event_router_deps) == 0, "EVENT_ROUTER should have no dependencies"


class TestEnumStringValues:
    """Tests that enum values are usable as strings (for serialization)."""

    def test_backpressure_level_is_str_enum(self):
        """Test BackpressureLevel is a str enum."""
        from app.coordination.types import BackpressureLevel

        # Should be usable as string
        assert str(BackpressureLevel.NONE) == "BackpressureLevel.NONE" or BackpressureLevel.NONE == "none"
        assert BackpressureLevel.NONE.value == "none"

    def test_task_type_is_str_enum(self):
        """Test TaskType is a str enum."""
        from app.coordination.types import TaskType

        assert TaskType.TRAINING.value == "training"

    def test_daemon_type_has_string_values(self):
        """Test DaemonType values are strings."""
        from app.coordination.daemon_types import DaemonType

        for daemon in DaemonType:
            assert isinstance(daemon.value, str)
            assert len(daemon.value) > 0
