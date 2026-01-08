"""Tests for ClusterResilienceOrchestrator - unified health aggregation.

These tests verify:
1. Configuration (ResilienceConfig, from_env)
2. Enums (ResilienceLevel, RecoveryAction)
3. Data classes (ComponentHealth, ResilienceScore)
4. ClusterResilienceOrchestrator initialization and lifecycle
5. Component health collection methods
6. Resilience score computation
7. Action recommendation logic
8. Callback registration
9. Health check method
10. Singleton pattern
11. Event handlers (Phase 6)

January 7, 2026 - Sprint 17: Added as part of Phase 3 test coverage.
"""

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.cluster_resilience_orchestrator import (
    ClusterResilienceOrchestrator,
    ComponentHealth,
    RecoveryAction,
    ResilienceConfig,
    ResilienceLevel,
    ResilienceScore,
    get_resilience_orchestrator,
    reset_resilience_orchestrator,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestResilienceLevel:
    """Tests for ResilienceLevel enum."""

    def test_level_values(self):
        """ResilienceLevel should have correct string values."""
        assert ResilienceLevel.HEALTHY.value == "healthy"
        assert ResilienceLevel.WARNING.value == "warning"
        assert ResilienceLevel.DEGRADED.value == "degraded"
        assert ResilienceLevel.CRITICAL.value == "critical"

    def test_level_count(self):
        """ResilienceLevel should have exactly 4 members."""
        assert len(ResilienceLevel) == 4


class TestRecoveryAction:
    """Tests for RecoveryAction enum."""

    def test_action_values(self):
        """RecoveryAction should have correct string values."""
        assert RecoveryAction.NONE.value == "none"
        assert RecoveryAction.PAUSE_SELFPLAY.value == "pause_selfplay"
        assert RecoveryAction.PAUSE_TRAINING.value == "pause_training"
        assert RecoveryAction.RESTART_DAEMONS.value == "restart_daemons"
        assert RecoveryAction.RESTART_P2P.value == "restart_p2p"
        assert RecoveryAction.TRIGGER_ELECTION.value == "trigger_election"
        assert RecoveryAction.TRIGGER_FAILOVER.value == "trigger_failover"
        assert RecoveryAction.GRACEFUL_SHUTDOWN.value == "graceful_shutdown"

    def test_action_count(self):
        """RecoveryAction should have exactly 8 members."""
        assert len(RecoveryAction) == 8


# =============================================================================
# Configuration Tests
# =============================================================================


class TestResilienceConfig:
    """Tests for ResilienceConfig dataclass."""

    def test_default_values(self):
        """ResilienceConfig should have sensible defaults."""
        config = ResilienceConfig()
        assert config.check_interval == 30.0
        assert config.early_warning_threshold == 0.70
        assert config.memory_weight == 0.30
        assert config.coordinator_weight == 0.30
        assert config.quorum_weight == 0.25
        assert config.daemon_weight == 0.15
        assert config.critical_threshold == 0.45
        assert config.degraded_threshold == 0.65
        assert config.warning_threshold == 0.85

    def test_custom_values(self):
        """ResilienceConfig should accept custom values."""
        config = ResilienceConfig(
            check_interval=60.0,
            early_warning_threshold=0.80,
            memory_weight=0.40,
        )
        assert config.check_interval == 60.0
        assert config.early_warning_threshold == 0.80
        assert config.memory_weight == 0.40

    def test_weights_sum_to_one(self):
        """Default weights should sum to 1.0."""
        config = ResilienceConfig()
        total = (
            config.memory_weight
            + config.coordinator_weight
            + config.quorum_weight
            + config.daemon_weight
        )
        assert abs(total - 1.0) < 0.01

    def test_from_env_defaults(self):
        """from_env should use defaults when env vars not set."""
        with patch.dict("os.environ", {}, clear=True):
            config = ResilienceConfig.from_env()
            assert config.check_interval == 30.0
            assert config.early_warning_threshold == 0.70

    def test_from_env_with_env_vars(self):
        """from_env should read from environment variables."""
        env_vars = {
            "RINGRIFT_RESILIENCE_CHECK_INTERVAL": "60.0",
            "RINGRIFT_RESILIENCE_WARNING_THRESHOLD": "0.80",
        }
        with patch.dict("os.environ", env_vars, clear=True):
            config = ResilienceConfig.from_env()
            assert config.check_interval == 60.0
            assert config.early_warning_threshold == 0.80


# =============================================================================
# Data Class Tests
# =============================================================================


class TestComponentHealth:
    """Tests for ComponentHealth dataclass."""

    def test_creation(self):
        """ComponentHealth should accept all parameters."""
        health = ComponentHealth(
            name="memory",
            score=0.75,
            is_healthy=True,
            message="Memory at 75%",
            details={"percent": 75},
            last_check_time=time.time(),
        )
        assert health.name == "memory"
        assert health.score == 0.75
        assert health.is_healthy is True
        assert health.message == "Memory at 75%"
        assert health.details["percent"] == 75

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        now = time.time()
        health = ComponentHealth(
            name="quorum",
            score=0.9,
            is_healthy=True,
            message="5 alive peers",
            details={"alive_peers": 5},
            last_check_time=now,
        )
        result = health.to_dict()
        assert result["name"] == "quorum"
        assert result["score"] == 0.9
        assert result["is_healthy"] is True
        assert result["message"] == "5 alive peers"
        assert result["details"]["alive_peers"] == 5
        assert result["last_check_time"] == now


class TestResilienceScore:
    """Tests for ResilienceScore dataclass."""

    def test_creation(self):
        """ResilienceScore should accept all parameters."""
        components = {
            "memory": ComponentHealth("memory", 0.8, True, "OK", {}, time.time()),
        }
        score = ResilienceScore(
            overall=0.85,
            level=ResilienceLevel.HEALTHY,
            components=components,
            recommended_actions=[RecoveryAction.NONE],
            timestamp=time.time(),
            degraded_components=[],
        )
        assert score.overall == 0.85
        assert score.level == ResilienceLevel.HEALTHY
        assert "memory" in score.components
        assert RecoveryAction.NONE in score.recommended_actions

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        now = time.time()
        components = {
            "memory": ComponentHealth("memory", 0.8, True, "OK", {}, now),
            "daemon": ComponentHealth("daemon", 0.3, False, "Failed", {}, now),
        }
        score = ResilienceScore(
            overall=0.55,
            level=ResilienceLevel.DEGRADED,
            components=components,
            recommended_actions=[RecoveryAction.RESTART_DAEMONS],
            timestamp=now,
            degraded_components=["daemon"],
        )
        result = score.to_dict()
        assert result["overall"] == 0.55
        assert result["level"] == "degraded"
        assert "memory" in result["components"]
        assert "daemon" in result["components"]
        assert result["recommended_actions"] == ["restart_daemons"]
        assert result["degraded_components"] == ["daemon"]


# =============================================================================
# Orchestrator Initialization Tests
# =============================================================================


class TestClusterResilienceOrchestratorInit:
    """Tests for ClusterResilienceOrchestrator initialization."""

    def test_default_config(self):
        """Orchestrator should use default config if none provided."""
        orchestrator = ClusterResilienceOrchestrator()
        assert orchestrator._config is not None
        assert orchestrator._config.check_interval == 30.0

    def test_custom_config(self):
        """Orchestrator should use provided config."""
        config = ResilienceConfig(check_interval=60.0)
        orchestrator = ClusterResilienceOrchestrator(config=config)
        assert orchestrator._config.check_interval == 60.0

    def test_initial_state(self):
        """Orchestrator should have zeroed state initially."""
        orchestrator = ClusterResilienceOrchestrator()
        assert orchestrator._running is False
        assert orchestrator._monitor_task is None
        assert orchestrator._last_score is None
        assert orchestrator._autonomous_queue_active is False
        assert orchestrator._fast_failure_tier == "healthy"
        assert orchestrator._escalation_count == 0

    def test_name_property(self):
        """name property should return coordinator name."""
        orchestrator = ClusterResilienceOrchestrator()
        assert orchestrator.name == "cluster_resilience_orchestrator"


# =============================================================================
# Component Health Collection Tests
# =============================================================================


class TestComponentHealthCollection:
    """Tests for component health collection methods."""

    def test_get_memory_health_fallback_no_psutil(self):
        """Memory health should handle missing psutil gracefully."""
        orchestrator = ClusterResilienceOrchestrator()

        with patch.dict("sys.modules", {"psutil": None}):
            # Force ImportError for memory_pressure_controller
            with patch(
                "app.coordination.cluster_resilience_orchestrator."
                "ClusterResilienceOrchestrator._get_memory_health"
            ) as mock:
                mock.return_value = ComponentHealth(
                    name="memory",
                    score=0.5,
                    is_healthy=True,
                    message="Memory status unknown",
                    last_check_time=time.time(),
                )
                health = orchestrator._get_memory_health()
                assert health.name == "memory"

    def test_get_coordinator_health_not_configured(self):
        """Coordinator health should handle missing StandbyCoordinator."""
        orchestrator = ClusterResilienceOrchestrator()

        # Mock the entire standby_coordinator module at the import location
        mock_module = MagicMock()
        mock_module.get_standby_coordinator.side_effect = TypeError("Not configured")

        with patch.dict(
            "sys.modules",
            {"app.coordination.standby_coordinator": mock_module},
        ):
            health = orchestrator._get_coordinator_health()
            assert health.name == "coordinator"
            assert health.score == 0.5
            assert "not configured" in health.message.lower()

    def test_get_quorum_health_fallback(self):
        """Quorum health should fallback gracefully when P2P unavailable."""
        orchestrator = ClusterResilienceOrchestrator()

        # Force import error for health_coordinator
        with patch.dict("sys.modules", {"scripts.p2p.health_coordinator": None}):
            # Also mock HTTP fallback to fail
            with patch("urllib.request.urlopen", side_effect=Exception("No P2P")):
                health = orchestrator._get_quorum_health()
                assert health.name == "quorum"
                assert health.score == 0.5
                assert "unknown" in health.message.lower()

    def test_get_daemon_health_no_daemons(self):
        """Daemon health should handle empty daemon manager."""
        orchestrator = ClusterResilienceOrchestrator()

        mock_dm = MagicMock()
        mock_dm.health_summary.return_value = {
            "total": 0,
            "running": 0,
            "failed": 0,
            "score": 1.0,
        }

        # Mock the daemon_manager module at the import location
        mock_module = MagicMock()
        mock_module.get_daemon_manager.return_value = mock_dm

        with patch.dict(
            "sys.modules",
            {"app.coordination.daemon_manager": mock_module},
        ):
            health = orchestrator._get_daemon_health()
            assert health.name == "daemon"
            assert health.score == 1.0
            assert "no daemons" in health.message.lower()


# =============================================================================
# Resilience Score Computation Tests
# =============================================================================


class TestResilienceScoreComputation:
    """Tests for resilience score computation."""

    def test_healthy_score(self):
        """Healthy components should yield HEALTHY level."""
        orchestrator = ClusterResilienceOrchestrator()

        # Mock all components as healthy
        def mock_healthy(*args, **kwargs):
            return ComponentHealth("test", 1.0, True, "OK", {}, time.time())

        with patch.multiple(
            orchestrator,
            _get_memory_health=mock_healthy,
            _get_coordinator_health=mock_healthy,
            _get_quorum_health=mock_healthy,
            _get_daemon_health=mock_healthy,
        ):
            score = orchestrator.get_resilience_score()
            assert score.overall >= 0.85
            assert score.level == ResilienceLevel.HEALTHY
            assert len(score.degraded_components) == 0

    def test_degraded_score(self):
        """Partially degraded components should yield DEGRADED level."""
        orchestrator = ClusterResilienceOrchestrator()

        # Mock some components as degraded
        with patch.multiple(
            orchestrator,
            _get_memory_health=lambda: ComponentHealth(
                "memory", 0.5, False, "Warning", {}, time.time()
            ),
            _get_coordinator_health=lambda: ComponentHealth(
                "coordinator", 0.5, False, "Degraded", {}, time.time()
            ),
            _get_quorum_health=lambda: ComponentHealth(
                "quorum", 0.7, True, "OK", {}, time.time()
            ),
            _get_daemon_health=lambda: ComponentHealth(
                "daemon", 0.8, True, "OK", {}, time.time()
            ),
        ):
            score = orchestrator.get_resilience_score()
            # (0.5*0.3) + (0.5*0.3) + (0.7*0.25) + (0.8*0.15) = 0.595
            assert score.overall < 0.65
            assert score.level == ResilienceLevel.DEGRADED
            assert "memory" in score.degraded_components
            assert "coordinator" in score.degraded_components

    def test_critical_score(self):
        """Severely degraded components should yield CRITICAL level."""
        orchestrator = ClusterResilienceOrchestrator()

        # Mock all components as critical
        with patch.multiple(
            orchestrator,
            _get_memory_health=lambda: ComponentHealth(
                "memory", 0.2, False, "Emergency", {}, time.time()
            ),
            _get_coordinator_health=lambda: ComponentHealth(
                "coordinator", 0.3, False, "Not responding", {}, time.time()
            ),
            _get_quorum_health=lambda: ComponentHealth(
                "quorum", 0.2, False, "Lost", {}, time.time()
            ),
            _get_daemon_health=lambda: ComponentHealth(
                "daemon", 0.3, False, "Failed", {}, time.time()
            ),
        ):
            score = orchestrator.get_resilience_score()
            # (0.2*0.3) + (0.3*0.3) + (0.2*0.25) + (0.3*0.15) = 0.245
            assert score.overall < 0.45
            assert score.level == ResilienceLevel.CRITICAL


# =============================================================================
# Action Recommendation Tests
# =============================================================================


class TestActionRecommendation:
    """Tests for recovery action recommendations."""

    def test_healthy_no_action(self):
        """Healthy state should recommend no action."""
        orchestrator = ClusterResilienceOrchestrator()
        actions = orchestrator._recommend_actions(
            ResilienceLevel.HEALTHY,
            [],
            {},
        )
        assert actions == [RecoveryAction.NONE]

    def test_memory_critical_recommends_shutdown(self):
        """Critical memory should recommend graceful shutdown."""
        orchestrator = ClusterResilienceOrchestrator()
        components = {
            "memory": ComponentHealth("memory", 0.2, False, "Critical", {}, 0),
        }
        actions = orchestrator._recommend_actions(
            ResilienceLevel.CRITICAL,
            ["memory"],
            components,
        )
        assert RecoveryAction.GRACEFUL_SHUTDOWN in actions

    def test_memory_warning_recommends_pause(self):
        """Warning memory should recommend pause training/selfplay."""
        orchestrator = ClusterResilienceOrchestrator()
        components = {
            "memory": ComponentHealth("memory", 0.4, False, "Warning", {}, 0),
        }
        actions = orchestrator._recommend_actions(
            ResilienceLevel.DEGRADED,
            ["memory"],
            components,
        )
        assert RecoveryAction.PAUSE_TRAINING in actions
        assert RecoveryAction.PAUSE_SELFPLAY in actions

    def test_coordinator_degraded_recommends_failover(self):
        """Degraded coordinator should recommend failover."""
        orchestrator = ClusterResilienceOrchestrator()
        components = {
            "coordinator": ComponentHealth("coordinator", 0.3, False, "Degraded", {}, 0),
        }
        actions = orchestrator._recommend_actions(
            ResilienceLevel.DEGRADED,
            ["coordinator"],
            components,
        )
        assert RecoveryAction.TRIGGER_FAILOVER in actions

    def test_quorum_lost_recommends_election(self):
        """Lost quorum should recommend trigger election."""
        orchestrator = ClusterResilienceOrchestrator()
        components = {
            "quorum": ComponentHealth("quorum", 0.2, False, "Lost", {}, 0),
        }
        actions = orchestrator._recommend_actions(
            ResilienceLevel.CRITICAL,
            ["quorum"],
            components,
        )
        assert RecoveryAction.TRIGGER_ELECTION in actions

    def test_daemon_degraded_recommends_restart(self):
        """Degraded daemons should recommend restart."""
        orchestrator = ClusterResilienceOrchestrator()
        components = {
            "daemon": ComponentHealth("daemon", 0.5, False, "Failed", {}, 0),
        }
        actions = orchestrator._recommend_actions(
            ResilienceLevel.DEGRADED,
            ["daemon"],
            components,
        )
        assert RecoveryAction.RESTART_DAEMONS in actions


# =============================================================================
# Callback Registration Tests
# =============================================================================


class TestCallbackRegistration:
    """Tests for callback registration."""

    def test_register_degraded_callback(self):
        """register_degraded_callback should add to callback list."""
        orchestrator = ClusterResilienceOrchestrator()
        callback = MagicMock()

        orchestrator.register_degraded_callback(callback)

        assert callback in orchestrator._on_degraded

    def test_register_critical_callback(self):
        """register_critical_callback should add to callback list."""
        orchestrator = ClusterResilienceOrchestrator()
        callback = MagicMock()

        orchestrator.register_critical_callback(callback)

        assert callback in orchestrator._on_critical

    def test_multiple_callbacks(self):
        """Multiple callbacks should all be registered."""
        orchestrator = ClusterResilienceOrchestrator()
        callbacks = [MagicMock() for _ in range(3)]

        for cb in callbacks:
            orchestrator.register_degraded_callback(cb)

        assert len(orchestrator._on_degraded) == 3


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health_check method."""

    def test_health_check_critical(self):
        """health_check should return unhealthy for CRITICAL."""
        orchestrator = ClusterResilienceOrchestrator()

        # Mock a critical score
        mock_score = ResilienceScore(
            overall=0.3,
            level=ResilienceLevel.CRITICAL,
            components={},
            recommended_actions=[RecoveryAction.GRACEFUL_SHUTDOWN],
            timestamp=time.time(),
            degraded_components=["memory", "quorum"],
        )
        orchestrator._last_score = mock_score

        result = orchestrator.health_check()
        assert result.healthy is False
        assert "critical" in result.message.lower()

    def test_health_check_degraded(self):
        """health_check should return degraded status for DEGRADED."""
        orchestrator = ClusterResilienceOrchestrator()

        mock_score = ResilienceScore(
            overall=0.6,
            level=ResilienceLevel.DEGRADED,
            components={},
            recommended_actions=[RecoveryAction.RESTART_DAEMONS],
            timestamp=time.time(),
            degraded_components=["daemon"],
        )
        orchestrator._last_score = mock_score

        result = orchestrator.health_check()
        # Degraded is still healthy=True but with degraded message
        assert "degraded" in result.message.lower()

    def test_health_check_healthy(self):
        """health_check should return healthy for HEALTHY/WARNING."""
        orchestrator = ClusterResilienceOrchestrator()

        mock_score = ResilienceScore(
            overall=0.9,
            level=ResilienceLevel.HEALTHY,
            components={},
            recommended_actions=[RecoveryAction.NONE],
            timestamp=time.time(),
            degraded_components=[],
        )
        orchestrator._last_score = mock_score

        result = orchestrator.health_check()
        assert result.healthy is True


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton pattern."""

    def test_returns_singleton(self):
        """get_resilience_orchestrator should return same instance."""
        reset_resilience_orchestrator()

        orch1 = get_resilience_orchestrator()
        orch2 = get_resilience_orchestrator()

        assert orch1 is orch2

    def test_reset_singleton(self):
        """reset_resilience_orchestrator should create new instance."""
        reset_resilience_orchestrator()

        orch1 = get_resilience_orchestrator()
        reset_resilience_orchestrator()
        orch2 = get_resilience_orchestrator()

        assert orch1 is not orch2


# =============================================================================
# Event Handler Tests (Phase 6)
# =============================================================================


class TestEventHandlers:
    """Tests for Phase 6 event handlers."""

    def test_on_autonomous_queue_activated(self):
        """_on_autonomous_queue_activated should update state."""
        orchestrator = ClusterResilienceOrchestrator()

        orchestrator._on_autonomous_queue_activated({
            "node_id": "test-node",
            "reason": "leader_unavailable",
        })

        assert orchestrator._autonomous_queue_active is True
        assert orchestrator._escalation_count == 1

    def test_on_autonomous_queue_deactivated(self):
        """_on_autonomous_queue_deactivated should update state."""
        orchestrator = ClusterResilienceOrchestrator()
        orchestrator._autonomous_queue_active = True

        orchestrator._on_autonomous_queue_deactivated({"node_id": "test-node"})

        assert orchestrator._autonomous_queue_active is False

    def test_on_utilization_recovery_started(self):
        """_on_utilization_recovery_started should update state."""
        orchestrator = ClusterResilienceOrchestrator()

        orchestrator._on_utilization_recovery_started({
            "node_id": "test-node",
            "idle_gpu_percent": 80,
            "work_items_injected": 10,
        })

        assert orchestrator._utilization_recovery_active is True

    def test_on_utilization_recovery_completed(self):
        """_on_utilization_recovery_completed should update state."""
        orchestrator = ClusterResilienceOrchestrator()
        orchestrator._utilization_recovery_active = True

        orchestrator._on_utilization_recovery_completed({
            "node_id": "test-node",
            "work_items_processed": 10,
        })

        assert orchestrator._utilization_recovery_active is False

    def test_on_utilization_recovery_failed(self):
        """_on_utilization_recovery_failed should update state and escalate."""
        orchestrator = ClusterResilienceOrchestrator()
        orchestrator._utilization_recovery_active = True

        orchestrator._on_utilization_recovery_failed({
            "node_id": "test-node",
            "error": "timeout",
        })

        assert orchestrator._utilization_recovery_active is False
        assert orchestrator._escalation_count == 1

    def test_on_fast_failure_alert(self):
        """_on_fast_failure_alert should update tier and escalate."""
        orchestrator = ClusterResilienceOrchestrator()

        with patch(
            "app.coordination.cluster_resilience_orchestrator."
            "ClusterResilienceOrchestrator._emit_escalation_event"
        ):
            orchestrator._on_fast_failure_alert({
                "tier": "alert",
                "signals": ["no_leader"],
                "no_leader_seconds": 300,
            })

        assert orchestrator._fast_failure_tier == "alert"
        assert orchestrator._escalation_count == 1

    def test_on_fast_failure_recovered(self):
        """_on_fast_failure_recovered should reset tier."""
        orchestrator = ClusterResilienceOrchestrator()
        orchestrator._fast_failure_tier = "recovery"

        orchestrator._on_fast_failure_recovered({
            "signals": ["leader_elected"],
        })

        assert orchestrator._fast_failure_tier == "healthy"

    def test_get_resilience_state(self):
        """get_resilience_state should return all state fields."""
        orchestrator = ClusterResilienceOrchestrator()
        orchestrator._autonomous_queue_active = True
        orchestrator._fast_failure_tier = "alert"
        orchestrator._utilization_recovery_active = True
        orchestrator._escalation_count = 3

        state = orchestrator.get_resilience_state()

        assert state["autonomous_queue_active"] is True
        assert state["fast_failure_tier"] == "alert"
        assert state["utilization_recovery_active"] is True
        assert state["escalation_count"] == 3


# =============================================================================
# Cleanup
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singleton_fixture():
    """Reset singleton before each test."""
    reset_resilience_orchestrator()
    yield
    reset_resilience_orchestrator()
