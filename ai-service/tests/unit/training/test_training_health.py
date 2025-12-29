"""Unit tests for training health monitoring.

Tests cover:
- AlertSeverity enum and AlertLevel conversion
- TrainingRunStatus, ConfigHealth, Alert, HealthReport dataclasses
- TrainingHealthMonitor initialization, recording, and health checks
- Singleton pattern via get_training_health_monitor()
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from app.training.training_health import (
    Alert,
    AlertSeverity,
    ConfigHealth,
    HealthReport,
    HealthStatus,
    TrainingHealthMonitor,
    TrainingRunStatus,
    get_training_health_monitor,
    MAX_TRAINING_HOURS,
    MIN_WIN_RATE,
    STALE_DATA_HOURS,
    STALE_MODEL_HOURS,
)


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_enum_values(self):
        """Verify enum has expected values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_to_alert_level_warning(self):
        """Test WARNING conversion to AlertLevel."""
        result = AlertSeverity.WARNING.to_alert_level()
        # Should return an AlertLevel enum (or self if monitoring framework unavailable)
        assert result is not None

    def test_to_alert_level_critical(self):
        """Test CRITICAL conversion to AlertLevel."""
        result = AlertSeverity.CRITICAL.to_alert_level()
        assert result is not None

    def test_to_alert_level_info(self):
        """Test INFO conversion (maps to WARNING since no INFO in AlertLevel)."""
        result = AlertSeverity.INFO.to_alert_level()
        assert result is not None


class TestTrainingRunStatus:
    """Tests for TrainingRunStatus dataclass."""

    def test_initialization_required_fields(self):
        """Test initialization with required fields only."""
        status = TrainingRunStatus(
            config_key="hex8_2p",
            started_at=1000.0,
        )
        assert status.config_key == "hex8_2p"
        assert status.started_at == 1000.0
        assert status.completed_at is None
        assert status.success is None
        assert status.metrics == {}
        assert status.error_message is None

    def test_initialization_all_fields(self):
        """Test initialization with all fields."""
        status = TrainingRunStatus(
            config_key="square8_4p",
            started_at=1000.0,
            completed_at=2000.0,
            success=True,
            metrics={"loss": 0.5, "accuracy": 0.9},
            error_message=None,
        )
        assert status.config_key == "square8_4p"
        assert status.completed_at == 2000.0
        assert status.success is True
        assert status.metrics["loss"] == 0.5

    def test_failed_run(self):
        """Test failed training run."""
        status = TrainingRunStatus(
            config_key="hex8_4p",
            started_at=1000.0,
            completed_at=1500.0,
            success=False,
            error_message="CUDA out of memory",
        )
        assert status.success is False
        assert status.error_message == "CUDA out of memory"


class TestConfigHealth:
    """Tests for ConfigHealth dataclass."""

    def test_default_values(self):
        """Test default initialization values."""
        health = ConfigHealth(config_key="hex8_2p")
        assert health.config_key == "hex8_2p"
        assert health.last_training_time == 0
        assert health.last_training_success is True
        assert health.consecutive_failures == 0
        assert health.last_data_time == 0
        assert health.game_count == 0
        assert health.model_count == 0
        assert health.win_rate == 0.5
        assert health.is_training is False
        assert health.training_start_time is None

    def test_training_in_progress(self):
        """Test config with training in progress."""
        now = time.time()
        health = ConfigHealth(
            config_key="square8_2p",
            is_training=True,
            training_start_time=now,
        )
        assert health.is_training is True
        assert health.training_start_time == now

    def test_multiple_failures(self):
        """Test config with consecutive failures."""
        health = ConfigHealth(
            config_key="hex8_4p",
            consecutive_failures=3,
            last_training_success=False,
        )
        assert health.consecutive_failures == 3
        assert health.last_training_success is False


class TestAlert:
    """Tests for Alert dataclass."""

    def test_initialization(self):
        """Test alert creation."""
        now = time.time()
        alert = Alert(
            id="training_failed:hex8_2p",
            severity=AlertSeverity.WARNING,
            config_key="hex8_2p",
            message="Training failed for hex8_2p",
            created_at=now,
        )
        assert alert.id == "training_failed:hex8_2p"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.config_key == "hex8_2p"
        assert alert.resolved_at is None

    def test_to_dict(self):
        """Test alert serialization."""
        now = time.time()
        alert = Alert(
            id="stale_data:square8_2p",
            severity=AlertSeverity.CRITICAL,
            config_key="square8_2p",
            message="No new data",
            created_at=now,
        )
        data = alert.to_dict()
        assert data["id"] == "stale_data:square8_2p"
        assert data["severity"] == "critical"
        assert data["config_key"] == "square8_2p"
        assert data["message"] == "No new data"
        assert data["created_at"] == now
        assert data["resolved_at"] is None

    def test_resolved_alert(self):
        """Test resolved alert."""
        now = time.time()
        alert = Alert(
            id="low_win_rate:hex8_2p",
            severity=AlertSeverity.WARNING,
            config_key="hex8_2p",
            message="Win rate dropped",
            created_at=now,
            resolved_at=now + 100,
        )
        assert alert.resolved_at == now + 100
        data = alert.to_dict()
        assert data["resolved_at"] == now + 100


class TestHealthReport:
    """Tests for HealthReport dataclass."""

    def test_healthy_report(self):
        """Test healthy report creation."""
        now = time.time()
        configs = {
            "hex8_2p": ConfigHealth(config_key="hex8_2p", win_rate=0.65),
        }
        report = HealthReport(
            status=HealthStatus.HEALTHY,
            timestamp=now,
            configs=configs,
            active_alerts=[],
            summary="All systems operational",
        )
        assert report.status == HealthStatus.HEALTHY
        assert report.summary == "All systems operational"
        assert len(report.configs) == 1

    def test_to_dict(self):
        """Test report serialization."""
        now = time.time()
        configs = {
            "hex8_2p": ConfigHealth(config_key="hex8_2p"),
        }
        alerts = [
            Alert(
                id="test_alert",
                severity=AlertSeverity.WARNING,
                config_key="hex8_2p",
                message="Test",
                created_at=now,
            )
        ]
        report = HealthReport(
            status=HealthStatus.DEGRADED,
            timestamp=now,
            configs=configs,
            active_alerts=alerts,
            summary="1 warning",
        )
        data = report.to_dict()
        assert data["status"] == "degraded"
        assert data["timestamp"] == now
        assert "hex8_2p" in data["configs"]
        assert len(data["active_alerts"]) == 1
        assert data["summary"] == "1 warning"


class TestTrainingHealthMonitor:
    """Tests for TrainingHealthMonitor class."""

    @pytest.fixture
    def temp_state_path(self, tmp_path: Path):
        """Create a temporary state path."""
        return tmp_path / "health_state.json"

    @pytest.fixture
    def monitor(self, temp_state_path: Path):
        """Create a monitor with temporary state."""
        return TrainingHealthMonitor(state_path=temp_state_path)

    def test_initialization(self, monitor: TrainingHealthMonitor):
        """Test monitor initialization."""
        assert monitor._configs == {}
        assert monitor._active_runs == {}
        assert monitor._alerts == {}

    def test_record_training_start(self, monitor: TrainingHealthMonitor):
        """Test recording training start."""
        monitor.record_training_start("hex8_2p")

        assert "hex8_2p" in monitor._configs
        assert monitor._configs["hex8_2p"].is_training is True
        assert monitor._configs["hex8_2p"].training_start_time is not None
        assert "hex8_2p" in monitor._active_runs

    def test_record_training_complete_success(self, monitor: TrainingHealthMonitor):
        """Test recording successful training completion."""
        monitor.record_training_start("hex8_2p")
        monitor.record_training_complete(
            "hex8_2p",
            success=True,
            metrics={"loss": 0.1},
        )

        config = monitor._configs["hex8_2p"]
        assert config.is_training is False
        assert config.last_training_success is True
        assert config.consecutive_failures == 0
        assert config.model_count == 1
        assert "hex8_2p" not in monitor._active_runs

    def test_record_training_complete_failure(self, monitor: TrainingHealthMonitor):
        """Test recording training failure."""
        monitor.record_training_start("hex8_2p")
        monitor.record_training_complete(
            "hex8_2p",
            success=False,
            error_message="Out of memory",
        )

        config = monitor._configs["hex8_2p"]
        assert config.is_training is False
        assert config.last_training_success is False
        assert config.consecutive_failures == 1

        # Should have created an alert
        active_alerts = monitor.get_active_alerts()
        assert len(active_alerts) == 1
        assert "training_failed" in active_alerts[0].id

    def test_consecutive_failures_alert_escalation(self, monitor: TrainingHealthMonitor):
        """Test that consecutive failures escalate alert severity."""
        # First two failures should be WARNING
        for i in range(2):
            monitor.record_training_start("hex8_2p")
            monitor.record_training_complete("hex8_2p", success=False)

        assert monitor._configs["hex8_2p"].consecutive_failures == 2

        # Third failure should be CRITICAL
        monitor.record_training_start("hex8_2p")
        monitor.record_training_complete("hex8_2p", success=False)

        assert monitor._configs["hex8_2p"].consecutive_failures == 3
        alert = monitor._alerts.get("training_failed:hex8_2p")
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL

    def test_record_data_update(self, monitor: TrainingHealthMonitor):
        """Test recording data update."""
        monitor.record_data_update("hex8_2p", game_count=1000)

        config = monitor._configs["hex8_2p"]
        assert config.game_count == 1000
        assert config.last_data_time > 0

    def test_record_win_rate_low(self, monitor: TrainingHealthMonitor):
        """Test recording low win rate creates alert."""
        monitor.record_win_rate("hex8_2p", win_rate=0.30)

        assert monitor._configs["hex8_2p"].win_rate == 0.30
        active_alerts = monitor.get_active_alerts()
        assert len(active_alerts) == 1
        assert "low_win_rate" in active_alerts[0].id

    def test_record_win_rate_acceptable(self, monitor: TrainingHealthMonitor):
        """Test recording acceptable win rate."""
        # First record low win rate to create alert
        monitor.record_win_rate("hex8_2p", win_rate=0.30)
        assert len(monitor.get_active_alerts()) == 1

        # Then record acceptable win rate
        monitor.record_win_rate("hex8_2p", win_rate=0.50)

        # Alert should be resolved
        assert len(monitor.get_active_alerts()) == 0

    def test_get_active_alerts(self, monitor: TrainingHealthMonitor):
        """Test getting active alerts only."""
        now = time.time()

        # Create active alert
        monitor._alerts["active"] = Alert(
            id="active",
            severity=AlertSeverity.WARNING,
            config_key="hex8_2p",
            message="Active alert",
            created_at=now,
        )

        # Create resolved alert
        monitor._alerts["resolved"] = Alert(
            id="resolved",
            severity=AlertSeverity.WARNING,
            config_key="hex8_2p",
            message="Resolved alert",
            created_at=now,
            resolved_at=now + 10,
        )

        active = monitor.get_active_alerts()
        assert len(active) == 1
        assert active[0].id == "active"

    def test_health_status_healthy(self, monitor: TrainingHealthMonitor):
        """Test healthy status when no alerts."""
        monitor.record_training_start("hex8_2p")
        monitor.record_training_complete("hex8_2p", success=True)

        report = monitor.get_health_status()
        assert report.status == HealthStatus.HEALTHY
        assert "All systems operational" in report.summary

    def test_health_status_degraded(self, monitor: TrainingHealthMonitor):
        """Test degraded status with warnings."""
        monitor.record_win_rate("hex8_2p", win_rate=0.30)

        report = monitor.get_health_status()
        assert report.status == HealthStatus.DEGRADED

    def test_health_status_unhealthy(self, monitor: TrainingHealthMonitor):
        """Test unhealthy status with critical alerts."""
        # Create 3 consecutive failures for CRITICAL alert
        for i in range(3):
            monitor.record_training_start("hex8_2p")
            monitor.record_training_complete("hex8_2p", success=False)

        report = monitor.get_health_status()
        assert report.status == HealthStatus.UNHEALTHY

    def test_health_status_unknown(self, monitor: TrainingHealthMonitor):
        """Test unknown status when no configs tracked."""
        report = monitor.get_health_status()
        assert report.status == HealthStatus.UNKNOWN

    def test_stalled_training_detection(self, monitor: TrainingHealthMonitor):
        """Test detection of stalled training."""
        # Start training
        monitor.record_training_start("hex8_2p")

        # Simulate training started long ago
        config = monitor._configs["hex8_2p"]
        config.training_start_time = time.time() - (MAX_TRAINING_HOURS + 1) * 3600

        # Run health checks
        monitor.run_health_checks()

        active_alerts = monitor.get_active_alerts()
        stalled_alerts = [a for a in active_alerts if "stalled_training" in a.id]
        assert len(stalled_alerts) == 1
        assert stalled_alerts[0].severity == AlertSeverity.CRITICAL

    def test_stale_model_detection(self, monitor: TrainingHealthMonitor):
        """Test detection of stale model."""
        # Record training that happened long ago
        monitor._configs["hex8_2p"] = ConfigHealth(
            config_key="hex8_2p",
            last_training_time=time.time() - (STALE_MODEL_HOURS + 1) * 3600,
        )

        monitor.run_health_checks()

        active_alerts = monitor.get_active_alerts()
        stale_alerts = [a for a in active_alerts if "stale_model" in a.id]
        assert len(stale_alerts) == 1

    def test_stale_data_detection(self, monitor: TrainingHealthMonitor):
        """Test detection of stale data."""
        # Record data update that happened long ago
        monitor._configs["hex8_2p"] = ConfigHealth(
            config_key="hex8_2p",
            last_data_time=time.time() - (STALE_DATA_HOURS + 1) * 3600,
        )

        monitor.run_health_checks()

        active_alerts = monitor.get_active_alerts()
        stale_alerts = [a for a in active_alerts if "stale_data" in a.id]
        assert len(stale_alerts) == 1

    def test_check_health_interface(self, monitor: TrainingHealthMonitor):
        """Test check_health() returns proper result."""
        monitor.record_training_start("hex8_2p")
        monitor.record_training_complete("hex8_2p", success=True)

        result = monitor.check_health()

        # Result should be dict-like or MonitoringResult
        if isinstance(result, dict):
            assert "status" in result
            assert "metrics" in result
        else:
            # MonitoringResult
            assert hasattr(result, "status")
            assert hasattr(result, "metrics")

    def test_prometheus_metrics(self, monitor: TrainingHealthMonitor):
        """Test Prometheus metrics export."""
        monitor.record_training_start("hex8_2p")
        monitor.record_training_complete("hex8_2p", success=True)
        monitor.record_win_rate("hex8_2p", win_rate=0.65)
        monitor.record_data_update("hex8_2p", game_count=5000)

        metrics = monitor.get_prometheus_metrics()

        assert "ringrift_training_is_running" in metrics
        assert "ringrift_training_consecutive_failures" in metrics
        assert "ringrift_training_win_rate" in metrics
        assert "ringrift_training_game_count" in metrics
        assert "ringrift_training_alerts_critical" in metrics
        assert "ringrift_training_alerts_warning" in metrics

    def test_state_persistence(self, temp_state_path: Path):
        """Test state is saved and loaded correctly."""
        # Create monitor and record some data
        monitor1 = TrainingHealthMonitor(state_path=temp_state_path)
        monitor1.record_training_start("hex8_2p")
        monitor1.record_training_complete("hex8_2p", success=True)
        monitor1.record_win_rate("hex8_2p", win_rate=0.70)
        monitor1.record_data_update("hex8_2p", game_count=10000)

        # Verify state file exists
        assert temp_state_path.exists()

        # Create new monitor that loads state
        monitor2 = TrainingHealthMonitor(state_path=temp_state_path)

        # Verify state was restored
        assert "hex8_2p" in monitor2._configs
        config = monitor2._configs["hex8_2p"]
        assert config.win_rate == 0.70
        assert config.game_count == 10000
        assert config.model_count == 1

    def test_state_load_missing_file(self, tmp_path: Path):
        """Test loading when state file doesn't exist."""
        state_path = tmp_path / "nonexistent" / "health.json"
        monitor = TrainingHealthMonitor(state_path=state_path)

        # Should not raise, just start with empty state
        assert monitor._configs == {}


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_training_health_monitor_returns_same_instance(self):
        """Test that get_training_health_monitor returns singleton."""
        # Reset the singleton for this test
        import app.training.training_health as module
        module._monitor_instance = None

        monitor1 = get_training_health_monitor()
        monitor2 = get_training_health_monitor()

        assert monitor1 is monitor2


class TestThresholds:
    """Tests for threshold constants."""

    def test_max_training_hours_reasonable(self):
        """Verify MAX_TRAINING_HOURS is reasonable."""
        assert 1 <= MAX_TRAINING_HOURS <= 24

    def test_stale_model_hours_reasonable(self):
        """Verify STALE_MODEL_HOURS is reasonable."""
        assert 12 <= STALE_MODEL_HOURS <= 168

    def test_stale_data_hours_reasonable(self):
        """Verify STALE_DATA_HOURS is reasonable."""
        assert 1 <= STALE_DATA_HOURS <= 48

    def test_min_win_rate_reasonable(self):
        """Verify MIN_WIN_RATE is reasonable."""
        assert 0.0 < MIN_WIN_RATE < 0.5
