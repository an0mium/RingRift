"""Unit tests for EloProgressDaemon velocity monitoring and alerts.

February 2026 (Sprint 18): Tests for the velocity monitoring and alert
system added to the existing Elo progress daemon.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.elo_progress_daemon import (
    ALERT_COOLDOWN_SECONDS,
    MIN_VELOCITY_ELO_PER_DAY,
    REGRESSION_THRESHOLD,
    STALL_WINDOW_HOURS,
    ConfigVelocity,
    EloProgressDaemon,
    EloProgressDaemonConfig,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton between tests."""
    EloProgressDaemon.reset_instance()
    yield
    EloProgressDaemon.reset_instance()


@pytest.fixture
def daemon():
    """Create daemon with alerts enabled."""
    return EloProgressDaemon(EloProgressDaemonConfig(enabled=True))


# =============================================================================
# Initialization
# =============================================================================


class TestInit:
    def test_default_config(self, daemon):
        assert daemon.config.enabled is True
        assert daemon.config.snapshot_interval == 900.0

    def test_all_configs_tracked(self, daemon):
        assert len(daemon._velocities) == 12
        assert "hex8_2p" in daemon._velocities
        assert "hexagonal_4p" in daemon._velocities

    def test_singleton(self):
        d1 = EloProgressDaemon.get_instance()
        d2 = EloProgressDaemon.get_instance()
        assert d1 is d2


# =============================================================================
# Velocity tracking
# =============================================================================


class TestVelocityTracking:
    @pytest.mark.asyncio
    async def test_evaluation_updates_velocity(self, daemon):
        """EVALUATION_COMPLETED updates real-time velocity."""
        state = daemon._velocities["hex8_2p"]
        state.last_elo = 1400.0
        state.last_snapshot_time = time.time() - 86400  # 1 day ago

        event = MagicMock()
        event.payload = {
            "config_key": "hex8_2p",
            "elo": 1410.0,
        }

        with patch.object(daemon, "_take_snapshots"):
            await daemon._on_evaluation_completed(event)

        assert state.last_elo == 1410.0
        assert state.velocity_per_day == pytest.approx(10.0, abs=1.0)

    @pytest.mark.asyncio
    async def test_promotion_resets_regression(self, daemon):
        """MODEL_PROMOTED clears regression tracking."""
        state = daemon._velocities["hex8_2p"]
        state.consecutive_regressions = 3
        state.alert_level = "alert"
        state.stall_since = time.time() - 3600

        event = MagicMock()
        event.payload = {"config_key": "hex8_2p"}

        with patch.object(daemon, "_take_snapshots"):
            await daemon._on_model_promoted(event)

        assert state.consecutive_regressions == 0
        assert state.alert_level == "none"
        assert state.stall_since is None


# =============================================================================
# Alert system
# =============================================================================


class TestAlerts:
    def test_no_alerts_without_data(self, daemon):
        """No alerts when no snapshots recorded."""
        with patch("app.coordination.elo_progress_daemon.safe_emit_event") as mock_emit:
            daemon._check_alerts()
            mock_emit.assert_not_called()

    def test_regression_alert(self, daemon):
        """Regression triggers alert event."""
        state = daemon._velocities["hex8_2p"]
        state.last_snapshot_time = time.time()
        state.velocity_per_day = -5.0  # Below REGRESSION_THRESHOLD
        state.last_elo = 1350.0

        with patch("app.coordination.elo_progress_daemon.safe_emit_event") as mock_emit:
            daemon._check_alerts()

            assert state.alert_level == "alert"
            assert state.consecutive_regressions == 1
            mock_emit.assert_called_once()
            call_args = mock_emit.call_args
            assert call_args[0][0] == "elo_regression_alert"
            assert call_args[0][1]["severity"] == "alert"
            assert call_args[0][1]["config_key"] == "hex8_2p"

    def test_stall_warning_after_window(self, daemon):
        """Stall warning emitted after STALL_WINDOW_HOURS."""
        state = daemon._velocities["hex8_2p"]
        state.last_snapshot_time = time.time()
        state.velocity_per_day = 0.1  # Below MIN_VELOCITY but not regression
        state.last_elo = 1400.0
        state.stall_since = time.time() - (STALL_WINDOW_HOURS * 3600 + 1)

        with patch("app.coordination.elo_progress_daemon.safe_emit_event") as mock_emit:
            daemon._check_alerts()

            assert state.alert_level == "warning"
            mock_emit.assert_called_once()
            assert mock_emit.call_args[0][0] == "elo_velocity_alert"

    def test_no_stall_warning_before_window(self, daemon):
        """No stall warning before STALL_WINDOW_HOURS elapsed."""
        state = daemon._velocities["hex8_2p"]
        state.last_snapshot_time = time.time()
        state.velocity_per_day = 0.1
        state.stall_since = time.time() - 60  # Only 1 minute

        with patch("app.coordination.elo_progress_daemon.safe_emit_event") as mock_emit:
            daemon._check_alerts()
            mock_emit.assert_not_called()

    def test_recovery_clears_alerts(self, daemon):
        """Good velocity clears previous alert state."""
        state = daemon._velocities["hex8_2p"]
        state.last_snapshot_time = time.time()
        state.velocity_per_day = 5.0  # Above MIN_VELOCITY
        state.alert_level = "warning"
        state.stall_since = time.time() - 86400
        state.consecutive_regressions = 2

        daemon._check_alerts()

        assert state.alert_level == "none"
        assert state.stall_since is None
        assert state.consecutive_regressions == 0

    def test_alert_cooldown(self, daemon):
        """Respects cooldown between repeated alerts."""
        state = daemon._velocities["hex8_2p"]
        state.last_snapshot_time = time.time()
        state.velocity_per_day = -5.0
        state.last_elo = 1350.0

        with patch("app.coordination.elo_progress_daemon.safe_emit_event") as mock_emit:
            # First alert
            daemon._check_alerts()
            assert mock_emit.call_count == 1

            # Second call — should be blocked by cooldown
            daemon._check_alerts()
            assert mock_emit.call_count == 1  # Still 1

    def test_critical_multi_regression(self, daemon):
        """Critical alert when 3+ configs regress."""
        for cfg in ["hex8_2p", "hex8_3p", "hex8_4p"]:
            state = daemon._velocities[cfg]
            state.last_snapshot_time = time.time()
            state.velocity_per_day = -5.0
            state.last_elo = 1300.0

        with patch("app.coordination.elo_progress_daemon.safe_emit_event") as mock_emit:
            daemon._check_alerts()

            # 3 individual alerts + 1 critical
            assert mock_emit.call_count == 4
            critical_calls = [
                c for c in mock_emit.call_args_list
                if c[0][1].get("severity") == "critical"
            ]
            assert len(critical_calls) == 1
            assert critical_calls[0][0][1]["count"] == 3


# =============================================================================
# Summary
# =============================================================================


class TestSummary:
    def test_summary_emitted_periodically(self, daemon):
        """Summary emitted when interval elapsed."""
        daemon._last_summary_time = 0  # Force emission

        state = daemon._velocities["hex8_2p"]
        state.last_snapshot_time = time.time()
        state.velocity_per_day = 5.0
        state.last_elo = 1400.0

        with patch("app.coordination.elo_progress_daemon.safe_emit_event") as mock_emit:
            daemon._maybe_emit_summary()

            mock_emit.assert_called_once()
            assert mock_emit.call_args[0][0] == "elo_progress_summary"
            payload = mock_emit.call_args[0][1]
            assert "hex8_2p" in payload["improving"]

    def test_summary_skipped_within_interval(self, daemon):
        """Summary not emitted if interval hasn't elapsed."""
        daemon._last_summary_time = time.time()

        with patch("app.coordination.elo_progress_daemon.safe_emit_event") as mock_emit:
            daemon._maybe_emit_summary()
            mock_emit.assert_not_called()


# =============================================================================
# Health check
# =============================================================================


class TestHealthCheck:
    def test_healthy_when_running(self, daemon):
        daemon._running = True
        result = daemon.health_check()
        assert result.healthy is True
        assert "Tracking" in result.message

    def test_unhealthy_when_stopped(self, daemon):
        result = daemon.health_check()
        assert result.healthy is False
        assert "not running" in result.message

    def test_degraded_on_error(self, daemon):
        daemon._running = True
        daemon._last_error = "DB connection failed"
        result = daemon.health_check()
        assert result.healthy is False
        assert "DB connection failed" in result.message

    def test_reports_regressing_configs(self, daemon):
        daemon._running = True
        daemon._velocities["hex8_2p"].alert_level = "alert"
        daemon._velocities["hex8_2p"].last_snapshot_time = time.time()

        result = daemon.health_check()
        assert "1 regressing" in result.message
        assert "hex8_2p" in result.details["regressing"]


# =============================================================================
# Velocity summary API
# =============================================================================


class TestVelocitySummary:
    def test_returns_tracked_configs(self, daemon):
        state = daemon._velocities["hex8_2p"]
        state.last_snapshot_time = time.time()
        state.velocity_per_day = 3.5
        state.velocity_7d_per_day = 2.1
        state.last_elo = 1420.0

        summary = daemon.get_velocity_summary()
        assert "hex8_2p" in summary
        assert summary["hex8_2p"]["velocity_1d"] == 3.5
        assert summary["hex8_2p"]["velocity_7d"] == 2.1
        assert summary["hex8_2p"]["current_elo"] == 1420.0

    def test_excludes_untracked_configs(self, daemon):
        summary = daemon.get_velocity_summary()
        assert len(summary) == 0  # No configs have snapshot_time set
