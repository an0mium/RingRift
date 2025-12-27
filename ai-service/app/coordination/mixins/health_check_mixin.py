"""HealthCheckMixin - Standard health check implementation for coordinators.

This mixin provides a consistent health_check() implementation that can be
used across the 76+ coordinator classes that need health monitoring.

December 2025: Created as part of Phase 2 consolidation.
Estimated LOC savings: ~600 lines across codebase.

Usage:
    from app.coordination.mixins import HealthCheckMixin

    class MyCoordinator(HealthCheckMixin):
        # Mixin will provide health_check() automatically
        # Override thresholds as needed:
        UNHEALTHY_THRESHOLD = 0.5  # 50% error rate
        DEGRADED_THRESHOLD = 0.1   # 10% error rate

    # For classes with existing attributes:
    class MyDaemon(HealthCheckMixin):
        def __init__(self):
            self._running = False
            self._cycle_count = 0
            self._error_count = 0
            self._start_time = time.time()

    # The mixin will automatically use these attributes
    health = daemon.health_check()
    # Returns: HealthCheckResult(healthy=True, status=RUNNING, ...)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.coordination.contracts import CoordinatorStatus, HealthCheckResult


class HealthCheckMixin:
    """Mixin providing standard health_check() implementation.

    Subclasses can override thresholds as class attributes:
    - UNHEALTHY_THRESHOLD: Error rate above which status is unhealthy (default: 0.5)
    - DEGRADED_THRESHOLD: Error rate above which status is degraded (default: 0.1)
    - HEALTH_CHECK_WINDOW_SECONDS: Time window for "stale" detection (default: 1800)

    Expected instance attributes (if present):
    - _running: bool - Whether the component is running
    - _cycle_count / total_runs: int - Total operation cycles
    - _error_count / failed_runs: int - Total error count
    - _start_time: float - Startup timestamp
    - _last_activity_time: float - Last activity timestamp
    """

    # Thresholds can be overridden in subclasses
    UNHEALTHY_THRESHOLD: float = 0.5  # 50% error rate = unhealthy
    DEGRADED_THRESHOLD: float = 0.1   # 10% error rate = degraded
    HEALTH_CHECK_WINDOW_SECONDS: float = 1800.0  # 30 minutes without activity = stale

    def health_check(self) -> "HealthCheckResult":
        """Return health status for DaemonManager integration.

        Returns:
            HealthCheckResult with status, message, and details.
        """
        from app.coordination.contracts import CoordinatorStatus, HealthCheckResult

        # Check if running
        running = self._get_running_state()

        if not running:
            return HealthCheckResult(
                healthy=True,
                status=CoordinatorStatus.STOPPED,
                message="Component is stopped",
            )

        # Calculate error rate
        error_rate = self._get_error_rate()
        cycles = self._get_cycle_count()
        errors = self._get_error_count()
        uptime = self._get_uptime()
        last_activity = self._get_last_activity_time()

        # Determine status based on error rate
        if error_rate > self.UNHEALTHY_THRESHOLD:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message=f"High error rate: {error_rate:.1%}",
                details=self._build_details(cycles, errors, uptime, error_rate, last_activity),
            )
        elif error_rate > self.DEGRADED_THRESHOLD:
            return HealthCheckResult(
                healthy=True,  # Still functional but degraded
                status=CoordinatorStatus.DEGRADED,
                message=f"Elevated error rate: {error_rate:.1%}",
                details=self._build_details(cycles, errors, uptime, error_rate, last_activity),
            )

        # Check for stale/inactive state
        if last_activity > 0:
            seconds_since_activity = time.time() - last_activity
            if seconds_since_activity > self.HEALTH_CHECK_WINDOW_SECONDS:
                return HealthCheckResult(
                    healthy=True,
                    status=CoordinatorStatus.DEGRADED,
                    message=f"No activity for {seconds_since_activity / 60:.1f} minutes",
                    details=self._build_details(cycles, errors, uptime, error_rate, last_activity),
                )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="Healthy",
            details=self._build_details(cycles, errors, uptime, error_rate, last_activity),
        )

    def _get_running_state(self) -> bool:
        """Get running state from instance attributes."""
        return getattr(self, "_running", True)

    def _get_cycle_count(self) -> int:
        """Get cycle count from various attribute names.

        Supports attribute names from:
        - BaseDaemon: _cycles_completed
        - HandlerBase: _cycle_count
        - Generic: total_runs, _total_cycles, operations_count
        """
        return (
            getattr(self, "_cycles_completed", 0)  # BaseDaemon pattern
            or getattr(self, "_cycle_count", 0)  # HandlerBase pattern
            or getattr(self, "total_runs", 0)
            or getattr(self, "_total_cycles", 0)
            or getattr(self, "operations_count", 0)
        )

    def _get_error_count(self) -> int:
        """Get error count from various attribute names.

        Supports attribute names from:
        - BaseDaemon: _errors_count
        - HandlerBase: _error_count
        - Generic: failed_runs, _total_errors, errors_count
        """
        return (
            getattr(self, "_errors_count", 0)  # BaseDaemon pattern
            or getattr(self, "_error_count", 0)  # HandlerBase pattern
            or getattr(self, "failed_runs", 0)
            or getattr(self, "_total_errors", 0)
            or getattr(self, "errors_count", 0)
        )

    def _get_error_rate(self) -> float:
        """Calculate error rate as errors / cycles."""
        cycles = self._get_cycle_count()
        errors = self._get_error_count()
        return errors / max(cycles, 1)

    def _get_uptime(self) -> float:
        """Get uptime in seconds."""
        start_time = getattr(self, "_start_time", None)
        if start_time:
            return time.time() - start_time
        return 0.0

    def _get_last_activity_time(self) -> float:
        """Get last activity timestamp."""
        return (
            getattr(self, "_last_activity_time", 0.0)
            or getattr(self, "_last_sync_execution", 0.0)
            or getattr(self, "last_run_time", 0.0)
        )

    def _build_details(
        self,
        cycles: int,
        errors: int,
        uptime: float,
        error_rate: float,
        last_activity: float,
    ) -> dict[str, Any]:
        """Build details dict for HealthCheckResult."""
        details: dict[str, Any] = {
            "cycles": cycles,
            "errors": errors,
            "error_rate": error_rate,
            "uptime_seconds": uptime,
        }
        if last_activity > 0:
            details["last_activity"] = last_activity
            details["seconds_since_activity"] = time.time() - last_activity
        return details

    def is_healthy(self) -> bool:
        """Quick check if component is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        result = self.health_check()
        return result.healthy

    def get_health_status(self) -> str:
        """Get health status as string.

        Returns:
            Status string: "healthy", "degraded", "unhealthy", or "stopped"
        """
        result = self.health_check()
        return result.status.value
