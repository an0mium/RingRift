"""Base classes for the unified monitoring framework.

This module provides abstract base classes that all monitors should inherit from,
ensuring a consistent interface across cluster, training, and data quality monitors.

Usage:
    from app.monitoring.base import HealthMonitor, HealthStatus, Alert

    class MyMonitor(HealthMonitor):
        def check_health(self) -> MonitoringResult:
            ...
        def should_alert(self) -> Optional[Alert]:
            ...
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from app.monitoring.thresholds import AlertLevel, THRESHOLDS


class HealthStatus(str, Enum):
    """Overall health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class Alert:
    """Represents a monitoring alert."""
    level: AlertLevel
    category: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    node: Optional[str] = None
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            "level": self.level.value,
            "category": self.category,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "node": self.node,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "details": self.details,
        }

    def __str__(self) -> str:
        """Human-readable alert string."""
        parts = [f"[{self.level.value.upper()}]", self.message]
        if self.node:
            parts.insert(1, f"({self.node})")
        if self.metric_value is not None and self.threshold is not None:
            parts.append(f"[{self.metric_value:.1f}/{self.threshold:.1f}]")
        return " ".join(parts)


@dataclass
class MonitoringResult:
    """Result of a health check."""
    status: HealthStatus
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metrics: Dict[str, Any] = field(default_factory=dict)
    alerts: List[Alert] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None

    @property
    def is_healthy(self) -> bool:
        """Check if status is healthy."""
        return self.status == HealthStatus.HEALTHY

    @property
    def has_alerts(self) -> bool:
        """Check if there are any alerts."""
        return len(self.alerts) > 0

    @property
    def critical_alerts(self) -> List[Alert]:
        """Get only critical alerts."""
        return [a for a in self.alerts if a.level == AlertLevel.CRITICAL]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "alerts": [a.to_dict() for a in self.alerts],
            "details": self.details,
            "duration_ms": self.duration_ms,
        }


class HealthMonitor(ABC):
    """Abstract base class for all health monitors.

    Concrete implementations should override:
    - check_health(): Perform the actual health check
    - get_name(): Return monitor name for logging

    Optional overrides:
    - should_alert(): Custom alert logic
    - format_report(): Custom report formatting
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize monitor.

        Args:
            name: Monitor name (default: class name)
        """
        self._name = name or self.__class__.__name__
        self._last_result: Optional[MonitoringResult] = None
        self._last_check: Optional[datetime] = None

    @property
    def name(self) -> str:
        """Get monitor name."""
        return self._name

    @property
    def last_result(self) -> Optional[MonitoringResult]:
        """Get result of last health check."""
        return self._last_result

    @abstractmethod
    def check_health(self) -> MonitoringResult:
        """Perform health check and return result.

        Implementations should:
        1. Gather relevant metrics
        2. Compare against thresholds
        3. Generate alerts if needed
        4. Return MonitoringResult

        Returns:
            MonitoringResult with status, metrics, and any alerts
        """
        pass

    def should_alert(self) -> Optional[Alert]:
        """Determine if an alert should be sent based on last check.

        Default implementation returns highest-severity alert from last check.
        Override for custom alert logic.

        Returns:
            Alert to send, or None if no alert needed
        """
        if not self._last_result or not self._last_result.alerts:
            return None

        # Return most severe alert
        severity_order = [AlertLevel.FATAL, AlertLevel.CRITICAL, AlertLevel.WARNING]
        for level in severity_order:
            for alert in self._last_result.alerts:
                if alert.level == level:
                    return alert
        return self._last_result.alerts[0] if self._last_result.alerts else None

    def format_report(self) -> str:
        """Format last result as human-readable report.

        Returns:
            Formatted report string
        """
        if not self._last_result:
            return f"{self.name}: No data"

        result = self._last_result
        lines = [
            f"=== {self.name} ===",
            f"Status: {result.status.value}",
            f"Time: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        if result.metrics:
            lines.append("Metrics:")
            for key, value in result.metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.2f}")
                else:
                    lines.append(f"  {key}: {value}")

        if result.alerts:
            lines.append(f"Alerts ({len(result.alerts)}):")
            for alert in result.alerts:
                lines.append(f"  {alert}")

        return "\n".join(lines)

    def run_check(self) -> MonitoringResult:
        """Run health check and update internal state.

        This is the main entry point for running checks.
        It handles timing, state updates, and error handling.

        Returns:
            MonitoringResult from the check
        """
        import time
        start = time.time()

        try:
            result = self.check_health()
        except Exception as e:
            result = MonitoringResult(
                status=HealthStatus.UNKNOWN,
                alerts=[Alert(
                    level=AlertLevel.CRITICAL,
                    category="monitor_error",
                    message=f"Health check failed: {str(e)}",
                )],
            )

        result.duration_ms = (time.time() - start) * 1000
        self._last_result = result
        self._last_check = datetime.utcnow()

        return result


class CompositeMonitor(HealthMonitor):
    """Monitor that aggregates results from multiple sub-monitors.

    Useful for creating unified health endpoints that combine
    cluster, training, and data quality monitors.
    """

    def __init__(self, name: str = "CompositeMonitor"):
        super().__init__(name)
        self._monitors: List[HealthMonitor] = []

    def add_monitor(self, monitor: HealthMonitor) -> None:
        """Add a sub-monitor."""
        self._monitors.append(monitor)

    def remove_monitor(self, monitor: HealthMonitor) -> None:
        """Remove a sub-monitor."""
        self._monitors.remove(monitor)

    def check_health(self) -> MonitoringResult:
        """Run all sub-monitors and aggregate results."""
        all_metrics: Dict[str, Any] = {}
        all_alerts: List[Alert] = []
        all_details: Dict[str, Any] = {}
        worst_status = HealthStatus.HEALTHY

        for monitor in self._monitors:
            try:
                result = monitor.run_check()

                # Aggregate metrics with monitor name prefix
                for key, value in result.metrics.items():
                    all_metrics[f"{monitor.name}.{key}"] = value

                # Collect all alerts
                all_alerts.extend(result.alerts)

                # Store sub-monitor details
                all_details[monitor.name] = result.to_dict()

                # Update worst status
                if result.status == HealthStatus.UNHEALTHY:
                    worst_status = HealthStatus.UNHEALTHY
                elif result.status == HealthStatus.DEGRADED and worst_status == HealthStatus.HEALTHY:
                    worst_status = HealthStatus.DEGRADED
                elif result.status == HealthStatus.UNKNOWN and worst_status == HealthStatus.HEALTHY:
                    worst_status = HealthStatus.UNKNOWN

            except Exception as e:
                all_alerts.append(Alert(
                    level=AlertLevel.WARNING,
                    category="sub_monitor_error",
                    message=f"Sub-monitor {monitor.name} failed: {str(e)}",
                ))

        return MonitoringResult(
            status=worst_status,
            metrics=all_metrics,
            alerts=all_alerts,
            details=all_details,
        )
