"""Metrics collection and health monitoring for SelfplayScheduler.

December 30, 2025: Extracted from selfplay_scheduler.py to:
1. Reduce SelfplayScheduler complexity (~4,200 LOC god object)
2. Enable independent testing of metrics logic
3. Provide clean observability layer for Prometheus integration

This class handles:
- Rolling window allocation tracking (games per hour)
- Total allocation counts
- Status reporting for health checks
- Throughput metrics for monitoring dashboards

Usage:
    from app.coordination.scheduler_metrics import SchedulerMetricsCollector

    metrics = SchedulerMetricsCollector(window_seconds=3600)

    # Record allocations as they happen
    metrics.record_allocation(games=50)

    # Get throughput metrics
    print(f"Games/hour: {metrics.get_games_per_hour()}")

    # Get full metrics dict for monitoring
    status = metrics.get_metrics()
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class AllocationRecord:
    """Single allocation record with timestamp.

    Attributes:
        timestamp: Unix timestamp when allocation occurred
        games: Number of games allocated
    """
    timestamp: float
    games: int


@dataclass
class SchedulerMetrics:
    """Metrics summary for scheduler health monitoring.

    Attributes:
        games_allocated_total: Total games allocated since startup
        games_allocated_window: Games allocated in current window
        games_per_hour: Computed throughput (games/hour)
        window_seconds: Size of rolling window in seconds
        allocation_count: Number of allocation events in window
        last_allocation_time: Unix timestamp of last allocation (or None)
    """
    games_allocated_total: int = 0
    games_allocated_window: int = 0
    games_per_hour: float = 0.0
    window_seconds: float = 3600.0
    allocation_count: int = 0
    last_allocation_time: Optional[float] = None


class SchedulerMetricsCollector:
    """Metrics collection and throughput tracking for SelfplayScheduler.

    Tracks allocation metrics using a rolling window approach. Designed
    for integration with monitoring systems (Prometheus, Grafana, etc.)
    and health check endpoints.

    Thread Safety:
        This class is NOT thread-safe. For concurrent access, use external
        synchronization or create per-thread instances.

    Attributes:
        window_seconds: Size of rolling window for throughput calculation
    """

    # Class-level constants
    DEFAULT_WINDOW_SECONDS = 3600.0  # 1 hour window
    MAX_HISTORY_SIZE = 10000  # Max allocation records to keep

    def __init__(
        self,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        max_history_size: int = MAX_HISTORY_SIZE,
    ):
        """Initialize the metrics collector.

        Args:
            window_seconds: Size of rolling window in seconds (default: 3600s)
            max_history_size: Max allocation records to keep (default: 10000)
        """
        self._window_seconds = window_seconds
        self._max_history_size = max_history_size

        # Rolling window for allocation tracking
        # Using deque with maxlen for automatic pruning
        self._allocation_history: deque[AllocationRecord] = deque(maxlen=max_history_size)

        # Total counters (since startup)
        self._games_allocated_total = 0
        self._allocation_events_total = 0

        # Logging prefix
        self._log_prefix = "[SchedulerMetrics]"

    @property
    def window_seconds(self) -> float:
        """Get the rolling window size in seconds."""
        return self._window_seconds

    @window_seconds.setter
    def window_seconds(self, value: float) -> None:
        """Set the rolling window size (prunes old entries on change)."""
        self._window_seconds = value
        self._prune_old_entries()

    def record_allocation(self, games: int) -> None:
        """Record a new allocation event.

        Args:
            games: Number of games allocated (must be positive)
        """
        if games <= 0:
            return

        now = time.time()
        self._games_allocated_total += games
        self._allocation_events_total += 1
        self._allocation_history.append(AllocationRecord(timestamp=now, games=games))

        # Prune old entries outside window
        self._prune_old_entries()

    def _prune_old_entries(self) -> int:
        """Remove entries older than the window.

        Returns:
            Number of entries pruned
        """
        if not self._allocation_history:
            return 0

        now = time.time()
        cutoff = now - self._window_seconds
        pruned = 0

        while self._allocation_history and self._allocation_history[0].timestamp < cutoff:
            self._allocation_history.popleft()
            pruned += 1

        return pruned

    def get_games_in_window(self) -> int:
        """Get total games allocated in current window.

        Returns:
            Sum of games allocated in rolling window
        """
        self._prune_old_entries()
        return sum(record.games for record in self._allocation_history)

    def get_games_per_hour(self) -> float:
        """Get games per hour throughput.

        Returns:
            Computed games/hour based on rolling window
        """
        window_games = self.get_games_in_window()
        window_hours = self._window_seconds / 3600.0
        return window_games / window_hours if window_hours > 0 else 0.0

    def get_allocation_rate(self) -> float:
        """Get allocation events per hour.

        Returns:
            Allocation events per hour in current window
        """
        self._prune_old_entries()
        event_count = len(self._allocation_history)
        window_hours = self._window_seconds / 3600.0
        return event_count / window_hours if window_hours > 0 else 0.0

    def get_average_allocation_size(self) -> float:
        """Get average games per allocation event.

        Returns:
            Average games per allocation, or 0.0 if no allocations
        """
        self._prune_old_entries()
        if not self._allocation_history:
            return 0.0
        total_games = sum(record.games for record in self._allocation_history)
        return total_games / len(self._allocation_history)

    def get_last_allocation_time(self) -> Optional[float]:
        """Get timestamp of last allocation.

        Returns:
            Unix timestamp or None if no allocations
        """
        if self._allocation_history:
            return self._allocation_history[-1].timestamp
        return None

    def get_metrics(self) -> dict[str, Any]:
        """Get throughput metrics for monitoring.

        Returns:
            Dict with all metrics for dashboards/health checks
        """
        self._prune_old_entries()
        window_games = sum(record.games for record in self._allocation_history)
        window_hours = self._window_seconds / 3600.0
        games_per_hour = window_games / window_hours if window_hours > 0 else 0.0

        return {
            "games_allocated_total": self._games_allocated_total,
            "games_allocated_last_hour": window_games,
            "games_per_hour": round(games_per_hour, 2),
            "allocation_window_seconds": self._window_seconds,
            "allocation_events_total": self._allocation_events_total,
            "allocation_events_in_window": len(self._allocation_history),
            "average_allocation_size": round(self.get_average_allocation_size(), 2),
            "last_allocation_time": self.get_last_allocation_time(),
        }

    def get_summary(self) -> SchedulerMetrics:
        """Get metrics as a structured dataclass.

        Returns:
            SchedulerMetrics instance with current values
        """
        self._prune_old_entries()
        window_games = sum(record.games for record in self._allocation_history)
        window_hours = self._window_seconds / 3600.0

        return SchedulerMetrics(
            games_allocated_total=self._games_allocated_total,
            games_allocated_window=window_games,
            games_per_hour=window_games / window_hours if window_hours > 0 else 0.0,
            window_seconds=self._window_seconds,
            allocation_count=len(self._allocation_history),
            last_allocation_time=self.get_last_allocation_time(),
        )

    def get_status(self, scheduler_state: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Get full status for health check endpoints.

        Args:
            scheduler_state: Optional dict with scheduler state to include

        Returns:
            Combined status dict with metrics and optional scheduler state
        """
        status = {
            "metrics": self.get_metrics(),
            "health": self._compute_health_status(),
        }

        if scheduler_state:
            status["scheduler"] = scheduler_state

        return status

    def _compute_health_status(self) -> dict[str, Any]:
        """Compute health status based on metrics.

        Returns:
            Dict with health indicators
        """
        games_per_hour = self.get_games_per_hour()
        last_alloc = self.get_last_allocation_time()
        now = time.time()

        # Health indicators
        is_active = last_alloc is not None and (now - last_alloc) < 300  # Active in last 5 min
        is_producing = games_per_hour > 0
        throughput_level = (
            "high" if games_per_hour > 1000 else
            "medium" if games_per_hour > 100 else
            "low" if games_per_hour > 0 else
            "none"
        )

        return {
            "is_active": is_active,
            "is_producing": is_producing,
            "throughput_level": throughput_level,
            "seconds_since_last_allocation": round(now - last_alloc, 1) if last_alloc else None,
        }

    def reset(self) -> None:
        """Reset all metrics to initial state.

        Call this when scheduler restarts or for testing.
        """
        self._allocation_history.clear()
        self._games_allocated_total = 0
        self._allocation_events_total = 0
        logger.debug(f"{self._log_prefix} Metrics reset")


# Convenience factory
def create_metrics_collector(
    window_seconds: float = SchedulerMetricsCollector.DEFAULT_WINDOW_SECONDS,
) -> SchedulerMetricsCollector:
    """Create a SchedulerMetricsCollector with default configuration.

    Args:
        window_seconds: Rolling window size in seconds

    Returns:
        Configured SchedulerMetricsCollector instance
    """
    return SchedulerMetricsCollector(window_seconds=window_seconds)
