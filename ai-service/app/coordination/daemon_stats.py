"""Base statistics classes for coordination daemons.

December 2025: Consolidates common metric patterns found across 29+ daemon
implementations. Subclasses extend with daemon-specific metrics.

Common patterns unified:
- Error tracking (errors_count, last_error)
- Time tracking (last_check_time, uptime)
- Operation counters (attempted, completed, failed)
- Health indicators (consecutive_failures, is_healthy)

Usage:
    from app.coordination.daemon_stats import DaemonStatsBase

    @dataclass
    class MySyncStats(DaemonStatsBase):
        '''Extended stats for sync daemon.'''
        nodes_synced: int = 0
        bytes_transferred: int = 0

    stats = MySyncStats()
    stats.record_success(duration_seconds=1.5)
    stats.record_item_processed(item_count=10, bytes_count=1024)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DaemonStatsBase:
    """Base statistics class for all coordination daemons.

    Provides consistent metric tracking patterns:
    - Lifecycle metrics (operations attempted/completed/failed)
    - Error tracking (counts, last error message)
    - Timing metrics (last check, uptime, average duration)
    - Health indicators (consecutive failures, error rate)

    Subclasses add daemon-specific metrics while inheriting
    common tracking methods.
    """

    # Core lifecycle metrics (universal)
    operations_attempted: int = 0
    operations_completed: int = 0
    operations_failed: int = 0

    # Error tracking
    errors_count: int = 0
    last_error: str | None = None
    critical_errors: int = 0

    # Timing metrics
    last_check_time: float = 0.0
    start_time: float = field(default_factory=time.time)
    avg_operation_duration: float = 0.0

    # Volume metrics (for data operations)
    total_items_processed: int = 0
    total_bytes_processed: int = 0

    # Health indicators
    consecutive_failures: int = 0
    recovery_attempts: int = 0

    def record_attempt(self) -> None:
        """Record an operation attempt."""
        self.operations_attempted += 1
        self.last_check_time = time.time()

    def record_success(self, duration_seconds: float = 0.0) -> None:
        """Record a successful operation.

        Args:
            duration_seconds: How long the operation took (for averaging)
        """
        self.operations_completed += 1
        self.consecutive_failures = 0
        self.last_check_time = time.time()
        if duration_seconds > 0:
            self._update_avg_duration(duration_seconds)

    def record_failure(self, error: str | Exception | None = None, critical: bool = False) -> None:
        """Record a failed operation.

        Args:
            error: Error message or exception
            critical: If True, increment critical_errors counter
        """
        self.operations_failed += 1
        self.errors_count += 1
        self.consecutive_failures += 1
        self.last_check_time = time.time()
        if error is not None:
            self.last_error = str(error)
        if critical:
            self.critical_errors += 1

    def record_item_processed(self, item_count: int = 1, bytes_count: int = 0) -> None:
        """Record items processed (useful for sync, cleanup, etc.).

        Args:
            item_count: Number of items processed
            bytes_count: Number of bytes processed (if applicable)
        """
        self.total_items_processed += item_count
        if bytes_count > 0:
            self.total_bytes_processed += bytes_count

    def record_recovery(self, success: bool = True) -> None:
        """Record a recovery attempt.

        Args:
            success: Whether recovery was successful
        """
        self.recovery_attempts += 1
        if success:
            self.consecutive_failures = 0

    def _update_avg_duration(self, new_duration: float) -> None:
        """Update running average of operation duration.

        Uses exponential moving average (90% old, 10% new).
        """
        if self.avg_operation_duration == 0:
            self.avg_operation_duration = new_duration
        else:
            self.avg_operation_duration = (
                0.9 * self.avg_operation_duration + 0.1 * new_duration
            )

    @property
    def uptime_seconds(self) -> float:
        """How long the daemon has been running."""
        return time.time() - self.start_time

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage (0.0 - 1.0)."""
        if self.operations_attempted == 0:
            return 1.0
        return self.operations_completed / self.operations_attempted

    @property
    def error_rate(self) -> float:
        """Error rate as a percentage (0.0 - 1.0)."""
        if self.operations_attempted == 0:
            return 0.0
        return self.errors_count / self.operations_attempted

    def is_healthy(self, max_error_rate: float = 0.1, max_consecutive_failures: int = 5) -> bool:
        """Simple health check based on error rate and failure streak.

        Args:
            max_error_rate: Maximum allowed error rate (default 10%)
            max_consecutive_failures: Maximum allowed consecutive failures

        Returns:
            True if daemon is healthy
        """
        if self.operations_attempted == 0:
            return True
        if self.consecutive_failures >= max_consecutive_failures:
            return False
        return self.error_rate < max_error_rate

    def reset_counters(self) -> None:
        """Reset all counters (useful for testing or periodic resets)."""
        self.operations_attempted = 0
        self.operations_completed = 0
        self.operations_failed = 0
        self.errors_count = 0
        self.critical_errors = 0
        self.consecutive_failures = 0
        self.recovery_attempts = 0
        self.total_items_processed = 0
        self.total_bytes_processed = 0
        self.last_error = None
        self.avg_operation_duration = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary for JSON serialization.

        Subclasses should override to include additional fields.
        """
        return {
            "operations_attempted": self.operations_attempted,
            "operations_completed": self.operations_completed,
            "operations_failed": self.operations_failed,
            "errors_count": self.errors_count,
            "last_error": self.last_error,
            "critical_errors": self.critical_errors,
            "last_check_time": self.last_check_time,
            "uptime_seconds": self.uptime_seconds,
            "avg_operation_duration": self.avg_operation_duration,
            "total_items_processed": self.total_items_processed,
            "total_bytes_processed": self.total_bytes_processed,
            "consecutive_failures": self.consecutive_failures,
            "recovery_attempts": self.recovery_attempts,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "is_healthy": self.is_healthy(),
        }


# =============================================================================
# Common Specialized Stats Classes
# =============================================================================


@dataclass
class SyncDaemonStats(DaemonStatsBase):
    """Extended stats for data synchronization daemons.

    Adds sync-specific metrics beyond base stats.
    """

    syncs_completed: int = 0
    syncs_failed: int = 0
    nodes_synced: int = 0
    files_synced: int = 0
    bytes_synced: int = 0
    last_sync_duration: float = 0.0

    def record_sync_success(self, duration: float, nodes: int = 0, files: int = 0, bytes_count: int = 0) -> None:
        """Record a successful sync operation."""
        self.record_success(duration_seconds=duration)
        self.syncs_completed += 1
        self.last_sync_duration = duration
        if nodes > 0:
            self.nodes_synced += nodes
        if files > 0:
            self.files_synced += files
        if bytes_count > 0:
            self.bytes_synced += bytes_count

    def record_sync_failure(self, error: str | Exception) -> None:
        """Record a failed sync operation."""
        self.record_failure(error)
        self.syncs_failed += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        base = super().to_dict()
        base.update({
            "syncs_completed": self.syncs_completed,
            "syncs_failed": self.syncs_failed,
            "nodes_synced": self.nodes_synced,
            "files_synced": self.files_synced,
            "bytes_synced": self.bytes_synced,
            "last_sync_duration": self.last_sync_duration,
        })
        return base


@dataclass
class CleanupDaemonStats(DaemonStatsBase):
    """Extended stats for cleanup/maintenance daemons."""

    items_scanned: int = 0
    items_cleaned: int = 0
    items_quarantined: int = 0
    bytes_reclaimed: int = 0
    last_scan_time: float = 0.0

    def record_cleanup(self, scanned: int = 0, cleaned: int = 0, quarantined: int = 0, bytes_reclaimed: int = 0) -> None:
        """Record a cleanup operation."""
        self.items_scanned += scanned
        self.items_cleaned += cleaned
        self.items_quarantined += quarantined
        self.bytes_reclaimed += bytes_reclaimed
        self.last_scan_time = time.time()
        self.record_item_processed(item_count=cleaned, bytes_count=bytes_reclaimed)

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        base = super().to_dict()
        base.update({
            "items_scanned": self.items_scanned,
            "items_cleaned": self.items_cleaned,
            "items_quarantined": self.items_quarantined,
            "bytes_reclaimed": self.bytes_reclaimed,
            "last_scan_time": self.last_scan_time,
        })
        return base


@dataclass
class JobDaemonStats(DaemonStatsBase):
    """Extended stats for job-related daemons (reaper, spawner, etc.)."""

    jobs_processed: int = 0
    jobs_succeeded: int = 0
    jobs_failed: int = 0
    jobs_timed_out: int = 0
    jobs_reassigned: int = 0
    last_job_time: float = 0.0

    def record_job_success(self, duration: float = 0.0) -> None:
        """Record a successfully processed job."""
        self.record_success(duration_seconds=duration)
        self.jobs_processed += 1
        self.jobs_succeeded += 1
        self.last_job_time = time.time()

    def record_job_failure(self, error: str | Exception | None = None) -> None:
        """Record a failed job."""
        self.record_failure(error)
        self.jobs_processed += 1
        self.jobs_failed += 1
        self.last_job_time = time.time()

    def record_job_timeout(self) -> None:
        """Record a timed-out job."""
        self.jobs_processed += 1
        self.jobs_timed_out += 1
        self.last_job_time = time.time()

    def record_job_reassigned(self) -> None:
        """Record a reassigned job."""
        self.jobs_reassigned += 1
        self.last_job_time = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        base = super().to_dict()
        base.update({
            "jobs_processed": self.jobs_processed,
            "jobs_succeeded": self.jobs_succeeded,
            "jobs_failed": self.jobs_failed,
            "jobs_timed_out": self.jobs_timed_out,
            "jobs_reassigned": self.jobs_reassigned,
            "last_job_time": self.last_job_time,
        })
        return base


@dataclass
class EvaluationDaemonStats(DaemonStatsBase):
    """Extended stats for evaluation/gauntlet daemons."""

    evaluations_triggered: int = 0
    evaluations_completed: int = 0
    evaluations_failed: int = 0
    games_played: int = 0
    models_evaluated: int = 0
    promotions_triggered: int = 0
    last_evaluation_time: float = 0.0
    avg_evaluation_duration: float = 0.0

    def record_evaluation_success(self, duration: float, games: int = 0, promoted: bool = False) -> None:
        """Record a successful evaluation."""
        self.record_success(duration_seconds=duration)
        self.evaluations_completed += 1
        self.models_evaluated += 1
        self.last_evaluation_time = time.time()
        if games > 0:
            self.games_played += games
        if promoted:
            self.promotions_triggered += 1
        # Update evaluation-specific average
        if self.avg_evaluation_duration == 0:
            self.avg_evaluation_duration = duration
        else:
            self.avg_evaluation_duration = 0.9 * self.avg_evaluation_duration + 0.1 * duration

    def record_evaluation_failure(self, error: str | Exception) -> None:
        """Record a failed evaluation."""
        self.record_failure(error)
        self.evaluations_failed += 1
        self.last_evaluation_time = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        base = super().to_dict()
        base.update({
            "evaluations_triggered": self.evaluations_triggered,
            "evaluations_completed": self.evaluations_completed,
            "evaluations_failed": self.evaluations_failed,
            "games_played": self.games_played,
            "models_evaluated": self.models_evaluated,
            "promotions_triggered": self.promotions_triggered,
            "last_evaluation_time": self.last_evaluation_time,
            "avg_evaluation_duration": self.avg_evaluation_duration,
        })
        return base


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "DaemonStatsBase",
    "SyncDaemonStats",
    "CleanupDaemonStats",
    "JobDaemonStats",
    "EvaluationDaemonStats",
]
