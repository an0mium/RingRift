"""Daemon metrics collection for Prometheus-style health endpoints.

Extracted from daemon_manager.py (December 2025) for cleaner separation of concerns.
This module collects metrics from various daemons and formats them in Prometheus format.

Usage:
    from app.coordination.daemon_metrics import DaemonMetricsCollector

    collector = DaemonMetricsCollector(health_summary_fn)
    metrics_text = collector.render_metrics()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DaemonMetricsCollector:
    """Collects and formats Prometheus-style metrics from daemons.

    This class centralizes metrics collection from various daemon subsystems:
    - Daemon counts and health scores (from DaemonManager)
    - Selfplay scheduler throughput
    - Cluster sync throughput
    - Event router statistics
    """

    def __init__(
        self,
        health_summary_fn: Callable[[], dict[str, Any]],
    ) -> None:
        """Initialize the metrics collector.

        Args:
            health_summary_fn: Callable that returns daemon health summary dict
                              (typically DaemonManager.health_summary)
        """
        self._health_summary_fn = health_summary_fn

    def render_metrics(self) -> str:
        """Render Prometheus-style metrics for the health server.

        Returns:
            Prometheus-formatted metrics text
        """
        metrics_blob = self._collect_prometheus_metrics()
        manual_metrics = self._collect_manual_metrics()

        if metrics_blob:
            return f"{metrics_blob.rstrip()}\n\n{manual_metrics}"
        return manual_metrics

    def _collect_prometheus_metrics(self) -> str:
        """Collect metrics from prometheus_client if available.

        Returns:
            Prometheus metrics blob as string, or empty string if unavailable
        """
        try:
            from app.utils.optional_imports import (
                PROMETHEUS_AVAILABLE,
                generate_latest,
            )

            if PROMETHEUS_AVAILABLE:
                payload = generate_latest()
                if isinstance(payload, bytes):
                    return payload.decode("utf-8", errors="replace")
                return str(payload)
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning(f"Failed to collect Prometheus metrics: {e}")

        return ""

    def _collect_manual_metrics(self) -> str:
        """Collect manually-defined metrics from daemon subsystems.

        Returns:
            Prometheus-formatted metrics text
        """
        lines: list[str] = []

        # Daemon health metrics
        lines.extend(self._collect_daemon_health_metrics())

        # Selfplay scheduler metrics
        lines.extend(self._collect_selfplay_metrics())

        # Cluster sync metrics
        lines.extend(self._collect_sync_metrics())

        # Event router metrics
        lines.extend(self._collect_event_router_metrics())

        return "\n".join(lines)

    def _collect_daemon_health_metrics(self) -> list[str]:
        """Collect daemon count and health metrics.

        Returns:
            List of Prometheus-formatted metric lines
        """
        summary = self._health_summary_fn()

        running = summary.get("running", 0)
        failed = summary.get("failed", 0)
        total = summary.get("total", 0)
        stopped = max(0, total - running - failed)
        health_score = summary.get("score", 0.0)

        return [
            "# HELP daemon_count Number of daemons",
            "# TYPE daemon_count gauge",
            f'daemon_count{{state="running"}} {running}',
            f'daemon_count{{state="stopped"}} {stopped}',
            f'daemon_count{{state="failed"}} {failed}',
            "",
            "# HELP daemon_health_score Overall health score (0-1)",
            "# TYPE daemon_health_score gauge",
            f"daemon_health_score {health_score}",
            "",
            "# HELP daemon_uptime_seconds Daemon manager uptime",
            "# TYPE daemon_uptime_seconds counter",
            f'daemon_uptime_seconds {summary.get("liveness", {}).get("uptime_seconds", 0)}',
        ]

    def _collect_selfplay_metrics(self) -> list[str]:
        """Collect selfplay scheduler throughput metrics.

        Returns:
            List of Prometheus-formatted metric lines
        """
        try:
            from app.coordination.selfplay_scheduler import get_selfplay_scheduler

            metrics = get_selfplay_scheduler().get_metrics()
            return [
                "",
                "# HELP selfplay_games_allocated_total Total selfplay games allocated",
                "# TYPE selfplay_games_allocated_total counter",
                f"selfplay_games_allocated_total {metrics.get('games_allocated_total', 0)}",
                "# HELP selfplay_games_allocated_last_hour Selfplay games allocated in last hour",
                "# TYPE selfplay_games_allocated_last_hour gauge",
                f"selfplay_games_allocated_last_hour {metrics.get('games_allocated_last_hour', 0)}",
                "# HELP selfplay_games_per_hour Current selfplay allocation rate",
                "# TYPE selfplay_games_per_hour gauge",
                f"selfplay_games_per_hour {metrics.get('games_per_hour', 0.0)}",
            ]
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning(f"Failed to collect selfplay scheduler metrics: {e}")
            return []

    def _collect_sync_metrics(self) -> list[str]:
        """Collect cluster sync throughput metrics.

        Returns:
            List of Prometheus-formatted metric lines
        """
        try:
            from app.coordination.auto_sync_daemon import get_auto_sync_daemon

            metrics = get_auto_sync_daemon().get_metrics()
            return [
                "",
                "# HELP cluster_sync_count_total Total sync cycles executed",
                "# TYPE cluster_sync_count_total counter",
                f"cluster_sync_count_total {metrics.get('sync_count', 0)}",
                "# HELP cluster_sync_bytes_last_cycle Bytes synced in last cycle",
                "# TYPE cluster_sync_bytes_last_cycle gauge",
                f"cluster_sync_bytes_last_cycle {metrics.get('last_sync_bytes', 0)}",
                "# HELP cluster_sync_throughput_bytes_per_sec Last cycle throughput (bytes/sec)",
                "# TYPE cluster_sync_throughput_bytes_per_sec gauge",
                f"cluster_sync_throughput_bytes_per_sec {metrics.get('last_sync_throughput_bps', 0.0)}",
                "# HELP cluster_sync_total_bytes Total bytes synced",
                "# TYPE cluster_sync_total_bytes counter",
                f"cluster_sync_total_bytes {metrics.get('total_bytes_synced', 0)}",
            ]
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning(f"Failed to collect cluster sync metrics: {e}")
            return []

    def _collect_event_router_metrics(self) -> list[str]:
        """Collect event router statistics.

        Returns:
            List of Prometheus-formatted metric lines
        """
        try:
            from app.coordination.event_router import get_router

            stats = get_router().get_stats()
            lines = [
                "",
                "# HELP event_router_events_routed_total Total events routed",
                "# TYPE event_router_events_routed_total counter",
                f"event_router_events_routed_total {stats.get('total_events_routed', 0)}",
                "# HELP event_router_duplicates_prevented_total Duplicate events prevented",
                "# TYPE event_router_duplicates_prevented_total counter",
                f"event_router_duplicates_prevented_total {stats.get('duplicates_prevented', 0)}",
                "# HELP event_router_content_duplicates_prevented_total Content-hash duplicates prevented",
                "# TYPE event_router_content_duplicates_prevented_total counter",
                f"event_router_content_duplicates_prevented_total {stats.get('content_duplicates_prevented', 0)}",
                "# HELP event_router_events_routed_by_type_total Events routed by type",
                "# TYPE event_router_events_routed_by_type_total counter",
            ]

            # Add per-event-type breakdown
            for event_type, count in stats.get("events_routed_by_type", {}).items():
                safe_event = str(event_type).replace('"', "'")
                lines.append(
                    f'event_router_events_routed_by_type_total{{event="{safe_event}"}} {count}'
                )

            return lines
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning(f"Failed to collect event router metrics: {e}")
            return []


# Singleton for shared access
_metrics_collector: DaemonMetricsCollector | None = None


def get_daemon_metrics_collector(
    health_summary_fn: Callable[[], dict[str, Any]] | None = None,
) -> DaemonMetricsCollector:
    """Get the shared DaemonMetricsCollector instance.

    Args:
        health_summary_fn: Health summary function (required on first call)

    Returns:
        DaemonMetricsCollector singleton instance
    """
    global _metrics_collector
    if _metrics_collector is None:
        if health_summary_fn is None:
            raise ValueError("health_summary_fn required for first initialization")
        _metrics_collector = DaemonMetricsCollector(health_summary_fn)
    return _metrics_collector
