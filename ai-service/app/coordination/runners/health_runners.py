"""Health and Monitoring daemon runners.

February 2026: Extracted from daemon_runners.py.

Contains runners for:
- Health & Monitoring Daemons (health_check through work_queue_monitor)
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from typing import Any

from app.coordination.runners import _wait_for_daemon

logger = logging.getLogger(__name__)


# =============================================================================
# Health & Monitoring Daemons
# =============================================================================


async def create_health_check() -> None:
    """Create and run health check daemon.

    DEPRECATED (December 2025): Use DaemonType.NODE_HEALTH_MONITOR instead.
    This runner is retained for backward compatibility and will be removed in Q2 2026.
    """
    warnings.warn(
        "DaemonType.HEALTH_CHECK is deprecated. Use DaemonType.NODE_HEALTH_MONITOR instead. "
        "Removal scheduled for Q2 2026.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from app.coordination.health_check_orchestrator import HealthCheckOrchestrator

        orchestrator = HealthCheckOrchestrator()
        await orchestrator.start()
        await _wait_for_daemon(orchestrator)
    except ImportError as e:
        logger.error(f"HealthCheckOrchestrator not available: {e}")
        raise


async def create_queue_monitor() -> None:
    """Create and run queue monitor daemon."""
    try:
        from app.coordination.queue_monitor import QueueMonitor

        monitor = QueueMonitor()
        await monitor.start()
        await _wait_for_daemon(monitor)
    except ImportError as e:
        logger.error(f"QueueMonitor not available: {e}")
        raise


async def create_daemon_watchdog() -> None:
    """Create and run daemon watchdog (December 2025)."""
    try:
        from app.coordination.daemon_watchdog import start_watchdog

        # start_watchdog() is async and already calls .start() internally
        watchdog = await start_watchdog()
        await _wait_for_daemon(watchdog)
    except ImportError as e:
        logger.error(f"DaemonWatchdog not available: {e}")
        raise


async def create_node_health_monitor() -> None:
    """Create and run node health monitor daemon.

    DEPRECATED (December 2025): Use HealthCheckOrchestrator directly via HEALTH_SERVER.
    The NODE_HEALTH_MONITOR daemon type is deprecated in favor of the unified
    health_check_orchestrator. This runner is retained for backward compatibility
    and will be removed in Q2 2026.
    """
    warnings.warn(
        "DaemonType.NODE_HEALTH_MONITOR is deprecated. Use HealthCheckOrchestrator "
        "(via DaemonType.HEALTH_SERVER) instead. Removal scheduled for Q2 2026.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from app.coordination.health_check_orchestrator import HealthCheckOrchestrator

        orchestrator = HealthCheckOrchestrator()
        await orchestrator.start()
        await _wait_for_daemon(orchestrator)
    except ImportError as e:
        logger.error(f"HealthCheckOrchestrator not available: {e}")
        raise


async def create_system_health_monitor() -> None:
    """Create and run system health monitor daemon (December 2025).

    DEPRECATED (December 2025): Use unified_health_manager.get_system_health_score() instead.
    This daemon type is deprecated in favor of unified_health_manager functions.
    This runner is retained for backward compatibility and will be removed in Q2 2026.
    """
    warnings.warn(
        "DaemonType.SYSTEM_HEALTH_MONITOR is deprecated. Use unified_health_manager."
        "get_system_health_score() instead. Removal scheduled for Q2 2026.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from app.coordination.unified_health_manager import UnifiedHealthManager

        manager = UnifiedHealthManager()
        await manager.start()
        await _wait_for_daemon(manager)
    except ImportError as e:
        logger.error(f"UnifiedHealthManager not available: {e}")
        raise


async def create_health_server() -> None:
    """Create and run HTTP health server (December 2025).

    This runner wraps the DaemonManager's _create_health_server method.
    The health server exposes endpoints at port 8790:
    - GET /health: Liveness probe
    - GET /ready: Readiness probe
    - GET /metrics: Prometheus-style metrics
    - GET /status: Detailed daemon status

    CIRCULAR DEPENDENCY NOTE (Dec 2025):
    This function imports get_daemon_manager() from daemon_manager.py.
    daemon_manager.py imports daemon_runners at top-level.
    This is SAFE because:
    1. This import is LAZY (inside function body, not at module load time)
    2. By the time this function is called, daemon_manager.py is fully loaded
    3. The circular reference is resolved at runtime, not import time
    """
    try:
        # Lazy import to avoid circular dependency with daemon_manager.py
        from app.coordination.daemon_manager import get_daemon_manager

        dm = get_daemon_manager()
        await dm._create_health_server()
    except ImportError as e:
        logger.error(f"DaemonManager not available for health server: {e}")
        raise


async def create_quality_monitor() -> None:
    """Create and run quality monitor daemon (December 2025)."""
    try:
        from app.coordination.quality_monitor_daemon import QualityMonitorDaemon

        daemon = QualityMonitorDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"QualityMonitorDaemon not available: {e}")
        raise


async def create_model_performance_watchdog() -> None:
    """Create and run model performance watchdog (December 2025)."""
    try:
        from app.coordination.model_performance_watchdog import ModelPerformanceWatchdog

        watchdog = ModelPerformanceWatchdog()
        await watchdog.start()
        await _wait_for_daemon(watchdog)
    except ImportError as e:
        logger.error(f"ModelPerformanceWatchdog not available: {e}")
        raise


async def create_cluster_monitor() -> None:
    """Create and run cluster monitor daemon."""
    try:
        from app.coordination.cluster_status_monitor import ClusterMonitor

        monitor = ClusterMonitor()
        await monitor.run_forever()
    except ImportError as e:
        logger.error(f"ClusterMonitor not available: {e}")
        raise


async def create_cluster_watchdog() -> None:
    """Create and run cluster watchdog daemon (December 2025)."""
    try:
        from app.coordination.cluster_watchdog_daemon import ClusterWatchdogDaemon

        daemon = ClusterWatchdogDaemon()
        await daemon.start()
        await _wait_for_daemon(daemon)
    except ImportError as e:
        logger.error(f"ClusterWatchdogDaemon not available: {e}")
        raise


async def create_coordinator_health_monitor() -> None:
    """Create and run coordinator health monitor daemon (December 2025).

    Subscribes to COORDINATOR_* events to track coordinator lifecycle:
    - Coordinator health status (healthy/unhealthy/degraded)
    - Heartbeat freshness monitoring
    - Init failure tracking
    """
    try:
        from app.coordination.coordinator_health_monitor_daemon import (
            get_coordinator_health_monitor,
        )

        monitor = get_coordinator_health_monitor()
        await monitor.start()
        await _wait_for_daemon(monitor)
    except ImportError as e:
        logger.error(f"CoordinatorHealthMonitorDaemon not available: {e}")
        raise


async def create_work_queue_monitor() -> None:
    """Create and run work queue monitor daemon (December 2025).

    Subscribes to WORK_* events to track queue lifecycle:
    - Queue depth tracking
    - Job latency monitoring
    - Stuck job detection
    - Backpressure signaling
    """
    try:
        from app.coordination.work_queue_monitor_daemon import (
            get_work_queue_monitor,
        )

        monitor = get_work_queue_monitor()
        await monitor.start()
        await _wait_for_daemon(monitor)
    except ImportError as e:
        logger.error(f"WorkQueueMonitorDaemon not available: {e}")
        raise
