"""Unified cluster health module (December 2025).

Consolidates health-related functionality from:
- unified_health_manager.py (error recovery, circuit breakers)
- host_health_policy.py (pre-spawn SSH health checks)
- node_health_monitor.py (async node monitoring, eviction)

This module re-exports all health-related APIs for unified access.

Usage:
    from app.coordination.cluster.health import (
        # Health manager
        UnifiedHealthManager,
        get_health_manager,
        wire_health_events,

        # Host health (pre-spawn checks)
        check_host_health,
        is_host_healthy,
        get_healthy_hosts,

        # Node monitoring
        NodeHealthMonitor,
        get_node_health_monitor,
        NodeStatus,
    )
"""

from __future__ import annotations

import warnings

# Re-export from unified_health_manager
from app.coordination.unified_health_manager import (
    UnifiedHealthManager,
    get_health_manager,
    wire_health_events,
    ErrorSeverity,
    RecoveryStatus,
)

# Re-export from host_health_policy
from app.coordination.host_health_policy import (
    HealthStatus,
    check_host_health,
    is_host_healthy,
    get_healthy_hosts,
    clear_health_cache,
    get_health_summary,
    is_cluster_healthy,
    check_cluster_health,
)

# Re-export from node_health_monitor (deprecated but kept for backward compatibility)
# December 2025: node_health_monitor is deprecated in favor of health_check_orchestrator,
# but these re-exports are kept for callers that depend on the NodeHealthMonitor API.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from app.coordination.node_health_monitor import (
        NodeHealthMonitor,
        get_node_health_monitor,
        NodeStatus,
        NodeHealth,
    )

__all__ = [
    # From unified_health_manager
    "UnifiedHealthManager",
    "get_health_manager",
    "wire_health_events",
    "ErrorSeverity",
    "RecoveryStatus",
    # From host_health_policy
    "HealthStatus",
    "check_host_health",
    "is_host_healthy",
    "get_healthy_hosts",
    "clear_health_cache",
    "get_health_summary",
    "is_cluster_healthy",
    "check_cluster_health",
    # From node_health_monitor
    "NodeHealthMonitor",
    "get_node_health_monitor",
    "NodeStatus",
    "NodeHealth",
]
