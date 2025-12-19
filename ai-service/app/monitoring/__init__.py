"""Unified monitoring framework for RingRift cluster health and training pipeline.

This module provides:
- Centralized alert thresholds (thresholds.py)
- Base classes for health monitors (base.py)
- P2P-integrated monitoring (p2p_monitoring.py)
- Predictive alerting (predictive_alerts.py)
- Training dashboard (training_dashboard.py)

Usage:
    from app.monitoring import HealthMonitor, THRESHOLDS, AlertLevel
    from app.monitoring.thresholds import get_threshold, should_alert

    # Check threshold
    if should_alert("disk", 75, "warning"):
        send_warning()

    # Create custom monitor
    class MyMonitor(HealthMonitor):
        def check_health(self) -> MonitoringResult:
            ...
"""

# Thresholds
from app.monitoring.thresholds import (
    THRESHOLDS,
    AlertLevel,
    get_threshold,
    should_alert,
    get_all_thresholds,
)

# Base classes
from app.monitoring.base import (
    HealthMonitor,
    HealthStatus,
    Alert,
    MonitoringResult,
    CompositeMonitor,
)

# P2P monitoring
from app.monitoring.p2p_monitoring import MonitoringManager

__all__ = [
    # Thresholds
    "THRESHOLDS",
    "AlertLevel",
    "get_threshold",
    "should_alert",
    "get_all_thresholds",
    # Base classes
    "HealthMonitor",
    "HealthStatus",
    "Alert",
    "MonitoringResult",
    "CompositeMonitor",
    # P2P monitoring
    "MonitoringManager",
]
