"""Centralized alert thresholds for all monitoring systems.

This module provides a single source of truth for all alert thresholds,
eliminating scattered hardcoded values across 21+ monitoring scripts.

Usage:
    from app.monitoring.thresholds import THRESHOLDS, get_threshold, should_alert

    # Get specific threshold
    disk_warn = get_threshold("disk", "warning")  # Returns 70

    # Check if value exceeds threshold
    if should_alert("gpu_utilization", current_util, level="warning"):
        send_alert(...)
"""

from enum import Enum
from typing import Any, Dict, Optional


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


# Master threshold configuration
# All monitoring scripts should reference these values instead of hardcoding
THRESHOLDS: Dict[str, Dict[str, Any]] = {
    # Disk space monitoring
    "disk": {
        "warning": 70,       # % used - send warning
        "critical": 85,      # % used - pause operations
        "fatal": 95,         # % used - stop all writes
        "unit": "percent",
        "description": "Disk space utilization thresholds",
    },

    # GPU monitoring
    "gpu_utilization": {
        "idle": 5,           # % - considered idle
        "low": 20,           # % - below expected
        "normal": 50,        # % - acceptable minimum
        "unit": "percent",
        "description": "GPU utilization levels",
    },
    "gpu_memory": {
        "warning": 85,       # % of VRAM used
        "critical": 95,      # % of VRAM used
        "unit": "percent",
        "description": "GPU memory utilization",
    },

    # Training monitoring
    "training": {
        "stale_hours": 24,           # Hours without progress
        "model_stale_hours": 48,     # Hours since last checkpoint
        "min_batch_rate": 10,        # Batches per minute minimum
        "max_loss_increase": 0.5,    # Loss increase threshold for divergence
        "description": "Training progress thresholds",
    },

    # Data quality monitoring
    "data_quality": {
        "draw_rate_threshold": 0.20,     # 20% draw rate is concerning
        "min_game_length": 10,           # Games shorter than this are suspect
        "max_game_length": 500,          # Games longer than this are suspect
        "nan_threshold": 0.001,          # 0.1% NaN rate in features
        "zero_feature_threshold": 0.05,  # 5% all-zero samples is concerning
        "description": "Data quality metrics",
    },

    # Cluster health
    "cluster": {
        "min_nodes_online": 5,           # Minimum nodes required
        "node_timeout_seconds": 30,      # Consider node dead after
        "heartbeat_interval": 60,        # Expected heartbeat frequency
        "max_coordinator_lag_seconds": 300,  # Max time without coordinator update
        "description": "Cluster health requirements",
    },

    # Selfplay monitoring
    "selfplay": {
        "min_games_per_hour": 100,       # Minimum generation rate
        "max_game_duration_seconds": 600,  # 10 minute max per game
        "min_move_count": 5,             # Suspiciously short game
        "description": "Selfplay generation metrics",
    },

    # Network/P2P monitoring
    "network": {
        "ping_timeout_ms": 5000,         # 5 second ping timeout
        "max_relay_latency_ms": 200,     # Direct connection preferred
        "reconnect_interval_seconds": 30,  # Time before reconnection attempt
        "description": "Network health thresholds",
    },

    # Memory monitoring
    "memory": {
        "warning": 80,       # % RAM used
        "critical": 90,      # % RAM used
        "unit": "percent",
        "description": "System memory thresholds",
    },
}


def get_threshold(
    category: str,
    key: str,
    default: Optional[Any] = None,
) -> Any:
    """Get a specific threshold value.

    Args:
        category: Threshold category (e.g., "disk", "gpu_utilization")
        key: Specific threshold key (e.g., "warning", "critical")
        default: Default value if not found

    Returns:
        Threshold value or default

    Example:
        disk_warning = get_threshold("disk", "warning")  # Returns 70
    """
    if category not in THRESHOLDS:
        return default
    return THRESHOLDS[category].get(key, default)


def should_alert(
    category: str,
    value: float,
    level: str = "warning",
    comparison: str = "gte",  # gte, lte, gt, lt, eq
) -> bool:
    """Check if a value exceeds the threshold for alert.

    Args:
        category: Threshold category
        value: Current value to check
        level: Alert level to check against
        comparison: Comparison type (gte=>=, lte=<=, gt=>, lt=<, eq===)

    Returns:
        True if value triggers alert at specified level

    Example:
        if should_alert("disk", 75, "warning"):  # disk at 75%
            send_warning()
    """
    threshold = get_threshold(category, level)
    if threshold is None:
        return False

    comparisons = {
        "gte": lambda v, t: v >= t,
        "lte": lambda v, t: v <= t,
        "gt": lambda v, t: v > t,
        "lt": lambda v, t: v < t,
        "eq": lambda v, t: v == t,
    }

    compare_fn = comparisons.get(comparison, comparisons["gte"])
    return compare_fn(value, threshold)


def get_all_thresholds() -> Dict[str, Dict[str, Any]]:
    """Get all thresholds for display/documentation."""
    return THRESHOLDS.copy()


def update_threshold(category: str, key: str, value: Any) -> None:
    """Update a threshold value at runtime.

    Use sparingly - primarily for testing or dynamic configuration.
    """
    if category in THRESHOLDS:
        THRESHOLDS[category][key] = value
