"""DEPRECATED: Use app.coordination.bandwidth_manager instead.

This module is a wrapper for backward compatibility.
Import directly from app.coordination.bandwidth_manager:

    from app.coordination.bandwidth_manager import (
        BandwidthManager,
        get_bandwidth_manager,
    )

Scheduled for removal: Q2 2026
"""
import warnings

# Issue deprecation warning on import
warnings.warn(
    "app.coordination.resources.bandwidth is deprecated. "
    "Use 'from app.coordination.bandwidth_manager import ...' instead. "
    "This module will be removed in Q2 2026.",
    DeprecationWarning,
    stacklevel=2,
)

from app.coordination.bandwidth_manager import (
    BandwidthManager,
    BandwidthAllocation,
    TransferPriority,
    get_bandwidth_manager,
    reset_bandwidth_manager,
    request_bandwidth,
    release_bandwidth,
    get_host_bandwidth_status,
    get_bandwidth_stats,
)

__all__ = [
    "BandwidthManager",
    "BandwidthAllocation",
    "TransferPriority",
    "get_bandwidth_manager",
    "reset_bandwidth_manager",
    "request_bandwidth",
    "release_bandwidth",
    "get_host_bandwidth_status",
    "get_bandwidth_stats",
]
