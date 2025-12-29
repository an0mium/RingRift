"""Bandwidth Manager - DEPRECATED re-export module.

.. deprecated:: December 2025
    This module is deprecated and only exists for backward compatibility.
    All functionality has been moved to ``app.coordination.sync_bandwidth``.

    Migration:

    .. code-block:: python

        # Old (deprecated)
        from app.coordination.bandwidth_manager import get_bandwidth_manager

        # New (canonical)
        from app.coordination.sync_bandwidth import get_bandwidth_manager

    This module will be archived in Q2 2026.

Original purpose: Network bandwidth allocation for data sync operations
with per-host tracking, priority scheduling, and rate limiting.
"""

from __future__ import annotations

import warnings

# Emit deprecation warning on import
warnings.warn(
    "app.coordination.bandwidth_manager is deprecated. "
    "Use app.coordination.sync_bandwidth instead. "
    "This module will be archived in Q2 2026.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical location for backward compatibility
from app.coordination.sync_bandwidth import (
    BandwidthAllocation,
    BandwidthConfig,
    BandwidthCoordinatedRsync,
    BandwidthManager,
    TransferPriority,
    bandwidth_allocation,
    get_bandwidth_manager,
    get_bandwidth_stats,
    get_host_bandwidth_status,
    get_optimal_transfer_time,
    release_bandwidth,
    request_bandwidth,
    reset_bandwidth_manager,
)

__all__ = [
    "BandwidthAllocation",
    "BandwidthConfig",
    "BandwidthCoordinatedRsync",
    "BandwidthManager",
    "TransferPriority",
    "bandwidth_allocation",
    "get_bandwidth_manager",
    "get_bandwidth_stats",
    "get_host_bandwidth_status",
    "get_optimal_transfer_time",
    "release_bandwidth",
    "request_bandwidth",
    "reset_bandwidth_manager",
]
