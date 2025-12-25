"""Bandwidth management (December 2025).

Re-exports from bandwidth_manager.py for unified access.

Usage:
    from app.coordination.resources.bandwidth import (
        BandwidthManager,
        get_bandwidth_manager,
    )
"""

from app.coordination.bandwidth_manager import (
    BandwidthManager,
    get_bandwidth_manager,
    BandwidthLimit,
    BandwidthPolicy,
    set_bandwidth_limit,
    get_current_bandwidth,
)

__all__ = [
    "BandwidthManager",
    "get_bandwidth_manager",
    "BandwidthLimit",
    "BandwidthPolicy",
    "set_bandwidth_limit",
    "get_current_bandwidth",
]
