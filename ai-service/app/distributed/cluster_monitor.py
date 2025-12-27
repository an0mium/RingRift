"""Backward-compatibility shim for cluster_monitor.

.. deprecated:: December 2025
    This module has been relocated to :mod:`app.coordination.cluster_status_monitor`
    for better organization. Update imports accordingly:

    # OLD (deprecated)
    from app.distributed.cluster_monitor import ClusterMonitor

    # NEW (recommended)
    from app.coordination.cluster_status_monitor import ClusterMonitor

This shim re-exports all public names from the new location.
"""

import warnings

warnings.warn(
    "app.distributed.cluster_monitor has been relocated to "
    "app.coordination.cluster_status_monitor for better organization. "
    "Update imports accordingly. This shim will be removed in Q2 2026.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all public names from the new location
from app.coordination.cluster_status_monitor import (
    ClusterMonitor,
    ClusterStatus,
    NodeStatus,
    main,
)

__all__ = [
    "ClusterMonitor",
    "ClusterStatus",
    "NodeStatus",
    "main",
]

# Support CLI usage
if __name__ == "__main__":
    main()
