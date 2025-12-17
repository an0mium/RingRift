#!/usr/bin/env python3
"""DEPRECATED: Use cluster_sync_coordinator.py instead.

This script has been deprecated as of 2025-12-16.
The functionality has been consolidated into cluster_sync_coordinator.py.

See docs/RESOURCE_MANAGEMENT.md for sync tool consolidation notes.
"""

import sys
import warnings

warnings.warn(
    "cluster_sync_integration.py is DEPRECATED. "
    "Use cluster_sync_coordinator.py instead. "
    "This script will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

print("\n" + "=" * 60)
print("DEPRECATED: cluster_sync_integration.py")
print("=" * 60)
print("\nThis script has been deprecated and moved to:")
print("  scripts/archive/deprecated_sync/cluster_sync_integration.py")
print("\nPlease use cluster_sync_coordinator.py instead:")
print("  python scripts/cluster_sync_coordinator.py --mode full")
print("\nFor more info, see docs/RESOURCE_MANAGEMENT.md")
print("=" * 60 + "\n")

sys.exit(1)
