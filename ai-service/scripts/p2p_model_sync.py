#!/usr/bin/env python3
"""DEPRECATED: Use aria2_data_sync.py instead.

This script has been deprecated as of 2025-12-16.
The aria2-based P2P functionality has been consolidated into aria2_data_sync.py.

See docs/RESOURCE_MANAGEMENT.md for sync tool consolidation notes.
"""

import sys
import warnings

warnings.warn(
    "p2p_model_sync.py is DEPRECATED. "
    "Use aria2_data_sync.py instead. "
    "This script will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

print("\n" + "=" * 60)
print("DEPRECATED: p2p_model_sync.py")
print("=" * 60)
print("\nThis script has been deprecated and moved to:")
print("  scripts/archive/deprecated_sync/p2p_model_sync.py")
print("\nPlease use aria2_data_sync.py instead:")
print("  python scripts/aria2_data_sync.py --mode models")
print("\nFor more info, see docs/RESOURCE_MANAGEMENT.md")
print("=" * 60 + "\n")

sys.exit(1)
