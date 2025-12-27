"""Resource coordination modules.

This package consolidates resource-related coordination:
- manager: Resource optimization and adaptive management
- bandwidth: Bandwidth coordination (re-exports from bandwidth_manager)
- thresholds: Dynamic resource thresholds (re-exports from dynamic_thresholds)

December 2025: Consolidation from 75 → 15 modules.

Usage:
    # Direct imports (preferred)
    from app.coordination.bandwidth_manager import BandwidthManager
    from app.coordination.dynamic_thresholds import DynamicThresholds

    # Or via package (deprecated wrappers)
    from app.coordination.resources.bandwidth import BandwidthManager
    from app.coordination.resources.thresholds import DynamicThresholds
"""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name in ("manager", "bandwidth", "thresholds"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Re-export common symbols for backward compatibility
from app.coordination.bandwidth_manager import BandwidthManager, get_bandwidth_manager
from app.coordination.dynamic_thresholds import DynamicThreshold, get_threshold_manager

__all__ = [
    "BandwidthManager",
    "get_bandwidth_manager",
    "DynamicThreshold",
    "get_threshold_manager",
]
