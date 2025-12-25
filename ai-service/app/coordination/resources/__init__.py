"""Resource coordination modules.

This package consolidates resource-related coordination:
- manager: Resource optimization and adaptive management
- bandwidth: Bandwidth coordination
- thresholds: Dynamic resource thresholds

December 2025: Consolidation from 75 → 15 modules.

Usage:
    from app.coordination.resources.manager import ResourceOptimizer
    from app.coordination.resources.bandwidth import BandwidthManager
"""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name in ("manager", "bandwidth", "thresholds"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
