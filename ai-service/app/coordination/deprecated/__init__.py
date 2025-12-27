"""Deprecated coordination modules.

These modules have been superseded by newer implementations.
See README.md in this directory for migration guide.

Moved to deprecated/ December 2025:
- _deprecated_cross_process_events.py -> Use event_router instead
- _deprecated_event_emitters.py -> Use event_router.emit() instead
- _deprecated_health_check_orchestrator.py -> Use cluster.health instead
- _deprecated_host_health_policy.py -> Use cluster.health instead
- _deprecated_system_health_monitor.py -> Use cluster.health instead
"""

import warnings

# Backwards-compatible imports with deprecation warnings
def __getattr__(name: str):
    """Lazy import with deprecation warning."""
    deprecated_modules = {
        "cross_process_events": "_deprecated_cross_process_events",
        "event_emitters": "_deprecated_event_emitters",
        "health_check_orchestrator": "_deprecated_health_check_orchestrator",
        "host_health_policy": "_deprecated_host_health_policy",
        "system_health_monitor": "_deprecated_system_health_monitor",
    }

    if name in deprecated_modules:
        warnings.warn(
            f"app.coordination.deprecated.{name} is deprecated. "
            f"See app/coordination/deprecated/README.md for migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )
        import importlib
        return importlib.import_module(f".{deprecated_modules[name]}", __package__)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
