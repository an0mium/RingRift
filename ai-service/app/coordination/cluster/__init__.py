"""Cluster coordination modules.

This package consolidates cluster-related coordination:
- health: Node and host health monitoring
- sync: Data synchronization
- transport: Cluster transport layer
- p2p: Peer-to-peer backend

December 2025: Consolidation from 75 → 15 modules.

Usage:
    from app.coordination.cluster.health import UnifiedHealthManager
    from app.coordination.cluster.sync import SyncScheduler
"""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name in ("health", "sync", "transport", "p2p"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
