"""Resource management (December 2025).

Consolidates resource management from resource_optimizer.py and adaptive_resource_manager.py.

Usage:
    from app.coordination.resources.manager import (
        ResourceOptimizer,
        AdaptiveResourceManager,
    )
"""

from __future__ import annotations

# Re-export from resource_optimizer
from app.coordination.resource_optimizer import (
    ResourceOptimizer,
    OptimizationResult,
    NodeResources,
    ClusterState,
    ResourceType,
    ScaleAction,
    PIDController,
)

# Re-export from adaptive_resource_manager
from app.coordination.adaptive_resource_manager import (
    AdaptiveResourceManager,
    ResourceStatus,
    CleanupResult,
)

# Re-export from resource_targets
from app.coordination.resource_targets import (
    ResourceTargetManager,
    HostTargets,
    get_resource_targets,
)

__all__ = [
    # From resource_optimizer
    "ResourceOptimizer",
    "OptimizationResult",
    "NodeResources",
    "ClusterState",
    "ResourceType",
    "ScaleAction",
    "PIDController",
    # From adaptive_resource_manager
    "AdaptiveResourceManager",
    "ResourceStatus",
    "CleanupResult",
    # From resource_targets
    "ResourceTargetManager",
    "HostTargets",
    "get_resource_targets",
]
