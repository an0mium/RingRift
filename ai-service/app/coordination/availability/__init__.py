"""Unified Cluster Availability Manager.

This submodule provides automated cluster availability management:
- NodeMonitor: Multi-layer health checking (P2P, SSH, GPU, Provider API)
- RecoveryEngine: Escalating recovery strategies
- Provisioner: Auto-provision new instances when capacity drops
- CapacityPlanner: Budget-aware capacity management

Created: Dec 28, 2025
"""

from .node_monitor import NodeMonitor, HealthCheckLayer, NodeHealthResult
from .recovery_engine import RecoveryEngine, RecoveryAction, RecoveryResult
from .provisioner import Provisioner, ProvisionResult
from .capacity_planner import CapacityPlanner, CapacityBudget, ScaleRecommendation

__all__ = [
    # NodeMonitor
    "NodeMonitor",
    "HealthCheckLayer",
    "NodeHealthResult",
    # RecoveryEngine
    "RecoveryEngine",
    "RecoveryAction",
    "RecoveryResult",
    # Provisioner
    "Provisioner",
    "ProvisionResult",
    # CapacityPlanner
    "CapacityPlanner",
    "CapacityBudget",
    "ScaleRecommendation",
]
