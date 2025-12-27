"""P2P Orchestrator Managers.

This module contains domain-specific manager classes extracted from
p2p_orchestrator.py for better modularity and testability.

Managers:
- StateManager: SQLite persistence, cluster epoch tracking
- NodeSelector: Node ranking and selection for job dispatch
- SyncPlanner: Manifest collection and sync planning (December 2025)
- JobManager: Job spawning and lifecycle management (December 2025)
- TrainingCoordinator: Training dispatch and promotion (December 2025)
- SelfplayScheduler: Selfplay config selection and diversity (December 2025)
"""

from .job_manager import JobManager
from .node_selector import NodeSelector
from .selfplay_scheduler import DiversityMetrics, SelfplayScheduler
from .state_manager import StateManager
from .sync_planner import SyncPlanner, SyncPlannerConfig, SyncStats
from .training_coordinator import TrainingCoordinator

__all__ = [
    "DiversityMetrics",
    "JobManager",
    "NodeSelector",
    "SelfplayScheduler",
    "StateManager",
    "SyncPlanner",
    "SyncPlannerConfig",
    "SyncStats",
    "TrainingCoordinator",
]
