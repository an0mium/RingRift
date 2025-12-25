"""Training orchestration (December 2025).

Consolidates training orchestration from:
- training_coordinator.py (training lifecycle)
- selfplay_orchestrator.py (selfplay coordination)

This module re-exports all training orchestration APIs.

Usage:
    from app.coordination.training.orchestrator import (
        TrainingCoordinator,
        get_training_coordinator,
        SelfplayOrchestrator,
        is_large_board,
    )
"""

from __future__ import annotations

# Re-export from training_coordinator
from app.coordination.training_coordinator import (
    TrainingCoordinator,
    get_training_coordinator,
    get_training_status,
    TrainingJob,
    can_train,
    request_training_slot,
    release_training_slot,
    wire_training_events,
)

# Re-export from selfplay_orchestrator
from app.coordination.selfplay_orchestrator import (
    SelfplayOrchestrator,
    get_selfplay_orchestrator,
    get_selfplay_stats,
    is_large_board,
    get_engine_for_board,
    get_simulation_budget_for_board,
    SelfplayStats,
    SelfplayType,
    wire_selfplay_events,
)

__all__ = [
    # From training_coordinator
    "TrainingCoordinator",
    "get_training_coordinator",
    "get_training_status",
    "TrainingJob",
    "can_train",
    "request_training_slot",
    "release_training_slot",
    "wire_training_events",
    # From selfplay_orchestrator
    "SelfplayOrchestrator",
    "get_selfplay_orchestrator",
    "get_selfplay_stats",
    "is_large_board",
    "get_engine_for_board",
    "get_simulation_budget_for_board",
    "SelfplayStats",
    "SelfplayType",
    "wire_selfplay_events",
]
