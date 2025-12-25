"""Training coordination modules.

This package consolidates training-related coordination:
- orchestrator: Training and selfplay orchestration
- scheduler: Job and duration scheduling

December 2025: Consolidation from 75 → 15 modules.

Usage:
    from app.coordination.training.orchestrator import TrainingCoordinator
    from app.coordination.training.scheduler import PriorityJobScheduler
"""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name in ("orchestrator", "scheduler"):
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
