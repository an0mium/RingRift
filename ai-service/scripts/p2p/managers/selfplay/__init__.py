"""Selfplay scheduler mixins extracted from selfplay_scheduler.py.

February 2026: P0 decomposition of the monolithic SelfplayScheduler (4,425 LOC)
into focused mixin modules for better maintainability.

Modules:
- engine_selection: Engine mode selection and diversity tracking
- job_targeting: Per-node job targeting and resource allocation
- event_handlers: Event subscription and handler methods
- priority: Priority calculation and architecture selection
"""

from .engine_selection import DiversityMetrics, EngineSelectionMixin
from .event_handlers import EventHandlersMixin
from .job_targeting import JobTargetingMixin
from .priority import ArchitectureSelectionMixin, PriorityCalculatorMixin

__all__ = [
    "ArchitectureSelectionMixin",
    "DiversityMetrics",
    "EngineSelectionMixin",
    "EventHandlersMixin",
    "JobTargetingMixin",
    "PriorityCalculatorMixin",
]
