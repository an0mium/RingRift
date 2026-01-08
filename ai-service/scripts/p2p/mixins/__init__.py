"""P2P Mixins Package.

January 2026 - Sprint 17: Shared mixin classes for P2P orchestrator components.

Mixins:
- HealthTrackingMixin: Per-entity failure tracking with backoff and health scoring

Usage:
    from scripts.p2p.mixins import HealthTrackingMixin, HealthTrackingConfig

    class MyLoop(BaseLoop, HealthTrackingMixin):
        def __init__(self):
            super().__init__()
            self.init_health_tracking(HealthTrackingConfig(failure_threshold=5))
"""

from .health_tracking import (
    HealthTrackingMixin,
    HealthTrackingConfig,
    EntityHealthSummary,
    EntityHealthState,
)

__all__ = [
    "HealthTrackingMixin",
    "HealthTrackingConfig",
    "EntityHealthSummary",
    "EntityHealthState",
]
