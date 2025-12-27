"""Coordination Mixins - Reusable behavior components.

This package provides mixins that can be used to add common functionality
to coordinators, daemons, and managers without requiring inheritance from
a specific base class.

December 2025: Created as part of Phase 2 consolidation to reduce code
duplication across 76+ files implementing similar patterns.

Available mixins:
- HealthCheckMixin: Standard health check implementation (~600 LOC savings)
"""

from app.coordination.mixins.health_check_mixin import (
    HealthCheckMixin,
)

__all__ = [
    "HealthCheckMixin",
]
