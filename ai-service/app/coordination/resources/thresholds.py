"""DEPRECATED: Use app.coordination.dynamic_thresholds instead.

This module is a wrapper for backward compatibility.
Import directly from app.coordination.dynamic_thresholds:

    from app.coordination.dynamic_thresholds import (
        DynamicThreshold,
        get_threshold_manager,
    )

Scheduled for removal: Q2 2026
"""
import warnings

# Issue deprecation warning on import
warnings.warn(
    "app.coordination.resources.thresholds is deprecated. "
    "Use 'from app.coordination.dynamic_thresholds import ...' instead. "
    "This module will be removed in Q2 2026.",
    DeprecationWarning,
    stacklevel=2,
)

from app.coordination.dynamic_thresholds import (
    DynamicThreshold,
    ThresholdManager,
    get_threshold_manager,
    reset_threshold_manager,
    AdjustmentStrategy,
)

__all__ = [
    "DynamicThreshold",
    "ThresholdManager",
    "get_threshold_manager",
    "reset_threshold_manager",
    "AdjustmentStrategy",
]
