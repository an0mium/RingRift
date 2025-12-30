"""Feature Registry for Training Module Optional Dependencies.

This module provides a centralized registry for checking availability of optional
training dependencies. It replaces scattered try/except ImportError blocks with
a unified interface.

December 2025: Created for Phase 6A of train.py refactoring.

Usage:
    from app.training.feature_registry import has_feature, get_feature

    # Check if feature is available
    if has_feature('data_validation'):
        validator = get_feature('data_validation', 'DataValidator')
        ...

    # Or use require_feature for clearer intent
    validator_class = require_feature('data_validation', 'DataValidator')
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class FeatureSpec:
    """Specification for an optional feature.

    Attributes:
        name: Feature name (e.g., 'data_validation')
        module_path: Import path (e.g., 'app.training.unified_data_validator')
        exports: List of names to import from the module
        description: Human-readable description
    """

    name: str
    module_path: str
    exports: list[str]
    description: str = ""


@dataclass
class FeatureStatus:
    """Runtime status of an optional feature.

    Attributes:
        available: Whether the feature is importable
        exports: Dict mapping export names to their values (if available)
        error: Error message if import failed
    """

    available: bool
    exports: dict[str, Any] = field(default_factory=dict)
    error: str = ""


# Registry of all optional features used by train.py
# Each entry specifies the module path and what to import from it
FEATURE_SPECS: dict[str, FeatureSpec] = {
    # Data validation features
    "data_validation": FeatureSpec(
        name="data_validation",
        module_path="app.training.unified_data_validator",
        exports=["DataValidator", "DataValidatorConfig", "validate_npz_file"],
        description="Unified data validation for NPZ files",
    ),
    "checksum_verification": FeatureSpec(
        name="checksum_verification",
        module_path="app.training.data_quality",
        exports=["verify_npz_checksums"],
        description="Checksum verification for data integrity",
    ),
    "npz_structure_validation": FeatureSpec(
        name="npz_structure_validation",
        module_path="app.coordination.npz_validation",
        exports=["validate_npz_structure", "NPZValidationResult"],
        description="NPZ structure validation for corruption detection",
    ),
    "train_validation": FeatureSpec(
        name="train_validation",
        module_path="app.training.train_validation",
        exports=[
            "validate_training_data_freshness",
            "validate_training_data_files",
            "validate_data_checksums",
            "FreshnessResult",
        ],
        description="Extracted training validation utilities",
    ),
    # Data handling features
    "hot_data_buffer": FeatureSpec(
        name="hot_data_buffer",
        module_path="app.training.hot_data_buffer",
        exports=["HotDataBuffer"],
        description="Hot data buffer for priority experience replay",
    ),
    "quality_bridge": FeatureSpec(
        name="quality_bridge",
        module_path="app.training.quality_bridge",
        exports=["QualityBridge", "get_quality_bridge"],
        description="Quality-aware data selection",
    ),
    "data_catalog": FeatureSpec(
        name="data_catalog",
        module_path="app.distributed.data_catalog",
        exports=["DataCatalog", "get_data_catalog"],
        description="Cluster-wide training data discovery",
    ),
    # Training enhancement features
    "integrated_enhancements": FeatureSpec(
        name="integrated_enhancements",
        module_path="app.training.integrated_enhancements",
        exports=["IntegratedEnhancementsConfig", "IntegratedTrainingManager"],
        description="Integrated training enhancements",
    ),
    "training_enhancements": FeatureSpec(
        name="training_enhancements",
        module_path="app.training.training_enhancements",
        exports=[
            "AdaptiveGradientClipper",
            "CheckpointAverager",
            "EvaluationFeedbackHandler",
            "TrainingAnomalyDetector",
        ],
        description="Training anomaly detection and enhancements",
    ),
    "hard_example_mining": FeatureSpec(
        name="hard_example_mining",
        module_path="app.training.enhancements.hard_example_mining",
        exports=["HardExampleMiner"],
        description="Hard example mining for curriculum learning",
    ),
    "per_sample_loss": FeatureSpec(
        name="per_sample_loss",
        module_path="app.training.enhancements.per_sample_loss",
        exports=["compute_per_sample_loss"],
        description="Per-sample loss computation",
    ),
    "training_facade": FeatureSpec(
        name="training_facade",
        module_path="app.training.enhancements.training_facade",
        exports=["FacadeConfig", "TrainingEnhancementsFacade"],
        description="Unified training enhancements facade",
    ),
    "quality_weighting": FeatureSpec(
        name="quality_weighting",
        module_path="app.training.quality_weighted_loss",
        exports=[
            "QualityWeightedTrainer",
            "compute_quality_weights",
            "quality_weighted_policy_loss",
            "ranking_loss_from_quality",
        ],
        description="Quality-weighted training loss",
    ),
    # Event and monitoring features
    "circuit_breaker": FeatureSpec(
        name="circuit_breaker",
        module_path="app.distributed.circuit_breaker",
        exports=["CircuitState", "get_training_breaker"],
        description="Circuit breaker for training fault tolerance",
    ),
    "event_bus": FeatureSpec(
        name="event_bus",
        module_path="app.coordination.event_router",
        exports=["get_router", "DataEvent", "DataEventType"],
        description="Event bus for training events",
    ),
    "training_events": FeatureSpec(
        name="training_events",
        module_path="app.coordination.event_router",
        exports=["emit_training_loss_anomaly", "emit_training_loss_trend"],
        description="Training event emission",
    ),
    "epoch_events": FeatureSpec(
        name="epoch_events",
        module_path="app.training.event_integration",
        exports=["publish_epoch_completed"],
        description="Epoch event emission for curriculum feedback",
    ),
    "regression_detector": FeatureSpec(
        name="regression_detector",
        module_path="app.training.regression_detector",
        exports=["RegressionSeverity", "get_regression_detector"],
        description="Regression detection for training quality",
    ),
    # Training coordination features
    "freshness_check": FeatureSpec(
        name="freshness_check",
        module_path="app.coordination.training_freshness",
        exports=["check_freshness_sync", "FreshnessConfig", "FreshnessResult"],
        description="Training data freshness checking",
    ),
    "stale_fallback": FeatureSpec(
        name="stale_fallback",
        module_path="app.coordination.stale_fallback",
        exports=["get_training_fallback_controller", "should_allow_stale_training"],
        description="Stale training fallback for autonomous operation",
    ),
}


class FeatureRegistry:
    """Registry for checking and accessing optional training features.

    This class provides a centralized way to:
    1. Check if optional features are available (importable)
    2. Access the imported modules/classes lazily
    3. Track which features are used for diagnostics

    Thread-safe singleton pattern.
    """

    _instance: "FeatureRegistry | None" = None
    _lock_initialized: bool = False

    def __new__(cls) -> "FeatureRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._status_cache: dict[str, FeatureStatus] = {}
        self._specs = FEATURE_SPECS.copy()

    def _check_feature(self, feature_name: str) -> FeatureStatus:
        """Check if a feature is available and cache the result."""
        if feature_name in self._status_cache:
            return self._status_cache[feature_name]

        spec = self._specs.get(feature_name)
        if spec is None:
            status = FeatureStatus(
                available=False,
                error=f"Unknown feature: {feature_name}",
            )
            self._status_cache[feature_name] = status
            return status

        try:
            # Import the module
            import importlib

            module = importlib.import_module(spec.module_path)

            # Get all exports
            exports: dict[str, Any] = {}
            for export_name in spec.exports:
                if hasattr(module, export_name):
                    exports[export_name] = getattr(module, export_name)
                else:
                    # Some exports might not exist in older versions
                    logger.debug(
                        f"Feature {feature_name}: export {export_name} not found"
                    )

            status = FeatureStatus(available=True, exports=exports)
        except ImportError as e:
            status = FeatureStatus(available=False, error=str(e))
        except Exception as e:
            status = FeatureStatus(available=False, error=f"Unexpected: {e}")

        self._status_cache[feature_name] = status
        return status

    def has_feature(self, feature_name: str) -> bool:
        """Check if a feature is available.

        Args:
            feature_name: Name of the feature to check.

        Returns:
            True if the feature is importable, False otherwise.
        """
        return self._check_feature(feature_name).available

    def get_feature(self, feature_name: str, export_name: str) -> Any | None:
        """Get an export from a feature.

        Args:
            feature_name: Name of the feature.
            export_name: Name of the export to retrieve.

        Returns:
            The exported value, or None if not available.
        """
        status = self._check_feature(feature_name)
        if not status.available:
            return None
        return status.exports.get(export_name)

    def require_feature(self, feature_name: str, export_name: str) -> Any:
        """Get an export from a feature, raising if not available.

        Args:
            feature_name: Name of the feature.
            export_name: Name of the export to retrieve.

        Returns:
            The exported value.

        Raises:
            ImportError: If the feature is not available.
        """
        status = self._check_feature(feature_name)
        if not status.available:
            raise ImportError(
                f"Feature {feature_name} not available: {status.error}"
            )
        if export_name not in status.exports:
            raise ImportError(
                f"Export {export_name} not found in feature {feature_name}"
            )
        return status.exports[export_name]

    def get_all_exports(self, feature_name: str) -> dict[str, Any]:
        """Get all exports from a feature.

        Args:
            feature_name: Name of the feature.

        Returns:
            Dict mapping export names to values, or empty dict if not available.
        """
        status = self._check_feature(feature_name)
        return status.exports.copy() if status.available else {}

    def get_status_report(self) -> dict[str, dict[str, Any]]:
        """Get status of all registered features.

        Returns:
            Dict mapping feature names to their status info.
        """
        report: dict[str, dict[str, Any]] = {}
        for feature_name in self._specs:
            status = self._check_feature(feature_name)
            report[feature_name] = {
                "available": status.available,
                "exports": list(status.exports.keys()) if status.available else [],
                "error": status.error if not status.available else None,
                "description": self._specs[feature_name].description,
            }
        return report

    def clear_cache(self) -> None:
        """Clear the status cache (useful for testing)."""
        self._status_cache.clear()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None


# Module-level convenience functions
_registry: FeatureRegistry | None = None


def _get_registry() -> FeatureRegistry:
    """Get or create the feature registry singleton."""
    global _registry
    if _registry is None:
        _registry = FeatureRegistry()
    return _registry


def has_feature(feature_name: str) -> bool:
    """Check if a feature is available.

    Args:
        feature_name: Name of the feature to check.

    Returns:
        True if the feature is importable, False otherwise.

    Example:
        >>> if has_feature('data_validation'):
        ...     # Use data validation
        ...     pass
    """
    return _get_registry().has_feature(feature_name)


def get_feature(feature_name: str, export_name: str) -> Any | None:
    """Get an export from a feature.

    Args:
        feature_name: Name of the feature.
        export_name: Name of the export to retrieve.

    Returns:
        The exported value, or None if not available.

    Example:
        >>> DataValidator = get_feature('data_validation', 'DataValidator')
        >>> if DataValidator:
        ...     validator = DataValidator()
    """
    return _get_registry().get_feature(feature_name, export_name)


def require_feature(feature_name: str, export_name: str) -> Any:
    """Get an export from a feature, raising if not available.

    Args:
        feature_name: Name of the feature.
        export_name: Name of the export to retrieve.

    Returns:
        The exported value.

    Raises:
        ImportError: If the feature is not available.

    Example:
        >>> try:
        ...     DataValidator = require_feature('data_validation', 'DataValidator')
        ...     validator = DataValidator()
        ... except ImportError:
        ...     logger.warning("Data validation not available")
    """
    return _get_registry().require_feature(feature_name, export_name)


def get_feature_status() -> dict[str, dict[str, Any]]:
    """Get status of all registered features.

    Returns:
        Dict mapping feature names to their status info.

    Example:
        >>> status = get_feature_status()
        >>> for name, info in status.items():
        ...     print(f"{name}: {'available' if info['available'] else 'missing'}")
    """
    return _get_registry().get_status_report()


def reset_feature_registry() -> None:
    """Reset the feature registry (for testing)."""
    global _registry
    _registry = None
    FeatureRegistry.reset_instance()


# Backward-compatible HAS_* constants
# These allow gradual migration - old code using HAS_DATA_VALIDATION still works
# New code should use has_feature('data_validation') instead
def __getattr__(name: str) -> Any:
    """Provide backward-compatible HAS_* constants.

    This allows code like:
        from app.training.feature_registry import HAS_DATA_VALIDATION

    To work while we migrate to:
        from app.training.feature_registry import has_feature
        if has_feature('data_validation'):
            ...
    """
    if name.startswith("HAS_"):
        # Convert HAS_DATA_VALIDATION to data_validation
        feature_name = name[4:].lower()
        return has_feature(feature_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
