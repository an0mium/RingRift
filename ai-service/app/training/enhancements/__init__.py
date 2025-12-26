"""
Training Enhancements Subpackage for RingRift AI.

This subpackage provides modularized training enhancement components:
- Training configuration
- Gradient management (accumulation and adaptive clipping)
- Checkpoint averaging
- Data quality scoring
- Hard example mining
- Learning rate schedulers
- Early stopping
- EWC regularization
- Model ensembles
- Anomaly detection
- Validation management
- Seed management

Extracted from training_enhancements.py (December 2025 modularization).
"""

from __future__ import annotations

# Phase 1 exports: Training config, gradient management, checkpoint averaging
from app.training.enhancements.checkpoint_averaging import (
    CheckpointAverager,
    average_checkpoints,
)
from app.training.enhancements.gradient_management import (
    AdaptiveGradientClipper,
    GradientAccumulator,
)
from app.training.enhancements.training_config import TrainingConfig

# Phase 2 exports: Learning rate scheduling, seed management, calibration
from app.training.enhancements.calibration import CalibrationAutomation
from app.training.enhancements.learning_rate_scheduling import (
    AdaptiveLRScheduler,
    WarmRestartsScheduler,
)
from app.training.enhancements.seed_management import (
    SeedManager,
    set_reproducible_seed,
)

# Phase 3 exports: EWC, Model Ensemble, Evaluation Feedback (December 2025)
from app.training.enhancements.evaluation_feedback import (
    EvaluationFeedbackHandler,
    create_evaluation_feedback_handler,
)
from app.training.enhancements.ewc_regularization import EWCRegularizer
from app.training.enhancements.model_ensemble import ModelEnsemble

__all__ = [
    # Training configuration
    "TrainingConfig",
    # Gradient management
    "GradientAccumulator",
    "AdaptiveGradientClipper",
    # Checkpoint averaging
    "CheckpointAverager",
    "average_checkpoints",
    # Learning rate scheduling
    "AdaptiveLRScheduler",
    "WarmRestartsScheduler",
    # Seed management
    "SeedManager",
    "set_reproducible_seed",
    # Calibration
    "CalibrationAutomation",
    # EWC Regularization (Phase 3 - December 2025)
    "EWCRegularizer",
    # Model Ensemble (Phase 3 - December 2025)
    "ModelEnsemble",
    # Evaluation Feedback (Phase 3 - December 2025)
    "EvaluationFeedbackHandler",
    "create_evaluation_feedback_handler",
]
