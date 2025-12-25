"""Training pipeline orchestration (December 2025).

Consolidates pipeline-related functionality from:
- data_pipeline_orchestrator.py (pipeline orchestration)
- pipeline_actions.py (stage action invokers)

This module re-exports all pipeline-related APIs for unified access.

Usage:
    from app.coordination.core.pipeline import (
        DataPipelineOrchestrator,
        get_pipeline_orchestrator,
        PipelineStage,
    )

    # Trigger a training pipeline
    orchestrator = get_pipeline_orchestrator()
    orchestrator.advance_stage(PipelineStage.EXPORT)
"""

from __future__ import annotations

# Re-export from data_pipeline_orchestrator
from app.coordination.data_pipeline_orchestrator import (
    DataPipelineOrchestrator,
    get_pipeline_orchestrator,
    get_pipeline_status,
    get_current_pipeline_stage,
    PipelineStage,
    PipelineStats,
    IterationRecord,
)

# Re-export from pipeline_actions
from app.coordination.pipeline_actions import (
    trigger_npz_export,
    trigger_evaluation,
    trigger_data_sync,
    ActionConfig,
    ActionPriority,
    StageCompletionResult,
)

__all__ = [
    # From data_pipeline_orchestrator
    "DataPipelineOrchestrator",
    "get_pipeline_orchestrator",
    "get_pipeline_status",
    "get_current_pipeline_stage",
    "PipelineStage",
    "PipelineStats",
    "IterationRecord",
    # From pipeline_actions
    "trigger_npz_export",
    "trigger_evaluation",
    "trigger_data_sync",
    "ActionConfig",
    "ActionPriority",
    "StageCompletionResult",
]
