"""Unified Event Type Definitions - Single source of truth for all RingRift events.

This module consolidates all event type enums from:
- app.distributed.data_events.DataEventType (100+ types)
- app.coordination.stage_events.StageEvent (22 types)
- Cross-process event patterns

December 2025: Created for Phase 2 consolidation to eliminate duplicate definitions.

Usage:
    from app.events.types import RingRiftEventType, EventCategory

    # Use unified event types
    event_type = RingRiftEventType.TRAINING_COMPLETED

    # Check event category
    category = EventCategory.from_event(event_type)

    # For backwards compatibility, aliases are provided:
    from app.events.types import DataEventType, StageEvent
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class EventCategory(Enum):
    """Categories for organizing events."""

    DATA = "data"  # Data collection, sync, freshness
    TRAINING = "training"  # Training lifecycle
    EVALUATION = "evaluation"  # Model evaluation
    PROMOTION = "promotion"  # Model promotion
    CURRICULUM = "curriculum"  # Curriculum management
    SELFPLAY = "selfplay"  # Selfplay operations
    OPTIMIZATION = "optimization"  # CMA-ES, NAS, PBT
    QUALITY = "quality"  # Data quality
    REGRESSION = "regression"  # Regression detection
    CLUSTER = "cluster"  # Cluster/P2P operations
    SYSTEM = "system"  # Daemons, health, resources
    WORK = "work"  # Work queue
    STAGE = "stage"  # Pipeline stage completion

    @classmethod
    def from_event(cls, event_type: RingRiftEventType) -> EventCategory:
        """Get the category for an event type."""
        return _EVENT_CATEGORIES.get(event_type, EventCategory.SYSTEM)


class RingRiftEventType(Enum):
    """Unified event types for all RingRift systems.

    This enum consolidates all event types from data_events.DataEventType
    and stage_events.StageEvent into a single source of truth.
    """

    # =========================================================================
    # DATA COLLECTION EVENTS
    # =========================================================================
    NEW_GAMES_AVAILABLE = "new_games"
    DATA_SYNC_STARTED = "sync_started"
    DATA_SYNC_COMPLETED = "sync_completed"
    DATA_SYNC_FAILED = "sync_failed"
    GAME_SYNCED = "game_synced"

    # =========================================================================
    # DATA FRESHNESS EVENTS
    # =========================================================================
    DATA_STALE = "data_stale"
    DATA_FRESH = "data_fresh"
    SYNC_TRIGGERED = "sync_triggered"
    SYNC_STALLED = "sync_stalled"

    # =========================================================================
    # TRAINING EVENTS
    # =========================================================================
    TRAINING_THRESHOLD_REACHED = "training_threshold"
    TRAINING_STARTED = "training_started"
    TRAINING_PROGRESS = "training_progress"
    TRAINING_COMPLETED = "training_completed"
    TRAINING_FAILED = "training_failed"
    TRAINING_LOSS_ANOMALY = "training_loss_anomaly"
    TRAINING_LOSS_TREND = "training_loss_trend"
    TRAINING_EARLY_STOPPED = "training_early_stopped"
    TRAINING_ROLLBACK_NEEDED = "training_rollback_needed"
    TRAINING_ROLLBACK_COMPLETED = "training_rollback_completed"

    # =========================================================================
    # EVALUATION EVENTS
    # =========================================================================
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_PROGRESS = "evaluation_progress"
    EVALUATION_COMPLETED = "evaluation_completed"
    EVALUATION_FAILED = "evaluation_failed"
    ELO_UPDATED = "elo_updated"
    ELO_SIGNIFICANT_CHANGE = "elo_significant_change"
    ELO_VELOCITY_CHANGED = "elo_velocity_changed"
    ADAPTIVE_PARAMS_CHANGED = "adaptive_params_changed"

    # =========================================================================
    # PROMOTION EVENTS
    # =========================================================================
    PROMOTION_CANDIDATE = "promotion_candidate"
    PROMOTION_STARTED = "promotion_started"
    MODEL_PROMOTED = "model_promoted"
    PROMOTION_FAILED = "promotion_failed"
    PROMOTION_REJECTED = "promotion_rejected"
    PROMOTION_ROLLED_BACK = "promotion_rolled_back"

    # =========================================================================
    # CURRICULUM EVENTS
    # =========================================================================
    CURRICULUM_REBALANCED = "curriculum_rebalanced"
    CURRICULUM_ADVANCED = "curriculum_advanced"
    WEIGHT_UPDATED = "weight_updated"

    # =========================================================================
    # SELFPLAY EVENTS
    # =========================================================================
    SELFPLAY_TARGET_UPDATED = "selfplay_target_updated"
    SELFPLAY_RATE_CHANGED = "selfplay_rate_changed"
    IDLE_RESOURCE_DETECTED = "idle_resource_detected"

    # =========================================================================
    # OPTIMIZATION EVENTS (CMA-ES, NAS, PBT)
    # =========================================================================
    CMAES_TRIGGERED = "cmaes_triggered"
    CMAES_COMPLETED = "cmaes_completed"
    NAS_TRIGGERED = "nas_triggered"
    NAS_STARTED = "nas_started"
    NAS_GENERATION_COMPLETE = "nas_generation_complete"
    NAS_COMPLETED = "nas_completed"
    NAS_BEST_ARCHITECTURE = "nas_best_architecture"
    PBT_STARTED = "pbt_started"
    PBT_GENERATION_COMPLETE = "pbt_generation_complete"
    PBT_COMPLETED = "pbt_completed"
    PLATEAU_DETECTED = "plateau_detected"
    HYPERPARAMETER_UPDATED = "hyperparameter_updated"

    # =========================================================================
    # PRIORITIZED EXPERIENCE REPLAY EVENTS
    # =========================================================================
    PER_BUFFER_REBUILT = "per_buffer_rebuilt"
    PER_PRIORITIES_UPDATED = "per_priorities_updated"

    # =========================================================================
    # TIER/GATING EVENTS
    # =========================================================================
    TIER_PROMOTION = "tier_promotion"
    CROSSBOARD_PROMOTION = "crossboard_promotion"

    # =========================================================================
    # PARITY VALIDATION EVENTS
    # =========================================================================
    PARITY_VALIDATION_STARTED = "parity_validation_started"
    PARITY_VALIDATION_COMPLETED = "parity_validation_completed"
    PARITY_FAILURE_RATE_CHANGED = "parity_failure_rate_changed"

    # =========================================================================
    # DATA QUALITY EVENTS
    # =========================================================================
    DATA_QUALITY_ALERT = "data_quality_alert"
    QUALITY_CHECK_FAILED = "quality_check_failed"
    QUALITY_SCORE_UPDATED = "quality_score_updated"
    QUALITY_DISTRIBUTION_CHANGED = "quality_distribution_changed"
    HIGH_QUALITY_DATA_AVAILABLE = "high_quality_data_available"
    QUALITY_DEGRADED = "quality_degraded"
    LOW_QUALITY_DATA_WARNING = "low_quality_data_warning"
    TRAINING_BLOCKED_BY_QUALITY = "training_blocked_by_quality"
    QUALITY_PENALTY_APPLIED = "quality_penalty_applied"

    # =========================================================================
    # REGISTRY & METRICS EVENTS
    # =========================================================================
    REGISTRY_UPDATED = "registry_updated"
    METRICS_UPDATED = "metrics_updated"
    CACHE_INVALIDATED = "cache_invalidated"

    # =========================================================================
    # REGRESSION DETECTION EVENTS
    # =========================================================================
    REGRESSION_DETECTED = "regression_detected"
    REGRESSION_MINOR = "regression_minor"
    REGRESSION_MODERATE = "regression_moderate"
    REGRESSION_SEVERE = "regression_severe"
    REGRESSION_CRITICAL = "regression_critical"
    REGRESSION_CLEARED = "regression_cleared"

    # =========================================================================
    # P2P/CLUSTER EVENTS
    # =========================================================================
    P2P_MODEL_SYNCED = "p2p_model_synced"
    P2P_CLUSTER_HEALTHY = "p2p_cluster_healthy"
    P2P_CLUSTER_UNHEALTHY = "p2p_cluster_unhealthy"
    P2P_NODES_DEAD = "p2p_nodes_dead"
    P2P_SELFPLAY_SCALED = "p2p_selfplay_scaled"
    CLUSTER_STATUS_CHANGED = "cluster_status_changed"
    CLUSTER_CAPACITY_CHANGED = "cluster_capacity_changed"
    NODE_UNHEALTHY = "node_unhealthy"
    NODE_RECOVERED = "node_recovered"
    NODE_ACTIVATED = "node_activated"
    NODE_CAPACITY_UPDATED = "node_capacity_updated"
    NODE_OVERLOADED = "node_overloaded"

    # =========================================================================
    # ORPHAN DETECTION EVENTS
    # =========================================================================
    ORPHAN_GAMES_DETECTED = "orphan_games_detected"
    ORPHAN_GAMES_REGISTERED = "orphan_games_registered"

    # =========================================================================
    # SYSTEM/DAEMON EVENTS
    # =========================================================================
    DAEMON_STARTED = "daemon_started"
    DAEMON_STOPPED = "daemon_stopped"
    DAEMON_STATUS_CHANGED = "daemon_status_changed"
    HOST_ONLINE = "host_online"
    HOST_OFFLINE = "host_offline"
    ERROR = "error"

    # =========================================================================
    # HEALTH & RECOVERY EVENTS
    # =========================================================================
    HEALTH_CHECK_PASSED = "health_check_passed"
    HEALTH_CHECK_FAILED = "health_check_failed"
    HEALTH_ALERT = "health_alert"
    RESOURCE_CONSTRAINT = "resource_constraint"
    RESOURCE_CONSTRAINT_DETECTED = "resource_constraint_detected"
    RECOVERY_INITIATED = "recovery_initiated"
    RECOVERY_COMPLETED = "recovery_completed"
    RECOVERY_FAILED = "recovery_failed"
    MODEL_CORRUPTED = "model_corrupted"
    COORDINATOR_HEALTH_DEGRADED = "coordinator_health_degraded"
    COORDINATOR_SHUTDOWN = "coordinator_shutdown"
    COORDINATOR_INIT_FAILED = "coordinator_init_failed"
    COORDINATOR_HEARTBEAT = "coordinator_heartbeat"
    HANDLER_TIMEOUT = "handler_timeout"
    HANDLER_FAILED = "handler_failed"

    # =========================================================================
    # WORK QUEUE EVENTS
    # =========================================================================
    WORK_QUEUED = "work_queued"
    WORK_CLAIMED = "work_claimed"
    WORK_STARTED = "work_started"
    WORK_COMPLETED = "work_completed"
    WORK_FAILED = "work_failed"
    WORK_RETRY = "work_retry"
    WORK_TIMEOUT = "work_timeout"
    WORK_CANCELLED = "work_cancelled"

    # =========================================================================
    # LOCK/SYNCHRONIZATION EVENTS
    # =========================================================================
    LOCK_ACQUIRED = "lock_acquired"
    LOCK_RELEASED = "lock_released"
    LOCK_TIMEOUT = "lock_timeout"
    DEADLOCK_DETECTED = "deadlock_detected"

    # =========================================================================
    # CHECKPOINT EVENTS
    # =========================================================================
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_LOADED = "checkpoint_loaded"

    # =========================================================================
    # TASK LIFECYCLE EVENTS
    # =========================================================================
    TASK_SPAWNED = "task_spawned"
    TASK_HEARTBEAT = "task_heartbeat"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_ORPHANED = "task_orphaned"
    TASK_CANCELLED = "task_cancelled"
    TASK_ABANDONED = "task_abandoned"

    # =========================================================================
    # CAPACITY/BACKPRESSURE EVENTS
    # =========================================================================
    BACKPRESSURE_ACTIVATED = "backpressure_activated"
    BACKPRESSURE_RELEASED = "backpressure_released"

    # =========================================================================
    # LEADER ELECTION EVENTS
    # =========================================================================
    LEADER_ELECTED = "leader_elected"
    LEADER_LOST = "leader_lost"
    LEADER_STEPDOWN = "leader_stepdown"

    # =========================================================================
    # ENCODING/PROCESSING EVENTS
    # =========================================================================
    ENCODING_BATCH_COMPLETED = "encoding_batch_completed"
    CALIBRATION_COMPLETED = "calibration_completed"

    # =========================================================================
    # STAGE COMPLETION EVENTS (from StageEvent)
    # =========================================================================
    # These map to pipeline stage completions with a STAGE_ prefix
    STAGE_SELFPLAY_COMPLETE = "selfplay_complete"
    STAGE_CANONICAL_SELFPLAY_COMPLETE = "canonical_selfplay_complete"
    STAGE_GPU_SELFPLAY_COMPLETE = "gpu_selfplay_complete"
    STAGE_SYNC_COMPLETE = "sync_complete"
    STAGE_PARITY_VALIDATION_COMPLETE = "parity_validation_complete"
    STAGE_NPZ_EXPORT_COMPLETE = "npz_export_complete"
    STAGE_TRAINING_COMPLETE = "stage_training_complete"  # Distinct from TRAINING_COMPLETED
    STAGE_TRAINING_STARTED = "stage_training_started"  # Distinct from TRAINING_STARTED
    STAGE_TRAINING_FAILED = "stage_training_failed"  # Distinct from TRAINING_FAILED
    STAGE_EVALUATION_COMPLETE = "stage_evaluation_complete"
    STAGE_SHADOW_TOURNAMENT_COMPLETE = "shadow_tournament_complete"
    STAGE_ELO_CALIBRATION_COMPLETE = "elo_calibration_complete"
    STAGE_CMAES_COMPLETE = "stage_cmaes_complete"
    STAGE_PBT_COMPLETE = "stage_pbt_complete"
    STAGE_NAS_COMPLETE = "stage_nas_complete"
    STAGE_PROMOTION_COMPLETE = "stage_promotion_complete"
    STAGE_TIER_GATING_COMPLETE = "tier_gating_complete"
    STAGE_ITERATION_COMPLETE = "iteration_complete"
    STAGE_CLUSTER_SYNC_COMPLETE = "cluster_sync_complete"
    STAGE_MODEL_SYNC_COMPLETE = "model_sync_complete"


# =============================================================================
# Event Category Mapping
# =============================================================================

_EVENT_CATEGORIES: dict[RingRiftEventType, EventCategory] = {
    # Data events
    RingRiftEventType.NEW_GAMES_AVAILABLE: EventCategory.DATA,
    RingRiftEventType.DATA_SYNC_STARTED: EventCategory.DATA,
    RingRiftEventType.DATA_SYNC_COMPLETED: EventCategory.DATA,
    RingRiftEventType.DATA_SYNC_FAILED: EventCategory.DATA,
    RingRiftEventType.GAME_SYNCED: EventCategory.DATA,
    RingRiftEventType.DATA_STALE: EventCategory.DATA,
    RingRiftEventType.DATA_FRESH: EventCategory.DATA,
    RingRiftEventType.SYNC_TRIGGERED: EventCategory.DATA,
    RingRiftEventType.SYNC_STALLED: EventCategory.DATA,

    # Training events
    RingRiftEventType.TRAINING_THRESHOLD_REACHED: EventCategory.TRAINING,
    RingRiftEventType.TRAINING_STARTED: EventCategory.TRAINING,
    RingRiftEventType.TRAINING_PROGRESS: EventCategory.TRAINING,
    RingRiftEventType.TRAINING_COMPLETED: EventCategory.TRAINING,
    RingRiftEventType.TRAINING_FAILED: EventCategory.TRAINING,
    RingRiftEventType.TRAINING_LOSS_ANOMALY: EventCategory.TRAINING,
    RingRiftEventType.TRAINING_LOSS_TREND: EventCategory.TRAINING,
    RingRiftEventType.TRAINING_EARLY_STOPPED: EventCategory.TRAINING,
    RingRiftEventType.TRAINING_ROLLBACK_NEEDED: EventCategory.TRAINING,
    RingRiftEventType.TRAINING_ROLLBACK_COMPLETED: EventCategory.TRAINING,

    # Evaluation events
    RingRiftEventType.EVALUATION_STARTED: EventCategory.EVALUATION,
    RingRiftEventType.EVALUATION_PROGRESS: EventCategory.EVALUATION,
    RingRiftEventType.EVALUATION_COMPLETED: EventCategory.EVALUATION,
    RingRiftEventType.EVALUATION_FAILED: EventCategory.EVALUATION,
    RingRiftEventType.ELO_UPDATED: EventCategory.EVALUATION,
    RingRiftEventType.ELO_SIGNIFICANT_CHANGE: EventCategory.EVALUATION,
    RingRiftEventType.ELO_VELOCITY_CHANGED: EventCategory.EVALUATION,
    RingRiftEventType.ADAPTIVE_PARAMS_CHANGED: EventCategory.EVALUATION,

    # Promotion events
    RingRiftEventType.PROMOTION_CANDIDATE: EventCategory.PROMOTION,
    RingRiftEventType.PROMOTION_STARTED: EventCategory.PROMOTION,
    RingRiftEventType.MODEL_PROMOTED: EventCategory.PROMOTION,
    RingRiftEventType.PROMOTION_FAILED: EventCategory.PROMOTION,
    RingRiftEventType.PROMOTION_REJECTED: EventCategory.PROMOTION,
    RingRiftEventType.PROMOTION_ROLLED_BACK: EventCategory.PROMOTION,
    RingRiftEventType.TIER_PROMOTION: EventCategory.PROMOTION,
    RingRiftEventType.CROSSBOARD_PROMOTION: EventCategory.PROMOTION,

    # Curriculum events
    RingRiftEventType.CURRICULUM_REBALANCED: EventCategory.CURRICULUM,
    RingRiftEventType.CURRICULUM_ADVANCED: EventCategory.CURRICULUM,
    RingRiftEventType.WEIGHT_UPDATED: EventCategory.CURRICULUM,

    # Selfplay events
    RingRiftEventType.SELFPLAY_TARGET_UPDATED: EventCategory.SELFPLAY,
    RingRiftEventType.SELFPLAY_RATE_CHANGED: EventCategory.SELFPLAY,
    RingRiftEventType.IDLE_RESOURCE_DETECTED: EventCategory.SELFPLAY,

    # Optimization events
    RingRiftEventType.CMAES_TRIGGERED: EventCategory.OPTIMIZATION,
    RingRiftEventType.CMAES_COMPLETED: EventCategory.OPTIMIZATION,
    RingRiftEventType.NAS_TRIGGERED: EventCategory.OPTIMIZATION,
    RingRiftEventType.NAS_STARTED: EventCategory.OPTIMIZATION,
    RingRiftEventType.NAS_GENERATION_COMPLETE: EventCategory.OPTIMIZATION,
    RingRiftEventType.NAS_COMPLETED: EventCategory.OPTIMIZATION,
    RingRiftEventType.NAS_BEST_ARCHITECTURE: EventCategory.OPTIMIZATION,
    RingRiftEventType.PBT_STARTED: EventCategory.OPTIMIZATION,
    RingRiftEventType.PBT_GENERATION_COMPLETE: EventCategory.OPTIMIZATION,
    RingRiftEventType.PBT_COMPLETED: EventCategory.OPTIMIZATION,
    RingRiftEventType.PLATEAU_DETECTED: EventCategory.OPTIMIZATION,
    RingRiftEventType.HYPERPARAMETER_UPDATED: EventCategory.OPTIMIZATION,

    # Quality events
    RingRiftEventType.DATA_QUALITY_ALERT: EventCategory.QUALITY,
    RingRiftEventType.QUALITY_CHECK_FAILED: EventCategory.QUALITY,
    RingRiftEventType.QUALITY_SCORE_UPDATED: EventCategory.QUALITY,
    RingRiftEventType.QUALITY_DISTRIBUTION_CHANGED: EventCategory.QUALITY,
    RingRiftEventType.HIGH_QUALITY_DATA_AVAILABLE: EventCategory.QUALITY,
    RingRiftEventType.QUALITY_DEGRADED: EventCategory.QUALITY,
    RingRiftEventType.LOW_QUALITY_DATA_WARNING: EventCategory.QUALITY,
    RingRiftEventType.TRAINING_BLOCKED_BY_QUALITY: EventCategory.QUALITY,
    RingRiftEventType.QUALITY_PENALTY_APPLIED: EventCategory.QUALITY,

    # Regression events
    RingRiftEventType.REGRESSION_DETECTED: EventCategory.REGRESSION,
    RingRiftEventType.REGRESSION_MINOR: EventCategory.REGRESSION,
    RingRiftEventType.REGRESSION_MODERATE: EventCategory.REGRESSION,
    RingRiftEventType.REGRESSION_SEVERE: EventCategory.REGRESSION,
    RingRiftEventType.REGRESSION_CRITICAL: EventCategory.REGRESSION,
    RingRiftEventType.REGRESSION_CLEARED: EventCategory.REGRESSION,

    # Cluster events
    RingRiftEventType.P2P_MODEL_SYNCED: EventCategory.CLUSTER,
    RingRiftEventType.P2P_CLUSTER_HEALTHY: EventCategory.CLUSTER,
    RingRiftEventType.P2P_CLUSTER_UNHEALTHY: EventCategory.CLUSTER,
    RingRiftEventType.P2P_NODES_DEAD: EventCategory.CLUSTER,
    RingRiftEventType.P2P_SELFPLAY_SCALED: EventCategory.CLUSTER,
    RingRiftEventType.CLUSTER_STATUS_CHANGED: EventCategory.CLUSTER,
    RingRiftEventType.CLUSTER_CAPACITY_CHANGED: EventCategory.CLUSTER,
    RingRiftEventType.NODE_UNHEALTHY: EventCategory.CLUSTER,
    RingRiftEventType.NODE_RECOVERED: EventCategory.CLUSTER,
    RingRiftEventType.NODE_ACTIVATED: EventCategory.CLUSTER,
    RingRiftEventType.NODE_CAPACITY_UPDATED: EventCategory.CLUSTER,
    RingRiftEventType.NODE_OVERLOADED: EventCategory.CLUSTER,

    # Work queue events
    RingRiftEventType.WORK_QUEUED: EventCategory.WORK,
    RingRiftEventType.WORK_CLAIMED: EventCategory.WORK,
    RingRiftEventType.WORK_STARTED: EventCategory.WORK,
    RingRiftEventType.WORK_COMPLETED: EventCategory.WORK,
    RingRiftEventType.WORK_FAILED: EventCategory.WORK,
    RingRiftEventType.WORK_RETRY: EventCategory.WORK,
    RingRiftEventType.WORK_TIMEOUT: EventCategory.WORK,
    RingRiftEventType.WORK_CANCELLED: EventCategory.WORK,

    # Stage events
    RingRiftEventType.STAGE_SELFPLAY_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_CANONICAL_SELFPLAY_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_GPU_SELFPLAY_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_SYNC_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_PARITY_VALIDATION_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_NPZ_EXPORT_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_TRAINING_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_TRAINING_STARTED: EventCategory.STAGE,
    RingRiftEventType.STAGE_TRAINING_FAILED: EventCategory.STAGE,
    RingRiftEventType.STAGE_EVALUATION_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_SHADOW_TOURNAMENT_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_ELO_CALIBRATION_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_CMAES_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_PBT_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_NAS_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_PROMOTION_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_TIER_GATING_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_ITERATION_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_CLUSTER_SYNC_COMPLETE: EventCategory.STAGE,
    RingRiftEventType.STAGE_MODEL_SYNC_COMPLETE: EventCategory.STAGE,
}


# =============================================================================
# Backwards Compatibility Aliases
# =============================================================================

# Alias the old names to the new unified type
DataEventType = RingRiftEventType

# Create a StageEvent alias class that maps to RingRiftEventType
class StageEvent(Enum):
    """Backwards compatibility alias for stage events.

    .. deprecated:: December 2025
        Use RingRiftEventType.STAGE_* instead.
    """

    SELFPLAY_COMPLETE = RingRiftEventType.STAGE_SELFPLAY_COMPLETE.value
    CANONICAL_SELFPLAY_COMPLETE = RingRiftEventType.STAGE_CANONICAL_SELFPLAY_COMPLETE.value
    GPU_SELFPLAY_COMPLETE = RingRiftEventType.STAGE_GPU_SELFPLAY_COMPLETE.value
    SYNC_COMPLETE = RingRiftEventType.STAGE_SYNC_COMPLETE.value
    PARITY_VALIDATION_COMPLETE = RingRiftEventType.STAGE_PARITY_VALIDATION_COMPLETE.value
    NPZ_EXPORT_COMPLETE = RingRiftEventType.STAGE_NPZ_EXPORT_COMPLETE.value
    TRAINING_COMPLETE = RingRiftEventType.STAGE_TRAINING_COMPLETE.value
    TRAINING_STARTED = RingRiftEventType.STAGE_TRAINING_STARTED.value
    TRAINING_FAILED = RingRiftEventType.STAGE_TRAINING_FAILED.value
    EVALUATION_COMPLETE = RingRiftEventType.STAGE_EVALUATION_COMPLETE.value
    SHADOW_TOURNAMENT_COMPLETE = RingRiftEventType.STAGE_SHADOW_TOURNAMENT_COMPLETE.value
    ELO_CALIBRATION_COMPLETE = RingRiftEventType.STAGE_ELO_CALIBRATION_COMPLETE.value
    CMAES_COMPLETE = RingRiftEventType.STAGE_CMAES_COMPLETE.value
    PBT_COMPLETE = RingRiftEventType.STAGE_PBT_COMPLETE.value
    NAS_COMPLETE = RingRiftEventType.STAGE_NAS_COMPLETE.value
    PROMOTION_COMPLETE = RingRiftEventType.STAGE_PROMOTION_COMPLETE.value
    TIER_GATING_COMPLETE = RingRiftEventType.STAGE_TIER_GATING_COMPLETE.value
    ITERATION_COMPLETE = RingRiftEventType.STAGE_ITERATION_COMPLETE.value
    CLUSTER_SYNC_COMPLETE = RingRiftEventType.STAGE_CLUSTER_SYNC_COMPLETE.value
    MODEL_SYNC_COMPLETE = RingRiftEventType.STAGE_MODEL_SYNC_COMPLETE.value


# =============================================================================
# Utility Functions
# =============================================================================

def get_events_by_category(category: EventCategory) -> list[RingRiftEventType]:
    """Get all event types in a category."""
    return [
        event_type
        for event_type, cat in _EVENT_CATEGORIES.items()
        if cat == category
    ]


def is_cross_process_event(event_type: RingRiftEventType) -> bool:
    """Check if an event should be propagated across processes.

    These events are important for distributed coordination.
    """
    return event_type in CROSS_PROCESS_EVENT_TYPES


# Events that should be propagated across processes
CROSS_PROCESS_EVENT_TYPES = {
    # Success events - coordination across processes
    RingRiftEventType.MODEL_PROMOTED,
    RingRiftEventType.TIER_PROMOTION,
    RingRiftEventType.TRAINING_STARTED,
    RingRiftEventType.TRAINING_COMPLETED,
    RingRiftEventType.EVALUATION_COMPLETED,
    RingRiftEventType.CURRICULUM_REBALANCED,
    RingRiftEventType.CURRICULUM_ADVANCED,
    RingRiftEventType.SELFPLAY_TARGET_UPDATED,
    RingRiftEventType.ELO_SIGNIFICANT_CHANGE,
    RingRiftEventType.P2P_MODEL_SYNCED,
    RingRiftEventType.PLATEAU_DETECTED,
    RingRiftEventType.DATA_SYNC_COMPLETED,
    RingRiftEventType.HYPERPARAMETER_UPDATED,
    RingRiftEventType.GAME_SYNCED,
    RingRiftEventType.DATA_STALE,

    # Failure events
    RingRiftEventType.TRAINING_FAILED,
    RingRiftEventType.EVALUATION_FAILED,
    RingRiftEventType.PROMOTION_FAILED,
    RingRiftEventType.DATA_SYNC_FAILED,

    # Host/cluster events
    RingRiftEventType.HOST_ONLINE,
    RingRiftEventType.HOST_OFFLINE,
    RingRiftEventType.DAEMON_STARTED,
    RingRiftEventType.DAEMON_STOPPED,
    RingRiftEventType.DAEMON_STATUS_CHANGED,

    # Trigger events
    RingRiftEventType.CMAES_TRIGGERED,
    RingRiftEventType.NAS_TRIGGERED,
    RingRiftEventType.TRAINING_THRESHOLD_REACHED,
    RingRiftEventType.CACHE_INVALIDATED,

    # Regression events
    RingRiftEventType.REGRESSION_DETECTED,
    RingRiftEventType.REGRESSION_SEVERE,
    RingRiftEventType.REGRESSION_CRITICAL,
    RingRiftEventType.REGRESSION_CLEARED,
}


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main enum
    "RingRiftEventType",
    # Category support
    "EventCategory",
    "get_events_by_category",
    # Cross-process support
    "CROSS_PROCESS_EVENT_TYPES",
    "is_cross_process_event",
    # Backwards compatibility
    "DataEventType",
    "StageEvent",
]
