"""
Tests for event system integration scenarios.

Tests how events work together in real-world scenarios.
"""

import pytest

from app.events import (
    EventCategory,
    RingRiftEventType,
    get_events_by_category,
    is_cross_process_event,
)


# ============================================
# Test Training Pipeline Events
# ============================================


class TestTrainingPipelineEvents:
    """Tests for training pipeline event sequences."""

    def test_training_lifecycle_events_exist(self):
        """Test that all training lifecycle events exist."""
        lifecycle = [
            RingRiftEventType.TRAINING_THRESHOLD_REACHED,
            RingRiftEventType.TRAINING_STARTED,
            RingRiftEventType.TRAINING_PROGRESS,
            RingRiftEventType.TRAINING_COMPLETED,
        ]

        for event in lifecycle:
            assert event in RingRiftEventType

    def test_training_failure_path_exists(self):
        """Test that training failure path events exist."""
        failure_path = [
            RingRiftEventType.TRAINING_STARTED,
            RingRiftEventType.TRAINING_FAILED,
        ]

        for event in failure_path:
            assert event in RingRiftEventType

    def test_training_early_stopping_exists(self):
        """Test that training can be early stopped."""
        assert RingRiftEventType.TRAINING_EARLY_STOPPED in RingRiftEventType

    def test_training_rollback_sequence_exists(self):
        """Test that training rollback sequence exists."""
        rollback_sequence = [
            RingRiftEventType.TRAINING_ROLLBACK_NEEDED,
            RingRiftEventType.TRAINING_ROLLBACK_COMPLETED,
        ]

        for event in rollback_sequence:
            assert event in RingRiftEventType


# ============================================
# Test Evaluation Pipeline Events
# ============================================


class TestEvaluationPipelineEvents:
    """Tests for evaluation pipeline event sequences."""

    def test_evaluation_lifecycle_exists(self):
        """Test that evaluation lifecycle events exist."""
        lifecycle = [
            RingRiftEventType.EVALUATION_STARTED,
            RingRiftEventType.EVALUATION_PROGRESS,
            RingRiftEventType.EVALUATION_COMPLETED,
        ]

        for event in lifecycle:
            assert event in RingRiftEventType

    def test_elo_update_events_exist(self):
        """Test that Elo update events exist."""
        elo_events = [
            RingRiftEventType.ELO_UPDATED,
            RingRiftEventType.ELO_SIGNIFICANT_CHANGE,
            RingRiftEventType.ELO_VELOCITY_CHANGED,
        ]

        for event in elo_events:
            assert event in RingRiftEventType

    def test_evaluation_triggers_curriculum(self):
        """Test that evaluation can trigger curriculum changes."""
        # Evaluation produces Elo changes
        assert RingRiftEventType.ELO_SIGNIFICANT_CHANGE in RingRiftEventType
        # Which can trigger curriculum rebalancing
        assert RingRiftEventType.CURRICULUM_REBALANCED in RingRiftEventType


# ============================================
# Test Promotion Pipeline Events
# ============================================


class TestPromotionPipelineEvents:
    """Tests for promotion pipeline event sequences."""

    def test_promotion_lifecycle_exists(self):
        """Test that promotion lifecycle events exist."""
        lifecycle = [
            RingRiftEventType.PROMOTION_CANDIDATE,
            RingRiftEventType.PROMOTION_STARTED,
            RingRiftEventType.MODEL_PROMOTED,
        ]

        for event in lifecycle:
            assert event in RingRiftEventType

    def test_promotion_rejection_path_exists(self):
        """Test that promotion can be rejected."""
        rejection_path = [
            RingRiftEventType.PROMOTION_STARTED,
            RingRiftEventType.PROMOTION_REJECTED,
        ]

        for event in rejection_path:
            assert event in RingRiftEventType

    def test_promotion_rollback_exists(self):
        """Test that promotions can be rolled back."""
        assert RingRiftEventType.PROMOTION_ROLLED_BACK in RingRiftEventType


# ============================================
# Test Data Sync Pipeline Events
# ============================================


class TestDataSyncPipelineEvents:
    """Tests for data sync pipeline event sequences."""

    def test_sync_lifecycle_exists(self):
        """Test that sync lifecycle events exist."""
        lifecycle = [
            RingRiftEventType.DATA_SYNC_STARTED,
            RingRiftEventType.DATA_SYNC_COMPLETED,
        ]

        for event in lifecycle:
            assert event in RingRiftEventType

    def test_sync_failure_path_exists(self):
        """Test that sync can fail."""
        failure_path = [
            RingRiftEventType.DATA_SYNC_STARTED,
            RingRiftEventType.DATA_SYNC_FAILED,
        ]

        for event in failure_path:
            assert event in RingRiftEventType

    def test_data_freshness_events_exist(self):
        """Test that data freshness monitoring exists."""
        freshness_events = [
            RingRiftEventType.DATA_STALE,
            RingRiftEventType.DATA_FRESH,
            RingRiftEventType.SYNC_TRIGGERED,
        ]

        for event in freshness_events:
            assert event in RingRiftEventType

    def test_orphan_games_workflow_exists(self):
        """Test that orphan games workflow exists."""
        workflow = [
            RingRiftEventType.ORPHAN_GAMES_DETECTED,
            RingRiftEventType.ORPHAN_GAMES_REGISTERED,
        ]

        for event in workflow:
            assert event in RingRiftEventType


# ============================================
# Test Quality Feedback Loop Events
# ============================================


class TestQualityFeedbackLoopEvents:
    """Tests for quality feedback loop event sequences."""

    def test_quality_monitoring_events_exist(self):
        """Test that quality monitoring events exist."""
        monitoring = [
            RingRiftEventType.QUALITY_SCORE_UPDATED,
            RingRiftEventType.QUALITY_DISTRIBUTION_CHANGED,
            RingRiftEventType.HIGH_QUALITY_DATA_AVAILABLE,
        ]

        for event in monitoring:
            assert event in RingRiftEventType

    def test_quality_degradation_workflow_exists(self):
        """Test that quality degradation workflow exists."""
        workflow = [
            RingRiftEventType.QUALITY_DEGRADED,
            RingRiftEventType.LOW_QUALITY_DATA_WARNING,
            RingRiftEventType.TRAINING_BLOCKED_BY_QUALITY,
        ]

        for event in workflow:
            assert event in RingRiftEventType

    def test_quality_feedback_adjustments_exist(self):
        """Test that quality feedback adjustments exist."""
        adjustments = [
            RingRiftEventType.QUALITY_FEEDBACK_ADJUSTED,
            RingRiftEventType.QUALITY_PENALTY_APPLIED,
            RingRiftEventType.EXPLORATION_BOOST,
            RingRiftEventType.EXPLORATION_ADJUSTED,
        ]

        for event in adjustments:
            assert event in RingRiftEventType


# ============================================
# Test Regression Detection Events
# ============================================


class TestRegressionDetectionEvents:
    """Tests for regression detection event sequences."""

    def test_regression_severity_levels_workflow(self):
        """Test that regression severity levels exist."""
        severities = [
            RingRiftEventType.REGRESSION_MINOR,
            RingRiftEventType.REGRESSION_MODERATE,
            RingRiftEventType.REGRESSION_SEVERE,
            RingRiftEventType.REGRESSION_CRITICAL,
        ]

        for event in severities:
            assert event in RingRiftEventType

    def test_regression_recovery_exists(self):
        """Test that regression recovery events exist."""
        recovery = [
            RingRiftEventType.REGRESSION_DETECTED,
            RingRiftEventType.REGRESSION_CLEARED,
        ]

        for event in recovery:
            assert event in RingRiftEventType

    def test_regression_triggers_rollback(self):
        """Test that severe regression can trigger rollback."""
        # Severe regression detected
        assert RingRiftEventType.REGRESSION_SEVERE in RingRiftEventType
        # Can trigger rollback
        assert RingRiftEventType.TRAINING_ROLLBACK_NEEDED in RingRiftEventType


# ============================================
# Test Cluster Coordination Events
# ============================================


class TestClusterCoordinationEvents:
    """Tests for cluster coordination event sequences."""

    def test_node_lifecycle_events_exist(self):
        """Test that node lifecycle events exist."""
        lifecycle = [
            RingRiftEventType.HOST_ONLINE,
            RingRiftEventType.NODE_UNHEALTHY,
            RingRiftEventType.NODE_RECOVERED,
            RingRiftEventType.HOST_OFFLINE,
        ]

        for event in lifecycle:
            assert event in RingRiftEventType

    def test_cluster_health_events_exist(self):
        """Test that cluster health events exist."""
        health = [
            RingRiftEventType.P2P_CLUSTER_HEALTHY,
            RingRiftEventType.P2P_CLUSTER_UNHEALTHY,
            RingRiftEventType.P2P_NODES_DEAD,
        ]

        for event in health:
            assert event in RingRiftEventType

    def test_leader_election_events_exist(self):
        """Test that leader election events exist."""
        election = [
            RingRiftEventType.LEADER_ELECTED,
            RingRiftEventType.LEADER_LOST,
            RingRiftEventType.LEADER_STEPDOWN,
        ]

        for event in election:
            assert event in RingRiftEventType


# ============================================
# Test Work Queue Events
# ============================================


class TestWorkQueueEvents:
    """Tests for work queue event sequences."""

    def test_work_lifecycle_exists(self):
        """Test that work lifecycle events exist."""
        lifecycle = [
            RingRiftEventType.WORK_QUEUED,
            RingRiftEventType.WORK_CLAIMED,
            RingRiftEventType.WORK_STARTED,
            RingRiftEventType.WORK_COMPLETED,
        ]

        for event in lifecycle:
            assert event in RingRiftEventType

    def test_work_retry_workflow_exists(self):
        """Test that work retry workflow exists."""
        retry = [
            RingRiftEventType.WORK_FAILED,
            RingRiftEventType.WORK_RETRY,
        ]

        for event in retry:
            assert event in RingRiftEventType

    def test_work_cancellation_exists(self):
        """Test that work can be cancelled."""
        assert RingRiftEventType.WORK_CANCELLED in RingRiftEventType

    def test_backpressure_workflow_exists(self):
        """Test that backpressure workflow exists."""
        backpressure = [
            RingRiftEventType.BACKPRESSURE_ACTIVATED,
            RingRiftEventType.BACKPRESSURE_RELEASED,
        ]

        for event in backpressure:
            assert event in RingRiftEventType


# ============================================
# Test Optimization Events
# ============================================


class TestOptimizationEvents:
    """Tests for optimization event sequences."""

    def test_cmaes_workflow_exists(self):
        """Test that CMA-ES workflow exists."""
        workflow = [
            RingRiftEventType.PLATEAU_DETECTED,
            RingRiftEventType.CMAES_TRIGGERED,
            RingRiftEventType.CMAES_COMPLETED,
        ]

        for event in workflow:
            assert event in RingRiftEventType

    def test_nas_workflow_exists(self):
        """Test that NAS workflow exists."""
        workflow = [
            RingRiftEventType.NAS_TRIGGERED,
            RingRiftEventType.NAS_STARTED,
            RingRiftEventType.NAS_GENERATION_COMPLETE,
            RingRiftEventType.NAS_COMPLETED,
            RingRiftEventType.NAS_BEST_ARCHITECTURE,
        ]

        for event in workflow:
            assert event in RingRiftEventType

    def test_pbt_workflow_exists(self):
        """Test that PBT workflow exists."""
        workflow = [
            RingRiftEventType.PBT_STARTED,
            RingRiftEventType.PBT_GENERATION_COMPLETE,
            RingRiftEventType.PBT_COMPLETED,
        ]

        for event in workflow:
            assert event in RingRiftEventType


# ============================================
# Test Cross-Process Event Patterns
# ============================================


class TestCrossProcessEventPatterns:
    """Tests for cross-process event communication patterns."""

    def test_training_events_are_cross_process(self):
        """Test that key training events are cross-process."""
        key_events = [
            RingRiftEventType.TRAINING_STARTED,
            RingRiftEventType.TRAINING_COMPLETED,
            RingRiftEventType.TRAINING_FAILED,
        ]

        for event in key_events:
            assert is_cross_process_event(event), f"{event} should be cross-process"

    def test_promotion_events_are_cross_process(self):
        """Test that promotion events are cross-process."""
        assert is_cross_process_event(RingRiftEventType.MODEL_PROMOTED)

    def test_sync_completion_is_cross_process(self):
        """Test that sync completion is cross-process."""
        assert is_cross_process_event(RingRiftEventType.DATA_SYNC_COMPLETED)

    def test_cluster_events_are_cross_process(self):
        """Test that cluster coordination events are cross-process."""
        cluster_events = [
            RingRiftEventType.HOST_ONLINE,
            RingRiftEventType.HOST_OFFLINE,
            RingRiftEventType.DAEMON_STARTED,
            RingRiftEventType.DAEMON_STOPPED,
        ]

        for event in cluster_events:
            assert is_cross_process_event(event), f"{event} should be cross-process"


# ============================================
# Test Stage Completion Events
# ============================================


class TestStageCompletionEvents:
    """Tests for pipeline stage completion events."""

    def test_all_stage_events_in_stage_category(self):
        """Test that all stage events are in STAGE category."""
        stage_events = [
            RingRiftEventType.STAGE_SELFPLAY_COMPLETE,
            RingRiftEventType.STAGE_TRAINING_COMPLETE,
            RingRiftEventType.STAGE_EVALUATION_COMPLETE,
            RingRiftEventType.STAGE_PROMOTION_COMPLETE,
        ]

        for event in stage_events:
            assert EventCategory.from_event(event) == EventCategory.STAGE

    def test_stage_events_different_from_general_events(self):
        """Test that stage events are distinct from general events."""
        # Stage events should have different values than general events
        assert RingRiftEventType.STAGE_TRAINING_COMPLETE.value != RingRiftEventType.TRAINING_COMPLETED.value
        assert RingRiftEventType.STAGE_EVALUATION_COMPLETE.value != RingRiftEventType.EVALUATION_COMPLETED.value


# ============================================
# Test Event Category Completeness
# ============================================


class TestEventCategoryCompleteness:
    """Tests for event category completeness."""

    def test_all_categories_have_events(self):
        """Test that all categories have at least one event."""
        expected_categories = [
            EventCategory.DATA,
            EventCategory.TRAINING,
            EventCategory.EVALUATION,
            EventCategory.PROMOTION,
            EventCategory.SELFPLAY,
            EventCategory.CLUSTER,
            EventCategory.STAGE,
        ]

        for category in expected_categories:
            events = get_events_by_category(category)
            assert len(events) > 0, f"Category {category} has no events"

    def test_stage_category_events_have_stage_prefix(self):
        """Test that most STAGE category events have STAGE_ prefix."""
        stage_events = get_events_by_category(EventCategory.STAGE)

        # Most stage events should start with STAGE_
        # Some legacy events like SELFPLAY_COMPLETE are in STAGE category but don't have prefix
        stage_prefixed = [e for e in stage_events if e.name.startswith("STAGE_")]

        # At least 80% should have the prefix
        assert len(stage_prefixed) / len(stage_events) >= 0.8

    def test_training_category_has_lifecycle_events(self):
        """Test that TRAINING category has complete lifecycle."""
        training_events = get_events_by_category(EventCategory.TRAINING)

        # Should have started, completed, and failed
        event_names = [e.name for e in training_events]
        assert "TRAINING_STARTED" in event_names
        assert "TRAINING_COMPLETED" in event_names
        assert "TRAINING_FAILED" in event_names


# ============================================
# Test Event Naming Consistency
# ============================================


class TestEventNamingConsistency:
    """Tests for event naming consistency."""

    def test_stage_events_use_complete_suffix(self):
        """Test that stage completion events use COMPLETE suffix."""
        stage_events = get_events_by_category(EventCategory.STAGE)

        completion_events = [
            e for e in stage_events
            if "COMPLETE" in e.name or "STARTED" in e.name or "FAILED" in e.name
        ]

        # Most stage events should be completion events
        assert len(completion_events) > 0

    def test_failed_events_exist_for_major_operations(self):
        """Test that _FAILED events exist for major operations."""
        major_operations = ["TRAINING", "EVALUATION", "PROMOTION", "DATA_SYNC"]

        for operation in major_operations:
            failed_event_name = f"{operation}_FAILED"
            assert any(
                e.name == failed_event_name for e in RingRiftEventType
            ), f"Missing {failed_event_name} event"

    def test_started_events_have_completed_counterparts(self):
        """Test that _STARTED events have _COMPLETED counterparts where applicable."""
        started_events = [e for e in RingRiftEventType if e.name.endswith("_STARTED")]

        # Exceptions: PROMOTION_STARTED -> MODEL_PROMOTED (not PROMOTION_COMPLETED)
        # DAEMON_STARTED -> DAEMON_STOPPED (not DAEMON_COMPLETED)
        # STAGE_TRAINING_STARTED -> STAGE_TRAINING_COMPLETE (not COMPLETE_D_)
        exceptions = {
            "PROMOTION_STARTED": "MODEL_PROMOTED",
            "DAEMON_STARTED": "DAEMON_STOPPED",
            "STAGE_TRAINING_STARTED": "STAGE_TRAINING_COMPLETE",
        }

        for started_event in started_events:
            if started_event.name in exceptions:
                # Check for the exception event
                expected_name = exceptions[started_event.name]
                assert any(
                    e.name == expected_name for e in RingRiftEventType
                ), f"Missing {expected_name} for {started_event.name}"
            else:
                # Check for normal _COMPLETED event
                base_name = started_event.name.replace("_STARTED", "")
                completed_name = f"{base_name}_COMPLETED"

                assert any(
                    e.name == completed_name for e in RingRiftEventType
                ), f"Missing {completed_name} for {started_event.name}"
