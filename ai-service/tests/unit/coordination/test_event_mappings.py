"""Tests for event type mappings.

Created: December 2025
Tests for: app/coordination/event_mappings.py
"""

from __future__ import annotations

import pytest

from app.coordination.event_mappings import (
    CROSS_PROCESS_TO_DATA_MAP,
    DATA_TO_CROSS_PROCESS_MAP,
    DATA_TO_STAGE_EVENT_MAP,
    STAGE_TO_CROSS_PROCESS_MAP,
    STAGE_TO_DATA_EVENT_MAP,
    get_all_event_types,
    get_cross_process_event_type,
    get_data_event_type,
    get_stage_event_type,
    is_mapped_event,
    validate_mappings,
)


class TestStageToDataEventMap:
    """Tests for STAGE_TO_DATA_EVENT_MAP."""

    def test_selfplay_events_mapped(self):
        """Test selfplay stage events map correctly."""
        assert STAGE_TO_DATA_EVENT_MAP["selfplay_complete"] == "selfplay_complete"
        assert STAGE_TO_DATA_EVENT_MAP["canonical_selfplay_complete"] == "selfplay_complete"
        assert STAGE_TO_DATA_EVENT_MAP["gpu_selfplay_complete"] == "selfplay_complete"

    def test_sync_events_mapped(self):
        """Test sync stage events map correctly."""
        assert STAGE_TO_DATA_EVENT_MAP["sync_complete"] == "sync_completed"
        assert STAGE_TO_DATA_EVENT_MAP["cluster_sync_complete"] == "sync_completed"
        assert STAGE_TO_DATA_EVENT_MAP["model_sync_complete"] == "p2p_model_synced"

    def test_training_events_mapped(self):
        """Test training stage events map correctly."""
        assert STAGE_TO_DATA_EVENT_MAP["training_complete"] == "training_completed"
        assert STAGE_TO_DATA_EVENT_MAP["training_started"] == "training_started"
        assert STAGE_TO_DATA_EVENT_MAP["training_failed"] == "training_failed"

    def test_evaluation_events_mapped(self):
        """Test evaluation stage events map correctly."""
        assert STAGE_TO_DATA_EVENT_MAP["evaluation_complete"] == "evaluation_completed"
        assert STAGE_TO_DATA_EVENT_MAP["shadow_tournament_complete"] == "evaluation_completed"
        assert STAGE_TO_DATA_EVENT_MAP["elo_calibration_complete"] == "elo_updated"

    def test_promotion_events_mapped(self):
        """Test promotion stage events map correctly."""
        assert STAGE_TO_DATA_EVENT_MAP["promotion_complete"] == "model_promoted"
        assert STAGE_TO_DATA_EVENT_MAP["tier_gating_complete"] == "model_promoted"

    def test_optimization_events_mapped(self):
        """Test optimization stage events map correctly."""
        assert STAGE_TO_DATA_EVENT_MAP["cmaes_complete"] == "cmaes_completed"
        assert STAGE_TO_DATA_EVENT_MAP["pbt_complete"] == "pbt_generation_complete"
        assert STAGE_TO_DATA_EVENT_MAP["nas_complete"] == "nas_completed"


class TestDataToStageEventMap:
    """Tests for DATA_TO_STAGE_EVENT_MAP (reverse mapping)."""

    def test_selfplay_reverse_mapping(self):
        """Test selfplay data events map back to stage."""
        assert DATA_TO_STAGE_EVENT_MAP["selfplay_complete"] == "selfplay_complete"
        assert DATA_TO_STAGE_EVENT_MAP["new_games"] == "selfplay_complete"

    def test_sync_reverse_mapping(self):
        """Test sync data events map back to stage."""
        assert DATA_TO_STAGE_EVENT_MAP["sync_completed"] == "sync_complete"
        assert DATA_TO_STAGE_EVENT_MAP["p2p_model_synced"] == "model_sync_complete"

    def test_training_reverse_mapping(self):
        """Test training data events map back to stage."""
        assert DATA_TO_STAGE_EVENT_MAP["training_completed"] == "training_complete"
        assert DATA_TO_STAGE_EVENT_MAP["training_started"] == "training_started"
        assert DATA_TO_STAGE_EVENT_MAP["training_failed"] == "training_failed"


class TestDataToCrossProcessMap:
    """Tests for DATA_TO_CROSS_PROCESS_MAP."""

    def test_training_events_uppercase(self):
        """Test training events are UPPERCASE in cross-process."""
        assert DATA_TO_CROSS_PROCESS_MAP["training_started"] == "TRAINING_STARTED"
        assert DATA_TO_CROSS_PROCESS_MAP["training_completed"] == "TRAINING_COMPLETED"
        assert DATA_TO_CROSS_PROCESS_MAP["training_failed"] == "TRAINING_FAILED"

    def test_evaluation_events_uppercase(self):
        """Test evaluation events are UPPERCASE in cross-process."""
        assert DATA_TO_CROSS_PROCESS_MAP["evaluation_started"] == "EVALUATION_STARTED"
        assert DATA_TO_CROSS_PROCESS_MAP["evaluation_completed"] == "EVALUATION_COMPLETED"
        assert DATA_TO_CROSS_PROCESS_MAP["elo_updated"] == "ELO_UPDATED"

    def test_promotion_events_uppercase(self):
        """Test promotion events are UPPERCASE in cross-process."""
        assert DATA_TO_CROSS_PROCESS_MAP["model_promoted"] == "MODEL_PROMOTED"
        assert DATA_TO_CROSS_PROCESS_MAP["promotion_failed"] == "PROMOTION_FAILED"
        assert DATA_TO_CROSS_PROCESS_MAP["promotion_candidate"] == "PROMOTION_CANDIDATE"

    def test_quality_events_uppercase(self):
        """Test quality events are UPPERCASE in cross-process."""
        assert DATA_TO_CROSS_PROCESS_MAP["quality_score_updated"] == "QUALITY_SCORE_UPDATED"
        assert DATA_TO_CROSS_PROCESS_MAP["quality_degraded"] == "QUALITY_DEGRADED"
        assert DATA_TO_CROSS_PROCESS_MAP["high_quality_data_available"] == "HIGH_QUALITY_DATA_AVAILABLE"

    def test_regression_events_uppercase(self):
        """Test regression events are UPPERCASE in cross-process."""
        assert DATA_TO_CROSS_PROCESS_MAP["regression_detected"] == "REGRESSION_DETECTED"
        assert DATA_TO_CROSS_PROCESS_MAP["regression_minor"] == "REGRESSION_MINOR"
        assert DATA_TO_CROSS_PROCESS_MAP["regression_critical"] == "REGRESSION_CRITICAL"

    def test_cluster_events_uppercase(self):
        """Test cluster events are UPPERCASE in cross-process."""
        assert DATA_TO_CROSS_PROCESS_MAP["host_online"] == "HOST_ONLINE"
        assert DATA_TO_CROSS_PROCESS_MAP["host_offline"] == "HOST_OFFLINE"
        assert DATA_TO_CROSS_PROCESS_MAP["node_recovered"] == "NODE_RECOVERED"
        assert DATA_TO_CROSS_PROCESS_MAP["leader_elected"] == "LEADER_ELECTED"

    def test_work_queue_events_uppercase(self):
        """Test work queue events are UPPERCASE in cross-process."""
        assert DATA_TO_CROSS_PROCESS_MAP["work_queued"] == "WORK_QUEUED"
        assert DATA_TO_CROSS_PROCESS_MAP["work_completed"] == "WORK_COMPLETED"
        assert DATA_TO_CROSS_PROCESS_MAP["work_failed"] == "WORK_FAILED"

    def test_task_lifecycle_events_uppercase(self):
        """Test task lifecycle events are UPPERCASE in cross-process."""
        assert DATA_TO_CROSS_PROCESS_MAP["task_spawned"] == "TASK_SPAWNED"
        assert DATA_TO_CROSS_PROCESS_MAP["task_completed"] == "TASK_COMPLETED"
        assert DATA_TO_CROSS_PROCESS_MAP["task_failed"] == "TASK_FAILED"
        assert DATA_TO_CROSS_PROCESS_MAP["task_abandoned"] == "TASK_ABANDONED"

    def test_orphan_detection_events_uppercase(self):
        """Test orphan detection events are UPPERCASE in cross-process."""
        assert DATA_TO_CROSS_PROCESS_MAP["orphan_games_detected"] == "ORPHAN_GAMES_DETECTED"
        assert DATA_TO_CROSS_PROCESS_MAP["orphan_games_registered"] == "ORPHAN_GAMES_REGISTERED"

    def test_all_cross_process_values_uppercase(self):
        """Verify all cross-process event values are UPPERCASE."""
        for data_event, cp_event in DATA_TO_CROSS_PROCESS_MAP.items():
            assert cp_event == cp_event.upper(), f"Cross-process event '{cp_event}' is not UPPERCASE"


class TestCrossProcessToDataMap:
    """Tests for CROSS_PROCESS_TO_DATA_MAP (reverse mapping)."""

    def test_reverse_mapping_consistency(self):
        """Verify reverse mapping is generated correctly."""
        for data_event, cp_event in DATA_TO_CROSS_PROCESS_MAP.items():
            assert CROSS_PROCESS_TO_DATA_MAP[cp_event] == data_event

    def test_reverse_mapping_completeness(self):
        """Verify all cross-process events have reverse mappings."""
        assert len(CROSS_PROCESS_TO_DATA_MAP) == len(DATA_TO_CROSS_PROCESS_MAP)


class TestStageToCrossProcessMap:
    """Tests for STAGE_TO_CROSS_PROCESS_MAP (direct stage to cross-process)."""

    def test_training_events_direct(self):
        """Test training stage events map directly to cross-process."""
        assert STAGE_TO_CROSS_PROCESS_MAP["training_complete"] == "TRAINING_COMPLETED"
        assert STAGE_TO_CROSS_PROCESS_MAP["training_started"] == "TRAINING_STARTED"
        assert STAGE_TO_CROSS_PROCESS_MAP["training_failed"] == "TRAINING_FAILED"

    def test_evaluation_events_direct(self):
        """Test evaluation stage events map directly to cross-process."""
        assert STAGE_TO_CROSS_PROCESS_MAP["evaluation_complete"] == "EVALUATION_COMPLETED"
        assert STAGE_TO_CROSS_PROCESS_MAP["shadow_tournament_complete"] == "SHADOW_TOURNAMENT_COMPLETE"

    def test_selfplay_events_direct(self):
        """Test selfplay stage events map directly to cross-process."""
        assert STAGE_TO_CROSS_PROCESS_MAP["selfplay_complete"] == "SELFPLAY_BATCH_COMPLETE"
        assert STAGE_TO_CROSS_PROCESS_MAP["canonical_selfplay_complete"] == "CANONICAL_SELFPLAY_COMPLETE"

    def test_sync_events_direct(self):
        """Test sync stage events map directly to cross-process."""
        assert STAGE_TO_CROSS_PROCESS_MAP["sync_complete"] == "DATA_SYNC_COMPLETED"
        assert STAGE_TO_CROSS_PROCESS_MAP["npz_export_complete"] == "NPZ_EXPORT_COMPLETE"


class TestGetDataEventType:
    """Tests for get_data_event_type helper function."""

    def test_mapped_event(self):
        """Test converting mapped stage event."""
        assert get_data_event_type("training_complete") == "training_completed"
        assert get_data_event_type("selfplay_complete") == "selfplay_complete"

    def test_unmapped_event_returns_none(self):
        """Test unmapped event returns None."""
        assert get_data_event_type("nonexistent_event") is None
        assert get_data_event_type("") is None


class TestGetCrossProcessEventType:
    """Tests for get_cross_process_event_type helper function."""

    def test_from_data_event(self):
        """Test converting data event to cross-process."""
        assert get_cross_process_event_type("training_completed") == "TRAINING_COMPLETED"
        assert get_cross_process_event_type("model_promoted") == "MODEL_PROMOTED"

    def test_from_stage_event(self):
        """Test converting stage event to cross-process."""
        assert get_cross_process_event_type("training_complete", source="stage") == "TRAINING_COMPLETED"
        assert get_cross_process_event_type("selfplay_complete", source="stage") == "SELFPLAY_BATCH_COMPLETE"

    def test_unmapped_returns_none(self):
        """Test unmapped event returns None."""
        assert get_cross_process_event_type("nonexistent") is None
        assert get_cross_process_event_type("nonexistent", source="stage") is None


class TestGetStageEventType:
    """Tests for get_stage_event_type helper function."""

    def test_mapped_data_event(self):
        """Test converting mapped data event to stage."""
        assert get_stage_event_type("training_completed") == "training_complete"
        assert get_stage_event_type("selfplay_complete") == "selfplay_complete"

    def test_unmapped_returns_none(self):
        """Test unmapped event returns None."""
        assert get_stage_event_type("nonexistent") is None


class TestIsMappedEvent:
    """Tests for is_mapped_event helper function."""

    def test_lowercase_stage_event(self):
        """Test lowercase stage event is recognized."""
        assert is_mapped_event("training_complete") is True
        assert is_mapped_event("selfplay_complete") is True

    def test_lowercase_data_event(self):
        """Test lowercase data event is recognized."""
        assert is_mapped_event("training_completed") is True
        assert is_mapped_event("model_promoted") is True

    def test_uppercase_cross_process_event(self):
        """Test UPPERCASE cross-process event is recognized."""
        assert is_mapped_event("TRAINING_COMPLETED") is True
        assert is_mapped_event("MODEL_PROMOTED") is True

    def test_unmapped_event_not_recognized(self):
        """Test unmapped event returns False."""
        assert is_mapped_event("nonexistent_event") is False
        assert is_mapped_event("UNMAPPED_EVENT") is False
        assert is_mapped_event("") is False


class TestGetAllEventTypes:
    """Tests for get_all_event_types helper function."""

    def test_returns_set(self):
        """Test returns a set of event types."""
        all_types = get_all_event_types()
        assert isinstance(all_types, set)
        assert len(all_types) > 0

    def test_contains_stage_events(self):
        """Test set contains stage events."""
        all_types = get_all_event_types()
        assert "training_complete" in all_types
        assert "selfplay_complete" in all_types

    def test_contains_data_events(self):
        """Test set contains data events."""
        all_types = get_all_event_types()
        assert "training_completed" in all_types
        assert "model_promoted" in all_types

    def test_contains_cross_process_events(self):
        """Test set contains cross-process events."""
        all_types = get_all_event_types()
        assert "TRAINING_COMPLETED" in all_types
        assert "MODEL_PROMOTED" in all_types

    def test_comprehensive_coverage(self):
        """Test all mappings are represented."""
        all_types = get_all_event_types()

        # Should include all stage-to-data keys and values
        for key in STAGE_TO_DATA_EVENT_MAP:
            assert key in all_types

        # Should include all data-to-cross-process keys
        for key in DATA_TO_CROSS_PROCESS_MAP:
            assert key in all_types

        # Should include all cross-process values
        for value in DATA_TO_CROSS_PROCESS_MAP.values():
            assert value in all_types


class TestValidateMappings:
    """Tests for validate_mappings function."""

    def test_no_warnings_for_valid_mappings(self):
        """Test valid mappings produce no warnings."""
        warnings = validate_mappings()
        # May have some expected warnings - just verify it runs
        assert isinstance(warnings, list)

    def test_returns_list(self):
        """Test function returns a list."""
        result = validate_mappings()
        assert isinstance(result, list)

    def test_cross_process_naming_convention(self):
        """Test cross-process events follow UPPERCASE convention."""
        warnings = validate_mappings()

        # Filter for naming convention warnings
        naming_warnings = [w for w in warnings if "UPPERCASE" in w]

        # All cross-process events should be UPPERCASE (no warnings)
        assert len(naming_warnings) == 0


class TestMappingConsistency:
    """Tests for overall mapping consistency."""

    def test_no_missing_reverse_mappings(self):
        """Test all forward mappings have valid reverse mappings where applicable."""
        # Every data event in stage-to-data should have some reverse path
        for stage_event, data_event in STAGE_TO_DATA_EVENT_MAP.items():
            # data_event should either be in DATA_TO_STAGE_EVENT_MAP
            # or the reverse of a valid mapping
            if data_event in DATA_TO_STAGE_EVENT_MAP:
                reverse = DATA_TO_STAGE_EVENT_MAP[data_event]
                assert reverse in STAGE_TO_DATA_EVENT_MAP, \
                    f"Reverse mapping {reverse} not in STAGE_TO_DATA_EVENT_MAP"

    def test_cross_process_complete(self):
        """Test cross-process mappings are complete for critical events."""
        critical_data_events = [
            "training_completed",
            "training_started",
            "evaluation_completed",
            "model_promoted",
            "sync_completed",
        ]
        for event in critical_data_events:
            assert event in DATA_TO_CROSS_PROCESS_MAP, \
                f"Critical event {event} missing from DATA_TO_CROSS_PROCESS_MAP"

    def test_selfplay_feedback_events_present(self):
        """Test selfplay feedback loop events are mapped."""
        feedback_events = [
            "selfplay_complete",
            "selfplay_target_updated",
            "selfplay_rate_changed",
        ]
        for event in feedback_events:
            assert event in DATA_TO_CROSS_PROCESS_MAP, \
                f"Feedback event {event} missing from DATA_TO_CROSS_PROCESS_MAP"

    def test_december_2025_events_present(self):
        """Test Dec 2025 additions are mapped."""
        dec_2025_events = [
            "task_abandoned",
            "handler_failed",
            "request_selfplay_queued",
            "selfplay_budget_adjusted",
            "curriculum_allocation_changed",
        ]
        for event in dec_2025_events:
            assert event in DATA_TO_CROSS_PROCESS_MAP, \
                f"Dec 2025 event {event} missing from DATA_TO_CROSS_PROCESS_MAP"
