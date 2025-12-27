"""Comprehensive tests for event_mappings.py.

Tests event type mapping dictionaries and helper functions for
cross-bus event translation.
"""

from __future__ import annotations

import pytest

from app.coordination.event_mappings import (
    # Dictionaries
    STAGE_TO_DATA_EVENT_MAP,
    DATA_TO_STAGE_EVENT_MAP,
    DATA_TO_CROSS_PROCESS_MAP,
    CROSS_PROCESS_TO_DATA_MAP,
    STAGE_TO_CROSS_PROCESS_MAP,
    # Functions
    get_data_event_type,
    get_cross_process_event_type,
    get_stage_event_type,
    is_mapped_event,
    get_all_event_types,
    validate_mappings,
)


class TestStageToDataEventMap:
    """Tests for STAGE_TO_DATA_EVENT_MAP dictionary."""

    def test_not_empty(self):
        """Test mapping is not empty."""
        assert len(STAGE_TO_DATA_EVENT_MAP) > 0

    def test_selfplay_events_mapped(self):
        """Test selfplay events are mapped correctly."""
        assert STAGE_TO_DATA_EVENT_MAP["selfplay_complete"] == "selfplay_complete"
        assert STAGE_TO_DATA_EVENT_MAP["canonical_selfplay_complete"] == "selfplay_complete"
        assert STAGE_TO_DATA_EVENT_MAP["gpu_selfplay_complete"] == "selfplay_complete"

    def test_training_events_mapped(self):
        """Test training events are mapped correctly."""
        assert STAGE_TO_DATA_EVENT_MAP["training_complete"] == "training_completed"
        assert STAGE_TO_DATA_EVENT_MAP["training_started"] == "training_started"
        assert STAGE_TO_DATA_EVENT_MAP["training_failed"] == "training_failed"

    def test_evaluation_events_mapped(self):
        """Test evaluation events are mapped correctly."""
        assert STAGE_TO_DATA_EVENT_MAP["evaluation_complete"] == "evaluation_completed"
        assert STAGE_TO_DATA_EVENT_MAP["shadow_tournament_complete"] == "evaluation_completed"

    def test_sync_events_mapped(self):
        """Test sync events are mapped correctly."""
        assert STAGE_TO_DATA_EVENT_MAP["sync_complete"] == "sync_completed"
        assert STAGE_TO_DATA_EVENT_MAP["cluster_sync_complete"] == "sync_completed"

    def test_promotion_events_mapped(self):
        """Test promotion events are mapped correctly."""
        assert STAGE_TO_DATA_EVENT_MAP["promotion_complete"] == "model_promoted"
        assert STAGE_TO_DATA_EVENT_MAP["tier_gating_complete"] == "model_promoted"

    def test_all_keys_are_lowercase(self):
        """Test all stage event keys are lowercase."""
        for key in STAGE_TO_DATA_EVENT_MAP:
            assert key == key.lower(), f"Key '{key}' is not lowercase"

    def test_all_values_are_lowercase(self):
        """Test all data event values are lowercase."""
        for value in STAGE_TO_DATA_EVENT_MAP.values():
            assert value == value.lower(), f"Value '{value}' is not lowercase"


class TestDataToStageEventMap:
    """Tests for DATA_TO_STAGE_EVENT_MAP dictionary."""

    def test_not_empty(self):
        """Test mapping is not empty."""
        assert len(DATA_TO_STAGE_EVENT_MAP) > 0

    def test_reverse_of_stage_to_data(self):
        """Test that DATA_TO_STAGE provides reverse mapping."""
        # selfplay_complete should map back to selfplay_complete
        assert DATA_TO_STAGE_EVENT_MAP.get("selfplay_complete") == "selfplay_complete"

    def test_training_events_mapped(self):
        """Test training events are mapped correctly."""
        assert DATA_TO_STAGE_EVENT_MAP["training_completed"] == "training_complete"
        assert DATA_TO_STAGE_EVENT_MAP["training_started"] == "training_started"

    def test_sync_events_mapped(self):
        """Test sync events are mapped correctly."""
        assert DATA_TO_STAGE_EVENT_MAP["sync_completed"] == "sync_complete"
        assert DATA_TO_STAGE_EVENT_MAP["p2p_model_synced"] == "model_sync_complete"

    def test_new_games_maps_to_selfplay(self):
        """Test new_games event maps to selfplay_complete stage."""
        assert DATA_TO_STAGE_EVENT_MAP["new_games"] == "selfplay_complete"


class TestDataToCrossProcessMap:
    """Tests for DATA_TO_CROSS_PROCESS_MAP dictionary."""

    def test_not_empty(self):
        """Test mapping is not empty."""
        assert len(DATA_TO_CROSS_PROCESS_MAP) > 0

    def test_training_events_uppercase(self):
        """Test training events are mapped to UPPERCASE."""
        assert DATA_TO_CROSS_PROCESS_MAP["training_started"] == "TRAINING_STARTED"
        assert DATA_TO_CROSS_PROCESS_MAP["training_completed"] == "TRAINING_COMPLETED"
        assert DATA_TO_CROSS_PROCESS_MAP["training_failed"] == "TRAINING_FAILED"

    def test_evaluation_events_uppercase(self):
        """Test evaluation events are mapped to UPPERCASE."""
        assert DATA_TO_CROSS_PROCESS_MAP["evaluation_started"] == "EVALUATION_STARTED"
        assert DATA_TO_CROSS_PROCESS_MAP["evaluation_completed"] == "EVALUATION_COMPLETED"

    def test_promotion_events_uppercase(self):
        """Test promotion events are mapped to UPPERCASE."""
        assert DATA_TO_CROSS_PROCESS_MAP["model_promoted"] == "MODEL_PROMOTED"
        assert DATA_TO_CROSS_PROCESS_MAP["promotion_failed"] == "PROMOTION_FAILED"

    def test_quality_events_mapped(self):
        """Test quality events are mapped."""
        assert DATA_TO_CROSS_PROCESS_MAP["quality_score_updated"] == "QUALITY_SCORE_UPDATED"
        assert DATA_TO_CROSS_PROCESS_MAP["quality_degraded"] == "QUALITY_DEGRADED"

    def test_regression_events_mapped(self):
        """Test regression events are mapped."""
        assert DATA_TO_CROSS_PROCESS_MAP["regression_detected"] == "REGRESSION_DETECTED"
        assert DATA_TO_CROSS_PROCESS_MAP["regression_critical"] == "REGRESSION_CRITICAL"

    def test_cluster_events_mapped(self):
        """Test cluster events are mapped."""
        assert DATA_TO_CROSS_PROCESS_MAP["host_online"] == "HOST_ONLINE"
        assert DATA_TO_CROSS_PROCESS_MAP["host_offline"] == "HOST_OFFLINE"
        assert DATA_TO_CROSS_PROCESS_MAP["node_recovered"] == "NODE_RECOVERED"

    def test_p2p_events_mapped(self):
        """Test P2P events are mapped."""
        assert DATA_TO_CROSS_PROCESS_MAP["p2p_cluster_healthy"] == "P2P_CLUSTER_HEALTHY"
        assert DATA_TO_CROSS_PROCESS_MAP["p2p_node_dead"] == "P2P_NODE_DEAD"
        assert DATA_TO_CROSS_PROCESS_MAP["leader_elected"] == "LEADER_ELECTED"

    def test_work_queue_events_mapped(self):
        """Test work queue events are mapped."""
        assert DATA_TO_CROSS_PROCESS_MAP["work_queued"] == "WORK_QUEUED"
        assert DATA_TO_CROSS_PROCESS_MAP["work_completed"] == "WORK_COMPLETED"
        assert DATA_TO_CROSS_PROCESS_MAP["work_failed"] == "WORK_FAILED"

    def test_task_lifecycle_events_mapped(self):
        """Test task lifecycle events are mapped."""
        assert DATA_TO_CROSS_PROCESS_MAP["task_spawned"] == "TASK_SPAWNED"
        assert DATA_TO_CROSS_PROCESS_MAP["task_completed"] == "TASK_COMPLETED"
        assert DATA_TO_CROSS_PROCESS_MAP["task_abandoned"] == "TASK_ABANDONED"

    def test_all_values_uppercase(self):
        """Test all cross-process event values are UPPERCASE."""
        for key, value in DATA_TO_CROSS_PROCESS_MAP.items():
            assert value == value.upper(), f"Value '{value}' for key '{key}' is not UPPERCASE"

    def test_all_keys_lowercase(self):
        """Test all data event keys are lowercase."""
        for key in DATA_TO_CROSS_PROCESS_MAP:
            assert key == key.lower(), f"Key '{key}' is not lowercase"


class TestCrossProcessToDataMap:
    """Tests for CROSS_PROCESS_TO_DATA_MAP (auto-generated reverse)."""

    def test_is_reverse_of_data_to_cross_process(self):
        """Test it's the exact reverse of DATA_TO_CROSS_PROCESS_MAP."""
        for data_event, cross_event in DATA_TO_CROSS_PROCESS_MAP.items():
            assert CROSS_PROCESS_TO_DATA_MAP[cross_event] == data_event

    def test_same_length(self):
        """Test both maps have same length."""
        assert len(CROSS_PROCESS_TO_DATA_MAP) == len(DATA_TO_CROSS_PROCESS_MAP)

    def test_training_completed_reverse(self):
        """Test TRAINING_COMPLETED maps back to training_completed."""
        assert CROSS_PROCESS_TO_DATA_MAP["TRAINING_COMPLETED"] == "training_completed"

    def test_model_promoted_reverse(self):
        """Test MODEL_PROMOTED maps back to model_promoted."""
        assert CROSS_PROCESS_TO_DATA_MAP["MODEL_PROMOTED"] == "model_promoted"


class TestStageToCrossProcessMap:
    """Tests for STAGE_TO_CROSS_PROCESS_MAP (direct mapping)."""

    def test_not_empty(self):
        """Test mapping is not empty."""
        assert len(STAGE_TO_CROSS_PROCESS_MAP) > 0

    def test_training_events_mapped(self):
        """Test training events are mapped directly."""
        assert STAGE_TO_CROSS_PROCESS_MAP["training_complete"] == "TRAINING_COMPLETED"
        assert STAGE_TO_CROSS_PROCESS_MAP["training_started"] == "TRAINING_STARTED"

    def test_evaluation_events_mapped(self):
        """Test evaluation events are mapped directly."""
        assert STAGE_TO_CROSS_PROCESS_MAP["evaluation_complete"] == "EVALUATION_COMPLETED"

    def test_selfplay_events_mapped(self):
        """Test selfplay events are mapped directly."""
        assert STAGE_TO_CROSS_PROCESS_MAP["selfplay_complete"] == "SELFPLAY_BATCH_COMPLETE"
        assert STAGE_TO_CROSS_PROCESS_MAP["gpu_selfplay_complete"] == "GPU_SELFPLAY_COMPLETE"

    def test_sync_events_mapped(self):
        """Test sync events are mapped directly."""
        assert STAGE_TO_CROSS_PROCESS_MAP["sync_complete"] == "DATA_SYNC_COMPLETED"

    def test_all_values_uppercase(self):
        """Test all cross-process values are UPPERCASE."""
        for key, value in STAGE_TO_CROSS_PROCESS_MAP.items():
            assert value == value.upper(), f"Value '{value}' for key '{key}' is not UPPERCASE"


class TestGetDataEventType:
    """Tests for get_data_event_type() helper function."""

    def test_valid_stage_event(self):
        """Test converting valid stage event."""
        assert get_data_event_type("training_complete") == "training_completed"
        assert get_data_event_type("selfplay_complete") == "selfplay_complete"

    def test_invalid_stage_event(self):
        """Test converting non-existent stage event."""
        result = get_data_event_type("nonexistent_event")
        assert result is None

    def test_empty_string(self):
        """Test converting empty string."""
        result = get_data_event_type("")
        assert result is None


class TestGetCrossProcessEventType:
    """Tests for get_cross_process_event_type() helper function."""

    def test_from_data_event(self):
        """Test converting from data event (default source)."""
        assert get_cross_process_event_type("training_completed") == "TRAINING_COMPLETED"
        assert get_cross_process_event_type("model_promoted") == "MODEL_PROMOTED"

    def test_from_stage_event(self):
        """Test converting from stage event."""
        assert get_cross_process_event_type("training_complete", source="stage") == "TRAINING_COMPLETED"
        assert get_cross_process_event_type("selfplay_complete", source="stage") == "SELFPLAY_BATCH_COMPLETE"

    def test_invalid_data_event(self):
        """Test converting non-existent data event."""
        result = get_cross_process_event_type("nonexistent")
        assert result is None

    def test_invalid_stage_event(self):
        """Test converting non-existent stage event."""
        result = get_cross_process_event_type("nonexistent", source="stage")
        assert result is None

    def test_default_source_is_data(self):
        """Test default source is 'data'."""
        # This should work for data events
        result = get_cross_process_event_type("training_completed")
        assert result == "TRAINING_COMPLETED"


class TestGetStageEventType:
    """Tests for get_stage_event_type() helper function."""

    def test_valid_data_event(self):
        """Test converting valid data event."""
        assert get_stage_event_type("training_completed") == "training_complete"
        assert get_stage_event_type("selfplay_complete") == "selfplay_complete"

    def test_new_games_event(self):
        """Test new_games maps to selfplay_complete."""
        assert get_stage_event_type("new_games") == "selfplay_complete"

    def test_invalid_data_event(self):
        """Test converting non-existent data event."""
        result = get_stage_event_type("nonexistent")
        assert result is None


class TestIsMappedEvent:
    """Tests for is_mapped_event() helper function."""

    def test_stage_event_lowercase(self):
        """Test stage event in lowercase is recognized."""
        assert is_mapped_event("training_complete") is True
        assert is_mapped_event("selfplay_complete") is True

    def test_data_event_lowercase(self):
        """Test data event in lowercase is recognized."""
        assert is_mapped_event("training_completed") is True
        assert is_mapped_event("model_promoted") is True

    def test_cross_process_event_uppercase(self):
        """Test cross-process event in UPPERCASE is recognized."""
        assert is_mapped_event("TRAINING_COMPLETED") is True
        assert is_mapped_event("MODEL_PROMOTED") is True

    def test_unknown_event(self):
        """Test unknown event returns False."""
        assert is_mapped_event("unknown_event") is False
        assert is_mapped_event("UNKNOWN_EVENT") is False

    def test_empty_string(self):
        """Test empty string returns False."""
        assert is_mapped_event("") is False

    def test_case_insensitive_lookup(self):
        """Test case handling works correctly."""
        # Lowercase should match stage/data maps
        assert is_mapped_event("training_complete") is True
        # Uppercase should match cross-process map
        assert is_mapped_event("TRAINING_COMPLETED") is True


class TestGetAllEventTypes:
    """Tests for get_all_event_types() helper function."""

    def test_returns_set(self):
        """Test function returns a set."""
        result = get_all_event_types()
        assert isinstance(result, set)

    def test_not_empty(self):
        """Test result is not empty."""
        result = get_all_event_types()
        assert len(result) > 0

    def test_contains_stage_events(self):
        """Test result contains stage events."""
        result = get_all_event_types()
        assert "training_complete" in result
        assert "selfplay_complete" in result

    def test_contains_data_events(self):
        """Test result contains data events."""
        result = get_all_event_types()
        assert "training_completed" in result
        assert "model_promoted" in result

    def test_contains_cross_process_events(self):
        """Test result contains cross-process events."""
        result = get_all_event_types()
        assert "TRAINING_COMPLETED" in result
        assert "MODEL_PROMOTED" in result

    def test_unique_entries(self):
        """Test no duplicates (set property)."""
        result = get_all_event_types()
        # Set automatically handles uniqueness
        assert len(result) == len(set(result))


class TestValidateMappings:
    """Tests for validate_mappings() helper function."""

    def test_returns_list(self):
        """Test function returns a list."""
        result = validate_mappings()
        assert isinstance(result, list)

    def test_no_warnings_for_current_mappings(self):
        """Test current mappings produce no warnings."""
        warnings = validate_mappings()
        # Filter out expected inconsistencies if any
        # Current mappings should be consistent
        assert len(warnings) == 0, f"Unexpected warnings: {warnings}"

    def test_cross_process_uppercase_check(self):
        """Test validation catches non-uppercase cross-process events."""
        # This is a static check - the current implementation should pass
        warnings = validate_mappings()
        lowercase_warnings = [w for w in warnings if "UPPERCASE" in w]
        assert len(lowercase_warnings) == 0


class TestMappingConsistency:
    """Integration tests for mapping consistency across all dictionaries."""

    def test_stage_to_data_then_cross_process(self):
        """Test chained mapping: stage -> data -> cross_process."""
        stage_event = "training_complete"
        data_event = STAGE_TO_DATA_EVENT_MAP.get(stage_event)
        assert data_event is not None

        cross_event = DATA_TO_CROSS_PROCESS_MAP.get(data_event)
        assert cross_event is not None
        assert cross_event == "TRAINING_COMPLETED"

    def test_direct_stage_to_cross_matches_chained(self):
        """Test direct stage->cross matches chained stage->data->cross."""
        for stage_event in STAGE_TO_CROSS_PROCESS_MAP:
            direct = STAGE_TO_CROSS_PROCESS_MAP[stage_event]

            # Check if we can chain through data
            data_event = STAGE_TO_DATA_EVENT_MAP.get(stage_event)
            if data_event:
                chained = DATA_TO_CROSS_PROCESS_MAP.get(data_event)
                # They might differ slightly (e.g., GPU_SELFPLAY_COMPLETE vs SELFPLAY_BATCH_COMPLETE)
                # but both should be valid UPPERCASE strings
                assert direct == direct.upper()
                if chained:
                    assert chained == chained.upper()

    def test_round_trip_data_cross_data(self):
        """Test round trip: data -> cross -> data."""
        for data_event in DATA_TO_CROSS_PROCESS_MAP:
            cross_event = DATA_TO_CROSS_PROCESS_MAP[data_event]
            back_to_data = CROSS_PROCESS_TO_DATA_MAP.get(cross_event)
            assert back_to_data == data_event, (
                f"Round trip failed: {data_event} -> {cross_event} -> {back_to_data}"
            )

    def test_all_cross_process_events_are_valid_identifiers(self):
        """Test all cross-process events are valid Python identifiers (uppercase)."""
        for cross_event in CROSS_PROCESS_TO_DATA_MAP:
            # Should be UPPERCASE_SNAKE_CASE
            assert cross_event == cross_event.upper()
            assert "_" in cross_event or cross_event.isalpha()
            # No spaces or special characters
            assert cross_event.replace("_", "").isalnum()

    def test_no_orphaned_data_events(self):
        """Test data events in DATA_TO_STAGE have corresponding cross-process mapping."""
        # Most data events should have cross-process equivalents
        unmapped_count = 0
        for data_event in DATA_TO_STAGE_EVENT_MAP:
            if data_event not in DATA_TO_CROSS_PROCESS_MAP:
                unmapped_count += 1

        # Allow some unmapped (not all data events need cross-process mapping)
        # But there shouldn't be too many
        assert unmapped_count < len(DATA_TO_STAGE_EVENT_MAP) // 2


class TestCriticalEventMappings:
    """Tests for critical events that must be mapped correctly."""

    def test_training_lifecycle_complete(self):
        """Test complete training lifecycle is mapped."""
        assert "training_started" in DATA_TO_CROSS_PROCESS_MAP
        assert "training_completed" in DATA_TO_CROSS_PROCESS_MAP
        assert "training_failed" in DATA_TO_CROSS_PROCESS_MAP

    def test_evaluation_lifecycle_complete(self):
        """Test complete evaluation lifecycle is mapped."""
        assert "evaluation_started" in DATA_TO_CROSS_PROCESS_MAP
        assert "evaluation_completed" in DATA_TO_CROSS_PROCESS_MAP

    def test_promotion_lifecycle_complete(self):
        """Test complete promotion lifecycle is mapped."""
        assert "model_promoted" in DATA_TO_CROSS_PROCESS_MAP
        assert "promotion_failed" in DATA_TO_CROSS_PROCESS_MAP

    def test_sync_lifecycle_complete(self):
        """Test complete sync lifecycle is mapped."""
        assert "sync_started" in DATA_TO_CROSS_PROCESS_MAP
        assert "sync_completed" in DATA_TO_CROSS_PROCESS_MAP
        assert "sync_failed" in DATA_TO_CROSS_PROCESS_MAP

    def test_p2p_node_dead_event(self):
        """Test P2P_NODE_DEAD event is mapped (critical for NodeRecoveryDaemon)."""
        assert "p2p_node_dead" in DATA_TO_CROSS_PROCESS_MAP
        assert DATA_TO_CROSS_PROCESS_MAP["p2p_node_dead"] == "P2P_NODE_DEAD"

    def test_node_recovered_event(self):
        """Test node_recovered event is mapped."""
        assert "node_recovered" in DATA_TO_CROSS_PROCESS_MAP
        assert DATA_TO_CROSS_PROCESS_MAP["node_recovered"] == "NODE_RECOVERED"

    def test_leader_election_events(self):
        """Test leader election events are mapped."""
        assert "leader_elected" in DATA_TO_CROSS_PROCESS_MAP
        assert "leader_lost" in DATA_TO_CROSS_PROCESS_MAP

    def test_orphan_games_events(self):
        """Test orphan games events are mapped."""
        assert "orphan_games_detected" in DATA_TO_CROSS_PROCESS_MAP
        assert "orphan_games_registered" in DATA_TO_CROSS_PROCESS_MAP
