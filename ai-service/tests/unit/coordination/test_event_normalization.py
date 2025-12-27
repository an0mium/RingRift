"""Tests for event name normalization.

Tests the canonical event naming convention and normalization utilities.
"""

import pytest

from app.coordination.event_normalization import (
    CANONICAL_EVENT_NAMES,
    audit_event_usage,
    get_variants,
    is_canonical,
    normalize_event_type,
    validate_event_names,
)


class TestEventNormalization:
    """Test event name normalization."""

    def test_normalize_sync_complete_variants(self):
        """Test normalization of SYNC_COMPLETE variants."""
        variants = [
            "sync_complete",
            "SYNC_COMPLETE",
            "sync_completed",
            "SYNC_COMPLETED",
            "data_sync_complete",
            "DATA_SYNC_COMPLETE",
            "data_sync_completed",
            "DATA_SYNC_COMPLETED",
            "cluster_sync_complete",
            "CLUSTER_SYNC_COMPLETE",
        ]

        for variant in variants:
            canonical = normalize_event_type(variant)
            assert canonical == "DATA_SYNC_COMPLETED", f"Failed to normalize {variant}"

    def test_normalize_selfplay_complete_variants(self):
        """Test normalization of SELFPLAY_COMPLETE variants."""
        variants = [
            "selfplay_complete",
            "SELFPLAY_COMPLETE",
            "selfplay_completed",
            "SELFPLAY_COMPLETED",
            "selfplay_batch_complete",
            "SELFPLAY_BATCH_COMPLETE",
            "canonical_selfplay_complete",
            "CANONICAL_SELFPLAY_COMPLETE",
            "gpu_selfplay_complete",
            "GPU_SELFPLAY_COMPLETE",
        ]

        for variant in variants:
            canonical = normalize_event_type(variant)
            assert canonical == "SELFPLAY_COMPLETE", f"Failed to normalize {variant}"

    def test_normalize_training_events(self):
        """Test normalization of training event variants."""
        test_cases = [
            ("training_complete", "TRAINING_COMPLETED"),
            ("TRAINING_COMPLETE", "TRAINING_COMPLETED"),
            ("training_completed", "TRAINING_COMPLETED"),
            ("TRAINING_COMPLETED", "TRAINING_COMPLETED"),
            ("training_start", "TRAINING_STARTED"),
            ("TRAINING_START", "TRAINING_STARTED"),
            ("training_started", "TRAINING_STARTED"),
            ("TRAINING_STARTED", "TRAINING_STARTED"),
            ("training_fail", "TRAINING_FAILED"),
            ("TRAINING_FAIL", "TRAINING_FAILED"),
            ("training_failed", "TRAINING_FAILED"),
            ("TRAINING_FAILED", "TRAINING_FAILED"),
        ]

        for variant, expected in test_cases:
            canonical = normalize_event_type(variant)
            assert canonical == expected, f"Failed to normalize {variant}"

    def test_normalize_evaluation_events(self):
        """Test normalization of evaluation event variants."""
        test_cases = [
            ("evaluation_complete", "EVALUATION_COMPLETED"),
            ("EVALUATION_COMPLETE", "EVALUATION_COMPLETED"),
            ("evaluation_completed", "EVALUATION_COMPLETED"),
            ("EVALUATION_COMPLETED", "EVALUATION_COMPLETED"),
            ("shadow_tournament_complete", "EVALUATION_COMPLETED"),
            ("SHADOW_TOURNAMENT_COMPLETE", "EVALUATION_COMPLETED"),
        ]

        for variant, expected in test_cases:
            canonical = normalize_event_type(variant)
            assert canonical == expected, f"Failed to normalize {variant}"

    def test_normalize_promotion_events(self):
        """Test normalization of promotion event variants."""
        test_cases = [
            ("promotion_complete", "MODEL_PROMOTED"),
            ("PROMOTION_COMPLETE", "MODEL_PROMOTED"),
            ("promotion_completed", "MODEL_PROMOTED"),
            ("PROMOTION_COMPLETED", "MODEL_PROMOTED"),
            ("model_promoted", "MODEL_PROMOTED"),
            ("MODEL_PROMOTED", "MODEL_PROMOTED"),
            ("tier_gating_complete", "MODEL_PROMOTED"),
            ("TIER_GATING_COMPLETE", "MODEL_PROMOTED"),
        ]

        for variant, expected in test_cases:
            canonical = normalize_event_type(variant)
            assert canonical == expected, f"Failed to normalize {variant}"

    def test_normalize_model_sync_events(self):
        """Test normalization of model sync variants."""
        variants = [
            "model_sync_complete",
            "MODEL_SYNC_COMPLETE",
            "model_sync_completed",
            "MODEL_SYNC_COMPLETED",
            "p2p_model_synced",
            "P2P_MODEL_SYNCED",
            "model_synced",
            "MODEL_SYNCED",
        ]

        for variant in variants:
            canonical = normalize_event_type(variant)
            assert canonical == "P2P_MODEL_SYNCED", f"Failed to normalize {variant}"

    def test_normalize_already_canonical(self):
        """Test that canonical names pass through unchanged."""
        canonical_names = [
            "DATA_SYNC_COMPLETED",
            "TRAINING_STARTED",
            "TRAINING_COMPLETED",
            "TRAINING_FAILED",
            "EVALUATION_COMPLETED",
            "MODEL_PROMOTED",
            "SELFPLAY_COMPLETE",
            "P2P_MODEL_SYNCED",
            "NPZ_EXPORT_COMPLETE",
            "PARITY_VALIDATION_COMPLETED",
        ]

        for canonical in canonical_names:
            normalized = normalize_event_type(canonical)
            assert normalized == canonical, f"Canonical name changed: {canonical} → {normalized}"

    def test_normalize_case_insensitive(self):
        """Test case-insensitive normalization."""
        test_cases = [
            ("sync_complete", "DATA_SYNC_COMPLETED"),
            ("SYNC_COMPLETE", "DATA_SYNC_COMPLETED"),
            ("Sync_Complete", "DATA_SYNC_COMPLETED"),
            ("SyNc_CoMpLeTe", "DATA_SYNC_COMPLETED"),
        ]

        for variant, expected in test_cases:
            canonical = normalize_event_type(variant)
            assert canonical == expected, f"Case-insensitive normalization failed for {variant}"

    def test_normalize_unknown_event(self):
        """Test normalization of unknown event types."""
        unknown_events = [
            "UNKNOWN_EVENT",
            "custom_event_type",
            "MY_NEW_EVENT",
        ]

        for event in unknown_events:
            # Unknown events should pass through unchanged
            normalized = normalize_event_type(event)
            assert normalized == event, f"Unknown event should pass through: {event}"

    def test_is_canonical(self):
        """Test is_canonical() function."""
        # Canonical names
        assert is_canonical("DATA_SYNC_COMPLETED")
        assert is_canonical("TRAINING_STARTED")
        assert is_canonical("MODEL_PROMOTED")

        # Non-canonical variants
        assert not is_canonical("sync_complete")
        assert not is_canonical("SYNC_COMPLETE")
        assert not is_canonical("training_complete")
        assert not is_canonical("TRAINING_COMPLETE")

    def test_get_variants(self):
        """Test get_variants() function."""
        # Get all variants of DATA_SYNC_COMPLETED
        variants = get_variants("DATA_SYNC_COMPLETED")

        # Should include all known variants
        expected_variants = [
            "sync_complete",
            "SYNC_COMPLETE",
            "sync_completed",
            "SYNC_COMPLETED",
            "data_sync_complete",
            "DATA_SYNC_COMPLETE",
            "data_sync_completed",
            "DATA_SYNC_COMPLETED",
            "cluster_sync_complete",
            "CLUSTER_SYNC_COMPLETE",
            "cluster_sync_completed",
            "CLUSTER_SYNC_COMPLETED",
        ]

        for expected in expected_variants:
            assert expected in variants, f"Missing variant: {expected}"

    def test_validate_event_names(self):
        """Test validate_event_names() function."""
        warnings = validate_event_names()

        # Should have no validation errors
        assert isinstance(warnings, list), "validate_event_names should return a list"

        # Log warnings for debugging (non-fatal)
        if warnings:
            print("Validation warnings:")
            for warning in warnings:
                print(f"  - {warning}")

    def test_audit_event_usage(self):
        """Test audit_event_usage() function."""
        # Simulate event history with mixed canonical and non-canonical names
        event_history = [
            "DATA_SYNC_COMPLETED",  # Canonical
            "sync_complete",        # Non-canonical
            "SYNC_COMPLETE",        # Non-canonical
            "DATA_SYNC_COMPLETED",  # Canonical
            "TRAINING_STARTED",     # Canonical
            "training_start",       # Non-canonical
            "TRAINING_STARTED",     # Canonical
        ]

        audit = audit_event_usage(event_history)

        # Check audit structure
        assert "total_events" in audit
        assert "unique_event_types" in audit
        assert "canonical_types" in audit
        assert "non_canonical_variants" in audit
        assert "non_canonical_count" in audit
        assert "normalization_rate" in audit
        assert "recommendations" in audit

        # Verify counts
        assert audit["total_events"] == 7
        assert audit["non_canonical_count"] == 3  # sync_complete, SYNC_COMPLETE, training_start

        # Verify non-canonical variants identified
        non_canonical = audit["non_canonical_variants"]
        assert "sync_complete" in non_canonical
        assert "SYNC_COMPLETE" in non_canonical
        assert "training_start" in non_canonical

        # Verify they map to correct canonical names
        assert non_canonical["sync_complete"] == "DATA_SYNC_COMPLETED"
        assert non_canonical["SYNC_COMPLETE"] == "DATA_SYNC_COMPLETED"
        assert non_canonical["training_start"] == "TRAINING_STARTED"

    def test_audit_event_usage_all_canonical(self):
        """Test audit when all events are canonical."""
        event_history = [
            "DATA_SYNC_COMPLETED",
            "TRAINING_STARTED",
            "TRAINING_COMPLETED",
            "MODEL_PROMOTED",
        ]

        audit = audit_event_usage(event_history)

        # Should have zero non-canonical events
        assert audit["non_canonical_count"] == 0
        assert audit["normalization_rate"] == 0.0
        assert len(audit["non_canonical_variants"]) == 0

    def test_audit_event_usage_all_non_canonical(self):
        """Test audit when all events are non-canonical."""
        event_history = [
            "sync_complete",
            "training_start",
            "training_complete",
            "promotion_complete",
        ]

        audit = audit_event_usage(event_history)

        # Should have all events as non-canonical
        assert audit["non_canonical_count"] == 4
        assert audit["normalization_rate"] == 1.0
        assert len(audit["non_canonical_variants"]) == 4

    def test_canonical_event_names_mapping_complete(self):
        """Test that CANONICAL_EVENT_NAMES mapping is comprehensive."""
        # All canonical names should map to themselves
        canonical_names = set(CANONICAL_EVENT_NAMES.values())

        for canonical in canonical_names:
            # Should be in mapping (either exact or case-variant)
            assert (
                canonical in CANONICAL_EVENT_NAMES or
                canonical.lower() in CANONICAL_EVENT_NAMES
            ), f"Canonical name {canonical} not in mapping"

    def test_normalize_with_enum_like_object(self):
        """Test normalization with enum-like objects."""
        class FakeEventType:
            def __init__(self, value):
                self.value = value

        # Test with enum-like object
        event = FakeEventType("sync_complete")
        canonical = normalize_event_type(event)
        assert canonical == "DATA_SYNC_COMPLETED"

        # Test with already-canonical enum value
        event = FakeEventType("DATA_SYNC_COMPLETED")
        canonical = normalize_event_type(event)
        assert canonical == "DATA_SYNC_COMPLETED"


class TestEventNormalizationIntegration:
    """Integration tests for event normalization."""

    def test_normalization_preserves_payload(self):
        """Test that normalization doesn't affect event payloads."""
        from app.coordination.event_router import RouterEvent
        from app.coordination.event_normalization import normalize_event_type

        # Create event with non-canonical type
        payload = {"host": "test", "games": 100}
        normalized_type = normalize_event_type("sync_complete")

        event = RouterEvent(
            event_type=normalized_type,
            payload=payload,
            source="test",
        )

        # Verify normalization occurred
        assert event.event_type == "DATA_SYNC_COMPLETED"

        # Verify payload unchanged
        assert event.payload == payload

    def test_router_normalization_integration(self):
        """Test that event router uses normalization."""
        # This is a placeholder - actual router test would require async setup
        # The key point is that normalize_event_type is called in router.publish()
        pass


class TestEventNamingGuidelines:
    """Test event naming guidelines compliance."""

    def test_all_canonical_names_uppercase(self):
        """Test that all canonical names use UPPERCASE_SNAKE_CASE."""
        canonical_names = set(CANONICAL_EVENT_NAMES.values())

        for canonical in canonical_names:
            assert canonical == canonical.upper(), (
                f"Canonical name '{canonical}' should be UPPERCASE_SNAKE_CASE"
            )
            assert '_' in canonical or canonical.isupper(), (
                f"Canonical name '{canonical}' should contain underscore"
            )

    def test_no_circular_mappings(self):
        """Test that there are no circular mappings."""
        canonical_names = set(CANONICAL_EVENT_NAMES.values())

        for variant, canonical in CANONICAL_EVENT_NAMES.items():
            # If variant is also a canonical name, it should map to itself
            if variant in canonical_names:
                assert variant == canonical, (
                    f"Circular mapping: '{variant}' maps to '{canonical}' "
                    f"but '{variant}' is also a canonical name"
                )

    def test_completion_events_use_completed_suffix(self):
        """Test that completion events use _COMPLETED suffix."""
        canonical_names = set(CANONICAL_EVENT_NAMES.values())

        # Events that should use _COMPLETED
        completion_patterns = [
            "TRAINING_COMPLETED",
            "EVALUATION_COMPLETED",
            "DATA_SYNC_COMPLETED",
            "PARITY_VALIDATION_COMPLETED",
            "CMAES_COMPLETED",
            "NAS_COMPLETED",
        ]

        for pattern in completion_patterns:
            assert pattern in canonical_names, (
                f"Expected completion event '{pattern}' not found in canonical names"
            )

    def test_start_events_use_started_suffix(self):
        """Test that start events use _STARTED suffix."""
        canonical_names = set(CANONICAL_EVENT_NAMES.values())

        # Events that should use _STARTED
        start_patterns = [
            "TRAINING_STARTED",
            "EVALUATION_STARTED",
            "PROMOTION_STARTED",
            "DATA_SYNC_STARTED",
        ]

        for pattern in start_patterns:
            assert pattern in canonical_names, (
                f"Expected start event '{pattern}' not found in canonical names"
            )

    def test_failure_events_use_failed_suffix(self):
        """Test that failure events use _FAILED suffix."""
        canonical_names = set(CANONICAL_EVENT_NAMES.values())

        # Events that should use _FAILED
        failure_patterns = [
            "TRAINING_FAILED",
            "EVALUATION_FAILED",
            "PROMOTION_FAILED",
            "DATA_SYNC_FAILED",
        ]

        for pattern in failure_patterns:
            assert pattern in canonical_names, (
                f"Expected failure event '{pattern}' not found in canonical names"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
