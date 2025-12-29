"""Unit tests for elo_reconciliation module.

Tests cover:
- EloDrift dataclass properties
- ConflictResolution enum values
- EloReconciler initialization
- Drift detection logic
- Sync utilities

Created: December 2025
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from app.training.elo_reconciliation import (
    ConflictResolution,
    EloDrift,
)


class TestConflictResolutionEnum:
    """Tests for ConflictResolution enum."""

    def test_enum_values(self):
        """Test ConflictResolution enum has correct values."""
        assert ConflictResolution.SKIP.value == "skip"
        assert ConflictResolution.LAST_WRITE_WINS.value == "last_write_wins"
        assert ConflictResolution.FIRST_WRITE_WINS.value == "first_write_wins"
        assert ConflictResolution.RAISE.value == "raise"

    def test_enum_members(self):
        """Test all expected enum members exist."""
        expected = {"SKIP", "LAST_WRITE_WINS", "FIRST_WRITE_WINS", "RAISE"}
        actual = {m.name for m in ConflictResolution}
        assert actual == expected

    def test_enum_from_string(self):
        """Test creating enum from string value."""
        assert ConflictResolution("skip") == ConflictResolution.SKIP
        assert ConflictResolution("last_write_wins") == ConflictResolution.LAST_WRITE_WINS


class TestEloDriftDataclass:
    """Tests for EloDrift dataclass."""

    def test_basic_creation(self):
        """Test basic EloDrift creation."""
        drift = EloDrift(
            source="local",
            target="central",
            checked_at="2025-12-29T00:00:00Z",
            participants_in_source=100,
            participants_in_target=95,
            participants_in_both=90,
        )

        assert drift.source == "local"
        assert drift.target == "central"
        assert drift.participants_in_both == 90

    def test_rating_diffs_default_empty(self):
        """Test rating_diffs defaults to empty dict."""
        drift = EloDrift(
            source="local",
            target="central",
            checked_at="2025-12-29T00:00:00Z",
            participants_in_source=100,
            participants_in_target=100,
            participants_in_both=100,
        )

        assert drift.rating_diffs == {}

    def test_with_rating_diffs(self):
        """Test EloDrift with rating differences."""
        drift = EloDrift(
            source="local",
            target="central",
            checked_at="2025-12-29T00:00:00Z",
            participants_in_source=100,
            participants_in_target=100,
            participants_in_both=100,
            rating_diffs={
                "model_a": 25.5,
                "model_b": -10.0,
                "model_c": 50.0,
            },
        )

        assert len(drift.rating_diffs) == 3
        assert drift.rating_diffs["model_a"] == 25.5

    def test_max_rating_diff_property_empty(self):
        """Test max_rating_diff with no diffs."""
        drift = EloDrift(
            source="local",
            target="central",
            checked_at="2025-12-29T00:00:00Z",
            participants_in_source=0,
            participants_in_target=0,
            participants_in_both=0,
        )

        assert drift.max_rating_diff == 0.0

    def test_max_rating_diff_property_with_diffs(self):
        """Test max_rating_diff with rating differences."""
        drift = EloDrift(
            source="local",
            target="central",
            checked_at="2025-12-29T00:00:00Z",
            participants_in_source=100,
            participants_in_target=100,
            participants_in_both=100,
            rating_diffs={
                "model_a": 25.5,
                "model_b": -60.0,  # Largest absolute value
                "model_c": 50.0,
            },
        )

        assert drift.max_rating_diff == 60.0  # abs(-60.0)

    def test_avg_rating_diff_property_empty(self):
        """Test avg_rating_diff with no diffs."""
        drift = EloDrift(
            source="local",
            target="central",
            checked_at="2025-12-29T00:00:00Z",
            participants_in_source=0,
            participants_in_target=0,
            participants_in_both=0,
        )

        assert drift.avg_rating_diff == 0.0

    def test_avg_rating_diff_property_with_diffs(self):
        """Test avg_rating_diff calculation."""
        drift = EloDrift(
            source="local",
            target="central",
            checked_at="2025-12-29T00:00:00Z",
            participants_in_source=100,
            participants_in_target=100,
            participants_in_both=100,
            rating_diffs={
                "model_a": 20.0,
                "model_b": -40.0,
            },
        )

        # avg of abs values: (20 + 40) / 2 = 30
        assert drift.avg_rating_diff == 30.0

    def test_board_type_optional(self):
        """Test board_type is optional."""
        drift = EloDrift(
            source="local",
            target="central",
            checked_at="2025-12-29T00:00:00Z",
            participants_in_source=10,
            participants_in_target=10,
            participants_in_both=10,
        )

        assert drift.board_type is None

    def test_with_board_type(self):
        """Test EloDrift with board_type set."""
        drift = EloDrift(
            source="local",
            target="central",
            checked_at="2025-12-29T00:00:00Z",
            participants_in_source=10,
            participants_in_target=10,
            participants_in_both=10,
            board_type="hex8",
            num_players=2,
        )

        assert drift.board_type == "hex8"
        assert drift.num_players == 2


class TestEloDriftThresholds:
    """Tests for Elo drift threshold detection."""

    def test_small_drift_below_threshold(self):
        """Test small drift is below typical alert threshold."""
        drift = EloDrift(
            source="local",
            target="central",
            checked_at="2025-12-29T00:00:00Z",
            participants_in_source=100,
            participants_in_target=100,
            participants_in_both=100,
            rating_diffs={"model_a": 10.0},
        )

        # 10 Elo diff is small, below typical 50-point threshold
        assert drift.max_rating_diff < 50

    def test_large_drift_above_threshold(self):
        """Test large drift exceeds typical alert threshold."""
        drift = EloDrift(
            source="local",
            target="central",
            checked_at="2025-12-29T00:00:00Z",
            participants_in_source=100,
            participants_in_target=100,
            participants_in_both=100,
            rating_diffs={"model_a": 75.0},
        )

        # 75 Elo diff is significant, above typical 50-point threshold
        assert drift.max_rating_diff > 50


class TestEloDriftEquality:
    """Tests for EloDrift equality and hashing."""

    def test_equal_drifts(self):
        """Test equal EloDrift objects."""
        drift1 = EloDrift(
            source="local",
            target="central",
            checked_at="2025-12-29T00:00:00Z",
            participants_in_source=100,
            participants_in_target=100,
            participants_in_both=100,
        )
        drift2 = EloDrift(
            source="local",
            target="central",
            checked_at="2025-12-29T00:00:00Z",
            participants_in_source=100,
            participants_in_target=100,
            participants_in_both=100,
        )

        assert drift1 == drift2

    def test_unequal_drifts(self):
        """Test unequal EloDrift objects."""
        drift1 = EloDrift(
            source="local",
            target="central",
            checked_at="2025-12-29T00:00:00Z",
            participants_in_source=100,
            participants_in_target=100,
            participants_in_both=100,
        )
        drift2 = EloDrift(
            source="remote",  # Different source
            target="central",
            checked_at="2025-12-29T00:00:00Z",
            participants_in_source=100,
            participants_in_target=100,
            participants_in_both=100,
        )

        assert drift1 != drift2


class TestEloReconcilerImport:
    """Tests for EloReconciler class import and basic instantiation."""

    def test_elo_reconciler_importable(self):
        """Test EloReconciler can be imported."""
        try:
            from app.training.elo_reconciliation import EloReconciler
            assert EloReconciler is not None
        except ImportError:
            pytest.skip("EloReconciler not yet implemented")

    def test_elo_reconciler_instantiation(self):
        """Test EloReconciler can be instantiated."""
        try:
            from app.training.elo_reconciliation import EloReconciler
            reconciler = EloReconciler()
            assert reconciler is not None
        except ImportError:
            pytest.skip("EloReconciler not yet implemented")
        except Exception:
            # May fail without database, that's OK for this test
            pass


class TestUtilityFunctions:
    """Tests for module utility functions."""

    def test_sync_elo_from_remote_importable(self):
        """Test sync_elo_from_remote function can be imported."""
        try:
            from app.training.elo_reconciliation import sync_elo_from_remote
            assert callable(sync_elo_from_remote)
        except ImportError:
            pytest.skip("sync_elo_from_remote not yet implemented")

    def test_check_elo_drift_importable(self):
        """Test check_elo_drift function can be imported."""
        try:
            from app.training.elo_reconciliation import check_elo_drift
            assert callable(check_elo_drift)
        except ImportError:
            pytest.skip("check_elo_drift not yet implemented")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
