"""Unit tests for elo_reconciliation module.

Comprehensive test coverage for distributed Elo rating reconciliation including:
- ConflictResolution enum values
- EloDrift dataclass and properties
- DriftHistory trend analysis and persistence tracking
- EloSyncResult (SyncResult) dataclass
- ReconciliationReport generation and summary
- EloReconciler class with all methods
- Conflict resolution strategies
- Timestamp comparison utilities
- Sync operations and error handling
- Convenience functions

Created: December 2025
"""

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import subprocess

import pytest

from app.training.elo_reconciliation import (
    ConflictResolution,
    EloDrift,
    DriftHistory,
    EloSyncResult,
    SyncResult,  # Backward compat alias
    ReconciliationReport,
    EloReconciler,
    sync_elo_from_remote,
    check_elo_drift,
)


# =============================================================================
# ConflictResolution Enum Tests
# =============================================================================

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
        assert ConflictResolution("first_write_wins") == ConflictResolution.FIRST_WRITE_WINS
        assert ConflictResolution("raise") == ConflictResolution.RAISE

    def test_enum_invalid_value_raises(self):
        """Test that invalid enum value raises ValueError."""
        with pytest.raises(ValueError):
            ConflictResolution("invalid_value")

    def test_enum_iteration(self):
        """Test enum can be iterated."""
        members = list(ConflictResolution)
        assert len(members) == 4


# =============================================================================
# EloDrift Dataclass Tests
# =============================================================================

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
        assert drift.participants_in_source == 100
        assert drift.participants_in_target == 95
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
        assert drift.rating_diffs["model_c"] == 50.0

    def test_max_rating_diff_property_empty(self):
        """Test max_rating_diff with no diffs returns 0."""
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
        """Test max_rating_diff returns absolute maximum."""
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

        assert drift.max_rating_diff == 60.0

    def test_avg_rating_diff_property_empty(self):
        """Test avg_rating_diff with no diffs returns 0."""
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

    def test_is_significant_low_drift(self):
        """Test is_significant returns False for low drift."""
        drift = EloDrift(
            source="local",
            target="central",
            checked_at="2025-12-29T00:00:00Z",
            participants_in_source=100,
            participants_in_target=100,
            participants_in_both=100,
            rating_diffs={"model_a": 10.0, "model_b": 15.0},
        )

        assert drift.is_significant is False

    def test_is_significant_high_max_drift(self):
        """Test is_significant returns True for high max drift > 50."""
        drift = EloDrift(
            source="local",
            target="central",
            checked_at="2025-12-29T00:00:00Z",
            participants_in_source=100,
            participants_in_target=100,
            participants_in_both=100,
            rating_diffs={"model_a": 60.0},
        )

        assert drift.is_significant is True

    def test_is_significant_high_avg_drift(self):
        """Test is_significant returns True for high avg drift > 25."""
        drift = EloDrift(
            source="local",
            target="central",
            checked_at="2025-12-29T00:00:00Z",
            participants_in_source=100,
            participants_in_target=100,
            participants_in_both=100,
            rating_diffs={
                "model_a": 30.0,
                "model_b": 30.0,
                "model_c": 30.0,
            },
        )

        # avg = 30 > 25, max = 30 < 50 but avg triggers it
        assert drift.is_significant is True

    def test_board_type_optional(self):
        """Test board_type is optional and defaults to None."""
        drift = EloDrift(
            source="local",
            target="central",
            checked_at="2025-12-29T00:00:00Z",
            participants_in_source=10,
            participants_in_target=10,
            participants_in_both=10,
        )

        assert drift.board_type is None
        assert drift.num_players is None

    def test_with_board_type_and_num_players(self):
        """Test EloDrift with board_type and num_players set."""
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

    def test_to_dict(self):
        """Test EloDrift to_dict serialization."""
        drift = EloDrift(
            source="local",
            target="central",
            checked_at="2025-12-29T00:00:00Z",
            participants_in_source=100,
            participants_in_target=95,
            participants_in_both=90,
            rating_diffs={"model_a": 10.0},  # Small diff, not significant
            board_type="square8",
            num_players=4,
        )

        d = drift.to_dict()

        assert d["source"] == "local"
        assert d["target"] == "central"
        assert d["checked_at"] == "2025-12-29T00:00:00Z"
        assert d["participants_in_source"] == 100
        assert d["participants_in_target"] == 95
        assert d["participants_in_both"] == 90
        assert d["max_rating_diff"] == 10.0
        assert d["avg_rating_diff"] == 10.0
        assert d["is_significant"] is False  # max < 50 and avg < 25
        assert d["rating_diffs"] == {"model_a": 10.0}
        assert d["board_type"] == "square8"
        assert d["num_players"] == 4

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


# =============================================================================
# DriftHistory Tests
# =============================================================================

class TestDriftHistory:
    """Tests for DriftHistory trend analysis."""

    def test_basic_creation(self):
        """Test DriftHistory creation."""
        history = DriftHistory(config_key="hex8_2p")

        assert history.config_key == "hex8_2p"
        assert history.snapshots == []
        assert history.max_snapshots == 100

    def test_add_snapshot(self):
        """Test adding a snapshot to history."""
        history = DriftHistory(config_key="hex8_2p")
        drift = EloDrift(
            source="local",
            target="central",
            checked_at="2025-12-29T00:00:00Z",
            participants_in_source=100,
            participants_in_target=100,
            participants_in_both=100,
            rating_diffs={"model_a": 25.0},
        )

        history.add_snapshot(drift)

        assert len(history.snapshots) == 1
        assert history.snapshots[0]["checked_at"] == "2025-12-29T00:00:00Z"
        assert history.snapshots[0]["max_rating_diff"] == 25.0

    def test_max_snapshots_limit(self):
        """Test that snapshots are limited to max_snapshots."""
        history = DriftHistory(config_key="hex8_2p", max_snapshots=5)

        for i in range(10):
            drift = EloDrift(
                source="local",
                target="central",
                checked_at=f"2025-12-29T00:0{i}:00Z",
                participants_in_source=100,
                participants_in_target=100,
                participants_in_both=100,
                rating_diffs={"model_a": float(i * 10)},
            )
            history.add_snapshot(drift)

        assert len(history.snapshots) == 5
        # Should keep the last 5 (indices 5-9)
        assert history.snapshots[0]["max_rating_diff"] == 50.0

    def test_trend_unknown_insufficient_data(self):
        """Test trend returns 'unknown' with < 3 snapshots."""
        history = DriftHistory(config_key="hex8_2p")

        assert history.trend == "unknown"

        # Add 2 snapshots
        for i in range(2):
            drift = EloDrift(
                source="local",
                target="central",
                checked_at=f"2025-12-29T00:0{i}:00Z",
                participants_in_source=100,
                participants_in_target=100,
                participants_in_both=100,
            )
            history.add_snapshot(drift)

        assert history.trend == "unknown"

    def test_trend_stable(self):
        """Test trend returns 'stable' for consistent drift."""
        history = DriftHistory(config_key="hex8_2p")

        for i in range(5):
            drift = EloDrift(
                source="local",
                target="central",
                checked_at=f"2025-12-29T00:0{i}:00Z",
                participants_in_source=100,
                participants_in_target=100,
                participants_in_both=100,
                rating_diffs={"model_a": 30.0},  # Same drift
            )
            history.add_snapshot(drift)

        assert history.trend == "stable"

    def test_trend_improving(self):
        """Test trend returns 'improving' when drift decreases."""
        history = DriftHistory(config_key="hex8_2p")

        # Start high, go low (improving)
        drifts = [100.0, 90.0, 50.0, 30.0, 20.0]
        for i, d in enumerate(drifts):
            drift = EloDrift(
                source="local",
                target="central",
                checked_at=f"2025-12-29T00:0{i}:00Z",
                participants_in_source=100,
                participants_in_target=100,
                participants_in_both=100,
                rating_diffs={"model_a": d},
            )
            history.add_snapshot(drift)

        assert history.trend == "improving"

    def test_trend_worsening(self):
        """Test trend returns 'worsening' when drift increases."""
        history = DriftHistory(config_key="hex8_2p")

        # Start low, go high (worsening)
        drifts = [20.0, 30.0, 50.0, 90.0, 100.0]
        for i, d in enumerate(drifts):
            drift = EloDrift(
                source="local",
                target="central",
                checked_at=f"2025-12-29T00:0{i}:00Z",
                participants_in_source=100,
                participants_in_target=100,
                participants_in_both=100,
                rating_diffs={"model_a": d},
            )
            history.add_snapshot(drift)

        assert history.trend == "worsening"

    def test_persistent_drift_false(self):
        """Test persistent_drift returns False with < 3 snapshots or not all significant."""
        history = DriftHistory(config_key="hex8_2p")

        # Less than 3 snapshots
        assert history.persistent_drift is False

        # Add 3 snapshots, only 2 significant
        drifts = [60.0, 10.0, 60.0]  # Middle one not significant
        for i, d in enumerate(drifts):
            drift = EloDrift(
                source="local",
                target="central",
                checked_at=f"2025-12-29T00:0{i}:00Z",
                participants_in_source=100,
                participants_in_target=100,
                participants_in_both=100,
                rating_diffs={"model_a": d},
            )
            history.add_snapshot(drift)

        assert history.persistent_drift is False

    def test_persistent_drift_true(self):
        """Test persistent_drift returns True when last 3 are all significant."""
        history = DriftHistory(config_key="hex8_2p")

        # All significant (max > 50)
        for i in range(3):
            drift = EloDrift(
                source="local",
                target="central",
                checked_at=f"2025-12-29T00:0{i}:00Z",
                participants_in_source=100,
                participants_in_target=100,
                participants_in_both=100,
                rating_diffs={"model_a": 60.0},
            )
            history.add_snapshot(drift)

        assert history.persistent_drift is True

    def test_avg_drift_last_hour_empty(self):
        """Test avg_drift_last_hour returns 0 with no snapshots."""
        history = DriftHistory(config_key="hex8_2p")
        assert history.avg_drift_last_hour == 0.0

    def test_avg_drift_last_hour(self):
        """Test avg_drift_last_hour calculation."""
        history = DriftHistory(config_key="hex8_2p")

        for i, d in enumerate([20.0, 40.0]):
            drift = EloDrift(
                source="local",
                target="central",
                checked_at=f"2025-12-29T00:0{i}:00Z",
                participants_in_source=100,
                participants_in_target=100,
                participants_in_both=100,
                rating_diffs={"model_a": d},
            )
            history.add_snapshot(drift)

        # (20 + 40) / 2 = 30
        assert history.avg_drift_last_hour == 30.0

    def test_to_dict(self):
        """Test DriftHistory to_dict serialization."""
        history = DriftHistory(config_key="hex8_2p")
        drift = EloDrift(
            source="local",
            target="central",
            checked_at="2025-12-29T00:00:00Z",
            participants_in_source=100,
            participants_in_target=100,
            participants_in_both=100,
        )
        history.add_snapshot(drift)

        d = history.to_dict()

        assert d["config_key"] == "hex8_2p"
        assert d["trend"] == "unknown"
        assert d["persistent_drift"] is False
        assert d["snapshot_count"] == 1
        assert len(d["recent_snapshots"]) == 1


# =============================================================================
# EloSyncResult Tests
# =============================================================================

class TestEloSyncResult:
    """Tests for EloSyncResult (SyncResult) dataclass."""

    def test_basic_creation(self):
        """Test EloSyncResult creation."""
        result = EloSyncResult(
            remote_host="192.168.1.100",
            synced_at="2025-12-29T00:00:00Z",
            matches_added=10,
            matches_skipped=5,
            matches_conflict=2,
        )

        assert result.remote_host == "192.168.1.100"
        assert result.matches_added == 10
        assert result.matches_skipped == 5
        assert result.matches_conflict == 2
        assert result.matches_resolved == 0  # Default
        assert result.participants_added == 0  # Default
        assert result.error is None

    def test_with_error(self):
        """Test EloSyncResult with error."""
        result = EloSyncResult(
            remote_host="192.168.1.100",
            synced_at="2025-12-29T00:00:00Z",
            matches_added=0,
            matches_skipped=0,
            matches_conflict=0,
            error="Connection refused",
        )

        assert result.error == "Connection refused"

    def test_backward_compat_alias(self):
        """Test SyncResult is an alias for EloSyncResult."""
        assert SyncResult is EloSyncResult

    def test_to_dict(self):
        """Test EloSyncResult to_dict serialization."""
        result = EloSyncResult(
            remote_host="192.168.1.100",
            synced_at="2025-12-29T00:00:00Z",
            matches_added=10,
            matches_skipped=5,
            matches_conflict=2,
            matches_resolved=1,
            participants_added=3,
            error=None,
        )

        d = result.to_dict()

        assert d["remote_host"] == "192.168.1.100"
        assert d["synced_at"] == "2025-12-29T00:00:00Z"
        assert d["matches_added"] == 10
        assert d["matches_skipped"] == 5
        assert d["matches_conflict"] == 2
        assert d["matches_resolved"] == 1
        assert d["participants_added"] == 3
        assert d["error"] is None


# =============================================================================
# ReconciliationReport Tests
# =============================================================================

class TestReconciliationReport:
    """Tests for ReconciliationReport class."""

    def test_basic_creation(self):
        """Test ReconciliationReport creation."""
        report = ReconciliationReport(
            started_at="2025-12-29T00:00:00Z",
            completed_at="2025-12-29T00:01:00Z",
            nodes_synced=["node1", "node2"],
            nodes_failed=["node3"],
            total_matches_added=100,
            total_matches_skipped=50,
            total_conflicts=5,
            total_resolved=3,
            drift_detected=True,
            max_drift=75.5,
        )

        assert report.started_at == "2025-12-29T00:00:00Z"
        assert report.completed_at == "2025-12-29T00:01:00Z"
        assert len(report.nodes_synced) == 2
        assert len(report.nodes_failed) == 1
        assert report.total_matches_added == 100
        assert report.drift_detected is True
        assert report.max_drift == 75.5

    def test_summary_no_drift(self):
        """Test summary output without significant drift."""
        report = ReconciliationReport(
            started_at="2025-12-29T00:00:00Z",
            completed_at="2025-12-29T00:01:00Z",
            nodes_synced=["node1", "node2"],
            nodes_failed=[],
            total_matches_added=100,
            total_matches_skipped=50,
            total_conflicts=0,
            total_resolved=0,
            drift_detected=False,
            max_drift=10.5,
        )

        summary = report.summary()

        assert "=== Elo Reconciliation Report ===" in summary
        assert "Nodes synced: 2" in summary
        assert "Nodes failed: 0" in summary
        assert "Matches added: 100" in summary
        assert "Max drift: 10.5" in summary
        assert "SIGNIFICANT DRIFT" not in summary

    def test_summary_with_drift(self):
        """Test summary output with significant drift."""
        report = ReconciliationReport(
            started_at="2025-12-29T00:00:00Z",
            completed_at="2025-12-29T00:01:00Z",
            nodes_synced=["node1"],
            nodes_failed=[],
            total_matches_added=50,
            total_matches_skipped=10,
            total_conflicts=2,
            total_resolved=1,
            drift_detected=True,
            max_drift=75.5,
        )

        summary = report.summary()

        assert "WARNING: SIGNIFICANT DRIFT DETECTED" in summary
        assert "Conflicts: 2" in summary
        assert "Resolved: 1" in summary

    def test_summary_with_failed_nodes(self):
        """Test summary lists failed nodes."""
        report = ReconciliationReport(
            started_at="2025-12-29T00:00:00Z",
            completed_at="2025-12-29T00:01:00Z",
            nodes_synced=["node1"],
            nodes_failed=["node2", "node3"],
            total_matches_added=20,
            total_matches_skipped=5,
            total_conflicts=0,
            total_resolved=0,
            drift_detected=False,
            max_drift=5.0,
        )

        summary = report.summary()

        assert "Failed nodes: node2, node3" in summary

    def test_sync_results_default_empty(self):
        """Test sync_results defaults to empty list."""
        report = ReconciliationReport(
            started_at="2025-12-29T00:00:00Z",
            completed_at="2025-12-29T00:01:00Z",
            nodes_synced=[],
            nodes_failed=[],
            total_matches_added=0,
            total_matches_skipped=0,
            total_conflicts=0,
            total_resolved=0,
            drift_detected=False,
            max_drift=0.0,
        )

        assert report.sync_results == []


# =============================================================================
# EloReconciler Tests
# =============================================================================

class TestEloReconcilerInit:
    """Tests for EloReconciler initialization."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        if db_path.exists():
            db_path.unlink()

    def test_default_initialization(self, temp_db):
        """Test EloReconciler default initialization."""
        reconciler = EloReconciler(local_db_path=temp_db, track_history=False)

        assert reconciler.local_db_path == temp_db
        assert reconciler.ssh_timeout == 30
        assert reconciler.conflict_resolution == ConflictResolution.SKIP
        assert reconciler.track_history is False

    def test_custom_conflict_resolution(self, temp_db):
        """Test EloReconciler with custom conflict resolution."""
        reconciler = EloReconciler(
            local_db_path=temp_db,
            conflict_resolution=ConflictResolution.LAST_WRITE_WINS,
            track_history=False,
        )

        assert reconciler.conflict_resolution == ConflictResolution.LAST_WRITE_WINS

    def test_custom_ssh_timeout(self, temp_db):
        """Test EloReconciler with custom SSH timeout."""
        reconciler = EloReconciler(
            local_db_path=temp_db,
            ssh_timeout=60,
            track_history=False,
        )

        assert reconciler.ssh_timeout == 60


class TestEloReconcilerDriftHistory:
    """Tests for EloReconciler drift history management."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_drift_history_path(self, temp_dir):
        """Test _drift_history_path property."""
        db_path = temp_dir / "unified_elo.db"
        reconciler = EloReconciler(
            local_db_path=db_path,
            track_history=False,
            persist_history=False,
        )

        assert reconciler._drift_history_path == temp_dir / "elo_drift_history.json"

    def test_save_and_load_drift_history(self, temp_dir):
        """Test saving and loading drift history."""
        db_path = temp_dir / "unified_elo.db"
        db_path.touch()

        reconciler = EloReconciler(
            local_db_path=db_path,
            track_history=True,
            persist_history=True,
        )

        # Add some history
        reconciler._drift_history["hex8_2p"] = DriftHistory(config_key="hex8_2p")
        reconciler.save_drift_history()

        # Verify file was created
        assert reconciler._drift_history_path.exists()

        # Create new reconciler and load
        reconciler2 = EloReconciler(
            local_db_path=db_path,
            track_history=True,
            persist_history=True,
        )

        assert "hex8_2p" in reconciler2._drift_history

    def test_get_drift_history(self, temp_dir):
        """Test get_drift_history method."""
        db_path = temp_dir / "unified_elo.db"
        reconciler = EloReconciler(
            local_db_path=db_path,
            track_history=True,
            persist_history=False,
        )

        # Initially empty
        assert reconciler.get_drift_history("hex8_2p") is None

        # Add history
        reconciler._drift_history["hex8_2p"] = DriftHistory(config_key="hex8_2p")
        history = reconciler.get_drift_history("hex8_2p")

        assert history is not None
        assert history.config_key == "hex8_2p"

    def test_get_all_drift_histories(self, temp_dir):
        """Test get_all_drift_histories method."""
        db_path = temp_dir / "unified_elo.db"
        reconciler = EloReconciler(
            local_db_path=db_path,
            track_history=True,
            persist_history=False,
        )

        reconciler._drift_history["hex8_2p"] = DriftHistory(config_key="hex8_2p")
        reconciler._drift_history["square8_4p"] = DriftHistory(config_key="square8_4p")

        histories = reconciler.get_all_drift_histories()

        assert len(histories) == 2
        assert "hex8_2p" in histories
        assert "square8_4p" in histories


class TestEloReconcilerCheckDrift:
    """Tests for EloReconciler.check_drift method."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_check_drift_local_db_missing(self, temp_dir):
        """Test check_drift when local DB doesn't exist."""
        db_path = temp_dir / "missing.db"
        reconciler = EloReconciler(
            local_db_path=db_path,
            track_history=False,
        )

        drift = reconciler.check_drift()

        assert drift.participants_in_source == 0

    def test_check_drift_remote_db_missing(self, temp_dir):
        """Test check_drift when remote DB doesn't exist."""
        local_db = temp_dir / "local.db"
        remote_db = temp_dir / "remote.db"
        local_db.touch()

        # Create local DB with schema
        conn = sqlite3.connect(str(local_db))
        conn.execute("""
            CREATE TABLE participants (
                participant_id TEXT PRIMARY KEY,
                rating REAL DEFAULT 1500.0
            )
        """)
        conn.execute("INSERT INTO participants VALUES ('model_a', 1500.0)")
        conn.commit()
        conn.close()

        reconciler = EloReconciler(
            local_db_path=local_db,
            track_history=False,
        )

        drift = reconciler.check_drift(remote_db_path=remote_db)

        assert drift.participants_in_source >= 1
        assert drift.participants_in_target == 0

    def test_check_drift_with_both_dbs(self, temp_dir):
        """Test check_drift with both databases."""
        local_db = temp_dir / "local.db"
        remote_db = temp_dir / "remote.db"

        # Create local DB
        conn = sqlite3.connect(str(local_db))
        conn.execute("""
            CREATE TABLE participants (
                participant_id TEXT PRIMARY KEY,
                rating REAL
            )
        """)
        conn.execute("INSERT INTO participants VALUES ('model_a', 1550.0)")
        conn.execute("INSERT INTO participants VALUES ('model_b', 1450.0)")
        conn.commit()
        conn.close()

        # Create remote DB with different ratings
        conn = sqlite3.connect(str(remote_db))
        conn.execute("""
            CREATE TABLE participants (
                participant_id TEXT PRIMARY KEY,
                rating REAL
            )
        """)
        conn.execute("INSERT INTO participants VALUES ('model_a', 1500.0)")  # Diff: 50
        conn.execute("INSERT INTO participants VALUES ('model_b', 1450.0)")  # Diff: 0
        conn.commit()
        conn.close()

        reconciler = EloReconciler(
            local_db_path=local_db,
            track_history=False,
        )

        with patch.object(reconciler, '_emit_drift_metrics'):
            drift = reconciler.check_drift(remote_db_path=remote_db)

        assert drift.participants_in_source == 2
        assert drift.participants_in_target == 2
        assert drift.participants_in_both == 2
        assert drift.max_rating_diff == 50.0


class TestEloReconcilerImportMatches:
    """Tests for EloReconciler._import_matches method."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_import_empty_matches(self, temp_dir):
        """Test importing empty matches list."""
        db_path = temp_dir / "unified_elo.db"
        reconciler = EloReconciler(
            local_db_path=db_path,
            track_history=False,
        )

        result = reconciler._import_matches(
            remote_host="192.168.1.100",
            synced_at="2025-12-29T00:00:00Z",
            matches=[],
        )

        assert result.matches_added == 0
        assert result.matches_skipped == 0
        assert result.matches_conflict == 0

    def test_import_new_matches(self, temp_dir):
        """Test importing new matches."""
        db_path = temp_dir / "unified_elo.db"
        reconciler = EloReconciler(
            local_db_path=db_path,
            track_history=False,
        )

        matches = [
            {
                "match_id": "match_001",
                "player1_id": "player_a",
                "player2_id": "player_b",
                "winner_id": "player_a",
                "timestamp": "2025-12-29T00:00:00Z",
            },
            {
                "match_id": "match_002",
                "player1_id": "player_a",
                "player2_id": "player_c",
                "winner_id": "player_c",
                "timestamp": "2025-12-29T00:01:00Z",
            },
        ]

        result = reconciler._import_matches(
            remote_host="192.168.1.100",
            synced_at="2025-12-29T00:00:00Z",
            matches=matches,
        )

        assert result.matches_added == 2
        assert result.matches_skipped == 0

    def test_import_duplicate_matches(self, temp_dir):
        """Test importing duplicate matches (same ID, same winner)."""
        db_path = temp_dir / "unified_elo.db"
        reconciler = EloReconciler(
            local_db_path=db_path,
            track_history=False,
        )

        matches = [
            {
                "match_id": "match_001",
                "player1_id": "player_a",
                "player2_id": "player_b",
                "winner_id": "player_a",
            },
        ]

        # Import first time
        reconciler._import_matches("192.168.1.100", "2025-12-29T00:00:00Z", matches)

        # Import same match again
        result = reconciler._import_matches(
            remote_host="192.168.1.100",
            synced_at="2025-12-29T00:01:00Z",
            matches=matches,
        )

        assert result.matches_added == 0
        assert result.matches_skipped == 1
        assert result.matches_conflict == 0

    def test_import_conflicting_matches_skip(self, temp_dir):
        """Test importing conflicting matches with SKIP resolution."""
        db_path = temp_dir / "unified_elo.db"
        reconciler = EloReconciler(
            local_db_path=db_path,
            conflict_resolution=ConflictResolution.SKIP,
            track_history=False,
        )

        # Import original match
        matches_orig = [
            {
                "match_id": "match_001",
                "player1_id": "player_a",
                "player2_id": "player_b",
                "winner_id": "player_a",
            },
        ]
        reconciler._import_matches("192.168.1.100", "2025-12-29T00:00:00Z", matches_orig)

        # Import conflicting match (different winner)
        matches_conflict = [
            {
                "match_id": "match_001",
                "player1_id": "player_a",
                "player2_id": "player_b",
                "winner_id": "player_b",  # Different winner!
            },
        ]

        result = reconciler._import_matches(
            remote_host="192.168.1.100",
            synced_at="2025-12-29T00:01:00Z",
            matches=matches_conflict,
        )

        assert result.matches_conflict == 1
        assert result.matches_resolved == 0

    def test_import_conflicting_matches_raise(self, temp_dir):
        """Test importing conflicting matches with RAISE resolution."""
        db_path = temp_dir / "unified_elo.db"
        reconciler = EloReconciler(
            local_db_path=db_path,
            conflict_resolution=ConflictResolution.RAISE,
            track_history=False,
        )

        # Import original match
        matches_orig = [
            {
                "match_id": "match_001",
                "player1_id": "player_a",
                "player2_id": "player_b",
                "winner_id": "player_a",
            },
        ]
        reconciler._import_matches("192.168.1.100", "2025-12-29T00:00:00Z", matches_orig)

        # Import conflicting match
        matches_conflict = [
            {
                "match_id": "match_001",
                "player1_id": "player_a",
                "player2_id": "player_b",
                "winner_id": "player_b",
            },
        ]

        with pytest.raises(ValueError, match="Match conflict"):
            reconciler._import_matches(
                remote_host="192.168.1.100",
                synced_at="2025-12-29T00:01:00Z",
                matches=matches_conflict,
            )


class TestEloReconcilerTimestamp:
    """Tests for EloReconciler timestamp comparison."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_is_newer_timestamp_incoming_none(self, temp_dir):
        """Test _is_newer_timestamp returns False when incoming is None."""
        db_path = temp_dir / "unified_elo.db"
        reconciler = EloReconciler(local_db_path=db_path, track_history=False)

        result = reconciler._is_newer_timestamp(None, "2025-12-29T00:00:00Z")
        assert result is False

    def test_is_newer_timestamp_existing_none(self, temp_dir):
        """Test _is_newer_timestamp returns True when existing is None."""
        db_path = temp_dir / "unified_elo.db"
        reconciler = EloReconciler(local_db_path=db_path, track_history=False)

        result = reconciler._is_newer_timestamp("2025-12-29T00:00:00Z", None)
        assert result is True

    def test_is_newer_timestamp_incoming_newer(self, temp_dir):
        """Test _is_newer_timestamp returns True when incoming is newer."""
        db_path = temp_dir / "unified_elo.db"
        reconciler = EloReconciler(local_db_path=db_path, track_history=False)

        result = reconciler._is_newer_timestamp(
            "2025-12-29T01:00:00Z",
            "2025-12-29T00:00:00Z",
        )
        assert result is True

    def test_is_newer_timestamp_existing_newer(self, temp_dir):
        """Test _is_newer_timestamp returns False when existing is newer."""
        db_path = temp_dir / "unified_elo.db"
        reconciler = EloReconciler(local_db_path=db_path, track_history=False)

        result = reconciler._is_newer_timestamp(
            "2025-12-29T00:00:00Z",
            "2025-12-29T01:00:00Z",
        )
        assert result is False

    def test_is_newer_timestamp_iso_with_z(self, temp_dir):
        """Test _is_newer_timestamp handles Z suffix."""
        db_path = temp_dir / "unified_elo.db"
        reconciler = EloReconciler(local_db_path=db_path, track_history=False)

        result = reconciler._is_newer_timestamp(
            "2025-12-29T01:00:00Z",
            "2025-12-29T00:00:00+00:00",
        )
        assert result is True


class TestEloReconcilerSyncFromRemote:
    """Tests for EloReconciler.sync_from_remote method."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_sync_ssh_failure(self, temp_dir):
        """Test sync_from_remote handles SSH failure."""
        db_path = temp_dir / "unified_elo.db"
        reconciler = EloReconciler(
            local_db_path=db_path,
            ssh_timeout=5,
            track_history=False,
        )

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stderr="Connection refused",
            )

            result = reconciler.sync_from_remote("192.168.1.100")

        assert result.error is not None
        assert "SSH failed" in result.error

    def test_sync_timeout(self, temp_dir):
        """Test sync_from_remote handles timeout."""
        db_path = temp_dir / "unified_elo.db"
        reconciler = EloReconciler(
            local_db_path=db_path,
            ssh_timeout=5,
            track_history=False,
        )

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("ssh", 5)

            with patch.object(reconciler, '_emit_sync_metrics'):
                result = reconciler.sync_from_remote("192.168.1.100")

        assert result.error == "SSH timeout"

    def test_sync_json_parse_error(self, temp_dir):
        """Test sync_from_remote handles JSON parse error."""
        db_path = temp_dir / "unified_elo.db"
        reconciler = EloReconciler(
            local_db_path=db_path,
            ssh_timeout=5,
            track_history=False,
        )

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="not valid json",
            )

            result = reconciler.sync_from_remote("192.168.1.100")

        assert result.error is not None
        assert "JSON parse error" in result.error

    def test_sync_remote_db_not_found(self, temp_dir):
        """Test sync_from_remote handles remote DB not found."""
        db_path = temp_dir / "unified_elo.db"
        reconciler = EloReconciler(
            local_db_path=db_path,
            ssh_timeout=5,
            track_history=False,
        )

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"error": "DB not found"}',
            )

            result = reconciler.sync_from_remote("192.168.1.100")

        assert result.error == "DB not found"

    def test_sync_success(self, temp_dir):
        """Test sync_from_remote successful sync."""
        db_path = temp_dir / "unified_elo.db"
        reconciler = EloReconciler(
            local_db_path=db_path,
            ssh_timeout=5,
            track_history=False,
        )

        matches_data = {
            "matches": [
                {
                    "match_id": "match_001",
                    "player1_id": "player_a",
                    "player2_id": "player_b",
                    "winner_id": "player_a",
                    "timestamp": "2025-12-29T00:00:00Z",
                },
            ],
        }

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(matches_data),
            )

            with patch.object(reconciler, '_emit_sync_metrics'):
                result = reconciler.sync_from_remote("192.168.1.100")

        assert result.error is None
        assert result.matches_added == 1


class TestEloReconcilerReconcileAll:
    """Tests for EloReconciler.reconcile_all method."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_reconcile_all_empty_hosts(self, temp_dir):
        """Test reconcile_all with empty hosts list."""
        db_path = temp_dir / "unified_elo.db"
        db_path.touch()

        reconciler = EloReconciler(
            local_db_path=db_path,
            track_history=False,
        )

        report = reconciler.reconcile_all(hosts=[])

        assert len(report.nodes_synced) == 0
        assert len(report.nodes_failed) == 0
        assert report.total_matches_added == 0

    def test_reconcile_all_single_host_success(self, temp_dir):
        """Test reconcile_all with single successful host."""
        db_path = temp_dir / "unified_elo.db"
        db_path.touch()

        reconciler = EloReconciler(
            local_db_path=db_path,
            track_history=False,
        )

        mock_result = SyncResult(
            remote_host="192.168.1.100",
            synced_at="2025-12-29T00:00:00Z",
            matches_added=10,
            matches_skipped=5,
            matches_conflict=0,
        )

        with patch.object(reconciler, 'sync_from_remote', return_value=mock_result):
            with patch.object(reconciler, 'check_drift') as mock_drift:
                mock_drift.return_value = EloDrift(
                    source="local",
                    target="local",
                    checked_at="2025-12-29T00:00:00Z",
                    participants_in_source=10,
                    participants_in_target=10,
                    participants_in_both=10,
                )

                report = reconciler.reconcile_all(hosts=["192.168.1.100"])

        assert len(report.nodes_synced) == 1
        assert len(report.nodes_failed) == 0
        assert report.total_matches_added == 10

    def test_reconcile_all_mixed_results(self, temp_dir):
        """Test reconcile_all with mixed success/failure."""
        db_path = temp_dir / "unified_elo.db"
        db_path.touch()

        reconciler = EloReconciler(
            local_db_path=db_path,
            track_history=False,
        )

        def mock_sync(host, *args, **kwargs):
            if host == "192.168.1.100":
                return SyncResult(
                    remote_host=host,
                    synced_at="2025-12-29T00:00:00Z",
                    matches_added=10,
                    matches_skipped=0,
                    matches_conflict=0,
                )
            else:
                return SyncResult(
                    remote_host=host,
                    synced_at="2025-12-29T00:00:00Z",
                    matches_added=0,
                    matches_skipped=0,
                    matches_conflict=0,
                    error="Connection refused",
                )

        with patch.object(reconciler, 'sync_from_remote', side_effect=mock_sync):
            with patch.object(reconciler, 'check_drift') as mock_drift:
                mock_drift.return_value = EloDrift(
                    source="local",
                    target="local",
                    checked_at="2025-12-29T00:00:00Z",
                    participants_in_source=10,
                    participants_in_target=10,
                    participants_in_both=10,
                )

                report = reconciler.reconcile_all(hosts=["192.168.1.100", "192.168.1.101"])

        assert len(report.nodes_synced) == 1
        assert len(report.nodes_failed) == 1
        assert "192.168.1.100" in report.nodes_synced
        assert "192.168.1.101" in report.nodes_failed


class TestEloReconcilerLoadHosts:
    """Tests for EloReconciler._load_p2p_hosts method."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_load_hosts_missing_config(self, temp_dir):
        """Test _load_p2p_hosts returns empty when config missing and no legacy fallback."""
        db_path = temp_dir / "unified_elo.db"
        config_path = temp_dir / "nonexistent.yaml"

        reconciler = EloReconciler(
            local_db_path=db_path,
            remote_hosts_config=config_path,
            track_history=False,
        )

        # Mock the AI_SERVICE_ROOT to prevent fallback to real config
        with patch('app.training.elo_reconciliation.AI_SERVICE_ROOT', temp_dir):
            hosts = reconciler._load_p2p_hosts()
            assert hosts == []

    def test_load_hosts_from_yaml(self, temp_dir):
        """Test _load_p2p_hosts parses YAML config."""
        db_path = temp_dir / "unified_elo.db"
        config_path = temp_dir / "hosts.yaml"

        config_content = """
hosts:
  node1:
    tailscale_ip: "100.64.0.1"
    status: "active"
  node2:
    ssh_host: "192.168.1.100"
    status: "active"
  node3:
    tailscale_ip: "100.64.0.3"
    status: "terminated"
"""
        config_path.write_text(config_content)

        reconciler = EloReconciler(
            local_db_path=db_path,
            remote_hosts_config=config_path,
            track_history=False,
        )

        hosts = reconciler._load_p2p_hosts()

        assert len(hosts) == 2
        assert "100.64.0.1" in hosts
        assert "192.168.1.100" in hosts
        # node3 is terminated, should be excluded


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_sync_elo_from_remote(self, temp_dir):
        """Test sync_elo_from_remote convenience function."""
        with patch('app.training.elo_reconciliation.EloReconciler') as mock_class:
            mock_instance = MagicMock()
            mock_instance.sync_from_remote.return_value = SyncResult(
                remote_host="192.168.1.100",
                synced_at="2025-12-29T00:00:00Z",
                matches_added=5,
                matches_skipped=0,
                matches_conflict=0,
            )
            mock_class.return_value = mock_instance

            result = sync_elo_from_remote("192.168.1.100")

            assert result.matches_added == 5
            mock_instance.sync_from_remote.assert_called_once()

    def test_check_elo_drift(self, temp_dir):
        """Test check_elo_drift convenience function."""
        with patch('app.training.elo_reconciliation.EloReconciler') as mock_class:
            mock_instance = MagicMock()
            mock_instance.check_drift.return_value = EloDrift(
                source="local",
                target="N/A",
                checked_at="2025-12-29T00:00:00Z",
                participants_in_source=50,
                participants_in_target=0,
                participants_in_both=0,
            )
            mock_class.return_value = mock_instance

            drift = check_elo_drift(board_type="hex8", num_players=2)

            assert drift.participants_in_source == 50
            mock_instance.check_drift.assert_called_once_with(
                board_type="hex8",
                num_players=2,
            )


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def test_import_match_without_match_id(self, temp_dir):
        """Test importing match without match_id is skipped."""
        db_path = temp_dir / "unified_elo.db"
        reconciler = EloReconciler(
            local_db_path=db_path,
            track_history=False,
        )

        matches = [
            {
                # No match_id
                "player1_id": "player_a",
                "player2_id": "player_b",
                "winner_id": "player_a",
            },
            {
                "match_id": "match_001",
                "player1_id": "player_a",
                "player2_id": "player_b",
                "winner_id": "player_a",
            },
        ]

        result = reconciler._import_matches(
            remote_host="192.168.1.100",
            synced_at="2025-12-29T00:00:00Z",
            matches=matches,
        )

        # Only the match with ID should be added
        assert result.matches_added == 1

    def test_conflicting_matches_first_write_wins(self, temp_dir):
        """Test FIRST_WRITE_WINS conflict resolution."""
        db_path = temp_dir / "unified_elo.db"
        reconciler = EloReconciler(
            local_db_path=db_path,
            conflict_resolution=ConflictResolution.FIRST_WRITE_WINS,
            track_history=False,
        )

        # Import original
        matches_orig = [
            {
                "match_id": "match_001",
                "player1_id": "player_a",
                "player2_id": "player_b",
                "winner_id": "player_a",
            },
        ]
        reconciler._import_matches("192.168.1.100", "2025-12-29T00:00:00Z", matches_orig)

        # Import conflicting
        matches_conflict = [
            {
                "match_id": "match_001",
                "player1_id": "player_a",
                "player2_id": "player_b",
                "winner_id": "player_b",
            },
        ]

        result = reconciler._import_matches(
            remote_host="192.168.1.100",
            synced_at="2025-12-29T00:01:00Z",
            matches=matches_conflict,
        )

        # Should be resolved, not conflict
        assert result.matches_resolved == 1
        assert result.matches_conflict == 0

    def test_conflicting_matches_last_write_wins_newer(self, temp_dir):
        """Test LAST_WRITE_WINS with newer incoming timestamp."""
        db_path = temp_dir / "unified_elo.db"
        reconciler = EloReconciler(
            local_db_path=db_path,
            conflict_resolution=ConflictResolution.LAST_WRITE_WINS,
            track_history=False,
        )

        # Import original with older timestamp
        matches_orig = [
            {
                "match_id": "match_001",
                "player1_id": "player_a",
                "player2_id": "player_b",
                "winner_id": "player_a",
                "timestamp": "2025-12-29T00:00:00Z",
            },
        ]
        reconciler._import_matches("192.168.1.100", "2025-12-29T00:00:00Z", matches_orig)

        # Import conflicting with newer timestamp
        matches_conflict = [
            {
                "match_id": "match_001",
                "player1_id": "player_a",
                "player2_id": "player_b",
                "winner_id": "player_b",
                "timestamp": "2025-12-29T01:00:00Z",  # Newer
            },
        ]

        result = reconciler._import_matches(
            remote_host="192.168.1.100",
            synced_at="2025-12-29T01:00:00Z",
            matches=matches_conflict,
        )

        assert result.matches_resolved == 1

    def test_get_ratings_elo_ratings_table(self, temp_dir):
        """Test _get_ratings with elo_ratings table schema."""
        db_path = temp_dir / "unified_elo.db"

        # Create DB with elo_ratings table (production schema)
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE elo_ratings (
                participant_id TEXT,
                board_type TEXT,
                num_players INTEGER,
                rating REAL,
                PRIMARY KEY (participant_id, board_type, num_players)
            )
        """)
        conn.execute(
            "INSERT INTO elo_ratings VALUES ('model_a', 'hex8', 2, 1600.0)"
        )
        conn.execute(
            "INSERT INTO elo_ratings VALUES ('model_a', 'square8', 2, 1550.0)"
        )
        conn.commit()
        conn.close()

        reconciler = EloReconciler(
            local_db_path=db_path,
            track_history=False,
        )

        ratings = reconciler._get_ratings(db_path, board_type="hex8", num_players=2)

        assert len(ratings) == 1
        assert ratings["model_a"] == 1600.0

    def test_get_ratings_no_tables(self, temp_dir):
        """Test _get_ratings with empty database."""
        db_path = temp_dir / "empty.db"

        # Create empty DB
        conn = sqlite3.connect(str(db_path))
        conn.close()

        reconciler = EloReconciler(
            local_db_path=db_path,
            track_history=False,
        )

        ratings = reconciler._get_ratings(db_path)

        assert ratings == {}

    def test_count_participants(self, temp_dir):
        """Test _count_participants method."""
        db_path = temp_dir / "unified_elo.db"

        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE participants (
                participant_id TEXT PRIMARY KEY,
                rating REAL
            )
        """)
        conn.execute("INSERT INTO participants VALUES ('model_a', 1500.0)")
        conn.execute("INSERT INTO participants VALUES ('model_b', 1550.0)")
        conn.commit()
        conn.close()

        reconciler = EloReconciler(
            local_db_path=db_path,
            track_history=False,
        )

        count = reconciler._count_participants(db_path)

        assert count == 2

    def test_record_drift_updates_history(self, temp_dir):
        """Test _record_drift adds to history."""
        db_path = temp_dir / "unified_elo.db"
        db_path.touch()

        reconciler = EloReconciler(
            local_db_path=db_path,
            track_history=True,
            persist_history=False,
        )

        drift = EloDrift(
            source="local",
            target="central",
            checked_at="2025-12-29T00:00:00Z",
            participants_in_source=100,
            participants_in_target=100,
            participants_in_both=100,
            rating_diffs={"model_a": 25.0},
        )

        reconciler._record_drift(drift, board_type="hex8", num_players=2)

        history = reconciler.get_drift_history("hex8_2")
        assert history is not None
        assert len(history.snapshots) == 1

    def test_record_drift_disabled(self, temp_dir):
        """Test _record_drift does nothing when track_history=False."""
        db_path = temp_dir / "unified_elo.db"
        db_path.touch()

        reconciler = EloReconciler(
            local_db_path=db_path,
            track_history=False,
            persist_history=False,
        )

        drift = EloDrift(
            source="local",
            target="central",
            checked_at="2025-12-29T00:00:00Z",
            participants_in_source=100,
            participants_in_target=100,
            participants_in_both=100,
        )

        reconciler._record_drift(drift, board_type="hex8", num_players=2)

        assert len(reconciler._drift_history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
