#!/usr/bin/env python3
"""Unit tests for app.db.validation module (December 2025).

Tests the game validation system for cross-validating recorded games:
- ValidationResult dataclass
- validate_game function
- validate_all_games function
- export_fixture function
- validate_database_summary function

These tests use mocks to avoid requiring actual game databases.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from dataclasses import asdict

from app.db.validation import (
    ValidationResult,
    validate_game,
    validate_all_games,
    export_fixture,
    validate_database_summary,
    _get_snapshots_with_hashes,
)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result_creation(self):
        """Test creating a valid result."""
        result = ValidationResult(
            game_id="game-123",
            valid=True,
            total_moves=50,
            validated_moves=50,
        )
        assert result.game_id == "game-123"
        assert result.valid is True
        assert result.total_moves == 50
        assert result.validated_moves == 50
        assert result.divergence_move is None
        assert result.error is None

    def test_invalid_result_with_divergence(self):
        """Test creating an invalid result with divergence."""
        mock_state = MagicMock()
        mock_move = MagicMock()

        result = ValidationResult(
            game_id="bad-game",
            valid=False,
            total_moves=100,
            validated_moves=45,
            divergence_move=45,
            expected_hash="expected123",
            computed_hash="computed456",
            expected_state=mock_state,
            computed_state=mock_state,
            move_at_divergence=mock_move,
        )

        assert result.valid is False
        assert result.divergence_move == 45
        assert result.expected_hash == "expected123"
        assert result.computed_hash == "computed456"

    def test_result_with_error(self):
        """Test result with error message."""
        result = ValidationResult(
            game_id="error-game",
            valid=False,
            total_moves=10,
            validated_moves=5,
            divergence_move=5,
            error="Error applying move 5: Invalid placement",
        )

        assert result.valid is False
        assert result.error is not None
        assert "Invalid placement" in result.error

    def test_result_default_lists(self):
        """Test default empty lists."""
        result = ValidationResult(
            game_id="test",
            valid=True,
            total_moves=0,
            validated_moves=0,
        )

        assert result.moves_up_to_divergence == []


class TestValidateGame:
    """Tests for validate_game function."""

    def test_validate_game_no_initial_state(self):
        """Test validation fails when initial state missing."""
        mock_db = MagicMock()
        mock_db.get_initial_state.return_value = None

        result = validate_game(mock_db, "game-123")

        assert result.valid is False
        assert result.error == "Initial state not found"
        assert result.total_moves == 0

    def test_validate_game_no_moves(self):
        """Test validation succeeds for game with no moves."""
        mock_db = MagicMock()
        mock_db.get_initial_state.return_value = MagicMock()
        mock_db.get_moves.return_value = []

        result = validate_game(mock_db, "empty-game")

        assert result.valid is True
        assert result.total_moves == 0
        assert result.validated_moves == 0

    def test_validate_game_successful(self):
        """Test successful validation of a game."""
        mock_db = MagicMock()
        mock_initial = MagicMock()
        mock_move = MagicMock()

        mock_db.get_initial_state.return_value = mock_initial
        mock_db.get_moves.return_value = [mock_move]

        # Mock the snapshots query
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_db._get_conn.return_value = mock_conn

        with patch("app.game_engine.GameEngine") as mock_engine:
            mock_next_state = MagicMock()
            mock_engine.apply_move.return_value = mock_next_state

            with patch("app.db.validation._compute_state_hash", return_value="hash123"):
                result = validate_game(mock_db, "game-123")

        assert result.valid is True
        assert result.total_moves == 1

    def test_validate_game_divergence_detected(self):
        """Test divergence is detected when hashes don't match."""
        mock_db = MagicMock()
        mock_initial = MagicMock()
        mock_move = MagicMock()

        mock_db.get_initial_state.return_value = mock_initial
        mock_db.get_moves.return_value = [mock_move]
        mock_db.get_state_at_move.return_value = MagicMock()

        # Mock snapshots with a hash that won't match
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = [
            {"move_number": 0, "state_hash": "expected_hash", "compressed": False}
        ]
        mock_db._get_conn.return_value = mock_conn

        with patch("app.game_engine.GameEngine") as mock_engine:
            mock_engine.apply_move.return_value = MagicMock()

            with patch("app.db.validation._compute_state_hash", return_value="different_hash"):
                result = validate_game(mock_db, "divergent-game")

        assert result.valid is False
        assert result.divergence_move == 0
        assert result.expected_hash == "expected_hash"
        assert result.computed_hash == "different_hash"

    def test_validate_game_apply_move_error(self):
        """Test handling of apply_move errors."""
        mock_db = MagicMock()
        mock_initial = MagicMock()
        mock_move = MagicMock()

        mock_db.get_initial_state.return_value = mock_initial
        mock_db.get_moves.return_value = [mock_move]

        # Mock snapshots
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_db._get_conn.return_value = mock_conn

        with patch("app.game_engine.GameEngine") as mock_engine:
            mock_engine.apply_move.side_effect = ValueError("Invalid move")

            result = validate_game(mock_db, "error-game")

        assert result.valid is False
        assert result.divergence_move == 0
        assert "Invalid move" in result.error


class TestValidateAllGames:
    """Tests for validate_all_games function."""

    def test_validate_all_games_empty_db(self):
        """Test validating empty database."""
        with patch("app.db.validation.GameReplayDB") as mock_db_class:
            mock_db = MagicMock()
            mock_db.query_games.return_value = []
            mock_db_class.return_value = mock_db

            results = validate_all_games("/path/db.db")

        assert results == []

    def test_validate_all_games_multiple(self):
        """Test validating multiple games."""
        with patch("app.db.validation.GameReplayDB") as mock_db_class:
            mock_db = MagicMock()
            mock_db.query_games.return_value = [
                {"game_id": "game-1"},
                {"game_id": "game-2"},
            ]
            mock_db.get_initial_state.return_value = MagicMock()
            mock_db.get_moves.return_value = []
            mock_db_class.return_value = mock_db

            results = validate_all_games("/path/db.db")

        assert len(results) == 2
        assert results[0].game_id == "game-1"
        assert results[1].game_id == "game-2"

    def test_validate_all_games_max_limit(self):
        """Test max_games parameter."""
        with patch("app.db.validation.GameReplayDB") as mock_db_class:
            mock_db = MagicMock()
            mock_db.query_games.return_value = [{"game_id": "game-1"}]
            mock_db.get_initial_state.return_value = MagicMock()
            mock_db.get_moves.return_value = []
            mock_db_class.return_value = mock_db

            validate_all_games("/path/db.db", max_games=5)

            mock_db.query_games.assert_called_with(limit=5)


class TestExportFixture:
    """Tests for export_fixture function."""

    def test_export_fixture_valid_result_returns_none(self):
        """Test export returns None for valid results."""
        result = ValidationResult(
            game_id="valid-game",
            valid=True,
            total_moves=10,
            validated_moves=10,
        )

        path = export_fixture(result)
        assert path is None

    def test_export_fixture_creates_directory(self, tmp_path):
        """Test export creates output directory."""
        result = ValidationResult(
            game_id="div-game",
            valid=False,
            total_moves=10,
            validated_moves=5,
            divergence_move=5,
        )

        output_dir = tmp_path / "new_fixtures"
        path = export_fixture(result, output_dir=str(output_dir))

        assert output_dir.exists()
        assert path is not None
        assert Path(path).exists()

    def test_export_fixture_file_content(self, tmp_path):
        """Test exported fixture contains expected data."""
        mock_initial = MagicMock()
        mock_move = MagicMock()
        mock_move.model_dump_json.return_value = '{"type": "placement", "cell": 42}'

        result = ValidationResult(
            game_id="content-game",
            valid=False,
            total_moves=20,
            validated_moves=15,
            divergence_move=15,
            expected_hash="exp_hash",
            computed_hash="comp_hash",
            move_at_divergence=mock_move,
            moves_up_to_divergence=[mock_move],
        )

        with patch("app.db.validation._serialize_state", return_value='{"state": "data"}'):
            result.initial_state = mock_initial
            path = export_fixture(result, output_dir=str(tmp_path))

        assert path is not None
        with open(path) as f:
            data = json.load(f)

        assert data["game_id"] == "content-game"
        assert data["divergence_move"] == 15
        assert data["expected_hash"] == "exp_hash"
        assert data["computed_hash"] == "comp_hash"

    def test_export_fixture_filename_format(self, tmp_path):
        """Test fixture filename format."""
        result = ValidationResult(
            game_id="test-123",
            valid=False,
            total_moves=10,
            validated_moves=7,
            divergence_move=7,
        )

        path = export_fixture(result, output_dir=str(tmp_path))

        assert path is not None
        assert "divergence_test-123_move7.json" in path


class TestValidateDatabaseSummary:
    """Tests for validate_database_summary function."""

    def test_summary_all_valid(self):
        """Test summary for database with all valid games."""
        with patch("app.db.validation.validate_all_games") as mock_validate:
            mock_validate.return_value = [
                ValidationResult("g1", True, 10, 10),
                ValidationResult("g2", True, 20, 20),
            ]

            summary = validate_database_summary("/path/db.db", sample_size=2)

        assert summary["total_validated"] == 2
        assert summary["valid_games"] == 2
        assert summary["invalid_games"] == 0
        assert summary["validation_rate"] == 1.0

    def test_summary_with_invalid_games(self):
        """Test summary with some invalid games."""
        with patch("app.db.validation.validate_all_games") as mock_validate:
            mock_validate.return_value = [
                ValidationResult("g1", True, 10, 10),
                ValidationResult("g2", False, 20, 15, divergence_move=15),
                ValidationResult("g3", False, 30, 5, divergence_move=5, error="Error"),
            ]

            summary = validate_database_summary("/path/db.db", sample_size=3)

        assert summary["total_validated"] == 3
        assert summary["valid_games"] == 1
        assert summary["invalid_games"] == 2
        assert summary["games_with_errors"] == 1
        assert len(summary["divergent_games"]) == 2

    def test_summary_divergent_games_list(self):
        """Test divergent_games list contains expected data."""
        with patch("app.db.validation.validate_all_games") as mock_validate:
            mock_validate.return_value = [
                ValidationResult(
                    "bad-game",
                    False,
                    50,
                    25,
                    divergence_move=25,
                    error="Hash mismatch"
                ),
            ]

            summary = validate_database_summary("/path/db.db")

        assert len(summary["divergent_games"]) == 1
        div = summary["divergent_games"][0]
        assert div["game_id"] == "bad-game"
        assert div["divergence_move"] == 25
        assert div["error"] == "Hash mismatch"

    def test_summary_empty_database(self):
        """Test summary for empty database."""
        with patch("app.db.validation.validate_all_games") as mock_validate:
            mock_validate.return_value = []

            summary = validate_database_summary("/path/db.db")

        assert summary["total_validated"] == 0
        assert summary["valid_games"] == 0
        assert summary["validation_rate"] == 0


class TestGetSnapshotsWithHashes:
    """Tests for _get_snapshots_with_hashes helper."""

    def test_get_snapshots_returns_dict(self):
        """Test snapshots are returned as dict mapping move_number to data."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = [
            {"move_number": 0, "state_hash": "hash0", "compressed": 0},
            {"move_number": 5, "state_hash": "hash5", "compressed": 1},
        ]
        mock_db._get_conn.return_value = mock_conn

        result = _get_snapshots_with_hashes(mock_db, "game-123")

        assert 0 in result
        assert 5 in result
        assert result[0]["hash"] == "hash0"
        assert result[5]["hash"] == "hash5"
        assert result[5]["compressed"] is True

    def test_get_snapshots_empty(self):
        """Test empty snapshots for game without snapshots."""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_db._get_conn.return_value = mock_conn

        result = _get_snapshots_with_hashes(mock_db, "no-snapshots")

        assert result == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
