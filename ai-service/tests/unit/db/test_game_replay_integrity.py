#!/usr/bin/env python3
"""Unit tests for move data integrity enforcement in game_replay.py (December 2025).

Tests the Phase 6 Move Data Integrity Enforcement:
- GameWriter.finalize() rejects games with no moves
- store_game() rejects games with fewer than MIN_MOVES_REQUIRED moves
- Placeholder games are cleaned up on exception
- v14 migration adds orphaned_games table and trigger

Test fixtures create temporary SQLite databases for testing without
affecting production data.
"""

import sqlite3
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.db.game_replay import GameReplayDB, SCHEMA_VERSION
from app.errors import InvalidGameError


class TestGameWriterIntegrityEnforcement:
    """Tests for GameWriter integrity enforcement."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary database for testing."""
        db_path = tmp_path / "test_game_replay.db"
        db = GameReplayDB(str(db_path))
        yield db
        db.close()

    @pytest.fixture
    def mock_game_state(self):
        """Create a mock game state for testing."""
        state = MagicMock()
        state.phase = "play"
        state.current_player = 0
        state.board = MagicMock()
        state.board.serialize.return_value = {"cells": []}
        state.players = [
            MagicMock(pieces=10, score=0, captured_pieces=0),
            MagicMock(pieces=10, score=0, captured_pieces=0),
        ]
        state.winner = None
        state.total_moves = 0
        state.serialize.return_value = {
            "phase": "play",
            "current_player": 0,
            "board": {"cells": []},
            "players": [],
        }
        return state

    @pytest.fixture
    def mock_move(self):
        """Create a mock move for testing."""
        move = MagicMock()
        move.move_type = "PLACE_RING"
        move.player = 0
        move.position = (0, 0)
        move.serialize.return_value = {
            "move_type": "PLACE_RING",
            "player": 0,
            "position": [0, 0],
        }
        return move

    def test_finalize_with_no_moves_raises_invalid_game_error(
        self, temp_db, mock_game_state
    ):
        """Test that finalize() raises InvalidGameError when no moves recorded."""
        with temp_db.incremental_writer(
            game_id="test-game-001",
            initial_state=mock_game_state,
            metadata={"board_type": "hex8", "num_players": 2},
        ) as writer:
            # Don't add any moves
            # Attempting to finalize should raise InvalidGameError
            with pytest.raises(InvalidGameError) as exc_info:
                writer.finalize(final_state=mock_game_state)

            assert "0 moves" in str(exc_info.value)
            assert "test-game-001" in str(exc_info.value)

    def test_finalize_with_moves_succeeds(
        self, temp_db, mock_game_state, mock_move
    ):
        """Test that finalize() succeeds when moves are recorded."""
        final_state = MagicMock()
        final_state.phase = "game_over"
        final_state.winner = 0
        final_state.serialize.return_value = {
            "phase": "game_over",
            "winner": 0,
        }

        with temp_db.incremental_writer(
            game_id="test-game-002",
            initial_state=mock_game_state,
            metadata={"board_type": "hex8", "num_players": 2},
        ) as writer:
            # Add a move
            writer.add_move(
                move=mock_move,
                state_before=mock_game_state,
                state_after=final_state,
            )
            # Finalize should succeed
            writer.finalize(final_state=final_state)

        # Verify game was stored
        games = temp_db.list_games()
        assert len(games) == 1
        assert games[0]["game_id"] == "test-game-002"

    def test_writer_cleans_up_placeholder_on_exception(
        self, temp_db, mock_game_state
    ):
        """Test that placeholder game is deleted when exception occurs."""
        game_id = "test-game-cleanup"

        class CustomError(Exception):
            pass

        try:
            with temp_db.incremental_writer(
                game_id=game_id,
                initial_state=mock_game_state,
                metadata={"board_type": "hex8", "num_players": 2},
            ) as writer:
                # Raise exception before finalizing
                raise CustomError("Simulated failure")
        except CustomError:
            pass

        # Verify placeholder was cleaned up
        games = temp_db.list_games()
        game_ids = [g["game_id"] for g in games]
        assert game_id not in game_ids


class TestStoreGameIntegrityEnforcement:
    """Tests for store_game() integrity enforcement."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary database for testing."""
        db_path = tmp_path / "test_store_game.db"
        db = GameReplayDB(str(db_path))
        yield db
        db.close()

    @pytest.fixture
    def mock_game_state(self):
        """Create a mock game state for testing."""
        state = MagicMock()
        state.serialize.return_value = {
            "phase": "play",
            "current_player": 0,
            "board": {"cells": []},
            "players": [],
        }
        return state

    @pytest.fixture
    def mock_move(self):
        """Create a mock move for testing."""
        move = MagicMock()
        move.serialize.return_value = {
            "move_type": "PLACE_RING",
            "player": 0,
            "position": [0, 0],
        }
        return move

    def test_store_game_with_no_moves_raises_invalid_game_error(
        self, temp_db, mock_game_state
    ):
        """Test that store_game() raises InvalidGameError with empty moves list."""
        with pytest.raises(InvalidGameError) as exc_info:
            temp_db.store_game(
                game_id="test-game-003",
                initial_state=mock_game_state,
                final_state=mock_game_state,
                moves=[],  # Empty moves list
                metadata={"board_type": "hex8", "num_players": 2},
            )

        assert "0 moves" in str(exc_info.value)

    def test_store_game_with_moves_succeeds(
        self, temp_db, mock_game_state, mock_move
    ):
        """Test that store_game() succeeds with valid moves."""
        temp_db.store_game(
            game_id="test-game-004",
            initial_state=mock_game_state,
            final_state=mock_game_state,
            moves=[mock_move],
            metadata={"board_type": "hex8", "num_players": 2},
        )

        # Verify game was stored
        games = temp_db.list_games()
        assert len(games) == 1
        assert games[0]["game_id"] == "test-game-004"


class TestSchemaVersion14Migration:
    """Tests for v14 schema migration."""

    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create a temporary database path."""
        return tmp_path / "test_migration.db"

    def test_schema_version_is_14(self):
        """Test that SCHEMA_VERSION is 14."""
        assert SCHEMA_VERSION == 14

    def test_fresh_database_has_orphaned_games_table(self, temp_db_path):
        """Test that fresh database has orphaned_games table."""
        db = GameReplayDB(str(temp_db_path))

        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='orphaned_games'"
        )
        result = cursor.fetchone()
        conn.close()
        db.close()

        assert result is not None
        assert result[0] == "orphaned_games"

    def test_fresh_database_has_enforcement_trigger(self, temp_db_path):
        """Test that fresh database has enforce_moves_on_complete trigger."""
        db = GameReplayDB(str(temp_db_path))

        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name='enforce_moves_on_complete'"
        )
        result = cursor.fetchone()
        conn.close()
        db.close()

        assert result is not None
        assert result[0] == "enforce_moves_on_complete"

    def test_trigger_blocks_completing_game_without_moves(self, temp_db_path):
        """Test that trigger prevents completing games with 0 moves."""
        db = GameReplayDB(str(temp_db_path))

        conn = sqlite3.connect(str(temp_db_path))
        # Insert a game with 0 moves
        conn.execute(
            """
            INSERT INTO games (game_id, board_type, num_players, total_moves, game_status, created_at)
            VALUES ('trigger-test', 'hex8', 2, 0, 'in_progress', datetime('now'))
        """
        )
        conn.commit()

        # Try to mark it as completed - should fail
        with pytest.raises(sqlite3.IntegrityError) as exc_info:
            conn.execute(
                "UPDATE games SET game_status = 'completed' WHERE game_id = 'trigger-test'"
            )
            conn.commit()

        assert "Cannot complete game without moves" in str(exc_info.value)
        conn.close()
        db.close()

    def test_trigger_allows_completing_game_with_moves(self, temp_db_path):
        """Test that trigger allows completing games with moves."""
        db = GameReplayDB(str(temp_db_path))

        conn = sqlite3.connect(str(temp_db_path))
        # Insert a game with moves
        conn.execute(
            """
            INSERT INTO games (game_id, board_type, num_players, total_moves, game_status, created_at)
            VALUES ('trigger-test-ok', 'hex8', 2, 10, 'in_progress', datetime('now'))
        """
        )
        conn.commit()

        # Mark it as completed - should succeed
        conn.execute(
            "UPDATE games SET game_status = 'completed' WHERE game_id = 'trigger-test-ok'"
        )
        conn.commit()

        # Verify it was updated
        cursor = conn.execute(
            "SELECT game_status FROM games WHERE game_id = 'trigger-test-ok'"
        )
        result = cursor.fetchone()
        assert result[0] == "completed"
        conn.close()
        db.close()


class TestOrphanedGamesTable:
    """Tests for orphaned_games table functionality."""

    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create a temporary database path."""
        return tmp_path / "test_orphaned.db"

    def test_can_insert_orphaned_game(self, temp_db_path):
        """Test that orphaned games can be inserted."""
        db = GameReplayDB(str(temp_db_path))

        conn = sqlite3.connect(str(temp_db_path))
        conn.execute(
            """
            INSERT INTO orphaned_games (game_id, detected_at, reason, original_status, board_type, num_players)
            VALUES ('orphan-001', datetime('now'), 'No move data found', 'completed', 'hex8', 2)
        """
        )
        conn.commit()

        cursor = conn.execute("SELECT * FROM orphaned_games WHERE game_id = 'orphan-001'")
        result = cursor.fetchone()
        assert result is not None
        assert result[0] == "orphan-001"
        conn.close()
        db.close()

    def test_orphaned_games_index_exists(self, temp_db_path):
        """Test that idx_orphaned_detected index exists."""
        db = GameReplayDB(str(temp_db_path))

        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_orphaned_detected'"
        )
        result = cursor.fetchone()
        conn.close()
        db.close()

        assert result is not None
        assert result[0] == "idx_orphaned_detected"
