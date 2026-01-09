#!/usr/bin/env python3
"""Unit tests for app.db.move_data_validator module (January 2026).

Tests the centralized move data validation system that enforces minimum move
requirements across all entry points (create, merge, sync, export).

Tests cover:
- MIN_MOVES_REQUIRED constant
- MoveValidationResult dataclass
- DatabaseValidationResult dataclass
- MoveDataValidator class methods
- validate_database_for_training convenience function
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from app.db.move_data_validator import (
    MIN_MOVES_REQUIRED,
    DatabaseValidationResult,
    MoveDataValidator,
    MoveValidationResult,
    validate_database_for_training,
)


class TestConstants:
    """Tests for module constants."""

    def test_min_moves_required_is_5(self):
        """MIN_MOVES_REQUIRED should be 5 (the minimum for valid training data)."""
        assert MIN_MOVES_REQUIRED == 5

    def test_min_moves_required_is_positive(self):
        """MIN_MOVES_REQUIRED must be positive."""
        assert MIN_MOVES_REQUIRED > 0


class TestMoveValidationResult:
    """Tests for MoveValidationResult dataclass."""

    def test_valid_result_creation(self):
        """Test creating a valid result."""
        result = MoveValidationResult(
            game_id="game-123",
            is_valid=True,
            move_count=10,
        )
        assert result.game_id == "game-123"
        assert result.is_valid is True
        assert result.move_count == 10
        assert result.error_message is None

    def test_invalid_result_creation(self):
        """Test creating an invalid result with error message."""
        result = MoveValidationResult(
            game_id="bad-game",
            is_valid=False,
            move_count=3,
            error_message="Game has only 3 moves (minimum required: 5)",
        )
        assert result.is_valid is False
        assert result.move_count == 3
        assert result.error_message is not None

    def test_bool_conversion_valid(self):
        """Valid result should be truthy."""
        result = MoveValidationResult(game_id="test", is_valid=True, move_count=10)
        assert bool(result) is True

    def test_bool_conversion_invalid(self):
        """Invalid result should be falsy."""
        result = MoveValidationResult(game_id="test", is_valid=False, move_count=2)
        assert bool(result) is False


class TestDatabaseValidationResult:
    """Tests for DatabaseValidationResult dataclass."""

    def test_valid_database_result(self):
        """Test result for valid database."""
        result = DatabaseValidationResult(
            db_path=Path("/test/db.db"),
            valid_count=100,
            invalid_count=0,
            total_games=100,
            has_game_moves_table=True,
        )
        assert result.is_valid is True
        assert result.validation_rate == 1.0

    def test_invalid_database_result(self):
        """Test result for database with invalid games."""
        result = DatabaseValidationResult(
            db_path=Path("/test/db.db"),
            valid_count=80,
            invalid_count=20,
            total_games=100,
            has_game_moves_table=True,
        )
        assert result.is_valid is False
        assert result.validation_rate == 0.8

    def test_no_game_moves_table(self):
        """Database without game_moves table is invalid."""
        result = DatabaseValidationResult(
            db_path=Path("/test/db.db"),
            valid_count=0,
            invalid_count=0,
            total_games=50,
            has_game_moves_table=False,
        )
        assert result.is_valid is False

    def test_empty_database(self):
        """Empty database has 0 validation rate."""
        result = DatabaseValidationResult(
            db_path=Path("/test/db.db"),
            valid_count=0,
            invalid_count=0,
            total_games=0,
            has_game_moves_table=True,
        )
        assert result.validation_rate == 0.0

    def test_bool_conversion(self):
        """Test bool conversion delegates to is_valid."""
        valid_result = DatabaseValidationResult(
            db_path=Path("/test/db.db"),
            valid_count=10,
            invalid_count=0,
            total_games=10,
            has_game_moves_table=True,
        )
        assert bool(valid_result) is True

        invalid_result = DatabaseValidationResult(
            db_path=Path("/test/db.db"),
            valid_count=5,
            invalid_count=5,
            total_games=10,
            has_game_moves_table=True,
        )
        assert bool(invalid_result) is False


class TestMoveDataValidatorStaticMethods:
    """Tests for MoveDataValidator static methods with mock databases."""

    @pytest.fixture
    def valid_db(self, tmp_path: Path):
        """Create a valid database with games and moves."""
        db_path = tmp_path / "valid.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT,
                num_players INTEGER
            );
            CREATE TABLE game_moves (
                game_id TEXT,
                move_number INTEGER,
                move_data TEXT,
                PRIMARY KEY (game_id, move_number)
            );

            -- Game with 10 moves (valid)
            INSERT INTO games (game_id, board_type, num_players) VALUES ('game1', 'hex8', 2);
            INSERT INTO game_moves (game_id, move_number, move_data) VALUES
                ('game1', 1, 'move1'), ('game1', 2, 'move2'), ('game1', 3, 'move3'),
                ('game1', 4, 'move4'), ('game1', 5, 'move5'), ('game1', 6, 'move6'),
                ('game1', 7, 'move7'), ('game1', 8, 'move8'), ('game1', 9, 'move9'),
                ('game1', 10, 'move10');

            -- Game with 6 moves (valid - exactly meets threshold with 1 margin)
            INSERT INTO games (game_id, board_type, num_players) VALUES ('game2', 'hex8', 2);
            INSERT INTO game_moves (game_id, move_number, move_data) VALUES
                ('game2', 1, 'move1'), ('game2', 2, 'move2'), ('game2', 3, 'move3'),
                ('game2', 4, 'move4'), ('game2', 5, 'move5'), ('game2', 6, 'move6');
        """)
        conn.commit()
        conn.close()
        return db_path

    @pytest.fixture
    def invalid_db(self, tmp_path: Path):
        """Create a database with games that have insufficient moves."""
        db_path = tmp_path / "invalid.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT,
                num_players INTEGER
            );
            CREATE TABLE game_moves (
                game_id TEXT,
                move_number INTEGER,
                move_data TEXT,
                PRIMARY KEY (game_id, move_number)
            );

            -- Game with 3 moves (invalid - below threshold)
            INSERT INTO games (game_id, board_type, num_players) VALUES ('game1', 'hex8', 2);
            INSERT INTO game_moves (game_id, move_number, move_data) VALUES
                ('game1', 1, 'move1'), ('game1', 2, 'move2'), ('game1', 3, 'move3');

            -- Game with 0 moves (invalid)
            INSERT INTO games (game_id, board_type, num_players) VALUES ('game2', 'hex8', 2);
        """)
        conn.commit()
        conn.close()
        return db_path

    @pytest.fixture
    def metadata_only_db(self, tmp_path: Path):
        """Create a metadata-only database (no game_moves table)."""
        db_path = tmp_path / "metadata_only.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE games (
                game_id TEXT PRIMARY KEY,
                board_type TEXT,
                num_players INTEGER
            );

            INSERT INTO games (game_id, board_type, num_players) VALUES ('game1', 'hex8', 2);
            INSERT INTO games (game_id, board_type, num_players) VALUES ('game2', 'hex8', 2);
        """)
        conn.commit()
        conn.close()
        return db_path

    def test_has_game_moves_table_true(self, valid_db: Path):
        """Test detecting game_moves table exists."""
        conn = sqlite3.connect(str(valid_db))
        assert MoveDataValidator.has_game_moves_table(conn) is True
        conn.close()

    def test_has_game_moves_table_false(self, metadata_only_db: Path):
        """Test detecting game_moves table is missing."""
        conn = sqlite3.connect(str(metadata_only_db))
        assert MoveDataValidator.has_game_moves_table(conn) is False
        conn.close()

    def test_has_games_table_true(self, valid_db: Path):
        """Test detecting games table exists."""
        conn = sqlite3.connect(str(valid_db))
        assert MoveDataValidator.has_games_table(conn) is True
        conn.close()

    def test_validate_game_valid(self, valid_db: Path):
        """Test validating a game with sufficient moves."""
        conn = sqlite3.connect(str(valid_db))
        result = MoveDataValidator.validate_game(conn, "game1")
        assert result.is_valid is True
        assert result.move_count == 10
        assert result.error_message is None
        conn.close()

    def test_validate_game_exactly_at_threshold(self, valid_db: Path):
        """Test validating a game with exactly MIN_MOVES_REQUIRED moves."""
        # game2 has 6 moves, which is >= 5 (MIN_MOVES_REQUIRED)
        conn = sqlite3.connect(str(valid_db))
        result = MoveDataValidator.validate_game(conn, "game2")
        assert result.is_valid is True
        assert result.move_count == 6
        conn.close()

    def test_validate_game_invalid(self, invalid_db: Path):
        """Test validating a game with insufficient moves."""
        conn = sqlite3.connect(str(invalid_db))
        result = MoveDataValidator.validate_game(conn, "game1")
        assert result.is_valid is False
        assert result.move_count == 3
        assert result.error_message is not None
        assert "3 moves" in result.error_message
        conn.close()

    def test_validate_game_no_moves(self, invalid_db: Path):
        """Test validating a game with zero moves."""
        conn = sqlite3.connect(str(invalid_db))
        result = MoveDataValidator.validate_game(conn, "game2")
        assert result.is_valid is False
        assert result.move_count == 0
        conn.close()

    def test_validate_database_all_valid(self, valid_db: Path):
        """Test validating a database where all games are valid."""
        result = MoveDataValidator.validate_database(valid_db)
        assert result.is_valid is True
        assert result.valid_count == 2
        assert result.invalid_count == 0
        assert result.total_games == 2
        assert result.has_game_moves_table is True

    def test_validate_database_with_invalid_games(self, invalid_db: Path):
        """Test validating a database with invalid games."""
        result = MoveDataValidator.validate_database(invalid_db)
        assert result.is_valid is False
        assert result.valid_count == 0
        assert result.invalid_count == 2
        assert result.total_games == 2
        assert len(result.errors) == 2

    def test_validate_database_metadata_only(self, metadata_only_db: Path):
        """Test validating a metadata-only database."""
        result = MoveDataValidator.validate_database(metadata_only_db)
        assert result.is_valid is False
        assert result.has_game_moves_table is False
        assert result.total_games == 2  # Games exist, but no moves

    def test_validate_database_nonexistent(self, tmp_path: Path):
        """Test validating a non-existent database."""
        result = MoveDataValidator.validate_database(tmp_path / "nonexistent.db")
        assert result.is_valid is False
        assert "not found" in result.errors[0].lower()

    def test_get_games_with_moves(self, valid_db: Path):
        """Test getting list of games with sufficient moves."""
        conn = sqlite3.connect(str(valid_db))
        game_ids = MoveDataValidator.get_games_with_moves(conn)
        assert len(game_ids) == 2
        assert "game1" in game_ids
        assert "game2" in game_ids
        conn.close()

    def test_get_games_without_moves(self, invalid_db: Path):
        """Test getting list of games without sufficient moves."""
        conn = sqlite3.connect(str(invalid_db))
        invalid_games = MoveDataValidator.get_games_without_moves(conn)
        assert len(invalid_games) == 2
        # Check that both games are reported with their move counts
        game_ids = [g[0] for g in invalid_games]
        assert "game1" in game_ids
        assert "game2" in game_ids
        conn.close()

    def test_get_games_without_moves_metadata_only(self, metadata_only_db: Path):
        """Test get_games_without_moves on metadata-only database."""
        conn = sqlite3.connect(str(metadata_only_db))
        invalid_games = MoveDataValidator.get_games_without_moves(conn)
        # All games should be invalid (0 moves each)
        assert len(invalid_games) == 2
        for game_id, move_count in invalid_games:
            assert move_count == 0
        conn.close()


class TestValidateDatabaseForTraining:
    """Tests for validate_database_for_training convenience function."""

    @pytest.fixture
    def valid_training_db(self, tmp_path: Path):
        """Create a valid training database."""
        db_path = tmp_path / "training.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE games (game_id TEXT PRIMARY KEY);
            CREATE TABLE game_moves (game_id TEXT, move_number INTEGER, PRIMARY KEY (game_id, move_number));

            INSERT INTO games (game_id) VALUES ('game1');
            INSERT INTO game_moves (game_id, move_number) VALUES
                ('game1', 1), ('game1', 2), ('game1', 3), ('game1', 4),
                ('game1', 5), ('game1', 6), ('game1', 7), ('game1', 8);
        """)
        conn.commit()
        conn.close()
        return db_path

    @pytest.fixture
    def invalid_training_db(self, tmp_path: Path):
        """Create an invalid training database."""
        db_path = tmp_path / "invalid_training.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE games (game_id TEXT PRIMARY KEY);
            CREATE TABLE game_moves (game_id TEXT, move_number INTEGER, PRIMARY KEY (game_id, move_number));

            INSERT INTO games (game_id) VALUES ('game1');
            INSERT INTO game_moves (game_id, move_number) VALUES ('game1', 1), ('game1', 2);
        """)
        conn.commit()
        conn.close()
        return db_path

    def test_validate_valid_database_non_strict(self, valid_training_db: Path):
        """Test validating valid database in non-strict mode."""
        result = validate_database_for_training(valid_training_db, strict=False)
        assert result.is_valid is True

    def test_validate_invalid_database_strict(self, invalid_training_db: Path):
        """Test validating invalid database in strict mode raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_database_for_training(invalid_training_db, strict=True)
        # Error message contains "without sufficient move data"
        assert "without sufficient move data" in str(exc_info.value).lower()

    def test_validate_invalid_database_non_strict(self, invalid_training_db: Path):
        """Test validating invalid database in non-strict mode returns result."""
        result = validate_database_for_training(invalid_training_db, strict=False)
        assert result.is_valid is False
        assert result.invalid_count > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_validate_game_nonexistent_game(self, tmp_path: Path):
        """Test validating a game that doesn't exist in the database."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE games (game_id TEXT PRIMARY KEY);
            CREATE TABLE game_moves (game_id TEXT, move_number INTEGER);
        """)
        conn.commit()

        result = MoveDataValidator.validate_game(conn, "nonexistent")
        assert result.is_valid is False
        assert result.move_count == 0
        conn.close()

    def test_validate_game_with_custom_min_moves(self, tmp_path: Path):
        """Test validating with custom minimum moves threshold."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE games (game_id TEXT PRIMARY KEY);
            CREATE TABLE game_moves (game_id TEXT, move_number INTEGER);

            INSERT INTO games (game_id) VALUES ('game1');
            INSERT INTO game_moves (game_id, move_number) VALUES ('game1', 1), ('game1', 2);
        """)
        conn.commit()

        # With min_moves=3, 2 moves is invalid
        result = MoveDataValidator.validate_game(conn, "game1", min_moves=3)
        assert result.is_valid is False

        # With min_moves=2, 2 moves is valid
        result = MoveDataValidator.validate_game(conn, "game1", min_moves=2)
        assert result.is_valid is True

        conn.close()

    def test_database_with_mixed_valid_invalid(self, tmp_path: Path):
        """Test database with a mix of valid and invalid games."""
        db_path = tmp_path / "mixed.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE games (game_id TEXT PRIMARY KEY);
            CREATE TABLE game_moves (game_id TEXT, move_number INTEGER);

            -- Valid game (10 moves)
            INSERT INTO games (game_id) VALUES ('valid1');
            INSERT INTO game_moves (game_id, move_number) VALUES
                ('valid1', 1), ('valid1', 2), ('valid1', 3), ('valid1', 4), ('valid1', 5),
                ('valid1', 6), ('valid1', 7), ('valid1', 8), ('valid1', 9), ('valid1', 10);

            -- Invalid game (2 moves)
            INSERT INTO games (game_id) VALUES ('invalid1');
            INSERT INTO game_moves (game_id, move_number) VALUES ('invalid1', 1), ('invalid1', 2);

            -- Valid game (5 moves - exactly at threshold)
            INSERT INTO games (game_id) VALUES ('valid2');
            INSERT INTO game_moves (game_id, move_number) VALUES
                ('valid2', 1), ('valid2', 2), ('valid2', 3), ('valid2', 4), ('valid2', 5);

            -- Invalid game (4 moves - one below threshold)
            INSERT INTO games (game_id) VALUES ('invalid2');
            INSERT INTO game_moves (game_id, move_number) VALUES
                ('invalid2', 1), ('invalid2', 2), ('invalid2', 3), ('invalid2', 4);
        """)
        conn.commit()
        conn.close()

        result = MoveDataValidator.validate_database(db_path)
        assert result.valid_count == 2  # valid1 and valid2
        assert result.invalid_count == 2  # invalid1 and invalid2
        assert result.total_games == 4
        assert result.validation_rate == 0.5

    def test_max_errors_limit(self, tmp_path: Path):
        """Test that max_errors parameter limits error collection."""
        db_path = tmp_path / "many_invalid.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE games (game_id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE game_moves (game_id TEXT, move_number INTEGER)")

        # Create 200 invalid games
        for i in range(200):
            conn.execute(f"INSERT INTO games (game_id) VALUES ('game{i}')")
            # Only 1 move each (invalid)
            conn.execute(f"INSERT INTO game_moves (game_id, move_number) VALUES ('game{i}', 1)")

        conn.commit()
        conn.close()

        # With max_errors=10, should only collect 10 error messages
        result = MoveDataValidator.validate_database(db_path, max_errors=10)
        assert result.invalid_count == 200  # All counted
        assert len(result.errors) == 10  # But only 10 errors recorded
