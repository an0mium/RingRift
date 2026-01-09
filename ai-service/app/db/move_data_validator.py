"""Move data validation for game databases.

This module provides centralized validation to ensure all game records have
sufficient move data for training. Games without moves are useless for training
and should be rejected at every entry point (create, merge, sync, export).

Usage:
    from app.db.move_data_validator import (
        MoveDataValidator,
        MoveDataValidationError,
        MIN_MOVES_REQUIRED,
        MoveValidationResult,
        DatabaseValidationResult,
    )

    # Validate a single game
    result = MoveDataValidator.validate_game(conn, game_id)
    if not result.is_valid:
        raise ValueError(result.error_message)

    # Validate entire database
    result = MoveDataValidator.validate_database(db_path)
    if result.invalid_count > 0:
        raise ValueError(f"{result.invalid_count} games without sufficient moves")

    # Check if database has game_moves table
    if not MoveDataValidator.has_game_moves_table(conn):
        raise ValueError("Database is metadata-only (no game_moves table)")
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

logger = logging.getLogger(__name__)

# CRITICAL: Minimum moves required for a game to be valid for training
# Games with fewer moves are rejected at all entry points
# This prevents "orphan" games (metadata without move data) from polluting databases
MIN_MOVES_REQUIRED = 5


class MoveDataValidationError(Exception):
    """Raised when game data fails move validation.

    This exception is raised when export or training operations encounter
    games that are missing move data. It provides details about which games
    are affected to aid in debugging data quality issues.
    """

    def __init__(
        self,
        game_ids: list[str],
        message: str = "Games missing move data",
    ) -> None:
        self.game_ids = game_ids
        super().__init__(f"{message}: {len(game_ids)} games affected")

    def __repr__(self) -> str:
        return f"MoveDataValidationError(game_ids={len(self.game_ids)} games)"


@dataclass
class MoveValidationResult:
    """Result of validating a single game's move data."""

    game_id: str
    is_valid: bool
    move_count: int
    error_message: Optional[str] = None

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.is_valid


@dataclass
class DatabaseValidationResult:
    """Result of validating all games in a database."""

    db_path: Path
    valid_count: int
    invalid_count: int
    total_games: int
    has_game_moves_table: bool
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Database is valid if all games have sufficient moves."""
        return self.invalid_count == 0 and self.has_game_moves_table

    @property
    def validation_rate(self) -> float:
        """Percentage of games with valid move data."""
        if self.total_games == 0:
            return 0.0
        return self.valid_count / self.total_games

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.is_valid


class MoveDataValidator:
    """Central validator for move data requirements.

    All operations that create, merge, sync, or export games should use this
    validator to ensure move data completeness. Games without sufficient moves
    are useless for training and must be rejected.
    """

    @staticmethod
    def validate_game(
        conn: sqlite3.Connection,
        game_id: str,
        min_moves: int = MIN_MOVES_REQUIRED,
    ) -> MoveValidationResult:
        """Validate that a single game has sufficient move data.

        Args:
            conn: SQLite connection to database
            game_id: ID of game to validate
            min_moves: Minimum moves required (default: MIN_MOVES_REQUIRED)

        Returns:
            MoveValidationResult with validation status
        """
        try:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM game_moves WHERE game_id = ?",
                (game_id,),
            )
            move_count = cursor.fetchone()[0]
        except sqlite3.OperationalError as e:
            # Table doesn't exist
            return MoveValidationResult(
                game_id=game_id,
                is_valid=False,
                move_count=0,
                error_message=f"Cannot validate game {game_id}: {e}",
            )

        if move_count < min_moves:
            return MoveValidationResult(
                game_id=game_id,
                is_valid=False,
                move_count=move_count,
                error_message=(
                    f"Game {game_id} has only {move_count} moves "
                    f"(minimum required: {min_moves}). "
                    f"Games without sufficient move data cannot be used for training."
                ),
            )

        return MoveValidationResult(
            game_id=game_id,
            is_valid=True,
            move_count=move_count,
        )

    @staticmethod
    def validate_database(
        db_path: Path | str,
        min_moves: int = MIN_MOVES_REQUIRED,
        max_errors: int = 100,
    ) -> DatabaseValidationResult:
        """Validate all games in a database have sufficient move data.

        Args:
            db_path: Path to SQLite database
            min_moves: Minimum moves required per game
            max_errors: Maximum errors to collect before stopping

        Returns:
            DatabaseValidationResult with validation statistics
        """
        db_path = Path(db_path)

        if not db_path.exists():
            return DatabaseValidationResult(
                db_path=db_path,
                valid_count=0,
                invalid_count=0,
                total_games=0,
                has_game_moves_table=False,
                errors=[f"Database not found: {db_path}"],
            )

        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            return DatabaseValidationResult(
                db_path=db_path,
                valid_count=0,
                invalid_count=0,
                total_games=0,
                has_game_moves_table=False,
                errors=[f"Cannot open database: {e}"],
            )

        try:
            # Check if game_moves table exists
            has_moves_table = MoveDataValidator.has_game_moves_table(conn)
            if not has_moves_table:
                return DatabaseValidationResult(
                    db_path=db_path,
                    valid_count=0,
                    invalid_count=0,
                    total_games=MoveDataValidator._count_games(conn),
                    has_game_moves_table=False,
                    errors=[
                        "Database is metadata-only: no game_moves table. "
                        "Cannot use for training."
                    ],
                )

            # Get game counts with move data using a single efficient query
            cursor = conn.execute(
                """
                SELECT
                    g.game_id,
                    COALESCE(m.move_count, 0) as move_count
                FROM games g
                LEFT JOIN (
                    SELECT game_id, COUNT(*) as move_count
                    FROM game_moves
                    GROUP BY game_id
                ) m ON g.game_id = m.game_id
                """
            )

            valid_count = 0
            invalid_count = 0
            errors: list[str] = []

            for row in cursor:
                game_id = row["game_id"]
                move_count = row["move_count"]

                if move_count >= min_moves:
                    valid_count += 1
                else:
                    invalid_count += 1
                    if len(errors) < max_errors:
                        errors.append(
                            f"Game {game_id}: {move_count} moves (min: {min_moves})"
                        )

            total_games = valid_count + invalid_count

            return DatabaseValidationResult(
                db_path=db_path,
                valid_count=valid_count,
                invalid_count=invalid_count,
                total_games=total_games,
                has_game_moves_table=True,
                errors=errors,
            )

        finally:
            conn.close()

    @staticmethod
    def has_game_moves_table(conn: sqlite3.Connection) -> bool:
        """Check if database has the game_moves table.

        Databases without this table are "metadata-only" and cannot be used
        for training since they contain no move data.

        Args:
            conn: SQLite connection

        Returns:
            True if game_moves table exists
        """
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'"
        )
        return cursor.fetchone() is not None

    @staticmethod
    def has_games_table(conn: sqlite3.Connection) -> bool:
        """Check if database has the games table.

        Args:
            conn: SQLite connection

        Returns:
            True if games table exists
        """
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='games'"
        )
        return cursor.fetchone() is not None

    @staticmethod
    def _count_games(conn: sqlite3.Connection) -> int:
        """Count total games in database."""
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM games")
            return cursor.fetchone()[0]
        except sqlite3.OperationalError:
            return 0

    @staticmethod
    def get_games_with_moves(
        conn: sqlite3.Connection,
        min_moves: int = MIN_MOVES_REQUIRED,
    ) -> list[str]:
        """Get list of game IDs that have sufficient move data.

        Args:
            conn: SQLite connection
            min_moves: Minimum moves required

        Returns:
            List of valid game IDs
        """
        if not MoveDataValidator.has_game_moves_table(conn):
            return []

        cursor = conn.execute(
            """
            SELECT game_id
            FROM game_moves
            GROUP BY game_id
            HAVING COUNT(*) >= ?
            """,
            (min_moves,),
        )
        return [row[0] for row in cursor.fetchall()]

    @staticmethod
    def get_games_without_moves(
        conn: sqlite3.Connection,
        min_moves: int = MIN_MOVES_REQUIRED,
    ) -> list[tuple[str, int]]:
        """Get list of game IDs that lack sufficient move data.

        Args:
            conn: SQLite connection
            min_moves: Minimum moves required

        Returns:
            List of (game_id, move_count) tuples for invalid games
        """
        if not MoveDataValidator.has_game_moves_table(conn):
            # All games are invalid if no moves table
            try:
                cursor = conn.execute("SELECT game_id FROM games")
                return [(row[0], 0) for row in cursor.fetchall()]
            except sqlite3.OperationalError:
                return []

        cursor = conn.execute(
            """
            SELECT g.game_id, COALESCE(m.move_count, 0) as move_count
            FROM games g
            LEFT JOIN (
                SELECT game_id, COUNT(*) as move_count
                FROM game_moves
                GROUP BY game_id
            ) m ON g.game_id = m.game_id
            WHERE COALESCE(m.move_count, 0) < ?
            """,
            (min_moves,),
        )
        return [(row[0], row[1]) for row in cursor.fetchall()]


def validate_database_for_training(
    db_path: Path | str,
    strict: bool = True,
) -> DatabaseValidationResult:
    """Convenience function to validate a database for training use.

    Args:
        db_path: Path to SQLite database
        strict: If True, raises ValueError on invalid databases

    Returns:
        DatabaseValidationResult

    Raises:
        ValueError: If strict=True and database is invalid
    """
    result = MoveDataValidator.validate_database(db_path)

    if strict and not result.is_valid:
        if not result.has_game_moves_table:
            raise ValueError(
                f"Database {db_path} is metadata-only (no game_moves table). "
                f"Cannot use for training. This database contains {result.total_games} "
                f"games but no move data."
            )
        raise ValueError(
            f"Database {db_path} has {result.invalid_count} games without "
            f"sufficient move data ({result.valid_count} valid, "
            f"{result.validation_rate:.1%} validation rate). "
            f"First errors: {result.errors[:5]}"
        )

    return result
