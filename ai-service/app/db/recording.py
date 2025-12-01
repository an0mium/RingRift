"""Game recording utilities for integrating GameReplayDB into scripts.

This module provides high-level helper functions for recording games to the
SQLite database. It is intended to be used by self-play, training, and
analysis scripts.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from app.db import GameReplayDB, GameWriter
from app.models import GameState, GameStatus, Move


class GameRecorder:
    """Context manager for recording a single game to the database.

    Usage:
        db = GameReplayDB("data/games.db")
        with GameRecorder(db, initial_state) as recorder:
            for move in game_loop():
                recorder.add_move(move)
            recorder.finalize(final_state, {"source": "self_play"})
    """

    def __init__(
        self,
        db: GameReplayDB,
        initial_state: GameState,
        game_id: Optional[str] = None,
    ):
        self.db = db
        self.initial_state = initial_state
        self.game_id = game_id or str(uuid.uuid4())
        self._writer: Optional[GameWriter] = None
        self._finalized = False

    def __enter__(self) -> "GameRecorder":
        self._writer = self.db.store_game_incremental(
            self.game_id,
            self.initial_state,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and self._writer is not None:
            # Exception occurred - abort the game recording
            self._writer.abort()
        elif not self._finalized and self._writer is not None:
            # Context exited without finalizing - abort
            self._writer.abort()
        return False  # Don't suppress exceptions

    def add_move(self, move: Move) -> None:
        """Add a move to the game record."""
        if self._writer is None:
            raise RuntimeError("GameRecorder not entered as context manager")
        self._writer.add_move(move)

    def finalize(
        self,
        final_state: GameState,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Finalize the game recording with the final state and metadata."""
        if self._writer is None:
            raise RuntimeError("GameRecorder not entered as context manager")
        self._writer.finalize(final_state, metadata)
        self._finalized = True


def record_completed_game(
    db: GameReplayDB,
    initial_state: GameState,
    final_state: GameState,
    moves: List[Move],
    metadata: Optional[Dict[str, Any]] = None,
    game_id: Optional[str] = None,
) -> str:
    """Record a completed game in one shot.

    This is a convenience function for scripts that collect moves in a list
    and want to store them all at once after the game ends.

    Args:
        db: The GameReplayDB instance
        initial_state: GameState at the start of the game
        final_state: GameState at the end of the game
        moves: List of all moves in the game
        metadata: Optional metadata dict (source, difficulty, etc.)
        game_id: Optional custom game ID

    Returns:
        The game ID that was stored
    """
    gid = game_id or str(uuid.uuid4())
    db.store_game(
        game_id=gid,
        initial_state=initial_state,
        final_state=final_state,
        moves=moves,
        metadata=metadata,
    )
    return gid


def get_or_create_db(
    db_path: Optional[str],
    default_path: str = "data/games/selfplay.db",
) -> Optional[GameReplayDB]:
    """Get or create a GameReplayDB instance.

    Args:
        db_path: Path to the database file, or None to disable recording
        default_path: Default path if db_path is empty string

    Returns:
        GameReplayDB instance or None if recording is disabled
    """
    if db_path is None:
        return None

    path = db_path if db_path else default_path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return GameReplayDB(path)
