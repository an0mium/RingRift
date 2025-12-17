"""Unified game recording with canonical naming.

This module provides a standardized interface for recording self-play games
across all sources (self-play, soak, CMA-ES, gauntlet, tournament scripts).

CANONICAL FORMAT:
    - Board types: "square8", "square19", "hex8", "hexagonal"
    - Config keys: "{board_type}_{num_players}p" (e.g., "square8_2p")
    - Database names: "{prefix}_{config_key}.db" (e.g., "selfplay_square8_2p.db")

Usage:
    from app.db.unified_recording import UnifiedGameRecorder, RecordingConfig

    config = RecordingConfig(
        board_type="sq8",  # Will be normalized to "square8"
        num_players=2,
        source="self_play",
    )

    with UnifiedGameRecorder(config, initial_state) as recorder:
        for move in game_loop():
            recorder.add_move(move, state_after)
        recorder.finalize(final_state)
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.db import GameReplayDB
from app.db.recording import (
    GameRecorder,
    get_or_create_db,
    is_recording_enabled,
    record_completed_game,
    record_completed_game_with_parity_check,
    _augment_metadata_with_env,
)
from app.models import BoardType, GameState, Move
from app.utils.canonical_naming import (
    normalize_board_type,
    make_config_key,
    normalize_database_filename,
    get_board_type_enum,
)


# Canonical source identifiers
class RecordSource:
    """Canonical source identifiers for game records."""

    SELF_PLAY = "self_play"
    SOAK_TEST = "soak_test"
    CMAES = "cmaes"
    GAUNTLET = "gauntlet"
    TOURNAMENT = "tournament"
    TRAINING = "training"
    MANUAL = "manual"


@dataclass
class RecordingConfig:
    """Configuration for unified game recording.

    All board type inputs are normalized to canonical values automatically.
    """

    board_type: str  # Will be normalized to canonical value
    num_players: int
    source: str = RecordSource.SELF_PLAY

    # Optional metadata
    difficulty: Optional[int] = None
    engine_mode: Optional[str] = None
    model_id: Optional[str] = None
    generation: Optional[int] = None
    candidate_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # Database configuration
    db_path: Optional[str] = None  # If None, auto-generated from board_type/num_players
    db_prefix: str = "selfplay"
    db_dir: str = "data/games"

    # Recording options
    store_history_entries: bool = True
    snapshot_interval: Optional[int] = None  # None = use env default (20)
    parity_mode: Optional[str] = None  # None = use env default
    fsm_validation: bool = False

    def __post_init__(self):
        # Normalize board type to canonical value
        self.board_type = normalize_board_type(self.board_type)

        # Validate num_players
        if not 2 <= self.num_players <= 4:
            raise ValueError(f"num_players must be 2-4, got {self.num_players}")

    @property
    def config_key(self) -> str:
        """Get the canonical config key (e.g., 'square8_2p')."""
        return make_config_key(self.board_type, self.num_players)

    @property
    def board_type_enum(self) -> BoardType:
        """Get the BoardType enum value."""
        return get_board_type_enum(self.board_type)

    def get_db_path(self) -> str:
        """Get the database path, auto-generating if not specified."""
        if self.db_path:
            return self.db_path

        filename = normalize_database_filename(
            self.board_type,
            self.num_players,
            prefix=self.db_prefix,
        )
        return str(Path(self.db_dir) / filename)

    def build_metadata(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build standardized metadata dict for this recording config."""
        metadata: Dict[str, Any] = {
            "source": self.source,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "config_key": self.config_key,
        }

        # Add optional fields if set
        if self.difficulty is not None:
            metadata["difficulty"] = self.difficulty
        if self.engine_mode:
            metadata["engine_mode"] = self.engine_mode
        if self.model_id:
            metadata["model_id"] = self.model_id
        if self.generation is not None:
            metadata["generation"] = self.generation
        if self.candidate_id:
            metadata["candidate_id"] = self.candidate_id
        if self.tags:
            metadata["tags"] = self.tags

        # Merge extra metadata
        if extra:
            metadata.update(extra)

        return metadata


class UnifiedGameRecorder:
    """Unified context manager for recording games with canonical naming.

    This is the RECOMMENDED way to record games across all scripts.
    It enforces canonical board type naming and consistent metadata.

    Usage:
        config = RecordingConfig(board_type="sq8", num_players=2, source="self_play")

        with UnifiedGameRecorder(config, initial_state) as recorder:
            for move in game_loop():
                recorder.add_move(move, state_after)
            recorder.finalize(final_state)
    """

    def __init__(
        self,
        config: RecordingConfig,
        initial_state: GameState,
        game_id: Optional[str] = None,
    ):
        self.config = config
        self.initial_state = initial_state
        self.game_id = game_id or str(uuid.uuid4())
        self._db: Optional[GameReplayDB] = None
        self._recorder: Optional[GameRecorder] = None
        self._finalized = False

    def __enter__(self) -> "UnifiedGameRecorder":
        if not is_recording_enabled():
            return self

        db_path = self.config.get_db_path()
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._db = GameReplayDB(db_path)

        self._recorder = GameRecorder(self._db, self.initial_state, self.game_id)
        self._recorder.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._recorder is not None:
            self._recorder.__exit__(exc_type, exc_val, exc_tb)
        return False

    def add_move(
        self,
        move: Move,
        state_after: Optional[GameState] = None,
        state_before: Optional[GameState] = None,
        available_moves_count: Optional[int] = None,
        engine_eval: Optional[float] = None,
        engine_depth: Optional[int] = None,
    ) -> None:
        """Add a move to the game record."""
        if self._recorder is None:
            return

        fsm_valid = None
        fsm_error_code = None

        # Optional FSM validation
        if self.config.fsm_validation and state_before:
            from app.db.recording import validate_move_fsm

            fsm_valid, fsm_error_code = validate_move_fsm(state_before, move)

        self._recorder.add_move(
            move,
            state_after=state_after,
            state_before=state_before,
            available_moves_count=available_moves_count,
            engine_eval=engine_eval,
            engine_depth=engine_depth,
            fsm_valid=fsm_valid,
            fsm_error_code=fsm_error_code,
        )

    def finalize(
        self,
        final_state: GameState,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Finalize the game recording with standardized metadata."""
        if self._recorder is None:
            return

        metadata = self.config.build_metadata(extra_metadata)
        self._recorder.finalize(final_state, metadata)
        self._finalized = True


def record_game_unified(
    config: RecordingConfig,
    initial_state: GameState,
    final_state: GameState,
    moves: List[Move],
    extra_metadata: Optional[Dict[str, Any]] = None,
    game_id: Optional[str] = None,
    with_parity_check: bool = False,
) -> Optional[str]:
    """Record a completed game with canonical naming (one-shot).

    This is the recommended function for recording completed games
    across all scripts.

    Args:
        config: RecordingConfig with board type, num_players, source, etc.
        initial_state: GameState at the start of the game
        final_state: GameState at the end of the game
        moves: List of all moves in the game
        extra_metadata: Additional metadata to merge
        game_id: Optional custom game ID
        with_parity_check: If True, validate parity with TS engine

    Returns:
        The game ID that was stored, or None if recording is disabled
    """
    if not is_recording_enabled():
        return None

    db_path = config.get_db_path()
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    db = GameReplayDB(db_path)

    metadata = config.build_metadata(extra_metadata)
    gid = game_id or str(uuid.uuid4())

    if with_parity_check:
        return record_completed_game_with_parity_check(
            db=db,
            initial_state=initial_state,
            final_state=final_state,
            moves=moves,
            metadata=metadata,
            game_id=gid,
            parity_mode=config.parity_mode,
            store_history_entries=config.store_history_entries,
            snapshot_interval=config.snapshot_interval,
        )
    else:
        return record_completed_game(
            db=db,
            initial_state=initial_state,
            final_state=final_state,
            moves=moves,
            metadata=metadata,
            game_id=gid,
            store_history_entries=config.store_history_entries,
            snapshot_interval=config.snapshot_interval,
        )


def get_unified_db(
    board_type: str,
    num_players: int,
    prefix: str = "selfplay",
    db_dir: str = "data/games",
) -> Optional[GameReplayDB]:
    """Get a GameReplayDB with canonical naming.

    Args:
        board_type: Board type (will be normalized to canonical value)
        num_players: Number of players (2-4)
        prefix: Database filename prefix (default: "selfplay")
        db_dir: Directory for database files (default: "data/games")

    Returns:
        GameReplayDB instance or None if recording is disabled
    """
    if not is_recording_enabled():
        return None

    canonical_board = normalize_board_type(board_type)
    filename = normalize_database_filename(canonical_board, num_players, prefix=prefix)
    db_path = str(Path(db_dir) / filename)

    os.makedirs(db_dir, exist_ok=True)
    return GameReplayDB(db_path)
