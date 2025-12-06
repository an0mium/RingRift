"""Database module for RingRift game storage and replay."""

from app.db.game_replay import GameReplayDB, GameWriter
from app.db.recording import (
    GameRecorder,
    record_completed_game,
    record_completed_game_with_parity_check,
    get_or_create_db,
)
from app.db.parity_validator import (
    validate_game_parity,
    ParityValidationError,
    ParityDivergence,
    ParityMode,
    get_parity_mode,
    is_parity_validation_enabled,
)

__all__ = [
    "GameReplayDB",
    "GameWriter",
    "GameRecorder",
    "record_completed_game",
    "record_completed_game_with_parity_check",
    "get_or_create_db",
    # Parity validation
    "validate_game_parity",
    "ParityValidationError",
    "ParityDivergence",
    "ParityMode",
    "get_parity_mode",
    "is_parity_validation_enabled",
]
