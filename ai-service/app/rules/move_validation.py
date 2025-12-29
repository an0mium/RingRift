"""Move position validation utilities.

This module provides validation for move position fields based on move type.
Different move types have different position requirements:

- PLACE_RING: requires 'to' (target position)
- MOVE_STACK: requires 'from_pos' and 'to'
- OVERTAKING_CAPTURE: requires 'from_pos', 'to', and 'capture_target'

Move types like SWAP_SIDES, FORCED_ELIMINATION, SKIP_PLACEMENT, etc.
do not require position fields.

December 28, 2025: Created to prevent null position corruption in game databases.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..models import MoveType

if TYPE_CHECKING:
    from ..models import Move


class MovePositionError(ValueError):
    """Raised when a move is missing required position fields."""

    def __init__(self, message: str, move_type: str | None = None, missing_field: str | None = None):
        super().__init__(message)
        self.move_type = move_type
        self.missing_field = missing_field


# Position requirements by move type
# Maps MoveType to dict of required position fields
# True means the field is required, False means optional
MOVE_TYPE_POSITION_REQUIREMENTS: dict[MoveType, dict[str, bool]] = {
    # Ring placement - requires target position
    MoveType.PLACE_RING: {"to": True},

    # Movement - requires source and destination
    MoveType.MOVE_STACK: {"from_pos": True, "to": True},
    MoveType.MOVE_RING: {"from_pos": True, "to": True},  # Legacy alias
    MoveType.BUILD_STACK: {"from_pos": True, "to": True},  # Legacy alias

    # Capture moves - require positions for movement and capture target
    MoveType.OVERTAKING_CAPTURE: {"from_pos": True, "to": True, "capture_target": True},
    MoveType.CONTINUE_CAPTURE_SEGMENT: {"to": True},

    # The following move types do NOT require positions:
    # - SWAP_SIDES: Pure identity swap, no board change
    # - FORCED_ELIMINATION: System-initiated, no position choice
    # - SKIP_PLACEMENT: Voluntary skip
    # - SKIP_CAPTURE: Voluntary skip
    # - NO_PLACEMENT_ACTION: Forced no-op
    # - NO_MOVEMENT_ACTION: Forced no-op
    # - PROCESS_LINE: Line processing decision
    # - CHOOSE_LINE_OPTION: Line choice decision
    # - CHOOSE_LINE_REWARD: Legacy alias
    # - CHOOSE_TERRITORY_OPTION: Territory decision
    # - PROCESS_TERRITORY_REGION: Legacy alias
}


def validate_move_positions(move: Move) -> tuple[bool, str | None]:
    """Validate that a move has all required position fields for its type.

    Args:
        move: The Move object to validate.

    Returns:
        A tuple of (is_valid, error_message).
        If valid, returns (True, None).
        If invalid, returns (False, "description of missing field").

    Example:
        >>> from app.models import Move, MoveType, Position
        >>> # Valid move
        >>> good_move = Move(id='1', type=MoveType.PLACE_RING, player=1, to=Position(x=3, y=4))
        >>> validate_move_positions(good_move)
        (True, None)

        >>> # Invalid move - missing 'to' position
        >>> bad_move = Move(id='2', type=MoveType.PLACE_RING, player=1, to=None)
        >>> validate_move_positions(bad_move)
        (False, "place_ring requires 'to' position")
    """
    # Allow bypass via environment variable (for emergency rollback)
    if os.environ.get("RINGRIFT_SKIP_MOVE_VALIDATION", "").lower() in ("true", "1", "yes"):
        return True, None

    requirements = MOVE_TYPE_POSITION_REQUIREMENTS.get(move.type, {})

    if requirements.get("to") and move.to is None:
        return False, f"{move.type.value} requires 'to' position"

    if requirements.get("from_pos") and move.from_pos is None:
        return False, f"{move.type.value} requires 'from_pos' position"

    if requirements.get("capture_target") and move.capture_target is None:
        return False, f"{move.type.value} requires 'capture_target' position"

    return True, None


def validate_move_positions_strict(move: Move) -> None:
    """Validate move positions, raising MovePositionError if invalid.

    This is a convenience wrapper around validate_move_positions() that
    raises an exception instead of returning a tuple.

    Args:
        move: The Move object to validate.

    Raises:
        MovePositionError: If the move is missing required position fields.
    """
    valid, error = validate_move_positions(move)
    if not valid:
        # Determine which field is missing for the error context
        missing_field = None
        requirements = MOVE_TYPE_POSITION_REQUIREMENTS.get(move.type, {})
        if requirements.get("to") and move.to is None:
            missing_field = "to"
        elif requirements.get("from_pos") and move.from_pos is None:
            missing_field = "from_pos"
        elif requirements.get("capture_target") and move.capture_target is None:
            missing_field = "capture_target"

        raise MovePositionError(
            f"Cannot process move {move.id} (type={move.type.value}): {error}",
            move_type=move.type.value,
            missing_field=missing_field,
        )


def is_position_required(move_type: MoveType, field: str) -> bool:
    """Check if a position field is required for a given move type.

    Args:
        move_type: The type of move.
        field: The field name ('to', 'from_pos', or 'capture_target').

    Returns:
        True if the field is required for this move type, False otherwise.
    """
    requirements = MOVE_TYPE_POSITION_REQUIREMENTS.get(move_type, {})
    return requirements.get(field, False)


def get_required_position_fields(move_type: MoveType) -> set[str]:
    """Get the set of required position fields for a move type.

    Args:
        move_type: The type of move.

    Returns:
        Set of required field names (e.g., {'to', 'from_pos'}).
    """
    requirements = MOVE_TYPE_POSITION_REQUIREMENTS.get(move_type, {})
    return {field for field, required in requirements.items() if required}


# String-based validation for use in consolidation scripts
# (which work with raw JSON, not parsed Move objects)
POSITION_REQUIREMENTS_BY_STRING: dict[str, set[str]] = {
    "place_ring": {"to"},
    "move_stack": {"from_pos", "to"},
    "move_ring": {"from_pos", "to"},
    "build_stack": {"from_pos", "to"},
    "overtaking_capture": {"from_pos", "to", "capture_target"},
    "continue_capture_segment": {"to"},
}


def validate_move_json(move_type: str, move_data: dict) -> tuple[bool, str | None]:
    """Validate move positions from raw JSON data.

    This is useful for validation in consolidation scripts where we have
    raw move JSON rather than parsed Move objects.

    Args:
        move_type: The move type as a string (e.g., 'place_ring').
        move_data: The move data dict from JSON.

    Returns:
        A tuple of (is_valid, error_message).
    """
    required = POSITION_REQUIREMENTS_BY_STRING.get(move_type, set())

    if "to" in required and move_data.get("to") is None:
        return False, f"{move_type} requires 'to' position"

    # Note: JSON uses "from" as the key, not "from_pos"
    if "from_pos" in required and move_data.get("from") is None:
        return False, f"{move_type} requires 'from' position"

    if "capture_target" in required and move_data.get("captureTarget") is None:
        return False, f"{move_type} requires 'captureTarget' position"

    return True, None
