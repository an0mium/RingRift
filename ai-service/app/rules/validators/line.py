from typing import Set

from app.models import GameState, Move, GamePhase, MoveType, Position
from app.rules.interfaces import Validator
from app.rules.core import get_line_length_for_board


def _position_to_string(pos: Position) -> str:
    """Convert position to string key for comparison."""
    if pos.z is not None:
        return f"{pos.x},{pos.y},{pos.z}"
    return f"{pos.x},{pos.y}"


class LineValidator(Validator):
    def validate(self, state: GameState, move: Move) -> bool:
        # 1. Phase Check
        if state.current_phase != GamePhase.LINE_PROCESSING:
            return False

        # 2. Turn Check
        if move.player != state.current_player:
            return False

        # 3. Move Type Check
        if move.type not in (
            MoveType.PROCESS_LINE,
            MoveType.CHOOSE_LINE_REWARD,
            MoveType.CHOOSE_LINE_OPTION,
        ):
            return False

        # 4. Line Existence & Ownership
        # The move should reference a line that exists in
        # state.board.formed_lines and belongs to the player.
        # Note: The Move object for line processing usually contains the line
        # info or an index/ID. The current Python Move model has `formed_lines`
        # tuple. We need to verify that the line being processed is actually
        # present in the board state.

        # For PROCESS_LINE (canonical auto-processing or explicit choice
        # start), we expect the move to correspond to one of the lines in
        # board.formed_lines.

        # Since the Python Move model doesn't strictly link to a specific line
        # ID in the same way the TS one might (TS uses line index or object
        # equality), we'll check if the player has ANY lines to process.

        player_lines = [
            line for line in state.board.formed_lines
            if line.player == move.player
        ]

        if not player_lines:
            return False

        # 5. Option Validity for CHOOSE_LINE_REWARD / CHOOSE_LINE_OPTION
        # Mirrors TS LineValidator.validateChooseLineReward:
        # If collapsed_markers provided, validate count & consecutiveness
        if move.type in (  # noqa: E501
            MoveType.CHOOSE_LINE_REWARD, MoveType.CHOOSE_LINE_OPTION
        ):
            return self._validate_line_reward_choice(state, move, player_lines)

        return True

    def _validate_line_reward_choice(
        self,
        state: GameState,
        move: Move,
        player_lines: list
    ) -> bool:
        """
        Validate collapsed_markers for line reward choice.
        Mirrors TS validateChooseLineReward logic.

        Option 1 (collapse all): collapsed_markers is None OR equals full line
        Option 2 (minimum collapse): collapsed_markers == required_length,
                                     must be consecutive positions from line
        """
        required_length = get_line_length_for_board(state.board_type)

        # Find the matching line from move.formed_lines or state lines
        target_line = None
        if move.formed_lines and len(move.formed_lines) > 0:
            target_line = move.formed_lines[0]
        elif player_lines:
            # Fall back to first player line in state
            target_line = player_lines[0]

        if target_line is None:
            return False

        # If collapsed_markers is not provided, it's Option 1 (collapse all)
        # which is always valid for any line.
        if not move.collapsed_markers:
            return True

        collapsed = move.collapsed_markers

        # If collapsed_markers length == line length, it's Option 1 (all)
        # This is valid for any overlength line.
        if len(collapsed) == target_line.length:
            # Verify all positions are actually from the line
            line_pos_keys: Set[str] = set(
                _position_to_string(p) for p in target_line.positions
            )
            for pos in collapsed:
                if _position_to_string(pos) not in line_pos_keys:
                    return False
            return True

        # Otherwise it's Option 2 (minimum collapse)
        # Must have exactly required_length positions
        if len(collapsed) != required_length:
            return False

        # Cannot do minimum collapse on exact-length line
        if target_line.length == required_length:
            return False

        # Verify all collapsed positions are part of the line
        line_pos_keys = set(
            _position_to_string(p) for p in target_line.positions
        )
        for pos in collapsed:
            if _position_to_string(pos) not in line_pos_keys:
                return False

        # Verify collapsed positions are consecutive within the line
        # Map collapsed positions to their indices in line.positions
        indices = []
        for pos in collapsed:
            pos_key = _position_to_string(pos)
            for idx, line_pos in enumerate(target_line.positions):
                if _position_to_string(line_pos) == pos_key:
                    indices.append(idx)
                    break

        if len(indices) != len(collapsed):
            # Some positions weren't found in line
            return False

        indices.sort()
        for i in range(len(indices) - 1):
            if indices[i + 1] != indices[i] + 1:
                return False

        return True
