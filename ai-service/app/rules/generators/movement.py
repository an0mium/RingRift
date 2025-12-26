"""Movement move generator.

This module implements non-capture movement enumeration, extracted from
GameEngine._get_movement_moves to establish SSoT.

Canonical Spec References:
- RR-CANON-R076: Interactive decision moves only
- RR-CANON-R085: Stack movement rules
- RR-CANON-R091/R092: Marker landing rules

Architecture Note (2025-12):
    This generator uses BoardManager for direction and position validation,
    mirroring TS MovementAggregate.enumerateMovementMoves. It enumerates
    MOVE_STACK moves for non-capture movements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.board_manager import BoardManager
from app.models import GameState, Move, MoveType
from app.rules.interfaces import Generator

if TYPE_CHECKING:
    pass


class MovementGenerator(Generator):
    """Generator for non-capture movement moves.

    Enumerates legal MOVE_STACK moves for a player (non-capture movements).

    Per RR-CANON-R085: Stacks move in straight lines, minimum distance
    equals stack height. Cannot land on other stacks (that's capture).
    Can land on markers (own or opponent - marker removed, ring eliminated).
    """

    def generate(
        self,
        state: GameState,
        player: int,
        *,
        limit: int | None = None,
        ignore_must_move_key: bool = False,
    ) -> list[Move]:
        """Generate all legal movement moves for the player.

        Args:
            state: Current game state
            player: Player number to generate moves for
            limit: If provided, return at most this many moves (early-return)
            ignore_must_move_key: If True, ignore must_move_from_stack_key
                constraint (used for FE eligibility checks)

        Returns:
            List of legal Move objects for movements
        """
        moves: list[Move] = []
        board = state.board

        # Per-turn must-move constraint: when a ring has been placed this
        # turn, only the updated stack may move or capture.
        must_move_key = None if ignore_must_move_key else state.must_move_from_stack_key

        directions = BoardManager._get_all_directions(board.type)

        for stack in board.stacks.values():
            if stack.controlling_player != player:
                continue

            from_pos = stack.position

            # If must move from specific stack, skip others
            if must_move_key is not None and from_pos.to_key() != must_move_key:
                continue

            # Minimum movement distance equals stack height
            min_distance = max(1, stack.stack_height)

            for direction in directions:
                distance = min_distance
                while True:
                    to_pos = BoardManager._add_direction(from_pos, direction, distance)

                    # Check if position is valid on board
                    if not BoardManager.is_valid_position(to_pos, board.type, board.size):
                        break

                    # Cannot move to collapsed spaces
                    if BoardManager.is_collapsed_space(to_pos, board):
                        break

                    # Check path is clear (no blocking stacks)
                    if not self._is_path_clear(board, from_pos, to_pos):
                        break

                    to_key = to_pos.to_key()
                    dest_stack = board.stacks.get(to_key)

                    if dest_stack is None or dest_stack.stack_height == 0:
                        # Empty cell or marker-only - valid landing
                        # Per RR-CANON-R091/R092: Can land on any marker
                        moves.append(
                            Move(
                                id="simulated",
                                type=MoveType.MOVE_STACK,
                                player=player,
                                from_pos=from_pos,
                                to=to_pos,
                                timestamp=state.last_move_at,
                                thinkTime=0,
                                moveNumber=len(state.move_history) + 1,
                            )
                        )
                        if limit is not None and len(moves) >= limit:
                            return moves
                    else:
                        # Destination has stack - cannot land, cannot continue ray
                        break

                    distance += 1

        return moves

    def _is_path_clear(self, board, from_pos, to_pos) -> bool:
        """Check if path between positions is clear of blocking obstacles.

        Returns True if no stacks or collapsed spaces block the path.
        The destination cell itself is not considered blocking.
        """
        # Get direction and distance using x,y coordinates
        dx = to_pos.x - from_pos.x
        dy = to_pos.y - from_pos.y

        # Normalize direction
        steps = max(abs(dx), abs(dy))
        if steps <= 1:
            return True  # Adjacent cells, no path to check

        step_x = dx // steps if steps > 0 else 0
        step_y = dy // steps if steps > 0 else 0

        from app.models import Position

        # Check each intermediate cell
        for i in range(1, steps):
            check_pos = Position(x=from_pos.x + step_x * i, y=from_pos.y + step_y * i)
            check_key = check_pos.to_key()

            # Collapsed space blocks
            if check_key in board.collapsed_spaces:
                return False

            # Stack blocks
            stack = board.stacks.get(check_key)
            if stack and stack.stack_height > 0:
                return False

        return True

    def has_any_movement(self, state: GameState, player: int) -> bool:
        """Check if player has any legal movement move.

        Optimized early-return check.
        """
        return len(self.generate(state, player, limit=1)) > 0
