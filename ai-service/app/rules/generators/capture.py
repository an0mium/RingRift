"""Capture move generator.

This module implements capture move enumeration, extracted from
GameEngine._get_capture_moves to establish SSoT.

Canonical Spec References:
- RR-CANON-R076: Interactive decision moves only
- RR-CANON-R095: Overtaking capture initiation
- RR-CANON-R096: Chain capture continuation

Architecture Note (2025-12):
    This generator wraps rules.capture_chain.enumerate_capture_moves_py,
    which mirrors TS CaptureAggregate.enumerateCaptureMoves. It handles
    both initial captures (from any controlled stack) and chain capture
    continuations (from the chain's current position).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.models import GameState, Move
from app.rules.capture_chain import enumerate_capture_moves_py
from app.rules.interfaces import Generator

if TYPE_CHECKING:
    pass


class CaptureGenerator(Generator):
    """Generator for capture moves.

    Enumerates legal overtaking capture moves for a player.

    Per RR-CANON-R076, this returns only interactive decision moves:
    - OVERTAKING_CAPTURE moves for initial captures from any controlled stack
    - CONTINUE_CAPTURE_SEGMENT moves during chain capture phase

    During CHAIN_CAPTURE phase (chain_capture_state set), enumerate only
    from the chain's current position. During MOVEMENT phase, enumerate
    from ALL controlled stacks.
    """

    def generate(
        self,
        state: GameState,
        player: int,
        *,
        limit: int | None = None,
    ) -> list[Move]:
        """Generate all legal capture moves for the player.

        Args:
            state: Current game state
            player: Player number to generate moves for
            limit: If provided, return at most this many moves (early-return optimization)

        Returns:
            List of legal Move objects for captures
        """
        moves: list[Move] = []
        move_number = len(state.move_history) + 1

        if state.chain_capture_state:
            # Chain capture in progress - enumerate only from the chain position
            return self._enumerate_chain_continuations(state, player, move_number)
        else:
            # Movement phase - enumerate from ALL player's stacks
            return self._enumerate_initial_captures(state, player, move_number, limit)

    def _enumerate_chain_continuations(
        self,
        state: GameState,
        player: int,
        move_number: int,
    ) -> list[Move]:
        """Enumerate chain capture continuation moves.

        Per RR-CANON-R096: During chain capture, only the attacker at the
        chain's current position may continue capturing. visited_positions
        tracks "from" positions (where attacker jumped FROM), not landing
        positions - enabling "bouncing" patterns.
        """
        attacker_pos = state.chain_capture_state.current_position
        return enumerate_capture_moves_py(
            state,
            player,
            attacker_pos,
            move_number=move_number,
            kind="continuation",
        )

    def _enumerate_initial_captures(
        self,
        state: GameState,
        player: int,
        move_number: int,
        limit: int | None,
    ) -> list[Move]:
        """Enumerate initial capture moves from all controlled stacks.

        Per RR-CANON-R095: During movement phase, captures may initiate
        from any stack controlled by the player. When must_move_from_stack_key
        is set (after place_ring), only captures from that stack are valid.
        """
        moves: list[Move] = []
        board = state.board
        must_move_key = state.must_move_from_stack_key

        for stack in board.stacks.values():
            if stack.controlling_player != player:
                continue
            if stack.stack_height <= 0:
                continue

            # Filter by must_move_from_stack_key if set
            if must_move_key is not None and stack.position.to_key() != must_move_key:
                continue

            stack_captures = enumerate_capture_moves_py(
                state,
                player,
                stack.position,
                move_number=move_number,
                kind="initial",
            )
            moves.extend(stack_captures)

            # Early return if limit reached
            if limit is not None and len(moves) >= limit:
                return moves[:limit]

        if limit is not None and len(moves) > limit:
            return moves[:limit]
        return moves

    def has_any_capture(self, state: GameState, player: int) -> bool:
        """Check if player has any legal capture move.

        Optimized early-return check - returns True immediately upon
        finding the first valid capture.
        """
        return len(self.generate(state, player, limit=1)) > 0
