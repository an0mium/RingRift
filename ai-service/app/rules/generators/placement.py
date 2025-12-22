"""Ring placement move generator.

This module implements ring placement enumeration, extracted from
GameEngine._get_ring_placement_moves to establish SSoT.

Canonical Spec References:
- RR-CANON-R076: Interactive decision moves only
- RR-CANON-R050: Ring placement rules
- RR-CANON-R055: Multi-ring placement (1-3 rings)
- RR-CANON-R060: No dead placement rule

Architecture Note (2025-12):
    This generator mirrors TS RuleEngine.getValidRingPlacements.
    It respects per-player ring caps, multi-ring placement rules,
    and enforces no-dead-placement by simulating post-placement
    movement/capture availability.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from app.board_manager import BoardManager
from app.models import GameState, Move, MoveType, Position, RingStack
from app.rules.core import count_rings_in_play_for_player
from app.rules.interfaces import Generator

if TYPE_CHECKING:
    from app.models import BoardState


class PlacementGenerator(Generator):
    """Generator for ring placement moves.

    Enumerates legal PLACE_RING moves for a player.

    Per RR-CANON-R050/R055: Respects per-player ring caps, allows
    multi-ring placement (1-3) on empty spaces, single ring on stacks.
    Per RR-CANON-R060: Enforces no-dead-placement (placed stack must
    have valid movement or capture after placement).
    """

    def generate(
        self,
        state: GameState,
        player: int,
        *,
        limit: int | None = None,
    ) -> list[Move]:
        """Generate all legal placement moves for the player.

        Args:
            state: Current game state
            player: Player number to generate moves for
            limit: If provided, return at most this many moves

        Returns:
            List of legal Move objects for placements
        """
        moves: list[Move] = []
        board = state.board

        # Check if player has rings in hand
        player_obj = next(
            (p for p in state.players if p.player_number == player),
            None,
        )
        if not player_obj or player_obj.rings_in_hand <= 0:
            return []

        rings_in_hand = player_obj.rings_in_hand

        # Per-player ring cap
        per_player_cap = self._estimate_rings_per_player(state)
        total_in_play = count_rings_in_play_for_player(state, player)
        rings_on_board = max(0, total_in_play - rings_in_hand)

        remaining_by_cap = per_player_cap - rings_on_board
        max_available = min(remaining_by_cap, rings_in_hand)
        if max_available <= 0:
            return []

        # Get all valid positions
        all_positions = self._generate_all_positions(board.type, board.size)

        for pos in all_positions:
            pos_key = pos.to_key()

            # Cannot place on collapsed spaces
            if pos_key in board.collapsed_spaces:
                continue

            # Cannot place on markers
            if pos_key in board.markers:
                continue

            existing_stack = board.stacks.get(pos_key)
            is_occupied = bool(existing_stack and existing_stack.stack_height > 0)

            if is_occupied:
                # On existing stacks, only single ring placement
                max_per_placement = 1
            else:
                # On empty spaces, allow multi-ring (up to 3)
                max_per_placement = min(3, max_available)

            if max_per_placement <= 0:
                continue

            # Optimization: Check highest count first - if valid, lower counts also valid
            valid_from_count = None
            for placement_count in range(max_per_placement, 0, -1):
                if placement_count > max_available:
                    continue

                hyp_board = self._create_hypothetical_board(
                    board, pos, player, placement_count
                )

                if self._has_movement_or_capture_after_placement(
                    state, player, pos, hyp_board
                ):
                    valid_from_count = placement_count
                    break

            # Add moves for all valid placement counts
            if valid_from_count is not None:
                for count in range(1, valid_from_count + 1):
                    if count > max_available:
                        break
                    moves.append(
                        Move(
                            id="simulated",
                            type=MoveType.PLACE_RING,
                            player=player,
                            to=pos,
                            timestamp=state.last_move_at,
                            thinkTime=0,
                            moveNumber=len(state.move_history) + 1,
                            placement_count=count,
                        )
                    )
                    if limit is not None and len(moves) >= limit:
                        return moves

        return moves

    def _estimate_rings_per_player(self, state: GameState) -> int:
        """Estimate per-player ring cap based on board config.

        Mirrors TS BOARD_CONFIGS[boardType].ringsPerPlayer.
        """
        board = state.board
        num_players = len(state.players)

        # Standard caps by board type
        caps = {
            "square8": 15,
            "square19": 50,
            "hexagonal": 25,
            "hex8": 15,
        }

        board_type_str = board.type.value if hasattr(board.type, "value") else str(board.type)
        base_cap = caps.get(board_type_str.lower(), 25)

        # Adjust for player count if needed
        if num_players > 2:
            return max(10, base_cap // num_players * 2)
        return base_cap

    def _generate_all_positions(self, board_type, board_size: int) -> list[Position]:
        """Generate all valid positions for the board type."""
        positions = []
        board_type_str = board_type.value if hasattr(board_type, "value") else str(board_type)

        if "hex" in board_type_str.lower():
            # Hexagonal grid
            for q in range(-board_size, board_size + 1):
                for r in range(-board_size, board_size + 1):
                    if abs(q + r) <= board_size:
                        positions.append(Position(q=q, r=r))
        else:
            # Square grid
            for q in range(board_size):
                for r in range(board_size):
                    positions.append(Position(q=q, r=r))

        return positions

    def _create_hypothetical_board(
        self,
        board: "BoardState",
        pos: Position,
        player: int,
        placement_count: int,
    ) -> "BoardState":
        """Create hypothetical board state after placement.

        Used for no-dead-placement validation.
        """
        hyp_board = copy.deepcopy(board)
        pos_key = pos.to_key()

        existing = hyp_board.stacks.get(pos_key)
        if existing and existing.stack_height > 0:
            # Add to existing stack
            new_height = existing.stack_height + placement_count
            new_rings = list(existing.rings) + [player] * placement_count
            hyp_board.stacks[pos_key] = RingStack(
                position=pos,
                rings=tuple(new_rings),
                stack_height=new_height,
                cap_height=new_height,
                controlling_player=player,
            )
        else:
            # New stack
            hyp_board.stacks[pos_key] = RingStack(
                position=pos,
                rings=tuple([player] * placement_count),
                stack_height=placement_count,
                cap_height=placement_count,
                controlling_player=player,
            )

        return hyp_board

    def _has_movement_or_capture_after_placement(
        self,
        state: GameState,
        player: int,
        placed_pos: Position,
        hyp_board: "BoardState",
    ) -> bool:
        """Check if the placed stack can move or capture.

        Enforces RR-CANON-R060: No dead placements.
        """
        # Create hypothetical state
        hyp_state = copy.copy(state)
        hyp_state.board = hyp_board
        hyp_state.must_move_from_stack_key = placed_pos.to_key()

        # Check for movement
        directions = BoardManager._get_all_directions(hyp_board.type)
        stack = hyp_board.stacks.get(placed_pos.to_key())
        if not stack:
            return False

        min_distance = max(1, stack.stack_height)

        # Quick movement check
        for direction in directions:
            to_pos = BoardManager._add_direction(placed_pos, direction, min_distance)
            if not BoardManager.is_valid_position(to_pos, hyp_board.type, hyp_board.size):
                continue
            if BoardManager.is_collapsed_space(to_pos, hyp_board):
                continue

            to_key = to_pos.to_key()
            dest = hyp_board.stacks.get(to_key)
            if dest is None or dest.stack_height == 0:
                return True  # Can move to empty space

            # Check capture potential (opponent stack)
            if dest.controlling_player != player and dest.stack_height < stack.stack_height:
                return True  # Can capture

        return False

    def has_any_placement(self, state: GameState, player: int) -> bool:
        """Check if player has any legal placement.

        Optimized early-return check.
        """
        return len(self.generate(state, player, limit=1)) > 0
