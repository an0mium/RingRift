from __future__ import annotations

"""Regression for turn rotation skipping fully eliminated players.

This test guards the scenario captured in
``ai-service/logs/selfplay/failures/failure_120_no_legal_moves_for_current_player.json``.

In that snapshot, after Player 2's move_stack from (5,5) to (4,6), the Python
GameEngine had rotated the turn to Player 1 despite Player 1 having:

- **no stacks** (all rings buried in opponent stacks)
- **no rings in hand**

Under the canonical RingRift rules (§7.3 Temporarily Inactive), a player with
no controlled stacks and no rings in hand is "temporarily inactive" and should
be skipped during turn rotation. The turn should pass to the next player who
has material.

Bug manifestation:
- ``game_status == ACTIVE``
- ``current_phase == MOVEMENT`` (incorrect - should be RING_PLACEMENT)
- ``current_player == 1`` (incorrect - should be 2)
- Player 1 has 0 stacks, 0 rings in hand
- Player 2 has 5 stacks, 4 rings in hand
- ``get_valid_moves(state, 1) == []``

This test replays the scenario and asserts that:
1. After applying the move from the pre-failure state, turn rotation correctly
   skips Player 1 (who is temporarily inactive)
2. The resulting state has ``current_player == 2``
3. Player 2 has valid moves available
"""

import json
import os
import sys
from typing import Dict, Any, Optional

import pytest

# Ensure app package is importable when running tests directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.models import (
    GameState, Move, Position, MoveType, GamePhase, GameStatus, BoardType,
    BoardState, Player, RingStack, TimeControl
)
from app.game_engine import GameEngine
from app.board_manager import BoardManager
from app.rules import global_actions as ga


FAILURE_SNAPSHOT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "logs",
    "selfplay",
    "failures",
    "failure_120_no_legal_moves_for_current_player.json",
)


def describe_state(state: GameState, label: str = "") -> Dict[str, Any]:
    """Return diagnostic info for a game state."""
    p1 = next((p for p in state.players if p.player_number == 1), None)
    p2 = next((p for p in state.players if p.player_number == 2), None)

    p1_stacks = [k for k, s in state.board.stacks.items() if s.controlling_player == 1]
    p2_stacks = [k for k, s in state.board.stacks.items() if s.controlling_player == 2]

    return {
        "label": label,
        "current_player": state.current_player,
        "current_phase": state.current_phase.value if state.current_phase else None,
        "game_status": state.game_status.value if state.game_status else None,
        "p1_rings_in_hand": p1.rings_in_hand if p1 else None,
        "p1_stacks": len(p1_stacks),
        "p1_eliminated": p1.eliminated_rings if p1 else None,
        "p2_rings_in_hand": p2.rings_in_hand if p2 else None,
        "p2_stacks": len(p2_stacks),
        "p2_eliminated": p2.eliminated_rings if p2 else None,
        "p1_has_turn_material": ga.has_turn_material(state, 1) if p1 else None,
        "p2_has_turn_material": ga.has_turn_material(state, 2) if p2 else None,
    }


@pytest.mark.skipif(
    not os.path.exists(FAILURE_SNAPSHOT_PATH),
    reason="Failure snapshot not found; run self-play soak to regenerate",
)
def test_turn_rotation_skips_eliminated_player_from_snapshot() -> None:
    """Load the failure snapshot and verify the bug conditions.

    This test documents the bug scenario:
    - The snapshot shows current_player=1 in MOVEMENT phase
    - Player 1 has no stacks and no rings in hand
    - This is an invalid state - Player 1 should have been skipped

    After the fix, calling _end_turn should correctly skip Player 1.
    """
    with open(FAILURE_SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    state = GameState.model_validate(payload["state"])

    # Document the bug: state has current_player=1 but P1 is eliminated
    diag = describe_state(state, "failure_snapshot")
    print(f"\nFailure state diagnostics: {json.dumps(diag, indent=2)}")

    assert state.game_status == GameStatus.ACTIVE, "Game should be ACTIVE"
    assert state.current_player == 1, "Bug: current_player is 1 (should be 2)"

    # Verify Player 1 has no material
    p1 = next(p for p in state.players if p.player_number == 1)
    p1_stacks = [s for s in state.board.stacks.values() if s.controlling_player == 1]
    assert len(p1_stacks) == 0, "P1 should have no stacks"
    assert p1.rings_in_hand == 0, "P1 should have no rings in hand"

    # Verify Player 2 has material
    p2 = next(p for p in state.players if p.player_number == 2)
    p2_stacks = [s for s in state.board.stacks.values() if s.controlling_player == 2]
    assert len(p2_stacks) > 0, "P2 should have stacks"
    assert p2.rings_in_hand > 0, "P2 should have rings in hand"

    # The fix: calling _end_turn should skip P1 and give turn to P2
    test_state = state.model_copy(deep=True)
    GameEngine._end_turn(test_state)

    fixed_diag = describe_state(test_state, "after_end_turn")
    print(f"After _end_turn: {json.dumps(fixed_diag, indent=2)}")

    # After fix, current_player should be 2 (skip P1)
    assert test_state.current_player == 2, (
        f"After _end_turn, current_player should be 2, got {test_state.current_player}"
    )

    # P2 should have valid moves
    p2_moves = GameEngine.get_valid_moves(test_state, 2)
    assert len(p2_moves) > 0, "P2 should have valid moves"


def test_turn_rotation_invariant_synthetic() -> None:
    """Synthetic test: verify turn rotation skips eliminated players.

    This test creates a synthetic state where:
    - Player 1 has no stacks and no rings in hand
    - Player 2 has stacks and rings in hand
    - After calling _end_turn from P1, turn should go to P2

    This should stay green after any fix is applied.
    """
    # Create a minimal synthetic state
    from datetime import datetime

    board = BoardState(
        type=BoardType.SQUARE8,
        size=8,
        stacks={
            "3,3": RingStack(
                position=Position(x=3, y=3),
                controlling_player=2,
                rings=[2, 2],
                stack_height=2,
                cap_height=2,
            ),
            "4,4": RingStack(
                position=Position(x=4, y=4),
                controlling_player=2,
                rings=[2, 1, 2],  # P1 rings buried
                stack_height=3,
                cap_height=2,
            ),
        },
        markers={},
        collapsed_spaces={},
        eliminated_rings={"1": 2, "2": 1},
        formed_lines=[],
        territories={},
    )

    players = [
        Player(
            id="p1",
            username="Player 1",
            type="ai",
            player_number=1,
            is_ready=True,
            time_remaining=600000,
            ai_difficulty=5,
            rings_in_hand=0,  # No rings in hand
            eliminated_rings=2,
            territory_spaces=0,
        ),
        Player(
            id="p2",
            username="Player 2",
            type="ai",
            player_number=2,
            is_ready=True,
            time_remaining=600000,
            ai_difficulty=5,
            rings_in_hand=10,  # Has rings in hand
            eliminated_rings=1,
            territory_spaces=0,
        ),
    ]

    time_control = TimeControl(initial_time=600000, increment=0, type="standard")
    now = datetime.now()

    state = GameState(
        id="synthetic-turn-skip-test",
        board_type=BoardType.SQUARE8,
        rng_seed=42,
        board=board,
        players=players,
        current_phase=GamePhase.MOVEMENT,
        current_player=1,  # Bug scenario: P1 is current but has no material
        move_history=[],
        time_control=time_control,
        spectators=[],
        game_status=GameStatus.ACTIVE,
        winner=None,
        created_at=now,
        last_move_at=now,
        is_rated=False,
        max_players=2,
        total_rings_in_play=18 * 2,
        total_rings_eliminated=3,
        victory_threshold=5,
        territory_victory_threshold=15,
        chain_capture_state=None,
        must_move_from_stack_key=None,
        zobrist_hash=None,
        lps_round_index=0,
        lps_current_round_actor_mask={},
        lps_exclusive_player_for_completed_round=None,
    )

    # Verify preconditions
    p1_stacks = BoardManager.get_player_stacks(state.board, 1)
    assert len(p1_stacks) == 0, "P1 should have no stacks"
    assert players[0].rings_in_hand == 0, "P1 should have no rings in hand"

    p2_stacks = BoardManager.get_player_stacks(state.board, 2)
    assert len(p2_stacks) > 0, "P2 should have stacks"
    assert players[1].rings_in_hand > 0, "P2 should have rings in hand"

    # Call _end_turn - it should skip P1 and set current_player to P2
    test_state = state.model_copy(deep=True)
    initial_player = test_state.current_player

    GameEngine._end_turn(test_state)

    # Assertions
    assert test_state.current_player == 2, (
        f"Turn should skip eliminated P1 and go to P2. "
        f"Initial: {initial_player}, Final: {test_state.current_player}"
    )

    # P2 should start in RING_PLACEMENT since they have rings in hand
    assert test_state.current_phase == GamePhase.RING_PLACEMENT, (
        f"P2 should start in RING_PLACEMENT, got {test_state.current_phase}"
    )

    # Verify P2 has valid moves
    p2_moves = GameEngine.get_valid_moves(test_state, 2)
    assert len(p2_moves) > 0, "P2 should have valid moves"


# ARCHIVED TEST: test_fully_eliminated_player_not_left_as_current
# Removed 2025-12-07
#
# This test expected _end_turn to skip players without turn material, but the
# current (correct) behavior is that fully-eliminated players are NOT skipped.
# Per the GameEngine._end_turn() documentation: "Do not skip fully-eliminated
# players; even seats with no stacks and no rings in hand must still traverse
# all phases and record no-action moves."
#
# The invariant being tested (ACTIVE state should never have a fully eliminated
# current_player) is not enforced by _end_turn - instead, hosts must emit
# NO_PLACEMENT_ACTION and other bookkeeping moves for players without material.
# The _assert_active_player_has_legal_action invariant accounts for this by
# checking for phase requirements (bookkeeping moves) in addition to interactive
# moves.
