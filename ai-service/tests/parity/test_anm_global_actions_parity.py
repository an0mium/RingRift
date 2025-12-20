from __future__ import annotations

"""
ANM global-actions parity tests (Python <-> TS).

These tests mirror the TS ANM fixtures in
tests/fixtures/anmFixtures.ts and the TS suites:

- MovementNoAction.anmParity.test.ts
- LineAndTerritoryNoAction.anmParity.test.ts
- VictoryAnmChains.scenarios.test.ts

The goal is to ensure that Python's global_legal_actions_summary and
is_anm_state agree with the TS behaviour for the core ANM scenarios:

- ANM-SCEN-01 – Movement, no moves but forced elimination available.
- ANM-SCEN-02 – Movement, placements-only global actions.
- ANM-SCEN-03 – Movement, current player fully eliminated.
- ANM-SCEN-04 – Territory processing, no remaining decisions.
- ANM-SCEN-05 – Line processing, no remaining decisions.
- ANM-SCEN-06 – Global stalemate on a bare board.
- mustMoveFromStackKey-constrained movement surface.
"""

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.game_engine import GameEngine
from app.models import (
    GamePhase,
    GameState,
    GameStatus,
    Position,
    RingStack,
)
from app.rules import global_actions as ga
from tests.rules.helpers import _make_base_game_state

# ---------------------------------------------------------------------------
# Scenario builders (Python analogues of tests/fixtures/anmFixtures.ts)
# ---------------------------------------------------------------------------


def make_anm_scen01_movement_no_moves_but_fe_available() -> GameState:
    """ANM-SCEN-01 – movement phase, no moves but forced elimination exists.

    Shape:

    - game_status == ACTIVE
    - current_phase == MOVEMENT
    - current_player controls exactly one stack
    - Every other space is collapsed territory
      - No legal movement or capture from the stack
      - No legal placements (rings_in_hand == 0)
    - Forced-elimination preconditions hold.
    """
    state = _make_base_game_state()
    state.game_status = GameStatus.ACTIVE
    state.current_phase = GamePhase.MOVEMENT
    state.current_player = 1

    board = state.board
    board.stacks.clear()
    board.markers.clear()
    board.collapsed_spaces.clear()

    origin = Position(x=3, y=3)
    origin_key = origin.to_key()

    board.stacks[origin_key] = RingStack(
        position=origin,
        rings=[1],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=1,
    )

    size = board.size
    for x in range(size):
        for y in range(size):
            pos = Position(x=x, y=y)
            key = pos.to_key()
            if key == origin_key:
                continue
            # Treat as opponent territory so movement/capture is impossible.
            board.collapsed_spaces[key] = 2

    # No rings in hand for any player ⇒ no global placements.
    for player in state.players:
        player.rings_in_hand = 0

    GameEngine.clear_cache()
    return state


def make_anm_scen02_movement_placements_only() -> GameState:
    """ANM-SCEN-02 – movement phase, placements-only global actions.

    Shape:

    - game_status == ACTIVE
    - current_phase == MOVEMENT
    - No stacks on the board for the current player
    - rings_in_hand > 0 so global placements exist
    - get_valid_moves (movement/capture/recovery) returns []
    - No forced elimination (no stacks controlled by the player)
    """
    state = _make_base_game_state()
    state.game_status = GameStatus.ACTIVE
    state.current_phase = GamePhase.MOVEMENT
    state.current_player = 1

    board = state.board
    board.stacks.clear()
    board.markers.clear()
    board.collapsed_spaces.clear()

    # No stacks anywhere; placements are the only global actions.
    players = state.players
    if len(players) >= 2:
        players[0].rings_in_hand = 3
        players[1].rings_in_hand = 3
    else:
        players[0].rings_in_hand = 3

    GameEngine.clear_cache()
    return state


def make_anm_scen03_movement_current_player_fully_eliminated() -> GameState:
    """ANM-SCEN-03 – movement phase with fully eliminated current player.

    Shape:

    - game_status == ACTIVE
    - current_phase == MOVEMENT
    - current_player P has:
      - no stacks (no stack.controlling_player == P)
      - rings_in_hand[P] == 0
    - Another player Q still has stacks and/or rings in hand.
    """
    state = _make_base_game_state()
    state.game_status = GameStatus.ACTIVE
    state.current_phase = GamePhase.MOVEMENT
    state.current_player = 1

    board = state.board
    board.stacks.clear()
    board.markers.clear()
    board.collapsed_spaces.clear()

    # Give player 2 a single stack so someone still has material.
    pos = Position(x=4, y=4)
    board.stacks[pos.to_key()] = RingStack(
        position=pos,
        rings=[2],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=2,
    )

    for player in state.players:
        if player.player_number == 1:
            player.rings_in_hand = 0
        else:
            player.rings_in_hand = 2

    GameEngine.clear_cache()
    return state


def make_anm_scen04_territory_no_remaining_decisions() -> GameState:
    """ANM-SCEN-04 – territory_processing with no remaining decisions.

    Shape:

    - game_status == ACTIVE
    - current_phase == TERRITORY_PROCESSING
    - No disconnected regions for current_player
    - No pending territory eliminations
    - Player still has legal placements available
    """
    state = _make_base_game_state()
    state.game_status = GameStatus.ACTIVE
    state.current_phase = GamePhase.TERRITORY_PROCESSING
    state.current_player = 1

    board = state.board
    board.stacks.clear()
    board.markers.clear()
    board.collapsed_spaces.clear()

    # One simple stack for player 1; no curated territory geometry.
    origin = Position(x=0, y=0)
    board.stacks[origin.to_key()] = RingStack(
        position=origin,
        rings=[1],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=1,
    )

    for player in state.players:
        player.rings_in_hand = 2

    GameEngine.clear_cache()
    return state


def make_anm_scen05_line_processing_no_remaining_decisions() -> GameState:
    """ANM-SCEN-05 – line_processing with no remaining decisions.

    Shape:

    - game_status == ACTIVE
    - current_phase == LINE_PROCESSING
    - No formed lines for current_player
    - Player still has legal placements available
    """
    state = _make_base_game_state()
    state.game_status = GameStatus.ACTIVE
    state.current_phase = GamePhase.LINE_PROCESSING
    state.current_player = 1

    board = state.board
    board.stacks.clear()
    board.markers.clear()
    board.collapsed_spaces.clear()
    board.formed_lines = []

    players = state.players
    if len(players) >= 2:
        players[0].rings_in_hand = 2
        players[1].rings_in_hand = 2
    else:
        players[0].rings_in_hand = 2

    GameEngine.clear_cache()
    return state


def make_anm_scen06_global_stalemate_bare_board() -> GameState:
    """ANM-SCEN-06 – global stalemate on a bare board.

    Shape:

    - game_status == ACTIVE (pre-termination snapshot)
    - No stacks on the board
    - All board spaces are collapsed
    - Both players still have rings in hand, but:
      - has_global_placement_action == False for every player
      - has_forced_elimination_action == False for every player
    """
    state = _make_base_game_state()
    state.game_status = GameStatus.ACTIVE

    board = state.board
    board.stacks.clear()
    board.markers.clear()
    board.collapsed_spaces.clear()

    size = board.size
    for x in range(size):
        for y in range(size):
            pos = Position(x=x, y=y)
            board.collapsed_spaces[pos.to_key()] = 1

    players = state.players
    if len(players) >= 2:
        players[0].rings_in_hand = 3
        players[1].rings_in_hand = 1
    else:
        players[0].rings_in_hand = 3

    # Initialise eliminated_rings map for parity with TS stalemate ladder.
    board.eliminated_rings.clear()
    for player in players[:2]:
        board.eliminated_rings[str(player.player_number)] = 0

    # Use a non-participating currentPlayer so that is_anm_state(state) is
    # false (has_turn_material(currentPlayer) == false) while stalemate
    # resolution still evaluates over players[1..N] in victory logic.
    max_seat = max(p.player_number for p in players)
    state.current_player = max_seat + 1
    state.current_phase = GamePhase.RING_PLACEMENT

    GameEngine.clear_cache()
    return state


def make_movement_must_move_from_stack_key_constrained_state() -> GameState:
    """Helper for mustMoveFromStackKey-constrained movement surface.

    Shape:

    - Two stacks for the current player:
      - A "stuck" stack at (3,3) surrounded by collapsed spaces.
      - A "free" stack at (6,6) with open space around it.
    - must_move_from_stack_key is set to the stuck stack's key.

    Expectations:

    - With must_move_from_stack_key set:
        has_phase_local_interactive_move == False (stuck stack has no moves).
    - With the constraint removed:
        has_phase_local_interactive_move == True (free stack can move).
    - In both cases placements remain available, so is_anm_state(state) is
      False.
    """
    state = _make_base_game_state()
    state.game_status = GameStatus.ACTIVE
    state.current_phase = GamePhase.MOVEMENT
    state.current_player = 1

    board = state.board
    board.stacks.clear()
    board.markers.clear()
    board.collapsed_spaces.clear()

    stuck = Position(x=3, y=3)
    free = Position(x=6, y=6)

    board.stacks[stuck.to_key()] = RingStack(
        position=stuck,
        rings=[1],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=1,
    )
    board.stacks[free.to_key()] = RingStack(
        position=free,
        rings=[1],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=1,
    )

    size = board.size
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            x = stuck.x + dx
            y = stuck.y + dy
            if x < 0 or x >= size or y < 0 or y >= size:
                continue
            pos = Position(x=x, y=y)
            key = pos.to_key()
            if key in board.stacks:
                continue
            board.collapsed_spaces[key] = 1

    players = state.players
    if len(players) >= 2:
        players[0].rings_in_hand = 2
        players[1].rings_in_hand = 2
    else:
        players[0].rings_in_hand = 2

    state.must_move_from_stack_key = stuck.to_key()

    GameEngine.clear_cache()
    return state


# ---------------------------------------------------------------------------
# Scenario tests
# ---------------------------------------------------------------------------


def test_anm_scen01_movement_no_moves_but_fe_available() -> None:
    state = make_anm_scen01_movement_no_moves_but_fe_available()
    player = state.current_player

    moves = GameEngine.get_valid_moves(state, player)
    assert moves == []

    summary = ga.global_legal_actions_summary(state, player)
    is_anm = ga.is_anm_state(state)

    assert summary.has_turn_material is True
    assert summary.has_global_placement_action is False
    assert summary.has_phase_local_interactive_move is False
    assert summary.has_forced_elimination_action is True
    assert is_anm is False


def test_anm_scen02_movement_placements_only_global_actions() -> None:
    state = make_anm_scen02_movement_placements_only()
    player = state.current_player

    moves = GameEngine.get_valid_moves(state, player)
    assert moves == []

    summary = ga.global_legal_actions_summary(state, player)
    is_anm = ga.is_anm_state(state)

    assert summary.has_turn_material is True
    assert summary.has_global_placement_action is True
    assert summary.has_phase_local_interactive_move is False
    assert summary.has_forced_elimination_action is False
    assert is_anm is False


def test_anm_scen03_current_player_fully_eliminated_not_anm() -> None:
    state = make_anm_scen03_movement_current_player_fully_eliminated()
    player = state.current_player

    summary = ga.global_legal_actions_summary(state, player)
    is_anm = ga.is_anm_state(state)

    assert summary.has_turn_material is False
    assert summary.has_global_placement_action is False
    assert summary.has_phase_local_interactive_move is False
    assert summary.has_forced_elimination_action is False
    assert is_anm is False


def test_anm_scen04_territory_no_remaining_decisions_not_anm() -> None:
    state = make_anm_scen04_territory_no_remaining_decisions()
    player = state.current_player

    assert state.current_phase == GamePhase.TERRITORY_PROCESSING
    assert state.game_status == GameStatus.ACTIVE

    moves = GameEngine.get_valid_moves(state, player)
    assert moves == []

    summary = ga.global_legal_actions_summary(state, player)
    is_anm = ga.is_anm_state(state)

    assert summary.has_turn_material is True
    assert summary.has_global_placement_action is True
    assert summary.has_phase_local_interactive_move is False
    assert summary.has_forced_elimination_action is False
    assert is_anm is False


def test_anm_scen05_line_processing_no_remaining_decisions_not_anm() -> None:
    state = make_anm_scen05_line_processing_no_remaining_decisions()
    player = state.current_player

    assert state.current_phase == GamePhase.LINE_PROCESSING
    assert state.game_status == GameStatus.ACTIVE

    moves = GameEngine.get_valid_moves(state, player)
    assert moves == []

    summary = ga.global_legal_actions_summary(state, player)
    is_anm = ga.is_anm_state(state)

    assert summary.has_turn_material is True
    assert summary.has_global_placement_action is True
    assert summary.has_phase_local_interactive_move is False
    assert summary.has_forced_elimination_action is False
    assert is_anm is False


def test_anm_scen06_global_stalemate_resolved_by_victory_not_anm() -> None:
    state = make_anm_scen06_global_stalemate_bare_board()

    board = state.board
    assert not board.stacks
    assert len(board.collapsed_spaces) == board.size * board.size
    assert state.game_status == GameStatus.ACTIVE

    # For every player with rings in hand, the global action surface is empty
    # except for has_turn_material.
    for player in state.players:
        summary = ga.global_legal_actions_summary(state, player.player_number)
        assert summary.has_turn_material is True
        assert summary.has_global_placement_action is False
        assert summary.has_phase_local_interactive_move is False
        assert summary.has_forced_elimination_action is False

    # The synthetic ACTIVE snapshot must not be considered ANM; victory logic
    # resolves the stalemate instead.
    assert ga.is_anm_state(state) is False

    # Apply Python victory logic and ensure the game terminates via the
    # stalemate ladder with a winner selected.
    GameEngine._check_victory(state)  # type: ignore[attr-defined]

    assert state.game_status != GameStatus.ACTIVE
    assert state.current_phase == GamePhase.GAME_OVER
    assert state.winner in {p.player_number for p in state.players}
    # With the chosen ring-in-hand counts, player 1 wins on effective
    # elimination score once hands are treated as eliminated.
    if len(state.players) >= 2:
        assert state.winner == state.players[0].player_number


def test_must_move_from_stack_key_constrained_surface_not_anm() -> None:
    constrained = make_movement_must_move_from_stack_key_constrained_state()
    player = constrained.current_player

    constrained_summary = ga.global_legal_actions_summary(
        constrained,
        player,
    )

    assert constrained_summary.has_turn_material is True
    assert constrained_summary.has_global_placement_action is True
    # Movement/capture surface is empty when constrained to the stuck stack.
    assert constrained_summary.has_phase_local_interactive_move is False
    assert constrained_summary.has_forced_elimination_action is False
    assert ga.is_anm_state(constrained) is False

    # Remove must_move_from_stack_key while keeping the same board so that the
    # free stack can move; this simulates the unconstrained global
    # reachability check.
    unconstrained = constrained.model_copy(deep=True)
    unconstrained.must_move_from_stack_key = None
    GameEngine.clear_cache()

    unconstrained_summary = ga.global_legal_actions_summary(
        unconstrained,
        player,
    )

    assert unconstrained_summary.has_turn_material is True
    assert unconstrained_summary.has_global_placement_action is True
    assert unconstrained_summary.has_phase_local_interactive_move is True
    assert unconstrained_summary.has_forced_elimination_action is False
    assert ga.is_anm_state(unconstrained) is False


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
