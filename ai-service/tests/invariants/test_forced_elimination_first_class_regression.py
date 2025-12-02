from __future__ import annotations

"""Regression tests for forced elimination as a first-class turn action.

These tests validate the fix for the bug where games terminated with
"no_legal_moves_for_current_player" when the player was in a forced
elimination state (has stacks but no placement/movement/capture).

Per RR-CANON-R072/R100/R205, forced elimination must be exposed as a
first-class action in get_valid_moves() so that:
1. Self-play and training loops can continue
2. AI players can choose which stack to eliminate from
3. Human players (via UI) can select their elimination target

Failure scenario captured in logs/selfplay/failures/:
- Game status: ACTIVE
- Current player has stacks on the board
- No placement, movement, or capture moves available
- Previously: get_valid_moves() returned [] → game terminated incorrectly
- After fix: get_valid_moves() returns FORCED_ELIMINATION moves
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

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


SELFPLAY_FAILURES_DIR = Path(__file__).parent.parent.parent / "logs" / "selfplay" / "failures"
RESULTS_FAILURES_DIR = Path(__file__).parent.parent.parent / "results" / "failures"


def load_failure_snapshot(path: Path) -> Optional[GameState]:
    """Load a failure snapshot and return the GameState."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return GameState.model_validate(payload["state"])


def describe_fe_state(state: GameState) -> dict:
    """Diagnostic info for a forced elimination state."""
    player = state.current_player
    player_stacks = [
        s for s in state.board.stacks.values()
        if s.controlling_player == player
    ]
    return {
        "game_status": state.game_status.value,
        "current_player": player,
        "current_phase": state.current_phase.value if state.current_phase else None,
        "player_stacks": len(player_stacks),
        "has_forced_elimination": ga.has_forced_elimination_action(state, player),
    }


class TestForcedEliminationFirstClass:
    """Test that forced elimination is exposed as first-class action."""

    def test_get_valid_moves_includes_forced_elimination_when_blocked(self):
        """When player has stacks but no other moves, FE moves are returned."""
        # Load the first failure snapshot if available
        snapshot_path = RESULTS_FAILURES_DIR / "failure_0_no_legal_moves_for_current_player.json"
        if not snapshot_path.exists():
            snapshot_path = SELFPLAY_FAILURES_DIR / "failure_0_no_legal_moves_for_current_player.json"

        if not snapshot_path.exists():
            pytest.skip("No failure snapshot available; run self-play soak first")

        state = load_failure_snapshot(snapshot_path)
        assert state is not None

        player = state.current_player

        # Verify preconditions: player has stacks but should be in FE state
        player_stacks = [
            s for s in state.board.stacks.values()
            if s.controlling_player == player
        ]
        if not player_stacks:
            pytest.skip(
                "Snapshot has no player stacks - not an FE scenario "
                "(may be a different failure type like ring_placement with no moves)"
            )

        # After the fix: get_valid_moves should return FE moves
        moves = GameEngine.get_valid_moves(state, player)
        assert len(moves) > 0, (
            f"get_valid_moves should return FE moves when player is blocked. "
            f"Diagnostics: {describe_fe_state(state)}"
        )

        # All returned moves should be FORCED_ELIMINATION
        fe_moves = [m for m in moves if m.type == MoveType.FORCED_ELIMINATION]
        assert len(fe_moves) == len(moves), (
            "When in FE state, all moves should be FORCED_ELIMINATION"
        )

        # FE moves should target player-controlled stacks
        for m in fe_moves:
            stack_key = f"{m.to.x},{m.to.y}"
            assert stack_key in state.board.stacks, f"FE move targets non-existent stack: {m.to}"
            stack = state.board.stacks[stack_key]
            assert stack.controlling_player == player, (
                f"FE move targets stack controlled by {stack.controlling_player}, not {player}"
            )

    def test_forced_elimination_moves_cover_all_player_stacks(self):
        """FE moves should be generated for all player-controlled stacks.

        This test uses the real failure snapshot to verify FE move coverage.
        """
        snapshot_path = RESULTS_FAILURES_DIR / "failure_0_no_legal_moves_for_current_player.json"
        if not snapshot_path.exists():
            snapshot_path = SELFPLAY_FAILURES_DIR / "failure_0_no_legal_moves_for_current_player.json"

        if not snapshot_path.exists():
            pytest.skip("No failure snapshot available")

        state = load_failure_snapshot(snapshot_path)
        assert state is not None

        player = state.current_player
        player_stacks = [
            s for s in state.board.stacks.values()
            if s.controlling_player == player
        ]

        if not player_stacks:
            pytest.skip("Snapshot has no player stacks")

        # Get moves - should include FE
        moves = GameEngine.get_valid_moves(state, player)
        fe_moves = [m for m in moves if m.type == MoveType.FORCED_ELIMINATION]

        if not fe_moves:
            pytest.skip("Snapshot player is not in FE state")

        # All FE moves should target player-controlled stacks
        for m in fe_moves:
            stack_key = f"{m.to.x},{m.to.y}"
            assert stack_key in state.board.stacks, f"FE move targets non-existent stack: {m.to}"
            stack = state.board.stacks[stack_key]
            assert stack.controlling_player == player, (
                f"FE move targets stack controlled by {stack.controlling_player}, not {player}"
            )

        # Number of FE moves should equal number of player stacks
        assert len(fe_moves) == len(player_stacks), (
            f"Expected {len(player_stacks)} FE moves for {len(player_stacks)} stacks, "
            f"got {len(fe_moves)}"
        )

    def test_applying_forced_elimination_progresses_game(self):
        """Applying an FE move should increase eliminated rings."""
        snapshot_path = RESULTS_FAILURES_DIR / "failure_0_no_legal_moves_for_current_player.json"
        if not snapshot_path.exists():
            snapshot_path = SELFPLAY_FAILURES_DIR / "failure_0_no_legal_moves_for_current_player.json"

        if not snapshot_path.exists():
            pytest.skip("No failure snapshot available")

        state = load_failure_snapshot(snapshot_path)
        assert state is not None

        player = state.current_player

        # Check if this is an FE scenario (player has stacks)
        player_stacks = [
            s for s in state.board.stacks.values()
            if s.controlling_player == player
        ]
        if not player_stacks:
            pytest.skip(
                "Snapshot has no player stacks - not an FE scenario "
                "(may be a different failure type)"
            )

        moves = GameEngine.get_valid_moves(state, player)

        if not moves:
            pytest.skip("No moves available in snapshot")

        fe_moves = [m for m in moves if m.type == MoveType.FORCED_ELIMINATION]
        if not fe_moves:
            pytest.skip("Snapshot is not in FE state")

        fe_move = fe_moves[0]

        before_eliminated = state.total_rings_eliminated

        # Apply the FE move
        new_state = GameEngine.apply_move(state, fe_move)

        after_eliminated = new_state.total_rings_eliminated

        assert after_eliminated > before_eliminated, (
            f"FE should increase eliminated rings. Before: {before_eliminated}, "
            f"After: {after_eliminated}"
        )

    def test_fe_satisfies_inv_active_has_moves(self):
        """After the fix, ACTIVE states with FE should not violate INV-ACTIVE-NO-MOVES.

        Note: This test only validates FE scenarios (where player has stacks but
        no other moves). Other failure types (e.g., ring_placement with exhausted
        rings) represent different bugs and are not validated here.
        """
        snapshot_path = RESULTS_FAILURES_DIR / "failure_0_no_legal_moves_for_current_player.json"
        if not snapshot_path.exists():
            snapshot_path = SELFPLAY_FAILURES_DIR / "failure_0_no_legal_moves_for_current_player.json"

        if not snapshot_path.exists():
            pytest.skip("No failure snapshot available")

        state = load_failure_snapshot(snapshot_path)
        assert state is not None

        assert state.game_status == GameStatus.ACTIVE

        player = state.current_player

        # This test focuses on FE scenarios only (player has stacks)
        player_stacks = [
            s for s in state.board.stacks.values()
            if s.controlling_player == player
        ]
        if not player_stacks:
            pytest.skip(
                "Snapshot has no player stacks - not an FE scenario. "
                "This may represent a different invariant violation."
            )

        # Check if this is specifically an FE scenario
        has_fe = ga.has_forced_elimination_action(state, player)
        if not has_fe:
            pytest.skip("Snapshot player is not in FE state")

        moves = GameEngine.get_valid_moves(state, player)

        # INV-ACTIVE-HAS-MOVES: ACTIVE state in FE scenario should have FE moves
        assert len(moves) > 0, (
            "INV-ACTIVE-HAS-MOVES violation in FE scenario: ACTIVE state has no legal moves. "
            f"Diagnostics: {describe_fe_state(state)}"
        )


class TestForcedEliminationMultipleSnapshots:
    """Test FE behavior across multiple failure snapshots."""

    @pytest.fixture
    def failure_snapshots(self) -> List[Path]:
        """Collect all available failure snapshots."""
        snapshots = []

        for failures_dir in [SELFPLAY_FAILURES_DIR, RESULTS_FAILURES_DIR]:
            if failures_dir.exists():
                for f in failures_dir.glob("failure_*_no_legal_moves_for_current_player.json"):
                    snapshots.append(f)

        return snapshots[:10]  # Limit to 10 for test speed

    def test_all_fe_snapshots_have_fe_moves_after_fix(self, failure_snapshots):
        """Failure snapshots in FE state should have FE moves available.

        Note: Some failure snapshots may be due to other phase issues (e.g.,
        RING_PLACEMENT with no rings in hand but movements available). This
        test only validates the FE-specific scenario.
        """
        if not failure_snapshots:
            pytest.skip("No failure snapshots available")

        fe_tested = 0
        for snapshot_path in failure_snapshots:
            state = load_failure_snapshot(snapshot_path)
            if state is None:
                continue

            if state.game_status != GameStatus.ACTIVE:
                continue

            player = state.current_player
            player_stacks = [
                s for s in state.board.stacks.values()
                if s.controlling_player == player
            ]

            # Only test FE scenario: player has stacks but no placements/movements
            if not player_stacks:
                continue

            # Check if this is a true FE scenario (no other moves available
            # except FE per the engine's internal check)
            has_fe = ga.has_forced_elimination_action(state, player)
            if not has_fe:
                # Not an FE scenario (player has placements or movements)
                continue

            fe_tested += 1
            moves = GameEngine.get_valid_moves(state, player)
            assert len(moves) > 0, (
                f"Snapshot {snapshot_path.name}: FE state but no moves. "
                f"Player {player} has {len(player_stacks)} stacks."
            )

        if fe_tested == 0:
            pytest.skip("No FE-state snapshots found in sample")


class TestForcedEliminationInvariant:
    """Test the formal FE invariant from RR-CANON-R072/R100/R205."""

    def test_invariant_has_stacks_implies_has_action(self):
        """If player has stacks, they must have at least one action available.

        This is the core invariant that the fix addresses:
        - has_stacks(P) → (has_placement ∨ has_movement ∨ has_capture ∨ has_fe)
        """
        from datetime import datetime

        # Create a blocked state where player has stacks but no regular moves
        board = BoardState(
            type=BoardType.SQUARE8,
            size=8,
            stacks={
                "3,3": RingStack(
                    position=Position(x=3, y=3),
                    controlling_player=1,
                    rings=[1],
                    stack_height=1,
                    cap_height=1,
                ),
            },
            # Surround with collapsed spaces to block movement
            collapsed_spaces={
                "2,2": 2, "2,3": 2, "2,4": 2,
                "3,2": 2, "3,4": 2,
                "4,2": 2, "4,3": 2, "4,4": 2,
            },
            markers={},
            eliminated_rings={"1": 0, "2": 0},
            formed_lines=[],
            territories={},
        )

        players = [
            Player(
                id="p1", username="P1", type="ai", player_number=1,
                is_ready=True, time_remaining=600000, ai_difficulty=5,
                rings_in_hand=0, eliminated_rings=0, territory_spaces=0,
            ),
            Player(
                id="p2", username="P2", type="ai", player_number=2,
                is_ready=True, time_remaining=600000, ai_difficulty=5,
                rings_in_hand=10, eliminated_rings=0, territory_spaces=0,
            ),
        ]

        time_control = TimeControl(initial_time=600000, increment=0, type="standard")
        now = datetime.now()

        state = GameState(
            id="invariant-test",
            board_type=BoardType.SQUARE8,
            rng_seed=42,
            board=board,
            players=players,
            current_phase=GamePhase.MOVEMENT,
            current_player=1,
            move_history=[],
            time_control=time_control,
            spectators=[],
            game_status=GameStatus.ACTIVE,
            winner=None,
            created_at=now,
            last_move_at=now,
            is_rated=False,
            max_players=2,
            total_rings_in_play=36,
            total_rings_eliminated=0,
            victory_threshold=19,
            territory_victory_threshold=33,
            chain_capture_state=None,
            must_move_from_stack_key=None,
            zobrist_hash=None,
            lps_round_index=0,
            lps_current_round_actor_mask={},
            lps_exclusive_player_for_completed_round=None,
        )

        # Player 1 has stacks
        p1_stacks = BoardManager.get_player_stacks(state.board, 1)
        assert len(p1_stacks) > 0

        # Therefore, player 1 must have at least one action
        moves = GameEngine.get_valid_moves(state, 1)

        # The invariant: has_stacks → has_action
        assert len(moves) > 0, (
            "INVARIANT VIOLATION: Player has stacks but no available actions. "
            "Either regular moves or FE moves must be available."
        )
