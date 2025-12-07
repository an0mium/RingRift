"""Tests for swap rule (pie rule) AI evaluation and training diversity.

This test suite verifies that:
1. HeuristicAI can evaluate swap opportunities strategically
2. Training randomness creates diverse swap decisions
3. Swap evaluation works correctly across board types
"""

import os
import sys

import numpy as np
import pytest

# Ensure app package is importable when running tests directly from ai-service/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app.ai.heuristic_ai import HeuristicAI
from app.game_engine import GameEngine
from app.models import AIConfig, MoveType, Position, RingStack
from tests.rules.helpers import _make_base_game_state, _make_place_ring_move


def _make_swap_eligible_state(p1_ring_pos: Position):
    """Create a 2-player state where P2 can offer swap after P1's first move.

    Sets up:
    - rules_options with swapRuleEnabled=True
    - A single P1 stack at the given position
    - currentPlayer=2 (P2's turn to decide swap)
    - move_history with one P1 move
    """
    state = _make_base_game_state()
    state.rules_options = {"swapRuleEnabled": True}
    state.current_player = 2

    # Place P1's ring on the board
    key = p1_ring_pos.to_key()
    state.board.stacks[key] = RingStack(
        position=p1_ring_pos,
        rings=[1],
        stackHeight=1,
        capHeight=1,
        controllingPlayer=1,
    )
    state.players[0].rings_in_hand = 17  # P1 used one ring
    state.total_rings_in_play = 1

    # Record P1's move in history so swap gate passes
    p1_move = _make_place_ring_move(player=1, x=p1_ring_pos.x, y=p1_ring_pos.y)
    state.move_history = [p1_move]

    return state


class TestSwapEvaluation:
    """Test swap rule evaluation in HeuristicAI."""

    def test_deterministic_swap_evaluation(self):
        """Test that swap evaluation is deterministic by default."""
        config = AIConfig(
            ai_type="heuristic",
            difficulty=5,
            randomness=0.0,
            heuristic_profile_id="heuristic_v1_balanced",
        )

        ai = HeuristicAI(player_number=2, config=config)

        # Verify temperature is <= 0 (deterministic - noise only added when > 0)
        assert ai.WEIGHT_SWAP_EXPLORATION_TEMPERATURE <= 0

    def test_swap_strong_center_opening(self):
        """Test that AI swaps when P1 plays a strong center opening."""
        # P1 placed ring in center (3,3) - strong opening
        center_pos = Position(x=3, y=3)
        state = _make_swap_eligible_state(center_pos)

        # P2 should see swap_sides as an option
        p2_moves = GameEngine.get_valid_moves(state, 2)
        swap_moves = [m for m in p2_moves if m.type == MoveType.SWAP_SIDES]
        assert len(swap_moves) == 1, "P2 should see exactly one swap move"

        # Evaluate swap opportunity (should be positive for center opening)
        config = AIConfig(ai_type="heuristic", difficulty=5, randomness=0.0)
        ai = HeuristicAI(player_number=2, config=config)

        swap_value = ai.evaluate_swap_opening_bonus(state)
        assert swap_value > 0, "Center opening should have positive swap value"

    def test_swap_weak_corner_opening(self):
        """Test that AI evaluates corner opening lower than center."""
        # P1 placed ring in corner (0,0) - weaker opening
        corner_pos = Position(x=0, y=0)
        state = _make_swap_eligible_state(corner_pos)

        # P2 should still see swap_sides as an option
        p2_moves = GameEngine.get_valid_moves(state, 2)
        swap_moves = [m for m in p2_moves if m.type == MoveType.SWAP_SIDES]
        assert len(swap_moves) == 1, "P2 should see exactly one swap move"

        # Evaluate swap opportunity (should be lower for corner)
        config = AIConfig(ai_type="heuristic", difficulty=5, randomness=0.0)
        ai = HeuristicAI(player_number=2, config=config)

        swap_value = ai.evaluate_swap_opening_bonus(state)
        # Corner should have lower value than center threshold
        assert swap_value < 15.0, "Corner opening should have lower swap value"

    def test_center_vs_corner_swap_values(self):
        """Verify center opening has higher swap value than corner."""
        config = AIConfig(ai_type="heuristic", difficulty=5, randomness=0.0)
        ai = HeuristicAI(player_number=2, config=config)

        # Center opening
        center_state = _make_swap_eligible_state(Position(x=3, y=3))
        center_value = ai.evaluate_swap_opening_bonus(center_state)

        # Corner opening
        corner_state = _make_swap_eligible_state(Position(x=0, y=0))
        corner_value = ai.evaluate_swap_opening_bonus(corner_state)

        assert (
            center_value > corner_value
        ), f"Center ({center_value}) should be valued higher than corner ({corner_value})"

    def test_stochastic_swap_creates_diversity(self):
        """Test that training randomness creates diverse swap decisions."""
        # Create game with center opening
        center_pos = Position(x=3, y=3)
        state = _make_swap_eligible_state(center_pos)

        # Create AI with training randomness enabled
        config = AIConfig(ai_type="heuristic", difficulty=5, randomness=0.0)
        ai = HeuristicAI(player_number=2, config=config)

        # Override temperature for training
        ai.WEIGHT_SWAP_EXPLORATION_TEMPERATURE = 10.0

        # Evaluate swap multiple times and collect values with added noise
        swap_values = []
        base_value = ai.evaluate_swap_opening_bonus(state)

        for _ in range(100):
            # Simulate the evaluation with randomness as done in select_move
            value = base_value
            if ai.WEIGHT_SWAP_EXPLORATION_TEMPERATURE > 0:
                noise = np.random.normal(0, ai.WEIGHT_SWAP_EXPLORATION_TEMPERATURE)
                value += noise
            swap_values.append(value)

        # Verify diversity: values should have variance
        assert len(swap_values) == 100
        variance = np.var(swap_values)
        assert variance > 0, "Stochastic mode should create variance in swap values"

        # With temperature=10.0, we should see significant spread
        assert variance > 50, f"Expected significant variance, got {variance}"

    def test_swap_training_mode_flag(self):
        """Test that training mode can be controlled via weight."""
        config = AIConfig(ai_type="heuristic", difficulty=5, randomness=0.0)
        ai = HeuristicAI(player_number=2, config=config)

        # Default: deterministic (temperature <= 0)
        assert ai.WEIGHT_SWAP_EXPLORATION_TEMPERATURE <= 0

        # Can be overridden for training
        ai.WEIGHT_SWAP_EXPLORATION_TEMPERATURE = 0.15
        assert ai.WEIGHT_SWAP_EXPLORATION_TEMPERATURE == 0.15


class TestSwapMultiplayer:
    """Test that swap does NOT apply in multiplayer games."""

    def test_no_swap_in_3player(self):
        """Verify swap is not offered in 3-player games."""
        state = _make_base_game_state()

        # Add a third player
        from app.models import Player

        p3 = Player(
            id="p3",
            username="p3",
            type="human",
            playerNumber=3,
            isReady=True,
            timeRemaining=60,
            aiDifficulty=None,
            ringsInHand=12,
            eliminatedRings=0,
            territorySpaces=0,
        )
        state.players.append(p3)
        state.max_players = 3

        # Enable swap rule
        state.rules_options = {"swapRuleEnabled": True}

        # P1 makes a move
        p1_move = _make_place_ring_move(player=1, x=3, y=3)
        state.move_history = [p1_move]
        state.board.stacks["3,3"] = RingStack(
            position=Position(x=3, y=3),
            rings=[1],
            stackHeight=1,
            capHeight=1,
            controllingPlayer=1,
        )
        state.players[0].rings_in_hand = 11
        state.current_player = 2

        # P2 should NOT see swap (only offered in 2-player games)
        p2_moves = GameEngine.get_valid_moves(state, 2)
        swap_moves = [m for m in p2_moves if m.type == MoveType.SWAP_SIDES]
        assert len(swap_moves) == 0, "Swap should not be offered in multiplayer"
