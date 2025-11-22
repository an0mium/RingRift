import unittest
from unittest.mock import MagicMock
from datetime import datetime
import torch
import random
import sys
import os

# Ensure app package is importable
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from app.training.train import seed_all  # noqa: E402
from app.ai.neural_net import RingRiftCNN  # noqa: E402
from app.ai.zobrist import ZobristHash  # noqa: E402
from app.ai.random_ai import RandomAI  # noqa: E402
from app.models import (  # noqa: E402
    AIConfig,
    BoardState,
    BoardType,
    GamePhase,
    GameState,
    GameStatus,
    TimeControl,
)


class TestDeterminism(unittest.TestCase):
    def _make_minimal_game_state(self) -> GameState:
        """Construct a minimal but valid GameState for RNG tests."""
        now = datetime.now()
        return GameState(
            id="det-test-game",
            boardType=BoardType.SQUARE8,
            board=BoardState(
                type=BoardType.SQUARE8,
                size=8,
                stacks={},
                markers={},
                collapsedSpaces={},
                eliminatedRings={},
            ),
            players=[],
            currentPhase=GamePhase.MOVEMENT,
            currentPlayer=1,
            moveHistory=[],
            timeControl=TimeControl(
                initialTime=600,
                increment=0,
                type="blitz",
            ),
            gameStatus=GameStatus.ACTIVE,
            createdAt=now,
            lastMoveAt=now,
            isRated=False,
            maxPlayers=2,
            totalRingsInPlay=36,
            totalRingsEliminated=0,
            victoryThreshold=19,
            territoryVictoryThreshold=33,
            chainCaptureState=None,
            mustMoveFromStackKey=None,
            zobristHash=None,
        )

    def test_seed_determinism(self):
        """
        Verify that setting the seed produces identical results for the CNN.
        """
        seed = 42

        # Run 1
        seed_all(seed)
        model1 = RingRiftCNN(
            board_size=8,
            in_channels=10,
            global_features=10,
        )
        input1 = torch.randn(4, 40, 8, 8)
        globals1 = torch.randn(4, 10)
        val1, pol1 = model1(input1, globals1)

        # Run 2
        seed_all(seed)
        model2 = RingRiftCNN(
            board_size=8,
            in_channels=10,
            global_features=10,
        )
        input2 = torch.randn(4, 40, 8, 8)
        globals2 = torch.randn(4, 10)
        val2, pol2 = model2(input2, globals2)

        # Check equality
        self.assertTrue(torch.allclose(val1, val2))
        self.assertTrue(torch.allclose(pol1, pol2))

        # Check that inputs were generated identically
        self.assertTrue(torch.allclose(input1, input2))
        self.assertTrue(torch.allclose(globals1, globals2))

        # Check weights
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_zobrist_does_not_mutate_global_rng(self):
        """ZobristHash must not touch the module-level random RNG state."""
        random.seed(12345)
        state_before = random.getstate()
        ZobristHash()
        state_after = random.getstate()
        self.assertEqual(state_before, state_after)

        game_state = self._make_minimal_game_state()
        random.seed(12345)
        state_before = random.getstate()
        ZobristHash().compute_initial_hash(game_state)
        state_after = random.getstate()
        self.assertEqual(state_before, state_after)

    def test_random_ai_determinism_with_fixed_seed(self):
        """RandomAI produces a reproducible move sequence for a fixed seed."""
        game_state = self._make_minimal_game_state()

        moves = [MagicMock(), MagicMock(), MagicMock()]

        cfg1 = AIConfig(
            difficulty=5,
            randomness=1.0,
            rngSeed=42,
        )
        ai1 = RandomAI(player_number=1, config=cfg1)
        ai1.simulate_thinking = MagicMock()
        ai1.rules_engine.get_valid_moves = MagicMock(return_value=moves)

        cfg2 = AIConfig(
            difficulty=5,
            randomness=1.0,
            rngSeed=42,
        )
        ai2 = RandomAI(player_number=1, config=cfg2)
        ai2.simulate_thinking = MagicMock()
        ai2.rules_engine.get_valid_moves = MagicMock(return_value=moves)

        seq1 = [ai1.select_move(game_state) for _ in range(3)]
        seq2 = [ai2.select_move(game_state) for _ in range(3)]

        self.assertEqual(seq1, seq2)


if __name__ == "__main__":
    unittest.main()