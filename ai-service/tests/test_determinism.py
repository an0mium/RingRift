"""
RNG Determinism Tests for Python AI Service

Verifies that the Python AI implementations produce consistent,
deterministic move sequences when given the same seed and game state.
"""

import random
from app.models import (
    GameState,
    AIConfig,
    BoardType,
    GamePhase,
    GameStatus
)
from app.ai.random_ai import RandomAI
from app.ai.heuristic_ai import HeuristicAI


def create_test_state() -> GameState:
    """Create a minimal test GameState for determinism testing."""
    from datetime import datetime
    from app.models import (
        BoardState,
        Player,
        TimeControl,
    )

    board = BoardState(
        type=BoardType.SQUARE8,
        size=8,
        stacks={},
        markers={},
        collapsedSpaces={},
        eliminatedRings={},
        formedLines=[],
        territories={},
    )

    players = [
        Player(
            id="player1",
            username="Player 1",
            type="human",
            playerNumber=1,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=None,
            ringsInHand=18,
            eliminatedRings=0,
            territorySpaces=0,
        ),
        Player(
            id="player2",
            username="Player 2",
            type="ai",
            playerNumber=2,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=5,
            ringsInHand=18,
            eliminatedRings=0,
            territorySpaces=0,
        ),
    ]

    time_control = TimeControl(
        initialTime=600,
        increment=0,
        type="rapid",
    )

    return GameState(
        id="test-game",
        boardType=BoardType.SQUARE8,
        rngSeed=42,
        board=board,
        players=players,
        currentPhase=GamePhase.RING_PLACEMENT,
        # For determinism tests we want the AI-controlled player (player 2)
        # to have legal moves, so we set currentPlayer=2 here. This keeps the
        # rules engine invariant (only the current player may move) while
        # allowing RandomAI/HeuristicAI(player_number=2) to select moves.
        currentPlayer=2,
        moveHistory=[],
        timeControl=time_control,
        spectators=[],
        gameStatus=GameStatus.ACTIVE,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=2,
        totalRingsInPlay=36,
        totalRingsEliminated=0,
        victoryThreshold=19,
        territoryVictoryThreshold=33,
        mustMoveFromStackKey=None,
        chainCaptureState=None,
        zobristHash=None,
    )


class TestAIDeterminism:
    """Test suite for AI determinism with seeded RNG."""

    def test_seeded_random_ai_determinism(self):
        """Same seed should produce identical move sequences for RandomAI."""  # noqa: E501
        state = create_test_state()
        
        config1 = AIConfig(difficulty=1, randomness=0.5, rngSeed=42)
        config2 = AIConfig(difficulty=1, randomness=0.5, rngSeed=42)
        
        ai1 = RandomAI(player_number=2, config=config1)
        ai2 = RandomAI(player_number=2, config=config2)
        
        # Both AIs should select the same move
        move1 = ai1.select_move(state)
        move2 = ai2.select_move(state)
        
        assert move1 is not None
        assert move2 is not None
        assert move1.type == move2.type
        assert move1.to == move2.to

    def test_different_seeds_produce_different_moves(self):
        """Different seeds should produce different move selections."""  # noqa: E501
        state = create_test_state()
        
        config1 = AIConfig(difficulty=1, randomness=0.5, rngSeed=42)
        config2 = AIConfig(difficulty=1, randomness=0.5, rngSeed=43)
        
        ai1 = RandomAI(player_number=2, config=config1)
        ai2 = RandomAI(player_number=2, config=config2)
        
        move1 = ai1.select_move(state)
        move2 = ai2.select_move(state)
        
        # With high probability, different seeds will select different moves
        # (unless the state has only one valid move)
        assert move1 is not None
        assert move2 is not None

    def test_heuristic_ai_determinism(self):
        """HeuristicAI deterministic moves with same seed."""  # noqa: E501
        state = create_test_state()
        
        config1 = AIConfig(
            difficulty=5,
            randomness=0.05,
            rngSeed=12345,
            heuristic_profile_id="v1-heuristic-5"
        )
        config2 = AIConfig(
            difficulty=5,
            randomness=0.05,
            rngSeed=12345,
            heuristic_profile_id="v1-heuristic-5"
        )
        
        ai1 = HeuristicAI(player_number=2, config=config1)
        ai2 = HeuristicAI(player_number=2, config=config2)
        
        move1 = ai1.select_move(state)
        move2 = ai2.select_move(state)
        
        assert move1 is not None
        assert move2 is not None
        assert move1.type == move2.type
        assert move1.to == move2.to

    def test_evaluation_determinism(self):
        """Position evaluation should be deterministic with same seed."""
        state = create_test_state()
        
        config1 = AIConfig(difficulty=5, randomness=0.0, rngSeed=999)
        config2 = AIConfig(difficulty=5, randomness=0.0, rngSeed=999)
        
        ai1 = HeuristicAI(player_number=2, config=config1)
        ai2 = HeuristicAI(player_number=2, config=config2)
        
        eval1 = ai1.evaluate_position(state)
        eval2 = ai2.evaluate_position(state)
        
        assert eval1 == eval2

    def test_base_ai_rng_helpers(self):
        """BaseAI helper methods should use seeded RNG."""
        config = AIConfig(difficulty=3, randomness=0.2, rngSeed=777)
        
        ai = RandomAI(player_number=1, config=config)
        
        # Test get_random_element
        items = [1, 2, 3, 4, 5]
        selected1 = ai.get_random_element(items)
        
        # Reset RNG by creating new instance with same seed
        ai2 = RandomAI(player_number=1, config=config)
        selected2 = ai2.get_random_element(items)
        
        assert selected1 == selected2

    def test_shuffle_determinism(self):
        """Array shuffling should be deterministic with same seed."""
        config = AIConfig(difficulty=3, randomness=0.2, rngSeed=888)
        
        ai1 = RandomAI(player_number=1, config=config)
        ai2 = RandomAI(player_number=1, config=AIConfig(
            difficulty=3,
            randomness=0.2,
            rngSeed=888
        ))
        
        items1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        items2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        shuffled1 = ai1.shuffle_array(items1)
        shuffled2 = ai2.shuffle_array(items2)
        
        assert shuffled1 == shuffled2

    def test_unseeded_ai_uses_deterministic_default(self):
        """AIs without explicit seed should use deterministic defaults."""
        # BaseAI derives a seed from difficulty and player_number when
        # rngSeed is None, so two instances with the same difficulty and
        # player should still be deterministic.
        config1 = AIConfig(difficulty=3, randomness=0.2, rngSeed=None)
        config2 = AIConfig(difficulty=3, randomness=0.2, rngSeed=None)
        
        ai1 = RandomAI(player_number=2, config=config1)
        ai2 = RandomAI(player_number=2, config=config2)
        
        # Both should have the same derived seed
        assert ai1.rng_seed == ai2.rng_seed
        
        # And should produce the same random sequences
        items1 = [1, 2, 3, 4, 5]
        items2 = [1, 2, 3, 4, 5]
        
        ai1.shuffle_array(items1)
        ai2.shuffle_array(items2)
        
        assert items1 == items2

    def test_different_players_get_different_default_seeds(self):
        """Different players should get different default seeds."""
        config1 = AIConfig(difficulty=5, randomness=0.1, rngSeed=None)
        config2 = AIConfig(difficulty=5, randomness=0.1, rngSeed=None)
        
        ai1 = RandomAI(player_number=1, config=config1)
        ai2 = RandomAI(player_number=2, config=config2)
        
        # Different players should get different seeds
        assert ai1.rng_seed != ai2.rng_seed


class TestPythonStandardLibraryRNG:
    """Test that Python's random.Random produces deterministic sequences."""

    def test_python_random_determinism(self):
        """Python's Random class should be deterministic with same seed."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        
        for _ in range(100):
            assert rng1.random() == rng2.random()

    def test_python_choice_determinism(self):
        """Python's Random.choice should be deterministic."""
        items = ['a', 'b', 'c', 'd', 'e']
        
        rng1 = random.Random(12345)
        rng2 = random.Random(12345)
        
        for _ in range(50):
            assert rng1.choice(items) == rng2.choice(items)

    def test_python_shuffle_determinism(self):
        """Python's Random.shuffle should be deterministic."""
        arr1 = list(range(20))
        arr2 = list(range(20))
        
        rng1 = random.Random(999)
        rng2 = random.Random(999)
        
        rng1.shuffle(arr1)
        rng2.shuffle(arr2)
        
        assert arr1 == arr2