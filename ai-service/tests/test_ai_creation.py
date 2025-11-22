import pytest
from app.main import (
    _create_ai_instance,
    _select_ai_type,
    _get_difficulty_profile,
)
from app.models import AIType, AIConfig
from app.ai.minimax_ai import MinimaxAI
from app.ai.mcts_ai import MCTSAI
from app.ai.heuristic_ai import HeuristicAI
from app.ai.random_ai import RandomAI


def test_select_ai_type():
    assert _select_ai_type(1) == AIType.RANDOM
    assert _select_ai_type(2) == AIType.RANDOM
    assert _select_ai_type(3) == AIType.HEURISTIC
    assert _select_ai_type(5) == AIType.HEURISTIC
    assert _select_ai_type(6) == AIType.MINIMAX
    assert _select_ai_type(8) == AIType.MINIMAX
    assert _select_ai_type(9) == AIType.MCTS
    assert _select_ai_type(10) == AIType.MCTS


def test_difficulty_profile_mapping():
    """Canonical ladder profiles for difficulties 1–10."""
    profiles = {
        1: (AIType.RANDOM, 0.5, 150, "v1-random-1"),
        2: (AIType.RANDOM, 0.3, 200, "v1-random-2"),
        3: (AIType.HEURISTIC, 0.2, 250, "v1-heuristic-3"),
        4: (AIType.HEURISTIC, 0.1, 300, "v1-heuristic-4"),
        5: (AIType.HEURISTIC, 0.05, 350, "v1-heuristic-5"),
        6: (AIType.MINIMAX, 0.02, 400, "v1-minimax-6"),
        7: (AIType.MINIMAX, 0.01, 450, "v1-minimax-7"),
        8: (AIType.MINIMAX, 0.0, 500, "v1-minimax-8"),
        9: (AIType.MCTS, 0.0, 600, "v1-mcts-9"),
        10: (AIType.MCTS, 0.0, 700, "v1-mcts-10"),
    }

    for difficulty, (
        expected_type,
        expected_randomness,
        expected_think_ms,
        expected_profile_id,
    ) in profiles.items():
        profile = _get_difficulty_profile(difficulty)
        assert profile["ai_type"] == expected_type
        assert profile["randomness"] == pytest.approx(expected_randomness)
        assert profile["think_time_ms"] == expected_think_ms
        assert profile["profile_id"] == expected_profile_id


def test_difficulty_profile_clamping():
    """Out-of-range difficulties are clamped into [1, 10] consistently."""
    low = _get_difficulty_profile(0)
    high = _get_difficulty_profile(11)

    assert low == _get_difficulty_profile(1)
    assert high == _get_difficulty_profile(10)


def test_create_ai_instance():
    config = AIConfig(difficulty=5, randomness=0.1, rngSeed=None)

    # Test Random
    ai = _create_ai_instance(AIType.RANDOM, 1, config)
    assert isinstance(ai, RandomAI)

    # Test Heuristic
    ai = _create_ai_instance(AIType.HEURISTIC, 1, config)
    assert isinstance(ai, HeuristicAI)

    # Test Minimax
    ai = _create_ai_instance(AIType.MINIMAX, 1, config)
    assert isinstance(ai, MinimaxAI)

    # Test MCTS
    ai = _create_ai_instance(AIType.MCTS, 1, config)
    assert isinstance(ai, MCTSAI)