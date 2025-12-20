#!/usr/bin/env python
"""Quick evaluation of EBMO models with direct evaluation mode."""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.ai.ebmo_ai import EBMO_AI
from app.ai.ebmo_network import EBMOConfig
from app.ai.factory import AIFactory
from app.models.core import AIType, AIConfig, BoardType
from app.game_engine import GameEngine
from app.training.generate_data import create_initial_state


def play_game(ai1, ai2, max_moves=200):
    """Play a single game and return winner."""
    state = create_initial_state(board_type=BoardType.SQUARE8, num_players=2)
    engine = GameEngine()

    for _ in range(max_moves):
        if state.winner is not None:
            return state.winner
        if hasattr(state.game_status, 'value') and state.game_status.value != 'active':
            return state.winner

        current = state.current_player
        ai = ai1 if current == 1 else ai2

        move = ai.select_move(state)
        if move is None:
            return 3 - current  # Other player wins

        state = engine.apply_move(state, move)

    return state.winner


def evaluate_model(model_path: str, num_games: int = 10):
    """Evaluate model with direct evaluation."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"Mode: Direct evaluation (no gradient descent)")
    print(f"{'='*60}\n")

    # Create EBMO with direct eval mode
    config = EBMOConfig(use_direct_eval=True)
    ebmo = EBMO_AI(
        player_number=1,
        config=AIConfig(difficulty=5),
        model_path=model_path,
        ebmo_config=config,
    )

    # Test vs Random
    random_ai = AIFactory.create(AIType.RANDOM, 2, AIConfig(difficulty=1))
    wins = 0
    for i in range(num_games):
        ebmo.reset_for_new_game()
        random_ai.reset_for_new_game()
        winner = play_game(ebmo, random_ai)
        if winner == 1:
            wins += 1
        result = "won" if winner == 1 else "lost"
        print(f"  Game {i+1}: EBMO {result}")

    print(f"vs Random: {wins}/{num_games} ({100*wins/num_games:.0f}%)")

    # Test vs Heuristic
    heuristic_ai = AIFactory.create(AIType.HEURISTIC, 2, AIConfig(difficulty=5))
    wins = 0
    for i in range(num_games):
        ebmo.reset_for_new_game()
        heuristic_ai.reset_for_new_game()
        winner = play_game(ebmo, heuristic_ai)
        if winner == 1:
            wins += 1
        result = "won" if winner == 1 else "lost"
        print(f"  Game {i+1}: EBMO {result}")

    print(f"vs Heuristic: {wins}/{num_games} ({100*wins/num_games:.0f}%)")

    return wins


if __name__ == "__main__":
    # Test both models
    models = [
        "models/ebmo/ebmo_improved_best.pt",
        "models/ebmo/ebmo_square8_epoch69.pt",  # Earlier self-play checkpoint
    ]

    for model in models:
        if os.path.exists(model):
            evaluate_model(model, num_games=10)
        else:
            print(f"Model not found: {model}")
