#!/usr/bin/env python3
"""Test EBMO Online Learning implementation."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from app.ai.ebmo_online import EBMOOnlineAI, EBMOOnlineConfig
from app.ai.random_ai import RandomAI
from app.models import AIConfig, BoardType, GameStatus
from app.game_engine import GameEngine
from app.training.initial_state import create_initial_state

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def play_game_with_learning(ebmo_ai, random_ai, max_moves: int = 500):
    """Play a game with online learning enabled."""
    state = create_initial_state(board_type=BoardType.SQUARE8, num_players=2)
    engine = GameEngine()
    move_count = 0

    # Determine player assignments
    ebmo_player = ebmo_ai.player_number
    random_player = random_ai.player_number
    ais = {ebmo_player: ebmo_ai, random_player: random_ai}

    ebmo_ai.reset_for_new_game()

    while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current_ai = ais[state.current_player]
        move = current_ai.select_move(state)

        if move is None:
            break

        state = engine.apply_move(state, move)
        move_count += 1

    # Report learning metrics
    winner = state.winner
    metrics = ebmo_ai.end_game(winner)

    return winner, move_count, metrics


def main():
    model_path = "models/ebmo_56ch/ebmo_quality_best.pt"
    num_games = 5

    logger.info("=" * 60)
    logger.info("EBMO Online Learning Test")
    logger.info("=" * 60)

    ai_config = AIConfig(difficulty=5)
    online_config = EBMOOnlineConfig(
        buffer_size=10,
        learning_rate=1e-5,
        td_lambda=0.9,
        gamma=0.99,
        batch_size=4,
    )

    # Create online learning AI
    ebmo_ai = EBMOOnlineAI(
        player_number=1,
        config=ai_config,
        model_path=model_path,
        online_config=online_config,
        enable_online_learning=True,
    )

    logger.info(f"Online learning enabled: {ebmo_ai.learner is not None}")

    ebmo_wins = 0
    total_losses = []

    for game_num in range(num_games):
        # Random plays as P2
        random_ai = RandomAI(2, ai_config)

        winner, moves, metrics = play_game_with_learning(ebmo_ai, random_ai)

        result = "EBMO wins" if winner == 1 else "Random wins"
        if winner == 1:
            ebmo_wins += 1

        logger.info(f"Game {game_num + 1}: {result} (moves={moves})")
        logger.info(f"  Metrics: {metrics}")

        if metrics and 'total_loss' in metrics:
            total_losses.append(metrics['total_loss'])

    logger.info("=" * 60)
    logger.info("Final Stats:")
    logger.info(f"  EBMO wins: {ebmo_wins}/{num_games}")
    logger.info(f"  Learning stats: {ebmo_ai.get_stats()}")
    if total_losses:
        logger.info(f"  Average loss: {sum(total_losses) / len(total_losses):.6f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
