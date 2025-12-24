#!/usr/bin/env python
"""Parallel MCTS self-play with batched neural network evaluation.

This script generates self-play games using BatchedGumbelMCTS, which runs
MCTS search on multiple games simultaneously for 3-4x speedup over sequential.

Key optimization:
- Standard: N games × 4 phases × batch = N×4 NN forward passes
- Batched: 4 phases × (N × batch) = 4 NN forward passes (larger batches)

Usage:
    # Run 16 games in parallel batches
    python scripts/run_parallel_mcts_selfplay.py \
        --board square8 --num-players 2 \
        --num-games 1000 --parallel-batch 16 \
        --output-dir data/selfplay/parallel_mcts

    # Use specific model
    python scripts/run_parallel_mcts_selfplay.py \
        --model-path models/nnue/nnue_square8_2p.pt \
        --board square8 --num-players 2 \
        --num-games 500
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add ai-service to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from app.ai.batched_gumbel_mcts import create_batched_gumbel_mcts
from app.db import get_or_create_db, record_completed_game_with_parity_check
from app.game_engine import GameEngine
from app.models import BoardType, Move
from app.rules.mutable_state import MutableGameState
from app.training.initial_state import create_initial_state
from app.training.selfplay_model_selector import SelfplayModelSelector
from app.utils.victory_type import derive_victory_type
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("run_parallel_mcts_selfplay")


def create_neural_net(board_type: BoardType, num_players: int, model_path: str | None = None):
    """Create neural network for evaluation.

    Uses NeuralNetAI which provides a GameState-based interface that
    the Gumbel MCTS expects. This works with both standard NN models
    and NNUE checkpoints (which NeuralNetAI can also load).
    """
    if model_path is None:
        logger.info("No model path provided, using random policy")
        return None

    try:
        from app.ai.neural_net import NeuralNetAI
        from app.models.core import AIConfig, AIType

        config = AIConfig(
            ai_type=AIType.NEURAL_NET,
            difficulty=7,
            use_neural_net=True,
        )

        nn = NeuralNetAI(
            player_number=1,
            config=config,
            board_type=board_type,
        )

        if Path(model_path).exists():
            nn.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")

        return nn

    except Exception as e:
        import traceback
        logger.warning(f"Failed to create neural network: {e}")
        logger.debug(traceback.format_exc())
        return None


def run_single_game(
    game_state,
    batched_mcts,
    max_moves: int,
    player_numbers: list[int],
) -> tuple[list[Move], int, str]:
    """Run a single game to completion.

    Args:
        game_state: Initial game state.
        batched_mcts: Batched MCTS AI (will be called with single game).
        max_moves: Maximum moves before timeout.
        player_numbers: List of player numbers.

    Returns:
        Tuple of (moves_played, winner, victory_type)
    """
    moves_played = []
    move_count = 0

    while game_state.game_status == "active" and move_count < max_moves:
        current_player = game_state.current_player

        # Get move from MCTS
        batched_mcts.player_number = current_player
        moves = batched_mcts.select_moves_batch([game_state])
        best_move = moves[0] if moves else None

        if best_move is None:
            # No valid moves - check victory
            GameEngine._check_victory(game_state)
            break

        # Apply move
        move_timestamp = datetime.now(timezone.utc)
        stamped_move = best_move.model_copy(
            update={
                "id": f"move-{move_count + 1}",
                "timestamp": move_timestamp,
                "think_time": 0,
                "move_number": move_count + 1,
            }
        )

        game_state = GameEngine.apply_move(game_state, stamped_move)
        moves_played.append(stamped_move)
        move_count += 1

    winner = game_state.winner or 0
    victory_type, _ = derive_victory_type(game_state, max_moves)

    return moves_played, winner, victory_type


def run_parallel_batch(
    game_states: list,
    batched_mcts,
    max_moves: int,
) -> list[tuple]:
    """Run multiple games in parallel with batched MCTS.

    Args:
        game_states: List of initial game states.
        batched_mcts: Batched MCTS AI.
        max_moves: Maximum moves per game.

    Returns:
        List of (game_state, moves, winner, victory_type) tuples.
    """
    n_games = len(game_states)
    results = [None] * n_games

    # Track which games are still active
    active_indices = list(range(n_games))
    moves_played = [[] for _ in range(n_games)]
    move_counts = [0] * n_games

    while active_indices:
        # Get current states for active games
        active_states = [game_states[i] for i in active_indices]

        # Update player number for all active games (assumes same player across batch)
        if active_states:
            batched_mcts.player_number = active_states[0].current_player

        # Batch select moves for all active games
        batch_moves = batched_mcts.select_moves_batch(active_states)

        # Apply moves and check for completion
        new_active = []
        for batch_idx, game_idx in enumerate(active_indices):
            move = batch_moves[batch_idx] if batch_idx < len(batch_moves) else None

            if move is None:
                # Game over - no valid moves
                GameEngine._check_victory(game_states[game_idx])
                winner = game_states[game_idx].winner or 0
                victory_type, _ = derive_victory_type(game_states[game_idx], max_moves)
                results[game_idx] = (
                    game_states[game_idx],
                    moves_played[game_idx],
                    winner,
                    victory_type,
                )
                continue

            # Apply move
            move_count = move_counts[game_idx]
            move_timestamp = datetime.now(timezone.utc)
            stamped_move = move.model_copy(
                update={
                    "id": f"move-{move_count + 1}",
                    "timestamp": move_timestamp,
                    "think_time": 0,
                    "move_number": move_count + 1,
                }
            )

            game_states[game_idx] = GameEngine.apply_move(game_states[game_idx], stamped_move)
            moves_played[game_idx].append(stamped_move)
            move_counts[game_idx] += 1

            # Check if game is over
            if game_states[game_idx].game_status != "active" or move_counts[game_idx] >= max_moves:
                winner = game_states[game_idx].winner or 0
                victory_type, _ = derive_victory_type(game_states[game_idx], max_moves)
                results[game_idx] = (
                    game_states[game_idx],
                    moves_played[game_idx],
                    winner,
                    victory_type,
                )
            else:
                new_active.append(game_idx)

        active_indices = new_active

        # Clear engine cache periodically
        if len(active_indices) < n_games // 2:
            GameEngine.clear_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Parallel MCTS self-play with batched NN evaluation"
    )
    parser.add_argument(
        "--board", "-b",
        type=str,
        default="square8",
        choices=["square8", "square19", "hex8", "hex", "hexagonal"],
        help="Board type",
    )
    parser.add_argument(
        "--num-players", "-p",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players",
    )
    parser.add_argument(
        "--num-games", "-n",
        type=int,
        default=100,
        help="Number of games to generate",
    )
    parser.add_argument(
        "--parallel-batch",
        type=int,
        default=16,
        help="Number of games to run in parallel",
    )
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=800,
        help="MCTS simulation budget per move (800+ recommended for quality)",
    )
    parser.add_argument(
        "--sampled-actions",
        type=int,
        default=16,
        help="Number of actions to sample (Gumbel-Top-K)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to neural network model",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/selfplay/parallel_mcts",
        help="Output directory for games",
    )
    parser.add_argument(
        "--record-db",
        type=str,
        default=None,
        help="SQLite database to record games",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=None,
        help="Maximum moves per game (auto-calculated if not set)",
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Parse board type
    board_type_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hex8": BoardType.HEX8,
        "hex": BoardType.HEXAGONAL,
        "hexagonal": BoardType.HEXAGONAL,
    }
    board_type = board_type_map.get(args.board.lower(), BoardType.SQUARE8)

    # Auto-calculate max_moves
    if args.max_moves is None:
        max_moves_table = {
            ("square8", 2): 500,
            ("square8", 3): 800,
            ("square8", 4): 1200,
            ("hex8", 2): 500,
            ("hex8", 3): 800,
            ("hex", 2): 1200,
            ("hexagonal", 2): 1200,
        }
        args.max_moves = max_moves_table.get(
            (args.board.lower(), args.num_players), 1000
        )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Auto-detect model if not specified
    model_path = args.model_path
    if model_path is None:
        logger.info("No model specified, using auto-detection...")
        selector = SelfplayModelSelector(
            board_type=args.board,
            num_players=args.num_players,
            prefer_nnue=True,  # Prefer NNUE for fast inference
        )
        detected_path = selector.get_current_model()
        if detected_path:
            model_path = str(detected_path)
            logger.info(f"Auto-detected model: {model_path}")
        else:
            logger.info("No model found, using random policy")

    # Create neural network
    neural_net = create_neural_net(board_type, args.num_players, model_path)

    # Create batched MCTS
    batched_mcts = create_batched_gumbel_mcts(
        board_type=board_type,
        num_players=args.num_players,
        batch_size=args.parallel_batch,
        num_sampled_actions=args.sampled_actions,
        simulation_budget=args.mcts_sims,
        neural_net=neural_net,
    )

    # Optional database recording
    replay_db = get_or_create_db(args.record_db) if args.record_db else None

    logger.info("=" * 60)
    logger.info("PARALLEL MCTS SELF-PLAY")
    logger.info("=" * 60)
    logger.info(f"Board: {args.board}")
    logger.info(f"Players: {args.num_players}")
    logger.info(f"Games: {args.num_games}")
    logger.info(f"Parallel batch size: {args.parallel_batch}")
    logger.info(f"MCTS simulations: {args.mcts_sims}")
    logger.info(f"Sampled actions: {args.sampled_actions}")
    logger.info(f"Model: {model_path or 'None (random policy)'}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("")

    # Track statistics
    total_games = 0
    total_moves = 0
    wins_by_player = {p: 0 for p in range(1, args.num_players + 1)}
    draws = 0
    game_records = []

    games_file = os.path.join(args.output_dir, "games.jsonl")
    start_time = time.time()

    with open(games_file, "w") as f:
        # Process games in batches
        for batch_start in range(0, args.num_games, args.parallel_batch):
            batch_size = min(args.parallel_batch, args.num_games - batch_start)

            # Create initial states for this batch
            game_states = []
            initial_snapshots = []
            for _ in range(batch_size):
                state = create_initial_state(
                    board_type=board_type,
                    num_players=args.num_players,
                )
                game_states.append(state)
                initial_snapshots.append(state.model_dump(mode="json"))

            # Run batch of games
            batch_start_time = time.time()
            results = run_parallel_batch(
                game_states,
                batched_mcts,
                args.max_moves,
            )
            batch_time = time.time() - batch_start_time

            # Process results
            for batch_idx, result in enumerate(results):
                final_state, moves, winner, victory_type = result
                game_idx = batch_start + batch_idx

                total_games += 1
                total_moves += len(moves)

                if winner == 0:
                    draws += 1
                else:
                    wins_by_player[winner] = wins_by_player.get(winner, 0) + 1

                # Create record
                record = {
                    "game_id": f"parallel_mcts_{args.board}_{args.num_players}p_{game_idx}_{int(time.time())}",
                    "board_type": args.board,
                    "num_players": args.num_players,
                    "winner": winner,
                    "move_count": len(moves),
                    "victory_type": victory_type,
                    "engine_mode": "gumbel-mcts-batched",
                    "mcts_sims": args.mcts_sims,
                    "parallel_batch": args.parallel_batch,
                    "moves": [
                        {
                            "type": m.type.value if hasattr(m.type, 'value') else str(m.type),
                            "player": m.player,
                            "to": {"x": m.to.x, "y": m.to.y} if hasattr(m, 'to') and m.to else None,
                        }
                        for m in moves
                    ],
                    "initial_state": initial_snapshots[batch_idx],
                    "timestamp": datetime.now().isoformat(),
                    "source": "run_parallel_mcts_selfplay.py",
                }

                f.write(json.dumps(record) + "\n")
                game_records.append(record)

                # Record to database if available
                if replay_db is not None:
                    try:
                        record_completed_game_with_parity_check(
                            db=replay_db,
                            initial_state=game_states[batch_idx],
                            final_state=final_state,
                            moves=moves,
                            metadata={
                                "source": "run_parallel_mcts_selfplay.py",
                                "engine_mode": "gumbel-mcts-batched",
                            },
                            game_id=record["game_id"],
                        )
                    except Exception as e:
                        logger.warning(f"Failed to record game: {e}")

            # Log batch progress
            elapsed = time.time() - start_time
            games_per_sec = total_games / elapsed
            logger.info(
                f"Batch {batch_start // args.parallel_batch + 1}: "
                f"{total_games}/{args.num_games} games, "
                f"{games_per_sec:.2f} g/s, "
                f"batch time: {batch_time:.1f}s"
            )

            # Clear engine cache
            GameEngine.clear_cache()

    total_time = time.time() - start_time

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total games: {total_games}")
    logger.info(f"Total moves: {total_moves}")
    logger.info(f"Avg moves/game: {total_moves / total_games:.1f}")
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Throughput: {total_games / total_time:.2f} games/sec")
    logger.info("")
    logger.info("Win rates:")
    for p in range(1, args.num_players + 1):
        wins = wins_by_player.get(p, 0)
        rate = wins / total_games * 100 if total_games > 0 else 0
        logger.info(f"  Player {p}: {wins} wins ({rate:.1f}%)")
    logger.info(f"  Draws: {draws}")
    logger.info("")
    logger.info(f"Games saved to: {games_file}")

    # Save stats
    stats_file = os.path.join(args.output_dir, "stats.json")
    stats = {
        "total_games": total_games,
        "total_moves": total_moves,
        "total_time": total_time,
        "games_per_second": total_games / total_time,
        "wins_by_player": wins_by_player,
        "draws": draws,
        "config": {
            "board": args.board,
            "num_players": args.num_players,
            "parallel_batch": args.parallel_batch,
            "mcts_sims": args.mcts_sims,
            "sampled_actions": args.sampled_actions,
            "model_path": model_path,
        },
    }
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Stats saved to: {stats_file}")


if __name__ == "__main__":
    main()
