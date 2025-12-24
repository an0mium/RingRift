#!/usr/bin/env python
"""Hybrid NN-value self-play data generation.

This script generates training data using the HybridNNValuePlayer,
which combines fast heuristic move generation with NN value ranking.

This provides 5-10x speedup over full MCTS while maintaining move quality,
helping to close the self-play training loop.

Usage:
    # Generate 1000 games
    PYTHONPATH=. python scripts/run_hybrid_selfplay.py \
        --num-games 1000 \
        --board square8 \
        --num-players 2 \
        --output-dir data/selfplay/hybrid_sq8_2p

    # With specific model
    PYTHONPATH=. python scripts/run_hybrid_selfplay.py \
        --num-games 500 \
        --board hex8 \
        --num-players 2 \
        --model-path models/canonical_hex8_2p.pth \
        --output-dir data/selfplay/hybrid_hex8_2p

Output:
    - games_*.jsonl: Game records with engine_mode="hybrid_nn_value"
    - stats.json: Aggregated statistics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add app/ to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("run_hybrid_selfplay")


def run_single_game(
    player1,
    player2,
    board_type_enum,
    num_players: int,
    max_moves: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """Run a single game between two hybrid players."""
    from app.game_engine import GameEngine
    from app.models import GameStatus
    from app.rules.serialization import serialize_game_state
    from app.training.initial_state import create_initial_state

    # Create initial state
    game_state = create_initial_state(board_type_enum, num_players=num_players)

    # Snapshot initial state
    initial_state_snapshot = serialize_game_state(game_state)

    moves_played = []
    move_count = 0
    players = [player1, player2] if num_players == 2 else [player1, player2, player1, player2][:num_players]

    while game_state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current_player_idx = game_state.current_player - 1
        player = players[current_player_idx % len(players)]

        # Get valid moves
        valid_moves = GameEngine.get_valid_moves(game_state, game_state.current_player)

        if not valid_moves:
            # Check for phase requirements that need bookkeeping moves
            requirement = GameEngine.get_phase_requirement(game_state, game_state.current_player)
            if requirement is not None:
                move = GameEngine.synthesize_bookkeeping_move(requirement, game_state)
                if move:
                    game_state = GameEngine.apply_move(game_state, move)
                    if hasattr(move, 'model_dump'):
                        moves_played.append(move.model_dump(mode="json"))
                    move_count += 1
                    continue
            break

        # Select move using hybrid player
        move = player.select_move(game_state, valid_moves)

        if move is None:
            break

        # Apply move
        game_state = GameEngine.apply_move(game_state, move)

        # Serialize move
        if hasattr(move, 'model_dump'):
            moves_played.append(move.model_dump(mode="json"))
        elif hasattr(move, 'to_dict'):
            moves_played.append(move.to_dict())
        else:
            moves_played.append(str(move))

        move_count += 1

    # Build game record
    board_type_str = board_type_enum.value if hasattr(board_type_enum, 'value') else str(board_type_enum)
    return {
        "game_id": f"hybrid_{board_type_str}_{num_players}p_{seed}_{int(datetime.now().timestamp())}",
        "board_type": board_type_str,
        "num_players": num_players,
        "seed": seed,
        "winner": game_state.winner,
        "move_count": move_count,
        "game_status": game_state.game_status,
        "moves": moves_played,
        "initial_state": initial_state_snapshot,
        "timestamp": datetime.now().isoformat(),
        "source": "run_hybrid_selfplay.py",
        "engine_mode": "hybrid_nn_value",
        "player_types": ["hybrid_nn_value"] * num_players,
    }


def run_hybrid_selfplay(
    board_type: str,
    num_players: int,
    num_games: int,
    output_dir: Path,
    model_path: str | None = None,
    top_k: int = 5,
    temperature: float = 0.1,
    max_moves: int = 500,
) -> dict[str, Any]:
    """Run hybrid selfplay and generate training data."""
    from app.ai.hybrid_gpu import HybridNNValuePlayer
    from app.models import BoardType

    # Convert board_type string to enum
    board_type_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hex8": BoardType.HEX8,
        "hexagonal": BoardType.HEXAGONAL,
    }
    board_type_enum = board_type_map.get(board_type.lower(), BoardType.SQUARE8)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize players
    logger.info(f"Initializing HybridNNValuePlayer (board={board_type}, players={num_players})")

    player1 = HybridNNValuePlayer(
        board_type=board_type,
        num_players=num_players,
        player_number=1,
        top_k=top_k,
        temperature=temperature,
        model_path=model_path,
    )

    player2 = HybridNNValuePlayer(
        board_type=board_type,
        num_players=num_players,
        player_number=2,
        top_k=top_k,
        temperature=temperature,
        model_path=model_path,
    )

    # Output file
    output_file = output_dir / f"games_hybrid_{board_type}_{num_players}p_{os.getpid()}.jsonl"

    # Stats tracking
    stats = {
        "games_completed": 0,
        "games_failed": 0,
        "total_moves": 0,
        "winners": {str(i): 0 for i in range(num_players + 1)},  # 0 = draw
        "start_time": datetime.now().isoformat(),
        "board_type": board_type,
        "num_players": num_players,
        "engine_mode": "hybrid_nn_value",
        "top_k": top_k,
        "temperature": temperature,
    }

    start_time = time.time()

    logger.info(f"Starting {num_games} games, output to {output_file}")

    with open(output_file, "w") as f:
        for game_idx in range(num_games):
            try:
                record = run_single_game(
                    player1=player1,
                    player2=player2,
                    board_type_enum=board_type_enum,
                    num_players=num_players,
                    max_moves=max_moves,
                    seed=game_idx,
                )

                f.write(json.dumps(record) + "\n")
                f.flush()

                stats["games_completed"] += 1
                stats["total_moves"] += record["move_count"]
                winner_key = str(record["winner"]) if record["winner"] else "0"
                stats["winners"][winner_key] = stats["winners"].get(winner_key, 0) + 1

                if (game_idx + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    games_per_hr = stats["games_completed"] / elapsed * 3600
                    logger.info(
                        f"Progress: {stats['games_completed']}/{num_games} games "
                        f"({games_per_hr:.1f} games/hr)"
                    )

            except Exception as e:
                logger.error(f"Game {game_idx} failed: {e}")
                stats["games_failed"] += 1

    # Finalize stats
    elapsed = time.time() - start_time
    stats["end_time"] = datetime.now().isoformat()
    stats["elapsed_seconds"] = elapsed
    stats["games_per_hour"] = stats["games_completed"] / elapsed * 3600 if elapsed > 0 else 0
    stats["avg_moves_per_game"] = stats["total_moves"] / max(1, stats["games_completed"])

    # Player stats
    stats["player1_stats"] = player1.get_stats()
    stats["player2_stats"] = player2.get_stats()

    # Save stats
    stats_file = output_dir / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Completed: {stats['games_completed']} games in {elapsed:.1f}s")
    logger.info(f"Rate: {stats['games_per_hour']:.1f} games/hr")
    logger.info(f"Output: {output_file}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate self-play data using HybridNNValuePlayer"
    )
    parser.add_argument(
        "--board",
        type=str,
        default="square8",
        choices=["square8", "square19", "hex8", "hexagonal"],
        help="Board type",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="Number of games to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/selfplay/hybrid",
        help="Output directory",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model (optional)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top moves to consider from heuristic",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Selection temperature (0=greedy)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=500,
        help="Maximum moves per game",
    )

    args = parser.parse_args()

    stats = run_hybrid_selfplay(
        board_type=args.board,
        num_players=args.num_players,
        num_games=args.num_games,
        output_dir=Path(args.output_dir),
        model_path=args.model_path,
        top_k=args.top_k,
        temperature=args.temperature,
        max_moves=args.max_moves,
    )

    print(f"\nFinal stats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
