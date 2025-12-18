#!/usr/bin/env python3
"""Distill CNN policy knowledge to NNUE training data.

This script generates soft policy targets from a trained CNN model,
enabling KL divergence training for NNUE without expensive MCTS search.

Usage:
    python scripts/distill_cnn_to_nnue.py \
        --input data/selfplay/games.jsonl \
        --output data/distilled/sq8_cnn_policy.jsonl \
        --board-type square8 \
        --batch-size 64
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.models import BoardType, GameState, Move, MoveType, Position
from app.ai.neural_net import (
    NeuralNetAI,
    get_policy_size_for_board,
    encode_state_for_nn,
)
from app.ai.nnue import extract_features_from_gamestate, get_board_size
from app.game_engine import GameEngine
from app.training.generate_data import create_initial_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_board_type(board_str: str) -> BoardType:
    """Parse board type string to enum."""
    board_str = board_str.lower()
    if "square8" in board_str or "sq8" in board_str:
        return BoardType.SQUARE8
    elif "square19" in board_str or "sq19" in board_str:
        return BoardType.SQUARE19
    elif "hex" in board_str:
        return BoardType.HEXAGONAL
    return BoardType.SQUARE8


def parse_move(move_dict: dict, move_number: int) -> Optional[Move]:
    """Parse a move dict into a Move object."""
    move_type_str = move_dict.get("type", "")
    try:
        move_type = MoveType(move_type_str)
    except ValueError:
        return None

    def parse_pos(pos_dict):
        if not pos_dict or not isinstance(pos_dict, dict):
            return None
        return Position(
            x=pos_dict.get("x", 0),
            y=pos_dict.get("y", 0),
            z=pos_dict.get("z"),
        )

    from_pos = parse_pos(move_dict.get("from") or move_dict.get("from_pos"))
    to_pos = parse_pos(move_dict.get("to"))
    capture_target = parse_pos(move_dict.get("capture_target"))

    return Move(
        id=move_dict.get("id", f"distill-{move_number}"),
        type=move_type,
        player=move_dict.get("player", 1),
        from_pos=from_pos,
        to=to_pos,
        capture_target=capture_target,
        move_number=move_number,
    )


class CNNDistiller:
    """Distills CNN policy knowledge into soft targets."""

    def __init__(
        self,
        board_type: BoardType,
        num_players: int = 2,
        device: str = "cuda",
        temperature: float = 1.0,
    ):
        self.board_type = board_type
        self.num_players = num_players
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.temperature = temperature

        # Load CNN model
        logger.info(f"Loading CNN model for {board_type.value}...")
        self.nn_ai = NeuralNetAI(
            player_number=1,
            board_type=board_type,
        )
        self.model = self.nn_ai.model
        self.model.eval()
        self.model.to(self.device)

        self.policy_size = get_policy_size_for_board(board_type)
        self.board_size = get_board_size(board_type)
        logger.info(f"CNN loaded: policy_size={self.policy_size}, device={self.device}")

    @torch.no_grad()
    def get_policy_batch(
        self,
        states: List[GameState],
        players: List[int],
    ) -> List[Dict[str, float]]:
        """Get CNN policy distributions for a batch of states.

        Returns list of sparse policy dicts {move_idx: prob}.
        """
        if not states:
            return []

        # Encode states for CNN
        batch_tensors = []
        for state, player in zip(states, players):
            encoded = encode_state_for_nn(state, player, self.board_type)
            batch_tensors.append(torch.from_numpy(encoded).float())

        batch = torch.stack(batch_tensors).to(self.device)

        # Get CNN policy output
        output = self.model(batch)
        if isinstance(output, tuple):
            if len(output) >= 2:
                _, policy_logits = output[0], output[1]
            else:
                policy_logits = output[0]
        else:
            policy_logits = output

        # Apply temperature and softmax
        policy_probs = torch.softmax(policy_logits / self.temperature, dim=-1)

        # Convert to sparse dicts (top-k moves with prob > threshold)
        results = []
        for i in range(len(states)):
            probs = policy_probs[i].cpu().numpy()

            # Get legal moves for this state
            legal_moves = GameEngine.get_valid_moves(states[i], players[i])

            # Create sparse policy dict for legal moves only
            policy_dict = {}
            for move_idx, move in enumerate(legal_moves[:128]):  # Max 128 moves
                # Get move encoding (simplified - use index directly)
                if move_idx < len(probs):
                    prob = float(probs[move_idx])
                    if prob > 1e-6:
                        policy_dict[str(move_idx)] = prob

            # Normalize
            total = sum(policy_dict.values())
            if total > 0:
                policy_dict = {k: v / total for k, v in policy_dict.items()}

            results.append(policy_dict)

        return results

    def distill_game(
        self,
        game: Dict[str, Any],
        sample_every: int = 1,
    ) -> Dict[str, Any]:
        """Add CNN policy distributions to a game record."""
        # Parse initial state
        board_type_str = game.get("board_type", "square8")
        num_players = game.get("num_players", 2)

        initial_state_dict = game.get("initial_state")
        if initial_state_dict:
            try:
                state = GameState(**initial_state_dict)
            except Exception:
                state = create_initial_state(self.board_type, num_players)
        else:
            state = create_initial_state(self.board_type, num_players)

        moves = game.get("moves", [])
        distilled_moves = []
        positions_distilled = 0

        for move_idx, move_dict in enumerate(moves):
            new_move = dict(move_dict)

            # Distill every Nth position
            game_status = state.game_status.value if hasattr(state.game_status, 'value') else str(state.game_status)
            if move_idx % sample_every == 0 and game_status == "active":
                current_player = state.current_player or 1

                # Get CNN policy for this position
                try:
                    policy_dicts = self.get_policy_batch([state], [current_player])
                    if policy_dicts and policy_dicts[0]:
                        new_move["cnn_policy"] = policy_dicts[0]
                        positions_distilled += 1
                except Exception as e:
                    logger.debug(f"Failed to distill position {move_idx}: {e}")

            distilled_moves.append(new_move)

            # Apply move to advance state
            try:
                move = parse_move(move_dict, move_idx)
                if move is not None:
                    state = GameEngine.apply_move(state, move)
            except Exception:
                break

        # Return game with distilled policies
        result = dict(game)
        result["moves"] = distilled_moves
        result["distilled"] = True
        result["positions_distilled"] = positions_distilled

        return result


def main():
    parser = argparse.ArgumentParser(description="Distill CNN policy to NNUE training data")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--board-type", type=str, default="square8", help="Board type")
    parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    parser.add_argument("--sample-every", type=int, default=1, help="Distill every Nth position")
    parser.add_argument("--max-games", type=int, default=None, help="Max games to process")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for CNN inference")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    board_type = parse_board_type(args.board_type)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"CNN Policy Distillation")
    logger.info(f"  Input: {args.input}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Board: {board_type.value}, Players: {args.num_players}")
    logger.info(f"  Temperature: {args.temperature}")

    # Initialize distiller
    distiller = CNNDistiller(
        board_type=board_type,
        num_players=args.num_players,
        device=args.device,
        temperature=args.temperature,
    )

    # Process games
    start_time = time.time()
    games_processed = 0
    total_positions = 0

    with open(args.input, 'r') as fin, open(output_path, 'w') as fout:
        for line_num, line in enumerate(fin):
            if args.max_games and games_processed >= args.max_games:
                break

            if not line.strip():
                continue

            try:
                game = json.loads(line)

                # Filter by board type and num_players
                game_board = game.get("board_type", "").lower()
                if board_type.value.lower() not in game_board:
                    continue
                if game.get("num_players") != args.num_players:
                    continue

                # Distill
                distilled = distiller.distill_game(game, sample_every=args.sample_every)
                fout.write(json.dumps(distilled) + "\n")

                games_processed += 1
                total_positions += distilled.get("positions_distilled", 0)

                if games_processed % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = games_processed / elapsed
                    logger.info(f"Processed {games_processed} games ({rate:.1f}/s), {total_positions} positions")

            except Exception as e:
                logger.warning(f"Failed to process game {line_num}: {e}")

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("DISTILLATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Games processed: {games_processed}")
    logger.info(f"Positions distilled: {total_positions}")
    logger.info(f"Time: {elapsed:.1f}s ({games_processed / elapsed:.1f} games/s)")
    logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
