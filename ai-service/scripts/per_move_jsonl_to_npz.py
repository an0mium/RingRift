#!/usr/bin/env python3
"""Convert per-move JSONL format to NPZ training data.

This handles the format from CPU heuristic selfplay where each line is:
{state, outcome, board_type, game_id, move_number, ply_to_end, move, metadata}

The Hetzner selfplay uses snake_case keys which need to be converted to
camelCase for the standard deserializers.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ai.neural_net import encode_move_for_board
from app.models import BoardType, Position
from app.rules.serialization import deserialize_game_state
from app.training.encoding import HexStateEncoder, SquareStateEncoder


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


def convert_keys_to_camel(obj: Any) -> Any:
    """Recursively convert all dict keys from snake_case to camelCase."""
    if isinstance(obj, dict):
        return {snake_to_camel(k): convert_keys_to_camel(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_camel(item) for item in obj]
    else:
        return obj


def get_encoder(board_type_str: str, num_players: int):
    """Get appropriate encoder for board type."""
    if board_type_str in ("square8", "square19"):
        # Use SquareStateEncoder for square boards
        board_size = 8 if board_type_str == "square8" else 19
        board_type = BoardType.SQUARE8 if board_type_str == "square8" else BoardType.SQUARE19
        return SquareStateEncoder(board_type=board_type, board_size=board_size)
    else:
        # Use HexStateEncoder for hex boards
        BOARD_CONFIGS = {
            "hexagonal": (25, 91876),
            "hex8": (9, 4500),
        }
        board_size, policy_size = BOARD_CONFIGS[board_type_str]
        return HexStateEncoder(board_size=board_size, policy_size=policy_size)


def parse_move(move_dict: dict) -> Any:
    """Parse move from JSON.

    Hetzner selfplay format uses:
    - type: "place_ring" (lowercase)
    - to: {x, y, z} for hex cube coordinates
    """
    from app.models import Move, MoveType

    move_type_str = move_dict.get("type", "PLACE_RING")
    # Convert snake_case to UPPER_CASE for enum
    move_type_str = move_type_str.upper()
    move_type = MoveType[move_type_str]

    # Check for 'to' field (Hetzner format) or 'position' field (other formats)
    to_pos = None
    pos_data = move_dict.get("to") or move_dict.get("position")
    if pos_data:
        # Handle x/y/z cube coordinates (hex boards)
        if "x" in pos_data:
            to_pos = Position(x=pos_data.get("x", 0), y=pos_data.get("y", 0), z=pos_data.get("z", 0))
        # Handle row/col format (square boards)
        elif "row" in pos_data:
            to_pos = Position(row=pos_data["row"], col=pos_data["col"])

    return Move(
        id=move_dict.get("id", "export"),
        player=move_dict.get("player", 1),
        type=move_type,
        to=to_pos,
    )


def main():
    parser = argparse.ArgumentParser(description="Convert per-move JSONL to NPZ")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output NPZ file")
    parser.add_argument("--board-type", required=True,
                        choices=["square8", "square19", "hexagonal", "hex8"])
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to extract")
    parser.add_argument("--sample-every", type=int, default=1,
                        help="Sample every Nth position")
    args = parser.parse_args()

    encoder = get_encoder(args.board_type, args.num_players)

    features_list = []
    values_list = []
    policy_indices_list = []
    policy_values_list = []

    samples_extracted = 0
    lines_processed = 0
    errors = 0

    print(f"Processing {args.input}...", flush=True)

    with open(args.input) as f:
        for line in f:
            lines_processed += 1
            if args.max_samples and samples_extracted >= args.max_samples:
                break

            # Sample every Nth line
            if lines_processed % args.sample_every != 0:
                continue

            try:
                record = json.loads(line.strip())

                # Get state and convert keys to camelCase
                state_dict = record.get("state")
                if not state_dict:
                    continue

                state_dict = convert_keys_to_camel(state_dict)

                # Remove move_history to avoid deserialization issues with null values
                # We don't need it for feature encoding anyway
                state_dict.pop("moveHistory", None)

                # Deserialize state
                state = deserialize_game_state(state_dict)

                # Get outcome value (-1, 0, or 1 from perspective of current player)
                # Hetzner format stores outcome as a float from player 1's perspective:
                #   1.0 = player 1 wins
                #  -1.0 = player 1 loses
                #   0.0 = draw
                outcome = record.get("outcome", 0.0)
                current_player = state.current_player

                if isinstance(outcome, dict):
                    winner = outcome.get("winner", -1)
                    if winner == current_player:
                        value = 1.0
                    elif winner == -1:  # Draw
                        value = 0.0
                    else:
                        value = -1.0
                elif isinstance(outcome, (int, float)):
                    # Hetzner format: outcome is from player 1's perspective
                    # Convert to current player's perspective
                    if current_player == 1:
                        value = float(outcome)
                    else:
                        # Flip the sign for other players
                        value = -float(outcome)
                else:
                    value = 0.0

                # Encode features (without history for per-move data)
                features, globals_vec = encoder.encode_state(state)

                # Get move and encode as policy target
                move_dict = record.get("move", {})
                if move_dict:
                    move = parse_move(move_dict)
                    action_idx = encode_move_for_board(move, state.board)
                    if action_idx >= 0:
                        policy_idx = np.array([action_idx], dtype=np.int32)
                        policy_val = np.array([1.0], dtype=np.float32)
                    else:
                        continue  # Skip invalid moves
                else:
                    continue  # Skip lines without moves

                features_list.append(features)
                values_list.append(value)
                policy_indices_list.append(policy_idx)
                policy_values_list.append(policy_val)
                samples_extracted += 1

                if samples_extracted % 100 == 0:
                    print(f"  Extracted {samples_extracted:,} samples...", flush=True)

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  Error at line {lines_processed}: {e}")
                continue

    if not features_list:
        print("No samples extracted!")
        return 1

    # Stack arrays
    features = np.stack(features_list)
    values = np.array(values_list, dtype=np.float32)

    # Save NPZ
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        features=features,
        values=values,
        policy_indices=np.array(policy_indices_list, dtype=object),
        policy_values=np.array(policy_values_list, dtype=object),
    )

    print(f"\nSaved {samples_extracted:,} samples to {args.output}")
    print(f"  Features shape: {features.shape}")
    print(f"  Lines processed: {lines_processed:,}")
    print(f"  Errors: {errors}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
