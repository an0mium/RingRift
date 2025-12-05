#!/usr/bin/env python
"""
Export training samples from existing GameReplayDB replays.

This script walks one or more GameReplayDB SQLite files and converts completed
games into a neural-net training dataset in the same NPZ format used by
app.training.generate_data:

    - features: (N, C, H, W) float32
    - globals:  (N, G)       float32
    - values:   (N,)         float32   (from final winner, per-state perspective)
    - policy_indices: (N,)   object    → np.ndarray[int32] of indices per sample
    - policy_values:  (N,)   object    → np.ndarray[float32] of probs per sample

Each sample corresponds to a (state_before_move, move_taken) pair drawn from
recorded games, with an outcome label derived from the final winner.

Usage examples (from ai-service/):

    # Basic: export square8 2p samples from a single DB
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \\
      python scripts/export_replay_dataset.py \\
        --db data/games/selfplay_square8_2p.db \\
        --board-type square8 \\
        --num-players 2 \\
        --output data/training/from_replays.square8_2p.npz

    # Limit to first 50 games and sample every 2nd move
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \\
      python scripts/export_replay_dataset.py \\
        --db data/games/selfplay_square8_2p.db \\
        --board-type square8 \\
        --num-players 2 \\
        --max-games 50 \\
        --sample-every 2 \\
        --output data/training/from_replays.square8_2p.npz
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.db import GameReplayDB
from app.models import AIConfig, BoardType, GameState, Move
from app.ai.neural_net import NeuralNetAI, INVALID_MOVE_INDEX


BOARD_TYPE_MAP: Dict[str, BoardType] = {
    "square8": BoardType.SQUARE8,
    "square19": BoardType.SQUARE19,
    "hexagonal": BoardType.HEXAGONAL,
}


def build_encoder(board_type: BoardType) -> NeuralNetAI:
    """
    Construct a NeuralNetAI instance for feature and policy encoding.

    This uses a lightweight AIConfig and treats player_number=1 purely as a
    placeholder; we never call select_move(), only the encoding helpers.
    """
    # Prefer CPU by default to avoid accidental MPS/OMP issues; callers can
    # override via env (e.g. RINGRIFT_FORCE_CPU=0) if they want GPU.
    os.environ.setdefault("RINGRIFT_FORCE_CPU", "1")

    config = AIConfig(
        difficulty=5,
        think_time=0,
        randomness=0.0,
        rngSeed=None,
        heuristic_profile_id=None,
        nn_model_id=None,
        heuristic_eval_mode=None,
        use_neural_net=True,
    )
    encoder = NeuralNetAI(player_number=1, config=config)
    # Ensure the encoder's board_size hint is consistent with the dataset.
    encoder.board_size = {
        BoardType.SQUARE8: 8,
        BoardType.SQUARE19: 19,
        BoardType.HEXAGONAL: 21,
    }.get(board_type, 8)
    return encoder


def encode_state_with_history(
    encoder: NeuralNetAI,
    state: GameState,
    history_frames: List[np.ndarray],
    history_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode a GameState + history into (stacked_features, globals_vec).

    This mirrors the stacking logic in NeuralNetAI.evaluate_batch /
    encode_state_for_model: current features followed by up to history_length
    previous feature frames, newest-first, padded with zeros as needed.
    """
    # Use the internal feature extractor; this is stable tooling code.
    features, globals_vec = encoder._extract_features(state)  # type: ignore[attr-defined]

    hist = history_frames[::-1][:history_length]
    while len(hist) < history_length:
        hist.append(np.zeros_like(features))

    stacked = np.concatenate([features] + hist, axis=0)
    return stacked.astype(np.float32), globals_vec.astype(np.float32)


def value_from_final_winner(final_state: GameState, perspective: int) -> float:
    """
    Map final winner to a scalar value from the perspective of `perspective`.
    """
    winner = getattr(final_state, "winner", None)
    if winner is None:
        return 0.0
    if winner == perspective:
        return 1.0
    return -1.0


def export_replay_dataset(
    db_path: str,
    board_type: BoardType,
    num_players: int,
    output_path: str,
    *,
    history_length: int = 3,
    sample_every: int = 1,
    max_games: Optional[int] = None,
) -> None:
    """
    Export training samples from a single GameReplayDB into an NPZ dataset.
    """
    db = GameReplayDB(db_path)
    encoder = build_encoder(board_type)

    features_list: List[np.ndarray] = []
    globals_list: List[np.ndarray] = []
    values_list: List[float] = []
    policy_indices_list: List[np.ndarray] = []
    policy_values_list: List[np.ndarray] = []

    games_iter = db.iterate_games(board_type=board_type, num_players=num_players)

    games_processed = 0
    for meta, initial_state, moves in games_iter:
        game_id = meta.get("game_id")
        total_moves = int(meta.get("total_moves", len(moves)))
        if total_moves <= 0 or not moves:
            continue

        # Compute final state once for value targets.
        final_state = db.get_state_at_move(game_id, total_moves - 1)
        if final_state is None:
            continue

        history_frames: List[np.ndarray] = []

        for move_index, move in enumerate(moves):
            if sample_every > 1 and (move_index % sample_every) != 0:
                continue

            # State BEFORE this move: initial_state for move 0, otherwise
            # state after the previous move.
            if move_index == 0:
                state_before = initial_state
            else:
                state_before = db.get_state_at_move(game_id, move_index - 1)
                if state_before is None:
                    break

            # Encode features + globals with history.
            stacked, globals_vec = encode_state_with_history(
                encoder,
                state_before,
                history_frames,
                history_length=history_length,
            )

            # Update history with the base features for this state.
            base_features, _ = encoder._extract_features(state_before)  # type: ignore[attr-defined]
            history_frames.append(base_features)
            if len(history_frames) > history_length + 1:
                history_frames.pop(0)

            # Encode the action taken at this state.
            idx = encoder.encode_move(move, state_before.board)
            if idx == INVALID_MOVE_INDEX:
                continue

            # Value from the perspective of the player to move at this state.
            value = value_from_final_winner(final_state, state_before.current_player)

            features_list.append(stacked)
            globals_list.append(globals_vec)
            values_list.append(float(value))
            policy_indices_list.append(np.array([idx], dtype=np.int32))
            policy_values_list.append(np.array([1.0], dtype=np.float32))

        games_processed += 1
        if max_games is not None and games_processed >= max_games:
            break

    if not features_list:
        print(f"No samples generated from {db_path} (board={board_type}, players={num_players}).")
        return

    # Stack into arrays; policies remain sparse per-sample arrays.
    features_arr = np.stack(features_list, axis=0).astype(np.float32)
    globals_arr = np.stack(globals_list, axis=0).astype(np.float32)
    values_arr = np.array(values_list, dtype=np.float32)
    policy_indices_arr = np.array(policy_indices_list, dtype=object)
    policy_values_arr = np.array(policy_values_list, dtype=object)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Append to existing dataset if present, mirroring generate_data.py.
    if os.path.exists(output_path):
        try:
            with np.load(output_path, allow_pickle=True) as data:
                if "features" in data:
                    existing_features = data["features"]
                    existing_globals = data["globals"]
                    existing_values = data["values"]
                    existing_policy_indices = data["policy_indices"]
                    existing_policy_values = data["policy_values"]

                    features_arr = np.concatenate(
                        [existing_features, features_arr],
                        axis=0,
                    )
                    globals_arr = np.concatenate(
                        [existing_globals, globals_arr],
                        axis=0,
                    )
                    values_arr = np.concatenate(
                        [existing_values, values_arr],
                        axis=0,
                    )
                    policy_indices_arr = np.concatenate(
                        [existing_policy_indices, policy_indices_arr],
                        axis=0,
                    )
                    policy_values_arr = np.concatenate(
                        [existing_policy_values, policy_values_arr],
                        axis=0,
                    )
                    print(
                        f"Appended to existing dataset at {output_path}; "
                        f"new total samples: {values_arr.shape[0]}"
                    )
        except Exception as exc:
            print(f"Warning: failed to append to existing {output_path}: {exc}")

    np.savez_compressed(
        output_path,
        features=features_arr,
        globals=globals_arr,
        values=values_arr,
        policy_indices=policy_indices_arr,
        policy_values=policy_values_arr,
    )

    print(
        f"Exported {features_arr.shape[0]} samples "
        f"from {games_processed} games into {output_path}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export NN training samples from existing GameReplayDB replays.",
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to a GameReplayDB SQLite file (e.g. data/games/selfplay_square8_2p.db).",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        choices=["square8", "square19", "hexagonal"],
        required=True,
        help="Board type to filter games by.",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        choices=[2, 3, 4],
        required=True,
        help="Number of players to filter games by.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output .npz dataset.",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=3,
        help="Number of past feature frames to stack (default: 3).",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Use every Nth move as a training sample (default: 1 = every move).",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Optional cap on number of games to process (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    board_type = BOARD_TYPE_MAP[args.board_type]
    if args.history_length < 0:
        raise ValueError("--history-length must be >= 0")
    if args.sample_every < 1:
        raise ValueError("--sample-every must be >= 1")

    export_replay_dataset(
        db_path=args.db,
        board_type=board_type,
        num_players=args.num_players,
        output_path=args.output,
        history_length=args.history_length,
        sample_every=args.sample_every,
        max_games=args.max_games,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
