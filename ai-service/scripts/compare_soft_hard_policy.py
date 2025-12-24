#!/usr/bin/env python3
"""Compare training with soft vs hard policy targets.

This script runs two training experiments:
1. Hard policy: One-hot targets (traditional)
2. Soft policy: Visit distribution targets (from GPU MCTS)

Usage:
    python scripts/compare_soft_hard_policy.py \
        --soft-data data/training/gpu_mcts_hex8_2p.npz \
        --hard-data data/training/hex8_2p.npz \
        --board-type hex8 --num-players 2 \
        --output-dir results/soft_vs_hard
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def convert_soft_to_hard(npz_path: str, output_path: str) -> None:
    """Convert soft policy targets to hard (one-hot) targets.

    Takes the highest probability action and creates a one-hot target.
    """
    data = np.load(npz_path, allow_pickle=True)

    # Get policy data
    policy_indices = data["policy_indices"]
    policy_values = data["policy_values"]

    # Convert to hard targets: keep only the max probability action
    hard_indices = []
    hard_values = []

    for i in range(len(policy_indices)):
        indices = policy_indices[i]
        values = policy_values[i]

        # Find non-zero entries
        mask = values > 0
        if not np.any(mask):
            # No valid actions, keep first
            hard_indices.append(indices[:1])
            hard_values.append(np.array([1.0], dtype=np.float32))
        else:
            # Keep only the max
            valid_indices = indices[mask]
            valid_values = values[mask]
            max_idx = np.argmax(valid_values)
            hard_indices.append(np.array([valid_indices[max_idx]], dtype=np.int64))
            hard_values.append(np.array([1.0], dtype=np.float32))

    # Pad to same shape
    max_len = max(len(x) for x in hard_indices)
    padded_indices = np.zeros((len(hard_indices), max_len), dtype=np.int64)
    padded_values = np.zeros((len(hard_values), max_len), dtype=np.float32)

    for i, (idx, val) in enumerate(zip(hard_indices, hard_values)):
        padded_indices[i, :len(idx)] = idx
        padded_values[i, :len(val)] = val

    # Save with same metadata
    save_dict = {k: data[k] for k in data.files if k not in ("policy_indices", "policy_values")}
    save_dict["policy_indices"] = padded_indices
    save_dict["policy_values"] = padded_values
    save_dict["source"] = np.asarray("hard_policy_converted")

    np.savez_compressed(output_path, **save_dict)
    logger.info(f"Converted {len(hard_indices)} samples to hard targets -> {output_path}")


def train_model(
    data_path: str,
    model_path: str,
    board_type: str,
    num_players: int,
    epochs: int = 20,
    batch_size: int = 256,
) -> dict:
    """Train a model and return metrics."""
    from app.models import BoardType
    from app.training.config import TrainConfig
    from app.training.train import train_model as do_train

    board_type_enum = getattr(BoardType, board_type.upper())

    config = TrainConfig(
        board_type=board_type_enum,
        batch_size=batch_size,
        learning_rate=1e-3,
        epochs_per_iter=epochs,
    )

    metrics = do_train(
        config=config,
        data_path=data_path,
        save_path=model_path,
        early_stopping_patience=5,
        num_players=num_players,
        model_version="v3",  # Match V3 encoder data
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compare soft vs hard policy training")
    parser.add_argument("--soft-data", required=True, help="Path to GPU MCTS NPZ (soft targets)")
    parser.add_argument("--hard-data", help="Path to hard target NPZ (or will convert from soft)")
    parser.add_argument("--board-type", default="hex8", help="Board type")
    parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    parser.add_argument("--output-dir", default="results/soft_vs_hard", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare hard data if not provided
    if args.hard_data:
        hard_data_path = args.hard_data
    else:
        hard_data_path = str(output_dir / "hard_targets.npz")
        logger.info("Converting soft targets to hard targets...")
        convert_soft_to_hard(args.soft_data, hard_data_path)

    results = {}

    # Train with soft targets
    logger.info("\n" + "=" * 50)
    logger.info("Training with SOFT policy targets (visit distributions)")
    logger.info("=" * 50)

    soft_model_path = str(output_dir / "model_soft.pt")
    soft_metrics = train_model(
        data_path=args.soft_data,
        model_path=soft_model_path,
        board_type=args.board_type,
        num_players=args.num_players,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    results["soft"] = soft_metrics

    # Train with hard targets
    logger.info("\n" + "=" * 50)
    logger.info("Training with HARD policy targets (one-hot)")
    logger.info("=" * 50)

    hard_model_path = str(output_dir / "model_hard.pt")
    hard_metrics = train_model(
        data_path=hard_data_path,
        model_path=hard_model_path,
        board_type=args.board_type,
        num_players=args.num_players,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    results["hard"] = hard_metrics

    # Compare results
    logger.info("\n" + "=" * 50)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 50)

    for key in ["policy_accuracy", "value_loss", "policy_loss"]:
        soft_val = soft_metrics.get(key, "N/A")
        hard_val = hard_metrics.get(key, "N/A")
        logger.info(f"{key}:")
        logger.info(f"  Soft: {soft_val}")
        logger.info(f"  Hard: {hard_val}")

    # Save results
    results_path = output_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
