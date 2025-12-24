#!/usr/bin/env python3
"""Multi-GPU selfplay launcher.

Launches GPU MCTS selfplay on all available GPUs in parallel.
Each GPU runs an independent instance with different game seeds.

Usage:
    # Auto-detect GPUs and run
    python scripts/multi_gpu_selfplay.py --board-type hex8 --num-players 2 --num-games 1000

    # Specify GPUs explicitly
    python scripts/multi_gpu_selfplay.py --gpus 0,1 --board-type hex8 --num-players 2 --num-games 500
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def get_available_gpus() -> list[int]:
    """Get list of available CUDA GPU indices."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def launch_gpu_worker(
    gpu_id: int,
    board_type: str,
    num_players: int,
    num_games: int,
    output_dir: str,
    simulation_budget: int,
) -> subprocess.Popen:
    """Launch a GPU MCTS worker on a specific GPU.

    Returns the Popen process handle.
    """
    output_path = Path(output_dir) / f"gpu{gpu_id}_{board_type}_{num_players}p.npz"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable,
        "-m",
        "app.training.gpu_mcts_selfplay",
        "--board-type",
        board_type,
        "--num-players",
        str(num_players),
        "--num-games",
        str(num_games),
        "--output",
        str(output_path),
        "--device",
        "cuda",
    ]

    if simulation_budget:
        cmd.extend(["--simulation-budget", str(simulation_budget)])

    logger.info(f"Launching worker on GPU {gpu_id} -> {output_path}")
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def merge_npz_files(input_files: list[str], output_path: str) -> None:
    """Merge multiple NPZ files into one."""
    import numpy as np

    all_data = {}
    metadata = {}

    for path in input_files:
        if not Path(path).exists():
            continue

        data = np.load(path)

        for key in data.files:
            arr = data[key]
            if arr.ndim == 0:
                # Scalar metadata
                metadata[key] = arr
            elif key not in all_data:
                all_data[key] = [arr]
            else:
                all_data[key].append(arr)

    if not all_data:
        logger.warning("No data to merge")
        return

    # Concatenate arrays
    merged = {}
    for key, arrays in all_data.items():
        merged[key] = np.concatenate(arrays, axis=0)

    # Add metadata
    merged.update({k: v for k, v in metadata.items()})
    merged["source"] = np.asarray("multi_gpu_merged")

    np.savez_compressed(output_path, **merged)
    logger.info(f"Merged {len(input_files)} files -> {output_path}")
    logger.info(f"  Total samples: {len(merged.get('features', []))}")


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU selfplay launcher")
    parser.add_argument("--board-type", default="hex8", help="Board type")
    parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    parser.add_argument("--num-games", type=int, default=1000, help="Total games (split across GPUs)")
    parser.add_argument("--gpus", help="Comma-separated GPU IDs (default: all)")
    parser.add_argument("--output-dir", default="data/training/gpu_mcts", help="Output directory")
    parser.add_argument("--output", help="Final merged output file")
    parser.add_argument("--simulation-budget", type=int, default=64, help="MCTS simulations")
    parser.add_argument("--no-merge", action="store_true", help="Don't merge outputs")

    args = parser.parse_args()

    # Determine GPUs to use
    if args.gpus:
        gpus = [int(g) for g in args.gpus.split(",")]
    else:
        gpus = get_available_gpus()

    if not gpus:
        logger.error("No GPUs available")
        sys.exit(1)

    logger.info(f"Using {len(gpus)} GPU(s): {gpus}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split games across GPUs
    games_per_gpu = args.num_games // len(gpus)
    remainder = args.num_games % len(gpus)

    # Launch workers
    processes = []
    output_files = []

    for i, gpu_id in enumerate(gpus):
        gpu_games = games_per_gpu + (1 if i < remainder else 0)
        output_path = output_dir / f"gpu{gpu_id}_{args.board_type}_{args.num_players}p.npz"
        output_files.append(str(output_path))

        proc = launch_gpu_worker(
            gpu_id=gpu_id,
            board_type=args.board_type,
            num_players=args.num_players,
            num_games=gpu_games,
            output_dir=str(output_dir),
            simulation_budget=args.simulation_budget,
        )
        processes.append((gpu_id, proc))

    # Wait for all workers
    logger.info("Waiting for workers to complete...")
    start_time = time.time()

    for gpu_id, proc in processes:
        stdout, _ = proc.communicate()
        elapsed = time.time() - start_time
        if proc.returncode == 0:
            logger.info(f"GPU {gpu_id} completed ({elapsed:.1f}s)")
        else:
            logger.error(f"GPU {gpu_id} failed (exit code {proc.returncode})")
            logger.error(stdout.decode() if stdout else "No output")

    total_elapsed = time.time() - start_time
    logger.info(f"All workers completed in {total_elapsed:.1f}s")

    # Merge outputs
    if not args.no_merge:
        final_output = args.output or str(
            output_dir / f"merged_{args.board_type}_{args.num_players}p.npz"
        )
        merge_npz_files(output_files, final_output)


if __name__ == "__main__":
    main()
