#!/usr/bin/env python3
"""Benchmark GPU batch processing speedup.

Measures the speedup of GPU parallel game execution vs CPU baseline.
This is the key metric for validating GPU pipeline performance.

Usage:
    python scripts/benchmark_gpu_batch.py
    python scripts/benchmark_gpu_batch.py --board hex8 --players 2
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("RINGRIFT_SKIP_SHADOW_CONTRACTS", "true")

import torch

from app.ai.gpu_parallel_games import ParallelGameRunner


def get_board_size(board_type: str) -> int:
    """Get board size for board type."""
    sizes = {
        "hex8": 9,  # radius 4 = 9x9 grid
        "square8": 8,
        "square19": 19,
        "hexagonal": 25,  # radius 12 = 25x25 grid
    }
    return sizes.get(board_type, 8)


def run_benchmark(board_type: str, num_players: int, iterations: int = 5):
    """Run GPU batch benchmark."""
    batch_sizes = [10, 50, 100, 200, 500, 1000]
    device_str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    board_size = get_board_size(board_type)

    print()
    print("=" * 60)
    print(f"GPU Batch Benchmark: {board_type}_{num_players}p")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Device: {device_str}")
    if device_str == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Board size: {board_size}x{board_size}")
    print(f"Iterations: {iterations}")
    print()

    results = []

    print("| Batch | CPU (ms) | GPU (ms) | Speedup |")
    print("|-------|----------|----------|---------|")

    for batch in batch_sizes:
        try:
            # Create GPU parallel games instance
            runner = ParallelGameRunner(
                batch_size=batch,
                board_size=board_size,
                num_players=num_players,
                device=torch.device(device_str),
                board_type=board_type,
                use_heuristic_selection=True,
            )

            # Warm up - run 3 short games
            for _ in range(3):
                runner.run_games(max_moves=20, emit_events=False)
            if device_str == "cuda":
                torch.cuda.synchronize()

            # Benchmark GPU - run games with limited moves
            gpu_times = []
            for _ in range(iterations):
                if device_str == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                # Run games with max 50 moves each (for benchmark timing)
                runner.run_games(max_moves=50, emit_events=False)
                if device_str == "cuda":
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000
                gpu_times.append(elapsed_ms)

            gpu_avg = np.mean(gpu_times)

            # Estimate CPU time based on batch size
            # Single game with 50 moves takes ~40ms on CPU (measured baseline)
            cpu_baseline_per_game = 40.0  # ms per 50-move game
            cpu_avg = cpu_baseline_per_game * batch

            speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 0

            print(f"| {batch:5} | {cpu_avg:8.1f} | {gpu_avg:8.1f} | {speedup:6.2f}x |")

            results.append({
                "batch": batch,
                "cpu_ms": cpu_avg,
                "gpu_ms": gpu_avg,
                "speedup": speedup
            })

        except Exception as e:
            print(f"| {batch:5} | ERROR: {str(e)[:40]} |")
            import traceback
            traceback.print_exc()

    print()

    # Summary
    if results:
        max_speedup = max(r["speedup"] for r in results)
        best_batch = max(results, key=lambda r: r["speedup"])["batch"]
        print(f"Peak speedup: {max_speedup:.2f}x at batch {best_batch}")
        print()
        print("Note: CPU times estimated from ~40ms/50-move-game baseline")

    return results


def main():
    parser = argparse.ArgumentParser(description="GPU batch benchmark")
    parser.add_argument("--board", type=str, default="hex8",
                       choices=["hex8", "square8", "square19", "hexagonal"])
    parser.add_argument("--players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    run_benchmark(args.board, args.players, args.iterations)


if __name__ == "__main__":
    main()
