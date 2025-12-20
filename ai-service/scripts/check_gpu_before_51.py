#!/usr/bin/env python
"""Check GPU state right before move 51."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from app.ai.gpu_parallel_games import ParallelGameRunner
import logging
logging.getLogger('app.ai.gpu_parallel_games').setLevel(logging.WARNING)


def print_gpu_board(runner, label):
    """Print GPU board state."""
    print(f"\n{label}:")
    print(f"  Move count: {int(runner.state.move_count[0].item())}")
    print(f"  Current player: {int(runner.state.current_player[0].item())}")
    print(f"  Current phase: {int(runner.state.current_phase[0].item())}")
    print(f"  In capture chain: {bool(runner.state.in_capture_chain[0].item())}")
    print(f"  Capture chain depth: {int(runner.state.capture_chain_depth[0].item())}")

    print("\n  Board (non-empty stacks):")
    for y in range(8):
        for x in range(8):
            owner = int(runner.state.stack_owner[0, y, x].item())
            height = int(runner.state.stack_height[0, y, x].item())
            if owner > 0 and height > 0:
                marker = int(runner.state.marker_owner[0, y, x].item())
                marker_str = f" M{marker}" if marker > 0 else ""
                print(f"    ({x},{y}): P{owner}h{height}{marker_str}")


def main():
    seed = 42
    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    # Run until just before move 51
    print("Running GPU game step by step...")
    step = 0
    while step < 200:
        mc = int(runner.state.move_count[0].item())

        # Print state before each move near the problem area
        if mc >= 49 and mc <= 52:
            print_gpu_board(runner, f"Before GPU step (move_count={mc})")

            # Show the last recorded move
            if mc > 0:
                last_idx = mc - 1
                mt = int(runner.state.move_history[0, last_idx, 0].item())
                player = int(runner.state.move_history[0, last_idx, 1].item())
                from_y = int(runner.state.move_history[0, last_idx, 2].item())
                from_x = int(runner.state.move_history[0, last_idx, 3].item())
                to_y = int(runner.state.move_history[0, last_idx, 4].item())
                to_x = int(runner.state.move_history[0, last_idx, 5].item())
                phase = int(runner.state.move_history[0, last_idx, 6].item())
                cap_y = int(runner.state.move_history[0, last_idx, 7].item())
                cap_x = int(runner.state.move_history[0, last_idx, 8].item())
                print(f"\n  Last move #{last_idx}: type={mt} player={player} from=({from_x},{from_y}) to=({to_x},{to_y}) phase={phase} cap=({cap_x},{cap_y})")

        if mc >= 52:
            break

        runner._step_games([{}])
        step += 1

    print("\n\nFinal state reached.")


if __name__ == '__main__':
    main()
