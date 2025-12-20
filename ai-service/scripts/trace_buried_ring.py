#!/usr/bin/env python
"""Trace P1's buried ring from creation to recovery."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from app.ai.gpu_parallel_games import ParallelGameRunner
import logging
logging.getLogger('app.ai.gpu_parallel_games').setLevel(logging.WARNING)


def print_buried_state(runner):
    """Print buried ring tracking state."""
    buried = [int(runner.state.buried_rings[0, p].item()) for p in range(3)]
    print(f"  Buried counts: {buried}")

    for p in [1, 2]:
        positions = []
        for y in range(8):
            for x in range(8):
                if runner.state.buried_at[0, p, y, x]:
                    positions.append(f"({x},{y})")
        if positions:
            print(f"  Buried_at P{p}: {positions}")
        else:
            print(f"  Buried_at P{p}: []")


def main():
    seed = 42
    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    prev_buried_1 = 0

    step = 0
    while step < 200:
        mc = int(runner.state.move_count[0].item())
        current_buried_1 = int(runner.state.buried_rings[0, 1].item())

        # Track when P1's buried ring changes
        if current_buried_1 != prev_buried_1:
            print(f"\n=== P1 buried rings changed at move_count={mc} ===")
            print(f"  P1 buried: {prev_buried_1} → {current_buried_1}")
            print_buried_state(runner)

            # Show last move
            if mc > 0:
                last = mc - 1
                mt = int(runner.state.move_history[0, last, 0].item())
                player = int(runner.state.move_history[0, last, 1].item())
                from_y = int(runner.state.move_history[0, last, 2].item())
                from_x = int(runner.state.move_history[0, last, 3].item())
                to_y = int(runner.state.move_history[0, last, 4].item())
                to_x = int(runner.state.move_history[0, last, 5].item())
                print(f"  Last move #{last}: type={mt} player={player} from=({from_x},{from_y}) to=({to_x},{to_y})")

            # Show stacks
            print("  Stacks:")
            for y in range(8):
                for x in range(8):
                    owner = int(runner.state.stack_owner[0, y, x].item())
                    height = int(runner.state.stack_height[0, y, x].item())
                    if owner > 0 and height > 0:
                        print(f"    ({x},{y}): P{owner}h{height}")

            prev_buried_1 = current_buried_1

        if mc >= 50:
            break

        runner._step_games([{}])
        step += 1

    print("\nDone.")


if __name__ == '__main__':
    main()
