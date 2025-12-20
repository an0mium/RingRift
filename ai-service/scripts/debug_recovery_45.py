#!/usr/bin/env python
"""Debug recovery at move 45."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from app.ai.gpu_parallel_games import ParallelGameRunner
import logging
logging.getLogger('app.ai.gpu_parallel_games').setLevel(logging.WARNING)


def print_gpu_state(runner, label):
    print(f"\n{label}:")
    print(f"  Move count: {int(runner.state.move_count[0].item())}")
    print(f"  Current player: {int(runner.state.current_player[0].item())}")
    print(f"  Buried rings: P1={int(runner.state.buried_rings[0,1].item())}, P2={int(runner.state.buried_rings[0,2].item())}")

    # Show buried_at positions
    for p in [1, 2]:
        buried_positions = []
        for y in range(8):
            for x in range(8):
                if runner.state.buried_at[0, p, y, x]:
                    buried_positions.append(f"({x},{y})")
        if buried_positions:
            print(f"  Buried_at P{p}: {buried_positions}")

    print("  Stacks:")
    for y in range(8):
        for x in range(8):
            owner = int(runner.state.stack_owner[0, y, x].item())
            height = int(runner.state.stack_height[0, y, x].item())
            if owner > 0 and height > 0:
                print(f"    ({x},{y}): P{owner}h{height}")


def main():
    seed = 42
    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    print("Running to move 44...")
    step = 0
    while step < 200:
        mc = int(runner.state.move_count[0].item())
        if mc == 44:
            print_gpu_state(runner, "Before move 44 step")
            runner._step_games([{}])
            print_gpu_state(runner, "After move 44 step")
        elif mc == 45:
            print_gpu_state(runner, "Before move 45 step")
            runner._step_games([{}])
            print_gpu_state(runner, "After move 45 step")
            break
        else:
            runner._step_games([{}])
        step += 1

        if mc >= 50:
            break


if __name__ == '__main__':
    main()
