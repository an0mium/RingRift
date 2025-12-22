#!/usr/bin/env python
"""Step through GPU moves one at a time to capture state at each point."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from app.ai.gpu_parallel_games import ParallelGameRunner
from app.ai.gpu_move_generation import generate_chain_capture_moves_from_position
import logging
logging.getLogger('app.ai.gpu_parallel_games').setLevel(logging.WARNING)


def trace_gpu_step_by_step(seed: int, target_move: int):
    """Trace GPU state by stepping one move at a time."""
    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    print(f"GPU step trace for seed {seed}, target move {target_move}")
    print("="*60)

    prev_move_count = -1
    step = 0
    
    while True:
        move_count = int(runner.state.move_count[0].item())
        
        # Print state at specific move counts
        if move_count >= target_move - 3 and move_count != prev_move_count:
            phase = int(runner.state.current_phase[0].item())
            player = int(runner.state.current_player[0].item())
            
            print(f"\n--- After move {move_count}, Phase={phase}, Player={player} ---")
            print("Stacks:")
            for y in range(8):
                for x in range(8):
                    owner = int(runner.state.stack_owner[0, y, x].item())
                    height = int(runner.state.stack_height[0, y, x].item())
                    if height > 0:
                        cap = int(runner.state.cap_height[0, y, x].item())
                        # Position format: x,y (to match CPU to_key)
                        print(f"  ({x},{y}): owner={owner}, h={height}, cap={cap}")
            
            # Check captures from key positions
            for y in range(8):
                for x in range(8):
                    owner = int(runner.state.stack_owner[0, y, x].item())
                    height = int(runner.state.stack_height[0, y, x].item())
                    if height > 0 and owner == player:
                        caps = generate_chain_capture_moves_from_position(runner.state, 0, y, x)
                        if caps:
                            print(f"  Captures from ({x},{y}): {[(cx, cy) for cy, cx in caps]}")
            
            prev_move_count = move_count
        
        if move_count >= target_move:
            break
            
        game_status = runner.state.game_status[0].item()
        if game_status != 0:
            print(f"Game ended at move {move_count}")
            break
            
        step += 1
        runner._step_games([{}])
        
        # Safety limit
        if step > 100:
            break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=18289)
    parser.add_argument("--target-move", type=int, default=48)
    args = parser.parse_args()

    trace_gpu_step_by_step(args.seed, args.target_move)
