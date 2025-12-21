#!/usr/bin/env python
"""Debug GPU state at failure point to find divergence."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from app.ai.gpu_parallel_games import ParallelGameRunner
from app.ai.gpu_canonical_export import export_game_to_canonical_dict
import logging
logging.getLogger('app.ai.gpu_parallel_games').setLevel(logging.WARNING)


def debug_gpu_state(seed: int, max_moves: int = 60) -> None:
    """Dump GPU state at each step to find divergence point."""
    print(f"\n{'='*60}")
    print(f"GPU State Debug for seed {seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    for step in range(100):
        game_status = runner.state.game_status[0].item()
        move_count = int(runner.state.move_count[0].item())
        if game_status != 0 or move_count >= max_moves:
            break

        # Get current player and phase
        current_player = int(runner.state.current_player[0].item())
        phase = int(runner.state.current_phase[0].item())

        # Get player stats
        rings_in_hand = [int(runner.state.rings_in_hand[0, p].item()) for p in range(3)]
        buried_rings = [int(runner.state.buried_rings[0, p].item()) for p in range(3)]

        # Check stack ownership (1 = P1, 2 = P2)
        stack_owner = runner.state.stack_owner[0]  # (8, 8)
        p1_stacks = (stack_owner == 1).sum().item()
        p2_stacks = (stack_owner == 2).sum().item()

        # Check marker ownership
        marker_owner = runner.state.marker_owner[0]  # (8, 8)
        p1_markers = (marker_owner == 1).sum().item()
        p2_markers = (marker_owner == 2).sum().item()

        # Check recovery eligibility
        p1_recovery = (p1_stacks == 0) and (p1_markers > 0) and (buried_rings[1] > 0)
        p2_recovery = (p2_stacks == 0) and (p2_markers > 0) and (buried_rings[2] > 0)

        print(f"\n--- Move {move_count} | Phase {phase} | Player {current_player} ---")
        print(f"  P1: rings_in_hand={rings_in_hand[1]}, stacks={p1_stacks}, markers={p1_markers}, buried={buried_rings[1]}, recovery={p1_recovery}")
        print(f"  P2: rings_in_hand={rings_in_hand[2]}, stacks={p2_stacks}, markers={p2_markers}, buried={buried_rings[2]}, recovery={p2_recovery}")

        # Show buried ring positions and stack info at those positions
        buried_at = runner.state.buried_at[0]  # (num_players+1, 8, 8)
        for p in [1, 2]:
            positions = torch.where(buried_at[p])
            if len(positions[0]) > 0:
                pos_list = []
                for i in range(len(positions[0])):
                    y, x = positions[0][i].item(), positions[1][i].item()
                    owner = runner.state.stack_owner[0, y, x].item()
                    height = runner.state.stack_height[0, y, x].item()
                    pos_list.append(f"({y},{x}) owner={owner} h={height}")
                print(f"    P{p} buried_at: {pos_list}")

        # Step the game
        runner._step_games([{}])

    # Final state and export
    print(f"\n{'='*60}")
    print("Final GPU State")
    print(f"{'='*60}")
    game_dict = export_game_to_canonical_dict(runner.state, 0, 'square8', 2)

    print(f"\nTotal moves exported: {len(game_dict['moves'])}")

    # Print last 10 moves
    print("\nLast 10 moves:")
    for i in range(max(0, len(game_dict['moves']) - 10), len(game_dict['moves'])):
        m = game_dict['moves'][i]
        print(f"  {i}: {m['type']} phase={m.get('phase')} player={m.get('player')}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int, help='Seed to debug')
    parser.add_argument('--max-moves', type=int, default=60, help='Max moves to run')
    args = parser.parse_args()

    debug_gpu_state(args.seed, args.max_moves)
