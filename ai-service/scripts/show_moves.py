#!/usr/bin/env python
"""Show exported GPU moves for a seed."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from app.ai.gpu_parallel_games import ParallelGameRunner
from app.ai.gpu_canonical_export import export_game_to_canonical_dict
import logging
logging.getLogger('app.ai.gpu_parallel_games').setLevel(logging.WARNING)


def show_moves(seed: int, start: int = 0, count: int = 20):
    """Show exported moves from a GPU game."""
    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    # Run GPU to completion
    for step in range(100):
        game_status = runner.state.game_status[0].item()
        if game_status != 0:
            break
        if int(runner.state.move_count[0].item()) >= 60:
            break
        runner._step_games([{}])

    # Export moves
    game_dict = export_game_to_canonical_dict(runner.state, 0, 'square8', 2)

    print(f"Total moves: {len(game_dict['moves'])}")
    print(f"\nMoves {start} to {start + count - 1}:")
    for i in range(start, min(start + count, len(game_dict['moves']))):
        m = game_dict['moves'][i]
        from_pos = m.get('from')
        to_pos = m.get('to')
        from_str = f"({from_pos['y']},{from_pos['x']})" if from_pos else "None"
        to_str = f"({to_pos['y']},{to_pos['x']})" if to_pos else "None"
        print(f"  [{i:2d}] {m['type']:30s} from={from_str:10s} to={to_str:10s} phase={m.get('phase', '?'):20s} player={m.get('player', '?')}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=18289)
    parser.add_argument("--start", type=int, default=40)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    show_moves(args.seed, args.start, args.count)
