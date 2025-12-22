#!/usr/bin/env python
"""Compare GPU and CPU state at a specific move."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from app.ai.gpu_parallel_games import ParallelGameRunner
from app.ai.gpu_canonical_export import export_game_to_canonical_dict
from app.game_engine import GameEngine
from app.training.initial_state import create_initial_state
from app.models import BoardType, MoveType, Position
import logging
logging.getLogger('app.ai.gpu_parallel_games').setLevel(logging.WARNING)


def compare_at_move(seed: int, target_move: int):
    """Compare GPU and CPU state after applying target_move moves."""
    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    # Run GPU step by step, carefully tracking move count
    prev_move_count = -1
    while True:
        move_count = int(runner.state.move_count[0].item())
        if move_count >= target_move:
            break
        if runner.state.game_status[0].item() != 0:
            break
        runner._step_games([{}])

    # Capture GPU state
    gpu_stacks = {}
    for y in range(8):
        for x in range(8):
            owner = int(runner.state.stack_owner[0, y, x].item())
            height = int(runner.state.stack_height[0, y, x].item())
            if height > 0:
                cap = int(runner.state.cap_height[0, y, x].item())
                key = f"{x},{y}"  # Position.to_key format
                gpu_stacks[key] = {'owner': owner, 'height': height, 'cap': cap}

    # Export GPU moves
    game_dict = export_game_to_canonical_dict(runner.state, 0, 'square8', 2)

    # Replay on CPU
    initial_state = create_initial_state(BoardType.SQUARE8, num_players=2)
    state = initial_state

    GPU_BOOKKEEPING = {'skip_capture', 'skip_recovery', 'no_placement_action',
                       'no_movement_action', 'no_line_action', 'no_territory_action', 'process_line'}

    cpu_move_count = 0
    for i, m in enumerate(game_dict['moves']):
        if cpu_move_count >= target_move:
            break

        move_type_str = m['type']

        # Skip bookkeeping
        if move_type_str in GPU_BOOKKEEPING:
            req = GameEngine.get_phase_requirement(state, state.current_player)
            if req:
                synth = GameEngine.synthesize_bookkeeping_move(req, state)
                state = GameEngine.apply_move(state, synth)
            continue

        from_pos = Position(**m['from']) if 'from' in m and m['from'] else None
        to_pos = Position(**m['to']) if 'to' in m and m['to'] else None
        move_type = MoveType(move_type_str)

        # Apply move on CPU
        valid = GameEngine.get_valid_moves(state, state.current_player)
        matched = None
        for v in valid:
            if v.type != move_type:
                continue
            if move_type == MoveType.PLACE_RING:
                if v.to and to_pos and v.to.to_key() == to_pos.to_key():
                    matched = v
                    break
            elif move_type in (MoveType.MOVE_STACK, MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT):
                v_to = v.to.to_key() if v.to else None
                m_to = to_pos.to_key() if to_pos else None
                v_from = v.from_pos.to_key() if v.from_pos else None
                m_from = from_pos.to_key() if from_pos else None
                if v_from == m_from and v_to == m_to:
                    matched = v
                    break
            else:
                matched = v
                break

        if matched:
            state = GameEngine.apply_move(state, matched)
            cpu_move_count += 1
        else:
            print(f"*** Cannot match move {i} ***")
            break

    # Capture CPU state
    cpu_stacks = {}
    for key, stack in state.board.stacks.items():
        if stack.stack_height > 0:
            cpu_stacks[key] = {'owner': stack.controlling_player, 'height': stack.stack_height, 'cap': stack.cap_height}

    # Compare
    print(f"Comparing state after {target_move} moves for seed {seed}")
    print("="*60)
    print(f"GPU move count: {int(runner.state.move_count[0].item())}")
    print(f"CPU moves applied: {cpu_move_count}")
    print()

    all_keys = sorted(set(gpu_stacks.keys()) | set(cpu_stacks.keys()))
    print(f"{'Position':<10} | {'GPU':^20} | {'CPU':^20}")
    print("-"*60)
    for key in all_keys:
        gpu_s = gpu_stacks.get(key)
        cpu_s = cpu_stacks.get(key)
        gpu_str = f"p={gpu_s['owner']},h={gpu_s['height']},c={gpu_s['cap']}" if gpu_s else "empty"
        cpu_str = f"p={cpu_s['owner']},h={cpu_s['height']},c={cpu_s['cap']}" if cpu_s else "empty"
        diff = " ***" if gpu_s != cpu_s else ""
        print(f"{key:<10} | {gpu_str:^20} | {cpu_str:^20}{diff}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=56985)
    parser.add_argument("--move", type=int, default=49)
    args = parser.parse_args()

    compare_at_move(args.seed, args.move)
