#!/usr/bin/env python
"""Trace early GPU vs CPU divergence."""
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
from app.board_manager import BoardManager
import logging
logging.getLogger('app.ai.gpu_parallel_games').setLevel(logging.WARNING)


GPU_BOOKKEEPING_MOVES = {
    'skip_capture', 'skip_recovery',
    'no_placement_action', 'no_movement_action', 'no_line_action', 'no_territory_action',
    'process_line', 'process_territory_region', 'choose_line_option', 'choose_territory_option',
}


def get_board_hash(state):
    """Get a simple hash of non-empty stacks."""
    stacks = []
    for y in range(8):
        for x in range(8):
            pos = Position(x=x, y=y)
            stack = BoardManager.get_stack(pos, state.board)
            if stack and stack.stack_height > 0:
                stacks.append(f"({x},{y}):P{stack.controlling_player}h{stack.stack_height}")
    return sorted(stacks)


def get_gpu_board_stacks(runner):
    """Get non-empty stacks from GPU."""
    stacks = []
    for y in range(8):
        for x in range(8):
            owner = int(runner.state.stack_owner[0, y, x].item())
            height = int(runner.state.stack_height[0, y, x].item())
            if owner > 0 and height > 0:
                stacks.append(f"({x},{y}):P{owner}h{height}")
    return sorted(stacks)


def main():
    seed = 42

    # Run GPU game to completion
    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')
    for step in range(100):
        if runner.state.move_count[0].item() >= 60:
            break
        runner._step_games([{}])

    game_dict = export_game_to_canonical_dict(runner.state, 0, 'square8', 2)

    # Now replay GPU step by step to get intermediate states
    print("Replaying GPU step-by-step and comparing to CPU...")
    torch.manual_seed(seed)
    gpu_runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    # Initialize CPU
    cpu_state = create_initial_state(BoardType.SQUARE8, num_players=2)

    gpu_move_idx = 0
    cpu_move_idx = 0

    # Track GPU states at each move_count
    for step in range(100):
        gpu_mc = int(gpu_runner.state.move_count[0].item())
        gpu_stacks = get_gpu_board_stacks(gpu_runner)

        # Apply corresponding CPU moves to match GPU move count
        while cpu_move_idx < len(game_dict['moves']) and cpu_move_idx < gpu_mc:
            m = game_dict['moves'][cpu_move_idx]
            move_type_str = m['type']

            if move_type_str in GPU_BOOKKEEPING_MOVES:
                # Advance CPU through bookkeeping
                for _ in range(10):
                    req = GameEngine.get_phase_requirement(cpu_state, cpu_state.current_player)
                    if req:
                        synth = GameEngine.synthesize_bookkeeping_move(req, cpu_state)
                        cpu_state = GameEngine.apply_move(cpu_state, synth)
                    else:
                        break
                cpu_move_idx += 1
                continue

            move_type = MoveType(move_type_str)
            from_pos = Position(**m['from']) if 'from' in m and m['from'] else None
            to_pos = Position(**m['to']) if 'to' in m and m['to'] else None

            # Advance CPU phases
            for _ in range(10):
                req = GameEngine.get_phase_requirement(cpu_state, cpu_state.current_player)
                if req:
                    synth = GameEngine.synthesize_bookkeeping_move(req, cpu_state)
                    cpu_state = GameEngine.apply_move(cpu_state, synth)
                else:
                    break

            valid = GameEngine.get_valid_moves(cpu_state, cpu_state.current_player)
            matched = None

            for v in valid:
                if v.type != move_type:
                    continue
                v_to = v.to.to_key() if v.to else None
                m_to = to_pos.to_key() if to_pos else None
                if move_type == MoveType.PLACE_RING:
                    if v_to == m_to:
                        matched = v
                        break
                else:
                    v_from = v.from_pos.to_key() if v.from_pos else None
                    m_from = from_pos.to_key() if from_pos else None
                    if v_from == m_from and v_to == m_to:
                        matched = v
                        break

            if matched:
                cpu_state = GameEngine.apply_move(cpu_state, matched)
            cpu_move_idx += 1

        cpu_stacks = get_board_hash(cpu_state)

        if gpu_stacks != cpu_stacks:
            print(f"\n=== DIVERGENCE at GPU move_count={gpu_mc} ===")
            print(f"GPU stacks: {gpu_stacks}")
            print(f"CPU stacks: {cpu_stacks}")
            print(f"GPU move_count: {gpu_mc}")
            print(f"CPU move_idx: {cpu_move_idx}")

            # Show last few moves
            if gpu_mc > 0:
                print("\nLast moves in history:")
                for i in range(max(0, gpu_mc - 3), gpu_mc):
                    if i < len(game_dict['moves']):
                        m = game_dict['moves'][i]
                        print(f"  {i}: {m['type']} from={m.get('from')} to={m.get('to')} player={m.get('player')}")
            break

        if gpu_mc >= 60:
            print(f"Reached move {gpu_mc} with no divergence!")
            break

        # Step GPU
        gpu_runner._step_games([{}])

    print("\nDone.")


if __name__ == '__main__':
    main()
