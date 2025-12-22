#!/usr/bin/env python
"""Debug script to find exact state divergence between GPU and CPU."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from app.ai.gpu_parallel_games import ParallelGameRunner
from app.ai.gpu_canonical_export import export_game_to_canonical_dict
from app.game_engine import GameEngine
from app.training.initial_state import create_initial_state
from app.models import BoardType, MoveType, Position
from app.game_engine import PhaseRequirementType
import logging
logging.getLogger('app.ai.gpu_parallel_games').setLevel(logging.WARNING)

GPU_BOOKKEEPING_MOVES = {
    'skip_capture', 'skip_recovery', 'no_placement_action',
    'no_movement_action', 'no_line_action', 'no_territory_action', 'process_line',
}


def get_board_state_summary(board) -> dict:
    """Extract stack positions from CPU board state."""
    stacks = {}
    for key, stack in board.stacks.items():
        stacks[key] = {
            'owner': stack.controlling_player,
            'height': stack.stack_height,
            'cap': stack.cap_height,
        }
    return stacks


def get_gpu_state_summary(state, game_idx: int, board_size: int) -> dict:
    """Extract stack positions from GPU state."""
    stacks = {}
    for y in range(board_size):
        for x in range(board_size):
            owner = int(state.stack_owner[game_idx, y, x].item())
            height = int(state.stack_height[game_idx, y, x].item())
            if height > 0:
                cap = int(state.cap_height[game_idx, y, x].item())
                key = f"{x},{y}"  # CPU uses x,y format
                stacks[key] = {'owner': owner, 'height': height, 'cap': cap}
    return stacks


def compare_states(gpu_stacks: dict, cpu_stacks: dict) -> list:
    """Compare GPU and CPU stack states, return differences."""
    diffs = []
    all_keys = set(gpu_stacks.keys()) | set(cpu_stacks.keys())

    for key in sorted(all_keys):
        gpu = gpu_stacks.get(key)
        cpu = cpu_stacks.get(key)

        if gpu and not cpu:
            diffs.append(f"  {key}: GPU has stack {gpu}, CPU has none")
        elif cpu and not gpu:
            diffs.append(f"  {key}: CPU has stack {cpu}, GPU has none")
        elif gpu != cpu:
            diffs.append(f"  {key}: GPU={gpu}, CPU={cpu}")

    return diffs


def find_matching_move(state, engine, move_type_str, from_pos, to_pos):
    """Find a matching valid move on CPU side."""
    valid_moves = engine.get_valid_moves(state, state.current_player)
    m_to = to_pos.to_key() if to_pos else None
    move_type = getattr(MoveType, move_type_str.upper(), None)

    for v in valid_moves:
        if v.type != move_type:
            continue
        v_to = v.to.to_key() if v.to else None

        if move_type in (MoveType.PLACE_RING,):
            if v_to == m_to:
                return v
        elif move_type in (MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT):
            if v_to == m_to:
                return v
        else:
            v_from = v.from_pos.to_key() if v.from_pos else None
            m_from = from_pos.to_key() if from_pos else None
            if v_from == m_from and v_to == m_to:
                return v
    return None


def debug_seed(seed: int):
    """Run detailed debug trace for a specific seed."""
    print(f"\n{'='*60}")
    print(f"DEBUGGING SEED {seed}")
    print(f"{'='*60}\n")

    # Run GPU game
    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    for step in range(100):
        move_count = int(runner.state.move_count[0].item())
        if runner.state.game_status[0].item() != 0 or move_count >= 60:
            break
        runner._step_games([{}])

    # Export GPU moves
    game_dict = export_game_to_canonical_dict(runner.state, 0, 'square8', 2)
    gpu_moves = game_dict['moves']

    print(f"GPU generated {len(gpu_moves)} moves")

    # Print first 40 moves for debugging
    print("\nFirst 40 GPU moves:")
    for i, m in enumerate(gpu_moves[:40]):
        mt = m['type']
        f = m.get('from', {})
        t = m.get('to', {})
        f_str = f"{f.get('x')},{f.get('y')}" if f else "None"
        t_str = f"{t.get('x')},{t.get('y')}" if t else "None"
        bk = " [BOOKKEEPING]" if mt in GPU_BOOKKEEPING_MOVES else ""
        print(f"  {i}: {mt} from={f_str} to={t_str} phase={m.get('phase')} player={m.get('player')}{bk}")

    # Replay on CPU with state comparison
    cpu_state = create_initial_state(BoardType.SQUARE8, num_players=2)
    engine = GameEngine()

    first_divergence = None
    moves_applied = 0

    for i, m in enumerate(gpu_moves):
        move_type_str = m['type']
        gpu_phase = m.get('phase', 'ring_placement')
        gpu_player = m.get('player', 1)

        # Skip bookkeeping moves
        if move_type_str in GPU_BOOKKEEPING_MOVES:
            continue

        from_pos = Position(**m['from']) if 'from' in m and m['from'] else None
        to_pos = Position(**m['to']) if 'to' in m and m['to'] else None

        # Try to find matching CPU move
        matching_move = find_matching_move(cpu_state, engine, move_type_str, from_pos, to_pos)

        move_desc = f"{move_type_str} from={from_pos.to_key() if from_pos else None} to={to_pos.to_key() if to_pos else None}"

        if matching_move is None:
            cpu_valid = engine.get_valid_moves(cpu_state, cpu_state.current_player)

            # Loop to handle multiple bookkeeping moves in sequence
            for _ in range(10):  # Safety limit
                if len(cpu_valid) == 0:
                    req = engine.get_phase_requirement(cpu_state, cpu_state.current_player)
                    if req:
                        synth = engine.synthesize_bookkeeping_move(req, cpu_state)
                        if synth:
                            print(f"  [CPU needs bookkeeping: {synth.type.name} for {req.type.name}]")
                            cpu_state = engine.apply_move(cpu_state, synth)
                            cpu_valid = engine.get_valid_moves(cpu_state, cpu_state.current_player)
                            # Try again to find matching move
                            matching_move = find_matching_move(cpu_state, engine, move_type_str, from_pos, to_pos)
                            if matching_move:
                                break
                            continue
                break

            if matching_move:
                cpu_state = engine.apply_move(cpu_state, matching_move)
                continue

            print(f"\nMove {i}: {move_desc} (player={gpu_player}, phase={gpu_phase})")
            print(f"  CPU phase: {cpu_state.current_phase}, player: {cpu_state.current_player}")
            print(f"  CPU board state:")
            cpu_stacks = get_board_state_summary(cpu_state.board)
            for pos, stack in sorted(cpu_stacks.items()):
                print(f"    {pos}: {stack}")
            print(f"  NO MATCHING CPU MOVE FOUND")
            print(f"  CPU valid moves ({len(cpu_valid)}):")
            for v in cpu_valid[:10]:
                v_from = v.from_pos.to_key() if v.from_pos else None
                v_to = v.to.to_key() if v.to else None
                print(f"    {v.type.name}: from={v_from} to={v_to}")
            if len(cpu_valid) > 10:
                print(f"    ... and {len(cpu_valid)-10} more")

            if first_divergence is None:
                first_divergence = i
            break
        else:
            # Apply move to CPU state
            cpu_state = engine.apply_move(cpu_state, matching_move)
            moves_applied += 1

    print(f"\nApplied {moves_applied} moves to CPU successfully")
    if first_divergence is not None:
        print(f"\n{'='*60}")
        print(f"FIRST DIVERGENCE AT MOVE {first_divergence}")
        print(f"{'='*60}")
    else:
        print(f"\nNo divergence found - all moves matched!")

    return first_divergence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, help='Seed to debug')
    args = parser.parse_args()

    debug_seed(args.seed)


if __name__ == '__main__':
    main()
