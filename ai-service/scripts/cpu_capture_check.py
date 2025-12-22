#!/usr/bin/env python
"""Check what captures CPU sees after move 44."""
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
from app.rules.capture_chain import enumerate_capture_moves_py
import logging
logging.getLogger('app.ai.gpu_parallel_games').setLevel(logging.WARNING)


def trace_cpu_captures(seed: int):
    """Trace CPU state and captures at each step."""
    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    # Run GPU to completion
    for step in range(100):
        if runner.state.game_status[0].item() != 0:
            break
        if int(runner.state.move_count[0].item()) >= 60:
            break
        runner._step_games([{}])

    # Export GPU moves
    game_dict = export_game_to_canonical_dict(runner.state, 0, 'square8', 2)

    # Replay on CPU
    initial_state = create_initial_state(BoardType.SQUARE8, num_players=2)
    state = initial_state

    GPU_BOOKKEEPING = {'skip_capture', 'skip_recovery', 'no_placement_action',
                       'no_movement_action', 'no_line_action', 'no_territory_action', 'process_line'}

    print(f"Seed {seed}: {len(game_dict['moves'])} moves")
    print("="*60)

    for i, m in enumerate(game_dict['moves']):
        if i > 46:
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

        # Show state before and after key moves
        if i in (43, 44, 45):
            print(f"\n--- Before applying move {i}: {move_type_str} ---")
            print(f"Phase: {state.current_phase.value}, Player: {state.current_player}")
            print("Stacks:")
            for key, stack in sorted(state.board.stacks.items()):
                if stack.stack_height > 0:
                    print(f"  {key}: p={stack.controlling_player}, h={stack.stack_height}, cap={stack.cap_height}")
            
            # Check captures for current player
            print(f"\nCaptures for player {state.current_player}:")
            for key, stack in sorted(state.board.stacks.items()):
                if stack.controlling_player == state.current_player and stack.stack_height > 0:
                    caps = enumerate_capture_moves_py(state, state.current_player, stack.position, kind="initial")
                    if caps:
                        print(f"  From {key}:")
                        for c in caps[:3]:
                            print(f"    -> {c.to.to_key()} (target={c.capture_target.to_key() if c.capture_target else None})")

        # Find and apply matching move
        valid = GameEngine.get_valid_moves(state, state.current_player)
        matched = None
        for v in valid:
            if v.type != move_type:
                continue
            if move_type == MoveType.PLACE_RING:
                if v.to and to_pos and v.to.to_key() == to_pos.to_key():
                    matched = v
                    break
            elif move_type in (MoveType.MOVE_STACK, MoveType.OVERTAKING_CAPTURE):
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
            if i in (43, 44):
                print(f"\n--- After applying move {i} ---")
                print(f"Phase: {state.current_phase.value}, Player: {state.current_player}")
                print("Stacks:")
                for key, stack in sorted(state.board.stacks.items()):
                    if stack.stack_height > 0:
                        print(f"  {key}: p={stack.controlling_player}, h={stack.stack_height}, cap={stack.cap_height}")
                
                # Check captures for current player
                print(f"\nCaptures for player {state.current_player}:")
                for key, stack in sorted(state.board.stacks.items()):
                    if stack.controlling_player == state.current_player and stack.stack_height > 0:
                        caps = enumerate_capture_moves_py(state, state.current_player, stack.position, kind="initial")
                        if caps:
                            print(f"  From {key}:")
                            for c in caps[:3]:
                                print(f"    -> {c.to.to_key()} (target={c.capture_target.to_key() if c.capture_target else None})")
                        else:
                            print(f"  From {key}: NO CAPTURES")
        else:
            print(f"\n*** Cannot match move {i}: {move_type_str} ***")
            print(f"GPU: from={from_pos.to_key() if from_pos else None}, to={to_pos.to_key() if to_pos else None}")
            print(f"Phase: {state.current_phase.value}, Player: {state.current_player}")
            break


if __name__ == "__main__":
    trace_cpu_captures(18289)
