#!/usr/bin/env python
"""Debug chain capture discrepancy at move 50-51."""
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


def print_stacks_around(state, center_x, center_y, label):
    """Print stack info around a position."""
    print(f"\n{label}:")
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            x, y = center_x + dx, center_y + dy
            if 0 <= x < 8 and 0 <= y < 8:
                pos = Position(x=x, y=y)
                stack = BoardManager.get_stack(pos, state.board)
                if stack and stack.stack_height > 0:
                    print(f"  ({x},{y}): P{stack.controlling_player}h{stack.stack_height}")


def main():
    seed = 42

    # Run GPU game to move 51
    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    target_move = 51
    for step in range(100):
        mc = runner.state.move_count[0].item()
        if mc >= target_move:
            break
        runner._step_games([{}])

    game_dict = export_game_to_canonical_dict(runner.state, 0, 'square8', 2)

    # Show GPU state after move 50 (before move 51)
    print("GPU state after move 50:")
    print(f"  Stack at (4,1): owner={int(runner.state.stack_owner[0, 1, 4].item())}, h={int(runner.state.stack_height[0, 1, 4].item())}")
    print(f"  Stack at (4,2): owner={int(runner.state.stack_owner[0, 2, 4].item())}, h={int(runner.state.stack_height[0, 2, 4].item())}")
    print(f"  Stack at (4,3): owner={int(runner.state.stack_owner[0, 3, 4].item())}, h={int(runner.state.stack_height[0, 3, 4].item())}")
    print(f"  Marker at (4,1): {int(runner.state.marker_owner[0, 1, 4].item())}")
    print(f"  GPU phase: {int(runner.state.current_phase[0].item())}")
    print(f"  GPU player: {int(runner.state.current_player[0].item())}")

    # Now replay CPU up to move 50
    print("\n\nReplaying CPU through move 50...")
    initial_state = create_initial_state(BoardType.SQUARE8, num_players=2)
    state = initial_state

    GPU_BOOKKEEPING_MOVES = {
        'skip_capture', 'skip_recovery',
        'no_placement_action', 'no_movement_action', 'no_line_action', 'no_territory_action',
        'process_line', 'process_territory_region', 'choose_line_option', 'choose_territory_option',
    }

    # Track which "real" move we're on
    real_moves = [m for m in game_dict['moves'] if m['type'] not in GPU_BOOKKEEPING_MOVES]

    for i, m in enumerate(game_dict['moves'][:51]):  # Through move 50
        move_type_str = m['type']
        gpu_phase = m.get('phase', 'ring_placement')
        gpu_player = m.get('player', 1)

        if move_type_str in GPU_BOOKKEEPING_MOVES:
            # Advance CPU through bookkeeping phases
            for _ in range(10):
                req = GameEngine.get_phase_requirement(state, state.current_player)
                if req:
                    synth = GameEngine.synthesize_bookkeeping_move(req, state)
                    state = GameEngine.apply_move(state, synth)
                else:
                    break
            continue

        move_type = MoveType(move_type_str)
        from_pos = Position(**m['from']) if 'from' in m and m['from'] else None
        to_pos = Position(**m['to']) if 'to' in m and m['to'] else None

        # Advance CPU phases first
        for _ in range(10):
            if state.current_phase.value == gpu_phase and state.current_player == gpu_player:
                break
            req = GameEngine.get_phase_requirement(state, state.current_player)
            if req:
                synth = GameEngine.synthesize_bookkeeping_move(req, state)
                state = GameEngine.apply_move(state, synth)
            else:
                break

        valid = GameEngine.get_valid_moves(state, state.current_player)
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
            # Show state before and after move 50
            if i == 50:
                print(f"\nBefore CPU move 50:")
                print(f"  Stack at (4,1): {BoardManager.get_stack(Position(x=4, y=1), state.board)}")
                print(f"  Stack at (4,2): {BoardManager.get_stack(Position(x=4, y=2), state.board)}")
                print(f"  Stack at (4,3): {BoardManager.get_stack(Position(x=4, y=3), state.board)}")
                print(f"  CPU phase: {state.current_phase.value}")
                print(f"  CPU player: {state.current_player}")

            state = GameEngine.apply_move(state, matched)

            if i == 50:
                print(f"\nAfter CPU move 50:")
                print(f"  Stack at (4,1): {BoardManager.get_stack(Position(x=4, y=1), state.board)}")
                print(f"  Stack at (4,2): {BoardManager.get_stack(Position(x=4, y=2), state.board)}")
                print(f"  Stack at (4,3): {BoardManager.get_stack(Position(x=4, y=3), state.board)}")
                print(f"  CPU phase: {state.current_phase.value}")
                print(f"  CPU player: {state.current_player}")

                # Check for chain capture continuations
                from app.rules.capture_chain import get_chain_capture_continuation_info_py
                cont_info = get_chain_capture_continuation_info_py(
                    state,
                    state.current_player if hasattr(matched, 'player') else 2,
                    matched.to
                )
                print(f"\n  Chain capture continuation info:")
                print(f"    must_continue: {cont_info.must_continue}")
                print(f"    available_continuations: {len(cont_info.available_continuations)}")
                for c in cont_info.available_continuations:
                    print(f"      {c.type.value} from={c.from_pos} to={c.to} target={c.capture_target}")
        else:
            req = GameEngine.get_phase_requirement(state, state.current_player)
            if req:
                synth = GameEngine.synthesize_bookkeeping_move(req, state)
                state = GameEngine.apply_move(state, synth)
            else:
                print(f"Move {i}: FAIL to match {move_type_str}")


if __name__ == '__main__':
    main()
