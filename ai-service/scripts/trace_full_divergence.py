#!/usr/bin/env python
"""Trace GPU vs CPU divergence point."""
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


def get_board_summary(state, positions):
    """Get stack info for positions."""
    result = {}
    for x, y in positions:
        pos = Position(x=x, y=y)
        stack = BoardManager.get_stack(pos, state.board)
        if stack and stack.stack_height > 0:
            result[(x, y)] = f"P{stack.controlling_player}h{stack.stack_height}"
        else:
            result[(x, y)] = "."
    return result


def main():
    seed = 42

    # Run GPU game
    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    for step in range(100):
        if runner.state.move_count[0].item() >= 60:
            break
        runner._step_games([{}])

    game_dict = export_game_to_canonical_dict(runner.state, 0, 'square8', 2)

    # Replay CPU, tracking divergence
    print("Tracing GPU→CPU divergence...")
    initial_state = create_initial_state(BoardType.SQUARE8, num_players=2)
    state = initial_state

    positions_of_interest = [(4, 1), (4, 2), (4, 3), (4, 4)]

    for i, m in enumerate(game_dict['moves']):
        move_type_str = m['type']
        gpu_phase = m.get('phase', 'ring_placement')
        gpu_player = m.get('player', 1)

        if move_type_str in GPU_BOOKKEEPING_MOVES:
            # Advance CPU through bookkeeping
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

        # Advance CPU phases
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
            state = GameEngine.apply_move(state, matched)
        else:
            # DIVERGENCE!
            print(f"\n=== DIVERGENCE at GPU move {i} ===")
            print(f"GPU move: {move_type_str} from={from_pos} to={to_pos} player={gpu_player} phase={gpu_phase}")
            print(f"CPU state: phase={state.current_phase.value} player={state.current_player}")
            print(f"Board at positions of interest:")
            for x, y in positions_of_interest:
                pos = Position(x=x, y=y)
                stack = BoardManager.get_stack(pos, state.board)
                marker = state.board.positions.get((x, y), {}).get('marker', None) if hasattr(state.board, 'positions') else None
                if stack and stack.stack_height > 0:
                    print(f"  ({x},{y}): P{stack.controlling_player}h{stack.stack_height}")
                else:
                    print(f"  ({x},{y}): empty")

            # Show what CPU thinks is valid
            print(f"\nValid moves for CPU ({len(valid)}):")
            for v in valid[:10]:
                print(f"  {v.type.value} from={v.from_pos} to={v.to}")
            if len(valid) > 10:
                print(f"  ... and {len(valid) - 10} more")

            # Try bookkeeping synthesis
            req = GameEngine.get_phase_requirement(state, state.current_player)
            if req:
                synth = GameEngine.synthesize_bookkeeping_move(req, state)
                state = GameEngine.apply_move(state, synth)
                print(f"\n(Applied bookkeeping: {synth.type.value})")

    print("\nDone.")


if __name__ == '__main__':
    main()
