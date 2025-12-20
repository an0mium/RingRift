#!/usr/bin/env python
"""Debug seed 456 - trace all moves."""
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
    'skip_capture', 'skip_recovery', 'no_placement_action', 'no_movement_action',
    'no_line_action', 'no_territory_action', 'process_line', 'process_territory_region',
    'choose_line_option', 'choose_territory_option',
}


def advance_cpu_through_phases(state, target_phase_str, target_player):
    max_iterations = 10
    for _ in range(max_iterations):
        current_phase = state.current_phase.value
        current_player = state.current_player
        if current_phase == 'ring_placement' and current_player == target_player:
            return state
        if current_phase == target_phase_str and current_player == target_player:
            return state
        req = GameEngine.get_phase_requirement(state, state.current_player)
        if req:
            synth = GameEngine.synthesize_bookkeeping_move(req, state)
            state = GameEngine.apply_move(state, synth)
        else:
            if current_phase == 'capture':
                valid = GameEngine.get_valid_moves(state, state.current_player)
                skip_moves = [v for v in valid if v.type == MoveType.SKIP_CAPTURE]
                captures = [v for v in valid if v.type in (MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT)]
                all_self_captures = True
                for c in captures:
                    if c.capture_target:
                        target_stack = BoardManager.get_stack(c.capture_target, state.board)
                        if target_stack and target_stack.controlling_player != state.current_player:
                            all_self_captures = False
                            break
                if all_self_captures and skip_moves:
                    state = GameEngine.apply_move(state, skip_moves[0])
                    continue
            break
    return state


def main():
    seed = 456
    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    for step in range(100):
        if runner.state.move_count[0].item() >= 60:
            break
        runner._step_games([{}])

    game_dict = export_game_to_canonical_dict(runner.state, 0, 'square8', 2)

    # Replay CPU with tracing for moves 44-50
    state = create_initial_state(BoardType.SQUARE8, num_players=2)

    for i, m in enumerate(game_dict['moves'][:51]):
        move_type_str = m['type']
        gpu_phase = m.get('phase', 'ring_placement')
        gpu_player = m.get('player', 1)

        if move_type_str in GPU_BOOKKEEPING_MOVES:
            state = advance_cpu_through_phases(state, gpu_phase, gpu_player)
            continue

        move_type = MoveType(move_type_str)
        from_pos = Position(**m['from']) if 'from' in m and m['from'] else None
        to_pos = Position(**m['to']) if 'to' in m and m['to'] else None

        state = advance_cpu_through_phases(state, gpu_phase, gpu_player)

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
            if i >= 44:
                print(f"  {i:2}: OK {move_type_str:25} player={gpu_player} -> cpu_phase={state.current_phase.value}")
            state = GameEngine.apply_move(state, matched)
        else:
            print(f"  {i:2}: FAIL {move_type_str:25}")
            print(f"        GPU: phase={gpu_phase} player={gpu_player} from={from_pos} to={to_pos}")
            print(f"        CPU: phase={state.current_phase.value} player={state.current_player}")
            print(f"        Valid moves ({len(valid)}):")
            for v in valid[:5]:
                print(f"          {v.type.value} from={v.from_pos} to={v.to}")
            return


if __name__ == '__main__':
    main()
