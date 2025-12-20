#!/usr/bin/env python
"""Debug GPU to CPU parity - detailed tracing."""
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

# GPU bookkeeping moves that CPU handles implicitly
GPU_BOOKKEEPING_MOVES = {
    'skip_capture', 'skip_recovery',
    'no_placement_action', 'no_movement_action', 'no_line_action', 'no_territory_action',
    'process_line', 'process_territory_region', 'choose_line_option', 'choose_territory_option',
}


def main():
    seed = 42
    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    for step in range(100):
        game_status = runner.state.game_status[0].item()
        move_count = runner.state.move_count[0].item()
        if game_status != 0 or move_count >= 60:
            break
        runner._step_games([{}])

    game_dict = export_game_to_canonical_dict(runner.state, 0, 'square8', 2)
    print(f"GPU game: {len(game_dict['moves'])} moves")

    # Show first 20 exported moves
    print("\nFirst 20 exported GPU moves:")
    for i, m in enumerate(game_dict['moves'][:20]):
        from_str = f"({m['from']['x']},{m['from']['y']})" if 'from' in m and m['from'] else "None"
        to_str = f"({m['to']['x']},{m['to']['y']})" if 'to' in m and m['to'] else "None"
        phase = m.get('phase', 'unknown')
        print(f"  {i:2}: [{phase:20}] {m['type']:25} from={from_str:10} to={to_str:10} player={m['player']}")

    # Now replay on CPU with detailed tracing
    print("\n\nCPU Replay (detailed):")
    initial_state = create_initial_state(BoardType.SQUARE8, num_players=2)
    state = initial_state

    for i, m in enumerate(game_dict['moves'][:20]):
        move_type_str = m['type']

        if move_type_str in GPU_BOOKKEEPING_MOVES:
            print(f"  {i:2}: SKIP {move_type_str}")
            continue

        move_type = MoveType(move_type_str)
        from_pos = Position(**m['from']) if 'from' in m and m['from'] else None
        to_pos = Position(**m['to']) if 'to' in m and m['to'] else None

        cpu_phase = state.current_phase.value
        cpu_player = state.current_player

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
            elif move_type in (MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT):
                v_from = v.from_pos.to_key() if v.from_pos else None
                m_from = from_pos.to_key() if from_pos else None
                if v_from == m_from and v_to == m_to:
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
            print(f"  {i:2}: OK   {move_type_str:25} cpu_phase={cpu_phase:15} cpu_player={cpu_player}")
        else:
            # Try bookkeeping synthesis
            req = GameEngine.get_phase_requirement(state, state.current_player)
            if req:
                synth = GameEngine.synthesize_bookkeeping_move(req, state)
                state = GameEngine.apply_move(state, synth)
                print(f"  {i:2}: SYNTH for {move_type_str:20} cpu_phase={cpu_phase:15} cpu_player={cpu_player} -> synth={synth.type.value}")
            else:
                print(f"  {i:2}: FAIL {move_type_str:25} cpu_phase={cpu_phase:15} cpu_player={cpu_player}")
                print(f"        Expected: from={from_pos} to={to_pos}")
                print(f"        Valid moves ({len(valid)}):")
                for v in valid[:5]:
                    print(f"          {v.type.value} from={v.from_pos} to={v.to}")
                if len(valid) > 5:
                    print(f"          ... and {len(valid)-5} more")


if __name__ == '__main__':
    main()
