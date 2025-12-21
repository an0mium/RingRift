#!/usr/bin/env python
"""Debug specific parity failures."""
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

GPU_BOOKKEEPING_MOVES = {
    'skip_capture', 'skip_recovery', 'no_placement_action',
    'no_movement_action', 'no_line_action', 'no_territory_action',
    'process_line', 'process_territory_region',
}


def debug_seed(seed: int, stop_at_error: bool = True) -> None:
    """Debug a specific seed with detailed output."""
    print(f"\n{'='*60}")
    print(f"Debugging seed {seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device='cpu')

    for step in range(100):
        game_status = runner.state.game_status[0].item()
        move_count = runner.state.move_count[0].item()
        if game_status != 0 or move_count >= 60:
            break
        runner._step_games([{}])

    game_dict = export_game_to_canonical_dict(runner.state, 0, 'square8', 2)
    initial_state = create_initial_state(BoardType.SQUARE8, num_players=2)
    state = initial_state

    from app.models import GamePhase, Move

    for i, m in enumerate(game_dict['moves']):
        move_type_str = m['type']
        gpu_phase = m.get('phase', 'ring_placement')
        gpu_player = m.get('player', 1)

        # Skip bookkeeping
        if move_type_str in GPU_BOOKKEEPING_MOVES:
            continue

        move_type = MoveType(move_type_str)
        from_pos = Position(**m['from']) if 'from' in m and m['from'] else None
        to_pos = Position(**m['to']) if 'to' in m and m['to'] else None

        # Try to advance CPU through phases
        for _ in range(10):
            current_phase = state.current_phase.value
            current_player = state.current_player

            if current_phase == gpu_phase and current_player == gpu_player:
                break

            req = GameEngine.get_phase_requirement(state, state.current_player)
            if req:
                synth = GameEngine.synthesize_bookkeeping_move(req, state)
                state = GameEngine.apply_move(state, synth)
            else:
                if current_phase in ('capture', 'chain_capture'):
                    valid = GameEngine.get_valid_moves(state, state.current_player)
                    skip_moves = [v for v in valid if v.type == MoveType.SKIP_CAPTURE]
                    if skip_moves:
                        state = GameEngine.apply_move(state, skip_moves[0])
                        continue
                elif current_phase == 'territory_processing':
                    valid = GameEngine.get_valid_moves(state, state.current_player)
                    skip_moves = [v for v in valid if v.type == MoveType.SKIP_TERRITORY_PROCESSING]
                    if skip_moves and gpu_player != current_player:
                        state = GameEngine.apply_move(state, skip_moves[0])
                        continue
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
            elif move_type in (MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT):
                v_from = v.from_pos.to_key() if v.from_pos else None
                m_from = from_pos.to_key() if from_pos else None
                if v_from == m_from and v_to == m_to:
                    matched = v
                    break
            elif move_type == MoveType.SKIP_PLACEMENT:
                matched = v
                break
            elif move_type == MoveType.RECOVERY_SLIDE:
                v_from = v.from_pos.to_key() if v.from_pos else None
                m_from = from_pos.to_key() if from_pos else None
                if v_from == m_from and v_to == m_to:
                    matched = v
                    break
            elif move_type in (MoveType.CHOOSE_LINE_OPTION, MoveType.PROCESS_LINE):
                matched = v
                break
            elif move_type in (MoveType.CHOOSE_TERRITORY_OPTION, MoveType.PROCESS_TERRITORY_REGION, MoveType.TERRITORY_CLAIM):
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
            # FAILURE - print detailed debug info
            print(f"\n--- FAILURE at move {i} ---")
            print(f"GPU move: {move_type_str} @ phase={gpu_phase} player={gpu_player}")
            print(f"  from={m.get('from')}, to={m.get('to')}")
            print(f"\nCPU state:")
            print(f"  phase={state.current_phase.value}, player={state.current_player}")

            # Get player info from CPU
            from app.rules.core import is_eligible_for_recovery, player_controls_any_stack, player_has_markers, count_buried_rings
            for p in [1, 2]:
                player = state.players[p - 1]  # 0-indexed
                has_stacks = player_controls_any_stack(state.board, p)
                has_markers = player_has_markers(state.board, p)
                buried = count_buried_rings(state.board, p)
                recovery = is_eligible_for_recovery(state, p)
                print(f"\n  Player {p}: rings={player.rings_in_hand}, stacks={has_stacks}, markers={has_markers}, buried={buried}, recovery={recovery}")

            print(f"\nCPU valid moves ({len(valid)}):")
            move_types = {}
            for v in valid:
                t = v.type.value
                if t not in move_types:
                    move_types[t] = []
                move_types[t].append(v)
            for t, moves in sorted(move_types.items()):
                print(f"  {t}: {len(moves)} moves")
                if len(moves) <= 5:
                    for mv in moves:
                        print(f"    from={mv.from_pos}, to={mv.to}")

            # Print last few moves from GPU
            print(f"\n--- Last 5 GPU moves before failure ---")
            for j in range(max(0, i-5), i+1):
                gm = game_dict['moves'][j]
                print(f"  Move {j}: {gm['type']} @ phase={gm.get('phase')} player={gm.get('player')}")
                print(f"           from={gm.get('from')}, to={gm.get('to')}")

            if stop_at_error:
                return

            # Try bookkeeping fallback
            req = GameEngine.get_phase_requirement(state, state.current_player)
            if req:
                synth = GameEngine.synthesize_bookkeeping_move(req, state)
                state = GameEngine.apply_move(state, synth)

    print(f"\nSeed {seed}: All moves processed successfully!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('seeds', type=int, nargs='+', help='Seeds to debug')
    parser.add_argument('--continue', dest='cont', action='store_true', help='Continue after errors')
    args = parser.parse_args()

    for seed in args.seeds:
        debug_seed(seed, stop_at_error=not args.cont)
