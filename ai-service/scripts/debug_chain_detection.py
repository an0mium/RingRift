#!/usr/bin/env python
"""Debug chain capture detection divergence at move 14/15."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from app.ai.gpu_parallel_games import ParallelGameRunner
from app.ai.gpu_canonical_export import export_game_to_canonical_dict
from app.ai.gpu_move_generation import generate_chain_capture_moves_from_position
from app.game_engine import GameEngine
from app.training.initial_state import create_initial_state
from app.models import BoardType, Move, MoveType, Position
from app.rules.capture_chain import get_chain_capture_continuation_info_py


def main():
    # Run GPU game up to move 15
    torch.manual_seed(42)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device="cpu")

    for step in range(100):
        if runner.state.move_count[0].item() >= 16:
            break
        runner._step_games([{}])

    # Print moves 12-15
    print("GPU Move History:")
    for i in range(12, 17):
        if i >= runner.state.move_count[0].item():
            break
        move = runner.state.move_history[0, i]
        from app.ai.gpu_game_types import MoveType as GMT, GamePhase as GP
        mt = GMT(int(move[0].item())).name
        player = int(move[1].item())
        from_y, from_x = int(move[2].item()), int(move[3].item())
        to_y, to_x = int(move[4].item()), int(move[5].item())
        phase = GP(int(move[6].item())).name
        print(f"  Move {i}: {mt} P{player} ({from_x},{from_y})->({to_x},{to_y}) phase={phase}")

    # Print board after these moves
    print("\nGPU Board after move 15+:")
    for y in range(8):
        row = ""
        for x in range(8):
            owner = runner.state.stack_owner[0, y, x].item()
            height = runner.state.stack_height[0, y, x].item()
            if owner > 0:
                row += f"P{int(owner)}h{int(height)} "
            else:
                row += "..... "
        print(f"  y={y}: {row}")

    # Now replay on CPU to understand the divergence
    game_dict = export_game_to_canonical_dict(runner.state, 0, "square8", 2)
    initial_state = create_initial_state(BoardType.SQUARE8, num_players=2)
    state = initial_state

    print("\nCPU Replay:")
    for i, m in enumerate(game_dict["moves"][:15]):  # Stop at move 14
        move_type = MoveType(m["type"])
        from_pos = Position(**m["from"]) if "from" in m and m["from"] else None
        to_pos = Position(**m["to"]) if "to" in m and m["to"] else None

        valid = GameEngine.get_valid_moves(state, state.current_player)
        matched = None

        # Find matching move
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
            if i >= 12:
                print(f"  Move {i}: OK - {matched.type.value}")
        else:
            req = GameEngine.get_phase_requirement(state, state.current_player)
            if req:
                synth = GameEngine.synthesize_bookkeeping_move(req, state)
                state = GameEngine.apply_move(state, synth)
                if i >= 12:
                    print(f"  Move {i}: Synthesized {synth.type.value}")
            else:
                print(f"  Move {i}: NO MATCH for {move_type.value}")
                print(f"    Phase: {state.current_phase.value}, Player: {state.current_player}")
                break

    print(f"\nAfter move 14:")
    print(f"  CPU Phase: {state.current_phase.value}")
    print(f"  CPU Player: {state.current_player}")

    # Check CPU chain capture detection from (0,4)
    from_pos = Position(x=0, y=4)
    chain_info = get_chain_capture_continuation_info_py(state, 2, from_pos)
    print(f"\nCPU chain capture from (0,4) for P2:")
    print(f"  must_continue: {chain_info.must_continue}")
    print(f"  available: {len(chain_info.available_continuations)}")
    for c in chain_info.available_continuations[:3]:
        print(f"    {c.from_pos} -> {c.to}")


if __name__ == "__main__":
    main()
