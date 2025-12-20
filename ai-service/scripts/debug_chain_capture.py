#!/usr/bin/env python
"""Debug chain capture divergence between GPU and CPU."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from app.ai.gpu_parallel_games import ParallelGameRunner
from app.ai.gpu_canonical_export import export_game_to_canonical_dict
from app.game_engine import GameEngine
from app.training.initial_state import create_initial_state
from app.models import BoardType, Move, MoveType, Position
from app.rules.capture_chain import get_chain_capture_continuation_info_py


def main():
    # Run GPU game
    torch.manual_seed(42)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device="cpu")
    for step in range(200):
        if runner.state.move_count[0].item() >= 40:
            break
        runner._step_games([{}])

    # Export to canonical format
    game_dict = export_game_to_canonical_dict(runner.state, 0, "square8", 2)

    # Replay moves on CPU until move 34
    initial_state = create_initial_state(BoardType.SQUARE8, num_players=2)
    state = initial_state

    # Parse and replay up to and including move 33 (place_ring)
    for i, m in enumerate(game_dict["moves"][:34]):  # Up to move 33
        move_type = MoveType(m["type"])
        from_pos = Position(**m["from"]) if "from" in m and m["from"] else None
        to_pos = Position(**m["to"]) if "to" in m and m["to"] else None

        # Find matching move or synthesize bookkeeping
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
            req = GameEngine.get_phase_requirement(state, state.current_player)
            if req:
                synth = GameEngine.synthesize_bookkeeping_move(req, state)
                state = GameEngine.apply_move(state, synth)

    print(f"After move 33:")
    print(f"  Phase: {state.current_phase.value}, Player: {state.current_player}")

    # Now apply move 34 (overtaking_capture from (3,2) to (0,5))
    m = game_dict["moves"][34]
    mtype = m["type"]
    mfrom = m.get("from")
    mto = m.get("to")
    print(f"\nGPU move 34: type={mtype}, from={mfrom}, to={mto}")

    move_type = MoveType(m["type"])
    from_pos = Position(**m["from"]) if "from" in m and m["from"] else None
    to_pos = Position(**m["to"]) if "to" in m and m["to"] else None

    valid = GameEngine.get_valid_moves(state, state.current_player)
    print(f"Valid moves: {len(valid)}")
    for v in valid[:5]:
        print(f"  {v.type.value}: from={v.from_pos}, to={v.to}")

    # Find matching capture
    matched = None
    for v in valid:
        if v.type in (MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT):
            v_from = v.from_pos.to_key() if v.from_pos else None
            v_to = v.to.to_key() if v.to else None
            m_from = from_pos.to_key() if from_pos else None
            m_to = to_pos.to_key() if to_pos else None
            if v_from == m_from and v_to == m_to:
                matched = v
                break

    if matched:
        print(f"\nMatched: {matched.type.value}")
        state_after = GameEngine.apply_move(state, matched)
        print(f"After move 34:")
        print(f"  Phase: {state_after.current_phase.value}, Player: {state_after.current_player}")

        # Check for chain capture
        from_pos_after = Position(x=0, y=5)
        chain_info = get_chain_capture_continuation_info_py(state_after, 1, from_pos_after)
        print(f"\nChain capture continuation from (0,5):")
        print(f"  must_continue: {chain_info.must_continue}")
        print(f"  available: {len(chain_info.available_continuations)}")
        for c in chain_info.available_continuations:
            print(f"    {c.from_pos} -> {c.to} (target: {c.capture_target})")
    else:
        print("\nNo match found!")


if __name__ == "__main__":
    main()
