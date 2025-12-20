#!/usr/bin/env python
"""Debug state after the first capture at move 14."""
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
from app.board_manager import BoardManager


def main():
    # Run GPU game up to just before move 15
    torch.manual_seed(42)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device="cpu")

    for step in range(100):
        if runner.state.move_count[0].item() >= 15:
            break
        runner._step_games([{}])

    # Export and replay on CPU up to move 14
    game_dict = export_game_to_canonical_dict(runner.state, 0, "square8", 2)
    initial_state = create_initial_state(BoardType.SQUARE8, num_players=2)
    state = initial_state

    # Replay up to and including move 13 (before the capture)
    for i, m in enumerate(game_dict["moves"][:14]):
        move_type = MoveType(m["type"])
        from_pos = Position(**m["from"]) if "from" in m and m["from"] else None
        to_pos = Position(**m["to"]) if "to" in m and m["to"] else None

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

    print("Before move 14 (capture):")
    print(f"  Phase: {state.current_phase.value}")
    print(f"  Player: {state.current_player}")

    # Show stacks
    print("\nCPU Board:")
    for y in range(8):
        row = ""
        for x in range(8):
            pos = Position(x=x, y=y)
            stack = BoardManager.get_stack(pos, state.board)
            if stack and stack.stack_height > 0:
                row += f"P{stack.controlling_player}h{stack.stack_height} "
            else:
                row += "..... "
        print(f"  y={y}: {row}")

    # Get the capture move
    print("\nMove 14 from GPU:", game_dict["moves"][14])

    # Apply the capture manually and check chain capture
    m = game_dict["moves"][14]
    move_type = MoveType(m["type"])
    from_pos = Position(**m["from"])
    to_pos = Position(**m["to"])

    valid = GameEngine.get_valid_moves(state, state.current_player)
    capture_moves = [v for v in valid if v.type in (MoveType.OVERTAKING_CAPTURE, MoveType.CONTINUE_CAPTURE_SEGMENT)]
    print(f"\nCPU valid capture moves: {len(capture_moves)}")
    for c in capture_moves[:5]:
        print(f"  {c.type.value}: {c.from_pos} -> {c.to} (target: {c.capture_target})")

    # Find and apply the matching capture
    matched = None
    for v in capture_moves:
        if v.from_pos.to_key() == from_pos.to_key() and v.to.to_key() == to_pos.to_key():
            matched = v
            break

    if matched:
        print(f"\nApplying capture: {matched.type.value} {matched.from_pos} -> {matched.to}")
        state_after = GameEngine.apply_move(state, matched)
        print(f"\nAfter capture:")
        print(f"  Phase: {state_after.current_phase.value}")
        print(f"  Player: {state_after.current_player}")

        # Show board after
        print("\nCPU Board after capture:")
        for y in range(8):
            row = ""
            for x in range(8):
                pos = Position(x=x, y=y)
                stack = BoardManager.get_stack(pos, state_after.board)
                if stack and stack.stack_height > 0:
                    row += f"P{stack.controlling_player}h{stack.stack_height} "
                else:
                    row += "..... "
            print(f"  y={y}: {row}")

        # Check chain capture state
        if hasattr(state_after, 'chain_capture_state') and state_after.chain_capture_state:
            print(f"\nChain capture state:")
            print(f"  current_position: {state_after.chain_capture_state.current_position}")
        else:
            print("\nNo chain capture state")

        # Check continuation
        landing = Position(x=0, y=4)
        chain_info = get_chain_capture_continuation_info_py(state_after, 2, landing)
        print(f"\nChain capture from (0,4):")
        print(f"  must_continue: {chain_info.must_continue}")
        print(f"  available: {len(chain_info.available_continuations)}")
    else:
        print("\nNo matching capture found!")


if __name__ == "__main__":
    main()
