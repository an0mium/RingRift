#!/usr/bin/env python
"""Compare GPU and CPU states at specific points."""
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
from app.board_manager import BoardManager


def print_board(name, board, board_size=8):
    print(f"\n{name}:")
    for y in range(board_size):
        row = ""
        for x in range(board_size):
            pos = Position(x=x, y=y)
            stack = BoardManager.get_stack(pos, board)
            if stack and stack.stack_height > 0:
                row += f"P{stack.controlling_player}h{stack.stack_height} "
            else:
                row += "..... "
        print(f"  y={y}: {row}")


def print_gpu_board(runner):
    print("\nGPU Board:")
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


def main():
    # Run GPU game
    torch.manual_seed(42)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device="cpu")

    # Run to at least move 14
    for step in range(50):
        if runner.state.move_count[0].item() >= 14:
            break
        runner._step_games([{}])

    # Export and replay on CPU
    game_dict = export_game_to_canonical_dict(runner.state, 0, "square8", 2)
    print(f"Exported {len(game_dict['moves'])} moves")

    # We want to compare states AFTER move 5 (P2 moves from 3,4 to 1,4)
    # This is where the P2h2 vs P2h1 discrepancy might originate

    initial_state = create_initial_state(BoardType.SQUARE8, num_players=2)
    state = initial_state

    print("Replaying moves 0-5:")
    for i, m in enumerate(game_dict["moves"][:8]):  # Through move 7
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
            print(f"  Move {i}: {matched.type.value} from={matched.from_pos} to={matched.to}")
        else:
            req = GameEngine.get_phase_requirement(state, state.current_player)
            if req:
                synth = GameEngine.synthesize_bookkeeping_move(req, state)
                state = GameEngine.apply_move(state, synth)
                print(f"  Move {i}: Synthesized {synth.type.value}")

        # Print board after key moves
        if i in [1, 4, 5]:
            print(f"\n  After move {i}:")
            for y in range(8):
                row = ""
                for x in range(8):
                    pos = Position(x=x, y=y)
                    stack = BoardManager.get_stack(pos, state.board)
                    if stack and stack.stack_height > 0:
                        row += f"P{stack.controlling_player}h{stack.stack_height} "
                    else:
                        row += "..... "
                print(f"    y={y}: {row}")

    # Now check GPU state - need to run it properly
    print("\n\nRunning GPU to compare:")
    torch.manual_seed(42)
    runner2 = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device="cpu")

    for step in range(20):
        mc = int(runner2.state.move_count[0].item())
        if mc >= 8:
            break
        runner2._step_games([{}])

    # Show GPU board
    print_gpu_board(runner2)

    # Compare specific stacks
    print(f"\nGPU stack at (1,4): owner={int(runner2.state.stack_owner[0, 4, 1].item())}, h={int(runner2.state.stack_height[0, 4, 1].item())}")

    # CPU stack at (1,4)
    cpu_stack = BoardManager.get_stack(Position(x=1, y=4), state.board)
    if cpu_stack:
        print(f"CPU stack at (1,4): owner={cpu_stack.controlling_player}, h={cpu_stack.stack_height}")
    else:
        print("CPU stack at (1,4): None")


if __name__ == "__main__":
    main()
