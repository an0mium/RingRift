#!/usr/bin/env python
"""Debug capture detection divergence between GPU and CPU at move 13."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from app.ai.gpu_parallel_games import ParallelGameRunner
from app.ai.gpu_canonical_export import export_game_to_canonical_dict
from app.ai.gpu_move_generation import generate_capture_moves_batch
from app.game_engine import GameEngine
from app.training.initial_state import create_initial_state
from app.models import BoardType, Move, MoveType, Position


def main():
    # Run GPU game up to move 13 (P2 movement)
    torch.manual_seed(42)
    runner = ParallelGameRunner(batch_size=1, board_size=8, num_players=2, device="cpu")

    # Run to just after move 13 (P2 movement from 4,5 to 4,4)
    for step in range(100):
        move_count = runner.state.move_count[0].item()
        if move_count >= 14:
            break
        runner._step_games([{}])

    print(f"Move count: {runner.state.move_count[0].item()}")

    # Print recent moves
    for i in range(14):
        move = runner.state.move_history[0, i]
        move_type = int(move[0].item())
        player = int(move[1].item())
        from_y = int(move[2].item())
        from_x = int(move[3].item())
        to_y = int(move[4].item())
        to_x = int(move[5].item())
        phase = int(move[6].item())
        print(f"  Move {i}: type={move_type} P{player} ({from_x},{from_y})->({to_x},{to_y}) phase={phase}")

    print("\nGPU Board state:")
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

    # Check what captures GPU sees
    print("\nGPU current player:", runner.state.current_player[0].item())
    print("GPU phase:", runner.state.current_phase[0].item())

    # Generate captures
    games_mask = torch.ones(1, dtype=torch.bool)
    captures = generate_capture_moves_batch(runner.state, games_mask)
    print(f"GPU captures available: {captures.moves_per_game[0].item()}")

    # Now let's replay on CPU to the same point
    game_dict = export_game_to_canonical_dict(runner.state, 0, "square8", 2)

    initial_state = create_initial_state(BoardType.SQUARE8, num_players=2)
    state = initial_state

    for i, m in enumerate(game_dict["moves"][:14]):  # Up to move 13
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

    print(f"\nCPU after move 13:")
    print(f"  Phase: {state.current_phase.value}, Player: {state.current_player}")

    # Get CPU valid moves
    valid = GameEngine.get_valid_moves(state, state.current_player)
    print(f"  Valid moves: {len(valid)}")
    capture_moves = [v for v in valid if "capture" in v.type.value]
    print(f"  Capture moves: {len(capture_moves)}")
    for c in capture_moves[:5]:
        print(f"    {c.type.value}: {c.from_pos} -> {c.to}")


if __name__ == "__main__":
    main()
