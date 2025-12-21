#!/usr/bin/env python3
"""Verify that AI moves have correct z coordinates matching engine moves."""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.models import AIConfig, BoardType
from app.rules.default_engine import DefaultRulesEngine
from app.training.initial_state import create_initial_state
from app.ai.gumbel_mcts_ai import GumbelMCTSAI


def compare_positions(ai_pos, engine_pos, label: str):
    """Compare two positions and report differences."""
    if ai_pos is None and engine_pos is None:
        return True
    if ai_pos is None or engine_pos is None:
        print(f"  {label}: AI={ai_pos}, Engine={engine_pos} - MISMATCH (one is None)")
        return False

    if ai_pos.x != engine_pos.x or ai_pos.y != engine_pos.y:
        print(f"  {label}: AI=({ai_pos.x},{ai_pos.y},{ai_pos.z}), Engine=({engine_pos.x},{engine_pos.y},{engine_pos.z}) - XY MISMATCH")
        return False

    if ai_pos.z != engine_pos.z:
        print(f"  {label}: AI=({ai_pos.x},{ai_pos.y},{ai_pos.z}), Engine=({engine_pos.x},{engine_pos.y},{engine_pos.z}) - Z MISMATCH!")
        return False

    return True


def main():
    engine = DefaultRulesEngine()
    state = create_initial_state(board_type=BoardType.HEX8, num_players=2)

    # Create AI with minimal config
    ai_config = AIConfig(difficulty=1, self_play=True)
    ai = GumbelMCTSAI(player_number=1, config=ai_config, board_type=BoardType.HEX8)

    print("Testing AI move positions vs engine legal moves for hex8...")
    print()

    mismatches = 0
    for move_num in range(10):  # Test first 10 moves
        current_player = state.current_player

        # Get legal moves from engine
        legal_moves = engine.get_valid_moves(state, current_player)

        # Get AI's move
        ai.player_number = current_player
        ai_move = ai.select_move(state)

        if ai_move is None:
            print(f"Move {move_num}: AI returned None, using first legal move")
            move = legal_moves[0] if legal_moves else None
        else:
            # Find matching legal move
            found = False
            for legal in legal_moves:
                if legal.type != ai_move.type:
                    continue

                from_match = compare_positions(ai_move.from_pos, legal.from_pos, "from")
                to_match = compare_positions(ai_move.to, legal.to, "to")

                if from_match and to_match:
                    found = True
                    move = legal
                    break

            if not found:
                print(f"Move {move_num}: AI move not found in legal moves!")
                print(f"  AI: type={ai_move.type.value}, from={ai_move.from_pos}, to={ai_move.to}")
                print(f"  Legal moves of same type:")
                for legal in legal_moves:
                    if legal.type == ai_move.type:
                        print(f"    from={legal.from_pos}, to={legal.to}")
                mismatches += 1
                move = legal_moves[0] if legal_moves else None

        if move is None:
            print(f"Move {move_num}: No valid move, breaking")
            break

        # Apply move
        state = engine.apply_move(state, move, trace_mode=True)
        print(f"Move {move_num}: OK - {move.type.value}")

    print()
    print(f"Total mismatches: {mismatches}")


if __name__ == "__main__":
    main()
