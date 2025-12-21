#!/usr/bin/env python3
"""Debug hex8 replay parity issues.

This script replays a hex8 JSONL game step by step, showing the phase
at each move and identifying exactly where the recording diverges from
the actual game state.
"""

import json
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.game_engine import GameEngine
from app.models import BoardType, GamePhase, GameStatus, Move, MoveType, Position
from app.rules.default_engine import DefaultRulesEngine
from app.training.initial_state import create_initial_state


def parse_move(move_data: dict, move_idx: int) -> Move:
    """Parse a move from JSONL format."""
    import uuid
    move_type = MoveType(move_data["type"])
    player = move_data["player"]

    from_pos = None
    if "from" in move_data:
        f = move_data["from"]
        from_pos = Position(x=f["x"], y=f["y"], z=f.get("z"))

    to_pos = None
    if "to" in move_data:
        t = move_data["to"]
        to_pos = Position(x=t["x"], y=t["y"], z=t.get("z"))

    capture_target = None
    if "capture_target" in move_data:
        ct = move_data["capture_target"]
        capture_target = Position(x=ct["x"], y=ct["y"], z=ct.get("z"))

    return Move(
        id=str(uuid.uuid4()),
        type=move_type,
        player=player,
        from_pos=from_pos,
        to=to_pos,
        capture_target=capture_target,
    )


def main():
    # Load the first game
    jsonl_path = sys.argv[1] if len(sys.argv) > 1 else "data/selfplay/hex8_fixed/test_fixed.jsonl"

    with open(jsonl_path) as f:
        game_data = json.loads(f.readline())

    print(f"Game ID: {game_data['game_id']}")
    print(f"Board type: {game_data['board_type']}")
    print(f"Total moves: {len(game_data['moves'])}")
    print()

    # Initialize engine
    engine = DefaultRulesEngine()
    state = create_initial_state(
        board_type=BoardType.HEX8,
        num_players=2,
    )

    print("Initial state:")
    print(f"  Phase: {state.current_phase.value}")
    print(f"  Player: {state.current_player}")
    print()

    # Replay move by move
    for i, move_data in enumerate(game_data["moves"]):
        recorded_type = move_data["type"]
        recorded_player = move_data["player"]
        has_policy = bool(move_data.get("mcts_policy"))

        # Current engine state
        engine_phase = state.current_phase.value
        engine_player = state.current_player

        # Check for mismatch
        phase_ok = True
        player_ok = recorded_player == engine_player

        # Parse the move
        try:
            move = parse_move(move_data, i)
        except Exception as e:
            print(f"Move {i}: PARSE ERROR - {e}")
            break

        # For move_stack and overtaking_capture, check if from position exists
        if move.type.value in ('move_stack', 'overtaking_capture') and move.from_pos:
            from_key = move.from_pos.to_key()
            if from_key not in state.board.stacks:
                print(f"Move {i:3d}: PRE-CHECK FAIL")
                print(f"          Recorded from_pos {from_key} not on board!")
                print(f"          Board stacks: {list(state.board.stacks.keys())}")
                print(f"          Full move data: {move_data}")
                break

        # Check valid moves from engine
        valid_moves = engine.get_valid_moves(state, engine_player)
        valid_types = set(m.type.value for m in valid_moves)

        # At moves 42-44, print board state for debugging
        if i in (42, 43, 44):
            print(f"          *** MOVE {i} DIAGNOSTICS ***")
            print(f"          Board stacks BEFORE move: {list(state.board.stacks.keys())}")
            for pos_key, stack in state.board.stacks.items():
                print(f"            Stack {pos_key}: height={stack.stack_height}, rings={stack.rings}")
            print(f"          Board markers BEFORE move: {list(state.board.markers.keys())}")
            print(f"          Recorded: type={recorded_type}, from={move_data.get('from')}, to={move_data.get('to')}")
            print(f"          Parsed move: type={move.type.value}, from={move.from_pos}, to={move.to}")
            if move.to:
                print(f"          Parsed move.to.to_key(): {move.to.to_key()}")
            if recorded_type == 'move_stack':
                print(f"          Valid move_stack moves with matching to:")
                target_to = move_data.get('to', {})
                for vm in valid_moves:
                    if vm.type.value == 'move_stack':
                        from_str = f"({vm.from_pos.x},{vm.from_pos.y},{vm.from_pos.z})" if vm.from_pos else "None"
                        to_str = f"({vm.to.x},{vm.to.y},{vm.to.z})" if vm.to else "None"
                        match = ""
                        if vm.to and target_to:
                            if vm.to.x == target_to.get('x') and vm.to.y == target_to.get('y'):
                                match = " <-- MATCHES x,y"
                        print(f"            from={from_str} to={to_str} key={vm.to.to_key() if vm.to else 'None'}{match}")

        # Check if recorded move type is valid for current phase
        if recorded_type not in valid_types:
            phase_ok = False

        # Print status
        status = "✓" if (phase_ok and player_ok) else "✗"
        print(f"Move {i:3d}: {status} {recorded_type:25s} player={recorded_player} policy={has_policy}")
        print(f"          Engine: phase={engine_phase:25s} player={engine_player}")

        if not phase_ok:
            print(f"          ERROR: Move type '{recorded_type}' not valid in phase '{engine_phase}'")
            print(f"          Valid move types: {sorted(valid_types)}")
            print()
            print("=== BOARD STATE AT FAILURE ===")
            print(f"Current phase: {state.current_phase.value}")
            print(f"Current player: {state.current_player}")
            print(f"Recorded move at failure: {move_data}")
            # Print the previous move to understand context
            if i > 0:
                prev_move = game_data["moves"][i-1]
                print(f"Previous move (move {i-1}): {prev_move}")
            print(f"Board stacks ({len(state.board.stacks)} total):")
            for pos_str, stack in sorted(state.board.stacks.items()):
                print(f"  {pos_str}: player={stack.controlling_player}, height={stack.stack_height}, rings={stack.rings}")
            # Show moves list
            print()
            print("=== VALID MOVES AT FAILURE ===")
            for vm in valid_moves[:10]:
                from_str = f"({vm.from_pos.x}, {vm.from_pos.y}, {vm.from_pos.z})" if vm.from_pos else "None"
                to_str = f"({vm.to.x}, {vm.to.y}, {vm.to.z})" if vm.to else "None"
                print(f"  {vm.type.value}: from={from_str}, to={to_str}")
            if len(valid_moves) > 10:
                print(f"  ... and {len(valid_moves) - 10} more")
            break

        if not player_ok:
            print(f"          ERROR: Recorded player {recorded_player} != engine player {engine_player}")
            break

        # Apply move
        try:
            state = engine.apply_move(state, move, trace_mode=True)
            # For move 30, show stacks immediately after apply
            if i == 30:
                print(f"          STACKS IMMEDIATELY AFTER apply_move: {list(state.board.stacks.keys())}")
        except Exception as e:
            print(f"          APPLY ERROR: {e}")
            break

        # Show new phase after move
        new_phase = state.current_phase.value
        new_player = state.current_player
        print(f"          -> New phase: {new_phase}, player: {new_player}")

        # Check for formed lines after move 30
        if i == 30:
            print(f"          FORMED LINES: {state.board.formed_lines}")

        # Check if the move's target position exists on the board after applying
        if move.to and move.type.value in ('move_stack', 'place_ring'):
            to_key = move.to.to_key()
            if to_key not in state.board.stacks:
                print(f"          WARNING: Expected stack at {to_key} but not found!")
                print(f"          Board stacks: {list(state.board.stacks.keys())}")
                # Check what move was actually valid and got selected
                print(f"          Recorded to position: ({move.to.x}, {move.to.y}, {move.to.z})")
                print(f"          Move z is None: {move.to.z is None}")
                # Check if there's a matching position without z
                alt_key = f"{move.to.x},{move.to.y}"
                if alt_key in state.board.stacks:
                    print(f"          FOUND at {alt_key} (without z)!")
                # Check for any position with same x,y
                for k in state.board.stacks:
                    parts = k.split(',')
                    if len(parts) >= 2 and int(parts[0]) == move.to.x and int(parts[1]) == move.to.y:
                        print(f"          FOUND matching x,y at key: {k}")
        print()

        if state.game_status != GameStatus.ACTIVE:
            print(f"Game ended: {state.game_status.value}, winner: {state.winner}")
            break

    print(f"\nReplayed {i+1} moves")


if __name__ == "__main__":
    main()
