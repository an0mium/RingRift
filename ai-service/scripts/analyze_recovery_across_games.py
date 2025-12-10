#!/usr/bin/env python3
"""
Analyze recovery eligibility across ALL selfplay games in a directory.

This script replays every game and checks at every move whether any player
meets the 4 conditions for recovery slide eligibility (RR-CANON-R110):
1. Zero rings in hand
2. Zero controlled stacks
3. At least one marker on board
4. At least one buried ring

It produces aggregate statistics and finds specific game states where
recovery should have been possible.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import traceback

from app.models import GameState, MoveType, GamePhase, Position
from app.game_engine import GameEngine
from app.rules.core import (
    is_eligible_for_recovery,
    count_buried_rings,
    player_has_markers,
    player_controls_any_stack,
)
from app.training.generate_data import create_initial_state


@dataclass
class RecoveryConditionStats:
    """Track how often each recovery condition is met."""
    total_states_checked: int = 0
    zero_rings_in_hand: int = 0
    zero_controlled_stacks: int = 0
    has_markers: int = 0
    has_buried_rings: int = 0
    # Combined conditions
    all_four_conditions_met: int = 0
    three_conditions_met: int = 0
    two_conditions_met: int = 0
    one_condition_met: int = 0
    zero_conditions_met: int = 0
    # Detailed breakdowns
    condition_combos: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # Track near-misses (3 conditions met)
    near_misses: List[Dict] = field(default_factory=list)
    # Track actual recovery-eligible states
    eligible_states: List[Dict] = field(default_factory=list)


@dataclass
class GameStats:
    """Stats for a single game."""
    game_file: str
    game_index: int
    board_type: str
    num_players: int
    total_moves: int
    moves_replayed: int
    had_forced_elimination: bool
    had_recovery_slide: bool
    recovery_eligible_states: int
    max_buried_rings_seen: int
    min_rings_in_hand_seen: int
    max_markers_seen: int
    error: Optional[str] = None


def analyze_player_recovery_conditions(state: GameState, player: int) -> Dict[str, Any]:
    """Analyze all 4 recovery conditions for a player."""
    board = state.board
    player_state = next((p for p in state.players if p.player_number == player), None)

    rings_in_hand = player_state.rings_in_hand if player_state else 0
    controls_stacks = player_controls_any_stack(board, player)
    has_markers_val = player_has_markers(board, player)
    buried_rings = count_buried_rings(board, player)
    is_eligible = is_eligible_for_recovery(state, player)

    # Count how many conditions are met
    conditions_met = 0
    condition_flags = []

    if rings_in_hand == 0:
        conditions_met += 1
        condition_flags.append('zero_rings')
    if not controls_stacks:
        conditions_met += 1
        condition_flags.append('no_stacks')
    if has_markers_val:
        conditions_met += 1
        condition_flags.append('has_markers')
    if buried_rings > 0:
        conditions_met += 1
        condition_flags.append('has_buried')

    return {
        'player': player,
        'rings_in_hand': rings_in_hand,
        'controls_stacks': controls_stacks,
        'has_markers': has_markers_val,
        'buried_rings': buried_rings,
        'is_eligible': is_eligible,
        'conditions_met': conditions_met,
        'condition_flags': condition_flags,
    }


def get_bookkeeping_move(state, player):
    """Get bookkeeping move if one is needed."""
    req = GameEngine.get_phase_requirement(state, player)
    if req is not None:
        return GameEngine.synthesize_bookkeeping_move(req, state)
    return None


def replay_and_analyze_game(
    game_data: Dict,
    game_file: str,
    game_index: int,
    stats: RecoveryConditionStats,
    verbose: bool = False,
) -> GameStats:
    """Replay a game and analyze recovery eligibility at each state."""
    moves_json = game_data.get('moves', [])
    board_type = game_data.get('board_type', 'square8')
    num_players = game_data.get('num_players', 2)

    # Create initial state
    state = create_initial_state(board_type=board_type, num_players=num_players)

    game_stats = GameStats(
        game_file=game_file,
        game_index=game_index,
        board_type=board_type,
        num_players=num_players,
        total_moves=len(moves_json),
        moves_replayed=0,
        had_forced_elimination=any(m.get('type') == 'forced_elimination' for m in moves_json),
        had_recovery_slide=any(m.get('type') == 'recovery_slide' for m in moves_json),
        recovery_eligible_states=0,
        max_buried_rings_seen=0,
        min_rings_in_hand_seen=99,
        max_markers_seen=0,
    )

    # Replay moves
    for move_idx, m_json in enumerate(moves_json):
        move_type_str = m_json.get('type')
        move_player = m_json.get('player', state.current_player)

        # Analyze recovery conditions for ALL players at this state
        for player in range(1, num_players + 1):
            analysis = analyze_player_recovery_conditions(state, player)
            stats.total_states_checked += 1

            # Update condition counts
            if analysis['rings_in_hand'] == 0:
                stats.zero_rings_in_hand += 1
            if not analysis['controls_stacks']:
                stats.zero_controlled_stacks += 1
            if analysis['has_markers']:
                stats.has_markers += 1
            if analysis['buried_rings'] > 0:
                stats.has_buried_rings += 1

            # Track game-level stats
            game_stats.max_buried_rings_seen = max(
                game_stats.max_buried_rings_seen, analysis['buried_rings']
            )
            game_stats.min_rings_in_hand_seen = min(
                game_stats.min_rings_in_hand_seen, analysis['rings_in_hand']
            )
            game_stats.max_markers_seen = max(
                game_stats.max_markers_seen, len(state.board.markers)
            )

            # Count conditions met
            n_met = analysis['conditions_met']
            if n_met == 4:
                stats.all_four_conditions_met += 1
            elif n_met == 3:
                stats.three_conditions_met += 1
            elif n_met == 2:
                stats.two_conditions_met += 1
            elif n_met == 1:
                stats.one_condition_met += 1
            else:
                stats.zero_conditions_met += 1

            # Track condition combos
            combo_key = ','.join(sorted(analysis['condition_flags'])) if analysis['condition_flags'] else 'none'
            stats.condition_combos[combo_key] += 1

            # Record near-misses (3 conditions met)
            if n_met == 3 and len(stats.near_misses) < 100:
                missing = set(['zero_rings', 'no_stacks', 'has_markers', 'has_buried']) - set(analysis['condition_flags'])
                stats.near_misses.append({
                    'game_file': game_file,
                    'game_index': game_index,
                    'move_index': move_idx,
                    'player': player,
                    'missing_condition': list(missing)[0] if missing else 'unknown',
                    'rings_in_hand': analysis['rings_in_hand'],
                    'controls_stacks': analysis['controls_stacks'],
                    'has_markers': analysis['has_markers'],
                    'buried_rings': analysis['buried_rings'],
                    'phase': state.current_phase.value if state.current_phase else None,
                })

            # Record actual eligible states
            if analysis['is_eligible']:
                game_stats.recovery_eligible_states += 1
                if len(stats.eligible_states) < 100:
                    stats.eligible_states.append({
                        'game_file': game_file,
                        'game_index': game_index,
                        'move_index': move_idx,
                        'player': player,
                        'rings_in_hand': analysis['rings_in_hand'],
                        'buried_rings': analysis['buried_rings'],
                        'phase': state.current_phase.value if state.current_phase else None,
                    })

        # Apply the move
        try:
            move_type = MoveType(move_type_str)
        except ValueError:
            game_stats.error = f"Unknown move type: {move_type_str}"
            break

        valid_moves = GameEngine.get_valid_moves(state, state.current_player)
        bookkeeping = get_bookkeeping_move(state, state.current_player)

        # Find matching move
        matched_move = None
        if bookkeeping and bookkeeping.type == move_type:
            matched_move = bookkeeping
        else:
            candidates = [vm for vm in valid_moves if vm.type == move_type]

            if m_json.get('to'):
                to_x, to_y = m_json['to']['x'], m_json['to']['y']
                candidates = [vm for vm in candidates if vm.to and vm.to.x == to_x and vm.to.y == to_y]

            if m_json.get('from'):
                from_x, from_y = m_json['from']['x'], m_json['from']['y']
                new_candidates = []
                for vm in candidates:
                    from_pos = getattr(vm, 'from_position', None) or getattr(vm, 'from_pos', None)
                    if from_pos and from_pos.x == from_x and from_pos.y == from_y:
                        new_candidates.append(vm)
                candidates = new_candidates

            if candidates:
                matched_move = candidates[0]

        if not matched_move:
            game_stats.error = f"No match for move {move_idx}: {move_type_str}"
            break

        try:
            state = GameEngine.apply_move(state, matched_move)
            game_stats.moves_replayed += 1
        except Exception as e:
            game_stats.error = f"Error at move {move_idx}: {str(e)}"
            break

    return game_stats


def load_games_from_file(filepath: Path) -> List[Dict]:
    """Load games from a JSONL file, skipping invalid lines."""
    games = []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    games.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip invalid lines (e.g., partial writes from concurrent processes)
                    pass
    return games


def main():
    parser = argparse.ArgumentParser(description='Analyze recovery eligibility across selfplay games')
    parser.add_argument('--input-dir', type=str, default='data/selfplay',
                        help='Directory containing selfplay game files')
    parser.add_argument('--pattern', type=str, default='**/games.jsonl',
                        help='Glob pattern for game files')
    parser.add_argument('--max-games', type=int, default=None,
                        help='Maximum number of games to analyze')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return

    # Find all game files
    game_files = list(input_dir.glob(args.pattern))
    print(f"Found {len(game_files)} game files")

    # Aggregate stats
    stats = RecoveryConditionStats()
    all_game_stats: List[GameStats] = []
    total_games = 0
    games_with_fe = 0
    games_with_recovery = 0
    games_with_errors = 0

    # Board type breakdown
    board_type_games = defaultdict(int)
    player_count_games = defaultdict(int)

    for game_file in game_files:
        print(f"\nProcessing {game_file}...")
        try:
            games = load_games_from_file(game_file)
        except Exception as e:
            print(f"  Error loading file: {e}")
            continue

        for i, game in enumerate(games):
            if args.max_games and total_games >= args.max_games:
                break

            total_games += 1
            board_type = game.get('board_type', 'square8')
            num_players = game.get('num_players', 2)
            board_type_games[board_type] += 1
            player_count_games[num_players] += 1

            try:
                game_stats = replay_and_analyze_game(
                    game, str(game_file), i, stats, args.verbose
                )
                all_game_stats.append(game_stats)

                if game_stats.had_forced_elimination:
                    games_with_fe += 1
                if game_stats.had_recovery_slide:
                    games_with_recovery += 1
                if game_stats.error:
                    games_with_errors += 1
                    if args.verbose:
                        print(f"  Game {i}: {game_stats.error}")

            except Exception as e:
                games_with_errors += 1
                if args.verbose:
                    print(f"  Game {i} error: {e}")
                    traceback.print_exc()

        if args.max_games and total_games >= args.max_games:
            break

    # Print results
    print("\n" + "=" * 70)
    print("RECOVERY ELIGIBILITY ANALYSIS RESULTS")
    print("=" * 70)

    print(f"\nGAMES ANALYZED:")
    print(f"  Total games: {total_games}")
    print(f"  Games with forced_elimination: {games_with_fe}")
    print(f"  Games with recovery_slide: {games_with_recovery}")
    print(f"  Games with replay errors: {games_with_errors}")

    print(f"\n  By board type:")
    for bt, count in sorted(board_type_games.items()):
        print(f"    {bt}: {count}")

    print(f"\n  By player count:")
    for pc, count in sorted(player_count_games.items()):
        print(f"    {pc}p: {count}")

    print(f"\nSTATES ANALYZED: {stats.total_states_checked:,}")

    print(f"\nINDIVIDUAL CONDITION FREQUENCIES:")
    if stats.total_states_checked > 0:
        pct = lambda x: 100.0 * x / stats.total_states_checked
        print(f"  Zero rings in hand:    {stats.zero_rings_in_hand:>8,} ({pct(stats.zero_rings_in_hand):>6.2f}%)")
        print(f"  Zero controlled stacks:{stats.zero_controlled_stacks:>8,} ({pct(stats.zero_controlled_stacks):>6.2f}%)")
        print(f"  Has markers on board:  {stats.has_markers:>8,} ({pct(stats.has_markers):>6.2f}%)")
        print(f"  Has buried rings:      {stats.has_buried_rings:>8,} ({pct(stats.has_buried_rings):>6.2f}%)")

    print(f"\nCONDITIONS MET DISTRIBUTION:")
    if stats.total_states_checked > 0:
        print(f"  4 conditions (ELIGIBLE): {stats.all_four_conditions_met:>8,} ({pct(stats.all_four_conditions_met):>6.2f}%)")
        print(f"  3 conditions (near-miss):{stats.three_conditions_met:>8,} ({pct(stats.three_conditions_met):>6.2f}%)")
        print(f"  2 conditions:            {stats.two_conditions_met:>8,} ({pct(stats.two_conditions_met):>6.2f}%)")
        print(f"  1 condition:             {stats.one_condition_met:>8,} ({pct(stats.one_condition_met):>6.2f}%)")
        print(f"  0 conditions:            {stats.zero_conditions_met:>8,} ({pct(stats.zero_conditions_met):>6.2f}%)")

    print(f"\nTOP CONDITION COMBINATIONS:")
    sorted_combos = sorted(stats.condition_combos.items(), key=lambda x: -x[1])[:15]
    for combo, count in sorted_combos:
        pct_val = 100.0 * count / stats.total_states_checked if stats.total_states_checked > 0 else 0
        print(f"  {combo:40s}: {count:>8,} ({pct_val:>6.2f}%)")

    if stats.near_misses:
        print(f"\nNEAR-MISS EXAMPLES (3 conditions met):")
        # Group by missing condition
        by_missing = defaultdict(list)
        for nm in stats.near_misses:
            by_missing[nm['missing_condition']].append(nm)

        for missing, examples in by_missing.items():
            print(f"\n  Missing '{missing}' ({len(examples)} cases):")
            for ex in examples[:3]:
                print(f"    Game {ex['game_index']} in {Path(ex['game_file']).name}, move {ex['move_index']}, P{ex['player']}")
                print(f"      rings_in_hand={ex['rings_in_hand']}, stacks={ex['controls_stacks']}, markers={ex['has_markers']}, buried={ex['buried_rings']}")

    if stats.eligible_states:
        print(f"\nRECOVERY-ELIGIBLE STATES FOUND: {len(stats.eligible_states)}")
        for es in stats.eligible_states[:10]:
            print(f"  Game {es['game_index']} in {Path(es['game_file']).name}, move {es['move_index']}, P{es['player']}")
            print(f"    buried_rings={es['buried_rings']}, phase={es['phase']}")

    # Game-level stats
    games_with_eligible = sum(1 for gs in all_game_stats if gs.recovery_eligible_states > 0)
    total_eligible_states = sum(gs.recovery_eligible_states for gs in all_game_stats)
    max_buried = max((gs.max_buried_rings_seen for gs in all_game_stats), default=0)
    min_rings = min((gs.min_rings_in_hand_seen for gs in all_game_stats if gs.min_rings_in_hand_seen < 99), default=99)
    max_markers = max((gs.max_markers_seen for gs in all_game_stats), default=0)

    print(f"\nGAME-LEVEL SUMMARY:")
    print(f"  Games with recovery-eligible states: {games_with_eligible}")
    print(f"  Total recovery-eligible states: {total_eligible_states}")
    print(f"  Max buried rings seen in any game: {max_buried}")
    print(f"  Min rings in hand seen in any game: {min_rings}")
    print(f"  Max markers seen in any game: {max_markers}")

    # Save results to JSON
    if args.output:
        results = {
            'total_games': total_games,
            'games_with_fe': games_with_fe,
            'games_with_recovery': games_with_recovery,
            'games_with_errors': games_with_errors,
            'states_analyzed': stats.total_states_checked,
            'condition_frequencies': {
                'zero_rings_in_hand': stats.zero_rings_in_hand,
                'zero_controlled_stacks': stats.zero_controlled_stacks,
                'has_markers': stats.has_markers,
                'has_buried_rings': stats.has_buried_rings,
            },
            'conditions_met_distribution': {
                '4_conditions': stats.all_four_conditions_met,
                '3_conditions': stats.three_conditions_met,
                '2_conditions': stats.two_conditions_met,
                '1_condition': stats.one_condition_met,
                '0_conditions': stats.zero_conditions_met,
            },
            'condition_combos': dict(stats.condition_combos),
            'near_misses': stats.near_misses[:50],
            'eligible_states': stats.eligible_states[:50],
            'board_type_games': dict(board_type_games),
            'player_count_games': {str(k): v for k, v in player_count_games.items()},
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
