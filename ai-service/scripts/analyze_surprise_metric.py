#!/usr/bin/env python3
"""
Surprise Metric Analysis for Titans/MIRAS Applicability Assessment

This script analyzes game databases to compute "surprise" metrics - how unexpected
each move was according to the policy network. High surprise indicates moves that
could benefit from Titans-style test-time memorization.

Surprise = -log(P(chosen_move)) where P is the policy probability from MCTS

Key metrics:
- Mean surprise per game
- Surprise distribution (histogram)
- Surprise by game phase (early/mid/late)
- Surprise correlation with game outcome
- "Surprising games" that could benefit from memory
"""

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MoveData:
    """Data for a single move."""
    game_id: str
    move_number: int
    turn_number: int
    player: int
    move_type: str
    move_json: str
    move_probs: Optional[str]  # JSON string of move probabilities


@dataclass
class GameSurpriseStats:
    """Surprise statistics for a single game."""
    game_id: str
    total_moves: int
    moves_with_probs: int
    surprises: List[float]
    mean_surprise: float
    max_surprise: float
    high_surprise_count: int  # moves with surprise > threshold
    winner: Optional[int]


def compute_surprise(move_probs_json: str, chosen_move_json: str) -> Optional[float]:
    """
    Compute surprise = -log(P(chosen_move)).

    Returns None if probability data is unavailable or move not found.
    """
    if not move_probs_json:
        return None

    try:
        probs = json.loads(move_probs_json)

        # Handle different formats of move_probs
        if isinstance(probs, dict):
            # Format: {"move_key": probability, ...}
            chosen_move = json.loads(chosen_move_json) if isinstance(chosen_move_json, str) else chosen_move_json

            # Try to find the probability for the chosen move
            move_key = None
            if isinstance(chosen_move, dict):
                # Try different key formats
                if 'action' in chosen_move:
                    move_key = str(chosen_move['action'])
                elif 'type' in chosen_move:
                    move_key = json.dumps(chosen_move, sort_keys=True)
                else:
                    move_key = json.dumps(chosen_move, sort_keys=True)
            else:
                move_key = str(chosen_move)

            # Look for probability
            prob = probs.get(move_key) or probs.get(str(move_key))

            # If not found, try to match by action index
            if prob is None and isinstance(chosen_move, dict) and 'actionIndex' in chosen_move:
                prob = probs.get(str(chosen_move['actionIndex']))

            if prob is None:
                # Last resort: if only one move, use that probability
                if len(probs) == 1:
                    prob = list(probs.values())[0]
                else:
                    return None

        elif isinstance(probs, list):
            # Format: [prob1, prob2, ...] indexed by action
            chosen_move = json.loads(chosen_move_json) if isinstance(chosen_move_json, str) else chosen_move_json
            if isinstance(chosen_move, dict) and 'actionIndex' in chosen_move:
                idx = chosen_move['actionIndex']
                if 0 <= idx < len(probs):
                    prob = probs[idx]
                else:
                    return None
            else:
                return None
        else:
            return None

        # Clamp probability to avoid log(0)
        prob = max(prob, 1e-10)
        surprise = -math.log(prob)
        return surprise

    except (json.JSONDecodeError, TypeError, KeyError, ValueError) as e:
        return None


def analyze_game(conn: sqlite3.Connection, game_id: str,
                 surprise_threshold: float = 2.0) -> Optional[GameSurpriseStats]:
    """Analyze surprise for a single game."""
    cursor = conn.cursor()

    # Get game info
    cursor.execute("""
        SELECT winner, total_moves FROM games WHERE game_id = ?
    """, (game_id,))
    game_row = cursor.fetchone()
    if not game_row:
        return None
    winner, total_moves = game_row

    # Get moves with probabilities
    cursor.execute("""
        SELECT move_number, turn_number, player, move_type, move_json, move_probs
        FROM game_moves
        WHERE game_id = ?
        ORDER BY move_number
    """, (game_id,))

    moves = cursor.fetchall()
    surprises = []
    moves_with_probs = 0

    for row in moves:
        move_number, turn_number, player, move_type, move_json, move_probs = row

        if move_probs:
            moves_with_probs += 1
            surprise = compute_surprise(move_probs, move_json)
            if surprise is not None:
                surprises.append(surprise)

    if not surprises:
        return None

    return GameSurpriseStats(
        game_id=game_id,
        total_moves=total_moves,
        moves_with_probs=moves_with_probs,
        surprises=surprises,
        mean_surprise=np.mean(surprises),
        max_surprise=np.max(surprises),
        high_surprise_count=sum(1 for s in surprises if s > surprise_threshold),
        winner=winner
    )


def analyze_database(db_path: str, sample_size: int = 500,
                     surprise_threshold: float = 2.0) -> Dict:
    """
    Analyze surprise metrics across games in a database.

    Args:
        db_path: Path to game database
        sample_size: Number of games to sample
        surprise_threshold: Threshold for "high surprise" moves (default: 2.0 = ~13.5% probability)

    Returns:
        Dictionary of analysis results
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get games with move probabilities
    cursor.execute("""
        SELECT DISTINCT g.game_id
        FROM games g
        JOIN game_moves m ON g.game_id = m.game_id
        WHERE m.move_probs IS NOT NULL
        AND g.game_status = 'completed'
        ORDER BY RANDOM()
        LIMIT ?
    """, (sample_size,))

    game_ids = [row[0] for row in cursor.fetchall()]

    print(f"Analyzing {len(game_ids)} games from {db_path}...")

    # Analyze each game
    game_stats = []
    all_surprises = []

    for i, game_id in enumerate(game_ids):
        stats = analyze_game(conn, game_id, surprise_threshold)
        if stats:
            game_stats.append(stats)
            all_surprises.extend(stats.surprises)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(game_ids)} games...")

    conn.close()

    if not game_stats:
        return {"error": "No games with move probabilities found"}

    # Aggregate statistics
    all_surprises = np.array(all_surprises)
    mean_surprises = [s.mean_surprise for s in game_stats]
    max_surprises = [s.max_surprise for s in game_stats]
    high_surprise_counts = [s.high_surprise_count for s in game_stats]

    # Compute percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    surprise_percentiles = {p: np.percentile(all_surprises, p) for p in percentiles}

    # Games with high surprise (could benefit from memory)
    high_surprise_games = [s for s in game_stats if s.mean_surprise > surprise_threshold]

    results = {
        "database": db_path,
        "games_analyzed": len(game_stats),
        "total_moves_with_probs": len(all_surprises),
        "surprise_stats": {
            "mean": float(np.mean(all_surprises)),
            "std": float(np.std(all_surprises)),
            "min": float(np.min(all_surprises)),
            "max": float(np.max(all_surprises)),
            "percentiles": {str(k): float(v) for k, v in surprise_percentiles.items()},
        },
        "per_game_stats": {
            "mean_of_means": float(np.mean(mean_surprises)),
            "std_of_means": float(np.std(mean_surprises)),
            "mean_max_surprise": float(np.mean(max_surprises)),
            "mean_high_surprise_moves": float(np.mean(high_surprise_counts)),
        },
        "high_surprise_games": {
            "count": len(high_surprise_games),
            "percentage": 100 * len(high_surprise_games) / len(game_stats),
            "threshold_used": surprise_threshold,
        },
        "titans_applicability": {
            "has_significant_surprise": np.mean(all_surprises) > 1.0,
            "high_surprise_moves_per_game": float(np.mean(high_surprise_counts)),
            "memory_benefit_score": min(100, 10 * np.mean(high_surprise_counts) + 20 * len(high_surprise_games) / len(game_stats)),
            "recommendation": "",
        }
    }

    # Add recommendation
    benefit_score = results["titans_applicability"]["memory_benefit_score"]
    if benefit_score > 50:
        results["titans_applicability"]["recommendation"] = "HIGH: Titans memory module would likely improve play significantly"
    elif benefit_score > 25:
        results["titans_applicability"]["recommendation"] = "MEDIUM: Titans could help with opponent modeling"
    else:
        results["titans_applicability"]["recommendation"] = "LOW: Current CNN+MCTS approach may be sufficient"

    return results


def print_results(results: Dict):
    """Pretty print analysis results."""
    print("\n" + "=" * 60)
    print("SURPRISE METRIC ANALYSIS FOR TITANS APPLICABILITY")
    print("=" * 60)

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print(f"\nDatabase: {results['database']}")
    print(f"Games analyzed: {results['games_analyzed']}")
    print(f"Total moves with probabilities: {results['total_moves_with_probs']}")

    print("\n--- Surprise Statistics ---")
    stats = results["surprise_stats"]
    print(f"  Mean surprise: {stats['mean']:.3f}")
    print(f"  Std deviation: {stats['std']:.3f}")
    print(f"  Min/Max: {stats['min']:.3f} / {stats['max']:.3f}")
    print("\n  Percentiles:")
    for p, v in stats["percentiles"].items():
        print(f"    {p}th: {v:.3f}")

    print("\n--- Per-Game Statistics ---")
    pg = results["per_game_stats"]
    print(f"  Mean of mean surprises: {pg['mean_of_means']:.3f}")
    print(f"  Mean max surprise per game: {pg['mean_max_surprise']:.3f}")
    print(f"  Mean high-surprise moves per game: {pg['mean_high_surprise_moves']:.1f}")

    print("\n--- High Surprise Games ---")
    hsg = results["high_surprise_games"]
    print(f"  Count: {hsg['count']} ({hsg['percentage']:.1f}%)")
    print(f"  Threshold: surprise > {hsg['threshold_used']}")

    print("\n--- TITANS APPLICABILITY ASSESSMENT ---")
    ta = results["titans_applicability"]
    print(f"  Has significant surprise: {ta['has_significant_surprise']}")
    print(f"  High-surprise moves per game: {ta['high_surprise_moves_per_game']:.1f}")
    print(f"  Memory benefit score: {ta['memory_benefit_score']:.1f}/100")
    print(f"\n  RECOMMENDATION: {ta['recommendation']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze surprise metrics for Titans applicability")
    parser.add_argument("--db", type=str, default="data/games/canonical_hex8_2p.db",
                        help="Path to game database")
    parser.add_argument("--sample-size", type=int, default=500,
                        help="Number of games to sample")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="Surprise threshold for 'high surprise' (default: 2.0)")
    parser.add_argument("--json", action="store_true",
                        help="Output raw JSON instead of formatted text")

    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"Error: Database not found: {args.db}")
        sys.exit(1)

    results = analyze_database(args.db, args.sample_size, args.threshold)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_results(results)


if __name__ == "__main__":
    main()
