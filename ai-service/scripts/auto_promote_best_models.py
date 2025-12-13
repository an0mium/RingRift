#!/usr/bin/env python
"""Auto-promote best Elo models to production ladder tiers.

This script queries the Elo leaderboard and automatically updates the
runtime ladder configuration to use the best-performing models for each
difficulty tier and board/player combination.

The promotion logic:
1. For each (board_type, num_players) combination:
   - Get the top-rated model from Elo leaderboard
   - For neural-network tiers (D6-D10), promote if model has sufficient games
   - Update runtime overrides in data/ladder_runtime_overrides.json

Usage:
    # Dry run (show what would be promoted)
    python scripts/auto_promote_best_models.py --dry-run

    # Actually promote
    python scripts/auto_promote_best_models.py --run

    # Promote for specific config only
    python scripts/auto_promote_best_models.py --run --board square8 --players 2

    # Require minimum games before promotion
    python scripts/auto_promote_best_models.py --run --min-games 50
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.models import BoardType
from app.config.ladder_config import (
    update_tier_model,
    get_effective_ladder_config,
    get_all_runtime_overrides,
    list_ladder_tiers,
    LadderTierConfig,
)

# Elo database path
ELO_DB_PATH = AI_SERVICE_ROOT / "data" / "elo_leaderboard.db"

# Minimum Elo games required for promotion
DEFAULT_MIN_GAMES = 20

# Difficulties that use neural network models (candidates for auto-promotion)
NN_DIFFICULTIES = [6, 7, 8, 9, 10]

# All board/player configurations
ALL_CONFIGS = [
    ("square8", 2),
    ("square8", 3),
    ("square8", 4),
    ("square19", 2),
    ("square19", 3),
    ("square19", 4),
    ("hexagonal", 2),
    ("hexagonal", 3),
    ("hexagonal", 4),
]

BOARD_TYPE_MAP = {
    "square8": BoardType.SQUARE8,
    "square19": BoardType.SQUARE19,
    "hexagonal": BoardType.HEXAGONAL,
}


@dataclass
class PromotionCandidate:
    """A model that could be promoted to a ladder tier."""
    model_id: str
    board_type: str
    num_players: int
    elo_rating: float
    games_played: int
    win_rate: float
    current_tier_model: Optional[str]
    should_promote: bool
    reason: str


def get_elo_leaderboard(
    board_type: str,
    num_players: int,
    limit: int = 10,
) -> List[Dict]:
    """Get top models from Elo leaderboard for a specific config."""
    if not ELO_DB_PATH.exists():
        return []

    conn = sqlite3.connect(ELO_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT e.model_id, e.rating, e.games_played, e.wins, e.losses, e.draws,
               m.model_path
        FROM elo_ratings e
        LEFT JOIN models m ON e.model_id = m.model_id
        WHERE e.board_type = ? AND e.num_players = ? AND e.games_played > 0
        ORDER BY e.rating DESC
        LIMIT ?
    """, (board_type, num_players, limit))

    results = []
    for row in cursor.fetchall():
        model_id, rating, games, wins, losses, draws, model_path = row
        win_rate = wins / games if games > 0 else 0.0
        results.append({
            "model_id": model_id,
            "rating": rating,
            "games_played": games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": win_rate,
            "model_path": model_path,
        })

    conn.close()
    return results


def get_current_tier_models(
    board_type: BoardType,
    num_players: int,
) -> Dict[int, str]:
    """Get currently configured models for each difficulty tier."""
    tier_models = {}

    for difficulty in NN_DIFFICULTIES:
        try:
            config = get_effective_ladder_config(difficulty, board_type, num_players)
            tier_models[difficulty] = config.model_id
        except KeyError:
            tier_models[difficulty] = None

    return tier_models


def evaluate_promotion(
    candidate: Dict,
    current_model: Optional[str],
    min_games: int,
) -> Tuple[bool, str]:
    """Evaluate whether a model should be promoted.

    Returns (should_promote, reason).
    """
    model_id = candidate["model_id"]
    games = candidate["games_played"]
    rating = candidate["rating"]

    # Check minimum games requirement
    if games < min_games:
        return False, f"Insufficient games ({games} < {min_games})"

    # If no current model, promote
    if current_model is None:
        return True, "No current model configured"

    # If same model, no need to promote
    if model_id == current_model:
        return False, "Already the current model"

    # Model is different and has enough games - promote
    return True, f"Higher Elo ({rating:.0f}) with {games} games"


def find_promotion_candidates(
    board_type_str: str,
    num_players: int,
    min_games: int,
) -> List[PromotionCandidate]:
    """Find models that should be promoted for a given config."""
    candidates = []

    board_type = BOARD_TYPE_MAP.get(board_type_str)
    if not board_type:
        return candidates

    # Get Elo leaderboard
    leaderboard = get_elo_leaderboard(board_type_str, num_players, limit=5)
    if not leaderboard:
        return candidates

    # Get best model (top of leaderboard)
    best_model = leaderboard[0]

    # Get current tier models
    current_models = get_current_tier_models(board_type, num_players)

    # Evaluate for each NN difficulty tier
    for difficulty in NN_DIFFICULTIES:
        current_model = current_models.get(difficulty)
        should_promote, reason = evaluate_promotion(
            best_model, current_model, min_games
        )

        candidates.append(PromotionCandidate(
            model_id=best_model["model_id"],
            board_type=board_type_str,
            num_players=num_players,
            elo_rating=best_model["rating"],
            games_played=best_model["games_played"],
            win_rate=best_model["win_rate"],
            current_tier_model=current_model,
            should_promote=should_promote,
            reason=reason,
        ))

    return candidates


def promote_model(
    model_id: str,
    board_type_str: str,
    num_players: int,
    difficulties: List[int],
) -> Dict[str, bool]:
    """Promote a model to specified difficulty tiers.

    Returns dict of difficulty -> success.
    """
    board_type = BOARD_TYPE_MAP.get(board_type_str)
    if not board_type:
        return {}

    results = {}
    for difficulty in difficulties:
        success = update_tier_model(
            difficulty=difficulty,
            board_type=board_type,
            num_players=num_players,
            model_id=model_id,
        )
        results[difficulty] = success

    return results


def run_auto_promotion(
    dry_run: bool = True,
    min_games: int = DEFAULT_MIN_GAMES,
    board_filter: Optional[str] = None,
    players_filter: Optional[int] = None,
) -> Dict:
    """Run automatic model promotion.

    Returns summary of promotions.
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "dry_run": dry_run,
        "min_games": min_games,
        "promotions": [],
        "skipped": [],
    }

    configs_to_check = ALL_CONFIGS
    if board_filter:
        configs_to_check = [(b, p) for b, p in configs_to_check if b == board_filter]
    if players_filter:
        configs_to_check = [(b, p) for b, p in configs_to_check if p == players_filter]

    print(f"\n{'='*70}")
    print(f" Auto-Promote Best Models {'(DRY RUN)' if dry_run else ''}")
    print(f"{'='*70}")
    print(f"Min games required: {min_games}")
    print(f"Checking {len(configs_to_check)} configurations\n")

    for board_type, num_players in configs_to_check:
        config_key = f"{board_type}_{num_players}p"
        print(f"\n--- {config_key} ---")

        # Get Elo leaderboard
        leaderboard = get_elo_leaderboard(board_type, num_players, limit=3)
        if not leaderboard:
            print(f"  No Elo data available")
            continue

        best = leaderboard[0]
        print(f"  Best model: {best['model_id'][:40]}")
        print(f"  Elo: {best['rating']:.0f} | Games: {best['games_played']} | Win%: {best['win_rate']:.1%}")

        # Check if meets promotion criteria
        if best['games_played'] < min_games:
            print(f"  SKIP: Insufficient games ({best['games_played']} < {min_games})")
            summary["skipped"].append({
                "config": config_key,
                "model": best["model_id"],
                "reason": f"Insufficient games ({best['games_played']} < {min_games})",
            })
            continue

        # Check current models for each tier
        board_type_enum = BOARD_TYPE_MAP[board_type]
        promotions_needed = []

        for difficulty in NN_DIFFICULTIES:
            try:
                current_config = get_effective_ladder_config(
                    difficulty, board_type_enum, num_players
                )
                current_model = current_config.model_id

                if current_model != best["model_id"]:
                    promotions_needed.append(difficulty)
                    print(f"  D{difficulty}: {current_model[:30]} -> {best['model_id'][:30]}")
                else:
                    print(f"  D{difficulty}: Already using best model")
            except KeyError:
                print(f"  D{difficulty}: No tier configured")

        if not promotions_needed:
            print(f"  No promotions needed")
            continue

        # Perform promotion
        if dry_run:
            print(f"  Would promote to: D{promotions_needed}")
            summary["promotions"].append({
                "config": config_key,
                "model": best["model_id"],
                "difficulties": promotions_needed,
                "dry_run": True,
            })
        else:
            results = promote_model(
                best["model_id"],
                board_type,
                num_players,
                promotions_needed,
            )
            successful = [d for d, ok in results.items() if ok]
            print(f"  PROMOTED to: D{successful}")
            summary["promotions"].append({
                "config": config_key,
                "model": best["model_id"],
                "difficulties": successful,
                "dry_run": False,
            })

    # Print summary
    print(f"\n{'='*70}")
    print(f" Summary")
    print(f"{'='*70}")
    print(f"Promotions: {len(summary['promotions'])}")
    print(f"Skipped: {len(summary['skipped'])}")

    if summary["promotions"] and not dry_run:
        print(f"\nRuntime overrides updated:")
        for key, overrides in get_all_runtime_overrides().items():
            print(f"  {key}: {overrides.get('model_id', 'N/A')}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Auto-promote best Elo models to production ladder"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be promoted without making changes",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually perform promotions",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=DEFAULT_MIN_GAMES,
        help=f"Minimum Elo games required for promotion (default: {DEFAULT_MIN_GAMES})",
    )
    parser.add_argument(
        "--board",
        type=str,
        choices=["square8", "square19", "hexagonal"],
        help="Only check specific board type",
    )
    parser.add_argument(
        "--players",
        type=int,
        choices=[2, 3, 4],
        help="Only check specific player count",
    )

    args = parser.parse_args()

    # Default to dry run if neither flag specified
    dry_run = not args.run

    summary = run_auto_promotion(
        dry_run=dry_run,
        min_games=args.min_games,
        board_filter=args.board,
        players_filter=args.players,
    )

    # Save summary to file
    summary_path = AI_SERVICE_ROOT / "data" / "auto_promotion_log.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Append to log
    log_entries = []
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                log_entries = json.load(f)
                if not isinstance(log_entries, list):
                    log_entries = [log_entries]
        except:
            log_entries = []

    log_entries.append(summary)

    # Keep last 100 entries
    log_entries = log_entries[-100:]

    with open(summary_path, "w") as f:
        json.dump(log_entries, f, indent=2)

    print(f"\nLog saved to: {summary_path}")


if __name__ == "__main__":
    main()
