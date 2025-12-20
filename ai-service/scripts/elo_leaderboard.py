#!/usr/bin/env python3
"""ELO Leaderboard CLI.

View model rankings by config, tier, or overall. Supports filtering and sorting.

Usage:
    python scripts/elo_leaderboard.py                    # Top 20 overall
    python scripts/elo_leaderboard.py --config square8_2p  # Specific config
    python scripts/elo_leaderboard.py --tier expert      # By tier
    python scripts/elo_leaderboard.py --baselines        # Show baselines
    python scripts/elo_leaderboard.py --json             # JSON output
    python scripts/elo_leaderboard.py --watch            # Live updates
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.config.thresholds import (
    PRODUCTION_ELO_THRESHOLD,
    PRODUCTION_MIN_GAMES,
    ELO_TIER_NOVICE,
    ELO_TIER_INTERMEDIATE,
    ELO_TIER_ADVANCED,
    ELO_TIER_EXPERT,
    ELO_TIER_MASTER,
    ELO_TIER_GRANDMASTER,
)

DEFAULT_DB = AI_SERVICE_ROOT / "data" / "unified_elo.db"

TIER_THRESHOLDS = [
    (ELO_TIER_GRANDMASTER, "Grandmaster", "GM"),
    (ELO_TIER_MASTER, "Master", "M"),
    (ELO_TIER_EXPERT, "Expert", "E"),
    (ELO_TIER_ADVANCED, "Advanced", "A"),
    (ELO_TIER_INTERMEDIATE, "Intermediate", "I"),
    (ELO_TIER_NOVICE, "Novice", "N"),
    (0, "Beginner", "B"),
]


def get_tier(rating: float) -> tuple[str, str]:
    """Get tier name and abbreviation for rating."""
    for threshold, name, abbr in TIER_THRESHOLDS:
        if rating >= threshold:
            return name, abbr
    return "Beginner", "B"


def get_leaderboard(db_path: Path, config: str = None, include_baselines: bool = False,
                    tier: str = None, limit: int = 20) -> list[dict]:
    """Get leaderboard data."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))

    query = """
        SELECT participant_id, rating, games_played, board_type, num_players
        FROM elo_ratings
        WHERE 1=1
    """
    params = []

    if not include_baselines:
        query += " AND participant_id NOT LIKE 'baseline_%'"

    if config:
        parts = config.rsplit("_", 1)
        if len(parts) == 2:
            board_type = parts[0]
            num_players = int(parts[1].replace("p", ""))
            query += " AND board_type = ? AND num_players = ?"
            params.extend([board_type, num_players])

    if tier:
        tier_lower = tier.lower()
        for threshold, name, _abbr in TIER_THRESHOLDS:
            if name.lower() == tier_lower:
                next_threshold = 10000  # Very high default
                for i, (t, n, _) in enumerate(TIER_THRESHOLDS):
                    if n.lower() == tier_lower and i > 0:
                        next_threshold = TIER_THRESHOLDS[i - 1][0]
                        break
                query += " AND rating >= ? AND rating < ?"
                params.extend([threshold, next_threshold])
                break

    query += " ORDER BY rating DESC LIMIT ?"
    params.append(limit)

    cursor = conn.execute(query, params)

    results = []
    for i, row in enumerate(cursor.fetchall(), 1):
        model_id, rating, games, board_type, num_players = row
        tier_name, tier_abbr = get_tier(rating)

        # Check production eligibility
        production_ready = rating >= PRODUCTION_ELO_THRESHOLD and games >= PRODUCTION_MIN_GAMES

        results.append({
            "rank": i,
            "model_id": model_id,
            "rating": round(rating, 1),
            "games": games,
            "board_type": board_type or "unknown",
            "num_players": num_players or 2,
            "tier": tier_name,
            "tier_abbr": tier_abbr,
            "production_ready": production_ready,
        })

    conn.close()
    return results


def format_leaderboard(data: list[dict], title: str = "ELO LEADERBOARD") -> str:
    """Format leaderboard as table."""
    lines = []
    lines.append("=" * 90)
    lines.append(f"{title:^90}")
    lines.append("=" * 90)
    lines.append("")

    if not data:
        lines.append("  No models found.")
        return "\n".join(lines)

    # Header
    header = f"{'#':>3}  {'Model ID':<45} {'ELO':>7} {'Games':>6} {'Tier':>6} {'Status':>8}"
    lines.append(header)
    lines.append("-" * 90)

    for entry in data:
        status = "PROD" if entry["production_ready"] else ""
        if entry["model_id"].startswith("baseline_"):
            status = "BASE"

        line = f"{entry['rank']:>3}  {entry['model_id'][:45]:<45} {entry['rating']:>7.1f} {entry['games']:>6} {entry['tier_abbr']:>6} {status:>8}"
        lines.append(line)

    lines.append("")
    lines.append("=" * 90)
    lines.append(f"Tiers: GM={ELO_TIER_GRANDMASTER}, M={ELO_TIER_MASTER}, E={ELO_TIER_EXPERT} (Prod), A={ELO_TIER_ADVANCED}, I={ELO_TIER_INTERMEDIATE}, N={ELO_TIER_NOVICE}")
    lines.append(f"Production: ELO >= {PRODUCTION_ELO_THRESHOLD}, Games >= {PRODUCTION_MIN_GAMES}")

    return "\n".join(lines)


def get_summary(db_path: Path) -> dict:
    """Get summary statistics."""
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(str(db_path))

    # Total models and games
    cursor = conn.execute("SELECT COUNT(*), SUM(games_played) FROM elo_ratings WHERE participant_id NOT LIKE 'baseline_%'")
    total_models, total_games = cursor.fetchone()

    # Models by tier
    tier_counts = {}
    for threshold, name, _abbr in TIER_THRESHOLDS:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM elo_ratings WHERE rating >= ? AND participant_id NOT LIKE 'baseline_%'",
            (threshold,)
        )
        count = cursor.fetchone()[0]
        tier_counts[name] = count

    # Production ready
    cursor = conn.execute(
        "SELECT COUNT(*) FROM elo_ratings WHERE rating >= ? AND games_played >= ? AND participant_id NOT LIKE 'baseline_%'",
        (PRODUCTION_ELO_THRESHOLD, PRODUCTION_MIN_GAMES)
    )
    production_ready = cursor.fetchone()[0]

    # Best model
    cursor = conn.execute(
        "SELECT participant_id, rating FROM elo_ratings WHERE participant_id NOT LIKE 'baseline_%' ORDER BY rating DESC LIMIT 1"
    )
    best = cursor.fetchone()

    conn.close()

    return {
        "total_models": total_models or 0,
        "total_games": total_games or 0,
        "tier_counts": tier_counts,
        "production_ready": production_ready,
        "best_model": best[0] if best else None,
        "best_rating": best[1] if best else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="ELO Leaderboard CLI")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to ELO database")
    parser.add_argument("--config", type=str, help="Filter by config (e.g., square8_2p)")
    parser.add_argument("--tier", type=str, help="Filter by tier (e.g., expert, master)")
    parser.add_argument("--baselines", action="store_true", help="Include baseline models")
    parser.add_argument("--limit", type=int, default=20, help="Number of models to show")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--summary", action="store_true", help="Show summary only")
    parser.add_argument("--watch", action="store_true", help="Live updates (refresh every 30s)")

    args = parser.parse_args()

    def show():
        if args.summary:
            summary = get_summary(args.db)
            if args.json:
                print(json.dumps(summary, indent=2))
            else:
                print("\n" + "=" * 50)
                print("ELO DATABASE SUMMARY")
                print("=" * 50)
                print(f"  Total Models: {summary['total_models']}")
                print(f"  Total Games: {summary['total_games']}")
                print(f"  Production Ready: {summary['production_ready']}")
                print(f"  Best Model: {summary['best_model']} ({summary['best_rating']:.1f})")
                print("")
                print("  Models by Tier:")
                for tier, count in summary.get('tier_counts', {}).items():
                    print(f"    {tier}: {count}")
                print("=" * 50)
            return

        title = "ELO LEADERBOARD"
        if args.config:
            title = f"ELO LEADERBOARD - {args.config}"
        elif args.tier:
            title = f"ELO LEADERBOARD - {args.tier.title()} Tier"

        data = get_leaderboard(
            args.db,
            config=args.config,
            include_baselines=args.baselines,
            tier=args.tier,
            limit=args.limit,
        )

        if args.json:
            print(json.dumps(data, indent=2))
        else:
            print(format_leaderboard(data, title))

    if args.watch:
        try:
            while True:
                print("\033[2J\033[H")  # Clear screen
                show()
                print(f"\n  Last updated: {time.strftime('%H:%M:%S')} (Ctrl+C to exit)")
                time.sleep(30)
        except KeyboardInterrupt:
            print("\nExiting...")
    else:
        show()


if __name__ == "__main__":
    main()
