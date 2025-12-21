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
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.config.thresholds import (
    ELO_TIER_ADVANCED,
    ELO_TIER_EXPERT,
    ELO_TIER_GRANDMASTER,
    ELO_TIER_INTERMEDIATE,
    ELO_TIER_MASTER,
    ELO_TIER_NOVICE,
)
from scripts.lib.elo_queries import (
    DEFAULT_DB,
    PRODUCTION_ELO_THRESHOLD,
    PRODUCTION_MIN_GAMES,
    get_model_stats,
    get_models_by_tier,
    get_tier_abbr,
    get_tier_name,
    get_top_models,
)


def get_leaderboard(db_path: Path, config: str = None, include_baselines: bool = False,
                    tier: str = None, limit: int = 20) -> list[dict]:
    """Get leaderboard data using unified query library."""
    models = get_top_models(
        db_path,
        limit=limit,
        include_baselines=include_baselines,
        config=config,
        tier=tier,
    )

    results = []
    for i, model in enumerate(models, 1):
        results.append({
            "rank": i,
            "model_id": model.participant_id,
            "rating": round(model.rating, 1),
            "games": model.games_played,
            "board_type": model.board_type or "unknown",
            "num_players": model.num_players or 2,
            "tier": get_tier_name(model.rating),
            "tier_abbr": get_tier_abbr(model.rating),
            "production_ready": model.is_production_ready,
        })

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
    """Get summary statistics using unified query library."""
    stats = get_model_stats(db_path)
    if not stats:
        return {}

    # Get tier counts using unified query
    tier_counts = get_models_by_tier(db_path, include_baselines=False)

    return {
        "total_models": stats.total_models,
        "total_games": stats.total_games,
        "tier_counts": tier_counts,
        "production_ready": stats.production_ready,
        "best_model": stats.best_model,
        "best_rating": stats.best_rating,
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
