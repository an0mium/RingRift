#!/usr/bin/env python3
"""Check for models that meet production promotion criteria.

Production Criteria (from app/config/thresholds.py):
- ELO >= 1650
- Games played >= 100
- Win rate vs heuristic >= 60%
- Win rate vs random >= 90%

Usage:
    python scripts/check_production_candidates.py           # Check all
    python scripts/check_production_candidates.py --promote # Promote eligible
    python scripts/check_production_candidates.py --slack   # Send to Slack
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.config.thresholds import (
    PRODUCTION_MIN_WIN_RATE_VS_HEURISTIC,
    PRODUCTION_MIN_WIN_RATE_VS_RANDOM,
)
from scripts.lib.elo_queries import (
    DEFAULT_DB,
    PRODUCTION_ELO_THRESHOLD,
    PRODUCTION_MIN_GAMES,
    get_games_by_config,
    get_near_production,
    get_production_candidates as _get_production_candidates,
    get_tier_name,
)


def check_candidates(db_path: Path) -> list[dict]:
    """Find models meeting production criteria using unified query library."""
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return []

    models = _get_production_candidates(db_path, include_baselines=False)
    return [
        {
            "model_id": m.participant_id,
            "rating": m.rating,
            "games": m.games_played,
            "board_type": m.board_type or "unknown",
            "num_players": m.num_players or 2,
            "tier": get_tier_name(m.rating),
            "meets_elo": m.rating >= PRODUCTION_ELO_THRESHOLD,
            "meets_games": m.games_played >= PRODUCTION_MIN_GAMES,
        }
        for m in models
    ]


def get_near_candidates(db_path: Path) -> list[dict]:
    """Find models close to production threshold using unified query library."""
    return get_near_production(db_path, min_games=50, min_elo=1500, limit=10)


def get_config_coverage(db_path: Path) -> dict:
    """Get game count coverage by config using unified query library."""
    return get_games_by_config(db_path)


def main():
    parser = argparse.ArgumentParser(description="Check production promotion candidates")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to ELO database")
    parser.add_argument("--promote", action="store_true", help="Promote eligible models")
    parser.add_argument("--slack", action="store_true", help="Send report to Slack")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    print("=" * 70)
    print("PRODUCTION PROMOTION CANDIDATES REPORT")
    print("=" * 70)
    print(f"\nProduction Criteria:")
    print(f"  - ELO Rating: >= {PRODUCTION_ELO_THRESHOLD}")
    print(f"  - Games Played: >= {PRODUCTION_MIN_GAMES}")
    print(f"  - Win Rate vs Heuristic: >= {PRODUCTION_MIN_WIN_RATE_VS_HEURISTIC*100:.0f}%")
    print(f"  - Win Rate vs Random: >= {PRODUCTION_MIN_WIN_RATE_VS_RANDOM*100:.0f}%")

    # Check production-ready candidates
    candidates = check_candidates(args.db)

    print(f"\n{'='*70}")
    print("PRODUCTION-READY MODELS")
    print("=" * 70)

    if candidates:
        for c in candidates:
            print(f"\n  {c['model_id']}")
            print(f"    ELO: {c['rating']:.1f} ({c['tier']})")
            print(f"    Games: {c['games']}")
            print(f"    Config: {c['board_type']} {c['num_players']}p")
    else:
        print("\n  No models currently meet all production criteria.")

    # Check near-candidates
    near = get_near_candidates(args.db)

    print(f"\n{'='*70}")
    print("NEAR PRODUCTION (Close but not ready)")
    print("=" * 70)

    if near:
        for n in near[:5]:
            print(f"\n  {n['model_id']}")
            print(f"    ELO: {n['rating']:.1f} ({n['tier']})")
            print(f"    Games: {n['games']}")
            print(f"    Needs: +{n['elo_needed']:.1f} ELO, +{n['games_needed']} games")
    else:
        print("\n  No models close to production threshold.")

    # Config coverage
    coverage = get_config_coverage(args.db)

    print(f"\n{'='*70}")
    print("CONFIG COVERAGE (Games by board/player)")
    print("=" * 70)

    all_configs = [
        "square8_2p", "square8_3p", "square8_4p",
        "square19_2p", "square19_3p", "square19_4p",
        "hex8_2p", "hex8_3p", "hex8_4p",
        "hexagonal_2p", "hexagonal_3p", "hexagonal_4p",
    ]

    for config in all_configs:
        games = coverage.get(config, 0)
        bar_len = min(40, games // 100)
        bar = "#" * bar_len + "." * (40 - bar_len)
        status = "OK" if games >= 500 else "LOW" if games >= 100 else "NEED"
        print(f"  {config:15} [{bar}] {games:5} games ({status})")

    print(f"\n{'='*70}")

    # Summary
    total_production = len(candidates)
    total_near = len(near)

    print(f"\nSUMMARY:")
    print(f"  Production-ready models: {total_production}")
    print(f"  Near-production models: {total_near}")
    print(f"  Configs needing games: {sum(1 for c in all_configs if coverage.get(c, 0) < 100)}/12")


if __name__ == "__main__":
    main()
