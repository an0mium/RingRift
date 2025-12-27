#!/usr/bin/env python3
"""Selfplay campaign for hex8_2p to generate 50K high-quality games.

December 2025 Data Generation Campaign
======================================

Current state: hex8_2p has only ~980 games (vs 16K for square8_2p)
Target: 50,000 games with Gumbel MCTS quality budget (800 simulations)

This script:
1. Runs selfplay on available cluster nodes
2. Uses GUMBEL_BUDGET_QUALITY (800) for high-quality training data
3. Syncs games to coordinator for consolidated training
4. Tracks progress toward 50K game target

Usage:
    # Run locally (single node)
    python scripts/selfplay_campaign_hex8_2p.py --local --games 1000

    # Run on cluster (distributed)
    python scripts/selfplay_campaign_hex8_2p.py --cluster --games 50000

    # Check progress
    python scripts/selfplay_campaign_hex8_2p.py --status

    # Dry run
    python scripts/selfplay_campaign_hex8_2p.py --dry-run
"""

import argparse
import json
import logging
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/selfplay_hex8_2p_{datetime.now():%Y%m%d_%H%M%S}.log"),
    ],
)
logger = logging.getLogger(__name__)

# Paths
AI_SERVICE_ROOT = Path(__file__).parent.parent
DATA_DIR = AI_SERVICE_ROOT / "data"
GAMES_DIR = DATA_DIR / "games"

# Configuration
CONFIG_KEY = "hex8_2p"
BOARD_TYPE = "hex8"
NUM_PLAYERS = 2

# Campaign targets
TARGET_GAMES = 50000
GUMBEL_BUDGET = 800  # GUMBEL_BUDGET_QUALITY from gumbel_common.py

# Database paths (check multiple locations)
CANONICAL_DB = GAMES_DIR / "canonical_hex8_2p.db"
SELFPLAY_DB = GAMES_DIR / "hex8_2p_selfplay.db"
CAMPAIGN_DB = GAMES_DIR / "hex8_2p_campaign.db"


def get_game_count(db_path: Path) -> int:
    """Get number of games in a database."""
    if not db_path.exists():
        return 0

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM games")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        logger.warning(f"Could not read {db_path}: {e}")
        return 0


def get_total_games() -> tuple[int, dict[str, int]]:
    """Get total games across all hex8_2p databases."""
    counts = {}

    # Check canonical database
    if CANONICAL_DB.exists():
        counts["canonical"] = get_game_count(CANONICAL_DB)

    # Check selfplay database
    if SELFPLAY_DB.exists():
        counts["selfplay"] = get_game_count(SELFPLAY_DB)

    # Check campaign database
    if CAMPAIGN_DB.exists():
        counts["campaign"] = get_game_count(CAMPAIGN_DB)

    # Also check for any other hex8_2p databases
    for db_path in GAMES_DIR.glob("*hex8*2p*.db"):
        if db_path not in [CANONICAL_DB, SELFPLAY_DB, CAMPAIGN_DB]:
            name = db_path.stem
            counts[name] = get_game_count(db_path)

    total = sum(counts.values())
    return total, counts


def show_status():
    """Show current campaign status."""
    total, counts = get_total_games()

    logger.info("=" * 60)
    logger.info("Hex8 2P Selfplay Campaign Status")
    logger.info("=" * 60)
    logger.info(f"Target: {TARGET_GAMES:,} games")
    logger.info(f"Current: {total:,} games ({100 * total / TARGET_GAMES:.1f}%)")
    logger.info(f"Remaining: {max(0, TARGET_GAMES - total):,} games")
    logger.info("")
    logger.info("Games by database:")
    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {name}: {count:,}")

    if total >= TARGET_GAMES:
        logger.info("")
        logger.info("TARGET REACHED! Ready for training.")
    else:
        logger.info("")
        remaining = TARGET_GAMES - total
        # Estimate time at 50 games/hour/node
        est_hours = remaining / 50
        logger.info(f"Estimated time (1 node, 50 games/hr): {est_hours:.0f} hours")
        logger.info(f"Estimated time (10 nodes): {est_hours / 10:.0f} hours")


def run_local_selfplay(
    games: int,
    output_db: Path,
    dry_run: bool = False,
) -> bool:
    """Run selfplay locally."""
    logger.info("=" * 60)
    logger.info(f"Running Local Selfplay: {games} games")
    logger.info("=" * 60)
    logger.info(f"Board: {BOARD_TYPE}")
    logger.info(f"Players: {NUM_PLAYERS}")
    logger.info(f"Gumbel budget: {GUMBEL_BUDGET}")
    logger.info(f"Output: {output_db}")

    cmd = [
        sys.executable,
        "scripts/selfplay.py",
        "--board", BOARD_TYPE,
        "--num-players", str(NUM_PLAYERS),
        "--num-games", str(games),
        "--engine", "gumbel",
        "--gumbel-budget", str(GUMBEL_BUDGET),
        "--output-dir", str(output_db.parent),
        "--db-name", output_db.name,
    ]

    cmd_str = " ".join(str(c) for c in cmd)
    logger.info(f"Command: {cmd_str}")

    if dry_run:
        logger.info("[DRY RUN] Would execute above command")
        return True

    result = subprocess.run(cmd, cwd=AI_SERVICE_ROOT)
    return result.returncode == 0


def run_cluster_selfplay(
    games: int,
    dry_run: bool = False,
) -> bool:
    """Submit selfplay jobs to cluster via P2P orchestrator."""
    logger.info("=" * 60)
    logger.info(f"Running Cluster Selfplay: {games} games")
    logger.info("=" * 60)

    # Calculate games per node (assume 10-20 nodes available)
    games_per_node = max(100, games // 20)
    logger.info(f"Target: {games_per_node} games per node")

    # Use P2P orchestrator to dispatch selfplay
    cmd = [
        sys.executable,
        "-c",
        f"""
import asyncio
import aiohttp

async def dispatch_selfplay():
    async with aiohttp.ClientSession() as session:
        # Check P2P status first
        try:
            async with session.get("http://localhost:8770/status") as resp:
                status = await resp.json()
                print(f"P2P Status: {{status.get('role')}}, Alive peers: {{status.get('alive_peers', 0)}}")
        except Exception as e:
            print(f"P2P not available: {{e}}")
            return False

        # Dispatch selfplay job
        payload = {{
            "board_type": "{BOARD_TYPE}",
            "num_players": {NUM_PLAYERS},
            "num_games": {games_per_node},
            "engine": "gumbel",
            "gumbel_budget": {GUMBEL_BUDGET},
        }}

        try:
            async with session.post(
                "http://localhost:8770/dispatch_selfplay",
                json=payload
            ) as resp:
                result = await resp.json()
                print(f"Dispatch result: {{result}}")
                return result.get("success", False)
        except Exception as e:
            print(f"Dispatch failed: {{e}}")
            return False

asyncio.run(dispatch_selfplay())
""",
    ]

    cmd_str = "python -c '<dispatch script>'"
    logger.info(f"Command: {cmd_str}")

    if dry_run:
        logger.info("[DRY RUN] Would dispatch selfplay to cluster via P2P")
        return True

    result = subprocess.run(cmd, cwd=AI_SERVICE_ROOT, capture_output=True, text=True)
    logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Selfplay campaign for hex8_2p"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run selfplay locally (single node)",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Run selfplay on cluster via P2P",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1000,
        help="Number of games to generate (default: 1000)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show campaign status and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without executing",
    )
    args = parser.parse_args()

    # Ensure logs directory exists
    (AI_SERVICE_ROOT / "logs").mkdir(exist_ok=True)

    # Status only
    if args.status:
        show_status()
        return 0

    # Check current progress
    total, counts = get_total_games()
    logger.info(f"Current games: {total:,} / {TARGET_GAMES:,}")

    if total >= TARGET_GAMES:
        logger.info("TARGET ALREADY REACHED! No more selfplay needed.")
        show_status()
        return 0

    remaining = TARGET_GAMES - total
    games_to_run = min(args.games, remaining)
    logger.info(f"Games to generate: {games_to_run:,}")

    # Run selfplay
    if args.local:
        success = run_local_selfplay(
            games=games_to_run,
            output_db=CAMPAIGN_DB,
            dry_run=args.dry_run,
        )
    elif args.cluster:
        success = run_cluster_selfplay(
            games=games_to_run,
            dry_run=args.dry_run,
        )
    else:
        logger.error("Must specify --local or --cluster")
        parser.print_help()
        return 1

    if not success:
        logger.error("Selfplay failed")
        return 1

    # Show updated status
    show_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
