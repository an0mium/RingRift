#!/usr/bin/env python
"""Bulk canonical data generation for all 9 board/player configurations.

This script orchestrates canonical selfplay data generation across all
supported board types and player counts, with built-in balancing to
prioritize underrepresented configurations.

Features:
- Parallel generation across configurations
- Automatic balancing based on current game counts
- Progress tracking and status reporting
- Integration with parity validation
- Checkpoint/resume support

Usage:
    # Generate 1000 games per config (balancing automatically)
    python scripts/run_bulk_canonical_generation.py --games-per-config 1000

    # Focus on underrepresented configs only
    python scripts/run_bulk_canonical_generation.py --underrepresented-only

    # Quick demo run
    python scripts/run_bulk_canonical_generation.py --demo

The script generates databases in data/games/ with format:
    canonical_<board>_<players>p.db
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
if str(AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.training.crossboard_strength import ALL_BOARD_CONFIGS, config_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Default output directory
DEFAULT_GAMES_DIR = AI_SERVICE_ROOT / "data" / "games"

# Minimum games threshold for "well-represented" configs
MIN_GAMES_THRESHOLD = 1000


@dataclass
class ConfigStatus:
    """Status of a board/player configuration."""

    board: str
    num_players: int
    current_games: int
    target_games: int
    priority: float
    db_path: Path


def get_db_game_count(db_path: Path, num_players: int | None = None) -> int:
    """Get game count from a database."""
    if not db_path.exists():
        return 0

    try:
        conn = sqlite3.connect(str(db_path))
        if num_players is not None:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM games WHERE num_players = ?",
                (num_players,)
            )
        else:
            cursor = conn.execute("SELECT COUNT(*) FROM games")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def get_all_config_status(
    games_dir: Path,
    target_games: int,
) -> list[ConfigStatus]:
    """Get status for all 9 configurations."""
    statuses = []

    for board, num_players in ALL_BOARD_CONFIGS:
        # Build database path
        db_name = f"canonical_{board}_{num_players}p.db"
        db_path = games_dir / db_name

        # Get current count
        current = get_db_game_count(db_path, num_players)

        # Calculate priority (higher for underrepresented)
        if current == 0:
            priority = 1000.0
        else:
            priority = target_games / current

        statuses.append(ConfigStatus(
            board=board,
            num_players=num_players,
            current_games=current,
            target_games=target_games,
            priority=priority,
            db_path=db_path,
        ))

    # Sort by priority (highest first)
    statuses.sort(key=lambda s: s.priority, reverse=True)
    return statuses


def run_selfplay_for_config(
    board: str,
    num_players: int,
    num_games: int,
    db_path: Path,
    demo: bool = False,
    verify_parity: bool = True,
) -> dict[str, Any]:
    """Run selfplay for a single configuration.

    Returns a result dictionary with generation stats.
    """
    cfg_key = config_key(board, num_players)
    start_time = time.time()

    logger.info(f"Generating {num_games} games for {cfg_key}")

    # Ensure output directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Build command for generate_canonical_selfplay.py
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "generate_canonical_selfplay.py"),
        "--board-type", board,
        "--num-players", str(num_players),
        "--num-games", str(num_games),
        "--db", str(db_path),
    ]

    if demo:
        # Demo mode: fewer games, faster AI
        cmd.extend(["--demo-mode"])

    if not verify_parity:
        cmd.append("--skip-parity")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(AI_SERVICE_ROOT),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout per config
        )

        elapsed = time.time() - start_time
        games_after = get_db_game_count(db_path, num_players)

        return {
            "config_key": cfg_key,
            "board": board,
            "num_players": num_players,
            "games_generated": num_games,
            "games_in_db": games_after,
            "elapsed_seconds": elapsed,
            "success": result.returncode == 0,
            "error": result.stderr if result.returncode != 0 else None,
        }

    except subprocess.TimeoutExpired:
        return {
            "config_key": cfg_key,
            "board": board,
            "num_players": num_players,
            "games_generated": 0,
            "elapsed_seconds": 3600,
            "success": False,
            "error": "Timeout after 1 hour",
        }
    except Exception as e:
        return {
            "config_key": cfg_key,
            "board": board,
            "num_players": num_players,
            "games_generated": 0,
            "elapsed_seconds": time.time() - start_time,
            "success": False,
            "error": str(e),
        }


def run_bulk_generation(
    configs: list[ConfigStatus],
    games_per_config: int,
    parallel: int = 3,
    demo: bool = False,
    verify_parity: bool = True,
) -> list[dict[str, Any]]:
    """Run bulk data generation for multiple configurations."""
    results = []

    if parallel > 1:
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {}

            for status in configs:
                # Calculate games needed
                games_needed = max(0, games_per_config - status.current_games)
                if games_needed == 0:
                    logger.info(
                        f"Skipping {config_key(status.board, status.num_players)}: "
                        f"already has {status.current_games} games"
                    )
                    continue

                future = executor.submit(
                    run_selfplay_for_config,
                    status.board,
                    status.num_players,
                    games_needed,
                    status.db_path,
                    demo,
                    verify_parity,
                )
                futures[future] = config_key(status.board, status.num_players)

            for future in as_completed(futures):
                cfg_key = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    status_str = "OK" if result["success"] else "FAILED"
                    logger.info(
                        f"Completed {cfg_key}: {status_str} "
                        f"({result.get('games_in_db', 0)} total games)"
                    )
                except Exception as e:
                    logger.error(f"Exception for {cfg_key}: {e}")
                    results.append({
                        "config_key": cfg_key,
                        "success": False,
                        "error": str(e),
                    })
    else:
        for status in configs:
            games_needed = max(0, games_per_config - status.current_games)
            if games_needed == 0:
                continue

            result = run_selfplay_for_config(
                status.board,
                status.num_players,
                games_needed,
                status.db_path,
                demo,
                verify_parity,
            )
            results.append(result)

    return results


def generate_report(
    results: list[dict[str, Any]],
    configs: list[ConfigStatus],
    target_games: int,
) -> dict[str, Any]:
    """Generate summary report."""
    total_games_generated = sum(r.get("games_generated", 0) for r in results)
    successful = sum(1 for r in results if r.get("success", False))
    failed = len(results) - successful

    # Config status summary
    config_summary = {}
    for status in configs:
        cfg_key = config_key(status.board, status.num_players)
        config_summary[cfg_key] = {
            "current_games": status.current_games,
            "target_games": status.target_games,
            "progress": min(1.0, status.current_games / status.target_games) if status.target_games > 0 else 0.0,
        }

    # Find result for each config
    for result in results:
        cfg_key = result.get("config_key")
        if cfg_key and cfg_key in config_summary:
            config_summary[cfg_key]["games_in_db"] = result.get("games_in_db", 0)
            config_summary[cfg_key]["success"] = result.get("success", False)

    return {
        "total_configs": len(configs),
        "configs_processed": len(results),
        "successful": successful,
        "failed": failed,
        "total_games_generated": total_games_generated,
        "target_games_per_config": target_games,
        "config_status": config_summary,
        "results": results,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bulk canonical data generation for all 9 board/player configurations. "
            "Generates parity-verified selfplay games for training."
        ),
    )
    parser.add_argument(
        "--games-per-config",
        type=int,
        default=1000,
        help="Target games per configuration (default: 1000).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_GAMES_DIR),
        help="Output directory for game databases.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=3,
        help="Number of parallel config generations (default: 3).",
    )
    parser.add_argument(
        "--underrepresented-only",
        action="store_true",
        help="Only generate for configs with < 1000 games.",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        help="Specific configs to generate (e.g., 'square8_2p hexagonal_3p').",
    )
    parser.add_argument(
        "--skip-parity",
        action="store_true",
        help="Skip parity verification (faster but less reliable).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use lightweight demo mode (fewer games, faster AI).",
    )
    parser.add_argument(
        "--output-json",
        help="Path to write JSON report.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    games_dir = Path(args.output_dir)
    games_dir.mkdir(parents=True, exist_ok=True)

    # Adjust target for demo mode
    target_games = args.games_per_config
    if args.demo:
        target_games = min(target_games, 10)

    # Get all config statuses
    all_statuses = get_all_config_status(games_dir, target_games)

    # Filter configs
    if args.configs:
        # Parse specific configs
        config_keys = set(args.configs)
        statuses = [
            s for s in all_statuses
            if config_key(s.board, s.num_players) in config_keys
        ]
    elif args.underrepresented_only:
        statuses = [s for s in all_statuses if s.current_games < MIN_GAMES_THRESHOLD]
    else:
        statuses = all_statuses

    if not statuses:
        logger.info("No configurations need data generation")
        return 0

    logger.info(
        f"Starting bulk canonical generation: {len(statuses)} configs, "
        f"target={target_games} games/config"
    )

    # Print current status
    print("\n" + "=" * 60)
    print("CURRENT STATUS")
    print("=" * 60)
    for status in all_statuses:
        cfg_key = config_key(status.board, status.num_players)
        bar_len = int(min(1.0, status.current_games / target_games) * 20)
        bar = "#" * bar_len + "-" * (20 - bar_len)
        print(f"  {cfg_key:15s} [{bar}] {status.current_games:5d}/{target_games}")
    print("=" * 60 + "\n")

    # Run generation
    start_time = time.time()
    results = run_bulk_generation(
        configs=statuses,
        games_per_config=target_games,
        parallel=args.parallel,
        demo=args.demo,
        verify_parity=not args.skip_parity,
    )

    elapsed = time.time() - start_time

    # Generate report
    report = generate_report(results, all_statuses, target_games)
    report["elapsed_seconds"] = elapsed

    # Write report
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        logger.info(f"Report written to {args.output_json}")

    # Print summary
    print("\n" + "=" * 60)
    print("BULK GENERATION SUMMARY")
    print("=" * 60)
    print(f"Configs processed: {report['configs_processed']}")
    print(f"Successful: {report['successful']}")
    print(f"Failed: {report['failed']}")
    print(f"Total games generated: {report['total_games_generated']}")
    print(f"Elapsed: {elapsed:.1f}s")
    print("=" * 60)

    return 0 if report['failed'] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
