#!/usr/bin/env python3
"""Run head-to-head tournaments between model generations.

This script runs tournaments between consecutive generations of trained models
to demonstrate progressive improvement. Results are stored in the
generation_tournaments table and statistical significance is calculated.

Usage:
    # Run tournaments for all configs with multiple generations
    python scripts/run_generation_tournaments.py

    # Run for a specific config
    python scripts/run_generation_tournaments.py --config hex8_2p

    # Specify number of games per tournament
    python scripts/run_generation_tournaments.py --games 100

    # Dry run (show what would be done)
    python scripts/run_generation_tournaments.py --dry-run

January 2026 - Created for demonstrating NN training improvement.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Add ai-service to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from app.coordination.generation_tracker import (
    GenerationInfo,
    GenerationTracker,
    get_generation_tracker,
)
from app.models import BoardType
from app.training.significance import wilson_score_interval
from app.training.tournament import run_tournament

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TournamentPair:
    """A pair of generations to compare in a tournament."""
    parent: GenerationInfo
    child: GenerationInfo
    config_key: str


@dataclass
class TournamentStats:
    """Statistics from a head-to-head tournament."""
    parent_id: int
    child_id: int
    child_wins: int
    parent_wins: int
    draws: int
    total_games: int
    win_rate: float
    ci_lower: float
    ci_upper: float
    is_significant: bool  # True if CI lower bound > 0.5


def find_tournament_pairs(tracker: GenerationTracker, config_key: str | None = None) -> list[TournamentPair]:
    """Find parent-child pairs that can be compared in tournaments.

    Args:
        tracker: The generation tracker instance.
        config_key: Optional filter for specific config (e.g., "hex8_2p").

    Returns:
        List of TournamentPair objects where both models exist.
    """
    pairs: list[TournamentPair] = []

    # Get all generations, optionally filtered
    board_type = None
    num_players = None
    if config_key:
        parts = config_key.rsplit("_", 1)
        if len(parts) == 2:
            board_type = parts[0]
            num_players = int(parts[1].rstrip("p"))

    generations = tracker.get_all_generations(
        board_type=board_type,
        num_players=num_players
    )

    # Build lookup by generation_id
    gen_by_id = {g.generation_id: g for g in generations}

    # Find pairs where child has a parent and both models exist
    for gen in generations:
        if gen.parent_generation is None:
            continue

        parent = gen_by_id.get(gen.parent_generation)
        if parent is None:
            continue

        # Check both model files exist
        parent_exists = parent.model_path and Path(parent.model_path).exists()
        child_exists = gen.model_path and Path(gen.model_path).exists()

        if parent_exists and child_exists:
            cfg_key = f"{gen.board_type}_{gen.num_players}p"
            pairs.append(TournamentPair(
                parent=parent,
                child=gen,
                config_key=cfg_key
            ))

    return pairs


def run_head_to_head(
    parent: GenerationInfo,
    child: GenerationInfo,
    num_games: int = 100,
) -> TournamentStats:
    """Run a head-to-head tournament between parent and child generations.

    Args:
        parent: The parent (older) generation.
        child: The child (newer) generation.
        num_games: Number of games to play.

    Returns:
        TournamentStats with results and statistical analysis.
    """
    # Map board_type string to BoardType enum
    board_type_map = {
        "hex8": BoardType.HEX8,
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hexagonal": BoardType.HEXAGONAL,
    }
    board_type = board_type_map.get(child.board_type, BoardType.HEX8)

    logger.info(
        f"Running tournament: Gen {child.generation_id} vs Gen {parent.generation_id} "
        f"({child.board_type}_{child.num_players}p, {num_games} games)"
    )

    # Run the tournament - child is model A (candidate), parent is model B (baseline)
    results = run_tournament(
        model_a_path=child.model_path,
        model_b_path=parent.model_path,
        num_games=num_games,
        board_type=board_type,
        num_players=child.num_players,
    )

    child_wins = results["model_a_wins"]
    parent_wins = results["model_b_wins"]
    draws = results["draws"]
    total = child_wins + parent_wins + draws

    win_rate = child_wins / total if total > 0 else 0.0
    ci_lower, ci_upper = wilson_score_interval(child_wins, total, confidence=0.95)

    # Significant improvement if lower bound > 50%
    is_significant = ci_lower > 0.5

    return TournamentStats(
        parent_id=parent.generation_id,
        child_id=child.generation_id,
        child_wins=child_wins,
        parent_wins=parent_wins,
        draws=draws,
        total_games=total,
        win_rate=win_rate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        is_significant=is_significant,
    )


def record_tournament_result(
    tracker: GenerationTracker,
    stats: TournamentStats,
) -> int:
    """Record tournament results in the database.

    Args:
        tracker: The generation tracker.
        stats: Tournament statistics to record.

    Returns:
        The tournament record ID.
    """
    return tracker.record_tournament(
        gen_a=stats.child_id,
        gen_b=stats.parent_id,
        gen_a_wins=stats.child_wins,
        gen_b_wins=stats.parent_wins,
        draws=stats.draws,
    )


def format_stats(stats: TournamentStats) -> str:
    """Format tournament stats for display."""
    sig_marker = "✓" if stats.is_significant else "✗"
    return (
        f"Gen {stats.child_id} vs Gen {stats.parent_id}: "
        f"{stats.child_wins}/{stats.total_games} wins ({stats.win_rate:.1%}) "
        f"Wilson 95% CI: [{stats.ci_lower:.2f}, {stats.ci_upper:.2f}] "
        f"Significant: {sig_marker}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run head-to-head tournaments between model generations."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Specific config to evaluate (e.g., hex8_2p). Default: all configs."
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games per tournament (default: 100)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without running tournaments."
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip pairs that already have tournament results."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/generation_tracking.db",
        help="Path to generation tracking database."
    )

    args = parser.parse_args()

    # Initialize tracker
    tracker = get_generation_tracker(args.db_path)

    # Find pairs to compare
    pairs = find_tournament_pairs(tracker, config_key=args.config)

    if not pairs:
        print("No tournament pairs found.")
        print("Pairs require:")
        print("  - Parent and child generations (parent_generation != NULL)")
        print("  - Both model files must exist")
        if args.config:
            print(f"  - Matching config: {args.config}")
        return 1

    print(f"Found {len(pairs)} tournament pair(s):\n")

    # Group by config for display
    by_config: dict[str, list[TournamentPair]] = {}
    for pair in pairs:
        if pair.config_key not in by_config:
            by_config[pair.config_key] = []
        by_config[pair.config_key].append(pair)

    for cfg, cfg_pairs in sorted(by_config.items()):
        print(f"=== {cfg} ===")
        for pair in cfg_pairs:
            print(f"  Gen {pair.child.generation_id} (child) vs Gen {pair.parent.generation_id} (parent)")
            print(f"    Child model: {pair.child.model_path}")
            print(f"    Parent model: {pair.parent.model_path}")
        print()

    if args.dry_run:
        print("Dry run - no tournaments will be run.")
        return 0

    # Run tournaments
    results: list[TournamentStats] = []

    for pair in pairs:
        # Check if we should skip existing
        if args.skip_existing:
            existing = tracker.get_tournaments_for_generation(pair.child.generation_id)
            already_compared = any(
                (t.gen_a == pair.child.generation_id and t.gen_b == pair.parent.generation_id) or
                (t.gen_a == pair.parent.generation_id and t.gen_b == pair.child.generation_id)
                for t in existing
            )
            if already_compared:
                logger.info(f"Skipping Gen {pair.child.generation_id} vs Gen {pair.parent.generation_id} (already compared)")
                continue

        try:
            stats = run_head_to_head(
                parent=pair.parent,
                child=pair.child,
                num_games=args.games,
            )

            # Record result
            record_tournament_result(tracker, stats)
            results.append(stats)

            print(f"\n{format_stats(stats)}")

        except Exception as e:
            logger.error(f"Failed tournament Gen {pair.child.generation_id} vs Gen {pair.parent.generation_id}: {e}")
            continue

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if not results:
        print("No tournaments completed.")
        return 1

    significant_count = sum(1 for s in results if s.is_significant)
    print(f"Total tournaments: {len(results)}")
    print(f"Statistically significant improvements: {significant_count}/{len(results)}")
    print()

    for stats in results:
        print(format_stats(stats))

    return 0


if __name__ == "__main__":
    sys.exit(main())
