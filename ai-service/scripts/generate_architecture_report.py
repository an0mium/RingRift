#!/usr/bin/env python3
"""Generate reports from architecture comparison results.

Reads comparison JSON output from compare_architectures.py and generates
formatted reports in multiple formats (console, JSON, Markdown).

Usage:
    # Generate console report (default)
    python scripts/generate_architecture_report.py \
        --input data/architecture_comparison_hex8_2p.json

    # Generate markdown report
    python scripts/generate_architecture_report.py \
        --input data/architecture_comparison_hex8_2p.json \
        --format markdown \
        --output docs/ARCHITECTURE_COMPARISON.md

    # Generate JSON summary
    python scripts/generate_architecture_report.py \
        --input data/architecture_comparison_hex8_2p.json \
        --format json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ArchitectureRanking:
    """Ranking entry for a single architecture."""

    rank: int
    architecture: str
    elo: float
    elo_ci_low: float
    elo_ci_high: float
    total_games: int
    win_rate: float


@dataclass
class HeadToHeadResult:
    """Head-to-head matchup result."""

    arch_a: str
    arch_b: str
    wins_a: int
    wins_b: int
    draws: int
    win_rate_a: float
    elo_diff: float


def compute_confidence_interval(
    games: int,
    elo: float,
    base_elo: float = 1500.0,
    k_factor: float = 32.0,
) -> tuple[float, float]:
    """Compute 95% confidence interval for Elo rating.

    Uses the approximation that Elo variance decreases with sqrt(games).
    """
    if games <= 0:
        return (elo - 200, elo + 200)

    # Standard error of Elo rating approximation
    # Assuming average game result variance of ~0.25
    se = k_factor * 0.5 / math.sqrt(games)

    # 95% CI uses 1.96 standard errors
    margin = 1.96 * se * math.sqrt(games / max(games, 10))

    # Scale margin based on distance from base Elo
    margin = max(margin, 50)  # Minimum margin of 50

    return (elo - margin, elo + margin)


def load_comparison_results(input_path: Path) -> dict[str, Any]:
    """Load comparison results from JSON file."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path) as f:
        return json.load(f)


def extract_rankings(results: dict[str, Any]) -> list[ArchitectureRanking]:
    """Extract and sort architecture rankings from comparison results."""
    elo_ratings = results.get("elo_ratings", {})
    matchups = results.get("matchups", [])

    # Calculate total games per architecture
    games_per_arch: dict[str, int] = {}
    wins_per_arch: dict[str, int] = {}

    for matchup in matchups:
        arch_a = matchup["arch_a"]
        arch_b = matchup["arch_b"]
        games = matchup.get("games_played", 0)

        games_per_arch[arch_a] = games_per_arch.get(arch_a, 0) + games
        games_per_arch[arch_b] = games_per_arch.get(arch_b, 0) + games

        wins_per_arch[arch_a] = wins_per_arch.get(arch_a, 0) + matchup.get("wins_a", 0)
        wins_per_arch[arch_b] = wins_per_arch.get(arch_b, 0) + matchup.get("wins_b", 0)

    # Build rankings
    rankings = []
    for arch, elo in elo_ratings.items():
        games = games_per_arch.get(arch, 0)
        wins = wins_per_arch.get(arch, 0)
        win_rate = wins / games if games > 0 else 0.0
        ci_low, ci_high = compute_confidence_interval(games, elo)

        rankings.append(
            ArchitectureRanking(
                rank=0,  # Will be set after sorting
                architecture=arch,
                elo=elo,
                elo_ci_low=ci_low,
                elo_ci_high=ci_high,
                total_games=games,
                win_rate=win_rate,
            )
        )

    # Sort by Elo descending
    rankings.sort(key=lambda r: r.elo, reverse=True)

    # Assign ranks
    for i, ranking in enumerate(rankings):
        ranking.rank = i + 1

    return rankings


def extract_head_to_head(results: dict[str, Any]) -> list[HeadToHeadResult]:
    """Extract head-to-head matchup results."""
    matchups = results.get("matchups", [])

    h2h_results = []
    for matchup in matchups:
        h2h_results.append(
            HeadToHeadResult(
                arch_a=matchup["arch_a"],
                arch_b=matchup["arch_b"],
                wins_a=matchup.get("wins_a", 0),
                wins_b=matchup.get("wins_b", 0),
                draws=matchup.get("draws", 0),
                win_rate_a=matchup.get("win_rate_a", 0.0),
                elo_diff=matchup.get("elo_diff", 0.0),
            )
        )

    return h2h_results


def format_console_report(results: dict[str, Any]) -> str:
    """Generate ASCII console report."""
    config = results.get("config", {})
    rankings = extract_rankings(results)
    h2h = extract_head_to_head(results)
    timestamp = results.get("timestamp", 0)

    lines = []

    # Header
    lines.append("=" * 70)
    lines.append("Architecture Comparison Report")
    lines.append(f"Board: {config.get('board_type', 'unknown')}, Players: {config.get('num_players', 0)}")
    lines.append(f"Harness: {config.get('harness', 'unknown')}")
    lines.append(f"Games per matchup: {config.get('games_per_matchup', 0)}")
    if timestamp:
        lines.append(f"Generated: {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")

    # Rankings table
    lines.append("## Elo Rankings")
    lines.append("")
    lines.append(f"{'Rank':<6}{'Architecture':<15}{'Elo':<10}{'95% CI':<20}{'Games':<8}{'Win %':<8}")
    lines.append("-" * 70)

    for r in rankings:
        ci_str = f"({r.elo_ci_low:.0f}-{r.elo_ci_high:.0f})"
        lines.append(
            f"{r.rank:<6}{r.architecture:<15}{r.elo:<10.0f}{ci_str:<20}{r.total_games:<8}{r.win_rate * 100:<.1f}%"
        )

    lines.append("")

    # Head-to-head table
    if h2h:
        lines.append("## Head-to-Head Results")
        lines.append("")
        lines.append(f"{'Matchup':<25}{'Result':<15}{'Win %':<10}{'Elo Diff':<12}")
        lines.append("-" * 70)

        for m in h2h:
            matchup_str = f"{m.arch_a} vs {m.arch_b}"
            result_str = f"{m.wins_a}-{m.wins_b}-{m.draws}"
            lines.append(
                f"{matchup_str:<25}{result_str:<15}{m.win_rate_a * 100:<10.1f}{m.elo_diff:<+12.0f}"
            )

        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def format_markdown_report(results: dict[str, Any]) -> str:
    """Generate Markdown report."""
    config = results.get("config", {})
    rankings = extract_rankings(results)
    h2h = extract_head_to_head(results)
    timestamp = results.get("timestamp", 0)

    lines = []

    # Header
    lines.append("# Architecture Comparison Report")
    lines.append("")
    lines.append(f"**Board Type:** {config.get('board_type', 'unknown')}")
    lines.append(f"**Players:** {config.get('num_players', 0)}")
    lines.append(f"**Harness:** {config.get('harness', 'unknown')}")
    lines.append(f"**Games per Matchup:** {config.get('games_per_matchup', 0)}")
    if timestamp:
        lines.append(f"**Generated:** {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Rankings table
    lines.append("## Elo Rankings")
    lines.append("")
    lines.append("| Rank | Architecture | Elo | 95% CI | Games | Win % |")
    lines.append("|------|--------------|-----|--------|-------|-------|")

    for r in rankings:
        ci_str = f"({r.elo_ci_low:.0f}-{r.elo_ci_high:.0f})"
        lines.append(
            f"| {r.rank} | {r.architecture} | {r.elo:.0f} | {ci_str} | {r.total_games} | {r.win_rate * 100:.1f}% |"
        )

    lines.append("")

    # Head-to-head table
    if h2h:
        lines.append("## Head-to-Head Results")
        lines.append("")
        lines.append("| Matchup | Result | Win % | Elo Diff |")
        lines.append("|---------|--------|-------|----------|")

        for m in h2h:
            matchup_str = f"{m.arch_a} vs {m.arch_b}"
            result_str = f"{m.wins_a}-{m.wins_b}-{m.draws}"
            lines.append(f"| {matchup_str} | {result_str} | {m.win_rate_a * 100:.1f}% | {m.elo_diff:+.0f} |")

        lines.append("")

    # Interpretation guide
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- **Elo ratings** are calculated using the Bradley-Terry model")
    lines.append("- **95% CI** (Confidence Interval) shows the estimated uncertainty range")
    lines.append("- **Win %** shows overall win percentage across all matchups")
    lines.append("- **Elo Diff** shows the Elo difference between architectures in a matchup")
    lines.append("")
    lines.append("### Elo Difference Guide")
    lines.append("")
    lines.append("| Elo Diff | Expected Win Rate |")
    lines.append("|----------|-------------------|")
    lines.append("| +100 | ~64% |")
    lines.append("| +200 | ~76% |")
    lines.append("| +300 | ~85% |")
    lines.append("| +400 | ~91% |")
    lines.append("")

    return "\n".join(lines)


def format_json_summary(results: dict[str, Any]) -> str:
    """Generate JSON summary report."""
    rankings = extract_rankings(results)
    h2h = extract_head_to_head(results)

    summary = {
        "config": results.get("config", {}),
        "rankings": [
            {
                "rank": r.rank,
                "architecture": r.architecture,
                "elo": round(r.elo, 1),
                "elo_ci_low": round(r.elo_ci_low, 1),
                "elo_ci_high": round(r.elo_ci_high, 1),
                "total_games": r.total_games,
                "win_rate": round(r.win_rate, 3),
            }
            for r in rankings
        ],
        "head_to_head": [
            {
                "matchup": f"{m.arch_a}_vs_{m.arch_b}",
                "winner": m.arch_a if m.wins_a > m.wins_b else (m.arch_b if m.wins_b > m.wins_a else "draw"),
                "score": f"{m.wins_a}-{m.wins_b}-{m.draws}",
                "elo_diff": round(m.elo_diff, 1),
            }
            for m in h2h
        ],
        "timestamp": results.get("timestamp"),
    }

    return json.dumps(summary, indent=2)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate reports from architecture comparison results"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to comparison results JSON file",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["console", "markdown", "json"],
        default="console",
        help="Output format (default: console)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (prints to stdout if not specified)",
    )

    args = parser.parse_args()

    try:
        # Load results
        input_path = Path(args.input)
        results = load_comparison_results(input_path)

        # Generate report
        if args.format == "console":
            report = format_console_report(results)
        elif args.format == "markdown":
            report = format_markdown_report(results)
        elif args.format == "json":
            report = format_json_summary(results)
        else:
            raise ValueError(f"Unknown format: {args.format}")

        # Output
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        else:
            print(report)

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in input file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
