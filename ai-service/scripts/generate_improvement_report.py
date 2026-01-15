#!/usr/bin/env python3
"""Generate improvement report for neural network training progress.

This script generates a comprehensive report showing progressive improvement
across model generations, including:
- Generation lineage per configuration
- Head-to-head tournament results
- Statistical significance analysis (Wilson CI)
- Elo progression over time

Usage:
    # Generate report for all configs
    python scripts/generate_improvement_report.py

    # Generate for specific config
    python scripts/generate_improvement_report.py --config hex8_2p

    # Output to specific file
    python scripts/generate_improvement_report.py --output reports/improvement.md

    # Generate JSON output
    python scripts/generate_improvement_report.py --format json

January 2026 - Created for demonstrating NN training improvement.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add ai-service to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from app.coordination.generation_tracker import (
    GenerationInfo,
    GenerationTracker,
    TournamentResult,
    get_generation_tracker,
)
from app.training.significance import wilson_score_interval

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    generation_id: int
    model_path: str
    version: int
    parent_id: int | None
    training_games: int
    training_samples: int
    elo: float | None
    elo_games: int


@dataclass
class TournamentAnalysis:
    """Analyzed tournament result with statistical significance."""
    child_id: int
    parent_id: int
    child_wins: int
    parent_wins: int
    draws: int
    total_games: int
    win_rate: float
    ci_lower: float
    ci_upper: float
    is_significant: bool
    timestamp: float


@dataclass
class ConfigReport:
    """Report for a single board configuration."""
    config_key: str
    board_type: str
    num_players: int
    generations: list[GenerationStats]
    tournaments: list[TournamentAnalysis]
    total_improvement: float | None
    is_improving: bool


@dataclass
class ImprovementReport:
    """Full improvement report across all configurations."""
    generated_at: str
    total_generations: int
    total_tournaments: int
    configs: list[ConfigReport]
    summary: dict[str, Any]


def get_elo_for_generation(tracker: GenerationTracker, generation_id: int) -> tuple[float | None, int]:
    """Get the latest Elo rating for a generation.

    Returns:
        Tuple of (elo_rating, games_played). Returns (None, 0) if no Elo data.
    """
    history = tracker.get_elo_history(generation_id)
    if history:
        latest = history[-1]
        return latest.elo, latest.games_played
    return None, 0


def analyze_tournament(result: TournamentResult) -> TournamentAnalysis:
    """Analyze a tournament result with Wilson CI."""
    total = result.gen_a_wins + result.gen_b_wins + result.draws
    win_rate = result.gen_a_wins / total if total > 0 else 0.0
    ci_lower, ci_upper = wilson_score_interval(result.gen_a_wins, total, confidence=0.95)

    return TournamentAnalysis(
        child_id=result.gen_a,  # gen_a is the child (newer generation)
        parent_id=result.gen_b,
        child_wins=result.gen_a_wins,
        parent_wins=result.gen_b_wins,
        draws=result.draws,
        total_games=total,
        win_rate=win_rate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        is_significant=ci_lower > 0.5,
        timestamp=result.timestamp,
    )


def build_config_report(
    tracker: GenerationTracker,
    board_type: str,
    num_players: int,
    generations: list[GenerationInfo]
) -> ConfigReport:
    """Build a report for a single configuration."""
    config_key = f"{board_type}_{num_players}p"

    # Build generation stats
    gen_stats: list[GenerationStats] = []
    for gen in generations:
        elo, elo_games = get_elo_for_generation(tracker, gen.generation_id)
        gen_stats.append(GenerationStats(
            generation_id=gen.generation_id,
            model_path=gen.model_path or "",
            version=gen.version,
            parent_id=gen.parent_generation,
            training_games=gen.training_games or 0,
            training_samples=gen.training_samples or 0,
            elo=elo,
            elo_games=elo_games,
        ))

    # Get all tournaments for this config
    tournament_analyses: list[TournamentAnalysis] = []
    seen_pairs: set[tuple[int, int]] = set()

    for gen in generations:
        results = tracker.get_tournaments_for_generation(gen.generation_id)
        for result in results:
            # Avoid duplicates
            pair = (min(result.gen_a, result.gen_b), max(result.gen_a, result.gen_b))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            # Only include if both generations are in this config
            gen_ids = {g.generation_id for g in generations}
            if result.gen_a in gen_ids and result.gen_b in gen_ids:
                tournament_analyses.append(analyze_tournament(result))

    # Sort tournaments by timestamp
    tournament_analyses.sort(key=lambda t: t.timestamp)

    # Calculate total improvement (first vs last with Elo data)
    elos_with_data = [(g.generation_id, g.elo) for g in gen_stats if g.elo is not None]
    if len(elos_with_data) >= 2:
        first_elo = elos_with_data[0][1]
        last_elo = elos_with_data[-1][1]
        total_improvement = last_elo - first_elo if first_elo and last_elo else None
    else:
        total_improvement = None

    # Determine if config is improving (majority of tournaments show improvement)
    significant_improvements = sum(1 for t in tournament_analyses if t.is_significant)
    is_improving = significant_improvements > len(tournament_analyses) // 2 if tournament_analyses else False

    return ConfigReport(
        config_key=config_key,
        board_type=board_type,
        num_players=num_players,
        generations=gen_stats,
        tournaments=tournament_analyses,
        total_improvement=total_improvement,
        is_improving=is_improving,
    )


def generate_report(tracker: GenerationTracker, config_filter: str | None = None) -> ImprovementReport:
    """Generate the full improvement report."""
    # Get all generations grouped by config
    all_generations = tracker.get_all_generations()

    by_config: dict[str, list[GenerationInfo]] = {}
    for gen in all_generations:
        config_key = f"{gen.board_type}_{gen.num_players}p"
        if config_filter and config_key != config_filter:
            continue
        if config_key not in by_config:
            by_config[config_key] = []
        by_config[config_key].append(gen)

    # Build reports per config
    config_reports: list[ConfigReport] = []
    for config_key, gens in sorted(by_config.items()):
        if len(gens) < 2:
            continue  # Skip configs with only one generation

        parts = config_key.rsplit("_", 1)
        board_type = parts[0]
        num_players = int(parts[1].rstrip("p"))

        report = build_config_report(tracker, board_type, num_players, gens)
        config_reports.append(report)

    # Calculate summary statistics
    total_tournaments = sum(len(r.tournaments) for r in config_reports)
    significant_tournaments = sum(
        sum(1 for t in r.tournaments if t.is_significant)
        for r in config_reports
    )
    improving_configs = sum(1 for r in config_reports if r.is_improving)

    summary = {
        "configs_analyzed": len(config_reports),
        "configs_improving": improving_configs,
        "total_tournaments": total_tournaments,
        "significant_improvements": significant_tournaments,
        "improvement_rate": significant_tournaments / total_tournaments if total_tournaments > 0 else 0.0,
    }

    return ImprovementReport(
        generated_at=datetime.now().isoformat(),
        total_generations=len(all_generations),
        total_tournaments=total_tournaments,
        configs=config_reports,
        summary=summary,
    )


def format_markdown(report: ImprovementReport) -> str:
    """Format the report as Markdown."""
    lines = [
        "# RingRift Neural Network Improvement Report",
        "",
        f"Generated: {report.generated_at}",
        "",
        "## Summary",
        "",
        f"- **Total Generations**: {report.total_generations}",
        f"- **Total Tournaments**: {report.total_tournaments}",
        f"- **Configs Analyzed**: {report.summary['configs_analyzed']}",
        f"- **Configs Showing Improvement**: {report.summary['configs_improving']}",
        f"- **Significant Improvements**: {report.summary['significant_improvements']}/{report.total_tournaments}",
        f"- **Improvement Rate**: {report.summary['improvement_rate']:.1%}",
        "",
    ]

    for config in report.configs:
        lines.append(f"## {config.config_key}")
        lines.append("")

        # Generation lineage
        lines.append("### Generations")
        lines.append("")
        lines.append("| Gen | Version | Parent | Training Games | Training Samples | Elo |")
        lines.append("|-----|---------|--------|----------------|------------------|-----|")

        for gen in config.generations:
            parent_str = str(gen.parent_id) if gen.parent_id else "-"
            elo_str = f"{gen.elo:.0f}" if gen.elo else "-"
            lines.append(
                f"| {gen.generation_id} | v{gen.version} | {parent_str} | "
                f"{gen.training_games:,} | {gen.training_samples:,} | {elo_str} |"
            )

        lines.append("")

        # Tournament results
        if config.tournaments:
            lines.append("### Head-to-Head Results")
            lines.append("")
            lines.append("| Child vs Parent | Wins | Win Rate | 95% CI | Significant |")
            lines.append("|-----------------|------|----------|--------|-------------|")

            for t in config.tournaments:
                sig = "✓" if t.is_significant else "✗"
                lines.append(
                    f"| Gen {t.child_id} vs Gen {t.parent_id} | "
                    f"{t.child_wins}/{t.total_games} | {t.win_rate:.1%} | "
                    f"[{t.ci_lower:.2f}, {t.ci_upper:.2f}] | {sig} |"
                )

            lines.append("")

        # Config summary
        status = "✓ Improving" if config.is_improving else "✗ Not yet improving"
        lines.append(f"**Status**: {status}")
        if config.total_improvement is not None:
            direction = "+" if config.total_improvement >= 0 else ""
            lines.append(f"**Total Elo Change**: {direction}{config.total_improvement:.0f}")
        lines.append("")

    # Overall conclusion
    lines.append("## Conclusion")
    lines.append("")

    if report.summary['improvement_rate'] >= 0.5:
        lines.append(
            f"The training pipeline is showing **positive improvement** with "
            f"{report.summary['significant_improvements']}/{report.total_tournaments} "
            f"({report.summary['improvement_rate']:.1%}) tournaments showing statistically "
            f"significant improvement of newer generations over older ones."
        )
    elif report.total_tournaments == 0:
        lines.append(
            "**No tournament data available.** Run `scripts/run_generation_tournaments.py` "
            "to generate head-to-head comparison data."
        )
    else:
        lines.append(
            f"The improvement rate ({report.summary['improvement_rate']:.1%}) is below 50%. "
            f"More training cycles may be needed to demonstrate consistent improvement."
        )

    lines.append("")
    return "\n".join(lines)


def format_json(report: ImprovementReport) -> str:
    """Format the report as JSON."""
    # Convert dataclasses to dicts recursively
    def to_dict(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {k: to_dict(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, list):
            return [to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        return obj

    return json.dumps(to_dict(report), indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate improvement report for neural network training."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Specific config to report on (e.g., hex8_2p). Default: all configs."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path. Default: stdout."
    )
    parser.add_argument(
        "--format", "-f",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)."
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

    # Generate report
    report = generate_report(tracker, config_filter=args.config)

    # Format output
    if args.format == "json":
        output = format_json(report)
    else:
        output = format_markdown(report)

    # Write output
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output)
        print(f"Report written to {args.output}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
