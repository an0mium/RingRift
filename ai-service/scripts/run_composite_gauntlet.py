#!/usr/bin/env python3
"""Run Composite Gauntlet - Two-phase evaluation for (NN, Algorithm) combinations.

Usage:
    # Two-phase gauntlet on all models in directory
    python scripts/run_composite_gauntlet.py --model-dir models/ --board square8 --players 2

    # Algorithm tournament with fixed NN
    python scripts/run_composite_gauntlet.py --algorithm-tournament models/best_model.pth

    # NN tournament with fixed algorithm
    python scripts/run_composite_gauntlet.py --nn-tournament models/*.pth --algorithm gumbel_mcts

    # Show aggregation report only
    python scripts/run_composite_gauntlet.py --report
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.tournament.composite_gauntlet import (
    CompositeGauntlet,
    CompositeGauntletConfig,
    GauntletPhaseConfig,
    run_two_phase_gauntlet,
    run_algorithm_tournament,
    run_nn_tournament,
)
from app.tournament.gauntlet_aggregation import (
    aggregate_by_nn,
    aggregate_by_algorithm,
    check_nn_ranking_consistency,
    print_aggregation_report,
    update_algorithm_baselines,
    update_nn_performance_summaries,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_models(model_dir: Path, pattern: str = "*.pth") -> list[Path]:
    """Find model files in directory."""
    models = list(model_dir.glob(pattern))
    # Filter out archived models
    models = [m for m in models if "archived" not in str(m)]
    return sorted(models)


async def main():
    parser = argparse.ArgumentParser(
        description="Run Composite Gauntlet for (NN, Algorithm) evaluation"
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--two-phase",
        action="store_true",
        help="Run two-phase gauntlet (default mode)"
    )
    mode_group.add_argument(
        "--algorithm-tournament",
        type=str,
        metavar="NN_PATH",
        help="Run algorithm tournament with fixed NN"
    )
    mode_group.add_argument(
        "--nn-tournament",
        action="store_true",
        help="Run NN tournament with fixed algorithm"
    )
    mode_group.add_argument(
        "--report",
        action="store_true",
        help="Show aggregation report only"
    )
    mode_group.add_argument(
        "--update-summaries",
        action="store_true",
        help="Update NN performance summaries"
    )

    # Model selection
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing model files"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Specific model paths"
    )

    # Game configuration
    parser.add_argument(
        "--board",
        type=str,
        default="square8",
        choices=["square8", "square19", "hexagonal"],
        help="Board type"
    )
    parser.add_argument(
        "--players",
        type=int,
        default=2,
        help="Number of players"
    )

    # Algorithm selection
    parser.add_argument(
        "--algorithm",
        type=str,
        default="gumbel_mcts",
        help="Reference algorithm for NN tournament"
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=["gumbel_mcts", "mcts", "descent"],
        help="Algorithms to test in Phase 2"
    )

    # Game counts
    parser.add_argument(
        "--phase1-games",
        type=int,
        default=50,
        help="Games per baseline in Phase 1"
    )
    parser.add_argument(
        "--phase2-games",
        type=int,
        default=20,
        help="Games per baseline in Phase 2"
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=0.5,
        help="Fraction of NNs to pass from Phase 1 to Phase 2"
    )

    # Other options
    parser.add_argument(
        "--min-games",
        type=int,
        default=5,
        help="Minimum games for report inclusion"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum models to evaluate"
    )

    args = parser.parse_args()

    # Handle report mode
    if args.report:
        print_aggregation_report(
            board_type=args.board,
            num_players=args.players,
            min_games=args.min_games,
        )
        return

    # Handle update summaries mode
    if args.update_summaries:
        print("Updating NN performance summaries...")
        count = update_nn_performance_summaries(
            board_type=args.board,
            num_players=args.players,
            min_games=args.min_games,
        )
        print(f"Updated {count} NN summaries")

        print("\nUpdating algorithm baselines...")
        baselines = update_algorithm_baselines(
            board_type=args.board,
            num_players=args.players,
            min_games=args.min_games * 2,
        )
        for algo, elo in baselines.items():
            print(f"  {algo}: {elo:.0f}")

        return

    # Get model paths
    if args.models:
        model_paths = [Path(p) for p in args.models]
    elif args.algorithm_tournament:
        model_paths = [Path(args.algorithm_tournament)]
    else:
        model_paths = find_models(args.model_dir)[:args.limit]

    if not model_paths:
        print("No models found")
        return

    print(f"Found {len(model_paths)} models")

    # Handle algorithm tournament
    if args.algorithm_tournament:
        print(f"\nRunning Algorithm Tournament with {args.algorithm_tournament}")
        print(f"Algorithms: {', '.join(args.algorithms)}")

        results = await run_algorithm_tournament(
            reference_nn=model_paths[0],
            algorithms=args.algorithms,
            board_type=args.board,
            num_players=args.players,
            games_per_algorithm=args.phase2_games * 2,
        )

        print("\nAlgorithm Rankings:")
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for rank, (algo, elo) in enumerate(sorted_results, 1):
            print(f"  {rank}. {algo}: {elo:.0f} Elo")

        return

    # Handle NN tournament
    if args.nn_tournament:
        print(f"\nRunning NN Tournament with {args.algorithm}")
        print(f"NNs: {len(model_paths)}")

        results = await run_nn_tournament(
            nn_paths=model_paths,
            reference_algorithm=args.algorithm,
            board_type=args.board,
            num_players=args.players,
            games_per_nn=args.phase2_games * 2,
        )

        print("\nNN Rankings:")
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for rank, (nn_id, elo) in enumerate(sorted_results[:20], 1):
            nn_short = nn_id[:40] + "..." if len(nn_id) > 40 else nn_id
            print(f"  {rank}. {nn_short}: {elo:.0f} Elo")

        return

    # Run two-phase gauntlet
    print(f"\nRunning Two-Phase Gauntlet")
    print(f"Phase 1: {args.phase1_games} games × policy_only")
    print(f"Phase 2: {args.phase2_games} games × {', '.join(args.algorithms)}")
    print(f"Pass threshold: {args.pass_threshold:.0%}")

    config = CompositeGauntletConfig(
        phase1=GauntletPhaseConfig(
            games_per_matchup=args.phase1_games,
            pass_threshold=args.pass_threshold,
        ),
        phase2=GauntletPhaseConfig(
            games_per_matchup=args.phase2_games,
        ),
        phase2_algorithms=args.algorithms,
    )

    gauntlet = CompositeGauntlet(
        board_type=args.board,
        num_players=args.players,
        config=config,
    )

    result = await gauntlet.run_two_phase_gauntlet(model_paths)

    # Print results
    print(f"\n{'='*60}")
    print(f"Gauntlet {result.run_id} Complete")
    print(f"{'='*60}")
    print(f"Status: {result.status}")
    print(f"Total games: {result.total_games}")
    duration = (result.completed_at or time.time()) - result.started_at
    print(f"Duration: {duration:.1f}s")

    if result.phase1_result:
        p1 = result.phase1_result
        print(f"\nPhase 1:")
        print(f"  NNs tested: {len(p1.nn_ids)}")
        print(f"  NNs passed: {len(p1.passed_nn_ids)}")
        print(f"  Games played: {p1.games_played}")

    if result.phase2_result:
        p2 = result.phase2_result
        print(f"\nPhase 2:")
        print(f"  NNs tested: {len(p2.nn_ids)}")
        print(f"  Algorithms: {', '.join(args.algorithms)}")
        print(f"  Games played: {p2.games_played}")

    if result.final_rankings:
        print(f"\nTop 10 Final Rankings:")
        for entry in result.final_rankings[:10]:
            pid = entry.get("participant_id", "")
            nn = entry.get("nn_model_id", "")[:30]
            algo = entry.get("ai_algorithm", "")
            elo = entry.get("rating", 1500)
            games = entry.get("games_played", 0)
            print(f"  {entry.get('rank', '?'):2}. {nn}:{algo} - {elo:.0f} ({games} games)")

    # Show aggregation report
    print("\n")
    print_aggregation_report(
        board_type=args.board,
        num_players=args.players,
        min_games=args.min_games,
    )


if __name__ == "__main__":
    asyncio.run(main())
