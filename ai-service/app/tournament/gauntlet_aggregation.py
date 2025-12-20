#!/usr/bin/env python3
"""Gauntlet Result Aggregation for Composite ELO System.

This module provides utilities for aggregating gauntlet results across:
- Multiple NNs (to find best NN per algorithm)
- Multiple algorithms (to find best algorithm per NN)
- Cross-algorithm analysis (NN ranking consistency)

Usage:
    from app.tournament.gauntlet_aggregation import (
        aggregate_by_nn,
        aggregate_by_algorithm,
        check_nn_ranking_consistency,
    )

    # Get best algorithm for each NN
    nn_summaries = aggregate_by_nn(board_type="square8", num_players=2)

    # Get best NN for each algorithm
    algo_summaries = aggregate_by_algorithm(board_type="square8", num_players=2)

    # Check if NN rankings are consistent across algorithms
    consistency = check_nn_ranking_consistency(board_type="square8", num_players=2)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.training.composite_participant import (
    extract_ai_type,
    extract_nn_id,
    is_composite_id,
)
from app.training.elo_service import get_elo_service

logger = logging.getLogger(__name__)


@dataclass
class NNSummary:
    """Aggregated summary for a neural network across algorithms."""
    nn_model_id: str
    best_algorithm: str
    best_elo: float
    avg_elo: float
    worst_elo: float
    elo_spread: float
    algorithms_tested: int
    total_games: int
    algorithm_ratings: dict[str, float] = field(default_factory=dict)


@dataclass
class AlgorithmSummary:
    """Aggregated summary for an algorithm across NNs."""
    ai_algorithm: str
    avg_elo: float
    best_elo: float
    worst_elo: float
    elo_spread: float
    nn_count: int
    total_games: int
    best_nn: str | None = None
    nn_ratings: dict[str, float] = field(default_factory=dict)


@dataclass
class ConsistencyReport:
    """Report on NN ranking consistency across algorithms."""
    board_type: str
    num_players: int
    algorithms_compared: list[str]
    nn_count: int
    avg_rank_correlation: float
    rank_violations: list[dict[str, Any]]
    is_consistent: bool


def aggregate_by_nn(
    board_type: str = "square8",
    num_players: int = 2,
    min_games: int = 5,
) -> list[NNSummary]:
    """Aggregate ratings by neural network across algorithms.

    Groups all (NN, Algorithm) combinations by NN and computes:
    - Best algorithm for each NN
    - Average Elo across algorithms
    - Elo spread (max - min)

    Args:
        board_type: Board type
        num_players: Number of players
        min_games: Minimum games for inclusion

    Returns:
        List of NNSummary objects sorted by best_elo descending
    """
    elo_service = get_elo_service()

    # Get all composite participants with ratings
    leaderboard = elo_service.get_composite_leaderboard(
        board_type=board_type,
        num_players=num_players,
        min_games=min_games,
        limit=1000,
    )

    # Group by NN
    nn_data: dict[str, dict[str, Any]] = {}

    for entry in leaderboard:
        nn_id = entry.get("nn_model_id")
        if not nn_id or nn_id == "none":
            continue

        if nn_id not in nn_data:
            nn_data[nn_id] = {
                "ratings": [],
                "algorithms": {},
                "total_games": 0,
            }

        algo = entry.get("ai_algorithm", "unknown")
        rating = entry.get("rating", 1500)
        games = entry.get("games_played", 0)

        nn_data[nn_id]["ratings"].append(rating)
        nn_data[nn_id]["algorithms"][algo] = rating
        nn_data[nn_id]["total_games"] += games

    # Build summaries
    summaries = []
    for nn_id, data in nn_data.items():
        ratings = data["ratings"]
        if not ratings:
            continue

        best_algo = max(data["algorithms"].items(), key=lambda x: x[1])

        summaries.append(NNSummary(
            nn_model_id=nn_id,
            best_algorithm=best_algo[0],
            best_elo=max(ratings),
            avg_elo=sum(ratings) / len(ratings),
            worst_elo=min(ratings),
            elo_spread=max(ratings) - min(ratings),
            algorithms_tested=len(data["algorithms"]),
            total_games=data["total_games"],
            algorithm_ratings=data["algorithms"],
        ))

    # Sort by best Elo
    summaries.sort(key=lambda x: x.best_elo, reverse=True)
    return summaries


def aggregate_by_algorithm(
    board_type: str = "square8",
    num_players: int = 2,
    min_games: int = 5,
) -> list[AlgorithmSummary]:
    """Aggregate ratings by algorithm across NNs.

    Groups all (NN, Algorithm) combinations by algorithm and computes:
    - Best NN for each algorithm
    - Average Elo across NNs
    - NN count using this algorithm

    Args:
        board_type: Board type
        num_players: Number of players
        min_games: Minimum games for inclusion

    Returns:
        List of AlgorithmSummary objects sorted by avg_elo descending
    """
    elo_service = get_elo_service()

    # Get algorithm rankings directly from EloService
    rankings = elo_service.get_algorithm_rankings(
        board_type=board_type,
        num_players=num_players,
        min_games=min_games,
    )

    # Get detailed data for each algorithm
    summaries = []
    for rank_data in rankings:
        algo = rank_data["ai_algorithm"]

        # Get all NNs using this algorithm
        algo_leaderboard = elo_service.get_composite_leaderboard(
            board_type=board_type,
            num_players=num_players,
            ai_algorithm=algo,
            min_games=min_games,
            limit=1000,
        )

        nn_ratings = {}
        for entry in algo_leaderboard:
            nn_id = entry.get("nn_model_id")
            if nn_id and nn_id != "none":
                nn_ratings[nn_id] = entry.get("rating", 1500)

        best_nn = max(nn_ratings.items(), key=lambda x: x[1])[0] if nn_ratings else None

        summaries.append(AlgorithmSummary(
            ai_algorithm=algo,
            avg_elo=rank_data["avg_elo"],
            best_elo=rank_data["best_elo"],
            worst_elo=rank_data["worst_elo"],
            elo_spread=rank_data["elo_spread"],
            nn_count=rank_data["nn_count"],
            total_games=rank_data["total_games"],
            best_nn=best_nn,
            nn_ratings=nn_ratings,
        ))

    return summaries


def check_nn_ranking_consistency(
    board_type: str = "square8",
    num_players: int = 2,
    min_games: int = 10,
    min_algorithms: int = 2,
) -> ConsistencyReport:
    """Check if NN rankings are consistent across algorithms.

    Computes rank correlation between algorithm pairs and identifies
    ranking violations (where NN_a > NN_b with one algorithm but
    NN_a < NN_b with another).

    Args:
        board_type: Board type
        num_players: Number of players
        min_games: Minimum games for inclusion
        min_algorithms: Minimum algorithms tested for NN inclusion

    Returns:
        ConsistencyReport with correlation and violation data
    """
    elo_service = get_elo_service()

    # Get all composite participants
    leaderboard = elo_service.get_composite_leaderboard(
        board_type=board_type,
        num_players=num_players,
        min_games=min_games,
        limit=1000,
    )

    # Build NN -> Algorithm -> Rating matrix
    nn_algo_ratings: dict[str, dict[str, float]] = {}

    for entry in leaderboard:
        nn_id = entry.get("nn_model_id")
        algo = entry.get("ai_algorithm")

        if not nn_id or nn_id == "none" or not algo:
            continue

        if nn_id not in nn_algo_ratings:
            nn_algo_ratings[nn_id] = {}
        nn_algo_ratings[nn_id][algo] = entry.get("rating", 1500)

    # Filter NNs with enough algorithm coverage
    qualified_nns = {
        nn: algos for nn, algos in nn_algo_ratings.items()
        if len(algos) >= min_algorithms
    }

    if len(qualified_nns) < 2:
        return ConsistencyReport(
            board_type=board_type,
            num_players=num_players,
            algorithms_compared=[],
            nn_count=len(qualified_nns),
            avg_rank_correlation=1.0,
            rank_violations=[],
            is_consistent=True,
        )

    # Find common algorithms
    all_algos = set()
    for algos in qualified_nns.values():
        all_algos.update(algos.keys())

    # Find ranking violations
    violations = []
    correlations = []

    algo_list = sorted(all_algos)
    for i, algo1 in enumerate(algo_list):
        for algo2 in algo_list[i+1:]:
            # Get NNs that have both algorithms
            common_nns = [
                nn for nn, algos in qualified_nns.items()
                if algo1 in algos and algo2 in algos
            ]

            if len(common_nns) < 2:
                continue

            # Check for violations
            for j, nn1 in enumerate(common_nns):
                for nn2 in common_nns[j+1:]:
                    rating1_algo1 = qualified_nns[nn1][algo1]
                    rating2_algo1 = qualified_nns[nn2][algo1]
                    rating1_algo2 = qualified_nns[nn1][algo2]
                    rating2_algo2 = qualified_nns[nn2][algo2]

                    # Check if ranking differs
                    order_algo1 = rating1_algo1 > rating2_algo1
                    order_algo2 = rating1_algo2 > rating2_algo2

                    if order_algo1 != order_algo2:
                        # Ranking violation
                        violations.append({
                            "nn_a": nn1,
                            "nn_b": nn2,
                            "algo_1": algo1,
                            "algo_2": algo2,
                            "rating_a_algo1": rating1_algo1,
                            "rating_b_algo1": rating2_algo1,
                            "rating_a_algo2": rating1_algo2,
                            "rating_b_algo2": rating2_algo2,
                        })

            # Compute rank correlation (Spearman-like)
            rankings1 = sorted(common_nns, key=lambda nn: qualified_nns[nn][algo1], reverse=True)
            rankings2 = sorted(common_nns, key=lambda nn: qualified_nns[nn][algo2], reverse=True)

            rank1 = {nn: i for i, nn in enumerate(rankings1)}
            rank2 = {nn: i for i, nn in enumerate(rankings2)}

            # Pearson correlation of ranks
            n = len(common_nns)
            if n > 1:
                mean_r1 = sum(rank1.values()) / n
                mean_r2 = sum(rank2.values()) / n

                cov = sum((rank1[nn] - mean_r1) * (rank2[nn] - mean_r2) for nn in common_nns)
                var1 = sum((rank1[nn] - mean_r1) ** 2 for nn in common_nns)
                var2 = sum((rank2[nn] - mean_r2) ** 2 for nn in common_nns)

                if var1 > 0 and var2 > 0:
                    correlation = cov / ((var1 * var2) ** 0.5)
                    correlations.append(correlation)

    avg_correlation = sum(correlations) / len(correlations) if correlations else 1.0
    is_consistent = avg_correlation > 0.7 and len(violations) < len(qualified_nns)

    return ConsistencyReport(
        board_type=board_type,
        num_players=num_players,
        algorithms_compared=algo_list,
        nn_count=len(qualified_nns),
        avg_rank_correlation=avg_correlation,
        rank_violations=violations[:10],  # Limit to 10 examples
        is_consistent=is_consistent,
    )


def update_nn_performance_summaries(
    board_type: str = "square8",
    num_players: int = 2,
    min_games: int = 5,
) -> int:
    """Update nn_performance_summary table from current ratings.

    Args:
        board_type: Board type
        num_players: Number of players
        min_games: Minimum games for inclusion

    Returns:
        Number of summaries updated
    """
    from app.training.composite_elo_migration import update_nn_performance_summaries as _update
    result = _update(board_type=board_type, num_players=num_players)
    return result.get("nn_models_updated", 0)


def update_algorithm_baselines(
    board_type: str = "square8",
    num_players: int = 2,
    min_games: int = 20,
) -> dict[str, float]:
    """Update algorithm baseline ratings from current data.

    Computes the average Elo for each algorithm and updates the
    algorithm_baselines table.

    Args:
        board_type: Board type
        num_players: Number of players
        min_games: Minimum games for reliable estimate

    Returns:
        Dict mapping algorithm -> updated baseline Elo
    """
    elo_service = get_elo_service()

    # Get algorithm rankings
    rankings = elo_service.get_algorithm_rankings(
        board_type=board_type,
        num_players=num_players,
        min_games=min_games,
    )

    updated = {}
    for rank_data in rankings:
        algo = rank_data["ai_algorithm"]
        avg_elo = rank_data["avg_elo"]
        total_games = rank_data["total_games"]

        elo_service.update_algorithm_baseline(
            ai_algorithm=algo,
            board_type=board_type,
            num_players=num_players,
            baseline_elo=avg_elo,
            games_played=total_games,
        )

        updated[algo] = avg_elo

    return updated


def print_aggregation_report(
    board_type: str = "square8",
    num_players: int = 2,
    min_games: int = 5,
) -> None:
    """Print comprehensive aggregation report to console."""
    print(f"\n{'='*60}")
    print(f"Composite ELO Aggregation Report: {board_type}/{num_players}p")
    print(f"{'='*60}")

    # NN Rankings
    print("\n--- Top NNs by Best Algorithm ---")
    nn_summaries = aggregate_by_nn(board_type, num_players, min_games)
    for i, s in enumerate(nn_summaries[:10], 1):
        nn_short = s.nn_model_id[:35] + "..." if len(s.nn_model_id) > 35 else s.nn_model_id
        print(f"{i:2}. {nn_short}")
        print(f"    Best: {s.best_elo:.0f} ({s.best_algorithm}), "
              f"Avg: {s.avg_elo:.0f}, Spread: {s.elo_spread:.0f}")

    # Algorithm Rankings
    print("\n--- Algorithm Rankings ---")
    algo_summaries = aggregate_by_algorithm(board_type, num_players, min_games)
    for s in algo_summaries:
        print(f"  {s.ai_algorithm:15} | Avg: {s.avg_elo:.0f} | "
              f"Best: {s.best_elo:.0f} | NNs: {s.nn_count}")

    # Consistency Check
    print("\n--- NN Ranking Consistency ---")
    consistency = check_nn_ranking_consistency(board_type, num_players, min_games)
    print(f"  Algorithms: {', '.join(consistency.algorithms_compared)}")
    print(f"  NNs analyzed: {consistency.nn_count}")
    print(f"  Rank correlation: {consistency.avg_rank_correlation:.2f}")
    print(f"  Ranking violations: {len(consistency.rank_violations)}")
    print(f"  Consistent: {'Yes' if consistency.is_consistent else 'No'}")

    if consistency.rank_violations:
        print("\n  Sample violations:")
        for v in consistency.rank_violations[:3]:
            print(f"    {v['nn_a'][:20]} vs {v['nn_b'][:20]}")
            print(f"      {v['algo_1']}: {v['rating_a_algo1']:.0f} vs {v['rating_b_algo1']:.0f}")
            print(f"      {v['algo_2']}: {v['rating_a_algo2']:.0f} vs {v['rating_b_algo2']:.0f}")

    print(f"\n{'='*60}\n")
