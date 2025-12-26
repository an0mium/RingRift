#!/usr/bin/env python3
"""Quick gauntlet evaluation for a model."""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.training.game_gauntlet import run_baseline_gauntlet, BaselineOpponent


def main():
    # Use the latest improved model (Dec 25)
    model_path = Path("models/hex8_2p_improved_dec25_20251225_230948.pth")
    if not model_path.exists():
        # Fall back to canonical
        model_path = Path("models/canonical_hex8_2p.pth")

    print(f"Running gauntlet for {model_path.name}...")
    print("=" * 60)

    result = run_baseline_gauntlet(
        model_path=str(model_path),
        board_type="hex8",  # String works now after gauntlet fix
        num_players=2,
        games_per_opponent=15,
        opponents=[BaselineOpponent.RANDOM, BaselineOpponent.HEURISTIC],
        verbose=True,
        early_stopping=True,
        parallel_opponents=False,  # Disable parallel to avoid multiprocessing issues
    )

    print("\n" + "=" * 60)
    print("GAUNTLET RESULTS")
    print("=" * 60)
    print(f"Model: {model_path.name}")
    print(f"Passed baseline gating: {result.passes_baseline_gating}")
    print(f"Total wins: {result.total_wins}/{result.total_games}")
    print(f"Win rate: {result.win_rate*100:.1f}%")

    for opp_name, stats in result.opponent_results.items():
        win_rate = stats.get('win_rate', 0)
        wins = stats.get('wins', 0)
        games = stats.get('games', 0)
        print(f"  vs {opp_name}: {wins}/{games} ({win_rate*100:.0f}%)")

    if hasattr(result, 'estimated_elo') and result.estimated_elo:
        print(f"\nEstimated Elo: {result.estimated_elo:.0f}")


if __name__ == "__main__":
    main()
