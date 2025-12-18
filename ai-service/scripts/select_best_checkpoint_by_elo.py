#!/usr/bin/env python
"""Select the best checkpoint based on Elo evaluation rather than loss.

This script addresses the loss/Elo disconnect where lower validation loss
doesn't always correlate with better playing strength. It:

1. Finds all checkpoints for a training run (epoch checkpoints + final)
2. Runs a mini-gauntlet (fast games against random/heuristic)
3. Selects the checkpoint with highest estimated Elo
4. Copies it as the "best" checkpoint

Usage:
    python scripts/select_best_checkpoint_by_elo.py \
        --candidate-id sq8_2p_d8_cand_20251218_040151 \
        --games 20
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.models import BoardType, AIType, AIConfig, GameStatus
from app.ai.heuristic_ai import HeuristicAI
from app.ai.random_ai import RandomAI
from app.ai.policy_only_ai import PolicyOnlyAI
from app.training.generate_data import create_initial_state
from app.rules.default_engine import DefaultRulesEngine


def is_versioned_checkpoint(checkpoint_path: Path) -> bool:
    """Check if a checkpoint has versioning metadata.

    Versioned checkpoints are safer to load as they include architecture info.
    """
    try:
        import torch
        data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return "version" in data or "__version__" in data or "architecture" in data
    except Exception:
        return False


def find_checkpoints(
    candidate_id: str,
    models_dir: str = "models",
    skip_checkpoint_dir: bool = True,
    versioned_only: bool = True,
) -> List[Path]:
    """Find all checkpoints for a candidate model.

    Looks for:
    - {candidate_id}.pth (final/best by loss)
    - {candidate_id}_*.pth (epoch checkpoints)
    - optionally checkpoints/{candidate_id}/*.pth (disabled by default)

    Args:
        candidate_id: Model candidate ID prefix
        models_dir: Base models directory
        skip_checkpoint_dir: If True, skip checkpoints/ subdirectory (default True)
            These legacy checkpoints often have architecture mismatches.
        versioned_only: If True, only include versioned checkpoints (default True)
    """
    models_path = Path(models_dir).resolve()
    checkpoints = []

    # Main checkpoint
    main_ckpt = models_path / f"{candidate_id}.pth"
    if main_ckpt.exists():
        checkpoints.append(main_ckpt.resolve())

    # Epoch checkpoints with timestamps
    for f in models_path.glob(f"{candidate_id}_*.pth"):
        if f.exists() and "_elo_best" not in f.name:  # Skip elo_best to avoid circular
            checkpoints.append(f.resolve())

    # Checkpoint directory (disabled by default due to legacy architecture issues)
    if not skip_checkpoint_dir:
        ckpt_dir = models_path / "checkpoints" / candidate_id
        if ckpt_dir.exists():
            for f in ckpt_dir.glob("*.pth"):
                checkpoints.append(f.resolve())

    # Filter to versioned-only if requested
    if versioned_only:
        original_count = len(checkpoints)
        checkpoints = [c for c in checkpoints if is_versioned_checkpoint(c)]
        skipped = original_count - len(checkpoints)
        if skipped > 0:
            print(f"  Skipped {skipped} legacy (unversioned) checkpoints")

    return sorted(checkpoints, key=lambda x: x.stat().st_mtime)


class CheckpointLoadError(Exception):
    """Raised when a checkpoint cannot be loaded due to architecture mismatch."""
    pass


def validate_checkpoint_loadable(
    checkpoint_path: Path,
    board_type: BoardType,
) -> bool:
    """Test if a checkpoint can be loaded without architecture errors.

    This performs a quick validation by attempting to load the model.
    Returns True if loadable, False otherwise.
    """
    try:
        config = AIConfig(
            ai_type=AIType.POLICY_ONLY,
            board_type=board_type,
            difficulty=8,
            use_neural_net=True,
            nn_model_id=str(checkpoint_path),
            policy_temperature=0.5,
        )
        # Try to create the AI - this will fail if architecture mismatches
        ai = PolicyOnlyAI(1, config, board_type=board_type)
        # Try to get a move to ensure model is fully loaded
        return True
    except RuntimeError as e:
        if "size mismatch" in str(e) or "Error(s) in loading state_dict" in str(e):
            return False
        raise
    except Exception:
        return False


def evaluate_checkpoint(
    checkpoint_path: Path,
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
    games_per_opponent: int = 10,
) -> Dict[str, Any]:
    """Evaluate a checkpoint via mini-gauntlet against baselines.

    Returns dict with win rates and estimated Elo.
    Raises CheckpointLoadError if checkpoint cannot be loaded.
    """
    from app.ai.neural_net import NeuralNetAI

    # Pre-validate checkpoint is loadable
    if not validate_checkpoint_loadable(checkpoint_path, board_type):
        raise CheckpointLoadError(
            f"Cannot load checkpoint {checkpoint_path}: architecture mismatch"
        )

    results = {
        "checkpoint": str(checkpoint_path),
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "vs_random": {"wins": 0, "games": 0},
        "vs_heuristic": {"wins": 0, "games": 0},
    }

    engine = DefaultRulesEngine()

    # Create neural net AI for the checkpoint
    def create_nn_ai(player: int) -> PolicyOnlyAI:
        config = AIConfig(
            ai_type=AIType.POLICY_ONLY,
            board_type=board_type,
            difficulty=8,
            use_neural_net=True,
            nn_model_id=str(checkpoint_path),
            policy_temperature=0.5,
        )
        return PolicyOnlyAI(player, config, board_type=board_type)

    baselines = [
        ("random", lambda p: RandomAI(p, AIConfig(ai_type=AIType.RANDOM, board_type=board_type, difficulty=1))),
        ("heuristic", lambda p: HeuristicAI(p, AIConfig(ai_type=AIType.HEURISTIC, board_type=board_type, difficulty=5))),
    ]

    for baseline_name, baseline_factory in baselines:
        for game_num in range(games_per_opponent):
            # Alternate who goes first
            nn_player = 1 if game_num % 2 == 0 else 2
            baseline_player = 2 if game_num % 2 == 0 else 1

            try:
                nn_ai = create_nn_ai(nn_player)
                baseline_ai = baseline_factory(baseline_player)

                state = create_initial_state(board_type, num_players)

                move_count = 0
                max_moves = 500

                while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
                    current_player = state.current_player

                    if current_player == nn_player:
                        move = nn_ai.select_move(state)
                    else:
                        move = baseline_ai.select_move(state)

                    if move:
                        state = engine.apply_move(state, move)
                    move_count += 1

                # Determine winner
                results["games"] += 1
                results[f"vs_{baseline_name}"]["games"] += 1

                if state.winner is not None:
                    if state.winner == nn_player:
                        results["wins"] += 1
                        results[f"vs_{baseline_name}"]["wins"] += 1
                    else:
                        results["losses"] += 1
                else:
                    results["draws"] += 1

            except Exception as e:
                import traceback
                print(f"  Error in game {game_num}: {type(e).__name__}: {e}")
                traceback.print_exc()
                continue

    # Calculate win rates
    if results["games"] > 0:
        results["win_rate"] = results["wins"] / results["games"]
    else:
        results["win_rate"] = 0.0

    for baseline_name in ["random", "heuristic"]:
        if results[f"vs_{baseline_name}"]["games"] > 0:
            results[f"vs_{baseline_name}"]["win_rate"] = (
                results[f"vs_{baseline_name}"]["wins"] /
                results[f"vs_{baseline_name}"]["games"]
            )
        else:
            results[f"vs_{baseline_name}"]["win_rate"] = 0.0

    # Estimate Elo based on win rates
    # Random = 400 Elo, Heuristic = 1200 Elo
    # Use weighted average based on performance
    random_elo = 400
    heuristic_elo = 1200

    def elo_from_winrate(win_rate: float, opponent_elo: float) -> float:
        """Estimate Elo from win rate against known opponent."""
        if win_rate <= 0:
            return opponent_elo - 400
        if win_rate >= 1:
            return opponent_elo + 400
        # Elo formula: E = 1 / (1 + 10^((Rb-Ra)/400))
        # Solving for Ra: Ra = Rb - 400 * log10(1/E - 1)
        import math
        return opponent_elo - 400 * math.log10(1/win_rate - 1)

    elo_vs_random = elo_from_winrate(results["vs_random"]["win_rate"], random_elo)
    elo_vs_heuristic = elo_from_winrate(results["vs_heuristic"]["win_rate"], heuristic_elo)

    # Weight by games played
    total_games = results["vs_random"]["games"] + results["vs_heuristic"]["games"]
    if total_games > 0:
        results["estimated_elo"] = (
            elo_vs_random * results["vs_random"]["games"] +
            elo_vs_heuristic * results["vs_heuristic"]["games"]
        ) / total_games
    else:
        results["estimated_elo"] = 1500.0

    return results


def select_best_checkpoint(
    candidate_id: str,
    models_dir: str = "models",
    games_per_opponent: int = 10,
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
    """Evaluate all checkpoints and select the best one by Elo.

    Returns (best_checkpoint_path, all_results).
    """
    checkpoints = find_checkpoints(candidate_id, models_dir)

    if not checkpoints:
        print(f"No checkpoints found for {candidate_id}")
        return None, []

    print(f"Found {len(checkpoints)} checkpoints for {candidate_id}")

    all_results = []
    best_elo = float("-inf")
    best_checkpoint = None

    for i, ckpt in enumerate(checkpoints):
        print(f"\nEvaluating [{i+1}/{len(checkpoints)}] {ckpt.name}...")

        try:
            result = evaluate_checkpoint(
                ckpt,
                board_type=board_type,
                num_players=num_players,
                games_per_opponent=games_per_opponent,
            )
            all_results.append(result)

            print(f"  Win rate: {result['win_rate']:.1%}")
            print(f"  vs Random: {result['vs_random']['win_rate']:.1%}")
            print(f"  vs Heuristic: {result['vs_heuristic']['win_rate']:.1%}")
            print(f"  Estimated Elo: {result['estimated_elo']:.0f}")

            if result["estimated_elo"] > best_elo:
                best_elo = result["estimated_elo"]
                best_checkpoint = ckpt

        except CheckpointLoadError as e:
            print(f"  SKIPPED: {e}")
            continue

        except Exception as e:
            print(f"  Error evaluating {ckpt.name}: {e}")
            continue

    return best_checkpoint, all_results


def main():
    parser = argparse.ArgumentParser(
        description="Select best checkpoint by Elo evaluation"
    )
    parser.add_argument(
        "--candidate-id",
        required=True,
        help="Candidate model ID to evaluate",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Games per opponent for evaluation",
    )
    parser.add_argument(
        "--board-type",
        default="square8",
        choices=["square8", "square19", "hexagonal"],
        help="Board type",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players",
    )
    parser.add_argument(
        "--copy-best",
        action="store_true",
        help="Copy best checkpoint to {candidate_id}_best.pth",
    )

    args = parser.parse_args()

    board_type_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hexagonal": BoardType.HEXAGONAL,
    }

    best_ckpt, results = select_best_checkpoint(
        candidate_id=args.candidate_id,
        models_dir=args.models_dir,
        games_per_opponent=args.games,
        board_type=board_type_map[args.board_type],
        num_players=args.num_players,
    )

    if best_ckpt:
        print(f"\n{'='*60}")
        print(f"Best checkpoint: {best_ckpt.name}")

        # Find result for best
        for r in results:
            if r["checkpoint"] == str(best_ckpt):
                print(f"Estimated Elo: {r['estimated_elo']:.0f}")
                break

        if args.copy_best:
            best_path = Path(args.models_dir) / f"{args.candidate_id}_elo_best.pth"
            shutil.copy2(best_ckpt, best_path)
            print(f"Copied to: {best_path}")
    else:
        print("No valid checkpoints found")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
