#!/usr/bin/env python
"""Cross-board tier orchestrator for RingRift multi-config promotion.

This script orchestrates tier evaluation and promotion across all 9 board/player
configurations (3 boards x 3 player counts). It:

1. Runs tier gating for each configuration in parallel or sequentially.
2. Aggregates Elo estimates across all configurations.
3. Makes automatic promotion decisions when target Elo is met.
4. Updates the ladder runtime overrides for promoted models.

Usage:
    python scripts/crossboard_tier_orchestrator.py \
        --candidate-id ringrift_v7_unified \
        --target-elo 2000 \
        --run-dir /path/to/run

The orchestrator respects the 2000+ Elo target across all 9 configurations
and only promotes when a model exceeds this threshold everywhere.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import ladder_config
from app.models import BoardType
from app.training.crossboard_strength import (
    ALL_BOARD_CONFIGS,
    aggregate_cross_board_elos,
    check_promotion_threshold,
    config_key,
)
from app.training.tier_promotion_registry import (
    load_config_registry,
    save_config_registry,
    update_config_registry_for_run,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Output filenames
ORCHESTRATOR_REPORT_FILENAME = "crossboard_orchestrator_report.json"
CONFIG_GATE_FILENAME = "config_gate_report.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Orchestrate tier evaluation and promotion across all 9 board/player "
            "configurations. Aggregates Elo estimates and makes promotion decisions."
        ),
    )
    parser.add_argument(
        "--candidate-id",
        required=True,
        help="Candidate model identifier (e.g., ringrift_v7_unified).",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Base run directory for storing results.",
    )
    parser.add_argument(
        "--target-elo",
        type=float,
        default=2000.0,
        help="Target Elo threshold for promotion (default: 2000).",
    )
    parser.add_argument(
        "--tier",
        default="D8",
        help="Difficulty tier to evaluate (default: D8).",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help=(
            "Specific configs to evaluate (e.g., 'square8_2p hexagonal_3p'). "
            "Defaults to all 9 configurations."
        ),
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=3,
        help="Number of parallel config evaluations (default: 3).",
    )
    parser.add_argument(
        "--games-per-config",
        type=int,
        default=200,
        help="Games per configuration for Elo estimation (default: 200).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow promotion if weakest config is within 50 Elo of target.",
    )
    parser.add_argument(
        "--auto-promote",
        action="store_true",
        help="Automatically update ladder runtime overrides on success.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use lightweight configs suitable for CI smoke runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate evaluation without actually running games.",
    )
    parser.add_argument(
        "--skip-parity-gate",
        action="store_true",
        help="Skip parity validation gate (not recommended for production).",
    )
    parser.add_argument(
        "--parity-databases",
        nargs="*",
        help="Specific databases for parity validation.",
    )
    return parser.parse_args(argv)


def run_parity_gate(
    run_dir: str,
    databases: list[str] | None = None,
    demo: bool = False,
) -> dict[str, Any]:
    """Run parity validation gate before tier evaluations.

    Args:
        run_dir: Directory to store parity gate results.
        databases: Optional list of database paths to validate.
        demo: If True, use reduced game counts.

    Returns:
        Parity gate summary dict with 'overall_passed' key.
    """
    logger.info("Running parity validation gate...")

    parity_output = os.path.join(run_dir, "parity_gate_report.json")

    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "run_automated_parity_gate.py"),
        "--output-json", parity_output,
        "--emit-events",
    ]

    if demo:
        cmd.append("--demo")

    if databases:
        cmd.extend(["--databases"] + databases)

    try:
        subprocess.run(cmd, check=False, cwd=PROJECT_ROOT, capture_output=True)

        # Load the parity gate results
        if os.path.exists(parity_output):
            with open(parity_output, encoding="utf-8") as f:
                return json.load(f)

    except Exception as e:
        logger.error(f"Parity gate failed: {e}")

    # Fallback: assume parity failed
    return {
        "overall_passed": False,
        "error": "Parity gate execution failed",
    }


def _parse_config_list(config_strs: list[str] | None) -> list[tuple[str, int]]:
    """Parse config strings like 'square8_2p' into (board, num_players) tuples."""
    if config_strs is None:
        return ALL_BOARD_CONFIGS

    configs = []
    for cfg_str in config_strs:
        parts = cfg_str.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].endswith("p"):
            logger.warning(f"Invalid config string: {cfg_str}, skipping")
            continue
        board = parts[0]
        num_players = int(parts[1][:-1])
        configs.append((board, num_players))

    return configs if configs else ALL_BOARD_CONFIGS


def run_config_evaluation(
    board: str,
    num_players: int,
    candidate_id: str,
    tier: str,
    run_dir: str,
    games: int,
    demo: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Run tier evaluation for a single board/player configuration.

    Returns a result dictionary with Elo estimate and gate decision.
    """
    cfg_key = config_key(board, num_players)
    config_run_dir = os.path.join(run_dir, cfg_key)
    os.makedirs(config_run_dir, exist_ok=True)

    logger.info(f"Evaluating {cfg_key}: tier={tier}, games={games}")

    if dry_run:
        # Simulate evaluation with synthetic Elo values
        import random

        base_elo = 1900 + random.randint(-100, 200)
        return {
            "config_key": cfg_key,
            "board": board,
            "num_players": num_players,
            "tier": tier,
            "elo_estimate": base_elo,
            "elo_std": 50.0,
            "games_played": games,
            "gate_pass": base_elo >= 2000,
            "dry_run": True,
            "run_dir": config_run_dir,
        }

    # Run the tier gating script for this configuration
    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "run_crossboard_tier_gating.py"),
        "--tier",
        tier,
        "--candidate-id",
        candidate_id,
        "--run-dir",
        config_run_dir,
        "--board",
        board,
        "--num-players",
        str(num_players),
    ]

    if demo:
        cmd.append("--demo")

    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, capture_output=True)

        # Load the gate report
        gate_path = os.path.join(config_run_dir, "crossboard_gate_report.json")
        if os.path.exists(gate_path):
            with open(gate_path, encoding="utf-8") as f:
                gate_report = json.load(f)

            # Extract Elo estimate from the gate report
            tier_gate = gate_report.get("tier_gate", {})
            elo_estimate = tier_gate.get("elo_estimate", 1500.0)

            return {
                "config_key": cfg_key,
                "board": board,
                "num_players": num_players,
                "tier": tier,
                "elo_estimate": elo_estimate,
                "elo_std": tier_gate.get("elo_std", 100.0),
                "games_played": tier_gate.get("games_played", games),
                "gate_pass": gate_report.get("final_decision") == "promote",
                "dry_run": False,
                "run_dir": config_run_dir,
            }

    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed for {cfg_key}: {e}")

    # Fallback for failed evaluations
    return {
        "config_key": cfg_key,
        "board": board,
        "num_players": num_players,
        "tier": tier,
        "elo_estimate": 1000.0,  # Pessimistic fallback
        "elo_std": 200.0,
        "games_played": 0,
        "gate_pass": False,
        "error": "Evaluation failed",
        "run_dir": config_run_dir,
    }


def run_all_evaluations(
    configs: list[tuple[str, int]],
    candidate_id: str,
    tier: str,
    run_dir: str,
    games: int,
    parallel: int,
    demo: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    """Run evaluations for all configurations, optionally in parallel."""
    results = []

    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(
                    run_config_evaluation,
                    board,
                    num_players,
                    candidate_id,
                    tier,
                    run_dir,
                    games,
                    demo,
                    dry_run,
                ): config_key(board, num_players)
                for board, num_players in configs
            }

            for future in as_completed(futures):
                cfg_key = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(
                        f"Completed {cfg_key}: Elo={result['elo_estimate']:.1f}, "
                        f"pass={result['gate_pass']}"
                    )
                except Exception as e:
                    logger.error(f"Exception for {cfg_key}: {e}")
                    results.append({
                        "config_key": cfg_key,
                        "error": str(e),
                        "elo_estimate": 1000.0,
                        "gate_pass": False,
                    })
    else:
        for board, num_players in configs:
            result = run_config_evaluation(
                board,
                num_players,
                candidate_id,
                tier,
                run_dir,
                games,
                demo,
                dry_run,
            )
            results.append(result)
            logger.info(
                f"Completed {config_key(board, num_players)}: "
                f"Elo={result['elo_estimate']:.1f}, pass={result['gate_pass']}"
            )

    return results


def apply_promotion(
    candidate_id: str,
    tier: str,
    configs: list[tuple[str, int]],
) -> bool:
    """Apply promotion by updating ladder runtime overrides."""
    tier_name = tier.upper()
    if not tier_name.startswith("D"):
        tier_name = f"D{tier_name}"
    difficulty = int(tier_name[1:])

    success_count = 0
    for board, num_players in configs:
        board_type_map = {
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "hexagonal": BoardType.HEXAGONAL,
        }
        board_type = board_type_map.get(board.lower())
        if board_type is None:
            logger.warning(f"Unknown board type: {board}")
            continue

        try:
            result = ladder_config.update_tier_model(
                difficulty=difficulty,
                board_type=board_type,
                num_players=num_players,
                model_id=candidate_id,
            )
            if result:
                success_count += 1
                logger.info(f"Updated ladder for {config_key(board, num_players)}")
            else:
                logger.warning(
                    f"Failed to update ladder for {config_key(board, num_players)}"
                )
        except Exception as e:
            logger.error(f"Error updating ladder for {config_key(board, num_players)}: {e}")

    return success_count == len(configs)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)

    configs = _parse_config_list(args.configs)
    logger.info(
        f"Starting cross-board orchestration: {len(configs)} configs, "
        f"target={args.target_elo} Elo, tier={args.tier}"
    )

    # Run parity gate first (unless skipped)
    parity_result = None
    if not args.skip_parity_gate and not args.dry_run:
        parity_result = run_parity_gate(
            run_dir=run_dir,
            databases=args.parity_databases,
            demo=args.demo,
        )

        if not parity_result.get("overall_passed", False):
            logger.error("Parity gate FAILED - blocking promotion")
            print("\n" + "=" * 60)
            print("PARITY GATE FAILED")
            print("=" * 60)
            print("Promotion blocked due to parity validation failures.")
            print("Fix parity issues before attempting promotion.")
            print("=" * 60)

            # Write partial report
            report = {
                "candidate_id": args.candidate_id,
                "tier": args.tier,
                "parity_gate": parity_result,
                "blocked": True,
                "reason": "Parity validation failed",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            report_path = os.path.join(run_dir, ORCHESTRATOR_REPORT_FILENAME)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, sort_keys=True)

            return 1

        logger.info("Parity gate PASSED - proceeding with tier evaluations")

    # Run evaluations
    results = run_all_evaluations(
        configs=configs,
        candidate_id=args.candidate_id,
        tier=args.tier,
        run_dir=run_dir,
        games=args.games_per_config,
        parallel=args.parallel,
        demo=args.demo,
        dry_run=args.dry_run,
    )

    # Aggregate Elos
    config_elos = {r["config_key"]: r["elo_estimate"] for r in results}
    aggregation = aggregate_cross_board_elos(config_elos)

    # Check promotion threshold
    promotion_check = check_promotion_threshold(
        config_elos,
        args.target_elo,
        min_configs=len(configs),
        allow_partial=args.allow_partial,
    )

    # Build orchestrator report
    report = {
        "candidate_id": args.candidate_id,
        "tier": args.tier,
        "target_elo": args.target_elo,
        "configs_evaluated": len(configs),
        "parity_gate": parity_result if parity_result else {"skipped": True},
        "aggregation": aggregation,
        "promotion_check": promotion_check,
        "per_config_results": results,
        "auto_promote": args.auto_promote,
        "promotion_applied": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Apply promotion if eligible and auto-promote is enabled
    if promotion_check["eligible"] and args.auto_promote:
        logger.info("Promotion criteria met, applying ladder updates...")
        promotion_success = apply_promotion(args.candidate_id, args.tier, configs)
        report["promotion_applied"] = promotion_success
        if promotion_success:
            logger.info("Ladder updates applied successfully")
        else:
            logger.warning("Some ladder updates failed")

    # Write report
    report_path = os.path.join(run_dir, ORCHESTRATOR_REPORT_FILENAME)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    logger.info(f"Orchestrator report written to {report_path}")

    # Summary output
    print("\n" + "=" * 60)
    print("CROSS-BOARD ORCHESTRATION SUMMARY")
    print("=" * 60)
    print(f"Candidate: {args.candidate_id}")
    print(f"Tier: {args.tier}")
    print(f"Target Elo: {args.target_elo}")
    print(f"Configs evaluated: {aggregation['num_configs']}")
    print(f"Mean Elo: {aggregation['mean_elo']:.1f}")
    print(f"Min Elo: {aggregation['min_elo']:.1f} ({aggregation['weakest_config']})")
    print(f"Max Elo: {aggregation['max_elo']:.1f} ({aggregation['strongest_config']})")
    print(f"Std Elo: {aggregation['std_elo']:.1f}")
    print("-" * 60)
    print(f"Promotion eligible: {promotion_check['eligible']}")
    print(f"Reason: {promotion_check['reason']}")
    if report["promotion_applied"]:
        print("Ladder updates: APPLIED")
    print("=" * 60)

    return 0 if promotion_check["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
