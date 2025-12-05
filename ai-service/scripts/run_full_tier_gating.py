#!/usr/bin/env python
"""Combined tier gating + perf benchmark wrapper for ladder tiers.

This script orchestrates the core checks described in the
AI_TIER_TRAINING_AND_PROMOTION_PIPELINE for a single difficulty tier:

- Runs the canonical difficulty-tier gate via run_tier_gate.py.
- Runs the small tier perf benchmark via run_tier_perf_benchmark.py.
- Aggregates results into a single gate_report JSON with:
  - tier, board_type, num_players
  - candidate_model_id, current_model_id (when available)
  - tier evaluation summary (overall_pass + key win-rate metrics)
  - perf benchmark summary (average/p95, budgets, within_avg/within_p95)

It does not modify ladder_config; promotions are still a manual,
explicit step that consumes the generated report.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config.perf_budgets import (  # noqa: E402
    TierPerfBudget,
    get_tier_perf_budget,
)
from app.training.tier_perf_benchmark import (  # noqa: E402
    TierPerfResult,
    run_tier_perf_benchmark,
)
from scripts.run_tier_gate import parse_args as parse_tier_gate_args  # type: ignore[import]  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run difficulty-tier gating and perf benchmark for a ladder tier "
            "(e.g. D4/D6/D8 on square8 2p) and emit a combined gate_report.json."
        ),
    )

    parser.add_argument(
        "--tier",
        required=True,
        help="Difficulty tier name (e.g. D2, D4, D6, D8).",
    )
    parser.add_argument(
        "--candidate-model-id",
        required=True,
        help="Candidate model identifier for the tier gate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base RNG seed for reproducible evaluations (default: 1).",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="logs/tier_gate",
        help="Directory to write tier gate and perf benchmark artefacts.",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=None,
        help=(
            "Optional override for games per opponent in difficulty-tier gate. "
            "When unset, TierEvaluationConfig defaults are used."
        ),
    )
    parser.add_argument(
        "--perf-num-games",
        type=int,
        default=1,
        help="Number of games for perf benchmark (default: 1).",
    )
    parser.add_argument(
        "--perf-moves-per-game",
        type=int,
        default=4,
        help="Maximum moves sampled per perf benchmark game (default: 4).",
    )

    return parser.parse_args(argv)


def _run_tier_gate_cli(args: argparse.Namespace, run_dir: str) -> Dict[str, Any]:
    """Invoke run_tier_gate.py in difficulty-tier mode and return its JSON."""
    tier = args.tier.upper()
    output_path = os.path.join(run_dir, f"{tier}_tier_eval.json")
    promotion_path = os.path.join(run_dir, f"{tier}_promotion_plan.json")

    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "run_tier_gate.py"),
    ]
    gate_args = [
        "--tier",
        tier,
        "--seed",
        str(args.seed),
        "--candidate-model-id",
        args.candidate_model_id,
        "--output-json",
        output_path,
        "--promotion-plan-out",
        promotion_path,
    ]

    if args.num_games is not None:
        gate_args.extend(["--num-games", str(args.num_games)])

    full_cmd = cmd + gate_args
    subprocess.run(full_cmd, check=True)

    with open(output_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload


def _eval_perf_budget(result: TierPerfResult) -> Dict[str, Any]:
    """Evaluate whether a perf benchmark result is within its tier budget."""
    within_avg = result.average_ms <= result.budget.max_avg_move_ms
    within_p95 = result.p95_ms <= result.budget.max_p95_move_ms
    overall = within_avg and within_p95
    return {
        "within_avg": within_avg,
        "within_p95": within_p95,
        "overall_pass": overall,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)

    tier_key = args.tier.upper()

    # Difficulty-tier gate JSON (TierEvaluationResult).
    tier_eval = _run_tier_gate_cli(args, run_dir)

    # Perf benchmark for tiers that have budgets; for others we still
    # measure latency but treat the budget as informational only.
    perf_budget: TierPerfBudget | None
    try:
        perf_budget = get_tier_perf_budget(tier_key)
    except KeyError:
        perf_budget = None

    perf_result: TierPerfResult | None = None
    perf_eval: Dict[str, Any] | None = None

    if perf_budget is not None:
        perf_result = run_tier_perf_benchmark(
            tier_name=tier_key,
            num_games=args.perf_num_games,
            moves_per_game=args.perf_moves_per_game,
            seed=args.seed,
        )
        perf_eval = _eval_perf_budget(perf_result)

        perf_json_path = os.path.join(run_dir, f"{tier_key}_perf.json")
        with open(perf_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tier_name": perf_result.tier_name,
                    "difficulty": perf_result.budget.difficulty,
                    "board_type": perf_result.budget.board_type.value,
                    "num_players": perf_result.budget.num_players,
                    "metrics": {
                        "average_ms": perf_result.average_ms,
                        "p95_ms": perf_result.p95_ms,
                    },
                    "budget": {
                        "max_avg_move_ms": perf_result.budget.max_avg_move_ms,
                        "max_p95_move_ms": perf_result.budget.max_p95_move_ms,
                    },
                    "evaluation": perf_eval,
                },
                f,
                indent=2,
                sort_keys=True,
            )

    # Combined gate report.
    ladder_info = tier_eval.get("ladder", {})
    report: Dict[str, Any] = {
        "tier": tier_key,
        "board_type": tier_eval.get("board_type"),
        "num_players": tier_eval.get("num_players"),
        "candidate_model_id": ladder_info.get("candidate_model_id"),
        "current_model_id": ladder_info.get("current_model_id"),
        "automated_gate": {
            "overall_pass": tier_eval.get("overall_pass"),
            "metrics": tier_eval.get("metrics", {}),
        },
    }

    if perf_result is not None and perf_eval is not None:
        report["perf"] = {
            "tier_name": perf_result.tier_name,
            "average_ms": perf_result.average_ms,
            "p95_ms": perf_result.p95_ms,
            "budget": {
                "max_avg_move_ms": perf_result.budget.max_avg_move_ms,
                "max_p95_move_ms": perf_result.budget.max_p95_move_ms,
            },
            "evaluation": perf_eval,
        }

    report_path = os.path.join(run_dir, f"{tier_key}_gate_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"\nWrote combined gate report to {report_path}")

    # For CI convenience, treat failure of either gate or perf budget as
    # a non-zero exit status so that pipelines can depend on this script.
    overall_gate_ok = bool(tier_eval.get("overall_pass"))
    perf_ok = True
    if perf_eval is not None:
        perf_ok = bool(perf_eval.get("overall_pass"))

    return 0 if (overall_gate_ok and perf_ok) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

