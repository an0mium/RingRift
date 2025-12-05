#!/usr/bin/env python
"""
run_tier_gate.py
================

Lightweight CLI wrapper for heuristic tier evaluation on eval pools.

This script is intentionally small and focused on the heuristic
`HeuristicTierSpec` configuration defined in
`app.training.tier_eval_config`. It runs a single heuristic tier
against its configured eval pool and prints a JSON summary that is
easy to consume from CI or dashboards.

Usage (from ai-service/):

  PYTHONPATH=. python scripts/run_tier_gate.py --tier-id sq8_heuristic_baseline_v1

  PYTHONPATH=. python scripts/run_tier_gate.py \\
    --tier-id sq8_heuristic_baseline_v1 \\
    --seed 123 \\
    --max-games 16 \\
    --output-json results/ai_eval/tier_gate.sq8_heuristic_baseline_v1.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.training.tier_eval_config import (  # noqa: E402
    HEURISTIC_TIER_SPECS,
    HeuristicTierSpec,
)
from app.training.eval_pools import (  # noqa: E402
    run_heuristic_tier_eval,
)


def _get_tier_spec(tier_id: str) -> HeuristicTierSpec:
    for spec in HEURISTIC_TIER_SPECS:
        if spec.id == tier_id:
            return spec
    available = ", ".join(sorted(t.id for t in HEURISTIC_TIER_SPECS))
    raise SystemExit(f"Unknown heuristic tier id {tier_id!r}. Available ids: {available}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single heuristic eval-pool tier gate and emit a JSON summary. "
            "Tiers are defined in app.training.tier_eval_config.HEURISTIC_TIER_SPECS."
        ),
    )
    parser.add_argument(
        "--tier-id",
        required=True,
        help="Heuristic tier id (e.g. sq8_heuristic_baseline_v1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base RNG seed for reproducible evaluations (default: 1).",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help=(
            "Optional cap on games for this run. When unset, the tier's "
            "configured num_games is used."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help=(
            "Optional path to write the JSON summary. When omitted, the "
            "summary is printed to stdout only."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    tier_spec = _get_tier_spec(args.tier_id)

    result = run_heuristic_tier_eval(
        tier_spec=tier_spec,
        rng_seed=args.seed,
        max_games=args.max_games,
    )

    payload: dict[str, Any] = {
        "tier_id": result.get("tier_id"),
        "tier_name": result.get("tier_name"),
        "board_type": result.get("board_type"),
        "num_players": result.get("num_players"),
        "eval_pool_id": result.get("eval_pool_id"),
        "candidate_profile_id": result.get("candidate_profile_id"),
        "baseline_profile_id": result.get("baseline_profile_id"),
        "games_requested": result.get("games_requested"),
        "games_played": result.get("games_played"),
        "results": result.get("results"),
        "margins": result.get("margins"),
        "latency_ms": result.get("latency_ms"),
        "total_moves": result.get("total_moves"),
        "victory_reasons": result.get("victory_reasons"),
        "config": {
            "tier_spec_id": tier_spec.id,
            "tier_spec_name": tier_spec.name,
            "description": tier_spec.description,
        },
    }

    json_text = json.dumps(payload, indent=2, sort_keys=True)
    print(json_text)

    if args.output_json:
        out_path = os.path.abspath(args.output_json)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json_text)
        print(f"\nWrote tier gate report to {out_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

