#!/usr/bin/env python
"""Automated parity gate for RingRift promotion pipeline.

This script runs comprehensive parity validation as a pre-condition for
model promotion. It integrates with the cross-board tier orchestrator
and blocks promotions when parity violations are detected.

The gate performs three levels of validation:

1. **Contract Vector Validation**: Tests TS↔Python turn/phase semantics
   using the v2 contract vectors suite.

2. **Database Parity Validation**: Validates that recorded games can be
   replayed identically by both Python and TypeScript engines.

3. **Healthcheck Suites**: Runs plateau snapshot and other parity fixtures.

Usage:
    # Run full parity gate
    python scripts/run_automated_parity_gate.py --output-json results.json

    # Quick CI mode (reduced game count)
    python scripts/run_automated_parity_gate.py --demo --fail-on-mismatch

    # Validate specific databases
    python scripts/run_automated_parity_gate.py --databases data/games/*.db

The gate exits with code 0 on success (parity passes) or 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Thresholds for parity validation
DEFAULT_MAX_DIVERGENCE_RATE = 0.05  # 5% maximum divergence
DEFAULT_MAX_CONTRACT_MISMATCHES = 0  # Zero tolerance for contract mismatches


@dataclass
class ParityGateResult:
    """Result of a parity gate check."""

    suite: str
    passed: bool
    total_cases: int
    mismatches: int
    mismatch_rate: float
    details: dict[str, Any]


def run_contract_vectors_check(demo: bool = False) -> ParityGateResult:
    """Run contract vector parity check.

    Returns:
        ParityGateResult with contract vector validation results.
    """
    logger.info("Running contract vectors parity check...")
    start_time = time.time()

    try:
        # Import the healthcheck module
        from scripts.run_parity_healthcheck import (
            run_contract_vectors_suite,
        )

        results = run_contract_vectors_suite()

        total = len(results)
        mismatches = sum(1 for r in results if r.mismatch_type is not None)
        mismatch_rate = mismatches / total if total > 0 else 0.0

        elapsed = time.time() - start_time
        logger.info(
            f"Contract vectors: {total - mismatches}/{total} passed "
            f"({mismatch_rate:.1%} divergence) in {elapsed:.1f}s"
        )

        # Collect sample mismatches for debugging
        samples = []
        for r in results:
            if r.mismatch_type is not None:
                samples.append({
                    "case_id": r.case_id,
                    "mismatch_type": r.mismatch_type,
                    "details": r.details,
                })
                if len(samples) >= 5:
                    break

        return ParityGateResult(
            suite="contract_vectors",
            passed=mismatches == 0,
            total_cases=total,
            mismatches=mismatches,
            mismatch_rate=mismatch_rate,
            details={
                "elapsed_seconds": elapsed,
                "sample_mismatches": samples,
            },
        )

    except Exception as e:
        logger.error(f"Contract vectors check failed: {e}")
        return ParityGateResult(
            suite="contract_vectors",
            passed=False,
            total_cases=0,
            mismatches=1,
            mismatch_rate=1.0,
            details={"error": str(e)},
        )


def run_plateau_snapshots_check() -> ParityGateResult:
    """Run plateau snapshot parity check.

    Returns:
        ParityGateResult with plateau snapshot validation results.
    """
    logger.info("Running plateau snapshots parity check...")
    start_time = time.time()

    try:
        from scripts.run_parity_healthcheck import (
            run_plateau_snapshots_suite,
        )

        results = run_plateau_snapshots_suite()

        total = len(results)
        mismatches = sum(1 for r in results if r.mismatch_type is not None)
        mismatch_rate = mismatches / total if total > 0 else 0.0

        elapsed = time.time() - start_time
        logger.info(
            f"Plateau snapshots: {total - mismatches}/{total} passed "
            f"({mismatch_rate:.1%} divergence) in {elapsed:.1f}s"
        )

        return ParityGateResult(
            suite="plateau_snapshots",
            passed=mismatches == 0,
            total_cases=total,
            mismatches=mismatches,
            mismatch_rate=mismatch_rate,
            details={"elapsed_seconds": elapsed},
        )

    except Exception as e:
        logger.error(f"Plateau snapshots check failed: {e}")
        return ParityGateResult(
            suite="plateau_snapshots",
            passed=False,
            total_cases=0,
            mismatches=1,
            mismatch_rate=1.0,
            details={"error": str(e)},
        )


def run_database_parity_check(
    databases: list[str] | None = None,
    max_games_per_db: int = 100,
    demo: bool = False,
) -> ParityGateResult:
    """Run database game replay parity check.

    Args:
        databases: List of database paths to validate. If None, searches
            for databases in the default data directory.
        max_games_per_db: Maximum games to validate per database.
        demo: If True, use reduced game count for CI.

    Returns:
        ParityGateResult with database validation results.
    """
    logger.info("Running database parity check...")
    start_time = time.time()

    # Find databases to validate
    if databases is None:
        data_dir = Path(PROJECT_ROOT) / "data" / "games"
        if data_dir.exists():
            databases = [str(p) for p in data_dir.glob("*.db")]
        else:
            databases = []

    if not databases:
        logger.info("No databases found to validate")
        return ParityGateResult(
            suite="database_parity",
            passed=True,
            total_cases=0,
            mismatches=0,
            mismatch_rate=0.0,
            details={"databases": [], "warning": "No databases found"},
        )

    # Limit games in demo mode
    if demo:
        max_games_per_db = min(max_games_per_db, 20)

    total_games = 0
    total_divergences = 0
    per_db_results = []

    try:
        from scripts.run_parity_validation import validate_database

        for db_path in databases:
            if not os.path.exists(db_path):
                continue

            logger.info(f"  Validating {os.path.basename(db_path)}...")
            result = validate_database(
                Path(db_path),
                mode="canonical",
                progress_every=0,
                max_games=max_games_per_db,
            )

            games = result.get("total_games_checked", 0)
            divergences = result.get("games_with_semantic_divergence", 0)

            total_games += games
            total_divergences += divergences

            per_db_results.append({
                "database": os.path.basename(db_path),
                "games_checked": games,
                "divergences": divergences,
            })

        mismatch_rate = total_divergences / total_games if total_games > 0 else 0.0
        passed = mismatch_rate < DEFAULT_MAX_DIVERGENCE_RATE

        elapsed = time.time() - start_time
        logger.info(
            f"Database parity: {total_games - total_divergences}/{total_games} passed "
            f"({mismatch_rate:.1%} divergence) in {elapsed:.1f}s"
        )

        return ParityGateResult(
            suite="database_parity",
            passed=passed,
            total_cases=total_games,
            mismatches=total_divergences,
            mismatch_rate=mismatch_rate,
            details={
                "elapsed_seconds": elapsed,
                "databases_checked": len(per_db_results),
                "per_database": per_db_results,
                "max_divergence_rate": DEFAULT_MAX_DIVERGENCE_RATE,
            },
        )

    except Exception as e:
        logger.error(f"Database parity check failed: {e}")
        return ParityGateResult(
            suite="database_parity",
            passed=False,
            total_cases=0,
            mismatches=1,
            mismatch_rate=1.0,
            details={"error": str(e)},
        )


def aggregate_results(results: list[ParityGateResult]) -> dict[str, Any]:
    """Aggregate parity gate results into a summary.

    Args:
        results: List of individual suite results.

    Returns:
        Aggregated summary dictionary.
    """
    all_passed = all(r.passed for r in results)
    total_cases = sum(r.total_cases for r in results)
    total_mismatches = sum(r.mismatches for r in results)

    overall_mismatch_rate = (
        total_mismatches / total_cases if total_cases > 0 else 0.0
    )

    suites_summary = {}
    for r in results:
        suites_summary[r.suite] = {
            "passed": r.passed,
            "total_cases": r.total_cases,
            "mismatches": r.mismatches,
            "mismatch_rate": round(r.mismatch_rate, 4),
            "details": r.details,
        }

    return {
        "overall_passed": all_passed,
        "total_cases": total_cases,
        "total_mismatches": total_mismatches,
        "overall_mismatch_rate": round(overall_mismatch_rate, 4),
        "suites": suites_summary,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def emit_parity_event(
    summary: dict[str, Any],
    source: str = "automated_parity_gate",
) -> None:
    """Emit parity validation event for monitoring.

    Args:
        summary: Aggregated parity summary.
        source: Event source identifier.
    """
    try:
        from app.distributed.event_helpers import emit_sync

        payload = {
            "overall_passed": summary["overall_passed"],
            "total_cases": summary["total_cases"],
            "total_mismatches": summary["total_mismatches"],
            "mismatch_rate": summary["overall_mismatch_rate"],
            "suites_passed": [
                s for s, d in summary["suites"].items() if d["passed"]
            ],
            "suites_failed": [
                s for s, d in summary["suites"].items() if not d["passed"]
            ],
        }

        emit_sync("PARITY_VALIDATION_COMPLETED", payload, source)
        logger.debug("Parity validation event emitted")

    except Exception as e:
        logger.debug(f"Could not emit parity event: {e}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run comprehensive parity validation as a promotion gate. "
            "Validates contract vectors, plateau snapshots, and game databases."
        ),
    )
    parser.add_argument(
        "--output-json",
        help="Path to write JSON results.",
    )
    parser.add_argument(
        "--databases",
        nargs="*",
        help="Specific database files to validate.",
    )
    parser.add_argument(
        "--max-games-per-db",
        type=int,
        default=100,
        help="Maximum games to validate per database (default: 100).",
    )
    parser.add_argument(
        "--skip-contract-vectors",
        action="store_true",
        help="Skip contract vector validation.",
    )
    parser.add_argument(
        "--skip-plateau-snapshots",
        action="store_true",
        help="Skip plateau snapshot validation.",
    )
    parser.add_argument(
        "--skip-database-parity",
        action="store_true",
        help="Skip database parity validation.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use reduced counts suitable for CI.",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit with non-zero status on any mismatch.",
    )
    parser.add_argument(
        "--emit-events",
        action="store_true",
        help="Emit events for monitoring integration.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    logger.info("Starting automated parity gate...")
    start_time = time.time()

    results: list[ParityGateResult] = []

    # Run contract vectors check
    if not args.skip_contract_vectors:
        result = run_contract_vectors_check(demo=args.demo)
        results.append(result)

    # Run plateau snapshots check
    if not args.skip_plateau_snapshots:
        result = run_plateau_snapshots_check()
        results.append(result)

    # Run database parity check
    if not args.skip_database_parity:
        result = run_database_parity_check(
            databases=args.databases,
            max_games_per_db=args.max_games_per_db,
            demo=args.demo,
        )
        results.append(result)

    # Aggregate results
    summary = aggregate_results(results)
    total_elapsed = time.time() - start_time
    summary["total_elapsed_seconds"] = round(total_elapsed, 2)

    # Emit events if requested
    if args.emit_events:
        emit_parity_event(summary)

    # Write output
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        logger.info(f"Results written to {args.output_json}")

    # Print summary
    print("\n" + "=" * 60)
    print("AUTOMATED PARITY GATE SUMMARY")
    print("=" * 60)
    for suite, data in summary["suites"].items():
        status = "PASS" if data["passed"] else "FAIL"
        print(
            f"  {suite}: {status} "
            f"({data['total_cases'] - data['mismatches']}/{data['total_cases']} passed)"
        )
    print("-" * 60)
    print(f"Overall: {'PASSED' if summary['overall_passed'] else 'FAILED'}")
    print(f"Total cases: {summary['total_cases']}")
    print(f"Total mismatches: {summary['total_mismatches']}")
    print(f"Mismatch rate: {summary['overall_mismatch_rate']:.2%}")
    print(f"Elapsed: {total_elapsed:.1f}s")
    print("=" * 60)

    # Determine exit code
    if args.fail_on_mismatch and summary["total_mismatches"] > 0:
        return 1

    return 0 if summary["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
