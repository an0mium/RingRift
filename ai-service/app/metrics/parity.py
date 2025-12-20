"""Parity healthcheck metrics for RingRift AI service.

This module provides Prometheus metrics for tracking parity healthcheck results,
which validate that GPU and CPU rules engines produce identical outputs.

Usage:
    from app.metrics.parity import (
        record_parity_mismatch,
        record_parity_case,
        update_parity_pass_rate,
    )

    # Record a mismatch
    record_parity_mismatch("hash", "contract_vectors")

    # Record a case result
    record_parity_case("contract_vectors", passed=True)

    # Update pass rate
    update_parity_pass_rate("contract_vectors", 0.98)
"""

from __future__ import annotations

import logging

from prometheus_client import Counter, Gauge

logger = logging.getLogger(__name__)

# Parity mismatch counter - tracks mismatches by type and suite
PARITY_MISMATCHES_TOTAL = Counter(
    "ringrift_parity_mismatches_total",
    "Total parity mismatches by type and suite.",
    ["mismatch_type", "suite"],
)

# Parity healthcheck cases counter - tracks cases by suite and result
PARITY_HEALTHCHECK_CASES_TOTAL = Counter(
    "ringrift_parity_healthcheck_cases_total",
    "Total parity healthcheck cases executed.",
    ["suite", "result"],
)

# Parity healthcheck pass rate gauge - current pass rate per suite
PARITY_HEALTHCHECK_PASS_RATE = Gauge(
    "ringrift_parity_healthcheck_pass_rate",
    "Parity healthcheck pass rate per suite (0-1).",
    ["suite"],
)


def record_parity_mismatch(mismatch_type: str, suite: str) -> None:
    """Record a parity mismatch.

    Args:
        mismatch_type: Type of mismatch (validation, status, hash, s_invariant, unknown)
        suite: Suite name (contract_vectors, plateau_snapshots)
    """
    try:
        PARITY_MISMATCHES_TOTAL.labels(mismatch_type=mismatch_type, suite=suite).inc()
    except Exception as e:
        logger.warning(f"Failed to record parity mismatch metric: {e}")


def record_parity_case(suite: str, passed: bool) -> None:
    """Record a parity healthcheck case result.

    Args:
        suite: Suite name (contract_vectors, plateau_snapshots)
        passed: Whether the case passed
    """
    try:
        result = "passed" if passed else "failed"
        PARITY_HEALTHCHECK_CASES_TOTAL.labels(suite=suite, result=result).inc()
    except Exception as e:
        logger.warning(f"Failed to record parity case metric: {e}")


def update_parity_pass_rate(suite: str, pass_rate: float) -> None:
    """Update the parity healthcheck pass rate for a suite.

    Args:
        suite: Suite name (contract_vectors, plateau_snapshots)
        pass_rate: Pass rate between 0 and 1
    """
    try:
        PARITY_HEALTHCHECK_PASS_RATE.labels(suite=suite).set(pass_rate)
    except Exception as e:
        logger.warning(f"Failed to update parity pass rate metric: {e}")


def emit_parity_summary_metrics(summary: dict) -> None:
    """Emit metrics from a parity healthcheck summary.

    This function takes the JSON summary from run_parity_healthcheck.py
    and emits corresponding Prometheus metrics.

    Args:
        summary: Parity healthcheck summary dict with keys:
            - total_cases: int
            - total_mismatches: int
            - mismatches_by_type: dict[str, int]
            - mismatches_by_suite: dict[str, int]
            - pass_rate_by_suite: dict[str, float] (optional)
    """
    try:
        # Emit mismatch counts by type
        for mismatch_type, count in summary.get("mismatches_by_type", {}).items():
            # We need to know the suite, so use mismatches_by_suite instead
            pass

        # Emit mismatch counts by suite (type=all for aggregate)
        for suite, count in summary.get("mismatches_by_suite", {}).items():
            # Increment by the count (note: Counter only supports inc())
            # For bulk updates, we'd need a different approach
            pass

        # Update pass rates per suite
        total_cases = summary.get("total_cases", 0)
        total_mismatches = summary.get("total_mismatches", 0)

        if total_cases > 0:
            # Overall pass rate
            overall_pass_rate = 1.0 - (total_mismatches / total_cases)
            update_parity_pass_rate("all", overall_pass_rate)

        # Per-suite pass rates if available
        for suite, pass_rate in summary.get("pass_rate_by_suite", {}).items():
            update_parity_pass_rate(suite, pass_rate)

    except Exception as e:
        logger.warning(f"Failed to emit parity summary metrics: {e}")


__all__ = [
    "PARITY_HEALTHCHECK_CASES_TOTAL",
    "PARITY_HEALTHCHECK_PASS_RATE",
    "PARITY_MISMATCHES_TOTAL",
    "emit_parity_summary_metrics",
    "record_parity_case",
    "record_parity_mismatch",
    "update_parity_pass_rate",
]
