"""Adaptive Timeout Management - Per-node latency-based timeout adjustment.

Jan 2026: Created as part of P2P Self-Healing Architecture.

This module provides adaptive timeout management that adjusts timeouts based on
observed latency rather than using fixed values. This reduces false positives
(marking healthy nodes as dead due to timeout) while maintaining quick failure
detection.

Key features:
- Per-node latency tracking with sliding window
- P99 percentile-based timeout calculation
- EWMA smoothing to prevent oscillation
- Integration with StabilityController for reactive adjustments
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Environment variable to disable adaptive timeouts
ADAPTIVE_TIMEOUTS_ENABLED = os.environ.get(
    "RINGRIFT_ADAPTIVE_TIMEOUTS_ENABLED", "true"
).lower() in ("true", "1", "yes")


@dataclass
class LatencyWindow:
    """Sliding window of latency samples for a node.

    Maintains a fixed-size deque of latency measurements and provides
    percentile calculations for adaptive timeout computation.
    """
    samples: deque = field(default_factory=lambda: deque(maxlen=100))

    def add(self, latency_ms: float) -> None:
        """Add a latency sample."""
        self.samples.append(latency_ms)

    def percentile(self, p: float) -> float:
        """Get p-th percentile (0-100).

        Args:
            p: Percentile to compute (e.g., 99 for p99)

        Returns:
            The percentile value in milliseconds, or 0 if no samples
        """
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * p / 100)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    def mean(self) -> float:
        """Get mean latency."""
        if not self.samples:
            return 0.0
        return sum(self.samples) / len(self.samples)

    def count(self) -> int:
        """Get number of samples."""
        return len(self.samples)


class AdaptiveTimeoutManager:
    """
    Manages per-node adaptive timeouts based on observed latency.

    The core formula is:
        timeout = p99_latency * 1.5 + safety_margin

    This ensures that:
    - 99% of healthy probes will complete within the timeout
    - The 1.5x multiplier and safety margin provide buffer for variance
    - EWMA smoothing prevents rapid oscillation

    Integration points:
    - record_latency(): Called after each successful probe
    - get_timeout(): Called before probing a node
    - update_timeouts(): Called periodically to recalculate all timeouts
    """

    MIN_TIMEOUT = 5.0   # Minimum timeout in seconds
    MAX_TIMEOUT = 60.0  # Maximum timeout in seconds
    SAFETY_MARGIN = 2.0  # Additional buffer in seconds
    EWMA_ALPHA = 0.1    # Slow adaptation rate (higher = faster adaptation)
    MIN_SAMPLES = 10    # Minimum samples before using adaptive timeout

    def __init__(self, default_timeout: float = 15.0):
        """Initialize the adaptive timeout manager.

        Args:
            default_timeout: Default timeout when no latency data available
        """
        self._default = default_timeout
        self._latencies: dict[str, LatencyWindow] = {}
        self._current_timeouts: dict[str, float] = {}
        self._last_update = 0.0

        logger.info(
            f"AdaptiveTimeoutManager initialized: enabled={ADAPTIVE_TIMEOUTS_ENABLED}, "
            f"default={default_timeout}s"
        )

    def record_latency(self, node_id: str, latency_ms: float) -> None:
        """Record observed latency for a node.

        Should be called after each successful probe with the measured
        round-trip time.

        Args:
            node_id: The node identifier
            latency_ms: Observed latency in milliseconds
        """
        if not ADAPTIVE_TIMEOUTS_ENABLED:
            return

        if node_id not in self._latencies:
            self._latencies[node_id] = LatencyWindow()
        self._latencies[node_id].add(latency_ms)

    def get_timeout(self, node_id: str) -> float:
        """Get adaptive timeout for a node.

        Returns the calculated adaptive timeout if sufficient samples exist,
        otherwise returns the default timeout.

        Args:
            node_id: The node identifier

        Returns:
            Timeout in seconds
        """
        if not ADAPTIVE_TIMEOUTS_ENABLED:
            return self._default

        # Use calculated timeout if available
        if node_id in self._current_timeouts:
            return self._current_timeouts[node_id]

        # Use default if insufficient data
        return self._default

    def update_timeouts(self) -> dict[str, float]:
        """Update all timeouts based on recent latency.

        Should be called periodically (e.g., every 30 seconds) to recalculate
        timeouts based on accumulated latency data.

        Returns:
            Dict of node_id -> new_timeout for nodes that changed
        """
        if not ADAPTIVE_TIMEOUTS_ENABLED:
            return {}

        updates = {}
        for node_id, window in self._latencies.items():
            if window.count() < self.MIN_SAMPLES:
                continue

            # Calculate target timeout from p99 latency
            p99_ms = window.percentile(99)
            target = (p99_ms / 1000) * 1.5 + self.SAFETY_MARGIN
            target = max(self.MIN_TIMEOUT, min(self.MAX_TIMEOUT, target))

            # Apply EWMA smoothing
            current = self._current_timeouts.get(node_id, self._default)
            new_timeout = current * (1 - self.EWMA_ALPHA) + target * self.EWMA_ALPHA

            # Only update if significant change (>0.5s)
            if abs(new_timeout - current) > 0.5:
                self._current_timeouts[node_id] = new_timeout
                updates[node_id] = new_timeout
                logger.debug(
                    f"Timeout updated for {node_id}: {current:.1f}s -> {new_timeout:.1f}s "
                    f"(p99={p99_ms:.0f}ms)"
                )

        self._last_update = time.time()
        return updates

    def increase_timeout(self, node_id: str, factor: float = 1.5) -> float:
        """Manually increase timeout for a node (called by StabilityController).

        Args:
            node_id: The node identifier
            factor: Multiplier for current timeout

        Returns:
            New timeout value
        """
        current = self._current_timeouts.get(node_id, self._default)
        new_timeout = min(current * factor, self.MAX_TIMEOUT)
        self._current_timeouts[node_id] = new_timeout
        logger.info(f"Timeout increased for {node_id}: {current:.1f}s -> {new_timeout:.1f}s")
        return new_timeout

    def decrease_timeout(self, node_id: str, factor: float = 0.75) -> float:
        """Manually decrease timeout for a node.

        Args:
            node_id: The node identifier
            factor: Multiplier for current timeout

        Returns:
            New timeout value
        """
        current = self._current_timeouts.get(node_id, self._default)
        new_timeout = max(current * factor, self.MIN_TIMEOUT)
        self._current_timeouts[node_id] = new_timeout
        logger.info(f"Timeout decreased for {node_id}: {current:.1f}s -> {new_timeout:.1f}s")
        return new_timeout

    def get_all_timeouts(self) -> dict[str, float]:
        """Return all current timeouts for status endpoint."""
        return dict(self._current_timeouts)

    def get_latency_stats(self, node_id: str) -> dict[str, Any] | None:
        """Get latency statistics for a specific node."""
        window = self._latencies.get(node_id)
        if not window or window.count() == 0:
            return None

        return {
            "samples": window.count(),
            "mean_ms": window.mean(),
            "p50_ms": window.percentile(50),
            "p90_ms": window.percentile(90),
            "p99_ms": window.percentile(99),
            "current_timeout": self._current_timeouts.get(node_id, self._default),
        }

    def get_status(self) -> dict[str, Any]:
        """Return status for HTTP endpoint."""
        return {
            "enabled": ADAPTIVE_TIMEOUTS_ENABLED,
            "default_timeout": self._default,
            "nodes_tracked": len(self._latencies),
            "nodes_with_adaptive_timeout": len(self._current_timeouts),
            "last_update": self._last_update,
            "timeouts": self.get_all_timeouts(),
            "config": {
                "min_timeout": self.MIN_TIMEOUT,
                "max_timeout": self.MAX_TIMEOUT,
                "safety_margin": self.SAFETY_MARGIN,
                "ewma_alpha": self.EWMA_ALPHA,
                "min_samples": self.MIN_SAMPLES,
            },
        }
