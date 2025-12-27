#!/usr/bin/env python3
"""Retry Strategy Patterns for Transport Operations.

Provides pluggable retry strategies that can be used with TransportBase
and TransportChain for flexible retry behavior.

Available Strategies:
- ExponentialBackoffStrategy: Classic exponential backoff (2^n delay)
- LinearBackoffStrategy: Linear increase in delay
- FibonacciBackoffStrategy: Fibonacci sequence delays
- AdaptiveStrategy: Adjusts based on recent failure patterns
- NoRetryStrategy: Fail immediately (for testing/critical paths)

Usage:
    from app.coordination.retry_strategies import (
        ExponentialBackoffStrategy,
        RetryContext,
    )

    strategy = ExponentialBackoffStrategy(
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
    )

    async def transfer_with_retry(data):
        ctx = RetryContext(target="node-1", operation="transfer")

        while strategy.should_retry(ctx):
            try:
                return await do_transfer(data)
            except Exception as e:
                ctx.record_failure(e)
                delay = strategy.get_delay(ctx)
                await asyncio.sleep(delay)

        raise RetryExhaustedError(ctx)

December 2025: Consolidates retry patterns from:
- cluster_transport.py (transport failover)
- handler_resilience.py (single retry)
- delivery_retry_queue.py (exponential backoff)
"""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RetryContext:
    """Context for tracking retry state.

    Maintains information about retry attempts for a single operation,
    allowing strategies to make informed decisions about delays and
    whether to continue retrying.
    """

    target: str = ""  # Target identifier (hostname, URL, etc.)
    operation: str = ""  # Operation name for logging
    attempt: int = 0  # Current attempt number (0 = first try)
    failures: list[Exception] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    last_failure_time: float = 0.0
    total_delay: float = 0.0  # Total time spent in delays
    metadata: dict[str, Any] = field(default_factory=dict)

    def record_failure(self, error: Exception) -> None:
        """Record a failure for this operation."""
        self.failures.append(error)
        self.last_failure_time = time.time()
        self.attempt += 1

    def record_delay(self, delay: float) -> None:
        """Record delay time."""
        self.total_delay += delay

    @property
    def elapsed_time(self) -> float:
        """Total elapsed time since start."""
        return time.time() - self.start_time

    @property
    def last_error(self) -> Exception | None:
        """Most recent error, if any."""
        return self.failures[-1] if self.failures else None

    @property
    def failure_count(self) -> int:
        """Number of failures so far."""
        return len(self.failures)


@dataclass
class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""

    context: RetryContext
    message: str = ""

    def __str__(self) -> str:
        msg = self.message or f"Retry exhausted for {self.context.operation}"
        return (
            f"{msg} "
            f"(attempts={self.context.attempt}, "
            f"target={self.context.target}, "
            f"elapsed={self.context.elapsed_time:.1f}s)"
        )


# =============================================================================
# Base Strategy Class
# =============================================================================


class RetryStrategy(ABC):
    """Abstract base class for retry strategies.

    Implementations define:
    - should_retry(): Whether to continue retrying
    - get_delay(): How long to wait before next attempt
    """

    def __init__(
        self,
        max_retries: int = 3,
        max_delay: float = 60.0,
        max_total_time: float = 300.0,
    ):
        """Initialize the retry strategy.

        Args:
            max_retries: Maximum number of retry attempts
            max_delay: Maximum delay between retries (seconds)
            max_total_time: Maximum total time for all retries (seconds)
        """
        self.max_retries = max_retries
        self.max_delay = max_delay
        self.max_total_time = max_total_time

    @abstractmethod
    def get_delay(self, ctx: RetryContext) -> float:
        """Calculate delay before next retry attempt.

        Args:
            ctx: Current retry context

        Returns:
            Delay in seconds before next attempt
        """
        pass

    def should_retry(self, ctx: RetryContext) -> bool:
        """Determine if another retry should be attempted.

        Args:
            ctx: Current retry context

        Returns:
            True if retry should be attempted
        """
        # Check attempt limit
        if ctx.attempt >= self.max_retries:
            return False

        # Check total time limit
        if ctx.elapsed_time >= self.max_total_time:
            return False

        return True

    def on_retry(self, ctx: RetryContext, delay: float) -> None:
        """Called before each retry. Override for custom logging/metrics."""
        logger.debug(
            f"[{self.__class__.__name__}] Retry {ctx.attempt}/{self.max_retries} "
            f"for {ctx.operation} on {ctx.target} in {delay:.2f}s"
        )


# =============================================================================
# Concrete Strategies
# =============================================================================


class ExponentialBackoffStrategy(RetryStrategy):
    """Exponential backoff retry strategy.

    Delay increases exponentially: base_delay * (multiplier ^ attempt)
    Common pattern: 1s, 2s, 4s, 8s, 16s, ...

    Optionally adds jitter to prevent thundering herd.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        multiplier: float = 2.0,
        max_delay: float = 60.0,
        max_total_time: float = 300.0,
        jitter: float = 0.1,
    ):
        """Initialize exponential backoff.

        Args:
            max_retries: Maximum retry attempts
            base_delay: Initial delay in seconds
            multiplier: Exponential multiplier (typically 2)
            max_delay: Maximum delay cap
            max_total_time: Maximum total retry time
            jitter: Random jitter factor (0.1 = ±10%)
        """
        super().__init__(max_retries, max_delay, max_total_time)
        self.base_delay = base_delay
        self.multiplier = multiplier
        self.jitter = jitter

    def get_delay(self, ctx: RetryContext) -> float:
        """Calculate exponential backoff delay with optional jitter."""
        # Calculate base exponential delay
        delay = self.base_delay * (self.multiplier ** ctx.attempt)

        # Add jitter
        if self.jitter > 0:
            jitter_range = delay * self.jitter
            delay += random.uniform(-jitter_range, jitter_range)

        # Cap at max delay
        return min(delay, self.max_delay)


class LinearBackoffStrategy(RetryStrategy):
    """Linear backoff retry strategy.

    Delay increases linearly: base_delay + (increment * attempt)
    Pattern: 1s, 3s, 5s, 7s, 9s, ...
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        increment: float = 2.0,
        max_delay: float = 30.0,
        max_total_time: float = 120.0,
    ):
        """Initialize linear backoff.

        Args:
            max_retries: Maximum retry attempts
            base_delay: Initial delay in seconds
            increment: Delay increase per attempt
            max_delay: Maximum delay cap
            max_total_time: Maximum total retry time
        """
        super().__init__(max_retries, max_delay, max_total_time)
        self.base_delay = base_delay
        self.increment = increment

    def get_delay(self, ctx: RetryContext) -> float:
        """Calculate linear backoff delay."""
        delay = self.base_delay + (self.increment * ctx.attempt)
        return min(delay, self.max_delay)


class FibonacciBackoffStrategy(RetryStrategy):
    """Fibonacci sequence backoff strategy.

    Delay follows Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, ...
    Provides smoother ramp-up than exponential.
    """

    def __init__(
        self,
        max_retries: int = 5,
        scale: float = 1.0,
        max_delay: float = 60.0,
        max_total_time: float = 180.0,
    ):
        """Initialize Fibonacci backoff.

        Args:
            max_retries: Maximum retry attempts
            scale: Multiplier for Fibonacci values
            max_delay: Maximum delay cap
            max_total_time: Maximum total retry time
        """
        super().__init__(max_retries, max_delay, max_total_time)
        self.scale = scale
        self._fib_cache = [1, 1]

    def _fib(self, n: int) -> int:
        """Get nth Fibonacci number."""
        while len(self._fib_cache) <= n:
            self._fib_cache.append(
                self._fib_cache[-1] + self._fib_cache[-2]
            )
        return self._fib_cache[n]

    def get_delay(self, ctx: RetryContext) -> float:
        """Calculate Fibonacci backoff delay."""
        fib_value = self._fib(ctx.attempt)
        delay = fib_value * self.scale
        return min(delay, self.max_delay)


class AdaptiveStrategy(RetryStrategy):
    """Adaptive retry strategy that adjusts based on failure patterns.

    Tracks recent failures across all operations and adjusts delay
    based on system-wide health. Increases delays when system is
    struggling, decreases when things improve.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        max_total_time: float = 300.0,
        min_delay: float = 0.5,
        backoff_multiplier: float = 2.0,
        success_window: int = 10,
    ):
        """Initialize adaptive strategy.

        Args:
            max_retries: Maximum retry attempts
            base_delay: Base delay when healthy
            max_delay: Maximum delay cap
            max_total_time: Maximum total retry time
            min_delay: Minimum delay floor
            backoff_multiplier: Multiplier per consecutive failure
            success_window: Number of recent ops to track
        """
        super().__init__(max_retries, max_delay, max_total_time)
        self.base_delay = base_delay
        self.min_delay = min_delay
        self.backoff_multiplier = backoff_multiplier
        self.success_window = success_window

        # Track recent outcomes globally
        self._recent_outcomes: list[bool] = []

    def record_outcome(self, success: bool) -> None:
        """Record operation outcome for adaptive adjustment."""
        self._recent_outcomes.append(success)
        if len(self._recent_outcomes) > self.success_window:
            self._recent_outcomes.pop(0)

    def _get_failure_rate(self) -> float:
        """Calculate recent failure rate."""
        if not self._recent_outcomes:
            return 0.0
        failures = sum(1 for s in self._recent_outcomes if not s)
        return failures / len(self._recent_outcomes)

    def get_delay(self, ctx: RetryContext) -> float:
        """Calculate adaptive delay based on system health."""
        # Base exponential backoff
        delay = self.base_delay * (self.backoff_multiplier ** ctx.attempt)

        # Adjust based on recent failure rate
        failure_rate = self._get_failure_rate()
        if failure_rate > 0.5:
            # System struggling - increase delay
            delay *= 1.0 + failure_rate
        elif failure_rate < 0.1:
            # System healthy - can be more aggressive
            delay *= 0.8

        # Clamp to bounds
        return max(self.min_delay, min(delay, self.max_delay))


class ConstantDelayStrategy(RetryStrategy):
    """Constant delay between retries.

    Simple strategy with fixed delay between attempts.
    Useful for rate-limited APIs or predictable resources.
    """

    def __init__(
        self,
        max_retries: int = 3,
        delay: float = 1.0,
        max_total_time: float = 60.0,
    ):
        """Initialize constant delay strategy.

        Args:
            max_retries: Maximum retry attempts
            delay: Fixed delay between attempts
            max_total_time: Maximum total retry time
        """
        super().__init__(max_retries, delay, max_total_time)
        self.delay = delay

    def get_delay(self, ctx: RetryContext) -> float:
        """Return constant delay."""
        return self.delay


class NoRetryStrategy(RetryStrategy):
    """No retry strategy - fail immediately.

    Useful for critical paths where retries are undesirable,
    or for testing failure handling.
    """

    def __init__(self):
        """Initialize no-retry strategy."""
        super().__init__(max_retries=0, max_delay=0, max_total_time=0)

    def should_retry(self, ctx: RetryContext) -> bool:
        """Never retry."""
        return False

    def get_delay(self, ctx: RetryContext) -> float:
        """No delay."""
        return 0.0


# =============================================================================
# Strategy Presets
# =============================================================================


def quick_retry() -> ExponentialBackoffStrategy:
    """Quick retry for responsive services."""
    return ExponentialBackoffStrategy(
        max_retries=2,
        base_delay=0.5,
        max_delay=5.0,
        max_total_time=15.0,
    )


def standard_retry() -> ExponentialBackoffStrategy:
    """Standard retry for typical services."""
    return ExponentialBackoffStrategy(
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        max_total_time=60.0,
    )


def patient_retry() -> ExponentialBackoffStrategy:
    """Patient retry for unreliable services."""
    return ExponentialBackoffStrategy(
        max_retries=5,
        base_delay=2.0,
        max_delay=60.0,
        max_total_time=300.0,
    )


def cluster_retry() -> AdaptiveStrategy:
    """Adaptive retry for cluster operations."""
    return AdaptiveStrategy(
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        success_window=20,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base class
    "RetryStrategy",
    # Context and errors
    "RetryContext",
    "RetryExhaustedError",
    # Concrete strategies
    "ExponentialBackoffStrategy",
    "LinearBackoffStrategy",
    "FibonacciBackoffStrategy",
    "AdaptiveStrategy",
    "ConstantDelayStrategy",
    "NoRetryStrategy",
    # Presets
    "quick_retry",
    "standard_retry",
    "patient_retry",
    "cluster_retry",
]
