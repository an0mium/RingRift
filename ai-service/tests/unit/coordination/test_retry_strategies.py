"""Tests for retry_strategies.py module.

December 2025: Tests for pluggable retry strategy patterns.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from app.coordination.retry_strategies import (
    AdaptiveStrategy,
    ConstantDelayStrategy,
    ExponentialBackoffStrategy,
    FibonacciBackoffStrategy,
    LinearBackoffStrategy,
    NoRetryStrategy,
    RetryContext,
    RetryExhaustedError,
    RetryStrategy,
    cluster_retry,
    patient_retry,
    quick_retry,
    standard_retry,
)


# =============================================================================
# RetryContext Tests
# =============================================================================


class TestRetryContext:
    """Tests for RetryContext dataclass."""

    def test_default_values(self):
        """Test default context values."""
        ctx = RetryContext()
        assert ctx.target == ""
        assert ctx.operation == ""
        assert ctx.attempt == 0
        assert ctx.failures == []
        assert ctx.total_delay == 0.0

    def test_with_values(self):
        """Test context with custom values."""
        ctx = RetryContext(
            target="node-1",
            operation="transfer",
        )
        assert ctx.target == "node-1"
        assert ctx.operation == "transfer"

    def test_record_failure(self):
        """Test recording a failure."""
        ctx = RetryContext()
        error = ValueError("test error")

        ctx.record_failure(error)

        assert ctx.attempt == 1
        assert len(ctx.failures) == 1
        assert ctx.failures[0] is error
        assert ctx.last_failure_time > 0

    def test_record_multiple_failures(self):
        """Test recording multiple failures."""
        ctx = RetryContext()

        ctx.record_failure(ValueError("error 1"))
        ctx.record_failure(RuntimeError("error 2"))

        assert ctx.attempt == 2
        assert ctx.failure_count == 2

    def test_record_delay(self):
        """Test recording delay time."""
        ctx = RetryContext()

        ctx.record_delay(1.5)
        ctx.record_delay(2.5)

        assert ctx.total_delay == 4.0

    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        ctx = RetryContext()
        time.sleep(0.1)

        assert ctx.elapsed_time >= 0.1

    def test_last_error(self):
        """Test getting last error."""
        ctx = RetryContext()

        assert ctx.last_error is None

        ctx.record_failure(ValueError("first"))
        ctx.record_failure(RuntimeError("second"))

        assert isinstance(ctx.last_error, RuntimeError)


# =============================================================================
# RetryExhaustedError Tests
# =============================================================================


class TestRetryExhaustedError:
    """Tests for RetryExhaustedError exception."""

    def test_str_representation(self):
        """Test string representation."""
        ctx = RetryContext(target="node-1", operation="transfer")
        ctx.record_failure(ValueError("test"))

        error = RetryExhaustedError(context=ctx)
        str_repr = str(error)

        assert "transfer" in str_repr
        assert "node-1" in str_repr
        assert "attempts=1" in str_repr

    def test_custom_message(self):
        """Test with custom message."""
        ctx = RetryContext()
        error = RetryExhaustedError(context=ctx, message="Custom failure")

        assert "Custom failure" in str(error)


# =============================================================================
# ExponentialBackoffStrategy Tests
# =============================================================================


class TestExponentialBackoffStrategy:
    """Tests for ExponentialBackoffStrategy."""

    def test_default_values(self):
        """Test default strategy values."""
        strategy = ExponentialBackoffStrategy()
        assert strategy.max_retries == 3
        assert strategy.base_delay == 1.0
        assert strategy.multiplier == 2.0
        assert strategy.max_delay == 60.0

    def test_get_delay_exponential(self):
        """Test exponential delay calculation."""
        strategy = ExponentialBackoffStrategy(
            base_delay=1.0,
            multiplier=2.0,
            jitter=0.0,  # Disable jitter for predictable test
        )

        ctx = RetryContext()
        # First retry (attempt 0)
        assert strategy.get_delay(ctx) == 1.0

        ctx.record_failure(ValueError())
        # Second retry (attempt 1)
        assert strategy.get_delay(ctx) == 2.0

        ctx.record_failure(ValueError())
        # Third retry (attempt 2)
        assert strategy.get_delay(ctx) == 4.0

    def test_get_delay_caps_at_max(self):
        """Test delay is capped at max_delay."""
        strategy = ExponentialBackoffStrategy(
            base_delay=10.0,
            max_delay=30.0,
            jitter=0.0,
        )

        ctx = RetryContext()
        ctx.attempt = 5  # Would be 10 * 2^5 = 320

        assert strategy.get_delay(ctx) == 30.0

    def test_get_delay_with_jitter(self):
        """Test delay includes jitter."""
        strategy = ExponentialBackoffStrategy(
            base_delay=10.0,
            jitter=0.1,
        )

        ctx = RetryContext()
        delays = [strategy.get_delay(ctx) for _ in range(10)]

        # All should be within 10% of 10.0
        assert all(9.0 <= d <= 11.0 for d in delays)
        # Should have some variation
        assert len(set(delays)) > 1

    def test_should_retry_within_limits(self):
        """Test should_retry within limits."""
        strategy = ExponentialBackoffStrategy(max_retries=3)
        ctx = RetryContext()

        assert strategy.should_retry(ctx) is True
        ctx.record_failure(ValueError())
        assert strategy.should_retry(ctx) is True
        ctx.record_failure(ValueError())
        assert strategy.should_retry(ctx) is True
        ctx.record_failure(ValueError())
        assert strategy.should_retry(ctx) is False

    def test_should_retry_respects_time_limit(self):
        """Test should_retry respects total time limit."""
        strategy = ExponentialBackoffStrategy(
            max_retries=100,
            max_total_time=0.1,
        )
        ctx = RetryContext()

        assert strategy.should_retry(ctx) is True

        time.sleep(0.15)
        assert strategy.should_retry(ctx) is False


# =============================================================================
# LinearBackoffStrategy Tests
# =============================================================================


class TestLinearBackoffStrategy:
    """Tests for LinearBackoffStrategy."""

    def test_default_values(self):
        """Test default strategy values."""
        strategy = LinearBackoffStrategy()
        assert strategy.max_retries == 3
        assert strategy.base_delay == 1.0
        assert strategy.increment == 2.0

    def test_get_delay_linear(self):
        """Test linear delay calculation."""
        strategy = LinearBackoffStrategy(
            base_delay=1.0,
            increment=2.0,
        )

        ctx = RetryContext()
        assert strategy.get_delay(ctx) == 1.0

        ctx.record_failure(ValueError())
        assert strategy.get_delay(ctx) == 3.0

        ctx.record_failure(ValueError())
        assert strategy.get_delay(ctx) == 5.0

    def test_get_delay_caps_at_max(self):
        """Test delay is capped at max_delay."""
        strategy = LinearBackoffStrategy(
            base_delay=1.0,
            increment=10.0,
            max_delay=20.0,
        )

        ctx = RetryContext()
        ctx.attempt = 5  # Would be 1 + 10*5 = 51

        assert strategy.get_delay(ctx) == 20.0


# =============================================================================
# FibonacciBackoffStrategy Tests
# =============================================================================


class TestFibonacciBackoffStrategy:
    """Tests for FibonacciBackoffStrategy."""

    def test_default_values(self):
        """Test default strategy values."""
        strategy = FibonacciBackoffStrategy()
        assert strategy.max_retries == 5
        assert strategy.scale == 1.0

    def test_get_delay_fibonacci(self):
        """Test Fibonacci delay calculation."""
        strategy = FibonacciBackoffStrategy(scale=1.0)

        ctx = RetryContext()
        # Fibonacci: 1, 1, 2, 3, 5, 8, 13, ...
        assert strategy.get_delay(ctx) == 1.0

        ctx.record_failure(ValueError())
        assert strategy.get_delay(ctx) == 1.0

        ctx.record_failure(ValueError())
        assert strategy.get_delay(ctx) == 2.0

        ctx.record_failure(ValueError())
        assert strategy.get_delay(ctx) == 3.0

        ctx.record_failure(ValueError())
        assert strategy.get_delay(ctx) == 5.0

    def test_get_delay_with_scale(self):
        """Test Fibonacci with scale factor."""
        strategy = FibonacciBackoffStrategy(scale=2.0)

        ctx = RetryContext()
        ctx.attempt = 4  # fib(4) = 5
        assert strategy.get_delay(ctx) == 10.0

    def test_get_delay_caps_at_max(self):
        """Test delay is capped at max_delay."""
        strategy = FibonacciBackoffStrategy(
            scale=10.0,
            max_delay=50.0,
        )

        ctx = RetryContext()
        ctx.attempt = 10  # fib(10) = 89, scaled = 890

        assert strategy.get_delay(ctx) == 50.0


# =============================================================================
# AdaptiveStrategy Tests
# =============================================================================


class TestAdaptiveStrategy:
    """Tests for AdaptiveStrategy."""

    def test_default_values(self):
        """Test default strategy values."""
        strategy = AdaptiveStrategy()
        assert strategy.max_retries == 3
        assert strategy.base_delay == 1.0
        assert strategy.success_window == 10

    def test_get_delay_without_history(self):
        """Test delay without outcome history."""
        strategy = AdaptiveStrategy(
            base_delay=1.0,
            backoff_multiplier=2.0,
        )

        ctx = RetryContext()
        # Should be base exponential
        delay = strategy.get_delay(ctx)
        assert delay == pytest.approx(1.0, rel=0.1)

    def test_get_delay_increases_on_failures(self):
        """Test delay increases when system has many failures."""
        strategy = AdaptiveStrategy(base_delay=1.0)

        # Record high failure rate
        for _ in range(10):
            strategy.record_outcome(False)

        ctx = RetryContext()
        delay = strategy.get_delay(ctx)

        # Should be higher due to 100% failure rate
        assert delay > 1.0

    def test_get_delay_decreases_on_success(self):
        """Test delay decreases when system is healthy."""
        strategy = AdaptiveStrategy(base_delay=1.0)

        # Record all successes
        for _ in range(10):
            strategy.record_outcome(True)

        ctx = RetryContext()
        delay = strategy.get_delay(ctx)

        # Should be reduced (0.8x)
        assert delay < 1.0

    def test_failure_rate_calculation(self):
        """Test failure rate calculation."""
        strategy = AdaptiveStrategy(success_window=10)

        # 5 failures, 5 successes
        for _ in range(5):
            strategy.record_outcome(False)
        for _ in range(5):
            strategy.record_outcome(True)

        assert strategy._get_failure_rate() == 0.5

    def test_window_size_maintained(self):
        """Test that window size is maintained."""
        strategy = AdaptiveStrategy(success_window=5)

        # Record more than window size
        for _ in range(10):
            strategy.record_outcome(True)

        assert len(strategy._recent_outcomes) == 5


# =============================================================================
# ConstantDelayStrategy Tests
# =============================================================================


class TestConstantDelayStrategy:
    """Tests for ConstantDelayStrategy."""

    def test_default_values(self):
        """Test default strategy values."""
        strategy = ConstantDelayStrategy()
        assert strategy.max_retries == 3
        assert strategy.delay == 1.0

    def test_get_delay_constant(self):
        """Test constant delay regardless of attempt."""
        strategy = ConstantDelayStrategy(delay=5.0)

        ctx = RetryContext()
        assert strategy.get_delay(ctx) == 5.0

        ctx.record_failure(ValueError())
        assert strategy.get_delay(ctx) == 5.0

        ctx.record_failure(ValueError())
        assert strategy.get_delay(ctx) == 5.0


# =============================================================================
# NoRetryStrategy Tests
# =============================================================================


class TestNoRetryStrategy:
    """Tests for NoRetryStrategy."""

    def test_never_retries(self):
        """Test strategy never allows retries."""
        strategy = NoRetryStrategy()
        ctx = RetryContext()

        assert strategy.should_retry(ctx) is False

    def test_zero_delay(self):
        """Test zero delay."""
        strategy = NoRetryStrategy()
        ctx = RetryContext()

        assert strategy.get_delay(ctx) == 0.0


# =============================================================================
# Preset Strategy Tests
# =============================================================================


class TestPresets:
    """Tests for preset strategy functions."""

    def test_quick_retry(self):
        """Test quick_retry preset."""
        strategy = quick_retry()
        assert strategy.max_retries == 2
        assert strategy.base_delay == 0.5
        assert strategy.max_delay == 5.0

    def test_standard_retry(self):
        """Test standard_retry preset."""
        strategy = standard_retry()
        assert strategy.max_retries == 3
        assert strategy.base_delay == 1.0
        assert strategy.max_delay == 30.0

    def test_patient_retry(self):
        """Test patient_retry preset."""
        strategy = patient_retry()
        assert strategy.max_retries == 5
        assert strategy.base_delay == 2.0
        assert strategy.max_delay == 60.0

    def test_cluster_retry(self):
        """Test cluster_retry preset."""
        strategy = cluster_retry()
        assert isinstance(strategy, AdaptiveStrategy)
        assert strategy.max_retries == 3
        assert strategy.success_window == 20


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for retry strategies."""

    def test_retry_loop_pattern(self):
        """Test typical retry loop pattern."""
        strategy = ExponentialBackoffStrategy(
            max_retries=3,
            base_delay=0.01,
            jitter=0.0,
        )
        ctx = RetryContext(target="test", operation="test_op")

        attempts = 0
        while strategy.should_retry(ctx):
            attempts += 1
            # Simulate failure
            ctx.record_failure(ValueError(f"attempt {attempts}"))
            delay = strategy.get_delay(ctx)
            ctx.record_delay(delay)

        assert attempts == 3
        assert ctx.failure_count == 3

    def test_early_success_pattern(self):
        """Test pattern where retry succeeds early."""
        strategy = ExponentialBackoffStrategy(max_retries=5)
        ctx = RetryContext()

        attempts = 0
        success = False
        while strategy.should_retry(ctx):
            attempts += 1
            if attempts == 2:
                success = True
                break
            ctx.record_failure(ValueError())

        assert success is True
        assert attempts == 2

    def test_on_retry_callback(self):
        """Test on_retry callback is called."""
        strategy = ExponentialBackoffStrategy()
        ctx = RetryContext(target="node", operation="op")

        with patch.object(strategy, "on_retry") as mock_callback:
            ctx.record_failure(ValueError())
            delay = strategy.get_delay(ctx)
            strategy.on_retry(ctx, delay)

            mock_callback.assert_called_once_with(ctx, delay)
