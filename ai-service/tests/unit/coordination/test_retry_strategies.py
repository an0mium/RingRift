#!/usr/bin/env python3
"""Unit tests for retry_strategies.py.

Tests all retry strategy implementations and presets:
- RetryContext dataclass
- RetryExhaustedError exception
- ExponentialBackoffStrategy
- LinearBackoffStrategy
- FibonacciBackoffStrategy
- AdaptiveStrategy
- ConstantDelayStrategy
- NoRetryStrategy
- Preset factory functions
"""

import pytest
import time
from unittest.mock import patch

from app.coordination.retry_strategies import (
    # Base classes and data classes
    RetryContext,
    RetryExhaustedError,
    RetryStrategy,
    # Concrete strategies
    ExponentialBackoffStrategy,
    LinearBackoffStrategy,
    FibonacciBackoffStrategy,
    AdaptiveStrategy,
    ConstantDelayStrategy,
    NoRetryStrategy,
    # Presets
    quick_retry,
    standard_retry,
    patient_retry,
    cluster_retry,
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
        assert ctx.last_failure_time == 0.0
        assert ctx.total_delay == 0.0
        assert ctx.metadata == {}
        assert ctx.start_time > 0

    def test_custom_values(self):
        """Test context with custom values."""
        ctx = RetryContext(
            target="node-1",
            operation="transfer",
            metadata={"file": "test.db"},
        )
        
        assert ctx.target == "node-1"
        assert ctx.operation == "transfer"
        assert ctx.metadata == {"file": "test.db"}

    def test_record_failure(self):
        """Test recording failures."""
        ctx = RetryContext()
        
        error = ValueError("test error")
        ctx.record_failure(error)
        
        assert ctx.attempt == 1
        assert ctx.failures == [error]
        assert ctx.last_failure_time > 0

    def test_record_multiple_failures(self):
        """Test recording multiple failures."""
        ctx = RetryContext()
        
        errors = [ValueError("e1"), OSError("e2"), RuntimeError("e3")]
        for e in errors:
            ctx.record_failure(e)
        
        assert ctx.attempt == 3
        assert ctx.failures == errors
        assert ctx.failure_count == 3

    def test_record_delay(self):
        """Test recording delay time."""
        ctx = RetryContext()
        
        ctx.record_delay(1.0)
        ctx.record_delay(2.0)
        ctx.record_delay(4.0)
        
        assert ctx.total_delay == 7.0

    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        ctx = RetryContext()
        time.sleep(0.1)
        
        elapsed = ctx.elapsed_time
        assert elapsed >= 0.1
        assert elapsed < 0.5

    def test_last_error_empty(self):
        """Test last_error with no failures."""
        ctx = RetryContext()
        assert ctx.last_error is None

    def test_last_error_with_failures(self):
        """Test last_error returns most recent."""
        ctx = RetryContext()
        
        ctx.record_failure(ValueError("first"))
        ctx.record_failure(OSError("last"))
        
        assert isinstance(ctx.last_error, OSError)
        assert str(ctx.last_error) == "last"


# =============================================================================
# RetryExhaustedError Tests
# =============================================================================


class TestRetryExhaustedError:
    """Tests for RetryExhaustedError exception."""

    def test_basic_creation(self):
        """Test creating error with context."""
        ctx = RetryContext(target="node-1", operation="sync")
        ctx.record_failure(ValueError("test"))
        
        error = RetryExhaustedError(ctx)
        
        assert error.context == ctx
        assert "sync" in str(error)
        assert "node-1" in str(error)
        assert "attempts=1" in str(error)

    def test_custom_message(self):
        """Test error with custom message."""
        ctx = RetryContext(target="node-1", operation="sync")
        
        error = RetryExhaustedError(ctx, message="Custom failure")
        
        assert "Custom failure" in str(error)

    def test_is_exception(self):
        """Test that error is an exception."""
        ctx = RetryContext()
        error = RetryExhaustedError(ctx)
        
        assert isinstance(error, Exception)
        
        with pytest.raises(RetryExhaustedError):
            raise error


# =============================================================================
# ExponentialBackoffStrategy Tests
# =============================================================================


class TestExponentialBackoffStrategy:
    """Tests for ExponentialBackoffStrategy."""

    def test_default_params(self):
        """Test default strategy parameters."""
        strategy = ExponentialBackoffStrategy()
        
        assert strategy.max_retries == 3
        assert strategy.base_delay == 1.0
        assert strategy.multiplier == 2.0
        assert strategy.max_delay == 60.0
        assert strategy.jitter == 0.1

    def test_custom_params(self):
        """Test custom strategy parameters."""
        strategy = ExponentialBackoffStrategy(
            max_retries=5,
            base_delay=0.5,
            multiplier=3.0,
            max_delay=30.0,
            jitter=0.2,
        )
        
        assert strategy.max_retries == 5
        assert strategy.base_delay == 0.5
        assert strategy.multiplier == 3.0

    def test_delay_exponential_growth(self):
        """Test delays grow exponentially."""
        strategy = ExponentialBackoffStrategy(
            base_delay=1.0,
            multiplier=2.0,
            jitter=0,  # Disable jitter for predictable test
        )
        
        # Expected: 1, 2, 4, 8
        expected_delays = [1.0, 2.0, 4.0, 8.0]
        
        for attempt, expected in enumerate(expected_delays):
            ctx = RetryContext()
            ctx.attempt = attempt
            delay = strategy.get_delay(ctx)
            assert delay == expected

    def test_delay_capped_at_max(self):
        """Test delay is capped at max_delay."""
        strategy = ExponentialBackoffStrategy(
            base_delay=10.0,
            multiplier=2.0,
            max_delay=30.0,
            jitter=0,
        )
        
        ctx = RetryContext()
        ctx.attempt = 5  # Would be 10 * 2^5 = 320
        delay = strategy.get_delay(ctx)
        
        assert delay == 30.0

    def test_jitter_adds_variance(self):
        """Test jitter adds randomness to delay."""
        strategy = ExponentialBackoffStrategy(
            base_delay=10.0,
            multiplier=1.0,  # Keep delay constant
            jitter=0.5,  # 50% jitter
        )
        
        ctx = RetryContext()
        ctx.attempt = 0
        
        # Collect multiple samples
        delays = [strategy.get_delay(ctx) for _ in range(100)]
        
        # Should have variance
        assert min(delays) < 10.0
        assert max(delays) > 10.0
        # All within jitter range
        assert all(5.0 <= d <= 15.0 for d in delays)

    def test_should_retry_within_limits(self):
        """Test should_retry returns True within limits."""
        strategy = ExponentialBackoffStrategy(max_retries=3)
        
        ctx = RetryContext()
        assert strategy.should_retry(ctx)  # attempt=0
        
        ctx.attempt = 1
        assert strategy.should_retry(ctx)
        
        ctx.attempt = 2
        assert strategy.should_retry(ctx)

    def test_should_retry_exhausted(self):
        """Test should_retry returns False when exhausted."""
        strategy = ExponentialBackoffStrategy(max_retries=3)
        
        ctx = RetryContext()
        ctx.attempt = 3
        
        assert not strategy.should_retry(ctx)

    def test_should_retry_time_exceeded(self):
        """Test should_retry respects max_total_time."""
        strategy = ExponentialBackoffStrategy(
            max_retries=100,
            max_total_time=0.1,  # Very short
        )
        
        ctx = RetryContext()
        time.sleep(0.15)  # Exceed max_total_time
        
        assert not strategy.should_retry(ctx)


# =============================================================================
# LinearBackoffStrategy Tests
# =============================================================================


class TestLinearBackoffStrategy:
    """Tests for LinearBackoffStrategy."""

    def test_default_params(self):
        """Test default strategy parameters."""
        strategy = LinearBackoffStrategy()
        
        assert strategy.max_retries == 3
        assert strategy.base_delay == 1.0
        assert strategy.increment == 2.0
        assert strategy.max_delay == 30.0

    def test_delay_linear_growth(self):
        """Test delays grow linearly."""
        strategy = LinearBackoffStrategy(
            base_delay=1.0,
            increment=2.0,
        )
        
        # Expected: 1, 3, 5, 7
        expected_delays = [1.0, 3.0, 5.0, 7.0]
        
        for attempt, expected in enumerate(expected_delays):
            ctx = RetryContext()
            ctx.attempt = attempt
            delay = strategy.get_delay(ctx)
            assert delay == expected

    def test_delay_capped_at_max(self):
        """Test delay is capped at max_delay."""
        strategy = LinearBackoffStrategy(
            base_delay=5.0,
            increment=5.0,
            max_delay=15.0,
        )
        
        ctx = RetryContext()
        ctx.attempt = 10  # Would be 5 + 5*10 = 55
        delay = strategy.get_delay(ctx)
        
        assert delay == 15.0


# =============================================================================
# FibonacciBackoffStrategy Tests
# =============================================================================


class TestFibonacciBackoffStrategy:
    """Tests for FibonacciBackoffStrategy."""

    def test_default_params(self):
        """Test default strategy parameters."""
        strategy = FibonacciBackoffStrategy()
        
        assert strategy.max_retries == 5
        assert strategy.scale == 1.0
        assert strategy.max_delay == 60.0

    def test_fibonacci_sequence(self):
        """Test delays follow Fibonacci sequence."""
        strategy = FibonacciBackoffStrategy(scale=1.0)
        
        # Fibonacci: 1, 1, 2, 3, 5, 8, 13
        expected_delays = [1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0]
        
        for attempt, expected in enumerate(expected_delays):
            ctx = RetryContext()
            ctx.attempt = attempt
            delay = strategy.get_delay(ctx)
            assert delay == expected

    def test_fibonacci_with_scale(self):
        """Test Fibonacci delays with scale multiplier."""
        strategy = FibonacciBackoffStrategy(scale=2.0)
        
        # Fibonacci * 2: 2, 2, 4, 6, 10
        expected_delays = [2.0, 2.0, 4.0, 6.0, 10.0]
        
        for attempt, expected in enumerate(expected_delays):
            ctx = RetryContext()
            ctx.attempt = attempt
            delay = strategy.get_delay(ctx)
            assert delay == expected

    def test_fibonacci_capped_at_max(self):
        """Test Fibonacci delay is capped at max_delay."""
        strategy = FibonacciBackoffStrategy(
            scale=1.0,
            max_delay=10.0,
        )
        
        ctx = RetryContext()
        ctx.attempt = 10  # Fib(10) = 55
        delay = strategy.get_delay(ctx)
        
        assert delay == 10.0

    def test_fibonacci_caching(self):
        """Test Fibonacci values are cached."""
        strategy = FibonacciBackoffStrategy()
        
        # Access high indices to populate cache
        ctx = RetryContext()
        ctx.attempt = 15
        strategy.get_delay(ctx)
        
        # Cache should be populated
        assert len(strategy._fib_cache) >= 16


# =============================================================================
# AdaptiveStrategy Tests
# =============================================================================


class TestAdaptiveStrategy:
    """Tests for AdaptiveStrategy."""

    def test_default_params(self):
        """Test default strategy parameters."""
        strategy = AdaptiveStrategy()
        
        assert strategy.max_retries == 3
        assert strategy.base_delay == 1.0
        assert strategy.min_delay == 0.5
        assert strategy.backoff_multiplier == 2.0
        assert strategy.success_window == 10

    def test_base_exponential_delay(self):
        """Test base delay follows exponential pattern."""
        strategy = AdaptiveStrategy(
            base_delay=1.0,
            backoff_multiplier=2.0,
        )
        
        # With no outcomes, should be exponential
        ctx = RetryContext()
        ctx.attempt = 0
        delay0 = strategy.get_delay(ctx)
        
        ctx.attempt = 1
        delay1 = strategy.get_delay(ctx)
        
        ctx.attempt = 2
        delay2 = strategy.get_delay(ctx)
        
        # Should roughly double each time (with small adjustments)
        assert delay1 > delay0
        assert delay2 > delay1

    def test_record_outcome(self):
        """Test recording outcomes updates history."""
        strategy = AdaptiveStrategy(success_window=5)
        
        assert strategy._recent_outcomes == []
        
        strategy.record_outcome(True)
        strategy.record_outcome(False)
        strategy.record_outcome(True)
        
        assert len(strategy._recent_outcomes) == 3
        assert strategy._recent_outcomes == [True, False, True]

    def test_outcome_window_limit(self):
        """Test outcome history respects window size."""
        strategy = AdaptiveStrategy(success_window=3)
        
        for _ in range(5):
            strategy.record_outcome(True)
        
        assert len(strategy._recent_outcomes) == 3

    def test_failure_rate_calculation(self):
        """Test failure rate calculation."""
        strategy = AdaptiveStrategy()
        
        # 50% failure rate
        strategy.record_outcome(True)
        strategy.record_outcome(False)
        
        rate = strategy._get_failure_rate()
        assert rate == 0.5

    def test_high_failure_increases_delay(self):
        """Test high failure rate increases delay."""
        strategy = AdaptiveStrategy(
            base_delay=1.0,
            backoff_multiplier=1.0,  # No exponential growth
        )
        
        # Record all failures
        for _ in range(10):
            strategy.record_outcome(False)
        
        ctx = RetryContext()
        ctx.attempt = 0
        delay = strategy.get_delay(ctx)
        
        # Should be higher than base_delay due to high failure rate
        assert delay > 1.0

    def test_low_failure_decreases_delay(self):
        """Test low failure rate decreases delay."""
        strategy = AdaptiveStrategy(
            base_delay=1.0,
            backoff_multiplier=1.0,
        )
        
        # Record all successes
        for _ in range(10):
            strategy.record_outcome(True)
        
        ctx = RetryContext()
        ctx.attempt = 0
        delay = strategy.get_delay(ctx)
        
        # Should be lower than base_delay due to healthy system
        assert delay < 1.0

    def test_delay_respects_bounds(self):
        """Test delay stays within min/max bounds."""
        strategy = AdaptiveStrategy(
            base_delay=1.0,
            min_delay=0.5,
            max_delay=10.0,
        )
        
        # Force very low delay scenario
        for _ in range(20):
            strategy.record_outcome(True)
        
        ctx = RetryContext()
        ctx.attempt = 0
        delay = strategy.get_delay(ctx)
        
        assert delay >= 0.5  # min_delay
        assert delay <= 10.0  # max_delay


# =============================================================================
# ConstantDelayStrategy Tests
# =============================================================================


class TestConstantDelayStrategy:
    """Tests for ConstantDelayStrategy."""

    def test_default_params(self):
        """Test default strategy parameters."""
        strategy = ConstantDelayStrategy()
        
        assert strategy.max_retries == 3
        assert strategy.delay == 1.0

    def test_constant_delay(self):
        """Test delay is always constant."""
        strategy = ConstantDelayStrategy(delay=5.0)
        
        for attempt in range(10):
            ctx = RetryContext()
            ctx.attempt = attempt
            delay = strategy.get_delay(ctx)
            assert delay == 5.0


# =============================================================================
# NoRetryStrategy Tests
# =============================================================================


class TestNoRetryStrategy:
    """Tests for NoRetryStrategy."""

    def test_init(self):
        """Test initialization sets zero limits."""
        strategy = NoRetryStrategy()
        
        assert strategy.max_retries == 0
        assert strategy.max_delay == 0
        assert strategy.max_total_time == 0

    def test_never_retry(self):
        """Test should_retry always returns False."""
        strategy = NoRetryStrategy()
        
        ctx = RetryContext()
        assert not strategy.should_retry(ctx)
        
        ctx.attempt = 0
        assert not strategy.should_retry(ctx)

    def test_zero_delay(self):
        """Test get_delay returns zero."""
        strategy = NoRetryStrategy()
        
        ctx = RetryContext()
        assert strategy.get_delay(ctx) == 0.0


# =============================================================================
# Strategy Preset Tests
# =============================================================================


class TestStrategyPresets:
    """Tests for strategy preset factory functions."""

    def test_quick_retry_config(self):
        """Test quick_retry preset configuration."""
        strategy = quick_retry()
        
        assert isinstance(strategy, ExponentialBackoffStrategy)
        assert strategy.max_retries == 2
        assert strategy.base_delay == 0.5
        assert strategy.max_delay == 5.0
        assert strategy.max_total_time == 15.0

    def test_standard_retry_config(self):
        """Test standard_retry preset configuration."""
        strategy = standard_retry()
        
        assert isinstance(strategy, ExponentialBackoffStrategy)
        assert strategy.max_retries == 3
        assert strategy.base_delay == 1.0
        assert strategy.max_delay == 30.0
        assert strategy.max_total_time == 60.0

    def test_patient_retry_config(self):
        """Test patient_retry preset configuration."""
        strategy = patient_retry()
        
        assert isinstance(strategy, ExponentialBackoffStrategy)
        assert strategy.max_retries == 5
        assert strategy.base_delay == 2.0
        assert strategy.max_delay == 60.0
        assert strategy.max_total_time == 300.0

    def test_cluster_retry_config(self):
        """Test cluster_retry preset configuration."""
        strategy = cluster_retry()
        
        assert isinstance(strategy, AdaptiveStrategy)
        assert strategy.max_retries == 3
        assert strategy.base_delay == 1.0
        assert strategy.max_delay == 30.0
        assert strategy.success_window == 20


# =============================================================================
# Integration Tests
# =============================================================================


class TestStrategyIntegration:
    """Integration tests for retry strategy usage patterns."""

    def test_retry_loop_pattern(self):
        """Test typical retry loop usage pattern."""
        strategy = ExponentialBackoffStrategy(
            max_retries=3,
            base_delay=0.01,  # Fast for testing
            jitter=0,
        )
        
        ctx = RetryContext(target="test", operation="test_op")
        attempts = 0
        
        while strategy.should_retry(ctx):
            attempts += 1
            try:
                if attempts < 3:
                    raise ValueError(f"Attempt {attempts} failed")
                # Success on 3rd attempt
                break
            except Exception as e:
                ctx.record_failure(e)
        
        assert attempts == 3
        assert ctx.attempt == 2  # 2 failures recorded

    def test_exhausted_raises_error(self):
        """Test raising RetryExhaustedError after exhaustion."""
        strategy = NoRetryStrategy()
        ctx = RetryContext(target="test", operation="test_op")
        
        # First check should already fail
        if not strategy.should_retry(ctx):
            with pytest.raises(RetryExhaustedError) as exc_info:
                raise RetryExhaustedError(ctx)
            
            assert "test_op" in str(exc_info.value)

    def test_on_retry_callback(self):
        """Test on_retry callback is called."""
        strategy = ExponentialBackoffStrategy()
        ctx = RetryContext()
        
        # Capture callback calls
        callback_called = []
        original_on_retry = strategy.on_retry
        
        def mock_on_retry(ctx, delay):
            callback_called.append((ctx, delay))
            original_on_retry(ctx, delay)
        
        strategy.on_retry = mock_on_retry
        
        delay = strategy.get_delay(ctx)
        strategy.on_retry(ctx, delay)
        
        assert len(callback_called) == 1
