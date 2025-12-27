"""Tests for app.core.error_handler module.

Comprehensive tests for error handling, retry logic, and emergency halt functionality.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.core.error_handler import (
    AGGRESSIVE_RETRY_POLICY,
    CONSERVATIVE_RETRY_POLICY,
    DATABASE_RETRY_POLICY,
    DEFAULT_RETRY_POLICY,
    EMERGENCY_HALT_FILE,
    FAST_RETRY_POLICY,
    HTTP_RETRY_POLICY,
    SSH_RETRY_POLICY,
    SYNC_RETRY_POLICY,
    EmergencyHaltError,
    ErrorAggregator,
    FatalError,
    RetryPolicy,
    RetryStrategy,
    RetryableError,
    RingRiftError,
    check_emergency_halt,
    clear_emergency_halt,
    retry,
    retry_async,
    safe_execute,
    safe_execute_async,
    set_emergency_halt,
    with_emergency_halt_check,
    with_emergency_halt_check_async,
    with_retry_policy,
    with_retry_policy_async,
)


# =============================================================================
# Test Emergency Halt Functions
# =============================================================================


class TestEmergencyHaltFunctions:
    """Tests for emergency halt functionality."""

    def setup_method(self):
        """Clean up halt file before each test."""
        if EMERGENCY_HALT_FILE.exists():
            EMERGENCY_HALT_FILE.unlink()

    def teardown_method(self):
        """Clean up halt file after each test."""
        if EMERGENCY_HALT_FILE.exists():
            EMERGENCY_HALT_FILE.unlink()

    def test_check_emergency_halt_returns_false_when_not_set(self):
        """check_emergency_halt returns False when halt file doesn't exist."""
        assert check_emergency_halt() is False

    def test_set_emergency_halt_creates_file(self):
        """set_emergency_halt creates the halt file."""
        set_emergency_halt("Test halt")
        assert EMERGENCY_HALT_FILE.exists()

    def test_check_emergency_halt_returns_true_when_set(self):
        """check_emergency_halt returns True after halt is set."""
        set_emergency_halt("Test halt")
        assert check_emergency_halt() is True

    def test_set_emergency_halt_writes_reason(self):
        """set_emergency_halt writes reason to file."""
        set_emergency_halt("Custom reason")
        content = EMERGENCY_HALT_FILE.read_text()
        assert "Custom reason" in content

    def test_clear_emergency_halt_removes_file(self):
        """clear_emergency_halt removes the halt file."""
        set_emergency_halt("Test")
        assert clear_emergency_halt() is True
        assert not EMERGENCY_HALT_FILE.exists()

    def test_clear_emergency_halt_returns_false_when_not_set(self):
        """clear_emergency_halt returns False when no file exists."""
        assert clear_emergency_halt() is False


class TestWithEmergencyHaltCheck:
    """Tests for the with_emergency_halt_check decorator."""

    def setup_method(self):
        """Clean up halt file before each test."""
        if EMERGENCY_HALT_FILE.exists():
            EMERGENCY_HALT_FILE.unlink()

    def teardown_method(self):
        """Clean up halt file after each test."""
        if EMERGENCY_HALT_FILE.exists():
            EMERGENCY_HALT_FILE.unlink()

    def test_decorator_allows_call_when_no_halt(self):
        """Function runs normally when no halt is set."""
        @with_emergency_halt_check
        def my_func():
            return 42

        assert my_func() == 42

    def test_decorator_raises_when_halt_is_set(self):
        """Function raises EmergencyHaltError when halt is set."""
        set_emergency_halt("Testing")

        @with_emergency_halt_check
        def my_func():
            return 42

        with pytest.raises(EmergencyHaltError) as exc_info:
            my_func()
        assert "my_func" in str(exc_info.value)

    def test_decorator_preserves_function_metadata(self):
        """Decorator preserves function name and docstring."""
        @with_emergency_halt_check
        def documented_func():
            """This is a docstring."""
            pass

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a docstring."


class TestWithEmergencyHaltCheckAsync:
    """Tests for the async emergency halt decorator."""

    def setup_method(self):
        """Clean up halt file before each test."""
        if EMERGENCY_HALT_FILE.exists():
            EMERGENCY_HALT_FILE.unlink()

    def teardown_method(self):
        """Clean up halt file after each test."""
        if EMERGENCY_HALT_FILE.exists():
            EMERGENCY_HALT_FILE.unlink()

    @pytest.mark.asyncio
    async def test_async_decorator_allows_call_when_no_halt(self):
        """Async function runs normally when no halt is set."""
        @with_emergency_halt_check_async
        async def my_async_func():
            return 42

        result = await my_async_func()
        assert result == 42

    @pytest.mark.asyncio
    async def test_async_decorator_raises_when_halt_is_set(self):
        """Async function raises EmergencyHaltError when halt is set."""
        set_emergency_halt("Testing")

        @with_emergency_halt_check_async
        async def my_async_func():
            return 42

        with pytest.raises(EmergencyHaltError):
            await my_async_func()


# =============================================================================
# Test Retry Decorator (Sync)
# =============================================================================


class TestRetryDecorator:
    """Tests for the @retry decorator."""

    def test_retry_succeeds_on_first_try(self):
        """Function succeeds on first try, no retries needed."""
        call_count = 0

        @retry(max_attempts=3, delay=0.01)
        def succeed_first():
            nonlocal call_count
            call_count += 1
            return "success"

        result = succeed_first()
        assert result == "success"
        assert call_count == 1

    def test_retry_succeeds_after_failures(self):
        """Function succeeds after some failures."""
        call_count = 0

        @retry(max_attempts=3, delay=0.01)
        def succeed_on_third():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"

        result = succeed_on_third()
        assert result == "success"
        assert call_count == 3

    def test_retry_raises_after_max_attempts(self):
        """Function raises after exhausting all attempts."""
        call_count = 0

        @retry(max_attempts=3, delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            always_fails()
        assert call_count == 3

    def test_retry_respects_exceptions_filter(self):
        """Retry only catches specified exception types."""
        call_count = 0

        @retry(max_attempts=3, delay=0.01, exceptions=(ValueError,))
        def raises_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("Wrong type")

        with pytest.raises(TypeError):
            raises_type_error()
        assert call_count == 1  # No retry for TypeError

    def test_retry_calls_on_retry_callback(self):
        """on_retry callback is called on each retry."""
        callbacks = []

        def callback(exc, attempt):
            callbacks.append((str(exc), attempt))

        @retry(max_attempts=3, delay=0.01, on_retry=callback)
        def fails_twice():
            if len(callbacks) < 2:
                raise ValueError("Fail")
            return "success"

        fails_twice()
        assert len(callbacks) == 2
        assert callbacks[0][1] == 1
        assert callbacks[1][1] == 2

    def test_retry_returns_none_when_reraise_false(self):
        """Returns None instead of raising when reraise=False."""
        @retry(max_attempts=2, delay=0.01, reraise=False)
        def always_fails():
            raise ValueError("Fail")

        result = always_fails()
        assert result is None

    def test_retry_never_retries_fatal_error(self):
        """FatalError is never retried."""
        call_count = 0

        @retry(max_attempts=5, delay=0.01)
        def raises_fatal():
            nonlocal call_count
            call_count += 1
            raise FatalError("Fatal")

        with pytest.raises(FatalError):
            raises_fatal()
        assert call_count == 1  # No retries

    def test_retry_applies_backoff(self):
        """Delay increases with backoff multiplier."""
        start_times = []

        @retry(max_attempts=3, delay=0.05, backoff=2.0)
        def record_times():
            start_times.append(time.time())
            if len(start_times) < 3:
                raise ValueError("Fail")
            return "done"

        record_times()
        assert len(start_times) == 3
        # Second gap should be ~2x the first gap (with some tolerance)
        gap1 = start_times[1] - start_times[0]
        gap2 = start_times[2] - start_times[1]
        assert gap2 > gap1 * 1.5  # Allow some tolerance

    def test_retry_respects_max_delay(self):
        """Delay is capped at max_delay."""
        delays = []

        @retry(max_attempts=4, delay=0.05, backoff=10.0, max_delay=0.05)
        def track_delays():
            delays.append(time.time())
            if len(delays) < 4:
                raise ValueError("Fail")
            return "done"

        start = time.time()
        track_delays()
        total_time = time.time() - start
        # With max_delay=0.05 and 3 retries, should take ~0.15s max
        assert total_time < 0.5

    def test_retry_with_jitter(self):
        """Jitter adds randomness to delays."""
        @retry(max_attempts=3, delay=0.05, jitter=True)
        def fails_twice():
            if not hasattr(fails_twice, "count"):
                fails_twice.count = 0
            fails_twice.count += 1
            if fails_twice.count < 3:
                raise ValueError("Fail")
            return "success"

        # Just verify it runs without error - jitter is random
        result = fails_twice()
        assert result == "success"


# =============================================================================
# Test Retry Async Decorator
# =============================================================================


class TestRetryAsyncDecorator:
    """Tests for the @retry_async decorator."""

    @pytest.mark.asyncio
    async def test_async_retry_succeeds_on_first_try(self):
        """Async function succeeds on first try."""
        call_count = 0

        @retry_async(max_attempts=3, delay=0.01)
        async def succeed_first():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await succeed_first()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_retry_succeeds_after_failures(self):
        """Async function succeeds after failures."""
        call_count = 0

        @retry_async(max_attempts=3, delay=0.01)
        async def succeed_on_third():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"

        result = await succeed_on_third()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_async_retry_raises_after_max_attempts(self):
        """Async function raises after exhausting attempts."""
        call_count = 0

        @retry_async(max_attempts=3, delay=0.01)
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            await always_fails()
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_async_retry_with_jitter(self):
        """Async retry applies jitter to delays."""
        @retry_async(max_attempts=3, delay=0.01, jitter=True)
        async def fails_twice():
            if not hasattr(fails_twice, "count"):
                fails_twice.count = 0
            fails_twice.count += 1
            if fails_twice.count < 3:
                raise ValueError("Fail")
            return "success"

        result = await fails_twice()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_async_retry_never_retries_fatal_error(self):
        """FatalError is never retried in async."""
        call_count = 0

        @retry_async(max_attempts=5, delay=0.01)
        async def raises_fatal():
            nonlocal call_count
            call_count += 1
            raise FatalError("Fatal")

        with pytest.raises(FatalError):
            await raises_fatal()
        assert call_count == 1


# =============================================================================
# Test RetryStrategy Enum
# =============================================================================


class TestRetryStrategy:
    """Tests for RetryStrategy enum."""

    def test_strategy_values(self):
        """Verify strategy enum values."""
        assert RetryStrategy.LINEAR.value == "linear"
        assert RetryStrategy.EXPONENTIAL.value == "exponential"
        assert RetryStrategy.EXPONENTIAL_JITTER.value == "exponential_jitter"

    def test_strategy_from_string(self):
        """Can create strategy from string value."""
        assert RetryStrategy("linear") == RetryStrategy.LINEAR
        assert RetryStrategy("exponential") == RetryStrategy.EXPONENTIAL


# =============================================================================
# Test RetryPolicy Dataclass
# =============================================================================


class TestRetryPolicy:
    """Tests for RetryPolicy dataclass."""

    def test_default_values(self):
        """Verify default policy values."""
        policy = RetryPolicy()
        assert policy.strategy == RetryStrategy.EXPONENTIAL
        assert policy.max_attempts == 3
        assert policy.initial_delay == 1.0
        assert policy.multiplier == 2.0
        assert policy.max_delay == 60.0

    def test_get_delay_linear(self):
        """Linear strategy uses constant delay."""
        policy = RetryPolicy(strategy=RetryStrategy.LINEAR, initial_delay=1.0)
        assert policy.get_delay(0) == 1.0
        assert policy.get_delay(1) == 1.0
        assert policy.get_delay(5) == 1.0

    def test_get_delay_exponential(self):
        """Exponential strategy doubles delay each attempt."""
        policy = RetryPolicy(
            strategy=RetryStrategy.EXPONENTIAL,
            initial_delay=1.0,
            multiplier=2.0,
        )
        assert policy.get_delay(0) == 1.0
        assert policy.get_delay(1) == 2.0
        assert policy.get_delay(2) == 4.0
        assert policy.get_delay(3) == 8.0

    def test_get_delay_respects_max_delay(self):
        """Delay is capped at max_delay."""
        policy = RetryPolicy(
            strategy=RetryStrategy.EXPONENTIAL,
            initial_delay=10.0,
            multiplier=2.0,
            max_delay=20.0,
        )
        assert policy.get_delay(0) == 10.0
        assert policy.get_delay(1) == 20.0
        assert policy.get_delay(2) == 20.0  # Capped

    def test_get_delay_exponential_jitter(self):
        """Exponential jitter adds randomness within range."""
        policy = RetryPolicy(
            strategy=RetryStrategy.EXPONENTIAL_JITTER,
            initial_delay=10.0,
            multiplier=1.0,  # No exponential growth for predictable test
            jitter_factor=0.2,
        )
        # With jitter_factor=0.2, delay should be in [8.0, 12.0]
        delays = [policy.get_delay(0) for _ in range(10)]
        assert all(8.0 <= d <= 12.0 for d in delays)

    def test_to_retry_kwargs(self):
        """to_retry_kwargs produces correct dict."""
        policy = RetryPolicy(
            strategy=RetryStrategy.EXPONENTIAL,
            max_attempts=5,
            initial_delay=0.5,
            multiplier=3.0,
            max_delay=30.0,
        )
        kwargs = policy.to_retry_kwargs()
        assert kwargs == {
            "max_attempts": 5,
            "delay": 0.5,
            "backoff": 3.0,
            "max_delay": 30.0,
        }

    def test_to_retry_kwargs_linear_has_backoff_1(self):
        """Linear strategy produces backoff=1.0."""
        policy = RetryPolicy(strategy=RetryStrategy.LINEAR, multiplier=5.0)
        kwargs = policy.to_retry_kwargs()
        assert kwargs["backoff"] == 1.0

    def test_from_config(self):
        """Can create policy from config dict."""
        config = {
            "strategy": "exponential_jitter",
            "max_attempts": 10,
            "initial_delay": 0.25,
            "multiplier": 1.5,
            "max_delay": 120.0,
            "jitter_factor": 0.3,
        }
        policy = RetryPolicy.from_config(config)
        assert policy.strategy == RetryStrategy.EXPONENTIAL_JITTER
        assert policy.max_attempts == 10
        assert policy.initial_delay == 0.25
        assert policy.multiplier == 1.5
        assert policy.max_delay == 120.0
        assert policy.jitter_factor == 0.3

    def test_from_config_with_defaults(self):
        """from_config uses defaults for missing values."""
        policy = RetryPolicy.from_config({})
        assert policy.max_attempts == 3
        assert policy.initial_delay == 1.0


# =============================================================================
# Test Predefined Retry Policies
# =============================================================================


class TestPredefinedRetryPolicies:
    """Tests for predefined retry policies."""

    def test_default_policy_exists(self):
        """DEFAULT_RETRY_POLICY is properly configured."""
        assert DEFAULT_RETRY_POLICY.strategy == RetryStrategy.EXPONENTIAL
        assert DEFAULT_RETRY_POLICY.max_attempts >= 3

    def test_aggressive_policy(self):
        """AGGRESSIVE_RETRY_POLICY has more attempts."""
        assert AGGRESSIVE_RETRY_POLICY.max_attempts >= DEFAULT_RETRY_POLICY.max_attempts
        assert AGGRESSIVE_RETRY_POLICY.strategy == RetryStrategy.EXPONENTIAL_JITTER

    def test_conservative_policy(self):
        """CONSERVATIVE_RETRY_POLICY has longer delays."""
        assert CONSERVATIVE_RETRY_POLICY.initial_delay >= DEFAULT_RETRY_POLICY.initial_delay

    def test_fast_policy(self):
        """FAST_RETRY_POLICY has short delays."""
        assert FAST_RETRY_POLICY.strategy == RetryStrategy.LINEAR
        assert FAST_RETRY_POLICY.initial_delay <= 1.0

    def test_sync_policy(self):
        """SYNC_RETRY_POLICY uses jitter."""
        assert SYNC_RETRY_POLICY.strategy == RetryStrategy.EXPONENTIAL_JITTER

    def test_http_policy(self):
        """HTTP_RETRY_POLICY uses jitter for thundering herd prevention."""
        assert HTTP_RETRY_POLICY.strategy == RetryStrategy.EXPONENTIAL_JITTER

    def test_ssh_policy(self):
        """SSH_RETRY_POLICY has longer initial delay."""
        assert SSH_RETRY_POLICY.initial_delay >= 1.0

    def test_database_policy(self):
        """DATABASE_RETRY_POLICY has fast initial delay for lock contention."""
        assert DATABASE_RETRY_POLICY.initial_delay < 1.0
        assert DATABASE_RETRY_POLICY.max_attempts >= 5


class TestWithRetryPolicy:
    """Tests for with_retry_policy decorator factory."""

    def test_with_retry_policy_applies_policy(self):
        """with_retry_policy applies the given policy."""
        policy = RetryPolicy(max_attempts=2, initial_delay=0.01)
        call_count = 0

        @with_retry_policy(policy)
        def my_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Fail")
            return "success"

        result = my_func()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_with_retry_policy_async(self):
        """with_retry_policy_async applies policy to async function."""
        policy = RetryPolicy(max_attempts=2, initial_delay=0.01)
        call_count = 0

        @with_retry_policy_async(policy)
        async def my_async_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Fail")
            return "success"

        result = await my_async_func()
        assert result == "success"
        assert call_count == 2


# =============================================================================
# Test Safe Execute Functions
# =============================================================================


class TestSafeExecute:
    """Tests for safe_execute function."""

    def test_safe_execute_returns_result_on_success(self):
        """safe_execute returns function result on success."""
        def add(a, b):
            return a + b

        result = safe_execute(add, 2, 3)
        assert result == 5

    def test_safe_execute_returns_default_on_error(self):
        """safe_execute returns default on error."""
        def fails():
            raise ValueError("Oops")

        result = safe_execute(fails, default="fallback")
        assert result == "fallback"

    def test_safe_execute_returns_none_without_default(self):
        """safe_execute returns None if no default specified."""
        def fails():
            raise ValueError("Oops")

        result = safe_execute(fails)
        assert result is None

    def test_safe_execute_logs_error_by_default(self):
        """safe_execute logs errors by default."""
        def fails():
            raise ValueError("Oops")

        with patch("app.core.error_handler.logger") as mock_logger:
            safe_execute(fails)
            mock_logger.warning.assert_called_once()

    def test_safe_execute_no_log_when_disabled(self):
        """safe_execute doesn't log when log_errors=False."""
        def fails():
            raise ValueError("Oops")

        with patch("app.core.error_handler.logger") as mock_logger:
            safe_execute(fails, log_errors=False)
            mock_logger.warning.assert_not_called()

    def test_safe_execute_with_kwargs(self):
        """safe_execute passes kwargs correctly."""
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = safe_execute(greet, "World", greeting="Hi")
        assert result == "Hi, World!"


class TestSafeExecuteAsync:
    """Tests for safe_execute_async function."""

    @pytest.mark.asyncio
    async def test_safe_execute_async_returns_result(self):
        """safe_execute_async returns result on success."""
        async def async_add(a, b):
            return a + b

        result = await safe_execute_async(async_add, 2, 3)
        assert result == 5

    @pytest.mark.asyncio
    async def test_safe_execute_async_returns_default_on_error(self):
        """safe_execute_async returns default on error."""
        async def async_fails():
            raise ValueError("Oops")

        result = await safe_execute_async(async_fails, default="fallback")
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_safe_execute_async_logs_error(self):
        """safe_execute_async logs errors by default."""
        async def async_fails():
            raise ValueError("Oops")

        with patch("app.core.error_handler.logger") as mock_logger:
            await safe_execute_async(async_fails)
            mock_logger.warning.assert_called_once()


# =============================================================================
# Test ErrorAggregator
# =============================================================================


class TestErrorAggregator:
    """Tests for ErrorAggregator class."""

    def test_new_aggregator_has_no_errors(self):
        """New ErrorAggregator has no errors."""
        agg = ErrorAggregator("test")
        assert agg.has_errors is False
        assert agg.count == 0

    def test_add_error_increases_count(self):
        """Adding errors increases count."""
        agg = ErrorAggregator("test")
        agg.add(ValueError("Error 1"))
        assert agg.count == 1
        agg.add(ValueError("Error 2"))
        assert agg.count == 2

    def test_has_errors_true_after_add(self):
        """has_errors is True after adding an error."""
        agg = ErrorAggregator("test")
        agg.add(ValueError("Error"))
        assert agg.has_errors is True

    def test_add_with_context(self):
        """Errors can be added with context."""
        agg = ErrorAggregator("test")
        agg.add(ValueError("Error"), context={"item": "foo"})
        assert agg.errors[0][1] == {"item": "foo"}

    def test_summary_no_errors(self):
        """Summary when no errors."""
        agg = ErrorAggregator("batch processing")
        summary = agg.summary()
        assert "No errors" in summary

    def test_summary_with_errors(self):
        """Summary includes error details."""
        agg = ErrorAggregator("batch processing")
        agg.add(ValueError("Error 1"), context={"item": 1})
        agg.add(TypeError("Error 2"))

        summary = agg.summary()
        assert "batch processing" in summary
        assert "2 error(s)" in summary
        assert "ValueError" in summary
        assert "TypeError" in summary

    def test_summary_truncates_at_10_errors(self):
        """Summary truncates to first 10 errors."""
        agg = ErrorAggregator("test")
        for i in range(15):
            agg.add(ValueError(f"Error {i}"))

        summary = agg.summary()
        assert "and 5 more" in summary

    def test_raise_if_any_does_nothing_when_no_errors(self):
        """raise_if_any doesn't raise when no errors."""
        agg = ErrorAggregator("test")
        agg.raise_if_any()  # Should not raise

    def test_raise_if_any_raises_when_errors_exist(self):
        """raise_if_any raises when errors exist."""
        agg = ErrorAggregator("test")
        agg.add(ValueError("Error"))

        with pytest.raises(RingRiftError):
            agg.raise_if_any()

    def test_raise_if_any_uses_custom_error_class(self):
        """raise_if_any uses specified error class."""
        agg = ErrorAggregator("test")
        agg.add(ValueError("Error"))

        with pytest.raises(ValueError):
            agg.raise_if_any(error_class=ValueError)


# =============================================================================
# Test Exception Types
# =============================================================================


class TestExceptionTypes:
    """Tests for exception types exported from error_handler."""

    def test_ringrift_error_is_base(self):
        """RingRiftError is the base exception."""
        err = RingRiftError("Test error")
        assert isinstance(err, Exception)
        assert err.message == "Test error"

    def test_retryable_error(self):
        """RetryableError indicates retriable condition."""
        err = RetryableError("Temporary failure")
        assert isinstance(err, RingRiftError)

    def test_fatal_error(self):
        """FatalError indicates non-retriable condition."""
        err = FatalError("Critical failure")
        assert isinstance(err, RingRiftError)

    def test_emergency_halt_error(self):
        """EmergencyHaltError indicates halt condition."""
        err = EmergencyHaltError("Halted")
        assert isinstance(err, RingRiftError)


# =============================================================================
# Integration Tests
# =============================================================================


class TestRetryIntegration:
    """Integration tests for retry functionality."""

    def test_retry_with_mixed_exceptions(self):
        """Retry handles a realistic scenario with mixed exceptions."""
        attempts = []

        @retry(max_attempts=5, delay=0.01, exceptions=(IOError, TimeoutError))
        def flaky_network_call():
            attempts.append(len(attempts))
            if len(attempts) == 1:
                raise IOError("Network error")
            if len(attempts) == 2:
                raise TimeoutError("Timed out")
            return "success"

        result = flaky_network_call()
        assert result == "success"
        assert len(attempts) == 3

    @pytest.mark.asyncio
    async def test_async_retry_without_circuit_breaker_key(self):
        """Async retry works when no circuit breaker key is provided."""
        @retry_async(max_attempts=2, delay=0.01)
        async def my_func():
            return "success"

        result = await my_func()
        assert result == "success"


class TestRetryPolicyIntegration:
    """Integration tests for RetryPolicy with decorators."""

    def test_policy_applied_to_function(self):
        """RetryPolicy integrates correctly with retry decorator."""
        policy = RetryPolicy(
            strategy=RetryStrategy.EXPONENTIAL,
            max_attempts=4,
            initial_delay=0.01,
            multiplier=2.0,
            max_delay=0.1,
        )

        call_count = 0

        @retry(**policy.to_retry_kwargs())
        def my_func():
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise ValueError("Not yet")
            return "done"

        result = my_func()
        assert result == "done"
        assert call_count == 4
