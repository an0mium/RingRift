"""Tests for scripts.lib.retry module.

Tests retry decorators and utilities.
"""

import pytest
from unittest.mock import patch, MagicMock
import time

from scripts.lib.retry import (
    RetryConfig,
    RetryAttempt,
    retry,
    retry_on_exception,
)


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_default_values(self):
        """Test RetryConfig with defaults."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential is True
        assert config.jitter == 0.1

    def test_custom_values(self):
        """Test RetryConfig with custom values."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential=False,
            jitter=0.2,
        )

        assert config.max_attempts == 5
        assert config.base_delay == 0.5
        assert config.exponential is False

    def test_get_delay_first_attempt(self):
        """Test no delay on first attempt."""
        config = RetryConfig(base_delay=1.0)
        assert config.get_delay(0) == 0.0

    def test_get_delay_exponential(self):
        """Test exponential backoff delays."""
        config = RetryConfig(base_delay=1.0, exponential=True, jitter=0.0)

        # Attempt 1: 1.0 * 2^0 = 1.0
        assert config.get_delay(1) == 1.0
        # Attempt 2: 1.0 * 2^1 = 2.0
        assert config.get_delay(2) == 2.0
        # Attempt 3: 1.0 * 2^2 = 4.0
        assert config.get_delay(3) == 4.0

    def test_get_delay_linear(self):
        """Test linear (non-exponential) delays."""
        config = RetryConfig(base_delay=2.0, exponential=False, jitter=0.0)

        assert config.get_delay(1) == 2.0
        assert config.get_delay(2) == 2.0
        assert config.get_delay(3) == 2.0

    def test_get_delay_max_cap(self):
        """Test delay is capped at max_delay."""
        config = RetryConfig(base_delay=10.0, max_delay=15.0, exponential=True, jitter=0.0)

        # Attempt 2: 10 * 2^1 = 20, but capped at 15
        assert config.get_delay(2) == 15.0


class TestRetryAttempt:
    """Tests for RetryAttempt class."""

    def test_attempt_attributes(self):
        """Test RetryAttempt stores attempt info."""
        attempt = RetryAttempt(
            number=2,
            max_attempts=5,
            delay=1.5,
        )

        assert attempt.number == 2
        assert attempt.max_attempts == 5
        assert attempt.delay == 1.5

    def test_should_retry(self):
        """Test should_retry logic."""
        # Can retry
        attempt = RetryAttempt(number=2, max_attempts=5, delay=1.0)
        assert attempt.should_retry is True

        # Last attempt - cannot retry
        last_attempt = RetryAttempt(number=5, max_attempts=5, delay=1.0)
        assert last_attempt.should_retry is False


class TestRetryDecorator:
    """Tests for @retry decorator."""

    def test_success_on_first_try(self):
        """Test function succeeds on first attempt."""
        call_count = 0

        @retry(max_attempts=3)
        def always_succeeds():
            nonlocal call_count
            call_count += 1
            return "success"

        result = always_succeeds()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_failure(self):
        """Test function retries on failure."""
        call_count = 0

        @retry(max_attempts=3, delay=0.01)
        def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "finally"

        result = fails_twice()
        assert result == "finally"
        assert call_count == 3

    def test_raises_after_max_attempts(self):
        """Test exception raised after all attempts fail."""
        call_count = 0

        @retry(max_attempts=3, delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Always fails")

        with pytest.raises(RuntimeError, match="Always fails"):
            always_fails()

        assert call_count == 3


class TestRetryOnException:
    """Tests for @retry_on_exception decorator."""

    def test_retries_on_specified_exception(self):
        """Test retries only on specified exceptions."""
        call_count = 0

        @retry_on_exception(ValueError, max_attempts=3, delay=0.01)
        def fails_with_value_error():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Retryable")
            return "ok"

        result = fails_with_value_error()
        assert result == "ok"
        assert call_count == 2

    def test_no_retry_on_other_exception(self):
        """Test does not retry on non-specified exceptions."""
        call_count = 0

        @retry_on_exception(ValueError, max_attempts=3, delay=0.01)
        def fails_with_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("Not retryable")

        with pytest.raises(TypeError, match="Not retryable"):
            fails_with_type_error()

        assert call_count == 1  # No retries

    def test_multiple_exception_types(self):
        """Test retries on multiple exception types."""
        call_count = 0
        exceptions = [ValueError, TypeError, RuntimeError]

        @retry_on_exception(ValueError, TypeError, max_attempts=4, delay=0.01)
        def alternating_errors():
            nonlocal call_count
            call_count += 1
            # Always raise an exception based on call count
            raise exceptions[(call_count - 1) % len(exceptions)]("Error")

        # Should fail on RuntimeError (not in retry list) after ValueError and TypeError
        with pytest.raises(RuntimeError):
            alternating_errors()

        assert call_count == 3


class TestRetryWithAsync:
    """Tests for async retry functionality."""

    @pytest.mark.asyncio
    async def test_async_retry_success(self):
        """Test async function retries."""
        from scripts.lib.retry import retry_async

        call_count = 0

        @retry_async(max_attempts=3, delay=0.01)
        async def async_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Retry me")
            return "async success"

        result = await async_operation()
        assert result == "async success"
        assert call_count == 2
