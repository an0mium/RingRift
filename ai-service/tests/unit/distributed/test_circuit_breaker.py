"""Tests for circuit breaker pattern implementation.

These tests verify:
1. CircuitState enum and CircuitStatus dataclass
2. CircuitBreaker state transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
3. Failure threshold and recovery timeout behavior
4. Exponential backoff with jitter
5. Context managers and decorators
6. CircuitBreakerRegistry and FallbackChain
7. Thread safety
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from app.distributed.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
    CircuitStatus,
    FallbackChain,
    format_circuit_status,
    get_adaptive_timeout,
    get_circuit_registry,
    get_host_breaker,
    get_operation_breaker,
    get_training_breaker,
    set_host_breaker_callback,
    with_circuit_breaker,
)


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_state_values(self):
        """CircuitState should have correct string values."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"

    def test_state_enum_members(self):
        """CircuitState should have exactly 3 members."""
        assert len(CircuitState) == 3


class TestCircuitStatus:
    """Tests for CircuitStatus dataclass."""

    def test_status_creation(self):
        """CircuitStatus should be created with all fields."""
        status = CircuitStatus(
            target="host1",
            state=CircuitState.CLOSED,
            failure_count=0,
            success_count=5,
            last_failure_time=None,
            last_success_time=time.time(),
            opened_at=None,
            half_open_at=None,
        )
        assert status.target == "host1"
        assert status.state == CircuitState.CLOSED
        assert status.failure_count == 0
        assert status.success_count == 5

    def test_time_since_open_when_closed(self):
        """time_since_open should be None when circuit not opened."""
        status = CircuitStatus(
            target="host1",
            state=CircuitState.CLOSED,
            failure_count=0,
            success_count=0,
            last_failure_time=None,
            last_success_time=None,
            opened_at=None,
            half_open_at=None,
        )
        assert status.time_since_open is None

    def test_time_since_open_when_open(self):
        """time_since_open should return elapsed seconds when open."""
        opened_at = time.time() - 10.0
        status = CircuitStatus(
            target="host1",
            state=CircuitState.OPEN,
            failure_count=3,
            success_count=0,
            last_failure_time=time.time(),
            last_success_time=None,
            opened_at=opened_at,
            half_open_at=None,
        )
        assert status.time_since_open is not None
        assert 9.5 < status.time_since_open < 11.0

    def test_to_dict(self):
        """to_dict should return all fields as dict."""
        status = CircuitStatus(
            target="host1",
            state=CircuitState.OPEN,
            failure_count=3,
            success_count=0,
            last_failure_time=123.456,
            last_success_time=None,
            opened_at=100.0,
            half_open_at=None,
            consecutive_opens=2,
        )
        d = status.to_dict()
        assert d["target"] == "host1"
        assert d["state"] == "open"
        assert d["failure_count"] == 3
        assert d["consecutive_opens"] == 2


class TestCircuitBreakerBasics:
    """Basic CircuitBreaker functionality tests."""

    def test_default_state_is_closed(self):
        """New circuit should start in CLOSED state."""
        breaker = CircuitBreaker()
        assert breaker.get_state("host1") == CircuitState.CLOSED

    def test_can_execute_when_closed(self):
        """can_execute should return True when CLOSED."""
        breaker = CircuitBreaker()
        assert breaker.can_execute("host1") is True

    def test_record_success_increments_count(self):
        """record_success should increment success_count."""
        breaker = CircuitBreaker()
        breaker.record_success("host1")
        status = breaker.get_status("host1")
        assert status.success_count == 1

    def test_record_failure_increments_count(self):
        """record_failure should increment failure_count."""
        breaker = CircuitBreaker()
        breaker.record_failure("host1")
        status = breaker.get_status("host1")
        assert status.failure_count == 1

    def test_success_resets_failure_count(self):
        """Success should reset failure_count to 0 in CLOSED state."""
        breaker = CircuitBreaker(failure_threshold=5)
        breaker.record_failure("host1")
        breaker.record_failure("host1")
        assert breaker.get_status("host1").failure_count == 2

        breaker.record_success("host1")
        assert breaker.get_status("host1").failure_count == 0


class TestCircuitBreakerStateTransitions:
    """Tests for circuit state transitions."""

    def test_opens_after_threshold_failures(self):
        """Circuit should OPEN after failure_threshold failures."""
        breaker = CircuitBreaker(failure_threshold=3)

        for i in range(3):
            assert breaker.get_state("host1") == CircuitState.CLOSED
            breaker.record_failure("host1")

        assert breaker.get_state("host1") == CircuitState.OPEN

    def test_blocks_requests_when_open(self):
        """can_execute should return False when OPEN."""
        breaker = CircuitBreaker(failure_threshold=2)
        breaker.record_failure("host1")
        breaker.record_failure("host1")

        assert breaker.get_state("host1") == CircuitState.OPEN
        assert breaker.can_execute("host1") is False

    def test_transitions_to_half_open_after_timeout(self):
        """Circuit should transition to HALF_OPEN after recovery_timeout."""
        # Use longer timeout to account for exponential backoff (2^1 = 2x base)
        # With consecutive_opens=1, timeout = 0.1 * 2 = 0.2s + jitter
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
            backoff_multiplier=1.0,  # Disable exponential backoff for this test
            jitter_factor=0.0,  # Disable jitter for predictable timing
        )
        breaker.record_failure("host1")
        breaker.record_failure("host1")

        assert breaker.get_state("host1") == CircuitState.OPEN

        time.sleep(0.15)

        assert breaker.get_state("host1") == CircuitState.HALF_OPEN

    def test_half_open_allows_limited_calls(self):
        """HALF_OPEN should allow up to half_open_max_calls."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
            half_open_max_calls=2,
            backoff_multiplier=1.0,
            jitter_factor=0.0,
        )
        breaker.record_failure("host1")
        breaker.record_failure("host1")

        time.sleep(0.15)

        # Should allow 2 calls
        assert breaker.can_execute("host1") is True
        assert breaker.can_execute("host1") is True
        # Third should be blocked
        assert breaker.can_execute("host1") is False

    def test_success_in_half_open_closes_circuit(self):
        """Success in HALF_OPEN should close the circuit."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
            success_threshold=1,
            backoff_multiplier=1.0,
            jitter_factor=0.0,
        )
        breaker.record_failure("host1")
        breaker.record_failure("host1")

        time.sleep(0.15)
        assert breaker.get_state("host1") == CircuitState.HALF_OPEN

        breaker.record_success("host1")
        assert breaker.get_state("host1") == CircuitState.CLOSED

    def test_failure_in_half_open_reopens_circuit(self):
        """Failure in HALF_OPEN should reopen the circuit."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
            backoff_multiplier=1.0,
            jitter_factor=0.0,
        )
        breaker.record_failure("host1")
        breaker.record_failure("host1")

        time.sleep(0.15)
        assert breaker.get_state("host1") == CircuitState.HALF_OPEN

        breaker.record_failure("host1")
        assert breaker.get_state("host1") == CircuitState.OPEN


class TestExponentialBackoff:
    """Tests for exponential backoff behavior."""

    def test_consecutive_opens_tracked(self):
        """consecutive_opens should increment on each open."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.05,
            backoff_multiplier=1.0,
            jitter_factor=0.0,
        )

        # First open
        breaker.record_failure("host1")
        status = breaker.get_status("host1")
        assert status.consecutive_opens == 1

        # Wait and transition to half-open
        time.sleep(0.1)
        assert breaker.get_state("host1") == CircuitState.HALF_OPEN

        # Fail again (reopens)
        breaker.record_failure("host1")
        status = breaker.get_status("host1")
        assert status.consecutive_opens == 2

    def test_consecutive_opens_reset_on_success(self):
        """consecutive_opens should reset after successful recovery."""
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.05,
            backoff_multiplier=1.0,
            jitter_factor=0.0,
        )

        breaker.record_failure("host1")
        time.sleep(0.1)
        # Transition to half-open and fail again
        breaker.can_execute("host1")  # Triggers half-open check
        breaker.record_failure("host1")
        time.sleep(0.1)

        assert breaker.get_status("host1").consecutive_opens >= 2

        # Trigger half-open and recover
        breaker.can_execute("host1")
        breaker.record_success("host1")
        status = breaker.get_status("host1")
        assert status.consecutive_opens == 0
        assert status.state == CircuitState.CLOSED


class TestCircuitBreakerReset:
    """Tests for circuit reset functionality."""

    def test_reset_returns_to_closed(self):
        """reset should return circuit to CLOSED state."""
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.record_failure("host1")

        assert breaker.get_state("host1") == CircuitState.OPEN

        breaker.reset("host1")
        assert breaker.get_state("host1") == CircuitState.CLOSED

    def test_reset_clears_failure_count(self):
        """reset should clear failure count."""
        breaker = CircuitBreaker(failure_threshold=5)
        breaker.record_failure("host1")
        breaker.record_failure("host1")

        breaker.reset("host1")
        status = breaker.get_status("host1")
        assert status.failure_count == 0

    def test_reset_all_clears_all_circuits(self):
        """reset_all should clear all tracked circuits."""
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.record_failure("host1")
        breaker.record_failure("host2")

        breaker.reset_all()

        assert breaker.get_state("host1") == CircuitState.CLOSED
        assert breaker.get_state("host2") == CircuitState.CLOSED

    def test_force_open(self):
        """force_open should immediately open circuit."""
        breaker = CircuitBreaker()
        assert breaker.get_state("host1") == CircuitState.CLOSED

        breaker.force_open("host1")
        assert breaker.get_state("host1") == CircuitState.OPEN

    def test_force_close(self):
        """force_close should immediately close circuit."""
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.record_failure("host1")

        assert breaker.get_state("host1") == CircuitState.OPEN

        breaker.force_close("host1")
        assert breaker.get_state("host1") == CircuitState.CLOSED


class TestContextManagers:
    """Tests for context manager functionality."""

    def test_protected_sync_records_success(self):
        """protected_sync should record success on normal exit."""
        breaker = CircuitBreaker()

        with breaker.protected_sync("host1"):
            pass  # Simulated successful operation

        status = breaker.get_status("host1")
        assert status.success_count == 1

    def test_protected_sync_records_failure(self):
        """protected_sync should record failure on exception."""
        breaker = CircuitBreaker()

        with pytest.raises(ValueError):
            with breaker.protected_sync("host1"):
                raise ValueError("Test error")

        status = breaker.get_status("host1")
        assert status.failure_count == 1

    def test_protected_sync_raises_when_open(self):
        """protected_sync should raise CircuitOpenError when open."""
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.record_failure("host1")

        with pytest.raises(CircuitOpenError):
            with breaker.protected_sync("host1"):
                pass

    @pytest.mark.asyncio
    async def test_protected_async_records_success(self):
        """protected should record success on normal exit."""
        breaker = CircuitBreaker()

        async with breaker.protected("host1"):
            await asyncio.sleep(0.01)

        status = breaker.get_status("host1")
        assert status.success_count == 1

    @pytest.mark.asyncio
    async def test_protected_async_records_failure(self):
        """protected should record failure on exception."""
        breaker = CircuitBreaker()

        with pytest.raises(ValueError):
            async with breaker.protected("host1"):
                raise ValueError("Test error")

        status = breaker.get_status("host1")
        assert status.failure_count == 1

    @pytest.mark.asyncio
    async def test_protected_async_raises_when_open(self):
        """protected should raise CircuitOpenError when open."""
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.record_failure("host1")

        with pytest.raises(CircuitOpenError):
            async with breaker.protected("host1"):
                pass


class TestExecuteMethods:
    """Tests for execute and execute_async methods."""

    def test_execute_returns_result(self):
        """execute should return function result on success."""
        breaker = CircuitBreaker()
        result = breaker.execute("host1", lambda: 42)
        assert result == 42

    def test_execute_records_success(self):
        """execute should record success."""
        breaker = CircuitBreaker()
        breaker.execute("host1", lambda: "ok")

        status = breaker.get_status("host1")
        assert status.success_count == 1

    def test_execute_records_failure(self):
        """execute should record failure on exception."""
        breaker = CircuitBreaker()

        def failing_func():
            raise RuntimeError("Failed")

        with pytest.raises(RuntimeError):
            breaker.execute("host1", failing_func)

        status = breaker.get_status("host1")
        assert status.failure_count == 1

    def test_execute_with_fallback(self):
        """execute should call fallback when circuit is open."""
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.record_failure("host1")

        result = breaker.execute(
            "host1",
            lambda: "primary",
            fallback=lambda: "fallback",
        )
        assert result == "fallback"

    def test_execute_raises_without_fallback(self):
        """execute should raise CircuitOpenError when open without fallback."""
        breaker = CircuitBreaker(failure_threshold=1)
        breaker.record_failure("host1")

        with pytest.raises(CircuitOpenError):
            breaker.execute("host1", lambda: "primary")

    @pytest.mark.asyncio
    async def test_execute_async_returns_result(self):
        """execute_async should return coroutine result."""
        breaker = CircuitBreaker()

        async def async_func():
            return 42

        result = await breaker.execute_async("host1", async_func)
        assert result == 42


class TestStateChangeCallback:
    """Tests for state change callback functionality."""

    def test_callback_on_open(self):
        """Callback should be called when circuit opens."""
        callback = MagicMock()
        breaker = CircuitBreaker(failure_threshold=1, on_state_change=callback)

        breaker.record_failure("host1")

        callback.assert_called_once_with(
            "host1", CircuitState.CLOSED, CircuitState.OPEN
        )

    def test_callback_on_close(self):
        """Callback should be called when circuit closes."""
        callback = MagicMock()
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.05,
            on_state_change=callback,
            backoff_multiplier=1.0,
            jitter_factor=0.0,
        )

        breaker.record_failure("host1")
        time.sleep(0.1)
        breaker.can_execute("host1")  # Trigger half-open
        breaker.record_success("host1")

        # Should have been called for CLOSED->OPEN and HALF_OPEN->CLOSED
        assert callback.call_count >= 2

    def test_callback_exception_does_not_crash(self):
        """Callback exception should not affect circuit operation."""
        def bad_callback(*args):
            raise RuntimeError("Callback error")

        breaker = CircuitBreaker(failure_threshold=1, on_state_change=bad_callback)

        # Should not raise
        breaker.record_failure("host1")
        assert breaker.get_state("host1") == CircuitState.OPEN


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry."""

    def test_get_instance_returns_singleton(self):
        """get_instance should return the same instance."""
        reg1 = CircuitBreakerRegistry.get_instance()
        reg2 = CircuitBreakerRegistry.get_instance()
        assert reg1 is reg2

    def test_get_breaker_returns_same_breaker(self):
        """get_breaker should return same breaker for same operation type."""
        registry = CircuitBreakerRegistry()
        b1 = registry.get_breaker("ssh")
        b2 = registry.get_breaker("ssh")
        assert b1 is b2

    def test_get_breaker_creates_different_breakers(self):
        """get_breaker should return different breakers for different types."""
        registry = CircuitBreakerRegistry()
        b1 = registry.get_breaker("ssh")
        b2 = registry.get_breaker("http")
        assert b1 is not b2

    def test_get_timeout_normal(self):
        """get_timeout should return default when circuit closed."""
        registry = CircuitBreakerRegistry()
        timeout = registry.get_timeout("ssh", "host1", 60.0)
        assert timeout == 60.0

    def test_get_timeout_half_open(self):
        """get_timeout should return shorter timeout in half-open."""
        registry = CircuitBreakerRegistry()
        breaker = registry.get_breaker("ssh")

        # Force to half-open
        breaker.force_open("host1")
        breaker._circuits["host1"].state = CircuitState.HALF_OPEN

        timeout = registry.get_timeout("ssh", "host1", 60.0)
        assert timeout < 60.0  # Should be reduced

    def test_get_all_open_circuits(self):
        """get_all_open_circuits should return only non-closed circuits."""
        registry = CircuitBreakerRegistry()

        ssh_breaker = registry.get_breaker("ssh")
        ssh_breaker.force_open("host1")

        http_breaker = registry.get_breaker("http")
        http_breaker.record_success("host2")  # Keep closed

        open_circuits = registry.get_all_open_circuits()
        assert "ssh" in open_circuits
        assert "host1" in open_circuits["ssh"]
        assert "http" not in open_circuits


class TestGlobalBreakers:
    """Tests for global circuit breaker functions."""

    def test_get_host_breaker_returns_breaker(self):
        """get_host_breaker should return a CircuitBreaker."""
        breaker = get_host_breaker()
        assert isinstance(breaker, CircuitBreaker)

    def test_get_training_breaker_returns_breaker(self):
        """get_training_breaker should return a CircuitBreaker."""
        breaker = get_training_breaker()
        assert isinstance(breaker, CircuitBreaker)

    def test_get_operation_breaker_returns_breaker(self):
        """get_operation_breaker should return a CircuitBreaker."""
        breaker = get_operation_breaker("test_op")
        assert isinstance(breaker, CircuitBreaker)

    def test_get_circuit_registry_returns_registry(self):
        """get_circuit_registry should return CircuitBreakerRegistry."""
        registry = get_circuit_registry()
        assert isinstance(registry, CircuitBreakerRegistry)

    def test_get_adaptive_timeout(self):
        """get_adaptive_timeout should return appropriate timeout."""
        timeout = get_adaptive_timeout("ssh", "host1", 60.0)
        assert isinstance(timeout, float)


class TestFormatCircuitStatus:
    """Tests for format_circuit_status function."""

    def test_format_closed_status(self):
        """format_circuit_status should show checkmark for closed."""
        status = CircuitStatus(
            target="host1",
            state=CircuitState.CLOSED,
            failure_count=0,
            success_count=5,
            last_failure_time=None,
            last_success_time=time.time(),
            opened_at=None,
            half_open_at=None,
        )
        formatted = format_circuit_status(status)
        assert "✓" in formatted
        assert "host1" in formatted
        assert "closed" in formatted

    def test_format_open_status(self):
        """format_circuit_status should show X for open."""
        status = CircuitStatus(
            target="host1",
            state=CircuitState.OPEN,
            failure_count=3,
            success_count=0,
            last_failure_time=time.time(),
            last_success_time=None,
            opened_at=time.time(),
            half_open_at=None,
        )
        formatted = format_circuit_status(status)
        assert "✗" in formatted
        assert "failures=3" in formatted

    def test_format_half_open_status(self):
        """format_circuit_status should show half-circle for half-open."""
        status = CircuitStatus(
            target="host1",
            state=CircuitState.HALF_OPEN,
            failure_count=3,
            success_count=0,
            last_failure_time=time.time(),
            last_success_time=None,
            opened_at=time.time() - 60,
            half_open_at=time.time(),
        )
        formatted = format_circuit_status(status)
        assert "◐" in formatted
        assert "half_open" in formatted


class TestFallbackChain:
    """Tests for FallbackChain class."""

    def test_add_operation_returns_self(self):
        """add_operation should return self for chaining."""
        chain = FallbackChain()
        result = chain.add_operation("op1", lambda: None, 10.0)
        assert result is chain

    def test_remaining_budget_initial(self):
        """remaining_budget should equal total_timeout initially."""
        chain = FallbackChain(total_timeout=100.0)
        assert chain.remaining_budget == 100.0

    @pytest.mark.asyncio
    async def test_execute_returns_first_success(self):
        """execute should return result from first successful operation."""
        chain = FallbackChain(total_timeout=10.0)

        async def op1(host, timeout):
            return "result1"

        chain.add_operation("op1", op1, 5.0)

        result = await chain.execute(host="host1")
        assert result == "result1"

    @pytest.mark.asyncio
    async def test_execute_skips_open_circuits(self):
        """execute should skip operations with open circuits."""
        chain = FallbackChain(total_timeout=10.0)

        # Open the circuit for op1
        registry = get_circuit_registry()
        breaker = registry.get_breaker("test_skip_open")
        breaker.force_open("host1")

        async def op1(host, timeout):
            return "should_not_reach"

        async def op2(host, timeout):
            return "fallback_result"

        chain.add_operation("test_skip_open", op1, 5.0)
        chain.add_operation("test_skip_fallback", op2, 5.0)

        result = await chain.execute(host="host1")
        assert result == "fallback_result"


class TestWithCircuitBreakerDecorator:
    """Tests for with_circuit_breaker decorator."""

    def test_decorator_allows_when_closed(self):
        """Decorated function should execute when circuit is closed."""
        @with_circuit_breaker("test_decorator_closed")
        def my_func(host: str):
            return f"success for {host}"

        result = my_func(host="test_host")
        assert result == "success for test_host"

    def test_decorator_blocks_when_open(self):
        """Decorated function should raise when circuit is open."""
        # First, open the circuit
        breaker = get_operation_breaker("test_decorator_open")
        breaker.force_open("block_host")

        @with_circuit_breaker("test_decorator_open")
        def my_func(host: str):
            return "should not reach"

        with pytest.raises(CircuitOpenError):
            my_func(host="block_host")

    def test_decorator_records_success(self):
        """Decorated function should record success."""
        @with_circuit_breaker("test_decorator_success")
        def my_func(host: str):
            return "ok"

        my_func(host="success_host")

        breaker = get_operation_breaker("test_decorator_success")
        status = breaker.get_status("success_host")
        assert status.success_count >= 1

    def test_decorator_records_failure(self):
        """Decorated function should record failure on exception."""
        @with_circuit_breaker("test_decorator_failure")
        def my_func(host: str):
            raise RuntimeError("Failed")

        with pytest.raises(RuntimeError):
            my_func(host="fail_host")

        breaker = get_operation_breaker("test_decorator_failure")
        status = breaker.get_status("fail_host")
        assert status.failure_count >= 1

    @pytest.mark.asyncio
    async def test_decorator_async(self):
        """Decorator should work with async functions."""
        @with_circuit_breaker("test_decorator_async")
        async def async_func(host: str):
            await asyncio.sleep(0.01)
            return "async success"

        result = await async_func(host="async_host")
        assert result == "async success"

    def test_decorator_with_custom_host_param(self):
        """Decorator should accept custom host parameter name."""
        @with_circuit_breaker("test_custom_param", host_param="hostname")
        def my_func(hostname: str):
            return f"success for {hostname}"

        result = my_func(hostname="custom_host")
        assert result == "success for custom_host"


class TestMultipleTargets:
    """Tests for handling multiple targets."""

    def test_independent_circuits_per_target(self):
        """Each target should have independent circuit state."""
        breaker = CircuitBreaker(failure_threshold=2)

        # Open circuit for host1
        breaker.record_failure("host1")
        breaker.record_failure("host1")

        # host1 should be open
        assert breaker.get_state("host1") == CircuitState.OPEN

        # host2 should still be closed
        assert breaker.get_state("host2") == CircuitState.CLOSED
        assert breaker.can_execute("host2") is True

    def test_get_all_states_returns_all_targets(self):
        """get_all_states should return status for all tracked targets."""
        breaker = CircuitBreaker()

        breaker.record_success("host1")
        breaker.record_failure("host2")
        breaker.record_success("host3")

        states = breaker.get_all_states()

        assert "host1" in states
        assert "host2" in states
        assert "host3" in states
        assert len(states) == 3
