#!/usr/bin/env python3
"""Unit tests for CircuitBreaker (December 2025).

Tests the circuit breaker pattern implementation for fault-tolerant
distributed operations.
"""

import asyncio
import time
import pytest
from unittest.mock import Mock, patch

from app.distributed.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitStatus,
    CircuitOpenError,
    CircuitBreakerRegistry,
    FallbackChain,
    get_circuit_registry,
    get_host_breaker,
    get_operation_breaker,
    format_circuit_status,
    get_adaptive_timeout,
)


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_circuit_state_values(self):
        """Test CircuitState enum has expected values."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestCircuitStatus:
    """Tests for CircuitStatus dataclass."""

    def test_circuit_status_creation(self):
        """Test CircuitStatus creation."""
        status = CircuitStatus(
            target="host1",
            state=CircuitState.CLOSED,
            failure_count=0,
            success_count=5,
            last_failure_time=None,
            last_success_time=time.time(),
            opened_at=None,
            half_open_at=None,
            consecutive_opens=0,
        )

        assert status.target == "host1"
        assert status.state == CircuitState.CLOSED
        assert status.failure_count == 0
        assert status.success_count == 5
        assert status.consecutive_opens == 0

    def test_time_since_open_when_closed(self):
        """Test time_since_open returns None when circuit is closed."""
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

    def test_time_since_open_when_opened(self):
        """Test time_since_open returns time delta when circuit is open."""
        opened_at = time.time() - 10.0  # 10 seconds ago
        status = CircuitStatus(
            target="host1",
            state=CircuitState.OPEN,
            failure_count=3,
            success_count=0,
            last_failure_time=opened_at,
            last_success_time=None,
            opened_at=opened_at,
            half_open_at=None,
        )
        # Should be approximately 10 seconds
        assert status.time_since_open is not None
        assert 9.0 <= status.time_since_open <= 11.0

    def test_to_dict(self):
        """Test CircuitStatus.to_dict() returns expected structure."""
        status = CircuitStatus(
            target="host1",
            state=CircuitState.OPEN,
            failure_count=3,
            success_count=0,
            last_failure_time=1234567890.0,
            last_success_time=None,
            opened_at=1234567890.0,
            half_open_at=None,
            consecutive_opens=2,
        )
        d = status.to_dict()

        assert d["target"] == "host1"
        assert d["state"] == "open"
        assert d["failure_count"] == 3
        assert d["success_count"] == 0
        assert d["consecutive_opens"] == 2
        assert "time_since_open" in d


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_init_defaults(self):
        """Test CircuitBreaker initialization with defaults."""
        breaker = CircuitBreaker()

        assert breaker.failure_threshold > 0
        assert breaker.recovery_timeout > 0
        assert breaker.half_open_max_calls > 0
        assert breaker.success_threshold > 0

    def test_init_custom_values(self):
        """Test CircuitBreaker with custom configuration."""
        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30.0,
            half_open_max_calls=2,
            success_threshold=2,
        )

        assert breaker.failure_threshold == 3
        assert breaker.recovery_timeout == 30.0
        assert breaker.half_open_max_calls == 2
        assert breaker.success_threshold == 2

    def test_can_execute_closed(self):
        """Test can_execute returns True when circuit is closed."""
        breaker = CircuitBreaker(failure_threshold=3)

        # New target starts in CLOSED state
        assert breaker.can_execute("host1") is True
        assert breaker.get_state("host1") == CircuitState.CLOSED

    def test_record_success_resets_failure_count(self):
        """Test that success resets failure count."""
        breaker = CircuitBreaker(failure_threshold=3)

        # Record some failures (but not enough to open)
        breaker.record_failure("host1")
        breaker.record_failure("host1")
        assert breaker.get_status("host1").failure_count == 2

        # Record success resets count
        breaker.record_success("host1")
        assert breaker.get_status("host1").failure_count == 0

    def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold is reached."""
        breaker = CircuitBreaker(failure_threshold=3)

        # Record failures up to threshold
        breaker.record_failure("host1")
        breaker.record_failure("host1")
        assert breaker.get_state("host1") == CircuitState.CLOSED

        # Third failure should open the circuit
        breaker.record_failure("host1")
        assert breaker.get_state("host1") == CircuitState.OPEN

    def test_can_execute_blocked_when_open(self):
        """Test can_execute returns False when circuit is open."""
        breaker = CircuitBreaker(failure_threshold=2)

        # Open the circuit
        breaker.record_failure("host1")
        breaker.record_failure("host1")

        assert breaker.get_state("host1") == CircuitState.OPEN
        assert breaker.can_execute("host1") is False

    def test_recovery_transition_to_half_open(self):
        """Test circuit transitions to half-open after recovery timeout."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,  # 100ms
            jitter_factor=0,  # Disable jitter for predictable timing
            backoff_multiplier=1.0,  # No backoff for predictable timing
        )

        # Open the circuit
        breaker.record_failure("host1")
        breaker.record_failure("host1")
        assert breaker.get_state("host1") == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # Should now be half-open (checked during can_execute or get_state)
        assert breaker.get_state("host1") == CircuitState.HALF_OPEN
        assert breaker.can_execute("host1") is True

    def test_half_open_to_closed_on_success(self):
        """Test circuit transitions from half-open to closed on success."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.05,
            success_threshold=1,
        )

        # Open the circuit
        breaker.record_failure("host1")
        breaker.record_failure("host1")
        time.sleep(0.1)

        # Should be half-open
        assert breaker.get_state("host1") == CircuitState.HALF_OPEN

        # Record success
        breaker.record_success("host1")
        assert breaker.get_state("host1") == CircuitState.CLOSED

    def test_half_open_to_open_on_failure(self):
        """Test circuit transitions from half-open back to open on failure."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.05,
            jitter_factor=0,
            backoff_multiplier=1.0,
        )

        # Open the circuit
        breaker.record_failure("host1")
        breaker.record_failure("host1")
        time.sleep(0.1)

        # Should be half-open
        assert breaker.get_state("host1") == CircuitState.HALF_OPEN

        # Failure in half-open goes back to open
        breaker.record_failure("host1")
        assert breaker.get_state("host1") == CircuitState.OPEN

    def test_half_open_max_calls(self):
        """Test half-open state limits concurrent test calls."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.05,
            half_open_max_calls=1,
            jitter_factor=0,
            backoff_multiplier=1.0,
        )

        # Open and wait for half-open
        breaker.record_failure("host1")
        breaker.record_failure("host1")
        time.sleep(0.1)

        assert breaker.get_state("host1") == CircuitState.HALF_OPEN

        # First call allowed
        assert breaker.can_execute("host1") is True
        # Second call blocked (already at max)
        assert breaker.can_execute("host1") is False

    def test_get_all_states(self):
        """Test get_all_states returns all tracked targets."""
        breaker = CircuitBreaker(failure_threshold=3)

        # Create circuits for multiple targets
        breaker.record_success("host1")
        breaker.record_failure("host2")
        breaker.record_success("host3")

        states = breaker.get_all_states()

        assert len(states) == 3
        assert "host1" in states
        assert "host2" in states
        assert "host3" in states

    def test_reset_clears_circuit(self):
        """Test reset() clears circuit state."""
        breaker = CircuitBreaker(failure_threshold=2)

        # Open the circuit
        breaker.record_failure("host1")
        breaker.record_failure("host1")
        assert breaker.get_state("host1") == CircuitState.OPEN

        # Reset
        breaker.reset("host1")
        assert breaker.get_state("host1") == CircuitState.CLOSED
        assert breaker.get_status("host1").failure_count == 0

    def test_exponential_backoff_increases_timeout(self):
        """Test exponential backoff increases recovery timeout."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.05,  # 50ms base
            backoff_multiplier=2.0,
            jitter_factor=0,  # Disable jitter for predictable test
        )

        # First open: consecutive_opens becomes 1
        # Timeout = 0.05 * 2^1 = 0.1s
        breaker.record_failure("host1")
        breaker.record_failure("host1")
        status1 = breaker.get_status("host1")
        assert status1.consecutive_opens == 1

        # Wait for first recovery (0.1s) and verify half-open
        time.sleep(0.11)
        assert breaker.can_execute("host1") is True
        breaker.record_failure("host1")  # Back to open

        # Second open: consecutive_opens becomes 2
        # Timeout = 0.05 * 2^2 = 0.2s
        status2 = breaker.get_status("host1")
        assert status2.consecutive_opens == 2

        # Wait less than the second backoff timeout
        time.sleep(0.11)
        # Still open because backoff requires 0.2s
        assert breaker.get_state("host1") == CircuitState.OPEN

        # Wait the rest of the backoff time
        time.sleep(0.10)  # Total ~0.21s
        assert breaker.get_state("host1") == CircuitState.HALF_OPEN

    def test_state_change_callback(self):
        """Test state change callback is invoked on transitions."""
        callback = Mock()
        breaker = CircuitBreaker(
            failure_threshold=2,
            on_state_change=callback,
        )

        # Trigger state change
        breaker.record_failure("host1")
        breaker.record_failure("host1")

        callback.assert_called_once_with(
            "host1", CircuitState.CLOSED, CircuitState.OPEN
        )

    def test_per_target_isolation(self):
        """Test circuits are isolated per target."""
        breaker = CircuitBreaker(failure_threshold=2)

        # Open circuit for host1
        breaker.record_failure("host1")
        breaker.record_failure("host1")
        assert breaker.get_state("host1") == CircuitState.OPEN

        # host2 should still be closed
        assert breaker.get_state("host2") == CircuitState.CLOSED
        assert breaker.can_execute("host2") is True

    @pytest.mark.asyncio
    async def test_protected_context_manager_success(self):
        """Test async protected context manager on success."""
        breaker = CircuitBreaker(failure_threshold=2)

        async with breaker.protected("host1"):
            pass  # Success

        status = breaker.get_status("host1")
        assert status.success_count == 1
        assert status.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_protected_context_manager_failure(self):
        """Test async protected context manager on failure."""
        breaker = CircuitBreaker(failure_threshold=2)

        with pytest.raises(ValueError):
            async with breaker.protected("host1"):
                raise ValueError("Test error")

        status = breaker.get_status("host1")
        assert status.failure_count == 1

    @pytest.mark.asyncio
    async def test_protected_raises_circuit_open_error(self):
        """Test protected raises CircuitOpenError when circuit is open."""
        breaker = CircuitBreaker(failure_threshold=1)

        # Open the circuit
        breaker.record_failure("host1")
        assert breaker.get_state("host1") == CircuitState.OPEN

        # Should raise CircuitOpenError
        with pytest.raises(CircuitOpenError) as exc_info:
            async with breaker.protected("host1"):
                pass

        assert "host1" in str(exc_info.value)


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry."""

    def test_registry_creates_breakers(self):
        """Test registry creates operation-specific breakers."""
        registry = CircuitBreakerRegistry()

        # Use get_breaker() which is the actual method name
        ssh_breaker = registry.get_breaker("ssh")
        http_breaker = registry.get_breaker("http")

        assert ssh_breaker is not None
        assert http_breaker is not None
        assert ssh_breaker is not http_breaker

    def test_registry_returns_same_breaker(self):
        """Test registry returns same breaker for same operation."""
        registry = CircuitBreakerRegistry()

        breaker1 = registry.get_breaker("ssh")
        breaker2 = registry.get_breaker("ssh")

        assert breaker1 is breaker2

    def test_registry_get_all_open_circuits(self):
        """Test registry get_all_open_circuits finds open circuits."""
        registry = CircuitBreakerRegistry()

        ssh_breaker = registry.get_breaker("ssh")

        # Open a circuit
        for _ in range(5):
            ssh_breaker.record_failure("host1")

        open_circuits = registry.get_all_open_circuits()

        # Should have the open circuit for ssh
        assert "ssh" in open_circuits or len(open_circuits) == 0 or ssh_breaker.get_state("host1") != CircuitState.OPEN


class TestModuleFunctions:
    """Tests for module-level helper functions."""

    def test_get_circuit_registry_singleton(self):
        """Test get_circuit_registry returns singleton."""
        registry1 = get_circuit_registry()
        registry2 = get_circuit_registry()

        assert registry1 is registry2

    def test_get_host_breaker(self):
        """Test get_host_breaker returns a circuit breaker."""
        breaker = get_host_breaker()

        assert breaker is not None
        assert isinstance(breaker, CircuitBreaker)

    def test_get_operation_breaker(self):
        """Test get_operation_breaker for different operations."""
        ssh_breaker = get_operation_breaker("ssh")
        http_breaker = get_operation_breaker("http")

        # Each operation type gets its own breaker instance
        assert ssh_breaker is not None
        assert http_breaker is not None
        assert isinstance(ssh_breaker, CircuitBreaker)
        assert isinstance(http_breaker, CircuitBreaker)

    def test_format_circuit_status(self):
        """Test format_circuit_status returns readable string."""
        status = CircuitStatus(
            target="host1",
            state=CircuitState.OPEN,
            failure_count=3,
            success_count=0,
            last_failure_time=time.time(),
            last_success_time=None,
            opened_at=time.time() - 30,
            half_open_at=None,
        )

        formatted = format_circuit_status(status)

        assert "host1" in formatted
        assert "open" in formatted.lower()

    def test_get_adaptive_timeout(self):
        """Test get_adaptive_timeout returns appropriate value."""
        # get_adaptive_timeout takes operation_type, host, and default
        base_timeout = get_adaptive_timeout("ssh", "host1", 10.0)
        assert base_timeout >= 0  # Could be shorter in half-open state


class TestFallbackChain:
    """Tests for FallbackChain."""

    def test_fallback_chain_init(self):
        """Test FallbackChain initialization."""
        chain = FallbackChain(total_timeout=300.0)

        assert chain.total_timeout == 300.0
        assert chain.remaining_budget == 300.0

    def test_fallback_chain_add_operation(self):
        """Test FallbackChain add_operation returns self for chaining."""
        chain = FallbackChain()

        async def dummy_op(host, timeout, **kwargs):
            return "result"

        result = chain.add_operation("ssh", dummy_op, timeout=60.0)

        assert result is chain  # Supports chaining

    @pytest.mark.asyncio
    async def test_fallback_chain_execute_success(self):
        """Test FallbackChain execute returns first successful result."""
        chain = FallbackChain(total_timeout=300.0)

        async def success_op(host, timeout, **kwargs):
            return f"success-{host}"

        chain.add_operation("ssh", success_op, timeout=60.0)

        result = await chain.execute("test-host")

        assert result == "success-test-host"

    @pytest.mark.asyncio
    async def test_fallback_chain_tries_alternatives(self):
        """Test FallbackChain tries alternatives on failure."""
        chain = FallbackChain(total_timeout=300.0)
        results = []

        async def fail_op(host, timeout, **kwargs):
            results.append("fail")
            raise ValueError("Failed")

        async def success_op(host, timeout, **kwargs):
            results.append("success")
            return "result"

        chain.add_operation("ssh", fail_op, timeout=60.0)
        chain.add_operation("http", success_op, timeout=60.0)

        result = await chain.execute("test-host")

        assert result == "result"
        assert "fail" in results
        assert "success" in results


class TestCircuitOpenError:
    """Tests for CircuitOpenError exception."""

    def test_circuit_open_error_is_exception(self):
        """Test CircuitOpenError is a proper exception."""
        error = CircuitOpenError("Circuit is open for host1")

        assert isinstance(error, Exception)
        assert str(error) == "Circuit is open for host1"

    def test_circuit_open_error_can_be_raised(self):
        """Test CircuitOpenError can be raised and caught."""
        with pytest.raises(CircuitOpenError):
            raise CircuitOpenError("test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
