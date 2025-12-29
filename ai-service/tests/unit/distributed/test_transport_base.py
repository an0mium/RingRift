#!/usr/bin/env python3
"""Unit tests for transport_base.py - Transport Base Class for Unified Transport Operations.

Tests cover:
1. TransportState enum values
2. TransportResult dataclass validation
3. TransportError and subclasses (RetryableTransportError, PermanentTransportError)
4. CircuitBreakerConfig factory methods and defaults
5. TimeoutConfig factory methods and defaults
6. TargetStatus dataclass
7. TransportBase abstract class:
   - Circuit breaker state machine (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
   - Success/failure recording
   - Timeout execution with retry
   - State persistence (load/save JSON)
   - Health check reporting

December 2025: Comprehensive test coverage for transport infrastructure.
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.transport_base import (
    CircuitBreakerConfig,
    PermanentTransportError,
    RetryableTransportError,
    TargetStatus,
    TimeoutConfig,
    TransportBase,
    TransportError,
    TransportResult,
    TransportState,
)


# =============================================================================
# TransportState Tests
# =============================================================================


class TestTransportState:
    """Tests for TransportState enum."""

    def test_closed_value(self):
        """CLOSED state has correct string value."""
        assert TransportState.CLOSED.value == "closed"

    def test_open_value(self):
        """OPEN state has correct string value."""
        assert TransportState.OPEN.value == "open"

    def test_half_open_value(self):
        """HALF_OPEN state has correct string value."""
        assert TransportState.HALF_OPEN.value == "half_open"

    def test_all_states_defined(self):
        """All three states are defined."""
        assert len(TransportState) == 3

    def test_state_membership(self):
        """States are members of the enum."""
        assert TransportState.CLOSED in TransportState
        assert TransportState.OPEN in TransportState
        assert TransportState.HALF_OPEN in TransportState


# =============================================================================
# TransportResult Tests
# =============================================================================


class TestTransportResult:
    """Tests for TransportResult dataclass."""

    def test_success_result(self):
        """Successful result has correct properties."""
        result = TransportResult(
            success=True,
            transport_used="rsync",
            latency_ms=150.5,
            bytes_transferred=1024,
        )
        assert result.success is True
        assert result.transport_used == "rsync"
        assert result.latency_ms == 150.5
        assert result.bytes_transferred == 1024
        assert result.error is None

    def test_failure_result_with_error(self):
        """Failed result with error message."""
        result = TransportResult(
            success=False,
            error="Connection refused",
            transport_used="ssh",
        )
        assert result.success is False
        assert result.error == "Connection refused"

    def test_failure_result_auto_error(self):
        """Failed result without error gets default message."""
        result = TransportResult(success=False)
        assert result.success is False
        assert result.error == "Unknown error"

    def test_default_values(self):
        """Default values are applied correctly."""
        result = TransportResult(success=True)
        assert result.transport_used == ""
        assert result.error is None
        assert result.latency_ms == 0.0
        assert result.bytes_transferred == 0
        assert result.data is None
        assert result.metadata == {}

    def test_data_field(self):
        """Data field holds arbitrary data."""
        data = {"key": "value", "count": 42}
        result = TransportResult(success=True, data=data)
        assert result.data == data

    def test_metadata_field(self):
        """Metadata field holds extra info."""
        meta = {"checksum": "abc123", "compression": "gzip"}
        result = TransportResult(success=True, metadata=meta)
        assert result.metadata == meta

    def test_to_dict(self):
        """to_dict serializes all fields."""
        result = TransportResult(
            success=True,
            transport_used="http",
            error=None,
            latency_ms=50.0,
            bytes_transferred=2048,
            data={"response": "ok"},
            metadata={"version": "1.0"},
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["transport_used"] == "http"
        assert d["error"] is None
        assert d["latency_ms"] == 50.0
        assert d["bytes_transferred"] == 2048
        assert d["data"] == {"response": "ok"}
        assert d["metadata"] == {"version": "1.0"}


# =============================================================================
# TransportError Tests
# =============================================================================


class TestTransportError:
    """Tests for TransportError exception class."""

    def test_basic_error(self):
        """Basic error with message."""
        err = TransportError(message="Connection failed")
        assert err.message == "Connection failed"
        assert str(err) == "Connection failed"

    def test_error_with_transport(self):
        """Error with transport name."""
        err = TransportError(message="Timeout", transport="rsync")
        assert err.transport == "rsync"
        assert "transport=rsync" in str(err)

    def test_error_with_target(self):
        """Error with target identifier."""
        err = TransportError(message="Host unreachable", target="node-1")
        assert err.target == "node-1"
        assert "target=node-1" in str(err)

    def test_error_with_cause(self):
        """Error with underlying cause."""
        cause = ValueError("Bad value")
        err = TransportError(message="Operation failed", cause=cause)
        assert err.cause is cause

    def test_full_error_string(self):
        """Error string includes all parts."""
        err = TransportError(
            message="Transfer failed",
            transport="ssh",
            target="host-a",
        )
        s = str(err)
        assert "Transfer failed" in s
        assert "transport=ssh" in s
        assert "target=host-a" in s


class TestRetryableTransportError:
    """Tests for RetryableTransportError subclass."""

    def test_is_transport_error(self):
        """RetryableTransportError is a TransportError."""
        err = RetryableTransportError(message="Temporary failure")
        assert isinstance(err, TransportError)

    def test_inheritance(self):
        """Can be caught as TransportError."""
        try:
            raise RetryableTransportError(message="Network timeout")
        except TransportError as e:
            assert "Network timeout" in str(e)


class TestPermanentTransportError:
    """Tests for PermanentTransportError subclass."""

    def test_is_transport_error(self):
        """PermanentTransportError is a TransportError."""
        err = PermanentTransportError(message="Auth failed")
        assert isinstance(err, TransportError)

    def test_distinct_from_retryable(self):
        """PermanentTransportError is distinct from RetryableTransportError."""
        perm = PermanentTransportError(message="Config error")
        retry = RetryableTransportError(message="Timeout")
        assert type(perm) is not type(retry)


# =============================================================================
# CircuitBreakerConfig Tests
# =============================================================================


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig dataclass."""

    def test_default_values(self):
        """Default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 3
        assert config.recovery_timeout == 300.0
        assert config.half_open_max_calls == 1

    def test_custom_values(self):
        """Custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=120.0,
            half_open_max_calls=2,
        )
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 120.0
        assert config.half_open_max_calls == 2

    def test_aggressive_factory(self):
        """Aggressive factory method returns quick-failing config."""
        config = CircuitBreakerConfig.aggressive()
        assert config.failure_threshold == 2
        assert config.recovery_timeout == 60.0

    def test_patient_factory(self):
        """Patient factory method returns tolerant config."""
        config = CircuitBreakerConfig.patient()
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 600.0


# =============================================================================
# TimeoutConfig Tests
# =============================================================================


class TestTimeoutConfig:
    """Tests for TimeoutConfig dataclass."""

    def test_default_values(self):
        """Default timeout values."""
        config = TimeoutConfig()
        assert config.connect_timeout == 30
        assert config.operation_timeout == 180
        assert config.http_timeout == 30

    def test_custom_values(self):
        """Custom timeout values."""
        config = TimeoutConfig(
            connect_timeout=15,
            operation_timeout=60,
            http_timeout=20,
        )
        assert config.connect_timeout == 15
        assert config.operation_timeout == 60
        assert config.http_timeout == 20

    def test_fast_factory(self):
        """Fast factory method returns quick timeouts."""
        config = TimeoutConfig.fast()
        assert config.connect_timeout == 10
        assert config.operation_timeout == 60
        assert config.http_timeout == 15

    def test_slow_factory(self):
        """Slow factory method returns extended timeouts."""
        config = TimeoutConfig.slow()
        assert config.connect_timeout == 60
        assert config.operation_timeout == 600
        assert config.http_timeout == 120


# =============================================================================
# TargetStatus Tests
# =============================================================================


class TestTargetStatus:
    """Tests for TargetStatus dataclass."""

    def test_default_values(self):
        """Default status values."""
        status = TargetStatus()
        assert status.state == TransportState.CLOSED
        assert status.failure_count == 0
        assert status.success_count == 0
        assert status.last_failure_time == 0.0
        assert status.last_success_time == 0.0
        assert status.last_error is None

    def test_custom_values(self):
        """Custom status values."""
        now = time.time()
        status = TargetStatus(
            state=TransportState.OPEN,
            failure_count=3,
            success_count=10,
            last_failure_time=now,
            last_error="Connection refused",
        )
        assert status.state == TransportState.OPEN
        assert status.failure_count == 3
        assert status.success_count == 10
        assert status.last_failure_time == now
        assert status.last_error == "Connection refused"


# =============================================================================
# TransportBase Implementation for Testing
# =============================================================================


class MockTransport(TransportBase):
    """Mock transport for testing TransportBase functionality."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transfer_calls = []
        self.should_fail = False
        self.fail_message = "Mock failure"

    async def transfer(
        self,
        source,
        destination,
        target,
        **kwargs,
    ) -> TransportResult:
        """Mock transfer implementation."""
        self.transfer_calls.append({
            "source": source,
            "destination": destination,
            "target": target,
            "kwargs": kwargs,
        })

        if self.should_fail:
            return TransportResult(
                success=False,
                error=self.fail_message,
                transport_used=self.name,
            )

        return TransportResult(
            success=True,
            transport_used=self.name,
            latency_ms=10.0,
            bytes_transferred=100,
        )


# =============================================================================
# TransportBase Tests - Initialization
# =============================================================================


class TestTransportBaseInit:
    """Tests for TransportBase initialization."""

    def test_default_name(self):
        """Transport uses class name as default name."""
        transport = MockTransport()
        assert transport.name == "MockTransport"

    def test_custom_name(self):
        """Transport can have custom name."""
        transport = MockTransport(name="CustomTransport")
        assert transport.name == "CustomTransport"

    def test_default_timeouts(self):
        """Default timeouts are applied."""
        transport = MockTransport()
        assert transport.connect_timeout == 30
        assert transport.operation_timeout == 180
        assert transport.http_timeout == 30

    def test_custom_timeout_config(self):
        """Custom timeout config is applied."""
        config = TimeoutConfig(connect_timeout=10, operation_timeout=60)
        transport = MockTransport(timeout_config=config)
        assert transport.connect_timeout == 10
        assert transport.operation_timeout == 60

    def test_default_circuit_breaker_config(self):
        """Default circuit breaker config is applied."""
        transport = MockTransport()
        assert transport._failure_threshold == 3
        assert transport._recovery_timeout == 300.0
        assert transport._half_open_max_calls == 1

    def test_custom_circuit_breaker_config(self):
        """Custom circuit breaker config is applied."""
        config = CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60.0)
        transport = MockTransport(circuit_breaker_config=config)
        assert transport._failure_threshold == 5
        assert transport._recovery_timeout == 60.0


# =============================================================================
# TransportBase Tests - Circuit Breaker State Machine
# =============================================================================


class TestCircuitBreakerStateMachine:
    """Tests for circuit breaker state machine behavior."""

    def test_initial_state_closed(self):
        """New targets start with CLOSED circuit."""
        transport = MockTransport()
        assert transport.get_circuit_state("new-target") == TransportState.CLOSED

    def test_can_attempt_when_closed(self):
        """can_attempt returns True when CLOSED."""
        transport = MockTransport()
        assert transport.can_attempt("target-a") is True

    def test_opens_after_threshold_failures(self):
        """Circuit opens after failure_threshold failures."""
        transport = MockTransport(
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=2)
        )
        transport.record_failure("target-a")
        assert transport.get_circuit_state("target-a") == TransportState.CLOSED

        transport.record_failure("target-a")
        assert transport.get_circuit_state("target-a") == TransportState.OPEN

    def test_blocks_when_open(self):
        """can_attempt returns False when OPEN (before recovery timeout)."""
        transport = MockTransport(
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=1, recovery_timeout=300.0
            )
        )
        transport.record_failure("target-a")
        assert transport.get_circuit_state("target-a") == TransportState.OPEN
        assert transport.can_attempt("target-a") is False

    def test_transitions_to_half_open_after_timeout(self):
        """Circuit transitions to HALF_OPEN after recovery timeout."""
        transport = MockTransport(
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=1, recovery_timeout=0.01
            )
        )
        transport.record_failure("target-a")
        assert transport.get_circuit_state("target-a") == TransportState.OPEN

        # Wait for recovery timeout
        time.sleep(0.02)

        # Calling can_attempt triggers transition check
        assert transport.can_attempt("target-a") is True
        assert transport.get_circuit_state("target-a") == TransportState.HALF_OPEN

    def test_success_in_half_open_closes_circuit(self):
        """Success in HALF_OPEN state closes the circuit."""
        transport = MockTransport(
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=1, recovery_timeout=0.01
            )
        )
        transport.record_failure("target-a")
        time.sleep(0.02)
        transport.can_attempt("target-a")  # Transition to HALF_OPEN

        transport.record_success("target-a")
        assert transport.get_circuit_state("target-a") == TransportState.CLOSED

    def test_failure_in_half_open_reopens_circuit(self):
        """Failure in HALF_OPEN state reopens the circuit."""
        transport = MockTransport(
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=1, recovery_timeout=0.01
            )
        )
        transport.record_failure("target-a")
        time.sleep(0.02)
        transport.can_attempt("target-a")  # Transition to HALF_OPEN

        transport.record_failure("target-a")
        assert transport.get_circuit_state("target-a") == TransportState.OPEN


# =============================================================================
# TransportBase Tests - Success/Failure Recording
# =============================================================================


class TestRecordSuccessFailure:
    """Tests for success and failure recording."""

    def test_record_success_increments_count(self):
        """record_success increments success count."""
        transport = MockTransport()
        transport.record_success("target-a")
        transport.record_success("target-a")

        status = transport._target_status["target-a"]
        assert status.success_count == 2

    def test_record_success_resets_failure_count(self):
        """record_success resets failure count."""
        transport = MockTransport(
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=5)
        )
        transport.record_failure("target-a")
        transport.record_failure("target-a")
        assert transport._target_status["target-a"].failure_count == 2

        transport.record_success("target-a")
        assert transport._target_status["target-a"].failure_count == 0

    def test_record_success_updates_timestamp(self):
        """record_success updates last_success_time."""
        transport = MockTransport()
        before = time.time()
        transport.record_success("target-a")
        after = time.time()

        status = transport._target_status["target-a"]
        assert before <= status.last_success_time <= after

    def test_record_failure_increments_count(self):
        """record_failure increments failure count."""
        transport = MockTransport(
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=10)
        )
        transport.record_failure("target-a")
        transport.record_failure("target-a")
        transport.record_failure("target-a")

        status = transport._target_status["target-a"]
        assert status.failure_count == 3

    def test_record_failure_with_error_message(self):
        """record_failure stores error message."""
        transport = MockTransport()
        transport.record_failure("target-a", "Connection refused")

        status = transport._target_status["target-a"]
        assert status.last_error == "Connection refused"

    def test_record_failure_with_exception(self):
        """record_failure stores exception as string."""
        transport = MockTransport()
        transport.record_failure("target-a", ValueError("Bad data"))

        status = transport._target_status["target-a"]
        assert "Bad data" in status.last_error

    def test_record_failure_updates_timestamp(self):
        """record_failure updates last_failure_time."""
        transport = MockTransport()
        before = time.time()
        transport.record_failure("target-a")
        after = time.time()

        status = transport._target_status["target-a"]
        assert before <= status.last_failure_time <= after


# =============================================================================
# TransportBase Tests - Circuit Reset
# =============================================================================


class TestCircuitReset:
    """Tests for circuit reset functionality."""

    def test_reset_circuit(self):
        """reset_circuit resets a single target."""
        transport = MockTransport(
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=1)
        )
        transport.record_failure("target-a")
        assert transport.get_circuit_state("target-a") == TransportState.OPEN

        transport.reset_circuit("target-a")
        assert transport.get_circuit_state("target-a") == TransportState.CLOSED

    def test_reset_circuit_clears_counts(self):
        """reset_circuit clears failure and success counts."""
        transport = MockTransport()
        transport.record_success("target-a")
        transport.record_failure("target-a")

        transport.reset_circuit("target-a")
        status = transport._target_status["target-a"]
        assert status.failure_count == 0
        assert status.success_count == 0

    def test_reset_all_circuits(self):
        """reset_all_circuits clears all tracked targets."""
        transport = MockTransport(
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=1)
        )
        transport.record_failure("target-a")
        transport.record_failure("target-b")

        transport.reset_all_circuits()
        assert len(transport._target_status) == 0
        assert transport.get_circuit_state("target-a") == TransportState.CLOSED
        assert transport.get_circuit_state("target-b") == TransportState.CLOSED

    def test_get_all_circuit_states(self):
        """get_all_circuit_states returns all tracked targets."""
        transport = MockTransport()
        transport.record_success("target-a")
        transport.record_failure("target-b")

        states = transport.get_all_circuit_states()
        assert "target-a" in states
        assert "target-b" in states


# =============================================================================
# TransportBase Tests - Timeout Execution
# =============================================================================


class TestTimeoutExecution:
    """Tests for timeout execution helpers."""

    @pytest.mark.asyncio
    async def test_execute_with_timeout_success(self):
        """execute_with_timeout returns result on success."""
        transport = MockTransport()

        async def quick_op():
            return "result"

        result = await transport.execute_with_timeout(quick_op(), timeout=1.0)
        assert result == "result"

    @pytest.mark.asyncio
    async def test_execute_with_timeout_raises_on_timeout(self):
        """execute_with_timeout raises TransportError on timeout."""
        transport = MockTransport()

        async def slow_op():
            await asyncio.sleep(1.0)
            return "result"

        with pytest.raises(TransportError) as exc_info:
            await transport.execute_with_timeout(slow_op(), timeout=0.01)

        assert "timed out" in str(exc_info.value).lower() or exc_info.value.transport

    @pytest.mark.asyncio
    async def test_execute_with_timeout_custom_message(self):
        """execute_with_timeout uses custom error message."""
        transport = MockTransport()

        async def slow_op():
            await asyncio.sleep(1.0)

        with pytest.raises(TransportError) as exc_info:
            await transport.execute_with_timeout(
                slow_op(),
                timeout=0.01,
                timeout_error_msg="Custom timeout message",
            )

        assert "Custom timeout message" in str(exc_info.value)


# =============================================================================
# TransportBase Tests - Retry Logic
# =============================================================================


class TestRetryLogic:
    """Tests for retry logic with exponential backoff."""

    @pytest.mark.asyncio
    async def test_execute_with_retry_success_first_try(self):
        """execute_with_retry returns on first success."""
        transport = MockTransport()
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await transport.execute_with_retry(
            operation,
            target="target-a",
            max_retries=3,
        )
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_execute_with_retry_retries_on_failure(self):
        """execute_with_retry retries on failure."""
        transport = MockTransport(
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=10)
        )
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = await transport.execute_with_retry(
            operation,
            target="target-a",
            max_retries=3,
            backoff_base=0.01,  # Fast backoff for testing
        )
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_execute_with_retry_fails_after_max_retries(self):
        """execute_with_retry raises after max retries exceeded."""
        transport = MockTransport(
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=10)
        )
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(TransportError) as exc_info:
            await transport.execute_with_retry(
                operation,
                target="target-a",
                max_retries=2,
                backoff_base=0.01,
            )

        assert "All 3 attempts failed" in str(exc_info.value)
        assert call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_execute_with_retry_respects_circuit_breaker(self):
        """execute_with_retry respects circuit breaker state."""
        transport = MockTransport(
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=1)
        )
        transport.record_failure("target-a")  # Open circuit

        async def operation():
            return "success"

        with pytest.raises(TransportError) as exc_info:
            await transport.execute_with_retry(
                operation,
                target="target-a",
            )

        assert "Circuit open" in str(exc_info.value)


# =============================================================================
# TransportBase Tests - State Persistence
# =============================================================================


class TestStatePersistence:
    """Tests for state persistence (load/save JSON)."""

    def test_load_state_no_file(self):
        """_load_state returns empty dict when no state file."""
        transport = MockTransport()
        state = transport._load_state()
        assert state == {}

    def test_load_state_no_path(self):
        """_load_state returns empty dict when no state_path configured."""
        transport = MockTransport()
        assert transport._state_path is None
        state = transport._load_state()
        assert state == {}

    def test_save_state_no_path(self):
        """_save_state returns False when no state_path configured."""
        transport = MockTransport()
        assert transport._save_state({"key": "value"}) is False

    def test_save_and_load_state(self):
        """_save_state and _load_state round-trip correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "transport_state.json"
            transport = MockTransport(state_path=state_path)

            data = {"targets": ["a", "b"], "count": 42}
            assert transport._save_state(data) is True
            assert state_path.exists()

            loaded = transport._load_state()
            assert loaded == data

    def test_load_state_handles_corrupt_json(self):
        """_load_state returns empty dict on corrupt JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "corrupt.json"
            state_path.write_text("not valid json {{{")

            transport = MockTransport(state_path=state_path)
            state = transport._load_state()
            assert state == {}

    def test_save_state_creates_parent_dirs(self):
        """_save_state creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "subdir" / "nested" / "state.json"
            transport = MockTransport(state_path=state_path)

            assert transport._save_state({"test": True}) is True
            assert state_path.exists()


# =============================================================================
# TransportBase Tests - Health Check
# =============================================================================


class TestHealthCheck:
    """Tests for health check functionality."""

    def test_health_check_no_targets(self):
        """Health check returns healthy when no targets tracked."""
        transport = MockTransport()
        result = transport.health_check()

        assert result.healthy is True
        assert "No targets tracked" in result.message
        assert result.details["total_targets"] == 0

    def test_health_check_all_healthy(self):
        """Health check returns healthy when all circuits closed."""
        transport = MockTransport()
        transport.record_success("target-a")
        transport.record_success("target-b")

        result = transport.health_check()
        assert result.healthy is True
        assert result.details["total_targets"] == 2
        assert result.details["open_circuits"] == 0

    def test_health_check_some_open(self):
        """Health check returns degraded when some circuits open."""
        transport = MockTransport(
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=1)
        )
        transport.record_success("target-a")
        transport.record_failure("target-b")  # Opens circuit

        result = transport.health_check()
        assert result.healthy is True  # Degraded is still operational
        assert "degraded" in result.status.value.lower()
        assert result.details["open_circuits"] == 1

    def test_health_check_all_open(self):
        """Health check returns unhealthy when all circuits open."""
        transport = MockTransport(
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=1)
        )
        transport.record_failure("target-a")
        transport.record_failure("target-b")

        result = transport.health_check()
        assert result.healthy is False
        assert result.details["open_circuits"] == 2

    def test_health_check_includes_statistics(self):
        """Health check includes operation statistics."""
        transport = MockTransport()
        transport.record_success("target-a")
        transport.record_success("target-a")
        transport.record_failure("target-b")

        result = transport.health_check()
        assert result.details["total_operations"] == 3
        assert result.details["total_successes"] == 2
        assert result.details["total_failures"] == 1
        assert result.details["success_rate"] == pytest.approx(2 / 3, rel=0.01)


# =============================================================================
# TransportBase Tests - Statistics
# =============================================================================


class TestStatistics:
    """Tests for statistics gathering."""

    def test_get_statistics_empty(self):
        """get_statistics returns zeros when no operations."""
        transport = MockTransport()
        stats = transport.get_statistics()

        assert stats["name"] == "MockTransport"
        assert stats["total_operations"] == 0
        assert stats["total_successes"] == 0
        assert stats["total_failures"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["targets_tracked"] == 0

    def test_get_statistics_with_operations(self):
        """get_statistics returns correct counts after operations."""
        transport = MockTransport(
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=10)
        )
        transport.record_success("target-a")
        transport.record_success("target-a")
        transport.record_failure("target-a")

        stats = transport.get_statistics()
        assert stats["total_operations"] == 3
        assert stats["total_successes"] == 2
        assert stats["total_failures"] == 1
        assert stats["success_rate"] == pytest.approx(2 / 3, rel=0.01)

    def test_get_statistics_counts_open_circuits(self):
        """get_statistics counts open circuits."""
        transport = MockTransport(
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=1)
        )
        transport.record_failure("target-a")
        transport.record_failure("target-b")
        transport.record_success("target-c")

        stats = transport.get_statistics()
        assert stats["circuits_open"] == 2


# =============================================================================
# TransportBase Tests - Independent Target State
# =============================================================================


class TestIndependentTargetState:
    """Tests for independent circuit state per target."""

    def test_targets_have_independent_state(self):
        """Each target has independent circuit state."""
        transport = MockTransport(
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=1)
        )

        transport.record_failure("target-a")  # Opens circuit
        transport.record_success("target-b")

        assert transport.get_circuit_state("target-a") == TransportState.OPEN
        assert transport.get_circuit_state("target-b") == TransportState.CLOSED
        assert transport.can_attempt("target-a") is False
        assert transport.can_attempt("target-b") is True

    def test_reset_circuit_only_affects_target(self):
        """reset_circuit only affects the specified target."""
        transport = MockTransport(
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=1)
        )

        transport.record_failure("target-a")
        transport.record_failure("target-b")

        transport.reset_circuit("target-a")
        assert transport.get_circuit_state("target-a") == TransportState.CLOSED
        assert transport.get_circuit_state("target-b") == TransportState.OPEN
