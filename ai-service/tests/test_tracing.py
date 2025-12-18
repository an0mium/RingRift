"""Tests for OpenTelemetry tracing module.

Tests the tracing infrastructure:
- Setup and configuration
- Decorators and context managers
- Graceful degradation when OTel not available
- Context propagation
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from app.tracing import (
    setup_tracing,
    shutdown_tracing,
    get_tracer,
    get_current_span,
    is_tracing_enabled,
    traced,
    traced_async,
    inject_trace_context,
    extract_trace_context,
    add_ai_move_attributes,
    add_training_attributes,
    NoOpSpan,
    NoOpTracer,
    HAS_OPENTELEMETRY,
)


class TestTracingSetup:
    """Tests for tracing setup and configuration."""

    def teardown_method(self):
        """Clean up after each test."""
        shutdown_tracing()

    def test_setup_with_none_exporter_disables_tracing(self):
        """Setting exporter to 'none' should disable tracing."""
        result = setup_tracing(exporter="none")
        assert result is False
        assert is_tracing_enabled() is False

    def test_setup_respects_env_disabled(self):
        """OTEL_TRACING_ENABLED=false should disable tracing."""
        with patch.dict(os.environ, {"OTEL_TRACING_ENABLED": "false"}):
            result = setup_tracing(exporter="console")
            assert result is False
            assert is_tracing_enabled() is False

    @pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
    def test_setup_with_console_exporter(self):
        """Console exporter should work."""
        result = setup_tracing(
            service_name="test-service",
            exporter="console",
        )
        assert result is True
        assert is_tracing_enabled() is True

    def test_shutdown_clears_state(self):
        """Shutdown should clear tracing state."""
        setup_tracing(exporter="console")
        shutdown_tracing()
        assert is_tracing_enabled() is False


class TestNoOpClasses:
    """Tests for no-op fallback classes."""

    def test_noop_span_methods(self):
        """NoOpSpan should accept all method calls without error."""
        span = NoOpSpan()

        # All methods should work without error
        span.set_attribute("key", "value")
        span.set_attributes({"key1": "value1", "key2": 2})
        span.add_event("event_name", {"attr": "value"})
        span.record_exception(ValueError("test"))
        span.set_status(None)
        span.end()

    def test_noop_span_context_manager(self):
        """NoOpSpan should work as context manager."""
        span = NoOpSpan()
        with span as s:
            s.set_attribute("key", "value")

    def test_noop_tracer_returns_noop_span(self):
        """NoOpTracer should return NoOpSpan."""
        tracer = NoOpTracer()

        span = tracer.start_span("test")
        assert isinstance(span, NoOpSpan)

        with tracer.start_as_current_span("test") as span:
            assert isinstance(span, NoOpSpan)


class TestTracedDecorator:
    """Tests for the @traced decorator."""

    def teardown_method(self):
        shutdown_tracing()

    def test_traced_function_works_when_disabled(self):
        """@traced should not affect function when tracing disabled."""
        shutdown_tracing()

        @traced("test_operation")
        def test_func(x, y):
            return x + y

        result = test_func(1, 2)
        assert result == 3

    def test_traced_preserves_exceptions(self):
        """@traced should preserve exceptions."""
        @traced("failing_operation")
        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_func()

    def test_traced_with_attributes(self):
        """@traced should accept static attributes."""
        @traced("test_op", attributes={"component": "test"})
        def test_func():
            return "ok"

        result = test_func()
        assert result == "ok"

    @pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
    def test_traced_creates_span_when_enabled(self):
        """@traced should create span when tracing enabled."""
        setup_tracing(exporter="console")

        @traced("test_operation")
        def test_func():
            span = get_current_span()
            # Span should exist when tracing enabled
            assert span is not None
            return "ok"

        result = test_func()
        assert result == "ok"


class TestTracedAsyncDecorator:
    """Tests for the @traced_async decorator."""

    def teardown_method(self):
        shutdown_tracing()

    @pytest.mark.asyncio
    async def test_traced_async_works_when_disabled(self):
        """@traced_async should not affect async function when tracing disabled."""
        shutdown_tracing()

        @traced_async("test_async_operation")
        async def test_async_func(x, y):
            return x * y

        result = await test_async_func(3, 4)
        assert result == 12

    @pytest.mark.asyncio
    async def test_traced_async_preserves_exceptions(self):
        """@traced_async should preserve async exceptions."""
        @traced_async("failing_async")
        async def failing_async():
            raise RuntimeError("async error")

        with pytest.raises(RuntimeError, match="async error"):
            await failing_async()


class TestContextPropagation:
    """Tests for trace context propagation."""

    def teardown_method(self):
        shutdown_tracing()

    def test_inject_noop_when_disabled(self):
        """inject_trace_context should be noop when disabled."""
        shutdown_tracing()
        carrier = {}
        inject_trace_context(carrier)
        # Should not raise, carrier may or may not be modified

    def test_extract_returns_none_when_disabled(self):
        """extract_trace_context should return None when disabled."""
        shutdown_tracing()
        result = extract_trace_context({"traceparent": "00-..."})
        assert result is None


class TestAttributeHelpers:
    """Tests for attribute helper functions."""

    def test_add_ai_move_attributes(self):
        """add_ai_move_attributes should work with NoOpSpan."""
        span = NoOpSpan()
        add_ai_move_attributes(
            span,
            board_type="square8",
            difficulty=3,
            engine_type="mcts",
            simulations=800,
            depth=0,
            time_ms=150.5,
        )
        # Should complete without error

    def test_add_training_attributes(self):
        """add_training_attributes should work with NoOpSpan."""
        span = NoOpSpan()
        add_training_attributes(
            span,
            epoch=5,
            batch_size=256,
            learning_rate=0.001,
            loss=0.5,
        )
        # Should complete without error


class TestGetTracer:
    """Tests for get_tracer function."""

    def teardown_method(self):
        shutdown_tracing()

    def test_get_tracer_returns_noop_when_disabled(self):
        """get_tracer should return NoOpTracer when OTel not available."""
        shutdown_tracing()

        if not HAS_OPENTELEMETRY:
            tracer = get_tracer("test")
            assert isinstance(tracer, NoOpTracer)

    @pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
    def test_get_tracer_returns_real_tracer_when_available(self):
        """get_tracer should return real tracer when OTel available."""
        # Even without setup, should return a tracer
        tracer = get_tracer("test")
        assert tracer is not None


class TestGetCurrentSpan:
    """Tests for get_current_span function."""

    def teardown_method(self):
        shutdown_tracing()

    def test_get_current_span_returns_noop_when_disabled(self):
        """get_current_span should return NoOpSpan when disabled."""
        shutdown_tracing()
        span = get_current_span()
        assert isinstance(span, NoOpSpan)
