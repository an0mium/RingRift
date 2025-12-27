"""Tests for tracing.py - Distributed tracing infrastructure.

Tests cover:
- TraceSpan dataclass
- TraceContext creation and span management
- Trace ID generation and propagation
- Context managers (new_trace, with_trace, span)
- traced decorator
- Event integration (inject/extract trace)
- HTTP header integration
- TraceCollector for trace storage and analysis
"""

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.tracing import (
    TraceSpan,
    TraceContext,
    TraceCollector,
    generate_trace_id,
    generate_span_id,
    get_trace_id,
    get_trace_context,
    set_trace_id,
    new_trace,
    with_trace,
    span,
    traced,
    inject_trace_into_event,
    extract_trace_from_event,
    inject_trace_into_headers,
    extract_trace_from_headers,
    get_trace_collector,
    collect_trace,
    _current_trace,
)


class TestTraceSpan:
    """Tests for TraceSpan dataclass."""

    def test_creation_minimal(self):
        """Test creating TraceSpan with minimal fields."""
        span = TraceSpan(
            span_id="span-12345678",
            name="test_operation",
            trace_id="trace-abcdef0123456789",
        )
        assert span.span_id == "span-12345678"
        assert span.name == "test_operation"
        assert span.trace_id == "trace-abcdef0123456789"
        assert span.parent_span_id is None
        assert span.status == "ok"

    def test_creation_with_all_fields(self):
        """Test creating TraceSpan with all fields."""
        span = TraceSpan(
            span_id="span-12345678",
            name="test_operation",
            trace_id="trace-abcdef0123456789",
            parent_span_id="span-parent",
            start_time=1000.0,
            end_time=1500.0,
            duration_ms=500.0,
            status="error",
            tags={"key": "value"},
            events=[{"name": "event1"}],
        )
        assert span.parent_span_id == "span-parent"
        assert span.duration_ms == 500.0
        assert span.status == "error"
        assert span.tags == {"key": "value"}
        assert len(span.events) == 1

    def test_to_dict(self):
        """Test TraceSpan.to_dict() serialization."""
        span = TraceSpan(
            span_id="span-123",
            name="test",
            trace_id="trace-abc",
            start_time=100.0,
            end_time=200.0,
            duration_ms=100000.0,
            tags={"tag1": "val1"},
        )
        d = span.to_dict()
        assert d["span_id"] == "span-123"
        assert d["name"] == "test"
        assert d["trace_id"] == "trace-abc"
        assert d["tags"] == {"tag1": "val1"}
        assert "events" in d


class TestTraceContext:
    """Tests for TraceContext dataclass."""

    def test_new_creation(self):
        """Test TraceContext.new() factory method."""
        ctx = TraceContext.new("test_trace", env="test")
        assert ctx.trace_id.startswith("trace-")
        assert ctx.name == "test_trace"
        assert ctx.tags["env"] == "test"
        assert ctx.start_time > 0

    def test_from_trace_id(self):
        """Test TraceContext.from_trace_id() factory method."""
        ctx = TraceContext.from_trace_id("trace-existing123", "continued_trace")
        assert ctx.trace_id == "trace-existing123"
        assert ctx.name == "continued_trace"

    def test_start_span(self):
        """Test starting a span within a trace."""
        ctx = TraceContext.new("test")
        span = ctx.start_span("operation1", key="value")

        assert span.name == "operation1"
        assert span.trace_id == ctx.trace_id
        assert span.parent_span_id is None
        assert span.tags["key"] == "value"
        assert len(ctx.spans) == 1

    def test_nested_spans(self):
        """Test nested span hierarchy."""
        ctx = TraceContext.new("test")
        span1 = ctx.start_span("parent")
        span2 = ctx.start_span("child")

        assert span2.parent_span_id == span1.span_id
        assert len(ctx.spans) == 2

    def test_end_span(self):
        """Test ending a span."""
        ctx = TraceContext.new("test")
        span = ctx.start_span("operation")
        time.sleep(0.01)  # Small delay
        ctx.end_span(status="ok", result="success")

        assert span.end_time > span.start_time
        assert span.duration_ms > 0
        assert span.status == "ok"
        assert span.tags["result"] == "success"

    def test_end_span_returns_to_parent(self):
        """Test that ending span returns to parent span."""
        ctx = TraceContext.new("test")
        parent = ctx.start_span("parent")
        ctx.start_span("child")
        ctx.end_span()

        assert ctx._current_span == parent

    def test_add_event(self):
        """Test adding event to current span."""
        ctx = TraceContext.new("test")
        ctx.start_span("operation")
        ctx.add_event("checkpoint", step=1, value=42)

        assert len(ctx._current_span.events) == 1
        event = ctx._current_span.events[0]
        assert event["name"] == "checkpoint"
        assert event["attributes"]["step"] == 1

    def test_set_tag_on_span(self):
        """Test setting tag on current span."""
        ctx = TraceContext.new("test")
        ctx.start_span("operation")
        ctx.set_tag("important", True)

        assert ctx._current_span.tags["important"] is True

    def test_set_tag_on_trace(self):
        """Test setting tag on trace (no active span)."""
        ctx = TraceContext.new("test")
        ctx.set_tag("trace_level", "value")

        assert ctx.tags["trace_level"] == "value"

    def test_to_dict(self):
        """Test TraceContext.to_dict() serialization."""
        ctx = TraceContext.new("test", key="value")
        ctx.start_span("span1")
        ctx.end_span()

        d = ctx.to_dict()
        assert d["trace_id"] == ctx.trace_id
        assert d["name"] == "test"
        assert d["tags"]["key"] == "value"
        assert len(d["spans"]) == 1

    def test_get_duration_ms(self):
        """Test get_duration_ms() method."""
        ctx = TraceContext.new("test")
        time.sleep(0.01)
        duration = ctx.get_duration_ms()
        assert duration >= 10.0  # At least 10ms


class TestIdGeneration:
    """Tests for ID generation functions."""

    def test_generate_trace_id_format(self):
        """Test trace ID format."""
        trace_id = generate_trace_id()
        assert trace_id.startswith("trace-")
        assert len(trace_id) == len("trace-") + 16

    def test_generate_trace_id_unique(self):
        """Test trace IDs are unique."""
        ids = [generate_trace_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_generate_span_id_format(self):
        """Test span ID format."""
        span_id = generate_span_id()
        assert span_id.startswith("span-")
        assert len(span_id) == len("span-") + 8

    def test_generate_span_id_unique(self):
        """Test span IDs are unique."""
        ids = [generate_span_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestTraceContextVars:
    """Tests for trace context variable management."""

    def teardown_method(self):
        """Clean up context after each test."""
        _current_trace.set(None)

    def test_get_trace_id_when_no_trace(self):
        """Test get_trace_id() returns None when no trace active."""
        assert get_trace_id() is None

    def test_get_trace_context_when_no_trace(self):
        """Test get_trace_context() returns None when no trace active."""
        assert get_trace_context() is None

    def test_set_trace_id(self):
        """Test set_trace_id() creates and sets context."""
        ctx = set_trace_id("trace-custom123", "my_trace")

        assert ctx.trace_id == "trace-custom123"
        assert ctx.name == "my_trace"
        assert get_trace_id() == "trace-custom123"
        assert get_trace_context() == ctx

    def test_trace_context_isolated_between_threads(self):
        """Test trace context is thread-isolated."""
        results = {}

        def thread_func(thread_id):
            with new_trace(f"thread_{thread_id}"):
                results[thread_id] = get_trace_id()
                time.sleep(0.01)  # Allow other thread to run

        t1 = threading.Thread(target=thread_func, args=("t1",))
        t2 = threading.Thread(target=thread_func, args=("t2",))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Each thread should have its own trace
        assert results["t1"] != results["t2"]


class TestNewTraceContextManager:
    """Tests for new_trace() context manager."""

    def teardown_method(self):
        """Clean up context after each test."""
        _current_trace.set(None)

    def test_new_trace_creates_context(self):
        """Test new_trace() creates and sets trace context."""
        with new_trace("test_trace") as ctx:
            assert ctx.trace_id.startswith("trace-")
            assert ctx.name == "test_trace"
            assert get_trace_id() == ctx.trace_id

    def test_new_trace_with_tags(self):
        """Test new_trace() with tags."""
        with new_trace("test", env="prod", version="1.0") as ctx:
            assert ctx.tags["env"] == "prod"
            assert ctx.tags["version"] == "1.0"

    def test_new_trace_clears_after_exit(self):
        """Test trace context is cleared after exiting."""
        with new_trace("test"):
            pass
        assert get_trace_id() is None

    def test_new_trace_clears_on_exception(self):
        """Test trace context is cleared even on exception."""
        with pytest.raises(ValueError):
            with new_trace("test"):
                raise ValueError("test error")
        assert get_trace_id() is None


class TestWithTraceContextManager:
    """Tests for with_trace() context manager."""

    def teardown_method(self):
        """Clean up context after each test."""
        _current_trace.set(None)

    def test_with_trace_continues_trace(self):
        """Test with_trace() continues an existing trace."""
        existing_id = "trace-existing12345"
        with with_trace(existing_id, "continued") as ctx:
            assert ctx.trace_id == existing_id
            assert ctx.name == "continued"
            assert get_trace_id() == existing_id

    def test_with_trace_with_tags(self):
        """Test with_trace() with additional tags."""
        with with_trace("trace-123", "segment", extra="value") as ctx:
            assert ctx.tags["extra"] == "value"

    def test_with_trace_clears_after_exit(self):
        """Test context is cleared after exiting."""
        with with_trace("trace-123", "test"):
            pass
        assert get_trace_id() is None


class TestSpanContextManager:
    """Tests for span() context manager."""

    def teardown_method(self):
        """Clean up context after each test."""
        _current_trace.set(None)

    def test_span_within_trace(self):
        """Test span() within an active trace."""
        with new_trace("test") as ctx:
            with span("operation") as s:
                assert s is not None
                assert s.name == "operation"
                assert s.trace_id == ctx.trace_id

    def test_span_with_tags(self):
        """Test span() with tags."""
        with new_trace("test"):
            with span("operation", key="value") as s:
                assert s.tags["key"] == "value"

    def test_span_auto_ends_on_success(self):
        """Test span ends with ok status on success."""
        with new_trace("test") as ctx:
            with span("operation"):
                pass
            assert len(ctx.spans) == 1
            assert ctx.spans[0].status == "ok"
            assert ctx.spans[0].end_time > 0

    def test_span_ends_with_error_on_exception(self):
        """Test span ends with error status on exception."""
        with new_trace("test") as ctx:
            with pytest.raises(ValueError):
                with span("operation"):
                    raise ValueError("test")
            assert ctx.spans[0].status == "error"
            assert "test" in ctx.spans[0].tags["error"]

    def test_span_creates_trace_if_none(self):
        """Test span() creates auto-trace if no trace active."""
        with span("standalone") as s:
            assert s is not None
            # Should have created an auto-trace
            assert get_trace_id() is not None


class TestTracedDecorator:
    """Tests for @traced decorator."""

    def teardown_method(self):
        """Clean up context after each test."""
        _current_trace.set(None)

    def test_traced_sync_function(self):
        """Test @traced on sync function."""
        @traced()
        def my_func(x):
            return x * 2

        with new_trace("test") as ctx:
            result = my_func(21)

        assert result == 42
        assert len(ctx.spans) == 1
        assert ctx.spans[0].name == "my_func"

    def test_traced_with_custom_name(self):
        """Test @traced with custom span name."""
        @traced(name="custom_operation")
        def my_func():
            pass

        with new_trace("test") as ctx:
            my_func()

        assert ctx.spans[0].name == "custom_operation"

    @pytest.mark.asyncio
    async def test_traced_async_function(self):
        """Test @traced on async function."""
        @traced()
        async def my_async_func(x):
            await asyncio.sleep(0)
            return x + 1

        with new_trace("test") as ctx:
            result = await my_async_func(5)

        assert result == 6
        assert len(ctx.spans) == 1
        assert ctx.spans[0].name == "my_async_func"


class TestEventIntegration:
    """Tests for event trace injection/extraction."""

    def teardown_method(self):
        """Clean up context after each test."""
        _current_trace.set(None)

    def test_inject_trace_into_event(self):
        """Test injecting trace_id into event payload."""
        event = MagicMock()
        event.payload = {"data": "value"}

        with new_trace("test") as ctx:
            inject_trace_into_event(event)

        assert event.payload["trace_id"] == ctx.trace_id

    def test_inject_trace_no_active_trace(self):
        """Test inject does nothing when no trace active."""
        event = MagicMock()
        event.payload = {"data": "value"}

        inject_trace_into_event(event)

        assert "trace_id" not in event.payload

    def test_inject_trace_non_dict_payload(self):
        """Test inject handles non-dict payload gracefully."""
        event = MagicMock()
        event.payload = "string_payload"

        with new_trace("test"):
            inject_trace_into_event(event)  # Should not crash

    def test_extract_trace_from_event(self):
        """Test extracting trace_id from event payload."""
        event = MagicMock()
        event.payload = {"trace_id": "trace-abc123", "data": "value"}

        result = extract_trace_from_event(event)

        assert result == "trace-abc123"

    def test_extract_trace_no_trace_id(self):
        """Test extract returns None when no trace_id."""
        event = MagicMock()
        event.payload = {"data": "value"}

        result = extract_trace_from_event(event)

        assert result is None


class TestHttpIntegration:
    """Tests for HTTP header trace injection/extraction."""

    def teardown_method(self):
        """Clean up context after each test."""
        _current_trace.set(None)

    def test_inject_trace_into_headers(self):
        """Test injecting trace_id into HTTP headers."""
        headers = {"Content-Type": "application/json"}

        with new_trace("test") as ctx:
            result = inject_trace_into_headers(headers)

        assert result["X-Trace-Id"] == ctx.trace_id

    def test_inject_span_id_into_headers(self):
        """Test injecting span_id into HTTP headers."""
        headers = {}

        with new_trace("test") as ctx:
            ctx.start_span("operation")
            result = inject_trace_into_headers(headers)

        assert "X-Span-Id" in result
        assert result["X-Span-Id"] == ctx._current_span.span_id

    def test_inject_no_active_trace(self):
        """Test inject preserves headers when no trace."""
        headers = {"Existing": "header"}
        result = inject_trace_into_headers(headers)

        assert result == {"Existing": "header"}

    def test_extract_trace_from_headers_uppercase(self):
        """Test extracting trace_id from headers (uppercase)."""
        headers = {"X-Trace-Id": "trace-from-header"}
        result = extract_trace_from_headers(headers)
        assert result == "trace-from-header"

    def test_extract_trace_from_headers_lowercase(self):
        """Test extracting trace_id from headers (lowercase)."""
        headers = {"x-trace-id": "trace-from-header"}
        result = extract_trace_from_headers(headers)
        assert result == "trace-from-header"

    def test_extract_trace_no_header(self):
        """Test extract returns None when no trace header."""
        headers = {"Content-Type": "application/json"}
        result = extract_trace_from_headers(headers)
        assert result is None


class TestTraceCollector:
    """Tests for TraceCollector class."""

    def test_collect_trace(self):
        """Test collecting a trace."""
        collector = TraceCollector()
        ctx = TraceContext.new("test")
        ctx.start_span("op")
        ctx.end_span()

        collector.collect(ctx)

        traces = collector.get_traces()
        assert len(traces) == 1
        assert traces[0]["name"] == "test"

    def test_max_traces_limit(self):
        """Test collector respects max_traces limit."""
        collector = TraceCollector(max_traces=5)

        for i in range(10):
            ctx = TraceContext.new(f"trace_{i}")
            collector.collect(ctx)

        traces = collector.get_traces()
        assert len(traces) == 5
        assert traces[-1]["name"] == "trace_9"  # Most recent

    def test_find_trace(self):
        """Test finding a trace by ID."""
        collector = TraceCollector()
        ctx = TraceContext.new("findable")
        collector.collect(ctx)

        found = collector.find_trace(ctx.trace_id)
        assert found is not None
        assert found["name"] == "findable"

    def test_find_trace_not_found(self):
        """Test finding non-existent trace."""
        collector = TraceCollector()
        result = collector.find_trace("trace-nonexistent")
        assert result is None

    def test_get_slow_traces(self):
        """Test getting slow traces."""
        collector = TraceCollector()

        # Fast trace - set duration_ms AFTER end_span (which calculates real duration)
        fast = TraceContext.new("fast")
        fast.start_span("op")
        fast.end_span()
        # Override the calculated duration with our test value
        fast.spans[-1].duration_ms = 100
        collector.collect(fast)

        # Slow trace - set duration_ms AFTER end_span
        slow = TraceContext.new("slow")
        slow.start_span("op")
        slow.end_span()
        # Override the calculated duration with our test value
        slow.spans[-1].duration_ms = 2000
        collector.collect(slow)

        slow_traces = collector.get_slow_traces(threshold_ms=1000)
        assert len(slow_traces) == 1
        assert slow_traces[0]["name"] == "slow"

    def test_thread_safe_collection(self):
        """Test collector is thread-safe."""
        collector = TraceCollector(max_traces=1000)
        errors = []

        def collect_traces():
            try:
                for i in range(100):
                    ctx = TraceContext.new(f"trace_{threading.current_thread().name}_{i}")
                    collector.collect(ctx)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=collect_traces) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(collector.get_traces()) <= 1000


class TestGlobalCollector:
    """Tests for global trace collector functions."""

    def test_get_trace_collector_singleton(self):
        """Test get_trace_collector returns same instance."""
        c1 = get_trace_collector()
        c2 = get_trace_collector()
        assert c1 is c2

    def test_collect_trace_function(self):
        """Test collect_trace() uses global collector."""
        ctx = TraceContext.new("global_test")
        collect_trace(ctx)

        collector = get_trace_collector()
        found = collector.find_trace(ctx.trace_id)
        assert found is not None


class TestExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Test that __all__ exports are defined."""
        from app.coordination import tracing
        expected = [
            "TraceCollector",
            "TraceContext",
            "TraceSpan",
            "collect_trace",
            "extract_trace_from_event",
            "extract_trace_from_headers",
            "generate_span_id",
            "generate_trace_id",
            "get_trace_collector",
            "get_trace_context",
            "get_trace_id",
            "inject_trace_into_event",
            "inject_trace_into_headers",
            "new_trace",
            "set_trace_id",
            "span",
            "traced",
            "with_trace",
        ]
        for name in expected:
            assert name in tracing.__all__
