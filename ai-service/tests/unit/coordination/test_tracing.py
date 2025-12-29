"""Tests for app.coordination.tracing module.

Comprehensive tests for distributed tracing infrastructure including:
- TraceSpan and TraceContext dataclasses
- Helper functions for trace/span ID generation
- Context managers (new_trace, with_trace, span)
- @traced decorator
- Event and header integration
- TraceCollector class
"""

import asyncio
import pytest
import time
import threading
from unittest.mock import MagicMock, patch

from app.coordination.tracing import (
    # Dataclasses
    TraceSpan,
    TraceContext,
    # Helper functions
    generate_trace_id,
    generate_span_id,
    get_trace_id,
    get_trace_context,
    set_trace_id,
    # Context managers
    new_trace,
    with_trace,
    span,
    # Decorator
    traced,
    # Event integration
    inject_trace_into_event,
    extract_trace_from_event,
    # Header integration
    inject_trace_into_headers,
    extract_trace_from_headers,
    # Trace collector
    TraceCollector,
    get_trace_collector,
    collect_trace,
    # Internal
    _current_trace,
)


class TestTraceSpan:
    """Tests for TraceSpan dataclass."""

    def test_default_values(self):
        """Test TraceSpan default field values."""
        span = TraceSpan(span_id="span-123", name="test_span", trace_id="trace-456")
        assert span.span_id == "span-123"
        assert span.name == "test_span"
        assert span.trace_id == "trace-456"
        assert span.parent_span_id is None
        assert span.start_time == 0.0
        assert span.end_time == 0.0
        assert span.duration_ms == 0.0
        assert span.status == "ok"
        assert span.tags == {}
        assert span.events == []

    def test_custom_values(self):
        """Test TraceSpan with custom values."""
        span = TraceSpan(
            span_id="span-abc",
            name="custom_span",
            trace_id="trace-xyz",
            parent_span_id="span-parent",
            start_time=1000.0,
            end_time=1005.5,
            duration_ms=5500.0,
            status="error",
            tags={"key": "value"},
            events=[{"name": "event1"}],
        )
        assert span.parent_span_id == "span-parent"
        assert span.start_time == 1000.0
        assert span.end_time == 1005.5
        assert span.duration_ms == 5500.0
        assert span.status == "error"
        assert span.tags == {"key": "value"}
        assert span.events == [{"name": "event1"}]

    def test_to_dict(self):
        """Test TraceSpan.to_dict() serialization."""
        span = TraceSpan(
            span_id="span-dict",
            name="dict_span",
            trace_id="trace-dict",
            parent_span_id="span-parent",
            start_time=100.0,
            end_time=200.0,
            duration_ms=100000.0,
            status="ok",
            tags={"foo": "bar"},
            events=[{"name": "e1"}, {"name": "e2"}],
        )
        d = span.to_dict()
        assert d["span_id"] == "span-dict"
        assert d["name"] == "dict_span"
        assert d["trace_id"] == "trace-dict"
        assert d["parent_span_id"] == "span-parent"
        assert d["start_time"] == 100.0
        assert d["end_time"] == 200.0
        assert d["duration_ms"] == 100000.0
        assert d["status"] == "ok"
        assert d["tags"] == {"foo": "bar"}
        assert d["events"] == [{"name": "e1"}, {"name": "e2"}]

    def test_to_dict_empty_collections(self):
        """Test to_dict with empty tags and events."""
        span = TraceSpan(span_id="s", name="n", trace_id="t")
        d = span.to_dict()
        assert d["tags"] == {}
        assert d["events"] == []


class TestTraceContext:
    """Tests for TraceContext dataclass."""

    def test_default_values(self):
        """Test TraceContext default values."""
        ctx = TraceContext(trace_id="trace-123", name="test_trace")
        assert ctx.trace_id == "trace-123"
        assert ctx.name == "test_trace"
        assert ctx.start_time > 0  # Should be set by __post_init__
        assert ctx.tags == {}
        assert ctx.spans == []
        assert ctx._current_span is None

    def test_post_init_sets_start_time(self):
        """Test __post_init__ sets start_time if not provided."""
        before = time.time()
        ctx = TraceContext(trace_id="t", name="n")
        after = time.time()
        assert before <= ctx.start_time <= after

    def test_post_init_preserves_start_time(self):
        """Test __post_init__ preserves explicit start_time."""
        ctx = TraceContext(trace_id="t", name="n", start_time=12345.0)
        assert ctx.start_time == 12345.0

    def test_new_classmethod(self):
        """Test TraceContext.new() factory."""
        ctx = TraceContext.new("my_trace", env="prod", version="1.0")
        assert ctx.name == "my_trace"
        assert ctx.trace_id.startswith("trace-")
        assert len(ctx.trace_id) == 22  # "trace-" + 16 hex chars
        assert ctx.tags == {"env": "prod", "version": "1.0"}
        assert ctx.start_time > 0

    def test_from_trace_id_classmethod(self):
        """Test TraceContext.from_trace_id() factory."""
        ctx = TraceContext.from_trace_id("existing-trace-id", "continued_segment")
        assert ctx.trace_id == "existing-trace-id"
        assert ctx.name == "continued_segment"
        assert ctx.start_time > 0

    def test_start_span(self):
        """Test starting a new span."""
        ctx = TraceContext.new("test")
        span = ctx.start_span("my_span", priority="high")

        assert span.span_id.startswith("span-")
        assert span.name == "my_span"
        assert span.trace_id == ctx.trace_id
        assert span.parent_span_id is None
        assert span.start_time > 0
        assert span.tags == {"priority": "high"}
        assert ctx._current_span is span
        assert len(ctx.spans) == 1

    def test_nested_spans(self):
        """Test nested span parent tracking."""
        ctx = TraceContext.new("test")
        outer = ctx.start_span("outer")
        inner = ctx.start_span("inner")

        assert inner.parent_span_id == outer.span_id
        assert ctx._current_span is inner

    def test_end_span(self):
        """Test ending the current span."""
        ctx = TraceContext.new("test")
        span = ctx.start_span("my_span")
        start = span.start_time

        time.sleep(0.01)  # Small delay
        ctx.end_span(status="ok", result="success")

        assert span.end_time > start
        assert span.duration_ms > 0
        assert span.status == "ok"
        assert span.tags["result"] == "success"
        assert ctx._current_span is None

    def test_end_nested_spans(self):
        """Test ending nested spans returns to parent."""
        ctx = TraceContext.new("test")
        outer = ctx.start_span("outer")
        inner = ctx.start_span("inner")

        ctx.end_span()
        assert ctx._current_span is outer

        ctx.end_span()
        assert ctx._current_span is None

    def test_end_span_no_current(self):
        """Test end_span when no span is active."""
        ctx = TraceContext.new("test")
        ctx.end_span()  # Should not raise
        assert ctx._current_span is None

    def test_add_event(self):
        """Test adding event to current span."""
        ctx = TraceContext.new("test")
        span = ctx.start_span("my_span")

        ctx.add_event("checkpoint", count=5, message="halfway")

        assert len(span.events) == 1
        event = span.events[0]
        assert event["name"] == "checkpoint"
        assert event["timestamp"] > 0
        assert event["attributes"] == {"count": 5, "message": "halfway"}

    def test_add_event_no_current_span(self):
        """Test add_event when no span is active."""
        ctx = TraceContext.new("test")
        ctx.add_event("orphan_event")  # Should not raise
        assert len(ctx.spans) == 0

    def test_set_tag_on_span(self):
        """Test setting tag on current span."""
        ctx = TraceContext.new("test")
        span = ctx.start_span("my_span")

        ctx.set_tag("version", "2.0")

        assert span.tags["version"] == "2.0"

    def test_set_tag_on_trace(self):
        """Test setting tag on trace when no span active."""
        ctx = TraceContext.new("test")
        ctx.set_tag("env", "test")
        assert ctx.tags["env"] == "test"

    def test_to_dict(self):
        """Test TraceContext.to_dict() serialization."""
        ctx = TraceContext.new("test", key="value")
        ctx.start_span("span1")
        ctx.end_span()

        d = ctx.to_dict()
        assert d["trace_id"] == ctx.trace_id
        assert d["name"] == "test"
        assert d["start_time"] == ctx.start_time
        assert d["tags"] == {"key": "value"}
        assert len(d["spans"]) == 1
        assert d["spans"][0]["name"] == "span1"

    def test_get_duration_ms(self):
        """Test get_duration_ms calculation."""
        ctx = TraceContext.new("test")
        time.sleep(0.05)  # 50ms
        duration = ctx.get_duration_ms()
        assert duration >= 50  # At least 50ms


class TestHelperFunctions:
    """Tests for trace helper functions."""

    def test_generate_trace_id(self):
        """Test trace ID generation format."""
        tid = generate_trace_id()
        assert tid.startswith("trace-")
        assert len(tid) == 22  # "trace-" (6) + 16 hex chars
        # Verify uniqueness
        tid2 = generate_trace_id()
        assert tid != tid2

    def test_generate_span_id(self):
        """Test span ID generation format."""
        sid = generate_span_id()
        assert sid.startswith("span-")
        assert len(sid) == 13  # "span-" (5) + 8 hex chars
        # Verify uniqueness
        sid2 = generate_span_id()
        assert sid != sid2

    def test_get_trace_id_no_context(self):
        """Test get_trace_id when no trace active."""
        _current_trace.set(None)
        assert get_trace_id() is None

    def test_get_trace_id_with_context(self):
        """Test get_trace_id returns current trace_id."""
        with new_trace("test") as ctx:
            assert get_trace_id() == ctx.trace_id

    def test_get_trace_context_no_context(self):
        """Test get_trace_context when no trace active."""
        _current_trace.set(None)
        assert get_trace_context() is None

    def test_get_trace_context_with_context(self):
        """Test get_trace_context returns current context."""
        with new_trace("test") as ctx:
            assert get_trace_context() is ctx

    def test_set_trace_id(self):
        """Test set_trace_id creates new context."""
        ctx = set_trace_id("my-custom-trace-id", "segment1")
        assert ctx.trace_id == "my-custom-trace-id"
        assert ctx.name == "segment1"
        assert get_trace_context() is ctx


class TestNewTraceContextManager:
    """Tests for new_trace context manager."""

    def test_creates_new_trace(self):
        """Test new_trace creates a new TraceContext."""
        with new_trace("my_operation") as ctx:
            assert ctx.name == "my_operation"
            assert ctx.trace_id.startswith("trace-")
            assert get_trace_context() is ctx

    def test_restores_context_on_exit(self):
        """Test new_trace restores previous context on exit."""
        _current_trace.set(None)
        with new_trace("op1"):
            with new_trace("op2") as inner:
                assert get_trace_context() is inner
            # After inner exits, outer is restored (but outer is a new context too)
        assert get_trace_context() is None

    def test_with_tags(self):
        """Test new_trace with tags."""
        with new_trace("tagged", env="prod", version="1.0") as ctx:
            assert ctx.tags == {"env": "prod", "version": "1.0"}

    def test_logs_on_exit(self):
        """Test new_trace logs summary on exit."""
        with patch("app.coordination.tracing.logger") as mock_logger:
            with new_trace("test_op"):
                pass
            mock_logger.debug.assert_called_once()
            call_args = mock_logger.debug.call_args[0][0]
            assert "test_op" in call_args
            assert "completed" in call_args


class TestWithTraceContextManager:
    """Tests for with_trace context manager."""

    def test_continues_existing_trace(self):
        """Test with_trace continues an existing trace ID."""
        with with_trace("existing-trace-123", "segment") as ctx:
            assert ctx.trace_id == "existing-trace-123"
            assert ctx.name == "segment"

    def test_with_tags(self):
        """Test with_trace with additional tags."""
        with with_trace("trace-id", "seg", key="value") as ctx:
            assert ctx.tags == {"key": "value"}

    def test_restores_context_on_exit(self):
        """Test with_trace restores previous context."""
        _current_trace.set(None)
        with with_trace("trace-1", "seg1"):
            pass
        assert get_trace_context() is None


class TestSpanContextManager:
    """Tests for span context manager."""

    def test_creates_span_in_existing_trace(self):
        """Test span creates span within existing trace."""
        with new_trace("outer") as ctx:
            with span("inner_span", priority="high") as s:
                assert s is not None
                assert s.name == "inner_span"
                assert s.tags["priority"] == "high"
                assert s.trace_id == ctx.trace_id

    def test_creates_auto_trace_if_none(self):
        """Test span creates auto-trace if no trace active."""
        _current_trace.set(None)
        with span("orphan_span") as s:
            # Should have created an auto-trace
            ctx = get_trace_context()
            assert ctx is not None
            assert ctx.name.startswith("auto-orphan_span")

    def test_ends_span_on_normal_exit(self):
        """Test span ends with ok status on normal exit."""
        with new_trace("test"):
            with span("my_span") as s:
                pass
            assert s.status == "ok"
            assert s.end_time > 0

    def test_ends_span_with_error_on_exception(self):
        """Test span ends with error status on exception."""
        with new_trace("test"):
            try:
                with span("failing_span") as s:
                    raise ValueError("test error")
            except ValueError:
                pass
            assert s.status == "error"
            assert "test error" in s.tags.get("error", "")


class TestTracedDecorator:
    """Tests for @traced decorator."""

    def test_traced_sync_function(self):
        """Test @traced on sync function."""
        @traced("my_func")
        def my_function(x, y):
            return x + y

        with new_trace("test") as ctx:
            result = my_function(2, 3)
            assert result == 5
            assert len(ctx.spans) == 1
            assert ctx.spans[0].name == "my_func"

    def test_traced_uses_function_name(self):
        """Test @traced uses function name if no name provided."""
        @traced()
        def auto_named_function():
            return 42

        with new_trace("test") as ctx:
            result = auto_named_function()
            assert result == 42
            assert ctx.spans[0].name == "auto_named_function"

    @pytest.mark.asyncio
    async def test_traced_async_function(self):
        """Test @traced on async function."""
        @traced("async_op")
        async def async_function(x):
            await asyncio.sleep(0.01)
            return x * 2

        with new_trace("test") as ctx:
            result = await async_function(5)
            assert result == 10
            assert len(ctx.spans) == 1
            assert ctx.spans[0].name == "async_op"


class TestEventIntegration:
    """Tests for event trace integration."""

    def test_inject_trace_into_event(self):
        """Test injecting trace_id into event payload."""
        event = MagicMock()
        event.payload = {"data": "value"}

        with new_trace("test") as ctx:
            inject_trace_into_event(event)
            assert event.payload["trace_id"] == ctx.trace_id

    def test_inject_no_trace_active(self):
        """Test inject does nothing when no trace active."""
        _current_trace.set(None)
        event = MagicMock()
        event.payload = {"data": "value"}

        inject_trace_into_event(event)
        assert "trace_id" not in event.payload

    def test_inject_non_dict_payload(self):
        """Test inject handles non-dict payload gracefully."""
        event = MagicMock()
        event.payload = "string payload"

        with new_trace("test"):
            inject_trace_into_event(event)  # Should not raise

    def test_extract_trace_from_event(self):
        """Test extracting trace_id from event payload."""
        event = MagicMock()
        event.payload = {"data": "value", "trace_id": "extracted-trace-123"}

        trace_id = extract_trace_from_event(event)
        assert trace_id == "extracted-trace-123"

    def test_extract_no_trace_id(self):
        """Test extract returns None when no trace_id."""
        event = MagicMock()
        event.payload = {"data": "value"}

        trace_id = extract_trace_from_event(event)
        assert trace_id is None

    def test_extract_non_dict_payload(self):
        """Test extract handles non-dict payload."""
        event = MagicMock()
        event.payload = None

        trace_id = extract_trace_from_event(event)
        assert trace_id is None


class TestHeaderIntegration:
    """Tests for HTTP header trace integration."""

    def test_inject_trace_into_headers(self):
        """Test injecting trace_id into headers."""
        headers = {"Content-Type": "application/json"}

        with new_trace("test") as ctx:
            ctx.start_span("request")
            result = inject_trace_into_headers(headers)

            assert result["X-Trace-Id"] == ctx.trace_id
            assert result["X-Span-Id"] == ctx._current_span.span_id
            assert result["Content-Type"] == "application/json"

    def test_inject_no_trace_active(self):
        """Test inject adds nothing when no trace active."""
        _current_trace.set(None)
        headers = {"Content-Type": "application/json"}

        result = inject_trace_into_headers(headers)
        assert "X-Trace-Id" not in result
        assert result["Content-Type"] == "application/json"

    def test_inject_no_current_span(self):
        """Test inject adds trace_id but not span_id when no span."""
        with new_trace("test") as ctx:
            headers = {}
            result = inject_trace_into_headers(headers)

            assert result["X-Trace-Id"] == ctx.trace_id
            assert "X-Span-Id" not in result

    def test_extract_trace_from_headers(self):
        """Test extracting trace_id from headers."""
        headers = {"X-Trace-Id": "header-trace-456"}

        trace_id = extract_trace_from_headers(headers)
        assert trace_id == "header-trace-456"

    def test_extract_lowercase_header(self):
        """Test extracting from lowercase header."""
        headers = {"x-trace-id": "lowercase-trace"}

        trace_id = extract_trace_from_headers(headers)
        assert trace_id == "lowercase-trace"

    def test_extract_no_header(self):
        """Test extract returns None when header missing."""
        headers = {"Content-Type": "application/json"}

        trace_id = extract_trace_from_headers(headers)
        assert trace_id is None


class TestTraceCollector:
    """Tests for TraceCollector class."""

    def test_initialization(self):
        """Test TraceCollector initialization."""
        collector = TraceCollector(max_traces=500, log_traces=True)
        assert collector.max_traces == 500
        assert collector.log_traces is True
        assert collector._traces == []

    def test_collect_trace(self):
        """Test collecting a trace."""
        collector = TraceCollector()
        ctx = TraceContext.new("test_trace")
        ctx.start_span("span1")
        ctx.end_span()

        collector.collect(ctx)

        assert len(collector._traces) == 1
        assert collector._traces[0]["trace_id"] == ctx.trace_id

    def test_collect_respects_max_traces(self):
        """Test collector respects max_traces limit."""
        collector = TraceCollector(max_traces=3)

        for i in range(5):
            ctx = TraceContext.new(f"trace_{i}")
            collector.collect(ctx)

        assert len(collector._traces) == 3
        # Should keep most recent
        names = [t["name"] for t in collector._traces]
        assert "trace_2" in names
        assert "trace_3" in names
        assert "trace_4" in names

    def test_get_traces(self):
        """Test getting recent traces."""
        collector = TraceCollector()
        for i in range(5):
            ctx = TraceContext.new(f"trace_{i}")
            collector.collect(ctx)

        traces = collector.get_traces(limit=3)
        assert len(traces) == 3
        assert traces[-1]["name"] == "trace_4"

    def test_find_trace(self):
        """Test finding trace by ID."""
        collector = TraceCollector()
        ctx = TraceContext.new("findable")
        collector.collect(ctx)

        found = collector.find_trace(ctx.trace_id)
        assert found is not None
        assert found["name"] == "findable"

    def test_find_trace_not_found(self):
        """Test finding non-existent trace."""
        collector = TraceCollector()
        found = collector.find_trace("non-existent-id")
        assert found is None

    def test_get_slow_traces(self):
        """Test getting traces exceeding threshold."""
        collector = TraceCollector()

        # Fast trace - set duration_ms AFTER end_span (since end_span recalculates it)
        fast = TraceContext.new("fast")
        fast_span = fast.start_span("fast_span")
        fast.end_span()
        fast_span.duration_ms = 50.0  # Override the calculated value
        collector.collect(fast)

        # Slow trace - set duration_ms AFTER end_span
        slow = TraceContext.new("slow")
        slow_span = slow.start_span("slow_span")
        slow.end_span()
        slow_span.duration_ms = 2000.0  # Override the calculated value
        collector.collect(slow)

        slow_traces = collector.get_slow_traces(threshold_ms=1000.0)
        assert len(slow_traces) == 1
        assert slow_traces[0]["name"] == "slow"

    def test_thread_safe_lazy_init(self):
        """Test thread-safe lazy initialization of lock."""
        collector = TraceCollector()
        assert collector._lock is None

        ctx = TraceContext.new("test")
        collector.collect(ctx)

        assert collector._lock is not None

    def test_concurrent_collection(self):
        """Test thread-safe concurrent collection."""
        collector = TraceCollector(max_traces=1000)

        def collect_traces():
            for i in range(100):
                ctx = TraceContext.new(f"trace_{threading.current_thread().name}_{i}")
                collector.collect(ctx)

        threads = [threading.Thread(target=collect_traces) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 500 traces (5 threads * 100 each)
        assert len(collector._traces) == 500


class TestGlobalCollector:
    """Tests for global trace collector functions."""

    def test_get_trace_collector_singleton(self):
        """Test get_trace_collector returns singleton."""
        import app.coordination.tracing as module
        module._collector = None  # Reset singleton

        c1 = get_trace_collector()
        c2 = get_trace_collector()
        assert c1 is c2

    def test_collect_trace_function(self):
        """Test collect_trace convenience function."""
        import app.coordination.tracing as module
        module._collector = None  # Reset singleton

        ctx = TraceContext.new("global_test")
        collect_trace(ctx)

        collector = get_trace_collector()
        assert len(collector._traces) == 1
        assert collector._traces[0]["trace_id"] == ctx.trace_id


class TestContextPropagation:
    """Tests for trace context propagation across threads/tasks."""

    def test_context_isolated_in_threads(self):
        """Test context is isolated between threads."""
        results = {}

        def thread_func(name):
            with new_trace(f"trace_{name}") as ctx:
                time.sleep(0.01)
                results[name] = ctx.trace_id

        t1 = threading.Thread(target=thread_func, args=("t1",))
        t2 = threading.Thread(target=thread_func, args=("t2",))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Each thread should have its own trace_id
        assert results["t1"] != results["t2"]

    @pytest.mark.asyncio
    async def test_context_in_async_tasks(self):
        """Test context works in async tasks."""
        async def async_op():
            with new_trace("async_test") as ctx:
                await asyncio.sleep(0.01)
                return ctx.trace_id

        trace_id = await async_op()
        assert trace_id.startswith("trace-")


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_span_with_none_yielded_on_error(self):
        """Test span handles errors in auto-trace creation gracefully."""
        _current_trace.set(None)

        # Span should still work even without existing trace
        with span("edge_case") as s:
            assert s is not None

    def test_trace_context_with_special_characters(self):
        """Test trace names handle special characters."""
        with new_trace("test/path:special@chars") as ctx:
            assert "test/path:special@chars" in ctx.name

    def test_empty_tags_handling(self):
        """Test handling of empty tags dict."""
        with new_trace("test") as ctx:
            ctx.start_span("span", **{})  # Empty kwargs
            assert ctx._current_span.tags == {}

    def test_deeply_nested_spans(self):
        """Test deeply nested span hierarchy."""
        with new_trace("test") as ctx:
            spans = []
            for i in range(10):
                s = ctx.start_span(f"level_{i}")
                spans.append(s)

            assert len(ctx.spans) == 10
            # Verify parent chain
            for i in range(1, 10):
                assert ctx.spans[i].parent_span_id == ctx.spans[i-1].span_id

            # End all spans
            for _ in range(10):
                ctx.end_span()

            assert ctx._current_span is None
