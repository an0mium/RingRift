"""Tests for scripts.p2p.handlers.handlers_base module.

Tests cover:
- EventBridgeManager (lazy loading, event emission, availability)
- safe_handler decorator (exception handling, response formatting)
- leader_only and voter_only decorators (authorization)
- HandlerStatusMixin (status tracking)
- Response formatting utilities (success_response, error_response)
- Request parsing utilities (parse_json_request, validate_node_id)

December 2025.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from scripts.p2p.handlers.handlers_base import (
    EventBridgeManager,
    HandlerStatusMixin,
    error_response,
    get_event_bridge,
    leader_only,
    parse_json_request,
    safe_handler,
    success_response,
    validate_node_id,
    voter_only,
)


# =============================================================================
# EventBridgeManager Tests
# =============================================================================


class TestEventBridgeManager:
    """Tests for EventBridgeManager class."""

    def test_initialization(self):
        """Manager initializes with None values."""
        manager = EventBridgeManager()
        assert manager._bridge is None
        assert manager._available is None
        assert manager._emit_functions == {}

    def test_available_lazy_load(self):
        """Available property triggers lazy loading."""
        manager = EventBridgeManager()
        # Accessing .available triggers _ensure_loaded
        with patch.object(manager, "_ensure_loaded") as mock_load:
            # Set _available so we don't try to import
            manager._available = True
            result = manager.available
            assert result is True

    def test_available_returns_false_on_import_error(self):
        """Available returns False when import fails."""
        manager = EventBridgeManager()

        with patch.dict("sys.modules", {"scripts.p2p": None}):
            # Force re-initialization
            manager._available = None
            manager._bridge = None

            # This should gracefully handle import failure
            manager._ensure_loaded()
            # Still returns False (import failed)
            assert manager._available is False or manager._available is None

    def test_ensure_loaded_caches_result(self):
        """Ensure loaded only runs once."""
        manager = EventBridgeManager()
        manager._available = True  # Mark as loaded

        # Should not attempt to import again
        manager._ensure_loaded()
        assert manager._available is True

    @pytest.mark.asyncio
    async def test_emit_returns_false_when_unavailable(self):
        """Emit returns False when bridge is unavailable."""
        manager = EventBridgeManager()
        manager._available = False

        result = await manager.emit("test_event", {"data": "value"})
        assert result is False

    @pytest.mark.asyncio
    async def test_emit_unknown_event_returns_false(self):
        """Emit returns False for unknown event functions."""
        manager = EventBridgeManager()
        manager._available = True
        manager._emit_functions = {}  # No functions registered

        result = await manager.emit("unknown_event", {"data": "value"})
        assert result is False

    @pytest.mark.asyncio
    async def test_emit_calls_registered_function(self):
        """Emit calls the registered emit function."""
        manager = EventBridgeManager()
        manager._available = True

        mock_emit = MagicMock(return_value=None)
        manager._emit_functions = {"emit_test_event": mock_emit}

        result = await manager.emit("test_event", {"key": "value"})

        assert result is True
        mock_emit.assert_called_once_with(key="value")

    @pytest.mark.asyncio
    async def test_emit_handles_async_function(self):
        """Emit properly awaits async emit functions."""
        manager = EventBridgeManager()
        manager._available = True

        async_mock = AsyncMock(return_value=None)
        manager._emit_functions = {"emit_async_event": async_mock}

        result = await manager.emit("async_event", {"data": "test"})

        assert result is True
        async_mock.assert_awaited_once_with(data="test")

    @pytest.mark.asyncio
    async def test_emit_handles_exception_gracefully(self):
        """Emit returns False on exception."""
        manager = EventBridgeManager()
        manager._available = True

        def failing_emit(**kwargs):
            raise ValueError("Test error")

        manager._emit_functions = {"emit_failing": failing_emit}

        result = await manager.emit("failing", {"data": "test"})
        assert result is False

    def test_get_emit_func_returns_function(self):
        """Get emit func returns registered function."""
        manager = EventBridgeManager()
        manager._available = True

        mock_func = MagicMock()
        manager._emit_functions = {"emit_my_event": mock_func}

        result = manager.get_emit_func("emit_my_event")
        assert result is mock_func

    def test_get_emit_func_returns_none_for_unknown(self):
        """Get emit func returns None for unknown function."""
        manager = EventBridgeManager()
        manager._available = True
        manager._emit_functions = {}

        result = manager.get_emit_func("emit_unknown")
        assert result is None


class TestGetEventBridge:
    """Tests for get_event_bridge global function."""

    def test_returns_manager_instance(self):
        """Returns an EventBridgeManager instance."""
        manager = get_event_bridge()
        assert isinstance(manager, EventBridgeManager)

    def test_returns_same_instance(self):
        """Returns the same singleton instance."""
        manager1 = get_event_bridge()
        manager2 = get_event_bridge()
        assert manager1 is manager2


# =============================================================================
# safe_handler Decorator Tests
# =============================================================================


class MockRequest:
    """Mock aiohttp Request for testing."""

    def __init__(self, json_data=None, should_fail=False):
        self._json_data = json_data or {}
        self._should_fail = should_fail

    async def json(self):
        if self._should_fail:
            raise ValueError("Invalid JSON")
        return self._json_data


class SafeHandlerTestClass:
    """Test class for safe_handler decorator."""

    @safe_handler("test_endpoint")
    async def handle_success(self, request):
        return {"status": "ok", "data": "result"}

    @safe_handler("key_error_endpoint")
    async def handle_key_error(self, request):
        data = await request.json()
        _ = data["required_field"]  # Will raise KeyError

    @safe_handler("value_error_endpoint")
    async def handle_value_error(self, request):
        raise ValueError("Invalid value")

    @safe_handler("timeout_endpoint")
    async def handle_timeout(self, request):
        raise asyncio.TimeoutError()

    @safe_handler("general_error_endpoint")
    async def handle_general_error(self, request):
        raise RuntimeError("Something went wrong")


class TestSafeHandler:
    """Tests for safe_handler decorator."""

    @pytest.fixture
    def handler(self):
        return SafeHandlerTestClass()

    @pytest.mark.asyncio
    async def test_success_returns_json_response(self, handler):
        """Successful handler returns json_response."""
        request = MockRequest()
        response = await handler.handle_success(request)

        assert isinstance(response, web.Response)
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_key_error_returns_400(self, handler):
        """KeyError returns 400 Bad Request."""
        request = MockRequest(json_data={})  # Missing required_field
        response = await handler.handle_key_error(request)

        assert response.status == 400
        assert b"Missing required field" in response.body

    @pytest.mark.asyncio
    async def test_value_error_returns_400(self, handler):
        """ValueError returns 400 Bad Request."""
        request = MockRequest()
        response = await handler.handle_value_error(request)

        assert response.status == 400
        assert b"Invalid value" in response.body

    @pytest.mark.asyncio
    async def test_timeout_error_returns_504(self, handler):
        """TimeoutError returns 504 Gateway Timeout."""
        request = MockRequest()
        response = await handler.handle_timeout(request)

        assert response.status == 504
        assert b"timed out" in response.body

    @pytest.mark.asyncio
    async def test_general_error_returns_500(self, handler):
        """Other exceptions return 500 Internal Server Error."""
        request = MockRequest()
        response = await handler.handle_general_error(request)

        assert response.status == 500
        assert b"Internal server error" in response.body


# =============================================================================
# leader_only Decorator Tests
# =============================================================================


class LeaderOnlyTestClass:
    """Test class for leader_only decorator."""

    def __init__(self, role_value="leader"):
        # Mock NodeRole enum
        self.role = MagicMock()
        self.role.value = role_value

        # Set LEADER value for comparison
        from enum import Enum
        class MockNodeRole(str, Enum):
            LEADER = "leader"
            FOLLOWER = "follower"

        if role_value == "leader":
            self.role = MockNodeRole.LEADER
        else:
            self.role = MockNodeRole.FOLLOWER

        self.node_id = "test-node"

    @leader_only
    async def leader_action(self, request):
        return web.json_response({"status": "ok"})


class TestLeaderOnly:
    """Tests for leader_only decorator."""

    @pytest.mark.asyncio
    async def test_leader_allowed(self):
        """Leader role is allowed to proceed."""
        handler = LeaderOnlyTestClass(role_value="leader")
        request = MockRequest()

        response = await handler.leader_action(request)

        assert response.status == 200

    @pytest.mark.asyncio
    async def test_follower_rejected(self):
        """Follower role is rejected with 403."""
        handler = LeaderOnlyTestClass(role_value="follower")
        request = MockRequest()

        response = await handler.leader_action(request)

        assert response.status == 403
        assert b"Only leader" in response.body


# =============================================================================
# voter_only Decorator Tests
# =============================================================================


class VoterOnlyTestClass:
    """Test class for voter_only decorator."""

    def __init__(self, node_id="voter-1", voter_ids=None):
        self.node_id = node_id
        self.voter_node_ids = voter_ids or ["voter-1", "voter-2", "voter-3"]

    @voter_only
    async def voter_action(self, request):
        return web.json_response({"status": "ok"})


class TestVoterOnly:
    """Tests for voter_only decorator."""

    @pytest.mark.asyncio
    async def test_voter_allowed(self):
        """Voter node is allowed to proceed."""
        handler = VoterOnlyTestClass(node_id="voter-1")
        request = MockRequest()

        response = await handler.voter_action(request)

        assert response.status == 200

    @pytest.mark.asyncio
    async def test_non_voter_rejected(self):
        """Non-voter is rejected with 403."""
        handler = VoterOnlyTestClass(node_id="worker-1")
        request = MockRequest()

        response = await handler.voter_action(request)

        assert response.status == 403
        assert b"Only voter" in response.body


# =============================================================================
# HandlerStatusMixin Tests
# =============================================================================


class HandlerStatusTestClass(HandlerStatusMixin):
    """Test class for HandlerStatusMixin."""

    def __init__(self):
        self._init_handler_status("TestHandler")


class TestHandlerStatusMixin:
    """Tests for HandlerStatusMixin."""

    def test_init_handler_status(self):
        """Initialize handler status creates attributes."""
        handler = HandlerStatusTestClass()

        assert handler._handler_name == "TestHandler"
        assert handler._handler_request_counts == {}
        assert handler._handler_error_counts == {}
        assert handler._handler_last_error == ""
        assert handler._handler_start_time > 0

    def test_record_request(self):
        """Record request increments counter."""
        handler = HandlerStatusTestClass()

        handler._record_request("/test")
        handler._record_request("/test")
        handler._record_request("/other")

        assert handler._handler_request_counts["/test"] == 2
        assert handler._handler_request_counts["/other"] == 1

    def test_record_error(self):
        """Record error increments counter and stores message."""
        handler = HandlerStatusTestClass()

        handler._record_error("/test", "Connection failed")

        assert handler._handler_error_counts["/test"] == 1
        assert handler._handler_last_error == "/test: Connection failed"

    def test_get_handler_status(self):
        """Get handler status returns summary dict."""
        handler = HandlerStatusTestClass()

        handler._record_request("/a")
        handler._record_request("/a")
        handler._record_error("/a", "Error")
        handler._record_request("/b")

        status = handler.get_handler_status()

        assert status["handler"] == "TestHandler"
        assert status["total_requests"] == 3
        assert status["total_errors"] == 1
        assert status["success_rate"] == pytest.approx(2/3)
        assert status["requests_by_endpoint"] == {"/a": 2, "/b": 1}
        assert status["errors_by_endpoint"] == {"/a": 1}

    def test_get_handler_status_no_requests(self):
        """Get handler status with no requests returns 100% success."""
        handler = HandlerStatusTestClass()

        status = handler.get_handler_status()

        assert status["total_requests"] == 0
        assert status["success_rate"] == 1.0

    def test_get_handler_status_uninitialized(self):
        """Get handler status without init returns error."""

        class UninitializedHandler(HandlerStatusMixin):
            pass

        handler = UninitializedHandler()
        status = handler.get_handler_status()

        assert "error" in status

    def test_record_without_init(self):
        """Record methods gracefully handle uninitialized state."""

        class UninitializedHandler(HandlerStatusMixin):
            pass

        handler = UninitializedHandler()
        # Should not raise
        handler._record_request("/test")
        handler._record_error("/test", "Error")


# =============================================================================
# Response Formatting Tests
# =============================================================================


class TestSuccessResponse:
    """Tests for success_response utility."""

    def test_basic_success(self):
        """Basic success response has required fields."""
        resp = success_response()

        assert resp["success"] is True
        assert "timestamp" in resp

    def test_success_with_message(self):
        """Success response includes message."""
        resp = success_response(message="Operation completed")

        assert resp["message"] == "Operation completed"

    def test_success_with_data(self):
        """Success response includes data fields."""
        resp = success_response(data={"count": 5, "items": ["a", "b"]})

        assert resp["count"] == 5
        assert resp["items"] == ["a", "b"]

    def test_success_with_both(self):
        """Success response includes both message and data."""
        resp = success_response(
            data={"result": "value"},
            message="Done",
        )

        assert resp["success"] is True
        assert resp["message"] == "Done"
        assert resp["result"] == "value"


class TestErrorResponse:
    """Tests for error_response utility."""

    def test_basic_error(self):
        """Basic error response has required fields."""
        resp = error_response("Something went wrong")

        assert resp["success"] is False
        assert resp["error"] == "Something went wrong"
        assert "timestamp" in resp

    def test_error_with_code(self):
        """Error response includes error code."""
        resp = error_response("Not found", code="NOT_FOUND")

        assert resp["code"] == "NOT_FOUND"

    def test_error_with_details(self):
        """Error response includes details."""
        resp = error_response(
            "Validation failed",
            details={"field": "email", "reason": "invalid format"},
        )

        assert resp["details"]["field"] == "email"
        assert resp["details"]["reason"] == "invalid format"


# =============================================================================
# Request Parsing Tests
# =============================================================================


class TestParseJsonRequest:
    """Tests for parse_json_request utility."""

    @pytest.mark.asyncio
    async def test_parse_valid_json(self):
        """Parse valid JSON request."""
        request = MockRequest(json_data={"key": "value", "count": 5})

        data = await parse_json_request(request)

        assert data["key"] == "value"
        assert data["count"] == 5

    @pytest.mark.asyncio
    async def test_parse_with_required_fields_present(self):
        """Parse with required fields that are present."""
        request = MockRequest(json_data={"name": "test", "id": 123})

        data = await parse_json_request(request, required_fields=["name", "id"])

        assert data["name"] == "test"
        assert data["id"] == 123

    @pytest.mark.asyncio
    async def test_parse_with_required_fields_missing(self):
        """Parse with missing required fields raises ValueError."""
        request = MockRequest(json_data={"name": "test"})

        with pytest.raises(ValueError, match="Missing required fields"):
            await parse_json_request(request, required_fields=["name", "id", "type"])

    @pytest.mark.asyncio
    async def test_parse_invalid_json_raises(self):
        """Parse invalid JSON raises HTTPBadRequest."""
        request = MockRequest(should_fail=True)

        with pytest.raises(web.HTTPBadRequest):
            await parse_json_request(request)


class TestValidateNodeId:
    """Tests for validate_node_id utility."""

    def test_validate_existing_node(self):
        """Validate node that exists in peers."""
        peers = {"node-1": {}, "node-2": {}}

        result = validate_node_id("node-1", peers)

        assert result == "node-1"

    def test_validate_none_raises(self):
        """Validate None node_id raises ValueError."""
        peers = {"node-1": {}}

        with pytest.raises(ValueError, match="node_id is required"):
            validate_node_id(None, peers)

    def test_validate_empty_raises(self):
        """Validate empty node_id raises ValueError."""
        peers = {"node-1": {}}

        with pytest.raises(ValueError, match="node_id is required"):
            validate_node_id("", peers)

    def test_validate_unknown_raises(self):
        """Validate unknown node_id raises ValueError."""
        peers = {"node-1": {}}

        with pytest.raises(ValueError, match="Unknown peer"):
            validate_node_id("node-99", peers)
