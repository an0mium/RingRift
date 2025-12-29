"""Tests for scripts.p2p.handlers.base module.

Tests cover:
- BaseP2PHandler (json_response, error_response, auth, body parsing)
- Module-level utilities (make_json_response, make_error_response)
- Gzip body parsing with magic byte detection

December 2025.
"""

from __future__ import annotations

import gzip
import json
import time
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from aiohttp import web

from scripts.p2p.handlers.base import (
    BaseP2PHandler,
    make_error_response,
    make_json_response,
)


# =============================================================================
# Mock Classes
# =============================================================================


class ConcreteHandler(BaseP2PHandler):
    """Concrete implementation of BaseP2PHandler for testing."""

    def __init__(
        self,
        node_id: str = "test-node",
        auth_token: str | None = None,
    ):
        self.node_id = node_id
        self.auth_token = auth_token

    def _is_request_authorized(self, request: web.Request) -> bool:
        """Default authorization check."""
        if not self.auth_token:
            return True

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:] == self.auth_token

        return request.headers.get("X-Auth-Token") == self.auth_token


class MockRequest:
    """Mock aiohttp Request for testing."""

    def __init__(
        self,
        headers: dict | None = None,
        json_data: dict | None = None,
        raw_body: bytes | None = None,
        transport=None,
    ):
        self.headers = headers or {}
        self._json_data = json_data or {}
        self._raw_body = raw_body
        self.transport = transport
        self.method = "GET"
        self.path = "/test"
        self.remote = "127.0.0.1"

    async def json(self):
        if self._json_data is None:
            raise json.JSONDecodeError("Invalid", "", 0)
        return self._json_data

    async def read(self):
        if self._raw_body is not None:
            return self._raw_body
        return json.dumps(self._json_data).encode()


class MockTransport:
    """Mock transport with peername."""

    def __init__(self, ip: str = "192.168.1.100", port: int = 12345):
        self._peername = (ip, port)

    def get_extra_info(self, key: str):
        if key == "peername":
            return self._peername
        return None


# =============================================================================
# BaseP2PHandler Response Tests
# =============================================================================


class TestJsonResponse:
    """Tests for json_response method."""

    @pytest.fixture
    def handler(self):
        return ConcreteHandler(node_id="my-node")

    def test_basic_response(self, handler):
        """Basic json response has correct content type."""
        response = handler.json_response({"status": "ok"})

        assert response.status == 200
        assert response.content_type == "application/json"

    def test_response_includes_node_id_header(self, handler):
        """Response includes X-Node-ID header."""
        response = handler.json_response({"data": "value"})

        assert response.headers.get("X-Node-ID") == "my-node"

    def test_custom_status_code(self, handler):
        """Response can have custom status code."""
        response = handler.json_response({"error": "bad"}, status=400)

        assert response.status == 400

    def test_custom_headers(self, handler):
        """Response can have custom headers."""
        response = handler.json_response(
            {"data": "value"},
            headers={"X-Custom": "header-value"},
        )

        assert response.headers.get("X-Custom") == "header-value"
        assert response.headers.get("X-Node-ID") == "my-node"

    def test_response_body_is_json(self, handler):
        """Response body is valid JSON."""
        response = handler.json_response({"key": "value", "count": 42})

        body = json.loads(response.body)
        assert body["key"] == "value"
        assert body["count"] == 42


class TestErrorResponse:
    """Tests for error_response method."""

    @pytest.fixture
    def handler(self):
        return ConcreteHandler()

    def test_basic_error(self, handler):
        """Basic error response has message."""
        response = handler.error_response("Something went wrong")

        body = json.loads(response.body)
        assert body["error"] == "Something went wrong"
        assert body["status"] == 500

    def test_custom_status(self, handler):
        """Error response with custom status."""
        response = handler.error_response("Not found", status=404)

        assert response.status == 404
        body = json.loads(response.body)
        assert body["status"] == 404

    def test_with_error_code(self, handler):
        """Error response with error code."""
        response = handler.error_response(
            "Validation failed",
            status=400,
            error_code="VALIDATION_ERROR",
        )

        body = json.loads(response.body)
        assert body["code"] == "VALIDATION_ERROR"

    def test_with_details(self, handler):
        """Error response with details."""
        response = handler.error_response(
            "Partial failure",
            status=500,
            details={"failed_nodes": ["a", "b"], "success_count": 5},
        )

        body = json.loads(response.body)
        assert body["details"]["failed_nodes"] == ["a", "b"]
        # Details are also projected to top level
        assert body["failed_nodes"] == ["a", "b"]
        assert body["success_count"] == 5

    def test_includes_timestamp(self, handler):
        """Error response includes timestamp."""
        before = time.time()
        response = handler.error_response("Error")
        after = time.time()

        body = json.loads(response.body)
        assert before <= body["timestamp"] <= after


class TestNotFound:
    """Tests for not_found method."""

    @pytest.fixture
    def handler(self):
        return ConcreteHandler()

    def test_default_message(self, handler):
        """Not found with default message."""
        response = handler.not_found()

        assert response.status == 404
        body = json.loads(response.body)
        assert "Resource not found" in body["error"]

    def test_custom_resource(self, handler):
        """Not found with custom resource name."""
        response = handler.not_found("Model")

        body = json.loads(response.body)
        assert "Model not found" in body["error"]


class TestBadRequest:
    """Tests for bad_request method."""

    @pytest.fixture
    def handler(self):
        return ConcreteHandler()

    def test_default_message(self, handler):
        """Bad request with default message."""
        response = handler.bad_request()

        assert response.status == 400
        body = json.loads(response.body)
        assert "Bad request" in body["error"]

    def test_custom_message(self, handler):
        """Bad request with custom message."""
        response = handler.bad_request("Invalid board_type")

        body = json.loads(response.body)
        assert body["error"] == "Invalid board_type"


# =============================================================================
# Authentication Tests
# =============================================================================


class TestCheckAuth:
    """Tests for check_auth method."""

    def test_no_auth_configured_allows_all(self):
        """No auth token configured allows all requests."""
        handler = ConcreteHandler(auth_token=None)
        request = MockRequest()

        result = handler.check_auth(request)

        assert result is True

    def test_bearer_token_valid(self):
        """Valid bearer token is accepted."""
        handler = ConcreteHandler(auth_token="secret123")
        request = MockRequest(headers={"Authorization": "Bearer secret123"})

        result = handler.check_auth(request)

        assert result is True

    def test_bearer_token_invalid(self):
        """Invalid bearer token is rejected."""
        handler = ConcreteHandler(auth_token="secret123")
        request = MockRequest(headers={"Authorization": "Bearer wrongtoken"})

        result = handler.check_auth(request)

        assert result is False

    def test_x_auth_token_header_valid(self):
        """Valid X-Auth-Token header is accepted."""
        handler = ConcreteHandler(auth_token="secret123")
        request = MockRequest(headers={"X-Auth-Token": "secret123"})

        result = handler.check_auth(request)

        assert result is True

    def test_no_token_provided_rejected(self):
        """Missing token when required is rejected."""
        handler = ConcreteHandler(auth_token="secret123")
        request = MockRequest(headers={})

        result = handler.check_auth(request)

        assert result is False


class TestAuthError:
    """Tests for auth_error method."""

    def test_returns_401(self):
        """Auth error returns 401 status."""
        handler = ConcreteHandler()
        response = handler.auth_error()

        assert response.status == 401

    def test_error_message(self):
        """Auth error has unauthorized message."""
        handler = ConcreteHandler()
        response = handler.auth_error()

        body = json.loads(response.body)
        assert body["error"] == "unauthorized"


# =============================================================================
# Body Parsing Tests
# =============================================================================


class TestParseJsonBody:
    """Tests for parse_json_body method."""

    @pytest.fixture
    def handler(self):
        return ConcreteHandler()

    @pytest.mark.asyncio
    async def test_parse_valid_json(self, handler):
        """Parse valid JSON body."""
        request = MockRequest(json_data={"key": "value", "num": 42})

        result = await handler.parse_json_body(request)

        assert result == {"key": "value", "num": 42}

    @pytest.mark.asyncio
    async def test_parse_invalid_json_returns_none(self, handler):
        """Parse invalid JSON returns None."""
        request = MockRequest()
        request._json_data = None  # Will cause JSONDecodeError

        result = await handler.parse_json_body(request)

        assert result is None


class TestParseGzipBody:
    """Tests for parse_gzip_body method."""

    @pytest.fixture
    def handler(self):
        return ConcreteHandler()

    @pytest.mark.asyncio
    async def test_parse_actual_gzip(self, handler):
        """Parse actually gzipped content."""
        data = {"key": "value", "compressed": True}
        json_bytes = json.dumps(data).encode()
        gzipped = gzip.compress(json_bytes)

        request = MockRequest(raw_body=gzipped)

        result = await handler.parse_gzip_body(request)

        assert result == data

    @pytest.mark.asyncio
    async def test_parse_raw_json_despite_gzip_header(self, handler):
        """Parse raw JSON even when client claims gzip."""
        data = {"key": "value"}
        raw_json = json.dumps(data).encode()

        request = MockRequest(raw_body=raw_json)

        result = await handler.parse_gzip_body(request)

        assert result == data

    @pytest.mark.asyncio
    async def test_parse_invalid_gzip_returns_none(self, handler):
        """Invalid gzip that looks like gzip returns None."""
        # Magic bytes but invalid content
        invalid = b"\x1f\x8b\x00\x00invalid"

        request = MockRequest(raw_body=invalid)

        result = await handler.parse_gzip_body(request)

        assert result is None

    @pytest.mark.asyncio
    async def test_empty_body_returns_none(self, handler):
        """Empty body returns None."""
        request = MockRequest(raw_body=b"")

        result = await handler.parse_gzip_body(request)

        # Will try to parse empty bytes as JSON and fail
        assert result is None


# =============================================================================
# Utility Methods Tests
# =============================================================================


class TestGetClientIp:
    """Tests for get_client_ip method."""

    @pytest.fixture
    def handler(self):
        return ConcreteHandler()

    def test_forwarded_for_header(self, handler):
        """Get IP from X-Forwarded-For header."""
        request = MockRequest(headers={"X-Forwarded-For": "10.0.0.1, 192.168.1.1"})

        result = handler.get_client_ip(request)

        assert result == "10.0.0.1"

    def test_transport_peername(self, handler):
        """Get IP from transport peername."""
        transport = MockTransport(ip="192.168.1.50")
        request = MockRequest(transport=transport)

        result = handler.get_client_ip(request)

        assert result == "192.168.1.50"

    def test_no_ip_available(self, handler):
        """Return 'unknown' when no IP available."""
        request = MockRequest()
        request.transport = None

        result = handler.get_client_ip(request)

        assert result == "unknown"


class TestLogRequest:
    """Tests for log_request method."""

    @pytest.fixture
    def handler(self):
        return ConcreteHandler(node_id="log-test")

    def test_log_request_no_error(self, handler):
        """Log request doesn't raise."""
        request = MockRequest(
            headers={"X-Forwarded-For": "10.0.0.1"},
        )
        request.method = "POST"
        request.path = "/api/test"

        # Should not raise
        handler.log_request(request, "Test message")


# =============================================================================
# Module-Level Function Tests
# =============================================================================


class TestMakeJsonResponse:
    """Tests for make_json_response standalone function."""

    def test_basic_response(self):
        """Create basic json response."""
        response = make_json_response({"key": "value"})

        assert response.status == 200
        body = json.loads(response.body)
        assert body["key"] == "value"

    def test_with_status(self):
        """Create response with custom status."""
        response = make_json_response({"error": True}, status=400)

        assert response.status == 400

    def test_with_node_id(self):
        """Create response with node ID header."""
        response = make_json_response({"data": 1}, node_id="custom-node")

        assert response.headers.get("X-Node-ID") == "custom-node"


class TestMakeErrorResponse:
    """Tests for make_error_response standalone function."""

    def test_basic_error(self):
        """Create basic error response."""
        response = make_error_response("Something failed")

        assert response.status == 500
        body = json.loads(response.body)
        assert body["error"] == "Something failed"

    def test_with_status(self):
        """Create error with custom status."""
        response = make_error_response("Not found", status=404)

        assert response.status == 404

    def test_with_node_id(self):
        """Create error with node ID header."""
        response = make_error_response("Error", node_id="error-node")

        assert response.headers.get("X-Node-ID") == "error-node"

    def test_includes_timestamp(self):
        """Error response includes timestamp."""
        before = time.time()
        response = make_error_response("Error")
        after = time.time()

        body = json.loads(response.body)
        assert before <= body["timestamp"] <= after
