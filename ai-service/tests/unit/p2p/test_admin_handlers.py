"""Tests for scripts.p2p.handlers.admin module.

Tests cover:
- AdminHandlersMixin git status endpoint
- Git update endpoint with auth token validation
- Admin restart endpoint with auth token validation
- Error handling and edge cases

December 2025.
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from scripts.p2p.handlers.admin import AdminHandlersMixin


# =============================================================================
# Test Class Implementation
# =============================================================================


class AdminHandlersTestClass(AdminHandlersMixin):
    """Test class that uses the admin handlers mixin."""

    def __init__(
        self,
        node_id: str = "test-node",
        local_commit: str = "abc123def456",
        local_branch: str = "main",
        has_local_changes: bool = False,
        has_updates: bool = False,
        remote_commit: str = "def456abc123",
        commits_behind: int = 0,
    ):
        self.node_id = node_id
        self.auth_token = None
        self.ringrift_path = "/home/user/ringrift"

        # Git status mocking
        self._local_commit = local_commit
        self._local_branch = local_branch
        self._has_local_changes = has_local_changes
        self._has_updates = has_updates
        self._remote_commit = remote_commit
        self._commits_behind = commits_behind

        # Update mocking
        self._update_success = True
        self._update_message = "Update completed"
        self._restart_called = False

    def _get_local_git_commit(self) -> str:
        return self._local_commit

    def _get_local_git_branch(self) -> str:
        return self._local_branch

    def _check_local_changes(self) -> bool:
        return self._has_local_changes

    def _check_for_updates(self) -> tuple[bool, str, str]:
        return (self._has_updates, self._local_commit, self._remote_commit)

    def _get_commits_behind(self, local: str, remote: str) -> int:
        return self._commits_behind

    async def _perform_git_update(self) -> tuple[bool, str]:
        return (self._update_success, self._update_message)

    async def _restart_orchestrator(self) -> None:
        self._restart_called = True
        await asyncio.sleep(0.01)  # Simulate brief delay


class MockRequest:
    """Mock aiohttp Request for testing."""

    def __init__(self, headers: dict | None = None, json_data: dict | None = None):
        self.headers = headers or {}
        self._json_data = json_data or {}

    async def json(self):
        return self._json_data


# =============================================================================
# Git Status Endpoint Tests
# =============================================================================


class TestHandleGitStatus:
    """Tests for handle_git_status endpoint."""

    @pytest.fixture
    def handler(self):
        return AdminHandlersTestClass()

    @pytest.mark.asyncio
    async def test_returns_local_commit(self, handler):
        """Response includes local commit info."""
        request = MockRequest()

        response = await handler.handle_git_status(request)

        body = json.loads(response.body)
        assert body["local_commit"] == "abc123de"  # First 8 chars
        assert body["local_commit_full"] == "abc123def456"

    @pytest.mark.asyncio
    async def test_returns_local_branch(self, handler):
        """Response includes local branch."""
        handler._local_branch = "feature-branch"
        request = MockRequest()

        response = await handler.handle_git_status(request)

        body = json.loads(response.body)
        assert body["local_branch"] == "feature-branch"

    @pytest.mark.asyncio
    async def test_returns_has_local_changes(self, handler):
        """Response includes local changes flag."""
        handler._has_local_changes = True
        request = MockRequest()

        response = await handler.handle_git_status(request)

        body = json.loads(response.body)
        assert body["has_local_changes"] is True

    @pytest.mark.asyncio
    async def test_returns_update_info(self, handler):
        """Response includes update availability."""
        handler._has_updates = True
        handler._commits_behind = 5
        request = MockRequest()

        response = await handler.handle_git_status(request)

        body = json.loads(response.body)
        assert body["has_updates"] is True
        assert body["commits_behind"] == 5

    @pytest.mark.asyncio
    async def test_returns_remote_commit(self, handler):
        """Response includes remote commit when updates available."""
        handler._has_updates = True
        handler._remote_commit = "xyz789abc"
        request = MockRequest()

        response = await handler.handle_git_status(request)

        body = json.loads(response.body)
        assert body["remote_commit"] == "xyz789ab"  # First 8 chars
        assert body["remote_commit_full"] == "xyz789abc"

    @pytest.mark.asyncio
    async def test_returns_ringrift_path(self, handler):
        """Response includes ringrift path."""
        request = MockRequest()

        response = await handler.handle_git_status(request)

        body = json.loads(response.body)
        assert body["ringrift_path"] == "/home/user/ringrift"

    @pytest.mark.asyncio
    async def test_handles_none_commits(self):
        """Handles None commits gracefully."""
        handler = AdminHandlersTestClass(local_commit="", remote_commit="")
        request = MockRequest()

        response = await handler.handle_git_status(request)

        body = json.loads(response.body)
        # Empty string now returns None (handler converts empty to None)
        assert body["local_commit"] is None or body["local_commit"] == ""

    @pytest.mark.asyncio
    async def test_error_returns_500(self, handler):
        """Exception returns 500 error."""
        handler._get_local_git_commit = MagicMock(side_effect=Exception("Git error"))
        request = MockRequest()

        response = await handler.handle_git_status(request)

        assert response.status == 500
        body = json.loads(response.body)
        assert "Git error" in body["error"]


# =============================================================================
# Git Update Endpoint Tests
# =============================================================================


class TestHandleGitUpdate:
    """Tests for handle_git_update endpoint."""

    @pytest.fixture
    def handler(self):
        handler = AdminHandlersTestClass()
        handler._has_updates = True
        return handler

    @pytest.mark.asyncio
    async def test_no_updates_returns_already_up_to_date(self, handler):
        """No updates available returns success with message."""
        handler._has_updates = False
        request = MockRequest()

        response = await handler.handle_git_update(request)

        body = json.loads(response.body)
        assert body["success"] is True
        assert "up to date" in body["message"].lower()

    @pytest.mark.asyncio
    async def test_successful_update(self, handler):
        """Successful update returns success and schedules restart."""
        request = MockRequest()

        response = await handler.handle_git_update(request)

        # Give async task time to run
        await asyncio.sleep(0.05)

        body = json.loads(response.body)
        assert body["success"] is True
        assert "restarting" in body["message"].lower()
        assert handler._restart_called is True

    @pytest.mark.asyncio
    async def test_failed_update(self, handler):
        """Failed update returns error message."""
        handler._update_success = False
        handler._update_message = "Merge conflict"
        request = MockRequest()

        response = await handler.handle_git_update(request)

        assert response.status == 400
        body = json.loads(response.body)
        assert body["success"] is False
        assert body["message"] == "Merge conflict"

    @pytest.mark.asyncio
    async def test_auth_token_required(self, handler):
        """Auth token required when RINGRIFT_ADMIN_TOKEN set."""
        with patch.dict(os.environ, {"RINGRIFT_ADMIN_TOKEN": "secret123"}):
            request = MockRequest(headers={})

            response = await handler.handle_git_update(request)

            assert response.status == 401

    @pytest.mark.asyncio
    async def test_auth_token_valid(self, handler):
        """Valid auth token allows update."""
        with patch.dict(os.environ, {"RINGRIFT_ADMIN_TOKEN": "secret123"}):
            request = MockRequest(headers={"X-Admin-Token": "secret123"})

            response = await handler.handle_git_update(request)

            # Should proceed to update check
            body = json.loads(response.body)
            assert body["success"] is True

    @pytest.mark.asyncio
    async def test_auth_token_invalid(self, handler):
        """Invalid auth token returns 401."""
        with patch.dict(os.environ, {"RINGRIFT_ADMIN_TOKEN": "secret123"}):
            request = MockRequest(headers={"X-Admin-Token": "wrongtoken"})

            response = await handler.handle_git_update(request)

            assert response.status == 401

    @pytest.mark.asyncio
    async def test_no_auth_when_not_configured(self, handler):
        """No auth required when RINGRIFT_ADMIN_TOKEN not set."""
        # Ensure env var is not set
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RINGRIFT_ADMIN_TOKEN", None)
            request = MockRequest(headers={})

            response = await handler.handle_git_update(request)

            # Should proceed to update check
            body = json.loads(response.body)
            assert body["success"] is True

    @pytest.mark.asyncio
    async def test_includes_commit_info(self, handler):
        """Response includes old and new commit info."""
        handler._local_commit = "oldcommit123"
        handler._remote_commit = "newcommit456"
        request = MockRequest()

        response = await handler.handle_git_update(request)

        body = json.loads(response.body)
        assert body["old_commit"] == "oldcommi"  # First 8 chars
        assert body["new_commit"] == "newcommi"

    @pytest.mark.asyncio
    async def test_exception_returns_500(self, handler):
        """Exception during update returns 500."""
        handler._check_for_updates = MagicMock(side_effect=Exception("Network error"))
        request = MockRequest()

        response = await handler.handle_git_update(request)

        assert response.status == 500
        body = json.loads(response.body)
        assert "Network error" in body["error"]


# =============================================================================
# Admin Restart Endpoint Tests
# =============================================================================


class TestHandleAdminRestart:
    """Tests for handle_admin_restart endpoint."""

    @pytest.fixture
    def handler(self):
        return AdminHandlersTestClass()

    @pytest.mark.asyncio
    async def test_restart_scheduled(self, handler):
        """Restart is scheduled on request."""
        request = MockRequest()

        response = await handler.handle_admin_restart(request)

        # Give async task time to run
        await asyncio.sleep(0.05)

        body = json.loads(response.body)
        assert body["success"] is True
        assert "restart" in body["message"].lower()
        assert handler._restart_called is True

    @pytest.mark.asyncio
    async def test_auth_token_required(self, handler):
        """Auth token required when RINGRIFT_ADMIN_TOKEN set."""
        with patch.dict(os.environ, {"RINGRIFT_ADMIN_TOKEN": "secret"}):
            request = MockRequest(headers={})

            response = await handler.handle_admin_restart(request)

            assert response.status == 401
            assert handler._restart_called is False

    @pytest.mark.asyncio
    async def test_auth_token_valid(self, handler):
        """Valid auth token allows restart."""
        with patch.dict(os.environ, {"RINGRIFT_ADMIN_TOKEN": "secret"}):
            request = MockRequest(headers={"X-Admin-Token": "secret"})

            response = await handler.handle_admin_restart(request)

            await asyncio.sleep(0.05)

            body = json.loads(response.body)
            assert body["success"] is True
            assert handler._restart_called is True

    @pytest.mark.asyncio
    async def test_no_auth_when_not_configured(self, handler):
        """No auth required when RINGRIFT_ADMIN_TOKEN not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RINGRIFT_ADMIN_TOKEN", None)
            request = MockRequest()

            response = await handler.handle_admin_restart(request)

            await asyncio.sleep(0.05)

            body = json.loads(response.body)
            assert body["success"] is True
            assert handler._restart_called is True

    @pytest.mark.asyncio
    async def test_restart_is_fire_and_forget(self, handler):
        """Restart is fire-and-forget, returns 200 immediately.

        Note: Restart is scheduled as a background task, so exceptions
        don't affect the response. The endpoint returns success before
        the restart actually executes.
        """
        restart_started = False

        async def slow_restart():
            nonlocal restart_started
            restart_started = True
            await asyncio.sleep(1.0)  # Simulate slow restart

        handler._restart_orchestrator = slow_restart
        request = MockRequest()

        response = await handler.handle_admin_restart(request)

        # Response returns immediately (fire-and-forget)
        assert response.status == 200
        body = json.loads(response.body)
        assert body["success"] is True
        # Restart may or may not have started yet (it's async)
        assert "restart" in body["message"].lower()


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestAdminHandlersEdgeCases:
    """Edge case tests for admin handlers."""

    @pytest.mark.asyncio
    async def test_commits_behind_not_calculated_without_updates(self):
        """Commits behind is 0 when no updates available."""
        handler = AdminHandlersTestClass(has_updates=False, commits_behind=10)
        request = MockRequest()

        response = await handler.handle_git_status(request)

        body = json.loads(response.body)
        # Should be 0 because has_updates is False
        assert body["commits_behind"] == 0

    @pytest.mark.asyncio
    async def test_empty_commit_handling(self):
        """Empty commits handled gracefully."""
        handler = AdminHandlersTestClass(
            local_commit="",
            remote_commit="",
            has_updates=True,
        )
        request = MockRequest()

        response = await handler.handle_git_status(request)

        # Should not error
        assert response.status == 200
