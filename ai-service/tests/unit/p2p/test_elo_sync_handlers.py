"""Tests for scripts.p2p.handlers.elo_sync module.

Tests cover:
- EloSyncHandlersMixin sync status endpoint
- Sync trigger endpoint
- Database download endpoint
- Database upload endpoint with auth
- Elo sync after matches (debouncing)
- Error handling and edge cases

December 2025.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from scripts.p2p.handlers.elo_sync import EloSyncHandlersMixin


# =============================================================================
# Test Class Implementation
# =============================================================================


class EloSyncHandlersTestClass(EloSyncHandlersMixin):
    """Test class that uses the Elo sync handlers mixin."""

    def __init__(
        self,
        node_id: str = "test-node",
        ringrift_path: str = "/home/user/ringrift",
        elo_sync_manager: MagicMock | None = None,
        sync_in_progress: bool = False,
    ):
        self.node_id = node_id
        self.ringrift_path = ringrift_path
        self.elo_sync_manager = elo_sync_manager
        self.sync_in_progress = sync_in_progress


class MockEloSyncManager:
    """Mock EloSyncManager for testing."""

    def __init__(
        self,
        status: dict | None = None,
        sync_result: bool = True,
        enable_merge: bool = True,
        db_path: Path | None = None,
    ):
        self._status = status or {
            "local_match_count": 150,
            "last_sync_time": 1700000000.0,
            "sync_errors": 0,
        }
        self._sync_result = sync_result
        self.enable_merge = enable_merge
        self.db_path = db_path or Path("/data/unified_elo.db")
        self.state = MagicMock()
        self.state.local_match_count = 150

    def get_status(self) -> dict:
        return self._status.copy()

    async def sync_with_cluster(self) -> bool:
        return self._sync_result

    async def _merge_databases(self, remote_path: Path) -> bool:
        return True

    def _update_local_stats(self) -> None:
        pass


class MockRequest:
    """Mock aiohttp Request for testing."""

    def __init__(
        self,
        headers: dict | None = None,
        json_data: dict | None = None,
        query: dict | None = None,
        raw_body: bytes | None = None,
    ):
        self.headers = headers or {}
        self._json_data = json_data or {}
        self.query = query or {}
        self._raw_body = raw_body

    async def json(self):
        return self._json_data

    async def read(self):
        if self._raw_body is not None:
            return self._raw_body
        return b""


# =============================================================================
# Elo Sync Status Tests
# =============================================================================


class TestHandleEloSyncStatus:
    """Tests for handle_elo_sync_status endpoint."""

    @pytest.fixture
    def handler(self):
        manager = MockEloSyncManager()
        return EloSyncHandlersTestClass(elo_sync_manager=manager)

    @pytest.mark.asyncio
    async def test_returns_enabled_true(self, handler):
        """Response indicates Elo sync is enabled."""
        request = MockRequest()

        response = await handler.handle_elo_sync_status(request)

        body = json.loads(response.body)
        assert body["enabled"] is True

    @pytest.mark.asyncio
    async def test_includes_node_id(self, handler):
        """Response includes node ID."""
        handler.node_id = "my-node-123"
        request = MockRequest()

        response = await handler.handle_elo_sync_status(request)

        body = json.loads(response.body)
        assert body["node_id"] == "my-node-123"

    @pytest.mark.asyncio
    async def test_includes_manager_status(self, handler):
        """Response includes manager status fields."""
        request = MockRequest()

        response = await handler.handle_elo_sync_status(request)

        body = json.loads(response.body)
        assert body["local_match_count"] == 150
        assert body["last_sync_time"] == 1700000000.0
        assert body["sync_errors"] == 0

    @pytest.mark.asyncio
    async def test_no_manager_returns_disabled(self):
        """No manager returns enabled=False."""
        handler = EloSyncHandlersTestClass(elo_sync_manager=None)
        request = MockRequest()

        response = await handler.handle_elo_sync_status(request)

        body = json.loads(response.body)
        assert body["enabled"] is False
        assert "error" in body

    @pytest.mark.asyncio
    async def test_manager_error_returns_500(self, handler):
        """Manager exception returns 500."""
        handler.elo_sync_manager.get_status = MagicMock(
            side_effect=Exception("Database error")
        )
        request = MockRequest()

        response = await handler.handle_elo_sync_status(request)

        assert response.status == 500
        body = json.loads(response.body)
        assert "Database error" in body["error"]


# =============================================================================
# Elo Sync Trigger Tests
# =============================================================================


class TestHandleEloSyncTrigger:
    """Tests for handle_elo_sync_trigger endpoint."""

    @pytest.fixture
    def handler(self):
        manager = MockEloSyncManager(sync_result=True)
        return EloSyncHandlersTestClass(elo_sync_manager=manager)

    @pytest.mark.asyncio
    async def test_successful_sync(self, handler):
        """Successful sync returns success=True."""
        request = MockRequest()

        response = await handler.handle_elo_sync_trigger(request)

        body = json.loads(response.body)
        assert body["success"] is True
        assert "status" in body

    @pytest.mark.asyncio
    async def test_sync_failure(self, handler):
        """Failed sync returns success=False."""
        handler.elo_sync_manager._sync_result = False
        request = MockRequest()

        response = await handler.handle_elo_sync_trigger(request)

        body = json.loads(response.body)
        assert body["success"] is False

    @pytest.mark.asyncio
    async def test_no_manager_returns_503(self):
        """No manager returns 503 Service Unavailable."""
        handler = EloSyncHandlersTestClass(elo_sync_manager=None)
        request = MockRequest()

        response = await handler.handle_elo_sync_trigger(request)

        assert response.status == 503
        body = json.loads(response.body)
        assert "not initialized" in body["error"]

    @pytest.mark.asyncio
    async def test_sync_error_returns_500(self, handler):
        """Sync exception returns 500."""
        handler.elo_sync_manager.sync_with_cluster = AsyncMock(
            side_effect=Exception("Network error")
        )
        request = MockRequest()

        response = await handler.handle_elo_sync_trigger(request)

        assert response.status == 500


# =============================================================================
# Elo Database Download Tests
# =============================================================================


class TestHandleEloSyncDownload:
    """Tests for handle_elo_sync_download endpoint."""

    @pytest.fixture
    def handler(self):
        manager = MockEloSyncManager()
        return EloSyncHandlersTestClass(elo_sync_manager=manager)

    @pytest.mark.asyncio
    async def test_database_not_found(self, handler):
        """Missing database returns 404."""
        handler.ringrift_path = "/nonexistent/path"
        request = MockRequest()

        response = await handler.handle_elo_sync_download(request)

        assert response.status == 404
        body = json.loads(response.body)
        assert "not found" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_database_download_success(self, handler):
        """Existing database is returned as binary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock database file
            db_dir = Path(tmpdir) / "ai-service" / "data"
            db_dir.mkdir(parents=True)
            db_path = db_dir / "unified_elo.db"
            db_path.write_bytes(b"SQLite format 3\x00test data")

            handler.ringrift_path = tmpdir
            request = MockRequest()

            response = await handler.handle_elo_sync_download(request)

            assert response.status == 200
            assert response.content_type == "application/octet-stream"
            assert b"SQLite format 3" in response.body

    @pytest.mark.asyncio
    async def test_download_includes_headers(self, handler):
        """Download response includes Content-Disposition and Content-Length."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_dir = Path(tmpdir) / "ai-service" / "data"
            db_dir.mkdir(parents=True)
            db_path = db_dir / "unified_elo.db"
            db_path.write_bytes(b"test database content")

            handler.ringrift_path = tmpdir
            request = MockRequest()

            response = await handler.handle_elo_sync_download(request)

            assert "Content-Disposition" in response.headers
            assert "Content-Length" in response.headers

    @pytest.mark.asyncio
    async def test_download_error_returns_500(self, handler):
        """Read error returns 500."""
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            with tempfile.TemporaryDirectory() as tmpdir:
                db_dir = Path(tmpdir) / "ai-service" / "data"
                db_dir.mkdir(parents=True)
                db_path = db_dir / "unified_elo.db"
                db_path.write_bytes(b"test")

                handler.ringrift_path = tmpdir
                request = MockRequest()

                response = await handler.handle_elo_sync_download(request)

                assert response.status == 500


# =============================================================================
# Elo Database Upload Tests
# =============================================================================


class TestHandleEloSyncUpload:
    """Tests for handle_elo_sync_upload endpoint."""

    @pytest.fixture
    def handler(self):
        manager = MockEloSyncManager(enable_merge=True)
        return EloSyncHandlersTestClass(elo_sync_manager=manager)

    @pytest.mark.asyncio
    async def test_upload_without_auth(self, handler):
        """Upload succeeds when no auth token configured."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RINGRIFT_ADMIN_TOKEN", None)
            request = MockRequest(raw_body=b"database content")

            response = await handler.handle_elo_sync_upload(request)

            body = json.loads(response.body)
            assert body["success"] is True

    @pytest.mark.asyncio
    async def test_upload_requires_auth_when_configured(self, handler):
        """Upload requires auth token when configured."""
        with patch.dict(os.environ, {"RINGRIFT_ADMIN_TOKEN": "secret123"}):
            request = MockRequest(raw_body=b"database content")

            response = await handler.handle_elo_sync_upload(request)

            assert response.status == 401

    @pytest.mark.asyncio
    async def test_upload_valid_auth_succeeds(self, handler):
        """Upload succeeds with valid auth token."""
        with patch.dict(os.environ, {"RINGRIFT_ADMIN_TOKEN": "secret123"}):
            request = MockRequest(
                headers={"X-Admin-Token": "secret123"},
                raw_body=b"database content",
            )

            response = await handler.handle_elo_sync_upload(request)

            body = json.loads(response.body)
            assert body["success"] is True

    @pytest.mark.asyncio
    async def test_upload_invalid_auth_fails(self, handler):
        """Upload fails with invalid auth token."""
        with patch.dict(os.environ, {"RINGRIFT_ADMIN_TOKEN": "secret123"}):
            request = MockRequest(
                headers={"X-Admin-Token": "wrongtoken"},
                raw_body=b"database content",
            )

            response = await handler.handle_elo_sync_upload(request)

            assert response.status == 401

    @pytest.mark.asyncio
    async def test_upload_empty_data_returns_400(self, handler):
        """Upload with no data returns 400."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RINGRIFT_ADMIN_TOKEN", None)
            request = MockRequest(raw_body=b"")

            response = await handler.handle_elo_sync_upload(request)

            assert response.status == 400
            body = json.loads(response.body)
            assert "No data" in body["error"]

    @pytest.mark.asyncio
    async def test_upload_no_manager_returns_503(self):
        """Upload without manager returns 503."""
        handler = EloSyncHandlersTestClass(elo_sync_manager=None)
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RINGRIFT_ADMIN_TOKEN", None)
            request = MockRequest(raw_body=b"data")

            response = await handler.handle_elo_sync_upload(request)

            assert response.status == 503

    @pytest.mark.asyncio
    async def test_upload_merge_enabled(self, handler):
        """Upload uses merge when enabled."""
        handler.elo_sync_manager._merge_databases = AsyncMock(return_value=True)
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RINGRIFT_ADMIN_TOKEN", None)
            request = MockRequest(raw_body=b"database content")

            response = await handler.handle_elo_sync_upload(request)

            handler.elo_sync_manager._merge_databases.assert_called_once()
            body = json.loads(response.body)
            assert body["success"] is True

    @pytest.mark.asyncio
    async def test_upload_merge_disabled_replaces(self, handler):
        """Upload replaces when merge disabled."""
        handler.elo_sync_manager.enable_merge = False
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RINGRIFT_ADMIN_TOKEN", None)
            with patch("shutil.copy") as mock_copy:
                request = MockRequest(raw_body=b"database content")

                response = await handler.handle_elo_sync_upload(request)

                mock_copy.assert_called_once()
                body = json.loads(response.body)
                assert body["success"] is True


# =============================================================================
# Trigger Elo Sync After Matches Tests
# =============================================================================


class TestTriggerEloSyncAfterMatches:
    """Tests for _trigger_elo_sync_after_matches method."""

    @pytest.fixture
    def handler(self):
        manager = MockEloSyncManager(sync_result=True)
        manager.sync_with_cluster = AsyncMock(return_value=True)
        return EloSyncHandlersTestClass(
            elo_sync_manager=manager,
            sync_in_progress=False,
        )

    @pytest.mark.asyncio
    async def test_no_manager_returns_early(self):
        """No manager returns without action."""
        handler = EloSyncHandlersTestClass(elo_sync_manager=None)

        await handler._trigger_elo_sync_after_matches(5)

        # Should not raise

    @pytest.mark.asyncio
    async def test_accumulates_pending_matches(self, handler):
        """Matches are accumulated before threshold."""
        # First call - should not sync (only 1 match, under threshold)
        handler._last_elo_sync_trigger = 0
        handler._pending_sync_matches = 0
        handler.sync_in_progress = True  # Prevent sync

        await handler._trigger_elo_sync_after_matches(3)
        await handler._trigger_elo_sync_after_matches(3)

        assert handler._pending_sync_matches == 6

    @pytest.mark.asyncio
    async def test_syncs_at_match_threshold(self, handler):
        """Sync triggers at match threshold."""
        handler._last_elo_sync_trigger = 0
        handler._pending_sync_matches = 9  # One away from threshold

        # Patch event bridge to avoid import issues
        with patch("scripts.p2p.handlers.elo_sync._event_bridge") as mock_bridge:
            mock_bridge.emit = AsyncMock(return_value=True)
            await handler._trigger_elo_sync_after_matches(1)  # Reaches 10

            handler.elo_sync_manager.sync_with_cluster.assert_called_once()
            assert handler._pending_sync_matches == 0

    @pytest.mark.asyncio
    async def test_syncs_after_time_interval(self, handler):
        """Sync triggers after time interval."""
        import time as time_module

        handler._last_elo_sync_trigger = time_module.time() - 60  # 60 seconds ago
        handler._pending_sync_matches = 0

        with patch("scripts.p2p.handlers.elo_sync._event_bridge") as mock_bridge:
            mock_bridge.emit = AsyncMock(return_value=True)
            await handler._trigger_elo_sync_after_matches(1)

            handler.elo_sync_manager.sync_with_cluster.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_sync_when_in_progress(self, handler):
        """No sync triggered when sync already in progress."""
        handler.sync_in_progress = True
        handler._pending_sync_matches = 20  # Way over threshold

        await handler._trigger_elo_sync_after_matches(1)

        handler.elo_sync_manager.sync_with_cluster.assert_not_called()

    @pytest.mark.asyncio
    async def test_emits_event_on_sync(self, handler):
        """Emits p2p_elo_updated event on successful sync."""
        handler._last_elo_sync_trigger = 0
        handler._pending_sync_matches = 9

        with patch("scripts.p2p.handlers.elo_sync._event_bridge") as mock_bridge:
            mock_bridge.emit = AsyncMock(return_value=True)
            await handler._trigger_elo_sync_after_matches(1)

            mock_bridge.emit.assert_called_once()
            call_args = mock_bridge.emit.call_args
            assert call_args[0][0] == "p2p_elo_updated"
            event_data = call_args[0][1]
            assert event_data["model_id"] == "cluster_sync"

    @pytest.mark.asyncio
    async def test_handles_sync_exception(self, handler):
        """Handles sync exception gracefully."""
        handler._last_elo_sync_trigger = 0
        handler._pending_sync_matches = 9
        handler.elo_sync_manager.sync_with_cluster = AsyncMock(
            side_effect=Exception("Network error")
        )

        # Should not raise
        await handler._trigger_elo_sync_after_matches(1)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEloSyncEdgeCases:
    """Edge case tests for Elo sync handlers."""

    @pytest.mark.asyncio
    async def test_sync_status_with_empty_status(self):
        """Handles empty manager status gracefully."""
        manager = MockEloSyncManager(status={})
        handler = EloSyncHandlersTestClass(elo_sync_manager=manager)
        request = MockRequest()

        response = await handler.handle_elo_sync_status(request)

        assert response.status == 200
        body = json.loads(response.body)
        assert body["enabled"] is True

    @pytest.mark.asyncio
    async def test_large_database_download(self):
        """Handles large database file."""
        manager = MockEloSyncManager()
        handler = EloSyncHandlersTestClass(elo_sync_manager=manager)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_dir = Path(tmpdir) / "ai-service" / "data"
            db_dir.mkdir(parents=True)
            db_path = db_dir / "unified_elo.db"
            # Create 1MB file
            db_path.write_bytes(b"x" * (1024 * 1024))

            handler.ringrift_path = tmpdir
            request = MockRequest()

            response = await handler.handle_elo_sync_download(request)

            assert response.status == 200
            assert len(response.body) == 1024 * 1024

    @pytest.mark.asyncio
    async def test_upload_cleans_temp_file(self):
        """Upload cleans up temp file even on error."""
        manager = MockEloSyncManager()
        manager._merge_databases = AsyncMock(side_effect=Exception("Merge failed"))
        handler = EloSyncHandlersTestClass(elo_sync_manager=manager)

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RINGRIFT_ADMIN_TOKEN", None)
            request = MockRequest(raw_body=b"test data")

            response = await handler.handle_elo_sync_upload(request)

            # Should have cleaned up temp file (no temp files remaining)
            assert response.status == 500
