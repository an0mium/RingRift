"""Tests for RegistrySyncManager.

Tests sync state management and local registry stats
without requiring network connections.
"""

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.training.registry_sync_manager import (
    NodeInfo,
    RegistrySyncManager,
    SyncState,
)


class TestSyncState:
    """Test SyncState (alias for DatabaseSyncState) dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        state = SyncState()
        assert state.last_sync_timestamp == 0.0
        assert state.local_record_count == 0
        assert state.synced_nodes == set()
        assert state.pending_syncs == set()

    def test_custom_values(self):
        """Test custom initialization."""
        state = SyncState(
            last_sync_timestamp=1000.0,
            local_record_count=5,
            synced_nodes={"node1"},
            pending_syncs={"node2"},
        )
        assert state.last_sync_timestamp == 1000.0
        assert state.local_record_count == 5
        assert state.synced_nodes == {"node1"}
        assert state.pending_syncs == {"node2"}


class TestNodeInfo:
    """Test NodeInfo (alias for SyncNodeInfo) dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        node = NodeInfo(name="test-node")
        assert node.name == "test-node"
        assert node.last_seen == 0.0
        assert node.record_count == 0
        assert node.reachable is True
        assert node.tailscale_ip is None
        assert node.ssh_port == 22

    def test_custom_values(self):
        """Test custom initialization."""
        node = NodeInfo(
            name="gpu-node",
            tailscale_ip="100.64.0.1",
            ssh_port=2222,
            record_count=3,
        )
        assert node.name == "gpu-node"
        assert node.tailscale_ip == "100.64.0.1"
        assert node.ssh_port == 2222
        assert node.record_count == 3


class TestRegistrySyncManager:
    """Test RegistrySyncManager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = Path(self.temp_dir) / "model_registry.db"

        # Create a test registry database
        self._create_test_registry()

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_registry(self):
        """Create a test registry database."""
        conn = sqlite3.connect(str(self.registry_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE models (
                model_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                model_type TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                version INTEGER,
                stage TEXT,
                file_path TEXT,
                file_hash TEXT,
                file_size_bytes INTEGER,
                metrics_json TEXT,
                training_config_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # Insert test data
        cursor.execute("""
            INSERT INTO models VALUES (?, ?, ?, ?, ?, ?)
        """, ("test_model", "Test Model", "Description", "policy_value",
              datetime.now().isoformat(), datetime.now().isoformat()))

        cursor.execute("""
            INSERT INTO versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (1, "test_model", 1, "production", "/path/to/model.pt",
              "abc123", 1024, "{}", "{}", datetime.now().isoformat(),
              datetime.now().isoformat()))

        cursor.execute("""
            INSERT INTO versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (2, "test_model", 2, "staging", "/path/to/model2.pt",
              "def456", 2048, "{}", "{}", datetime.now().isoformat(),
              datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def test_initialization(self):
        """Test manager initializes correctly."""
        manager = RegistrySyncManager(registry_path=self.registry_path)
        assert manager.registry_path == self.registry_path
        assert manager.sync_interval == 600
        assert manager.state is not None

    def test_custom_sync_interval(self):
        """Test custom sync interval."""
        manager = RegistrySyncManager(
            registry_path=self.registry_path,
            sync_interval=300,
        )
        assert manager.sync_interval == 300

    def test_update_local_stats(self):
        """Test local stats are updated from registry."""
        manager = RegistrySyncManager(registry_path=self.registry_path)
        manager._update_local_stats()

        assert manager._model_count == 1
        assert manager._version_count == 2

    def test_update_local_stats_missing_registry(self):
        """Test local stats with missing registry."""
        missing_path = Path(self.temp_dir) / "nonexistent.db"
        manager = RegistrySyncManager(registry_path=missing_path)
        manager._update_local_stats()

        # Should not raise, just leave counts at 0
        assert manager._model_count == 0
        assert manager._version_count == 0

    def test_get_sync_status(self):
        """Test get_sync_status returns expected structure."""
        manager = RegistrySyncManager(registry_path=self.registry_path)
        manager._update_local_stats()

        status = manager.get_sync_status()

        assert 'last_sync' in status
        assert 'local_models' in status
        assert 'local_versions' in status
        assert 'synced_nodes' in status
        assert 'nodes_available' in status
        assert 'circuit_breakers' in status

        assert status['local_models'] == 1
        assert status['local_versions'] == 2

    def test_on_sync_callbacks(self):
        """Test callback registration."""
        manager = RegistrySyncManager(registry_path=self.registry_path)

        callback_called = []

        def on_complete():
            callback_called.append("complete")

        def on_failed():
            callback_called.append("failed")

        manager.on_sync_complete(on_complete)
        manager.on_sync_failed(on_failed)

        assert len(manager._on_sync_complete_callbacks) == 1
        assert len(manager._on_sync_failed_callbacks) == 1


class TestRegistrySyncManagerAsync:
    """Test async methods of RegistrySyncManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = Path(self.temp_dir) / "model_registry.db"

        # Create test registry
        conn = sqlite3.connect(str(self.registry_path))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE models (
                model_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                model_type TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                version INTEGER,
                stage TEXT,
                file_path TEXT,
                file_hash TEXT,
                file_size_bytes INTEGER,
                metrics_json TEXT,
                training_config_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        conn.commit()
        conn.close()

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test async initialization."""
        manager = RegistrySyncManager(registry_path=self.registry_path)

        # Mock node discovery to avoid network calls
        with patch.object(manager, 'discover_nodes', new_callable=AsyncMock):
            await manager.initialize()

        assert manager._model_count == 0  # Empty test DB
        assert manager._version_count == 0

    @pytest.mark.asyncio
    async def test_sync_via_tailscale_no_ip(self):
        """Test tailscale sync fails without IP."""
        manager = RegistrySyncManager(registry_path=self.registry_path)
        node = NodeInfo(name="test-node", tailscale_ip=None)

        result = await manager._sync_via_tailscale(node)

        assert result is False


class TestMergeDatabases:
    """Test database merging functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.local_path = Path(self.temp_dir) / "local.db"
        self.remote_path = Path(self.temp_dir) / "remote.db"

        # Create local database
        self._create_db(self.local_path, [
            ("model1", "Model 1", "local"),
        ], [
            ("model1", 1, "production"),
        ])

        # Create remote database with additional data
        self._create_db(self.remote_path, [
            ("model1", "Model 1", "remote"),  # Same model
            ("model2", "Model 2", "remote"),  # New model
        ], [
            ("model1", 1, "production"),  # Same version
            ("model1", 2, "staging"),     # New version
            ("model2", 1, "development"), # New model version
        ])

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_db(self, path: Path, models: list, versions: list):
        """Create a test database."""
        conn = sqlite3.connect(str(path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE models (
                model_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                model_type TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                version INTEGER,
                stage TEXT,
                file_path TEXT,
                file_hash TEXT,
                file_size_bytes INTEGER,
                metrics_json TEXT,
                training_config_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        now = datetime.now().isoformat()
        for model_id, name, desc in models:
            cursor.execute(
                "INSERT INTO models VALUES (?, ?, ?, ?, ?, ?)",
                (model_id, name, desc, "policy_value", now, now)
            )

        for model_id, version, stage in versions:
            cursor.execute(
                "INSERT INTO versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (None, model_id, version, stage, f"/path/{model_id}_v{version}.pt",
                 "hash", 1024, "{}", "{}", now, now)
            )

        conn.commit()
        conn.close()

    @pytest.mark.asyncio
    async def test_merge_databases(self):
        """Test merging remote database into local."""
        manager = RegistrySyncManager(registry_path=self.local_path)

        result = await manager._merge_databases(self.remote_path)

        assert result is True

        # Verify local database now has all data
        conn = sqlite3.connect(str(self.local_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM models")
        assert cursor.fetchone()[0] == 2

        cursor.execute("SELECT COUNT(*) FROM versions")
        assert cursor.fetchone()[0] == 3

        conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
