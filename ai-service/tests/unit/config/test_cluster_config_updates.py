"""Unit tests for cluster_config dynamic update functions.

December 2025: Tests for add_or_update_node, update_node_status, remove_node.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from app.config.cluster_config import (
    ClusterNode,
    add_or_update_node,
    update_node_status,
    remove_node,
    cluster_node_to_dict,
    clear_cluster_config_cache,
    load_cluster_config,
)


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    initial_config = {
        "hosts": {
            "test-node-1": {
                "status": "ready",
                "ssh_host": "192.168.1.1",
                "gpu": "RTX 4090",
                "gpu_vram_gb": 24,
            },
            "test-node-2": {
                "status": "offline",
                "ssh_host": "192.168.1.2",
            },
        },
        "sync_routing": {},
        "auto_sync": {},
        "elo_sync": {},
        "p2p_voters": [],
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.safe_dump(initial_config, f)
        config_path = Path(f.name)

    yield config_path

    # Cleanup
    config_path.unlink(missing_ok=True)
    clear_cluster_config_cache()


class TestAddOrUpdateNode:
    """Tests for add_or_update_node function."""

    def test_add_new_node(self, temp_config_file):
        """Test adding a completely new node."""
        result = add_or_update_node(
            "new-node",
            {
                "status": "ready",
                "ssh_host": "192.168.1.100",
                "gpu": "H100",
                "gpu_vram_gb": 80,
            },
            config_path=temp_config_file,
        )

        assert result is True

        # Verify node was added
        with open(temp_config_file) as f:
            config = yaml.safe_load(f)

        assert "new-node" in config["hosts"]
        assert config["hosts"]["new-node"]["ssh_host"] == "192.168.1.100"
        assert config["hosts"]["new-node"]["gpu"] == "H100"

    def test_update_existing_node(self, temp_config_file):
        """Test updating an existing node."""
        result = add_or_update_node(
            "test-node-1",
            {
                "status": "offline",
                "ssh_host": "192.168.1.50",
            },
            config_path=temp_config_file,
        )

        assert result is True

        # Verify node was updated
        with open(temp_config_file) as f:
            config = yaml.safe_load(f)

        assert config["hosts"]["test-node-1"]["status"] == "offline"
        assert config["hosts"]["test-node-1"]["ssh_host"] == "192.168.1.50"
        # Original fields should be preserved
        assert config["hosts"]["test-node-1"]["gpu"] == "RTX 4090"
        assert config["hosts"]["test-node-1"]["gpu_vram_gb"] == 24

    def test_nonexistent_config_file(self):
        """Test with nonexistent config file."""
        result = add_or_update_node(
            "node",
            {"status": "ready"},
            config_path="/nonexistent/path.yaml",
        )
        assert result is False

    def test_clears_cache(self, temp_config_file):
        """Test that cache is cleared after update."""
        # Load config to populate cache
        config1 = load_cluster_config(temp_config_file)
        assert "test-node-1" in config1.hosts_raw

        # Add new node
        add_or_update_node(
            "cache-test-node",
            {"status": "ready"},
            config_path=temp_config_file,
        )

        # Reload should show new node
        config2 = load_cluster_config(temp_config_file, force_reload=True)
        assert "cache-test-node" in config2.hosts_raw


class TestUpdateNodeStatus:
    """Tests for update_node_status function."""

    def test_update_status_only(self, temp_config_file):
        """Test updating just the status."""
        result = update_node_status(
            "test-node-1",
            "terminated",
            config_path=temp_config_file,
        )

        assert result is True

        with open(temp_config_file) as f:
            config = yaml.safe_load(f)

        assert config["hosts"]["test-node-1"]["status"] == "terminated"
        # Other fields preserved
        assert config["hosts"]["test-node-1"]["gpu"] == "RTX 4090"

    def test_update_status_with_extra_fields(self, temp_config_file):
        """Test updating status with additional fields."""
        result = update_node_status(
            "test-node-2",
            "ready",
            config_path=temp_config_file,
            ssh_host="10.0.0.1",
            gpu="A100",
            gpu_vram_gb=80,
        )

        assert result is True

        with open(temp_config_file) as f:
            config = yaml.safe_load(f)

        assert config["hosts"]["test-node-2"]["status"] == "ready"
        assert config["hosts"]["test-node-2"]["ssh_host"] == "10.0.0.1"
        assert config["hosts"]["test-node-2"]["gpu"] == "A100"
        assert config["hosts"]["test-node-2"]["gpu_vram_gb"] == 80


class TestRemoveNode:
    """Tests for remove_node function."""

    def test_remove_existing_node(self, temp_config_file):
        """Test removing an existing node."""
        result = remove_node("test-node-2", config_path=temp_config_file)

        assert result is True

        with open(temp_config_file) as f:
            config = yaml.safe_load(f)

        assert "test-node-2" not in config["hosts"]
        assert "test-node-1" in config["hosts"]  # Other nodes preserved

    def test_remove_nonexistent_node(self, temp_config_file):
        """Test removing a node that doesn't exist."""
        result = remove_node("nonexistent-node", config_path=temp_config_file)

        # Should succeed (already removed)
        assert result is True

    def test_remove_nonexistent_config(self):
        """Test with nonexistent config file."""
        result = remove_node("node", config_path="/nonexistent/path.yaml")
        assert result is False


class TestClusterNodeToDict:
    """Tests for cluster_node_to_dict function."""

    def test_minimal_node(self):
        """Test converting a node with minimal fields."""
        node = ClusterNode(name="test", status="ready")
        result = cluster_node_to_dict(node)

        assert result == {"status": "ready"}

    def test_full_node(self):
        """Test converting a node with all fields."""
        node = ClusterNode(
            name="full-node",
            tailscale_ip="100.1.2.3",
            ssh_host="192.168.1.1",
            ssh_user="root",
            ssh_key="~/.ssh/id_ed25519",
            ssh_port=2222,
            ringrift_path="/opt/ringrift",
            status="ready",
            role="gpu-worker",
            memory_gb=128,
            cpus=64,
            gpu="H100",
            gpu_vram_gb=80,
            bandwidth_mbps=1000,
            cuda_capable=True,
            selfplay_enabled=True,
            training_enabled=True,
        )
        result = cluster_node_to_dict(node)

        assert result["tailscale_ip"] == "100.1.2.3"
        assert result["ssh_host"] == "192.168.1.1"
        assert result["ssh_user"] == "root"
        assert result["ssh_port"] == 2222
        assert result["ringrift_path"] == "/opt/ringrift"
        assert result["gpu"] == "H100"
        assert result["gpu_vram_gb"] == 80
        assert result["cuda_capable"] is True
        assert result["training_enabled"] is True

    def test_default_values_not_included(self):
        """Test that default values are not included."""
        node = ClusterNode(
            name="default-node",
            status="ready",
            ssh_user="ubuntu",  # Default
            ssh_port=22,  # Default
            ringrift_path="~/ringrift/ai-service",  # Default
            selfplay_enabled=True,  # Default
            data_server_port=8766,  # Default
        )
        result = cluster_node_to_dict(node)

        # Only status should be included
        assert "ssh_user" not in result
        assert "ssh_port" not in result
        assert "ringrift_path" not in result
        assert "selfplay_enabled" not in result
        assert "data_server_port" not in result
        assert result == {"status": "ready"}


class TestIntegration:
    """Integration tests for config update workflow."""

    def test_provisioning_workflow(self, temp_config_file):
        """Test the full provisioning workflow."""
        # 1. Provision a new node
        add_or_update_node(
            "runpod-abc123",
            {
                "status": "setup",
                "ssh_host": "10.0.0.100",
                "gpu": "A100",
                "gpu_vram_gb": 80,
                "cuda_capable": True,
                "selfplay_enabled": True,
                "training_enabled": True,
            },
            config_path=temp_config_file,
        )

        # 2. Update to ready when P2P connects
        update_node_status(
            "runpod-abc123",
            "ready",
            config_path=temp_config_file,
            tailscale_ip="100.64.0.50",
        )

        # 3. Verify final state
        config = load_cluster_config(temp_config_file, force_reload=True)
        node = config.hosts_raw["runpod-abc123"]

        assert node["status"] == "ready"
        assert node["tailscale_ip"] == "100.64.0.50"
        assert node["gpu"] == "A100"
        assert node["training_enabled"] is True

    def test_recovery_workflow(self, temp_config_file):
        """Test the recovery workflow (recreate instance)."""
        # 1. Node goes offline
        update_node_status(
            "test-node-1",
            "offline",
            config_path=temp_config_file,
        )

        # 2. Instance is recreated with new IP
        update_node_status(
            "test-node-1",
            "ready",
            config_path=temp_config_file,
            ssh_host="192.168.1.200",  # New IP
        )

        # 3. Verify
        config = load_cluster_config(temp_config_file, force_reload=True)
        node = config.hosts_raw["test-node-1"]

        assert node["status"] == "ready"
        assert node["ssh_host"] == "192.168.1.200"
        # GPU info preserved
        assert node["gpu"] == "RTX 4090"
        assert node["gpu_vram_gb"] == 24

    def test_termination_workflow(self, temp_config_file):
        """Test the termination workflow."""
        # 1. Mark as terminated
        update_node_status(
            "test-node-2",
            "terminated",
            config_path=temp_config_file,
        )

        # 2. Remove from config
        remove_node("test-node-2", config_path=temp_config_file)

        # 3. Verify
        config = load_cluster_config(temp_config_file, force_reload=True)
        assert "test-node-2" not in config.hosts_raw
