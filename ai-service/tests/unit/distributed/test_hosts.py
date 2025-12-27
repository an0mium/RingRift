"""Tests for app/distributed/hosts.py - cluster host configuration."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.distributed.hosts import (
    BOARD_MEMORY_REQUIREMENTS,
    DEFAULT_SSH_KEY,
    HostConfig,
    HostMemoryInfo,
    get_local_memory_gb,
)


class TestHostMemoryInfo:
    """Tests for HostMemoryInfo dataclass."""

    def test_basic_creation(self):
        """Test creating HostMemoryInfo."""
        info = HostMemoryInfo(total_gb=64, available_gb=48)
        assert info.total_gb == 64
        assert info.available_gb == 48

    def test_str_representation(self):
        """Test string representation."""
        info = HostMemoryInfo(total_gb=96, available_gb=80)
        assert str(info) == "96GB total, 80GB available"

    def test_is_high_memory_true(self):
        """Test high memory detection for 48GB+ hosts."""
        info = HostMemoryInfo(total_gb=48, available_gb=40)
        assert info.is_high_memory is True

        info = HostMemoryInfo(total_gb=96, available_gb=80)
        assert info.is_high_memory is True

    def test_is_high_memory_false(self):
        """Test low memory detection for <48GB hosts."""
        info = HostMemoryInfo(total_gb=32, available_gb=24)
        assert info.is_high_memory is False

        info = HostMemoryInfo(total_gb=16, available_gb=12)
        assert info.is_high_memory is False


class TestHostConfig:
    """Tests for HostConfig dataclass."""

    def test_basic_creation(self):
        """Test creating HostConfig with required fields."""
        config = HostConfig(name="test-host", ssh_host="192.168.1.1")
        assert config.name == "test-host"
        assert config.ssh_host == "192.168.1.1"
        assert config.ssh_port == 22  # default
        assert config.max_parallel_jobs == 1  # default
        assert config.worker_port == 8765  # default

    def test_ssh_target_simple(self):
        """Test ssh_target with simple host."""
        config = HostConfig(name="test", ssh_host="192.168.1.1")
        assert config.ssh_target == "192.168.1.1"

    def test_ssh_target_with_user(self):
        """Test ssh_target with user specified."""
        config = HostConfig(name="test", ssh_host="192.168.1.1", ssh_user="ubuntu")
        assert config.ssh_target == "ubuntu@192.168.1.1"

    def test_ssh_target_with_user_in_host(self):
        """Test ssh_target when user is in host string."""
        config = HostConfig(name="test", ssh_host="root@192.168.1.1")
        assert config.ssh_target == "root@192.168.1.1"

    def test_ssh_targets_with_tailscale(self):
        """Test ssh_targets prefers tailscale IP."""
        config = HostConfig(
            name="test",
            ssh_host="192.168.1.1",
            tailscale_ip="100.64.0.1",
            ssh_user="ubuntu",
        )
        targets = config.ssh_targets
        assert len(targets) == 2
        assert targets[0] == "ubuntu@100.64.0.1"  # Tailscale preferred
        assert targets[1] == "ubuntu@192.168.1.1"  # Fallback

    def test_ssh_targets_no_tailscale(self):
        """Test ssh_targets without tailscale."""
        config = HostConfig(name="test", ssh_host="192.168.1.1", ssh_user="ubuntu")
        targets = config.ssh_targets
        assert len(targets) == 1
        assert targets[0] == "ubuntu@192.168.1.1"

    def test_ssh_key_path_default(self):
        """Test default SSH key path."""
        config = HostConfig(name="test", ssh_host="192.168.1.1")
        expected = os.path.expanduser(DEFAULT_SSH_KEY)
        assert config.ssh_key_path == expected

    def test_ssh_key_path_custom(self):
        """Test custom SSH key path."""
        config = HostConfig(name="test", ssh_host="192.168.1.1", ssh_key="~/.ssh/custom")
        expected = os.path.expanduser("~/.ssh/custom")
        assert config.ssh_key_path == expected

    def test_all_optional_fields(self):
        """Test HostConfig with all optional fields."""
        config = HostConfig(
            name="full-test",
            ssh_host="192.168.1.1",
            tailscale_ip="100.64.0.1",
            ssh_user="admin",
            ssh_port=2222,
            ssh_key="~/.ssh/special",
            memory_gb=96,
            work_dir="/opt/ringrift",
            venv_activate="/opt/ringrift/venv/bin/activate",
            python_path="/opt/ringrift/venv/bin/python",
            max_parallel_jobs=4,
            worker_port=9000,
            worker_url="http://localhost:9000",
            cloudflare_tunnel="host.example.com",
            cloudflare_service_token_id="token123",
            cloudflare_service_token_secret="secret456",
            properties={"gpu": "H100", "role": "training"},
        )
        assert config.name == "full-test"
        assert config.ssh_port == 2222
        assert config.memory_gb == 96
        assert config.max_parallel_jobs == 4
        assert config.worker_port == 9000
        assert config.properties["gpu"] == "H100"


class TestBoardMemoryRequirements:
    """Tests for memory requirements constants."""

    def test_square8_requirement(self):
        """Test square8 has lowest memory requirement."""
        assert BOARD_MEMORY_REQUIREMENTS["square8"] == 8

    def test_large_boards_requirement(self):
        """Test large boards have high memory requirement."""
        assert BOARD_MEMORY_REQUIREMENTS["square19"] == 48
        assert BOARD_MEMORY_REQUIREMENTS["hexagonal"] == 48

    def test_all_boards_have_requirements(self):
        """Test all expected board types have requirements."""
        expected_boards = ["square8", "square19", "hexagonal"]
        for board in expected_boards:
            assert board in BOARD_MEMORY_REQUIREMENTS


class TestGetLocalMemoryGb:
    """Tests for get_local_memory_gb function."""

    def test_returns_tuple(self):
        """Test that local memory detection returns tuple (total, available)."""
        memory = get_local_memory_gb()
        assert isinstance(memory, tuple)
        assert len(memory) == 2
        total, available = memory
        assert isinstance(total, int)
        assert isinstance(available, int)
        assert total > 0
        assert available >= 0

    def test_memory_is_reasonable(self):
        """Test that memory values are in reasonable range."""
        total, available = get_local_memory_gb()
        # Most machines have between 4GB and 2TB
        assert total >= 1
        assert total <= 2048
        # Available should be <= total
        assert available <= total

    def test_available_less_than_total(self):
        """Test available memory is at most total."""
        total, available = get_local_memory_gb()
        assert available <= total


class TestHostConfigProperties:
    """Tests for computed properties on HostConfig."""

    def test_properties_dict_default_empty(self):
        """Test properties dict defaults to empty."""
        config = HostConfig(name="test", ssh_host="192.168.1.1")
        assert config.properties == {}

    def test_properties_dict_mutable(self):
        """Test properties dict can be modified."""
        config = HostConfig(name="test", ssh_host="192.168.1.1")
        config.properties["custom"] = "value"
        assert config.properties["custom"] == "value"

    def test_cloudflare_fields_optional(self):
        """Test Cloudflare fields are optional."""
        config = HostConfig(name="test", ssh_host="192.168.1.1")
        assert config.cloudflare_tunnel is None
        assert config.cloudflare_service_token_id is None
        assert config.cloudflare_service_token_secret is None


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_tailscale_ip_string(self):
        """Test empty tailscale IP is handled."""
        config = HostConfig(
            name="test",
            ssh_host="192.168.1.1",
            tailscale_ip="",  # Empty string
            ssh_user="ubuntu",
        )
        targets = config.ssh_targets
        assert len(targets) == 1
        assert targets[0] == "ubuntu@192.168.1.1"

    def test_whitespace_in_host(self):
        """Test whitespace is trimmed from hosts."""
        config = HostConfig(
            name="test",
            ssh_host="192.168.1.1",
            tailscale_ip="  100.64.0.1  ",  # Whitespace
            ssh_user="ubuntu",
        )
        targets = config.ssh_targets
        # Whitespace should be trimmed
        assert any("100.64.0.1" in t for t in targets)

    def test_duplicate_hosts_deduplicated(self):
        """Test duplicate hosts are deduplicated in targets."""
        config = HostConfig(
            name="test",
            ssh_host="192.168.1.1",
            tailscale_ip="192.168.1.1",  # Same as ssh_host
            ssh_user="ubuntu",
        )
        targets = config.ssh_targets
        # Should only have one entry
        assert len(targets) == 1
        assert targets[0] == "ubuntu@192.168.1.1"


class TestDefaultSshKey:
    """Tests for DEFAULT_SSH_KEY constant."""

    def test_default_key_is_valid_path(self):
        """Test default SSH key is a valid path format."""
        assert DEFAULT_SSH_KEY.startswith("~/.ssh/")

    def test_default_key_expandable(self):
        """Test default SSH key path can be expanded."""
        expanded = os.path.expanduser(DEFAULT_SSH_KEY)
        assert not expanded.startswith("~")
        assert "/.ssh/" in expanded
