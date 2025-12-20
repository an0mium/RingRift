"""Tests for Vast.ai P2P Sync script."""

import json

# Import the module under test
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[3]))

from scripts.vast_p2p_sync import (
    GPU_ROLES,
    PREFERRED_GPUS,
    P2PNode,
    VastInstance,
    match_vast_to_p2p,
)


class TestVastInstance:
    """Tests for VastInstance dataclass."""

    def test_vast_instance_creation(self):
        """Test creating a VastInstance."""
        inst = VastInstance(
            id=12345,
            machine_id=67890,
            gpu_name="RTX 3070",
            num_gpus=1,
            vcpus=8.0,
            ram_gb=32.0,
            ssh_host="ssh1.vast.ai",
            ssh_port=22222,
            status="running",
            hourly_cost=0.05,
            uptime_mins=120.0,
        )
        assert inst.id == 12345
        assert inst.gpu_name == "RTX 3070"
        assert inst.status == "running"
        assert inst.hourly_cost == 0.05


class TestP2PNode:
    """Tests for P2PNode dataclass."""

    def test_p2p_node_creation(self):
        """Test creating a P2PNode."""
        node = P2PNode(
            node_id="vast-12345",
            host="100.64.0.1",
            retired=False,
            selfplay_jobs=5,
            healthy=True,
            gpu_name="RTX 3070",
        )
        assert node.node_id == "vast-12345"
        assert node.retired is False
        assert node.selfplay_jobs == 5


class TestMatchVastToP2P:
    """Tests for matching Vast instances to P2P nodes."""

    def test_match_by_instance_id_in_node_id(self):
        """Test matching by instance ID pattern in node_id."""
        vast_instances = [
            VastInstance(
                id=12345, machine_id=999, gpu_name="RTX 3070",
                num_gpus=1, vcpus=8, ram_gb=32, ssh_host="host1",
                ssh_port=22, status="running", hourly_cost=0.05, uptime_mins=10,
            ),
        ]
        p2p_nodes = [
            P2PNode(
                node_id="vast-12345", host="100.64.0.1",
                retired=False, selfplay_jobs=3, healthy=True, gpu_name="RTX 3070",
            ),
        ]

        # Mock get_vast_tailscale_ip to return None (no Tailscale IP)
        with patch("scripts.vast_p2p_sync.get_vast_tailscale_ip", return_value=None):
            matches = match_vast_to_p2p(vast_instances, p2p_nodes)

        assert 12345 in matches
        assert matches[12345].node_id == "vast-12345"

    def test_match_by_machine_id(self):
        """Test matching by machine_id pattern in node_id."""
        vast_instances = [
            VastInstance(
                id=11111, machine_id=99999, gpu_name="RTX 3060",
                num_gpus=1, vcpus=8, ram_gb=16, ssh_host="host2",
                ssh_port=22, status="running", hourly_cost=0.04, uptime_mins=30,
            ),
        ]
        p2p_nodes = [
            P2PNode(
                node_id="vast-99999", host="100.64.0.2",
                retired=True, selfplay_jobs=0, healthy=False, gpu_name="RTX 3060",
            ),
        ]

        with patch("scripts.vast_p2p_sync.get_vast_tailscale_ip", return_value=None):
            matches = match_vast_to_p2p(vast_instances, p2p_nodes)

        assert 11111 in matches
        assert matches[11111].node_id == "vast-99999"

    def test_no_match_for_unknown_instance(self):
        """Test that unknown instances don't match."""
        vast_instances = [
            VastInstance(
                id=77777, machine_id=88888, gpu_name="RTX 4090",
                num_gpus=1, vcpus=16, ram_gb=64, ssh_host="host3",
                ssh_port=22, status="running", hourly_cost=0.50, uptime_mins=5,
            ),
        ]
        p2p_nodes = [
            P2PNode(
                node_id="lambda-gh200-a", host="192.222.51.29",
                retired=False, selfplay_jobs=10, healthy=True, gpu_name="GH200",
            ),
        ]

        with patch("scripts.vast_p2p_sync.get_vast_tailscale_ip", return_value=None):
            matches = match_vast_to_p2p(vast_instances, p2p_nodes)

        assert 77777 not in matches
        assert len(matches) == 0


class TestGPURoles:
    """Tests for GPU role mapping."""

    def test_gpu_roles_defined(self):
        """Test that GPU roles are properly defined."""
        assert "RTX 3070" in GPU_ROLES
        assert "RTX 3060" in GPU_ROLES
        assert "H100" in GPU_ROLES
        assert "A100" in GPU_ROLES

    def test_selfplay_gpus(self):
        """Test that consumer GPUs map to selfplay role."""
        selfplay_gpus = ["RTX 3070", "RTX 3060", "RTX 2060S", "RTX 2060 SUPER", "RTX 2080 Ti"]
        for gpu in selfplay_gpus:
            assert GPU_ROLES.get(gpu) == "gpu_selfplay", f"{gpu} should be gpu_selfplay"

    def test_training_gpus(self):
        """Test that datacenter GPUs map to training role."""
        training_gpus = ["A10", "A40", "A100", "H100", "RTX 4080S", "RTX 4080 SUPER"]
        for gpu in training_gpus:
            assert GPU_ROLES.get(gpu) == "nn_training_primary", f"{gpu} should be nn_training_primary"


class TestPreferredGPUs:
    """Tests for preferred GPU configuration."""

    def test_preferred_gpus_structure(self):
        """Test that PREFERRED_GPUS has correct structure."""
        assert len(PREFERRED_GPUS) > 0
        for pref in PREFERRED_GPUS:
            assert "name" in pref
            assert "max_price" in pref
            assert "role" in pref
            assert pref["max_price"] > 0

    def test_preferred_gpus_ordered_by_value(self):
        """Test that preferred GPUs are reasonably ordered."""
        # First few should be cost-effective options
        assert PREFERRED_GPUS[0]["name"] in ["RTX 3070", "RTX 3060"]


class TestDryRunMode:
    """Tests for dry-run mode functionality."""

    def test_argument_parsing(self):
        """Test that --dry-run flag is parsed correctly."""
        import argparse

        # Simulate argument parsing
        parser = argparse.ArgumentParser()
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--full', action='store_true')

        args = parser.parse_args(['--dry-run', '--full'])
        assert args.dry_run is True
        assert args.full is True

    def test_deprovision_parsing(self):
        """Test that --deprovision parses comma-separated IDs."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--deprovision', type=str)

        args = parser.parse_args(['--deprovision', '123,456,789'])
        instance_ids = [int(x.strip()) for x in args.deprovision.split(',') if x.strip().isdigit()]
        assert instance_ids == [123, 456, 789]


class TestAsyncWrappers:
    """Tests for async wrapper functions."""

    @pytest.mark.asyncio
    async def test_get_vast_instances_async(self):
        """Test async wrapper returns list."""
        from scripts.vast_p2p_sync import get_vast_instances_async

        with patch("scripts.vast_p2p_sync.get_vast_instances", return_value=[]):
            result = await get_vast_instances_async()
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_p2p_nodes_async(self):
        """Test async wrapper returns list."""
        from scripts.vast_p2p_sync import get_p2p_nodes_async

        with patch("scripts.vast_p2p_sync.get_p2p_nodes", return_value=[]):
            result = await get_p2p_nodes_async()
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_provision_instances_async(self):
        """Test async provision wrapper."""
        from scripts.vast_p2p_sync import provision_instances_async

        with patch("scripts.vast_p2p_sync.provision_instances", return_value=2):
            with patch("scripts.vast_p2p_sync.get_vast_instances", return_value=[]):
                count, ids = await provision_instances_async(count=2, max_total_hourly=0.50)
                assert count == 2
                assert isinstance(ids, list)

    @pytest.mark.asyncio
    async def test_deprovision_instances_async(self):
        """Test async deprovision wrapper."""
        from scripts.vast_p2p_sync import deprovision_instances_async

        with patch("scripts.vast_p2p_sync.deprovision_instances", return_value=3):
            count = await deprovision_instances_async([123, 456, 789], destroy=True)
            assert count == 3


class TestNodeToVastMapping:
    """Tests for node-to-Vast instance mapping."""

    def test_get_node_to_vast_mapping(self):
        """Test mapping from node_id to vast_id."""
        from scripts.vast_p2p_sync import get_node_to_vast_mapping

        mock_vast = [
            VastInstance(
                id=12345, machine_id=999, gpu_name="RTX 3070",
                num_gpus=1, vcpus=8, ram_gb=32, ssh_host="host1",
                ssh_port=22, status="running", hourly_cost=0.05, uptime_mins=10,
            ),
        ]
        mock_p2p = [
            P2PNode(
                node_id="vast-12345", host="100.64.0.1",
                retired=False, selfplay_jobs=3, healthy=True, gpu_name="RTX 3070",
            ),
        ]

        with patch("scripts.vast_p2p_sync.get_vast_instances", return_value=mock_vast):
            with patch("scripts.vast_p2p_sync.get_p2p_nodes", return_value=mock_p2p):
                with patch("scripts.vast_p2p_sync.get_vast_tailscale_ip", return_value=None):
                    mapping = get_node_to_vast_mapping()

        assert "vast-12345" in mapping
        assert mapping["vast-12345"] == 12345


class TestCostFunctions:
    """Tests for cost-related functions."""

    def test_get_vast_instance_costs(self):
        """Test getting instance costs."""
        from scripts.vast_p2p_sync import get_vast_instance_costs

        mock_instances = [
            VastInstance(
                id=111, machine_id=1, gpu_name="RTX 3070",
                num_gpus=1, vcpus=8, ram_gb=32, ssh_host="h1",
                ssh_port=22, status="running", hourly_cost=0.05, uptime_mins=10,
            ),
            VastInstance(
                id=222, machine_id=2, gpu_name="RTX 3060",
                num_gpus=1, vcpus=8, ram_gb=16, ssh_host="h2",
                ssh_port=22, status="running", hourly_cost=0.04, uptime_mins=20,
            ),
        ]

        with patch("scripts.vast_p2p_sync.get_vast_instances", return_value=mock_instances):
            costs = get_vast_instance_costs()

        assert costs[111] == 0.05
        assert costs[222] == 0.04
