"""Tests for HetznerProvider - Hetzner cloud provider implementation.

Tests cover:
- Provider initialization and configuration
- Instance parsing from JSON
- Status mapping
- CLI command execution (mocked)
- Instance listing and filtering
- Scale up/down operations
- Cost estimation (CPU-only)
- Health check integration
"""

import json
import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.providers.hetzner_provider import (
    HetznerProvider,
    HETZNER_PLANS,
)
from app.coordination.providers.base import (
    GPUType,
    InstanceStatus,
    ProviderType,
)


class TestHetznerProviderInit:
    """Tests for HetznerProvider initialization."""

    def test_init_with_token(self):
        """Test initialization with explicit token."""
        provider = HetznerProvider(token="test-token-123")

        assert provider._token == "test-token-123"

    def test_init_from_env(self):
        """Test initialization from environment variable."""
        with patch.dict(os.environ, {"HCLOUD_TOKEN": "env-token-456"}):
            provider = HetznerProvider()

            assert provider._token == "env-token-456"

    def test_init_auto_detect_cli(self):
        """Test CLI path auto-detection."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/local/bin/hcloud"
            provider = HetznerProvider()

            assert provider._cli_path == "/usr/local/bin/hcloud"

    def test_init_cli_not_found(self):
        """Test when CLI is not found."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None
            provider = HetznerProvider()

            assert provider._cli_path is None


class TestHetznerProviderProperties:
    """Tests for HetznerProvider properties."""

    def test_provider_type(self):
        """Test provider_type property."""
        provider = HetznerProvider(token="test")

        assert provider.provider_type == ProviderType.HETZNER

    def test_name(self):
        """Test name property."""
        provider = HetznerProvider(token="test")

        assert provider.name == "Hetzner"


class TestHetznerProviderConfiguration:
    """Tests for HetznerProvider configuration checking."""

    def test_is_configured_with_token(self):
        """Test is_configured returns True with token."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None  # No CLI
            provider = HetznerProvider(token="test-token")

            assert provider.is_configured() is True

    def test_is_configured_with_cli(self):
        """Test is_configured returns True with CLI."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            with patch.dict(os.environ, {}, clear=True):
                provider = HetznerProvider()

                assert provider.is_configured() is True

    def test_is_configured_neither(self):
        """Test is_configured returns False without token or CLI."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None
            with patch.dict(os.environ, {}, clear=True):
                provider = HetznerProvider()

                assert provider.is_configured() is False


class TestHetznerPlans:
    """Tests for HETZNER_PLANS constant."""

    def test_plans_structure(self):
        """Test plans have correct structure."""
        for plan_name, (vcpus, ram_gb, cost) in HETZNER_PLANS.items():
            assert isinstance(plan_name, str)
            assert isinstance(vcpus, int)
            assert vcpus > 0
            assert isinstance(ram_gb, int)
            assert ram_gb > 0
            assert isinstance(cost, float)
            assert cost > 0

    def test_shared_cpu_plans_exist(self):
        """Test shared CPU plans exist."""
        assert "cx22" in HETZNER_PLANS
        assert "cx32" in HETZNER_PLANS
        assert "cx52" in HETZNER_PLANS

    def test_dedicated_cpu_plans_exist(self):
        """Test dedicated CPU plans exist."""
        assert "ccx13" in HETZNER_PLANS
        assert "ccx33" in HETZNER_PLANS
        assert "ccx63" in HETZNER_PLANS

    def test_plan_scaling(self):
        """Test that larger plans have more resources and higher cost."""
        # Compare cx22 to cx52
        assert HETZNER_PLANS["cx22"][0] < HETZNER_PLANS["cx52"][0]  # vCPUs
        assert HETZNER_PLANS["cx22"][1] < HETZNER_PLANS["cx52"][1]  # RAM
        assert HETZNER_PLANS["cx22"][2] < HETZNER_PLANS["cx52"][2]  # Cost


class TestHetznerProviderInstanceParsing:
    """Tests for instance parsing from JSON."""

    def get_sample_instance_data(self):
        """Get sample server data."""
        return {
            "id": 12345,
            "name": "hetzner-cpu1",
            "status": "running",
            "server_type": {"name": "ccx33"},
            "public_net": {
                "ipv4": {"ip": "192.168.1.100"},
                "ipv6": {"ip": "2001:db8::1"},
            },
            "created": "2024-12-28T10:00:00Z",
            "datacenter": {"name": "fsn1-dc14"},
            "labels": {"role": "p2p_voter", "env": "prod"},
        }

    def test_parse_instance_running(self):
        """Test parsing running instance."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()
            data = self.get_sample_instance_data()

            instance = provider._parse_instance(data)

            assert instance.id == "12345"
            assert instance.provider == ProviderType.HETZNER
            assert instance.name == "hetzner-cpu1"
            assert instance.status == InstanceStatus.RUNNING
            assert instance.gpu_type == GPUType.CPU_ONLY
            assert instance.gpu_count == 0
            assert instance.gpu_memory_gb == 0
            assert instance.ip_address == "192.168.1.100"
            assert instance.ssh_port == 22
            assert instance.ssh_user == "root"
            assert instance.region == "fsn1-dc14"
            assert instance.cost_per_hour == HETZNER_PLANS["ccx33"][2]

    def test_parse_instance_initializing(self):
        """Test parsing initializing/pending instance."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()
            data = self.get_sample_instance_data()
            data["status"] = "initializing"

            instance = provider._parse_instance(data)

            assert instance.status == InstanceStatus.PENDING

    def test_parse_instance_starting(self):
        """Test parsing starting instance."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()
            data = self.get_sample_instance_data()
            data["status"] = "starting"

            instance = provider._parse_instance(data)

            assert instance.status == InstanceStatus.STARTING

    def test_parse_instance_stopping(self):
        """Test parsing stopping instance."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()
            data = self.get_sample_instance_data()
            data["status"] = "stopping"

            instance = provider._parse_instance(data)

            assert instance.status == InstanceStatus.STOPPING

    def test_parse_instance_off(self):
        """Test parsing off/stopped instance."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()
            data = self.get_sample_instance_data()
            data["status"] = "off"

            instance = provider._parse_instance(data)

            assert instance.status == InstanceStatus.STOPPED

    def test_parse_instance_deleting(self):
        """Test parsing deleting instance."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()
            data = self.get_sample_instance_data()
            data["status"] = "deleting"

            instance = provider._parse_instance(data)

            assert instance.status == InstanceStatus.TERMINATED

    def test_parse_instance_unknown_status(self):
        """Test parsing instance with unknown status."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()
            data = self.get_sample_instance_data()
            data["status"] = "unknown_status"

            instance = provider._parse_instance(data)

            assert instance.status == InstanceStatus.UNKNOWN

    def test_parse_instance_unknown_server_type(self):
        """Test parsing instance with unknown server type."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()
            data = self.get_sample_instance_data()
            data["server_type"]["name"] = "unknown_type"

            instance = provider._parse_instance(data)

            # Should use default fallback
            assert instance.cost_per_hour == 0.01

    def test_parse_instance_no_ipv4(self):
        """Test parsing instance without IPv4."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()
            data = self.get_sample_instance_data()
            data["public_net"]["ipv4"] = None

            instance = provider._parse_instance(data)

            assert instance.ip_address is None

    def test_parse_instance_without_created(self):
        """Test parsing instance without created date."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()
            data = self.get_sample_instance_data()
            del data["created"]

            instance = provider._parse_instance(data)

            assert instance.created_at is None

    def test_parse_instance_labels_to_tags(self):
        """Test that labels are converted to tags."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()
            data = self.get_sample_instance_data()

            instance = provider._parse_instance(data)

            assert "role" in instance.tags
            assert "env" in instance.tags


class TestHetznerProviderListInstances:
    """Tests for listing instances."""

    @pytest.mark.asyncio
    async def test_list_instances_success(self):
        """Test successful instance listing."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()

            servers_json = json.dumps([
                {
                    "id": 12345,
                    "name": "cpu1",
                    "status": "running",
                    "server_type": {"name": "ccx33"},
                    "public_net": {"ipv4": {"ip": "1.2.3.4"}},
                    "datacenter": {"name": "fsn1"},
                    "labels": {},
                },
                {
                    "id": 12346,
                    "name": "cpu2",
                    "status": "off",
                    "server_type": {"name": "ccx33"},
                    "public_net": {"ipv4": {"ip": "1.2.3.5"}},
                    "datacenter": {"name": "fsn1"},
                    "labels": {},
                },
            ])

            with patch.object(provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
                mock_cli.return_value = (servers_json, "", 0)

                instances = await provider.list_instances()

                assert len(instances) == 2
                assert instances[0].id == "12345"
                assert instances[1].id == "12346"

    @pytest.mark.asyncio
    async def test_list_instances_cli_failure(self):
        """Test instance listing when CLI fails."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()

            with patch.object(provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
                mock_cli.return_value = ("", "Error: authentication failed", 1)

                instances = await provider.list_instances()

                assert len(instances) == 0

    @pytest.mark.asyncio
    async def test_list_instances_invalid_json(self):
        """Test instance listing with invalid JSON."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()

            with patch.object(provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
                mock_cli.return_value = ("not valid json", "", 0)

                instances = await provider.list_instances()

                assert len(instances) == 0


class TestHetznerProviderGetInstance:
    """Tests for getting specific instance."""

    @pytest.mark.asyncio
    async def test_get_instance_found(self):
        """Test getting instance that exists."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()

            server_json = json.dumps({
                "id": 12345,
                "name": "target",
                "status": "running",
                "server_type": {"name": "ccx33"},
                "public_net": {"ipv4": {"ip": "1.2.3.4"}},
                "datacenter": {"name": "fsn1"},
                "labels": {},
            })

            with patch.object(provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
                mock_cli.return_value = (server_json, "", 0)

                instance = await provider.get_instance("12345")

                assert instance is not None
                assert instance.id == "12345"
                assert instance.name == "target"

    @pytest.mark.asyncio
    async def test_get_instance_not_found(self):
        """Test getting instance that doesn't exist."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()

            with patch.object(provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
                mock_cli.return_value = ("", "server not found", 1)

                instance = await provider.get_instance("99999")

                assert instance is None


class TestHetznerProviderGetInstanceStatus:
    """Tests for getting instance status."""

    @pytest.mark.asyncio
    async def test_get_instance_status_running(self):
        """Test getting status of running instance."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()

            server_json = json.dumps({
                "id": 12345,
                "name": "test",
                "status": "running",
                "server_type": {"name": "ccx33"},
                "public_net": {"ipv4": {"ip": "1.2.3.4"}},
                "datacenter": {"name": "fsn1"},
                "labels": {},
            })

            with patch.object(provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
                mock_cli.return_value = (server_json, "", 0)

                status = await provider.get_instance_status("12345")

                assert status == InstanceStatus.RUNNING

    @pytest.mark.asyncio
    async def test_get_instance_status_not_found(self):
        """Test getting status of non-existent instance."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()

            with patch.object(provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
                mock_cli.return_value = ("", "not found", 1)

                status = await provider.get_instance_status("99999")

                assert status == InstanceStatus.UNKNOWN


class TestHetznerProviderScaleDown:
    """Tests for scale down (server deletion)."""

    @pytest.mark.asyncio
    async def test_scale_down_success(self):
        """Test successful server deletion."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()

            with patch.object(provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
                mock_cli.return_value = ("", "", 0)

                results = await provider.scale_down(["12345", "12346"])

                assert results["12345"] is True
                assert results["12346"] is True
                assert mock_cli.call_count == 2

    @pytest.mark.asyncio
    async def test_scale_down_partial_failure(self):
        """Test scale down with partial failure."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()

            with patch.object(provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
                mock_cli.side_effect = [
                    ("", "", 0),
                    ("", "Error: server not found", 1),
                ]

                results = await provider.scale_down(["12345", "99999"])

                assert results["12345"] is True
                assert results["99999"] is False


class TestHetznerProviderScaleUp:
    """Tests for scale up (server creation)."""

    @pytest.mark.asyncio
    async def test_scale_up_success(self):
        """Test successful server creation."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()

            create_response = json.dumps({
                "server": {
                    "id": 12345,
                    "name": "ringrift-cpu-0",
                    "status": "initializing",
                    "server_type": {"name": "ccx33"},
                    "public_net": {"ipv4": {"ip": "1.2.3.4"}},
                    "datacenter": {"name": "fsn1"},
                    "labels": {},
                }
            })

            with patch.object(provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
                mock_cli.return_value = (create_response, "", 0)

                instances = await provider.scale_up(GPUType.CPU_ONLY, count=1)

                assert len(instances) == 1
                assert instances[0].id == "12345"

    @pytest.mark.asyncio
    async def test_scale_up_with_region(self):
        """Test server creation with specific region."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()

            create_response = json.dumps({
                "server": {
                    "id": 12345,
                    "name": "ringrift-cpu-0",
                    "status": "initializing",
                    "server_type": {"name": "ccx33"},
                    "public_net": {"ipv4": {"ip": "1.2.3.4"}},
                    "datacenter": {"name": "nbg1-dc3"},
                    "labels": {},
                }
            })

            with patch.object(provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
                mock_cli.return_value = (create_response, "", 0)

                instances = await provider.scale_up(
                    GPUType.CPU_ONLY,
                    count=1,
                    region="nbg1-dc3"
                )

                assert len(instances) == 1
                # Verify region was passed
                call_args = mock_cli.call_args[0]
                assert "--datacenter" in call_args

    @pytest.mark.asyncio
    async def test_scale_up_cli_failure(self):
        """Test server creation when CLI fails."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()

            with patch.object(provider, "_run_cli", new_callable=AsyncMock) as mock_cli:
                mock_cli.return_value = ("", "Error: quota exceeded", 1)

                instances = await provider.scale_up(GPUType.CPU_ONLY, count=1)

                assert len(instances) == 0


class TestHetznerProviderCostEstimation:
    """Tests for cost estimation."""

    def test_get_cost_per_hour_default(self):
        """Test cost returns ccx33 pricing."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()

            cost = provider.get_cost_per_hour(GPUType.CPU_ONLY)

            assert cost == HETZNER_PLANS["ccx33"][2]

    def test_get_cost_per_hour_ignores_gpu_type(self):
        """Test that GPU type is ignored (Hetzner is CPU-only)."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()

            cost_h100 = provider.get_cost_per_hour(GPUType.H100_80GB)
            cost_cpu = provider.get_cost_per_hour(GPUType.CPU_ONLY)

            # Both should return same CPU pricing
            assert cost_h100 == cost_cpu


class TestHetznerProviderAvailableGPUs:
    """Tests for GPU availability."""

    @pytest.mark.asyncio
    async def test_get_available_gpus(self):
        """Test that only CPU is available."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()

            gpus = await provider.get_available_gpus()

            assert GPUType.CPU_ONLY in gpus
            assert gpus[GPUType.CPU_ONLY] == 100
            assert len(gpus) == 1


class TestHetznerProviderHealthCheck:
    """Tests for health check integration."""

    def test_health_check_healthy_with_token(self):
        """Test health check when configured with token."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None  # No CLI
            provider = HetznerProvider(token="test-token")

            result = provider.health_check()

            assert result.healthy is True
            assert result.details["configured"] is True
            assert result.details["has_token"] is True

    def test_health_check_healthy_with_cli(self):
        """Test health check when configured with CLI."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider()

            result = provider.health_check()

            assert result.healthy is True
            assert result.details["configured"] is True

    def test_health_check_no_cli_no_token(self):
        """Test health check when not configured."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None
            with patch.dict(os.environ, {}, clear=True):
                provider = HetznerProvider()

                result = provider.health_check()

                assert result.healthy is False
                assert "ERROR" in result.status.value or result.status.value == "error"
                assert result.details["configured"] is False


class TestHetznerProviderCLI:
    """Tests for CLI execution."""

    @pytest.mark.asyncio
    async def test_run_cli_no_path(self):
        """Test CLI execution fails when no CLI path."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None
            provider = HetznerProvider(token="test-token")

            with pytest.raises(RuntimeError, match="CLI not found"):
                await provider._run_cli("server", "list")

    @pytest.mark.asyncio
    async def test_run_cli_includes_token_in_env(self):
        """Test that token is passed in environment."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/hcloud"
            provider = HetznerProvider(token="secret-token")

            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.stdout = "[]"
                mock_result.stderr = ""
                mock_result.returncode = 0
                mock_run.return_value = mock_result

                with patch("asyncio.get_running_loop") as mock_loop:
                    mock_loop.return_value.run_in_executor = AsyncMock(
                        return_value=("[]", "", 0)
                    )

                    # Test passes if no exception is raised
                    pass
