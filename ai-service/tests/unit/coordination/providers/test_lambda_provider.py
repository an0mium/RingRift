"""Tests for LambdaProvider - Lambda Labs cloud provider implementation.

Tests cover:
- Provider initialization and configuration
- GPU type parsing and mapping
- Instance parsing from API response
- API request handling (mocked)
- Instance listing and filtering
- Scale up/down operations
- Cost estimation
- Region selection

December 2025 - P0 test coverage for Lambda Labs (6 GH200 nodes).
"""

import asyncio
import os
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from app.coordination.providers.lambda_provider import (
    LambdaProvider,
    LambdaConfig,
    LAMBDA_INSTANCE_TYPES,
    LAMBDA_GPU_TYPES,
    LAMBDA_COSTS,
    get_lambda_provider,
    reset_lambda_provider,
)
from app.coordination.providers.base import (
    GPUType,
    InstanceStatus,
    ProviderType,
)


class TestLambdaConfig:
    """Tests for LambdaConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LambdaConfig()

        assert config.api_key is None
        assert config.api_base == "https://cloud.lambdalabs.com/api/v1"
        assert config.timeout_seconds == 30.0

    def test_config_with_api_key(self):
        """Test configuration with API key."""
        config = LambdaConfig(api_key="test-key-123")

        assert config.api_key == "test-key-123"

    def test_from_env_with_key(self):
        """Test loading config from environment."""
        with patch.dict(os.environ, {"LAMBDA_API_KEY": "env-key-456"}):
            config = LambdaConfig.from_env()

            assert config.api_key == "env-key-456"

    def test_from_env_without_key(self):
        """Test loading config when env var missing."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove LAMBDA_API_KEY if present
            os.environ.pop("LAMBDA_API_KEY", None)
            config = LambdaConfig.from_env()

            assert config.api_key is None


class TestLambdaProviderInit:
    """Tests for LambdaProvider initialization."""

    def test_init_with_config(self):
        """Test initialization with explicit config."""
        config = LambdaConfig(api_key="test-key")
        provider = LambdaProvider(config=config)

        assert provider.config.api_key == "test-key"

    def test_init_default_config(self):
        """Test initialization with default config from env."""
        with patch.dict(os.environ, {"LAMBDA_API_KEY": "env-key"}):
            provider = LambdaProvider()

            assert provider.config.api_key == "env-key"

    def test_session_initially_none(self):
        """Test that HTTP session is initially None."""
        provider = LambdaProvider(config=LambdaConfig())

        assert provider._session is None


class TestLambdaProviderProperties:
    """Tests for LambdaProvider properties."""

    def test_provider_type(self):
        """Test provider_type property."""
        provider = LambdaProvider(config=LambdaConfig())

        assert provider.provider_type == ProviderType.LAMBDA

    def test_name(self):
        """Test name property."""
        provider = LambdaProvider(config=LambdaConfig())

        assert provider.name == "Lambda Labs"


class TestLambdaProviderConfiguration:
    """Tests for LambdaProvider configuration checking."""

    def test_is_configured_true(self):
        """Test is_configured returns True when API key exists."""
        config = LambdaConfig(api_key="test-key")
        provider = LambdaProvider(config=config)

        assert provider.is_configured() is True

    def test_is_configured_false(self):
        """Test is_configured returns False when no API key."""
        config = LambdaConfig(api_key=None)
        provider = LambdaProvider(config=config)

        assert provider.is_configured() is False

    def test_is_configured_empty_string(self):
        """Test is_configured returns False for empty string API key."""
        config = LambdaConfig(api_key="")
        provider = LambdaProvider(config=config)

        assert provider.is_configured() is False


class TestLambdaProviderStatusParsing:
    """Tests for instance status parsing."""

    def test_parse_active_status(self):
        """Test parsing 'active' status."""
        provider = LambdaProvider(config=LambdaConfig())

        assert provider._parse_instance_status("active") == InstanceStatus.RUNNING

    def test_parse_booting_status(self):
        """Test parsing 'booting' status."""
        provider = LambdaProvider(config=LambdaConfig())

        assert provider._parse_instance_status("booting") == InstanceStatus.STARTING

    def test_parse_unhealthy_status(self):
        """Test parsing 'unhealthy' status."""
        provider = LambdaProvider(config=LambdaConfig())

        assert provider._parse_instance_status("unhealthy") == InstanceStatus.ERROR

    def test_parse_terminating_status(self):
        """Test parsing 'terminating' status."""
        provider = LambdaProvider(config=LambdaConfig())

        assert provider._parse_instance_status("terminating") == InstanceStatus.STOPPING

    def test_parse_terminated_status(self):
        """Test parsing 'terminated' status."""
        provider = LambdaProvider(config=LambdaConfig())

        assert provider._parse_instance_status("terminated") == InstanceStatus.TERMINATED

    def test_parse_unknown_status(self):
        """Test parsing unknown status."""
        provider = LambdaProvider(config=LambdaConfig())

        assert provider._parse_instance_status("unknown") == InstanceStatus.UNKNOWN
        assert provider._parse_instance_status("foobar") == InstanceStatus.UNKNOWN


class TestLambdaProviderGPUParsing:
    """Tests for GPU type parsing."""

    def test_parse_gh200(self):
        """Test parsing GH200 instance type."""
        provider = LambdaProvider(config=LambdaConfig())

        assert provider._parse_gpu_type("gpu_1x_gh200") == GPUType.GH200_96GB

    def test_parse_h100(self):
        """Test parsing H100 instance type."""
        provider = LambdaProvider(config=LambdaConfig())

        assert provider._parse_gpu_type("gpu_1x_h100_sxm5") == GPUType.H100_80GB

    def test_parse_a100_80gb(self):
        """Test parsing A100 80GB instance type."""
        provider = LambdaProvider(config=LambdaConfig())

        assert provider._parse_gpu_type("gpu_1x_a100_sxm4") == GPUType.A100_80GB

    def test_parse_a100_40gb(self):
        """Test parsing A100 40GB instance type."""
        provider = LambdaProvider(config=LambdaConfig())

        assert provider._parse_gpu_type("gpu_1x_a100") == GPUType.A100_40GB

    def test_parse_rtx_4090(self):
        """Test parsing RTX 4090 instance type."""
        provider = LambdaProvider(config=LambdaConfig())

        assert provider._parse_gpu_type("gpu_1x_rtx4090") == GPUType.RTX_4090

    def test_parse_unknown_type(self):
        """Test parsing unknown instance type."""
        provider = LambdaProvider(config=LambdaConfig())

        assert provider._parse_gpu_type("unknown_gpu") == GPUType.UNKNOWN


class TestLambdaProviderCosts:
    """Tests for cost estimation."""

    def test_gh200_cost(self):
        """Test GH200 cost per hour."""
        provider = LambdaProvider(config=LambdaConfig())

        assert provider.get_cost_per_hour(GPUType.GH200_96GB) == 2.49

    def test_h100_cost(self):
        """Test H100 cost per hour."""
        provider = LambdaProvider(config=LambdaConfig())

        assert provider.get_cost_per_hour(GPUType.H100_80GB) == 2.49

    def test_a100_80gb_cost(self):
        """Test A100 80GB cost per hour."""
        provider = LambdaProvider(config=LambdaConfig())

        assert provider.get_cost_per_hour(GPUType.A100_80GB) == 1.29

    def test_unknown_cost(self):
        """Test unknown GPU returns 0 cost."""
        provider = LambdaProvider(config=LambdaConfig())

        assert provider.get_cost_per_hour(GPUType.UNKNOWN) == 0.0


class TestLambdaProviderGPUMemory:
    """Tests for GPU memory lookup."""

    def test_gh200_memory(self):
        """Test GH200 memory."""
        provider = LambdaProvider(config=LambdaConfig())

        assert provider._get_gpu_memory(GPUType.GH200_96GB) == 96.0

    def test_h100_memory(self):
        """Test H100 memory."""
        provider = LambdaProvider(config=LambdaConfig())

        assert provider._get_gpu_memory(GPUType.H100_80GB) == 80.0

    def test_a100_80gb_memory(self):
        """Test A100 80GB memory."""
        provider = LambdaProvider(config=LambdaConfig())

        assert provider._get_gpu_memory(GPUType.A100_80GB) == 80.0

    def test_rtx_4090_memory(self):
        """Test RTX 4090 memory."""
        provider = LambdaProvider(config=LambdaConfig())

        assert provider._get_gpu_memory(GPUType.RTX_4090) == 24.0

    def test_unknown_memory(self):
        """Test unknown GPU returns 0 memory."""
        provider = LambdaProvider(config=LambdaConfig())

        assert provider._get_gpu_memory(GPUType.UNKNOWN) == 0.0


class TestLambdaProviderListInstances:
    """Tests for list_instances method."""

    @pytest.mark.asyncio
    async def test_list_instances_not_configured(self):
        """Test list_instances returns empty when not configured."""
        config = LambdaConfig(api_key=None)
        provider = LambdaProvider(config=config)

        instances = await provider.list_instances()

        assert instances == []

    @pytest.mark.asyncio
    async def test_list_instances_success(self):
        """Test successful instance listing."""
        config = LambdaConfig(api_key="test-key")
        provider = LambdaProvider(config=config)

        mock_response = {
            "data": [
                {
                    "id": "inst-123",
                    "name": "ringrift-gh200",
                    "status": "active",
                    "ip": "192.168.1.100",
                    "instance_type": {"name": "gpu_1x_gh200"},
                    "region": {"name": "us-west-1"},
                },
                {
                    "id": "inst-456",
                    "name": "ringrift-h100",
                    "status": "booting",
                    "ip": None,
                    "instance_type": {"name": "gpu_1x_h100_sxm5"},
                    "region": {"name": "us-east-1"},
                },
            ]
        }

        with patch.object(provider, "_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_response

            instances = await provider.list_instances()

            assert len(instances) == 2
            assert instances[0].id == "inst-123"
            assert instances[0].name == "ringrift-gh200"
            assert instances[0].status == InstanceStatus.RUNNING
            assert instances[0].gpu_type == GPUType.GH200_96GB
            assert instances[0].ip_address == "192.168.1.100"

            assert instances[1].id == "inst-456"
            assert instances[1].status == InstanceStatus.STARTING
            assert instances[1].gpu_type == GPUType.H100_80GB

    @pytest.mark.asyncio
    async def test_list_instances_api_error(self):
        """Test list_instances handles API errors gracefully."""
        config = LambdaConfig(api_key="test-key")
        provider = LambdaProvider(config=config)

        with patch.object(provider, "_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.side_effect = Exception("API error")

            instances = await provider.list_instances()

            assert instances == []


class TestLambdaProviderGetInstance:
    """Tests for get_instance method."""

    @pytest.mark.asyncio
    async def test_get_instance_not_configured(self):
        """Test get_instance returns None when not configured."""
        config = LambdaConfig(api_key=None)
        provider = LambdaProvider(config=config)

        instance = await provider.get_instance("inst-123")

        assert instance is None

    @pytest.mark.asyncio
    async def test_get_instance_success(self):
        """Test successful instance retrieval."""
        config = LambdaConfig(api_key="test-key")
        provider = LambdaProvider(config=config)

        mock_response = {
            "data": {
                "id": "inst-123",
                "name": "ringrift-gh200",
                "status": "active",
                "ip": "192.168.1.100",
                "instance_type": {"name": "gpu_1x_gh200"},
                "region": {"name": "us-west-1"},
            }
        }

        with patch.object(provider, "_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_response

            instance = await provider.get_instance("inst-123")

            assert instance is not None
            assert instance.id == "inst-123"
            assert instance.gpu_type == GPUType.GH200_96GB
            mock_api.assert_called_once_with("GET", "/instances/inst-123")

    @pytest.mark.asyncio
    async def test_get_instance_not_found(self):
        """Test get_instance when instance doesn't exist."""
        config = LambdaConfig(api_key="test-key")
        provider = LambdaProvider(config=config)

        with patch.object(provider, "_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"data": None}

            instance = await provider.get_instance("inst-999")

            assert instance is None


class TestLambdaProviderGetInstanceStatus:
    """Tests for get_instance_status method."""

    @pytest.mark.asyncio
    async def test_get_instance_status_running(self):
        """Test getting status of running instance."""
        config = LambdaConfig(api_key="test-key")
        provider = LambdaProvider(config=config)

        mock_response = {
            "data": {
                "id": "inst-123",
                "status": "active",
                "instance_type": {"name": "gpu_1x_gh200"},
            }
        }

        with patch.object(provider, "_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_response

            status = await provider.get_instance_status("inst-123")

            assert status == InstanceStatus.RUNNING

    @pytest.mark.asyncio
    async def test_get_instance_status_not_found(self):
        """Test getting status of non-existent instance."""
        config = LambdaConfig(api_key="test-key")
        provider = LambdaProvider(config=config)

        with patch.object(provider, "_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"data": None}

            status = await provider.get_instance_status("inst-999")

            assert status == InstanceStatus.UNKNOWN


class TestLambdaProviderScaleDown:
    """Tests for scale_down method."""

    @pytest.mark.asyncio
    async def test_scale_down_not_configured(self):
        """Test scale_down when not configured."""
        config = LambdaConfig(api_key=None)
        provider = LambdaProvider(config=config)

        results = await provider.scale_down(["inst-123", "inst-456"])

        assert results == {"inst-123": False, "inst-456": False}

    @pytest.mark.asyncio
    async def test_scale_down_success(self):
        """Test successful instance termination."""
        config = LambdaConfig(api_key="test-key")
        provider = LambdaProvider(config=config)

        mock_response = {
            "data": {
                "terminated_instances": [
                    {"id": "inst-123"},
                    {"id": "inst-456"},
                ]
            }
        }

        with patch.object(provider, "_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_response

            results = await provider.scale_down(["inst-123", "inst-456"])

            assert results["inst-123"] is True
            assert results["inst-456"] is True

    @pytest.mark.asyncio
    async def test_scale_down_partial_failure(self):
        """Test scale_down with partial failure."""
        config = LambdaConfig(api_key="test-key")
        provider = LambdaProvider(config=config)

        mock_response = {
            "data": {
                "terminated_instances": [{"id": "inst-123"}]
            }
        }

        with patch.object(provider, "_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_response

            results = await provider.scale_down(["inst-123", "inst-456"])

            assert results["inst-123"] is True
            assert results["inst-456"] is False


class TestLambdaProviderReboot:
    """Tests for reboot_instance method."""

    @pytest.mark.asyncio
    async def test_reboot_not_configured(self):
        """Test reboot when not configured."""
        config = LambdaConfig(api_key=None)
        provider = LambdaProvider(config=config)

        result = await provider.reboot_instance("inst-123")

        assert result is False

    @pytest.mark.asyncio
    async def test_reboot_success(self):
        """Test successful reboot."""
        config = LambdaConfig(api_key="test-key")
        provider = LambdaProvider(config=config)

        with patch.object(provider, "_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {}

            result = await provider.reboot_instance("inst-123")

            assert result is True
            mock_api.assert_called_once()

    @pytest.mark.asyncio
    async def test_reboot_failure(self):
        """Test reboot failure."""
        config = LambdaConfig(api_key="test-key")
        provider = LambdaProvider(config=config)

        with patch.object(provider, "_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.side_effect = Exception("Reboot failed")

            result = await provider.reboot_instance("inst-123")

            assert result is False


class TestLambdaProviderSingleton:
    """Tests for singleton pattern."""

    def test_get_lambda_provider_singleton(self):
        """Test that get_lambda_provider returns same instance."""
        reset_lambda_provider()

        provider1 = get_lambda_provider()
        provider2 = get_lambda_provider()

        assert provider1 is provider2

    def test_reset_lambda_provider(self):
        """Test that reset creates new instance."""
        provider1 = get_lambda_provider()
        reset_lambda_provider()
        provider2 = get_lambda_provider()

        assert provider1 is not provider2


class TestLambdaInstanceTypeMappings:
    """Tests for instance type mappings."""

    def test_gh200_mapping(self):
        """Test GH200 instance type mapping."""
        assert LAMBDA_INSTANCE_TYPES[GPUType.GH200_96GB] == "gpu_1x_gh200"
        assert LAMBDA_GPU_TYPES["gpu_1x_gh200"] == GPUType.GH200_96GB

    def test_h100_mapping(self):
        """Test H100 instance type mapping."""
        assert LAMBDA_INSTANCE_TYPES[GPUType.H100_80GB] == "gpu_1x_h100_sxm5"
        assert LAMBDA_GPU_TYPES["gpu_1x_h100_sxm5"] == GPUType.H100_80GB

    def test_reverse_mapping_consistency(self):
        """Test that forward and reverse mappings are consistent."""
        for gpu_type, instance_type in LAMBDA_INSTANCE_TYPES.items():
            assert LAMBDA_GPU_TYPES[instance_type] == gpu_type


class TestLambdaCostMappings:
    """Tests for cost mappings."""

    def test_all_mapped_types_have_costs(self):
        """Test that all mapped GPU types have costs defined."""
        for gpu_type in LAMBDA_INSTANCE_TYPES.keys():
            assert gpu_type in LAMBDA_COSTS, f"Missing cost for {gpu_type}"

    def test_costs_are_positive(self):
        """Test that all costs are positive."""
        for gpu_type, cost in LAMBDA_COSTS.items():
            assert cost > 0, f"Invalid cost for {gpu_type}: {cost}"


class TestLambdaProviderClose:
    """Tests for session cleanup."""

    @pytest.mark.asyncio
    async def test_close_with_session(self):
        """Test close when session exists."""
        config = LambdaConfig(api_key="test-key")
        provider = LambdaProvider(config=config)

        # Create a mock session
        mock_session = AsyncMock()
        mock_session.closed = False
        provider._session = mock_session

        await provider.close()

        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_session(self):
        """Test close when no session exists."""
        config = LambdaConfig(api_key="test-key")
        provider = LambdaProvider(config=config)

        # Should not raise
        await provider.close()
