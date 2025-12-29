"""Tests for Lambda Labs cloud provider integration.

Tests cover:
- Configuration loading from env
- GPU type mappings
- Instance status parsing
- API response handling
- Error handling

Created: Dec 29, 2025
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.providers.lambda_provider import (
    LAMBDA_COSTS,
    LAMBDA_GPU_TYPES,
    LAMBDA_INSTANCE_TYPES,
    LambdaConfig,
    LambdaProvider,
)
from app.coordination.providers.base import GPUType, InstanceStatus, ProviderType


class TestLambdaConfig:
    """Tests for LambdaConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LambdaConfig()
        assert config.api_key is None
        assert config.api_base == "https://cloud.lambdalabs.com/api/v1"
        assert config.timeout_seconds == 30.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LambdaConfig(
            api_key="test-key-123",
            api_base="https://custom.api.lambda/v1",
            timeout_seconds=60.0,
        )
        assert config.api_key == "test-key-123"
        assert config.api_base == "https://custom.api.lambda/v1"
        assert config.timeout_seconds == 60.0

    def test_from_env_with_env_var(self):
        """Test loading API key from environment variable."""
        with patch.dict(os.environ, {"LAMBDA_API_KEY": "env-key-456"}):
            config = LambdaConfig.from_env()
            assert config.api_key == "env-key-456"

    def test_from_env_no_key(self):
        """Test from_env when no key is set."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear LAMBDA_API_KEY if it exists
            if "LAMBDA_API_KEY" in os.environ:
                del os.environ["LAMBDA_API_KEY"]
            config = LambdaConfig.from_env()
            assert config.api_key is None


class TestGPUTypeMappings:
    """Tests for GPU type mappings."""

    def test_gh200_mapping(self):
        """Test GH200 GPU mapping."""
        assert LAMBDA_INSTANCE_TYPES[GPUType.GH200_96GB] == "gpu_1x_gh200"

    def test_h100_mapping(self):
        """Test H100 GPU mapping."""
        assert LAMBDA_INSTANCE_TYPES[GPUType.H100_80GB] == "gpu_1x_h100_sxm5"

    def test_a100_mappings(self):
        """Test A100 GPU mappings."""
        assert LAMBDA_INSTANCE_TYPES[GPUType.A100_80GB] == "gpu_1x_a100_sxm4"
        assert LAMBDA_INSTANCE_TYPES[GPUType.A100_40GB] == "gpu_1x_a100"

    def test_consumer_gpu_mappings(self):
        """Test consumer GPU mappings."""
        assert LAMBDA_INSTANCE_TYPES[GPUType.RTX_4090] == "gpu_1x_rtx4090"
        assert LAMBDA_INSTANCE_TYPES[GPUType.RTX_3090] == "gpu_1x_rtx3090"

    def test_a10_mapping(self):
        """Test A10 GPU mapping."""
        assert LAMBDA_INSTANCE_TYPES[GPUType.A10] == "gpu_1x_a10"

    def test_reverse_mappings(self):
        """Test reverse mappings (instance type -> GPU type)."""
        assert LAMBDA_GPU_TYPES["gpu_1x_gh200"] == GPUType.GH200_96GB
        assert LAMBDA_GPU_TYPES["gpu_1x_h100_sxm5"] == GPUType.H100_80GB
        assert LAMBDA_GPU_TYPES["gpu_1x_a100_sxm4"] == GPUType.A100_80GB


class TestLambdaCosts:
    """Tests for cost mappings."""

    def test_gh200_cost(self):
        """Test GH200 cost."""
        assert LAMBDA_COSTS[GPUType.GH200_96GB] == 2.49

    def test_h100_cost(self):
        """Test H100 cost."""
        assert LAMBDA_COSTS[GPUType.H100_80GB] == 2.49

    def test_a100_costs(self):
        """Test A100 costs."""
        assert LAMBDA_COSTS[GPUType.A100_80GB] == 1.29
        assert LAMBDA_COSTS[GPUType.A100_40GB] == 1.10

    def test_consumer_costs(self):
        """Test consumer GPU costs."""
        assert LAMBDA_COSTS[GPUType.RTX_4090] == 0.80
        assert LAMBDA_COSTS[GPUType.RTX_3090] == 0.50

    def test_all_costs_positive(self):
        """Test all costs are positive."""
        for gpu_type, cost in LAMBDA_COSTS.items():
            assert cost > 0, f"Cost for {gpu_type} should be positive"


class TestLambdaProvider:
    """Tests for LambdaProvider class."""

    def test_provider_type(self):
        """Test provider type property."""
        provider = LambdaProvider(LambdaConfig(api_key="test"))
        assert provider.provider_type == ProviderType.LAMBDA

    def test_name(self):
        """Test provider name property."""
        provider = LambdaProvider(LambdaConfig(api_key="test"))
        assert provider.name == "Lambda Labs"

    def test_is_configured_with_key(self):
        """Test is_configured with API key."""
        provider = LambdaProvider(LambdaConfig(api_key="test-key"))
        assert provider.is_configured() is True

    def test_is_configured_without_key(self):
        """Test is_configured without API key."""
        provider = LambdaProvider(LambdaConfig(api_key=None))
        assert provider.is_configured() is False

    def test_is_configured_empty_key(self):
        """Test is_configured with empty string API key."""
        provider = LambdaProvider(LambdaConfig(api_key=""))
        assert provider.is_configured() is False

    def test_default_config_from_env(self):
        """Test that provider uses config from env by default."""
        with patch.dict(os.environ, {"LAMBDA_API_KEY": "auto-loaded-key"}):
            provider = LambdaProvider()
            assert provider.config.api_key == "auto-loaded-key"


class TestLambdaProviderAsync:
    """Async tests for LambdaProvider."""

    @pytest.fixture
    def provider(self):
        """Create a configured provider."""
        return LambdaProvider(LambdaConfig(api_key="test-key"))

    @pytest.mark.asyncio
    async def test_list_instances_not_configured(self):
        """Test list_instances when provider is not configured."""
        provider = LambdaProvider(LambdaConfig(api_key=None))
        instances = await provider.list_instances()
        assert instances == []

    @pytest.mark.asyncio
    async def test_get_instance_not_configured(self):
        """Test get_instance when provider is not configured."""
        provider = LambdaProvider(LambdaConfig(api_key=None))
        instance = await provider.get_instance("inst-123")
        assert instance is None

    @pytest.mark.asyncio
    async def test_scale_down_not_configured(self):
        """Test scale_down when provider is not configured."""
        provider = LambdaProvider(LambdaConfig(api_key=None))
        result = await provider.scale_down(["inst-123"])
        assert result == {"inst-123": False}

    @pytest.mark.asyncio
    async def test_get_instance_status_not_configured(self):
        """Test get_instance_status when provider is not configured."""
        provider = LambdaProvider(LambdaConfig(api_key=None))
        status = await provider.get_instance_status("inst-123")
        assert status == InstanceStatus.UNKNOWN


class TestInstanceStatusMapping:
    """Tests for instance status mapping."""

    def test_active_status(self):
        """Test mapping of active status."""
        provider = LambdaProvider(LambdaConfig(api_key="test"))
        assert provider._parse_instance_status("active") == InstanceStatus.RUNNING

    def test_booting_status(self):
        """Test mapping of booting status."""
        provider = LambdaProvider(LambdaConfig(api_key="test"))
        assert provider._parse_instance_status("booting") == InstanceStatus.STARTING

    def test_unhealthy_status(self):
        """Test mapping of unhealthy status."""
        provider = LambdaProvider(LambdaConfig(api_key="test"))
        assert provider._parse_instance_status("unhealthy") == InstanceStatus.ERROR

    def test_terminating_status(self):
        """Test mapping of terminating status."""
        provider = LambdaProvider(LambdaConfig(api_key="test"))
        assert provider._parse_instance_status("terminating") == InstanceStatus.STOPPING

    def test_terminated_status(self):
        """Test mapping of terminated status."""
        provider = LambdaProvider(LambdaConfig(api_key="test"))
        assert provider._parse_instance_status("terminated") == InstanceStatus.TERMINATED

    def test_unknown_status(self):
        """Test mapping of unknown status."""
        provider = LambdaProvider(LambdaConfig(api_key="test"))
        assert provider._parse_instance_status("something_else") == InstanceStatus.UNKNOWN


class TestGPUTypeParsing:
    """Tests for GPU type parsing."""

    def test_parse_gh200(self):
        """Test parsing GH200 instance type."""
        provider = LambdaProvider(LambdaConfig(api_key="test"))
        assert provider._parse_gpu_type("gpu_1x_gh200") == GPUType.GH200_96GB

    def test_parse_h100(self):
        """Test parsing H100 instance type."""
        provider = LambdaProvider(LambdaConfig(api_key="test"))
        assert provider._parse_gpu_type("gpu_1x_h100_sxm5") == GPUType.H100_80GB

    def test_parse_a100_80gb(self):
        """Test parsing A100 80GB instance type."""
        provider = LambdaProvider(LambdaConfig(api_key="test"))
        assert provider._parse_gpu_type("gpu_1x_a100_sxm4") == GPUType.A100_80GB

    def test_parse_unknown(self):
        """Test parsing unknown instance type."""
        provider = LambdaProvider(LambdaConfig(api_key="test"))
        assert provider._parse_gpu_type("unknown_gpu") == GPUType.UNKNOWN


class TestCostLookup:
    """Tests for cost lookup methods."""

    def test_get_cost_per_hour_gh200(self):
        """Test cost lookup for GH200."""
        provider = LambdaProvider(LambdaConfig(api_key="test"))
        assert provider.get_cost_per_hour(GPUType.GH200_96GB) == 2.49

    def test_get_cost_per_hour_h100(self):
        """Test cost lookup for H100."""
        provider = LambdaProvider(LambdaConfig(api_key="test"))
        assert provider.get_cost_per_hour(GPUType.H100_80GB) == 2.49

    def test_get_cost_per_hour_unknown(self):
        """Test cost lookup for unknown GPU type."""
        provider = LambdaProvider(LambdaConfig(api_key="test"))
        # Unknown GPU should return 0.0 or default
        cost = provider.get_cost_per_hour(GPUType.UNKNOWN)
        assert cost >= 0.0
