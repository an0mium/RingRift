"""Tests for cloud provider base classes and types.

Created: December 2025
Tests for: app/coordination/providers/base.py
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.providers.base import (
    CloudProvider,
    GPUType,
    Instance,
    InstanceStatus,
    ProviderType,
)


class TestProviderType:
    """Tests for ProviderType enum."""

    def test_all_provider_types_exist(self):
        """Verify all expected provider types are defined."""
        assert ProviderType.LAMBDA
        assert ProviderType.VULTR
        assert ProviderType.VAST
        assert ProviderType.HETZNER
        assert ProviderType.RUNPOD

    def test_provider_types_are_unique(self):
        """Verify all provider types have unique values."""
        values = [p.value for p in ProviderType]
        assert len(values) == len(set(values))

    def test_provider_type_count(self):
        """Verify expected number of provider types."""
        assert len(ProviderType) == 5


class TestGPUType:
    """Tests for GPUType enum."""

    def test_all_consumer_gpu_types(self):
        """Verify consumer GPU types."""
        assert GPUType.RTX_3090
        assert GPUType.RTX_4090
        assert GPUType.RTX_5090

    def test_all_datacenter_gpu_types(self):
        """Verify data center GPU types."""
        assert GPUType.A10
        assert GPUType.A100_40GB
        assert GPUType.A100_80GB
        assert GPUType.H100_80GB
        assert GPUType.GH200_96GB

    def test_cpu_only_type(self):
        """Verify CPU-only type exists."""
        assert GPUType.CPU_ONLY
        assert GPUType.CPU_ONLY.value == "cpu"

    def test_unknown_type(self):
        """Verify unknown type exists."""
        assert GPUType.UNKNOWN
        assert GPUType.UNKNOWN.value == "unknown"

    def test_from_string_gh200(self):
        """Test GPU type parsing for GH200."""
        assert GPUType.from_string("gh200") == GPUType.GH200_96GB
        assert GPUType.from_string("GH200-96GB") == GPUType.GH200_96GB

    def test_from_string_h100(self):
        """Test GPU type parsing for H100."""
        assert GPUType.from_string("h100") == GPUType.H100_80GB
        assert GPUType.from_string("H100-80GB") == GPUType.H100_80GB

    def test_from_string_a100_80gb(self):
        """Test GPU type parsing for A100 80GB."""
        assert GPUType.from_string("a100-80gb") == GPUType.A100_80GB
        assert GPUType.from_string("NVIDIA A100 80GB") == GPUType.A100_80GB

    def test_from_string_a100_40gb(self):
        """Test GPU type parsing for A100 40GB (default)."""
        assert GPUType.from_string("a100") == GPUType.A100_40GB

    def test_from_string_a10_not_a100(self):
        """Test A10 is correctly identified (not confused with A100)."""
        assert GPUType.from_string("a10") == GPUType.A10
        assert GPUType.from_string("NVIDIA A10") == GPUType.A10

    def test_from_string_rtx_5090(self):
        """Test GPU type parsing for RTX 5090."""
        assert GPUType.from_string("rtx5090") == GPUType.RTX_5090
        assert GPUType.from_string("RTX 5090") == GPUType.RTX_5090

    def test_from_string_rtx_4090(self):
        """Test GPU type parsing for RTX 4090."""
        assert GPUType.from_string("rtx4090") == GPUType.RTX_4090

    def test_from_string_rtx_3090(self):
        """Test GPU type parsing for RTX 3090."""
        assert GPUType.from_string("rtx3090") == GPUType.RTX_3090

    def test_from_string_unknown(self):
        """Test unknown GPU types return UNKNOWN."""
        assert GPUType.from_string("rtx2080") == GPUType.UNKNOWN
        assert GPUType.from_string("random_gpu") == GPUType.UNKNOWN
        assert GPUType.from_string("") == GPUType.UNKNOWN

    def test_from_string_case_insensitive(self):
        """Test GPU parsing is case insensitive."""
        assert GPUType.from_string("H100") == GPUType.from_string("h100")


class TestInstanceStatus:
    """Tests for InstanceStatus enum."""

    def test_all_statuses_exist(self):
        """Verify all expected statuses are defined."""
        assert InstanceStatus.PENDING
        assert InstanceStatus.STARTING
        assert InstanceStatus.RUNNING
        assert InstanceStatus.STOPPING
        assert InstanceStatus.STOPPED
        assert InstanceStatus.TERMINATED
        assert InstanceStatus.ERROR
        assert InstanceStatus.UNKNOWN

    def test_status_values_are_strings(self):
        """Verify status values are lowercase strings."""
        for status in InstanceStatus:
            assert isinstance(status.value, str)
            assert status.value == status.value.lower()

    def test_status_count(self):
        """Verify expected number of statuses."""
        assert len(InstanceStatus) == 8


class TestInstance:
    """Tests for Instance dataclass."""

    def test_instance_creation_minimal(self):
        """Test creating instance with minimal required fields."""
        instance = Instance(
            id="inst-123",
            provider=ProviderType.VAST,
            name="test-instance",
            status=InstanceStatus.RUNNING,
            gpu_type=GPUType.RTX_4090,
        )
        assert instance.id == "inst-123"
        assert instance.provider == ProviderType.VAST

    def test_instance_defaults(self):
        """Test instance default values."""
        instance = Instance(
            id="inst-123",
            provider=ProviderType.VAST,
            name="test",
            status=InstanceStatus.PENDING,
            gpu_type=GPUType.RTX_4090,
        )
        assert instance.gpu_count == 1
        assert instance.gpu_memory_gb == 0.0
        assert instance.ip_address is None
        assert instance.ssh_port == 22
        assert instance.ssh_user == "root"

    def test_is_running_true(self):
        """Test is_running property when running with IP."""
        instance = Instance(
            id="inst-123",
            provider=ProviderType.VAST,
            name="test",
            status=InstanceStatus.RUNNING,
            gpu_type=GPUType.RTX_4090,
            ip_address="10.0.0.1",
        )
        assert instance.is_running is True

    def test_is_running_false_no_ip(self):
        """Test is_running is False when no IP address."""
        instance = Instance(
            id="inst-123",
            provider=ProviderType.VAST,
            name="test",
            status=InstanceStatus.RUNNING,
            gpu_type=GPUType.RTX_4090,
            ip_address=None,
        )
        assert instance.is_running is False

    def test_is_running_false_not_running(self):
        """Test is_running is False when not RUNNING status."""
        for status in [InstanceStatus.PENDING, InstanceStatus.STOPPED]:
            instance = Instance(
                id="inst-123",
                provider=ProviderType.VAST,
                name="test",
                status=status,
                gpu_type=GPUType.RTX_4090,
                ip_address="10.0.0.1",
            )
            assert instance.is_running is False

    def test_ssh_host_with_ip(self):
        """Test ssh_host property with IP address."""
        instance = Instance(
            id="inst-123",
            provider=ProviderType.VAST,
            name="test",
            status=InstanceStatus.RUNNING,
            gpu_type=GPUType.RTX_4090,
            ip_address="192.168.1.50",
            ssh_user="ubuntu",
        )
        assert instance.ssh_host == "ubuntu@192.168.1.50"

    def test_ssh_host_without_ip(self):
        """Test ssh_host property without IP address."""
        instance = Instance(
            id="inst-123",
            provider=ProviderType.VAST,
            name="test",
            status=InstanceStatus.RUNNING,
            gpu_type=GPUType.RTX_4090,
            ip_address=None,
        )
        assert instance.ssh_host == ""


class MockCloudProvider(CloudProvider):
    """Mock provider for testing CloudProvider abstract class."""

    def __init__(self, configured: bool = True):
        self._configured = configured
        self._instances: list[Instance] = []

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.VAST

    @property
    def name(self) -> str:
        return "Mock Provider"

    def is_configured(self) -> bool:
        return self._configured

    async def list_instances(self) -> list[Instance]:
        return self._instances

    async def get_instance(self, instance_id: str) -> Instance | None:
        for inst in self._instances:
            if inst.id == instance_id:
                return inst
        return None

    async def get_instance_status(self, instance_id: str) -> InstanceStatus:
        inst = await self.get_instance(instance_id)
        return inst.status if inst else InstanceStatus.UNKNOWN

    async def scale_up(
        self,
        gpu_type: GPUType,
        count: int = 1,
        region: str | None = None,
        name_prefix: str = "ringrift",
    ) -> list[Instance]:
        new_instances = []
        for i in range(count):
            inst = Instance(
                id=f"new-{i}",
                provider=self.provider_type,
                name=f"{name_prefix}-{i}",
                status=InstanceStatus.PENDING,
                gpu_type=gpu_type,
            )
            new_instances.append(inst)
            self._instances.append(inst)
        return new_instances

    async def scale_down(self, instance_ids: list[str]) -> dict[str, bool]:
        results = {}
        for inst_id in instance_ids:
            found = False
            for inst in self._instances:
                if inst.id == inst_id:
                    inst.status = InstanceStatus.TERMINATED
                    found = True
                    break
            results[inst_id] = found
        return results

    def get_cost_per_hour(self, gpu_type: GPUType) -> float:
        costs = {
            GPUType.RTX_3090: 0.50,
            GPUType.RTX_4090: 0.80,
            GPUType.H100_80GB: 3.00,
        }
        return costs.get(gpu_type, 0.0)


class TestCloudProvider:
    """Tests for CloudProvider abstract base class."""

    @pytest.fixture
    def provider(self):
        """Create mock provider."""
        return MockCloudProvider()

    @pytest.fixture
    def provider_with_instances(self, provider: MockCloudProvider):
        """Create provider with test instances."""
        provider._instances = [
            Instance(
                id="inst-1",
                provider=ProviderType.VAST,
                name="test-1",
                status=InstanceStatus.RUNNING,
                gpu_type=GPUType.RTX_4090,
                gpu_count=1,
                ip_address="10.0.0.1",
                cost_per_hour=0.80,
            ),
            Instance(
                id="inst-2",
                provider=ProviderType.VAST,
                name="test-2",
                status=InstanceStatus.RUNNING,
                gpu_type=GPUType.H100_80GB,
                gpu_count=2,
                ip_address="10.0.0.2",
                cost_per_hour=6.00,
            ),
            Instance(
                id="inst-3",
                provider=ProviderType.VAST,
                name="test-3",
                status=InstanceStatus.STOPPED,
                gpu_type=GPUType.RTX_3090,
                gpu_count=1,
                ip_address="10.0.0.3",
                cost_per_hour=0.50,
            ),
        ]
        return provider

    def test_provider_type(self, provider: MockCloudProvider):
        """Test provider_type property."""
        assert provider.provider_type == ProviderType.VAST

    def test_provider_name(self, provider: MockCloudProvider):
        """Test name property."""
        assert provider.name == "Mock Provider"

    def test_is_configured_true(self, provider: MockCloudProvider):
        """Test is_configured when configured."""
        assert provider.is_configured() is True

    def test_is_configured_false(self):
        """Test is_configured when not configured."""
        provider = MockCloudProvider(configured=False)
        assert provider.is_configured() is False

    @pytest.mark.asyncio
    async def test_list_instances_empty(self, provider: MockCloudProvider):
        """Test list_instances with no instances."""
        instances = await provider.list_instances()
        assert instances == []

    @pytest.mark.asyncio
    async def test_list_instances(self, provider_with_instances: MockCloudProvider):
        """Test list_instances with instances."""
        instances = await provider_with_instances.list_instances()
        assert len(instances) == 3

    @pytest.mark.asyncio
    async def test_get_instance_found(self, provider_with_instances: MockCloudProvider):
        """Test get_instance when instance exists."""
        instance = await provider_with_instances.get_instance("inst-1")
        assert instance is not None
        assert instance.id == "inst-1"

    @pytest.mark.asyncio
    async def test_get_instance_not_found(self, provider_with_instances: MockCloudProvider):
        """Test get_instance when instance doesn't exist."""
        instance = await provider_with_instances.get_instance("nonexistent")
        assert instance is None

    @pytest.mark.asyncio
    async def test_get_instance_status(self, provider_with_instances: MockCloudProvider):
        """Test get_instance_status."""
        status = await provider_with_instances.get_instance_status("inst-1")
        assert status == InstanceStatus.RUNNING

        status = await provider_with_instances.get_instance_status("nonexistent")
        assert status == InstanceStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_scale_up(self, provider: MockCloudProvider):
        """Test scale_up creates instances."""
        instances = await provider.scale_up(GPUType.RTX_4090, count=2)
        assert len(instances) == 2
        assert all(i.gpu_type == GPUType.RTX_4090 for i in instances)

    @pytest.mark.asyncio
    async def test_scale_down(self, provider_with_instances: MockCloudProvider):
        """Test scale_down terminates instances."""
        results = await provider_with_instances.scale_down(["inst-1"])
        assert results["inst-1"] is True

    def test_get_cost_per_hour(self, provider: MockCloudProvider):
        """Test get_cost_per_hour."""
        assert provider.get_cost_per_hour(GPUType.RTX_4090) == 0.80
        assert provider.get_cost_per_hour(GPUType.UNKNOWN) == 0.0

    @pytest.mark.asyncio
    async def test_get_available_gpus(self, provider_with_instances: MockCloudProvider):
        """Test get_available_gpus aggregates running instances."""
        gpus = await provider_with_instances.get_available_gpus()
        assert gpus[GPUType.RTX_4090] == 1
        assert gpus[GPUType.H100_80GB] == 2

    @pytest.mark.asyncio
    async def test_get_total_cost_per_hour(self, provider_with_instances: MockCloudProvider):
        """Test get_total_cost_per_hour sums running instances."""
        total = await provider_with_instances.get_total_cost_per_hour()
        assert total == 6.80

    @pytest.mark.asyncio
    async def test_health_check_not_running(self, provider_with_instances: MockCloudProvider):
        """Test health_check returns False for non-running instance."""
        instance = await provider_with_instances.get_instance("inst-3")
        result = await provider_with_instances.health_check(instance)
        assert result is False
