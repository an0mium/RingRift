"""Tests for node_availability state_checker module.

December 2025: Phase 1 of Node Availability Test Coverage.

Tests:
- ProviderInstanceState enum values
- STATE_TO_YAML_STATUS mapping
- InstanceInfo dataclass
- StateChecker abstract base class
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock

from app.coordination.node_availability.state_checker import (
    ProviderInstanceState,
    STATE_TO_YAML_STATUS,
    InstanceInfo,
    StateChecker,
)


class TestProviderInstanceState:
    """Tests for ProviderInstanceState enum."""

    def test_all_states_defined(self) -> None:
        """Verify all expected states are defined."""
        expected_states = {"RUNNING", "STARTING", "STOPPING", "STOPPED", "TERMINATED", "UNKNOWN"}
        actual_states = {state.name for state in ProviderInstanceState}
        assert actual_states == expected_states

    def test_state_values_are_lowercase(self) -> None:
        """State values should be lowercase strings."""
        for state in ProviderInstanceState:
            assert state.value == state.value.lower()
            assert state.value == state.name.lower()

    def test_running_state(self) -> None:
        """Test RUNNING state."""
        assert ProviderInstanceState.RUNNING.value == "running"

    def test_terminated_state(self) -> None:
        """Test TERMINATED state."""
        assert ProviderInstanceState.TERMINATED.value == "terminated"

    def test_unknown_state(self) -> None:
        """Test UNKNOWN state."""
        assert ProviderInstanceState.UNKNOWN.value == "unknown"


class TestStateToYamlMapping:
    """Tests for STATE_TO_YAML_STATUS mapping."""

    def test_all_states_have_mapping(self) -> None:
        """Every ProviderInstanceState should have a YAML status mapping."""
        for state in ProviderInstanceState:
            assert state in STATE_TO_YAML_STATUS, f"Missing mapping for {state}"

    def test_running_maps_to_ready(self) -> None:
        """RUNNING state should map to 'ready' status."""
        assert STATE_TO_YAML_STATUS[ProviderInstanceState.RUNNING] == "ready"

    def test_starting_maps_to_setup(self) -> None:
        """STARTING state should map to 'setup' status."""
        assert STATE_TO_YAML_STATUS[ProviderInstanceState.STARTING] == "setup"

    def test_stopping_maps_to_offline(self) -> None:
        """STOPPING state should map to 'offline' status."""
        assert STATE_TO_YAML_STATUS[ProviderInstanceState.STOPPING] == "offline"

    def test_stopped_maps_to_offline(self) -> None:
        """STOPPED state should map to 'offline' status."""
        assert STATE_TO_YAML_STATUS[ProviderInstanceState.STOPPED] == "offline"

    def test_terminated_maps_to_retired(self) -> None:
        """TERMINATED state should map to 'retired' status."""
        assert STATE_TO_YAML_STATUS[ProviderInstanceState.TERMINATED] == "retired"

    def test_unknown_maps_to_offline(self) -> None:
        """UNKNOWN state should map to 'offline' status."""
        assert STATE_TO_YAML_STATUS[ProviderInstanceState.UNKNOWN] == "offline"


class TestInstanceInfo:
    """Tests for InstanceInfo dataclass."""

    def test_create_with_required_fields(self) -> None:
        """Can create with only required fields."""
        info = InstanceInfo(
            instance_id="i-12345",
            state=ProviderInstanceState.RUNNING,
            provider="vast",
        )
        assert info.instance_id == "i-12345"
        assert info.state == ProviderInstanceState.RUNNING
        assert info.provider == "vast"

    def test_optional_fields_default_none(self) -> None:
        """Optional fields default to None."""
        info = InstanceInfo(
            instance_id="i-12345",
            state=ProviderInstanceState.RUNNING,
            provider="vast",
        )
        assert info.node_name is None
        assert info.tailscale_ip is None
        assert info.public_ip is None
        assert info.ssh_host is None
        assert info.gpu_type is None
        assert info.hostname is None
        assert info.created_at is None
        assert info.last_seen is None

    def test_ssh_port_defaults_to_22(self) -> None:
        """SSH port defaults to 22."""
        info = InstanceInfo(
            instance_id="i-12345",
            state=ProviderInstanceState.RUNNING,
            provider="vast",
        )
        assert info.ssh_port == 22

    def test_gpu_fields_default_to_zero(self) -> None:
        """GPU fields default to 0."""
        info = InstanceInfo(
            instance_id="i-12345",
            state=ProviderInstanceState.RUNNING,
            provider="vast",
        )
        assert info.gpu_count == 0
        assert info.gpu_vram_gb == 0.0

    def test_raw_data_defaults_to_empty_dict(self) -> None:
        """raw_data defaults to empty dict."""
        info = InstanceInfo(
            instance_id="i-12345",
            state=ProviderInstanceState.RUNNING,
            provider="vast",
        )
        assert info.raw_data == {}

    def test_yaml_status_property_running(self) -> None:
        """yaml_status property returns correct status for RUNNING."""
        info = InstanceInfo(
            instance_id="i-12345",
            state=ProviderInstanceState.RUNNING,
            provider="vast",
        )
        assert info.yaml_status == "ready"

    def test_yaml_status_property_terminated(self) -> None:
        """yaml_status property returns correct status for TERMINATED."""
        info = InstanceInfo(
            instance_id="i-12345",
            state=ProviderInstanceState.TERMINATED,
            provider="vast",
        )
        assert info.yaml_status == "retired"

    def test_str_with_node_name(self) -> None:
        """__str__ uses node_name when available."""
        info = InstanceInfo(
            instance_id="i-12345",
            state=ProviderInstanceState.RUNNING,
            provider="vast",
            node_name="vast-mynode",
        )
        result = str(info)
        assert "vast-mynode" in result
        assert "vast" in result
        assert "running" in result

    def test_str_without_node_name(self) -> None:
        """__str__ falls back to instance_id when no node_name."""
        info = InstanceInfo(
            instance_id="i-12345",
            state=ProviderInstanceState.RUNNING,
            provider="vast",
        )
        result = str(info)
        assert "i-12345" in result
        assert "vast" in result
        assert "running" in result

    def test_all_fields_populated(self) -> None:
        """Can create with all fields populated."""
        now = datetime.now()
        info = InstanceInfo(
            instance_id="i-12345",
            state=ProviderInstanceState.RUNNING,
            provider="vast",
            node_name="vast-mynode",
            tailscale_ip="100.64.1.1",
            public_ip="203.0.113.1",
            ssh_host="ssh.vast.ai",
            ssh_port=30022,
            gpu_type="RTX 4090",
            gpu_count=2,
            gpu_vram_gb=24.0,
            hostname="node1",
            created_at=now,
            last_seen=now,
            raw_data={"key": "value"},
        )
        assert info.node_name == "vast-mynode"
        assert info.tailscale_ip == "100.64.1.1"
        assert info.public_ip == "203.0.113.1"
        assert info.ssh_host == "ssh.vast.ai"
        assert info.ssh_port == 30022
        assert info.gpu_type == "RTX 4090"
        assert info.gpu_count == 2
        assert info.gpu_vram_gb == 24.0
        assert info.hostname == "node1"
        assert info.created_at == now
        assert info.last_seen == now
        assert info.raw_data == {"key": "value"}


class ConcreteStateChecker(StateChecker):
    """Concrete implementation for testing abstract base class."""

    def __init__(self, provider_name: str = "test"):
        super().__init__(provider_name)
        self.get_states_called = False
        self.check_api_called = False
        self.correlate_called = False

    async def get_instance_states(self) -> list[InstanceInfo]:
        self.get_states_called = True
        return []

    async def check_api_availability(self) -> bool:
        self.check_api_called = True
        return True

    def correlate_with_config(
        self,
        instances: list[InstanceInfo],
        config_hosts: dict[str, dict],
    ) -> list[InstanceInfo]:
        self.correlate_called = True
        return instances


class TestStateChecker:
    """Tests for StateChecker abstract base class."""

    def test_is_abstract(self) -> None:
        """StateChecker cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            StateChecker("test")  # type: ignore

    def test_concrete_can_be_instantiated(self) -> None:
        """Concrete subclass can be instantiated."""
        checker = ConcreteStateChecker("vast")
        assert checker.provider_name == "vast"

    def test_is_enabled_default_true(self) -> None:
        """is_enabled defaults to True."""
        checker = ConcreteStateChecker()
        assert checker.is_enabled is True

    def test_disable_sets_enabled_false(self) -> None:
        """disable() sets is_enabled to False."""
        checker = ConcreteStateChecker()
        assert checker.is_enabled is True
        checker.disable("test reason")
        assert checker.is_enabled is False

    def test_enable_sets_enabled_true(self) -> None:
        """enable() sets is_enabled to True."""
        checker = ConcreteStateChecker()
        checker.disable("test")
        assert checker.is_enabled is False
        checker.enable()
        assert checker.is_enabled is True

    def test_get_status_returns_dict(self) -> None:
        """get_status returns a dict with expected keys."""
        checker = ConcreteStateChecker("vast")
        status = checker.get_status()

        assert isinstance(status, dict)
        assert status["provider"] == "vast"
        assert status["enabled"] is True
        assert status["last_check"] is None
        assert status["last_error"] is None

    def test_get_status_after_disable(self) -> None:
        """get_status reflects disabled state."""
        checker = ConcreteStateChecker("vast")
        checker.disable("No API key")
        status = checker.get_status()

        assert status["enabled"] is False

    @pytest.mark.asyncio
    async def test_abstract_methods_can_be_called(self) -> None:
        """Abstract methods can be called on concrete class."""
        checker = ConcreteStateChecker("test")

        instances = await checker.get_instance_states()
        assert instances == []
        assert checker.get_states_called is True

        available = await checker.check_api_availability()
        assert available is True
        assert checker.check_api_called is True

        result = checker.correlate_with_config([], {})
        assert result == []
        assert checker.correlate_called is True

    def test_last_check_initially_none(self) -> None:
        """_last_check is initially None."""
        checker = ConcreteStateChecker()
        assert checker._last_check is None

    def test_last_error_initially_none(self) -> None:
        """_last_error is initially None."""
        checker = ConcreteStateChecker()
        assert checker._last_error is None
