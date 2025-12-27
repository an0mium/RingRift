"""Tests for Hetzner manager."""

import pytest
from unittest.mock import AsyncMock, patch
import json

from app.providers.base import (
    InstanceState,
    Provider,
    ProviderInstance,
)
from app.providers.hetzner_manager import (
    HetznerManager,
    HETZNER_SERVER_TYPES,
    _parse_server_state,
)


class TestHetznerServerTypes:
    """Tests for Hetzner server type definitions."""

    def test_cpx_types_defined(self):
        """CPX types are defined."""
        assert "cpx11" in HETZNER_SERVER_TYPES
        assert "cpx51" in HETZNER_SERVER_TYPES

    def test_cx_types_defined(self):
        """CX types are defined."""
        assert "cx22" in HETZNER_SERVER_TYPES

    def test_types_have_costs(self):
        """All types have hourly costs."""
        for name, info in HETZNER_SERVER_TYPES.items():
            assert info["hourly_cost"] > 0, f"{name} missing cost"


class TestParseServerState:
    """Tests for state parsing."""

    def test_running(self):
        assert _parse_server_state("running") == InstanceState.RUNNING

    def test_starting(self):
        assert _parse_server_state("starting") == InstanceState.STARTING

    def test_stopping(self):
        assert _parse_server_state("stopping") == InstanceState.STOPPING

    def test_off(self):
        assert _parse_server_state("off") == InstanceState.STOPPED

    def test_unknown(self):
        assert _parse_server_state("weird") == InstanceState.UNKNOWN


class TestHetznerManager:
    """Tests for HetznerManager class."""

    def test_init(self):
        """Can initialize manager."""
        manager = HetznerManager()
        assert manager.provider == Provider.HETZNER

    @pytest.mark.asyncio
    async def test_list_instances_no_cli(self):
        """Returns empty list if CLI not available."""
        manager = HetznerManager()

        with patch.object(manager, "_check_hcloud_available", new_callable=AsyncMock) as mock:
            mock.return_value = False
            instances = await manager.list_instances()
            assert instances == []

    @pytest.mark.asyncio
    async def test_list_instances_parses_response(self):
        """Correctly parses hcloud output."""
        manager = HetznerManager()

        mock_response = [
            {
                "id": 12345,
                "name": "cpu1",
                "status": "running",
                "public_net": {
                    "ipv4": {"ip": "1.2.3.4"},
                },
                "server_type": {"name": "cpx31"},
                "datacenter": {"name": "fsn1-dc14"},
            }
        ]

        with patch.object(manager, "_run_hcloud", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            instances = await manager.list_instances()

            assert len(instances) == 1
            inst = instances[0]
            assert inst.provider == Provider.HETZNER
            assert inst.name == "cpu1"
            assert inst.state == InstanceState.RUNNING
            assert inst.public_ip == "1.2.3.4"

    @pytest.mark.asyncio
    async def test_reboot_instance(self):
        """Can reboot instance."""
        manager = HetznerManager()

        with patch.object(manager, "_run_hcloud_action", new_callable=AsyncMock) as mock:
            mock.return_value = True
            result = await manager.reboot_instance("12345")

            assert result is True
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_terminate_instance(self):
        """Can terminate instance."""
        manager = HetznerManager()

        with patch.object(manager, "_run_hcloud_action", new_callable=AsyncMock) as mock:
            mock.return_value = True
            result = await manager.terminate_instance("12345")

            assert result is True


class TestHetznerHealthCheck:
    """Tests for health checking."""

    @pytest.mark.asyncio
    async def test_check_health_ssh_only(self):
        """Health check uses SSH (Hetzner are CPU nodes so no GPU check)."""
        manager = HetznerManager()

        instance = ProviderInstance(
            instance_id="123",
            provider=Provider.HETZNER,
            name="cpu1",
            public_ip="1.2.3.4",
        )

        from app.providers.base import HealthCheckResult

        # Mock all health checks
        with patch.object(manager, "check_ssh_connectivity", new_callable=AsyncMock) as ssh:
            with patch.object(manager, "check_p2p_health", new_callable=AsyncMock) as p2p:
                with patch.object(manager, "check_tailscale", new_callable=AsyncMock) as ts:
                    ssh.return_value = HealthCheckResult(healthy=True, check_type="ssh", message="OK")
                    p2p.return_value = HealthCheckResult(healthy=True, check_type="p2p", message="OK")
                    ts.return_value = HealthCheckResult(healthy=True, check_type="tailscale", message="OK")

                    result = await manager.check_health(instance)
                    assert result.healthy is True
