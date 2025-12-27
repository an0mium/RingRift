"""Tests for AWS manager."""

import pytest
from unittest.mock import AsyncMock, patch
import json

from app.providers.base import (
    InstanceState,
    Provider,
    ProviderInstance,
)
from app.providers.aws_manager import (
    AWSManager,
    AWS_INSTANCE_TYPES,
    _parse_instance_state,
)


class TestAwsInstanceTypes:
    """Tests for AWS instance type definitions."""

    def test_gpu_types_defined(self):
        """GPU types are defined."""
        assert "p3.2xlarge" in AWS_INSTANCE_TYPES or "p4d.24xlarge" in AWS_INSTANCE_TYPES
        assert "g4dn.xlarge" in AWS_INSTANCE_TYPES

    def test_cpu_types_defined(self):
        """CPU types are defined."""
        assert "t3.medium" in AWS_INSTANCE_TYPES
        assert "c5.xlarge" in AWS_INSTANCE_TYPES


class TestParseInstanceState:
    """Tests for state parsing."""

    def test_running(self):
        assert _parse_instance_state("running") == InstanceState.RUNNING

    def test_pending(self):
        assert _parse_instance_state("pending") == InstanceState.STARTING

    def test_stopping(self):
        assert _parse_instance_state("stopping") == InstanceState.STOPPING

    def test_stopped(self):
        assert _parse_instance_state("stopped") == InstanceState.STOPPED

    def test_terminated(self):
        assert _parse_instance_state("terminated") == InstanceState.TERMINATED

    def test_unknown(self):
        assert _parse_instance_state("weird") == InstanceState.UNKNOWN


class TestAWSManager:
    """Tests for AWSManager class."""

    def test_init(self):
        """Can initialize manager."""
        manager = AWSManager()
        assert manager.provider == Provider.AWS

    def test_init_with_region(self):
        """Can initialize with specific region."""
        manager = AWSManager(region="us-west-2")
        assert manager.region == "us-west-2"

    @pytest.mark.asyncio
    async def test_list_instances_no_cli(self):
        """Returns empty list if CLI not available."""
        manager = AWSManager()

        with patch.object(manager, "_check_aws_available", new_callable=AsyncMock) as mock:
            mock.return_value = False
            instances = await manager.list_instances()
            assert instances == []

    @pytest.mark.asyncio
    async def test_list_instances_parses_response(self):
        """Correctly parses AWS CLI output."""
        manager = AWSManager()

        mock_response = {
            "Reservations": [
                {
                    "Instances": [
                        {
                            "InstanceId": "i-12345",
                            "State": {"Name": "running"},
                            "PublicIpAddress": "1.2.3.4",
                            "PrivateIpAddress": "10.0.0.1",
                            "InstanceType": "t3.medium",
                            "Tags": [{"Key": "Name", "Value": "aws-staging"}],
                        }
                    ]
                }
            ]
        }

        with patch.object(manager, "_run_aws", new_callable=AsyncMock) as mock:
            mock.return_value = mock_response
            instances = await manager.list_instances()

            assert len(instances) == 1
            inst = instances[0]
            assert inst.provider == Provider.AWS
            assert inst.instance_id == "i-12345"
            assert inst.name == "aws-staging"
            assert inst.state == InstanceState.RUNNING

    @pytest.mark.asyncio
    async def test_reboot_instance(self):
        """Can reboot instance."""
        manager = AWSManager()

        with patch.object(manager, "_run_aws_action", new_callable=AsyncMock) as mock:
            mock.return_value = True
            result = await manager.reboot_instance("i-12345")

            assert result is True

    @pytest.mark.asyncio
    async def test_terminate_instance(self):
        """Can terminate instance."""
        manager = AWSManager()

        with patch.object(manager, "_run_aws_action", new_callable=AsyncMock) as mock:
            mock.return_value = True
            result = await manager.terminate_instance("i-12345")

            assert result is True

    @pytest.mark.asyncio
    async def test_start_instance(self):
        """Can start instance."""
        manager = AWSManager()

        with patch.object(manager, "_run_aws_action", new_callable=AsyncMock) as mock:
            mock.return_value = True
            result = await manager.start_instance("i-12345")

            assert result is True

    @pytest.mark.asyncio
    async def test_stop_instance(self):
        """Can stop instance."""
        manager = AWSManager()

        with patch.object(manager, "_run_aws_action", new_callable=AsyncMock) as mock:
            mock.return_value = True
            result = await manager.stop_instance("i-12345")

            assert result is True


class TestAWSHealthCheck:
    """Tests for health checking."""

    @pytest.mark.asyncio
    async def test_check_health_ssh_first(self):
        """Health check starts with SSH."""
        manager = AWSManager()

        instance = ProviderInstance(
            instance_id="i-123",
            provider=Provider.AWS,
            name="aws-staging",
            public_ip="1.2.3.4",
        )

        with patch.object(manager, "check_ssh_connectivity", new_callable=AsyncMock) as ssh:
            from app.providers.base import HealthCheckResult
            ssh.return_value = HealthCheckResult(
                healthy=False,
                check_type="ssh",
                message="SSH timeout",
            )
            result = await manager.check_health(instance)

            assert result.healthy is False
            ssh.assert_called_once()
