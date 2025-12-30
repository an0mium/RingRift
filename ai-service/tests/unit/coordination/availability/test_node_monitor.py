"""Tests for NodeMonitor - multi-layer node health monitoring.

Created: Dec 29, 2025
Phase 3: Test coverage for critical untested modules.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.coordination.availability.node_monitor import (
    HealthCheckLayer,
    NodeHealthResult,
    NodeMonitor,
    NodeMonitorConfig,
    get_node_monitor,
    reset_node_monitor,
)


# --- Mock ClusterNode for testing ---


@dataclass
class MockClusterNode:
    """Mock ClusterNode for testing."""

    name: str
    tailscale_ip: str | None = None
    best_ip: str | None = None
    ssh_user: str = "root"
    ssh_port: int = 22
    is_gpu_node: bool = False
    provider: str | None = None
    instance_id: str | None = None


# --- HealthCheckLayer Tests ---


class TestHealthCheckLayer:
    """Tests for HealthCheckLayer enum."""

    def test_layer_values_exist(self) -> None:
        """All expected layer values should exist."""
        assert HealthCheckLayer.P2P.value == "p2p"
        assert HealthCheckLayer.SSH.value == "ssh"
        assert HealthCheckLayer.GPU.value == "gpu"
        assert HealthCheckLayer.PROVIDER_API.value == "provider_api"
        assert HealthCheckLayer.ALL.value == "all"

    def test_layer_is_string_enum(self) -> None:
        """HealthCheckLayer should be a string enum."""
        assert isinstance(HealthCheckLayer.P2P.value, str)
        assert str(HealthCheckLayer.P2P) == "HealthCheckLayer.P2P"

    def test_all_layers(self) -> None:
        """All layers should be enumerable."""
        layers = list(HealthCheckLayer)
        assert len(layers) == 5


# --- NodeHealthResult Tests ---


class TestNodeHealthResult:
    """Tests for NodeHealthResult dataclass."""

    def test_basic_result(self) -> None:
        """Basic healthy result should have correct fields."""
        result = NodeHealthResult(
            node_id="test-node",
            layer=HealthCheckLayer.P2P,
            healthy=True,
            latency_ms=50.0,
        )
        assert result.node_id == "test-node"
        assert result.layer == HealthCheckLayer.P2P
        assert result.healthy is True
        assert result.latency_ms == 50.0
        assert result.error is None
        assert isinstance(result.timestamp, datetime)
        assert result.details == {}

    def test_unhealthy_result_with_error(self) -> None:
        """Unhealthy result should include error message."""
        result = NodeHealthResult(
            node_id="test-node",
            layer=HealthCheckLayer.SSH,
            healthy=False,
            latency_ms=30000.0,
            error="SSH timeout",
        )
        assert result.healthy is False
        assert result.error == "SSH timeout"

    def test_result_with_details(self) -> None:
        """Result can include additional details."""
        result = NodeHealthResult(
            node_id="test-node",
            layer=HealthCheckLayer.GPU,
            healthy=True,
            latency_ms=100.0,
            details={"gpu_info": "NVIDIA A100, 80GB"},
        )
        assert result.details["gpu_info"] == "NVIDIA A100, 80GB"

    def test_to_dict(self) -> None:
        """to_dict should return serializable dictionary."""
        result = NodeHealthResult(
            node_id="test-node",
            layer=HealthCheckLayer.P2P,
            healthy=True,
            latency_ms=25.5,
            details={"status": "ok"},
        )
        d = result.to_dict()
        assert d["node_id"] == "test-node"
        assert d["layer"] == "p2p"
        assert d["healthy"] is True
        assert d["latency_ms"] == 25.5
        assert d["error"] is None
        assert "timestamp" in d
        assert d["details"]["status"] == "ok"


# --- NodeMonitorConfig Tests ---


class TestNodeMonitorConfig:
    """Tests for NodeMonitorConfig."""

    def test_default_config(self) -> None:
        """Default config should have expected values."""
        config = NodeMonitorConfig()
        assert config.check_interval_seconds == 30
        assert config.p2p_timeout_seconds == 15.0
        assert config.ssh_timeout_seconds == 30.0
        assert config.gpu_check_enabled is True
        assert config.provider_check_enabled is True
        assert config.consecutive_failures_before_unhealthy == 3
        assert config.consecutive_failures_before_recovery == 5
        assert config.p2p_port == 8770

    def test_custom_config(self) -> None:
        """Custom config values should be accepted."""
        config = NodeMonitorConfig(
            check_interval_seconds=60,
            p2p_timeout_seconds=10.0,
            ssh_timeout_seconds=20.0,
            gpu_check_enabled=False,
            consecutive_failures_before_unhealthy=5,
        )
        assert config.check_interval_seconds == 60
        assert config.p2p_timeout_seconds == 10.0
        assert config.gpu_check_enabled is False
        assert config.consecutive_failures_before_unhealthy == 5


# --- NodeMonitor Initialization Tests ---


class TestNodeMonitorInit:
    """Tests for NodeMonitor initialization."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_node_monitor()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_node_monitor()

    def test_default_init(self) -> None:
        """NodeMonitor should initialize with defaults."""
        monitor = NodeMonitor()
        assert monitor.config is not None
        assert monitor.config.check_interval_seconds == 30
        assert monitor._nodes == []
        assert monitor._failure_counts == {}
        assert monitor._health_history == {}

    def test_init_with_config(self) -> None:
        """NodeMonitor should accept custom config."""
        config = NodeMonitorConfig(
            check_interval_seconds=120,
            p2p_timeout_seconds=20.0,
        )
        monitor = NodeMonitor(config=config)
        assert monitor.config.check_interval_seconds == 120
        assert monitor.config.p2p_timeout_seconds == 20.0

    def test_init_with_nodes(self) -> None:
        """NodeMonitor should accept initial node list."""
        nodes = [
            MockClusterNode(name="node-1", tailscale_ip="100.1.2.3"),
            MockClusterNode(name="node-2", tailscale_ip="100.1.2.4"),
        ]
        monitor = NodeMonitor(nodes=nodes)
        assert len(monitor._nodes) == 2

    def test_config_property(self) -> None:
        """config property should return NodeMonitorConfig."""
        monitor = NodeMonitor()
        assert isinstance(monitor.config, NodeMonitorConfig)

    def test_name_attribute(self) -> None:
        """name attribute should be 'NodeMonitor' (HandlerBase pattern)."""
        monitor = NodeMonitor()
        assert monitor.name == "NodeMonitor"


# --- NodeMonitor set_nodes Tests ---


class TestNodeMonitorSetNodes:
    """Tests for NodeMonitor.set_nodes()."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_node_monitor()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_node_monitor()

    def test_set_nodes_updates_list(self) -> None:
        """set_nodes should update the node list."""
        monitor = NodeMonitor()
        nodes = [MockClusterNode(name="node-1", tailscale_ip="100.1.2.3")]
        monitor.set_nodes(nodes)
        assert len(monitor._nodes) == 1
        assert monitor._nodes[0].name == "node-1"

    def test_set_nodes_initializes_failure_counts(self) -> None:
        """set_nodes should initialize failure counts for new nodes."""
        monitor = NodeMonitor()
        nodes = [
            MockClusterNode(name="node-1", tailscale_ip="100.1.2.3"),
            MockClusterNode(name="node-2", tailscale_ip="100.1.2.4"),
        ]
        monitor.set_nodes(nodes)
        assert "node-1" in monitor._failure_counts
        assert "node-2" in monitor._failure_counts
        assert monitor._failure_counts["node-1"] == 0

    def test_set_nodes_initializes_health_history(self) -> None:
        """set_nodes should initialize health history for new nodes."""
        monitor = NodeMonitor()
        nodes = [MockClusterNode(name="node-1", tailscale_ip="100.1.2.3")]
        monitor.set_nodes(nodes)
        assert "node-1" in monitor._health_history
        assert monitor._health_history["node-1"] == []

    def test_set_nodes_preserves_existing_state(self) -> None:
        """set_nodes should preserve state for existing nodes."""
        monitor = NodeMonitor()
        monitor._failure_counts["node-1"] = 3
        monitor._health_history["node-1"] = ["some_history"]

        nodes = [
            MockClusterNode(name="node-1", tailscale_ip="100.1.2.3"),
            MockClusterNode(name="node-2", tailscale_ip="100.1.2.4"),
        ]
        monitor.set_nodes(nodes)

        # Existing node state preserved
        assert monitor._failure_counts["node-1"] == 3
        assert monitor._health_history["node-1"] == ["some_history"]
        # New node initialized
        assert monitor._failure_counts["node-2"] == 0


# --- NodeMonitor P2P Check Tests ---


class TestNodeMonitorP2PCheck:
    """Tests for NodeMonitor._check_p2p()."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_node_monitor()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_node_monitor()

    @pytest.mark.asyncio
    async def test_check_p2p_no_ip(self) -> None:
        """P2P check should fail if node has no IP."""
        monitor = NodeMonitor()
        node = MockClusterNode(name="node-1")  # No IP
        result = await monitor._check_p2p(node)
        assert result.healthy is False
        assert result.error == "No IP address configured"
        assert result.layer == HealthCheckLayer.P2P

    @pytest.mark.asyncio
    async def test_check_p2p_uses_best_ip(self) -> None:
        """P2P check should prefer best_ip over tailscale_ip."""
        monitor = NodeMonitor()
        node = MockClusterNode(
            name="node-1",
            best_ip="192.168.1.1",
            tailscale_ip="100.1.2.3",
        )

        # Create a mock that tracks calls
        call_url: str | None = None

        class MockResponse:
            status = 200

            async def json(self):
                return {"status": "healthy"}

        class MockSession:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def get(self, url, *args, **kwargs):
                nonlocal call_url
                call_url = url

                class MockContext:
                    async def __aenter__(inner_self):
                        return MockResponse()

                    async def __aexit__(inner_self, *args):
                        pass

                return MockContext()

        with patch("aiohttp.ClientSession", MockSession):
            result = await monitor._check_p2p(node)

            # Verify best_ip was used
            assert call_url is not None
            assert "192.168.1.1" in call_url

    @pytest.mark.asyncio
    async def test_check_p2p_success(self) -> None:
        """P2P check should return healthy on 200 response."""
        monitor = NodeMonitor()
        node = MockClusterNode(name="node-1", tailscale_ip="100.1.2.3")

        class MockResponse:
            status = 200

            async def json(self):
                return {"status": "healthy"}

        class MockSession:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def get(self, url, *args, **kwargs):
                class MockContext:
                    async def __aenter__(inner_self):
                        return MockResponse()

                    async def __aexit__(inner_self, *args):
                        pass

                return MockContext()

        with patch("aiohttp.ClientSession", MockSession):
            result = await monitor._check_p2p(node)
            assert result.healthy is True
            assert result.layer == HealthCheckLayer.P2P
            assert result.details["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_check_p2p_http_error(self) -> None:
        """P2P check should return unhealthy on non-200 response."""
        monitor = NodeMonitor()
        node = MockClusterNode(name="node-1", tailscale_ip="100.1.2.3")

        class MockResponse:
            status = 500

        class MockSession:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def get(self, url, *args, **kwargs):
                class MockContext:
                    async def __aenter__(inner_self):
                        return MockResponse()

                    async def __aexit__(inner_self, *args):
                        pass

                return MockContext()

        with patch("aiohttp.ClientSession", MockSession):
            result = await monitor._check_p2p(node)
            assert result.healthy is False
            assert "HTTP 500" in result.error

    @pytest.mark.asyncio
    async def test_check_p2p_timeout(self) -> None:
        """P2P check should handle timeout."""
        monitor = NodeMonitor()
        node = MockClusterNode(name="node-1", tailscale_ip="100.1.2.3")

        class MockSession:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def get(self, url, *args, **kwargs):
                class MockContext:
                    async def __aenter__(inner_self):
                        raise asyncio.TimeoutError()

                    async def __aexit__(inner_self, *args):
                        pass

                return MockContext()

        with patch("aiohttp.ClientSession", MockSession):
            result = await monitor._check_p2p(node)
            assert result.healthy is False
            assert "timeout" in result.error.lower()


# --- NodeMonitor SSH Check Tests ---


class TestNodeMonitorSSHCheck:
    """Tests for NodeMonitor._check_ssh()."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_node_monitor()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_node_monitor()

    @pytest.mark.asyncio
    async def test_check_ssh_no_ip(self) -> None:
        """SSH check should fail if node has no IP."""
        monitor = NodeMonitor()
        node = MockClusterNode(name="node-1")  # No IP
        result = await monitor._check_ssh(node)
        assert result.healthy is False
        assert result.error == "No IP address configured"
        assert result.layer == HealthCheckLayer.SSH

    @pytest.mark.asyncio
    async def test_check_ssh_success(self) -> None:
        """SSH check should return healthy on successful echo."""
        monitor = NodeMonitor()
        node = MockClusterNode(name="node-1", tailscale_ip="100.1.2.3")

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.wait = AsyncMock()
        mock_proc.stderr = None

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await monitor._check_ssh(node)
            assert result.healthy is True
            assert result.layer == HealthCheckLayer.SSH

    @pytest.mark.asyncio
    async def test_check_ssh_failure(self) -> None:
        """SSH check should return unhealthy on non-zero exit."""
        monitor = NodeMonitor()
        node = MockClusterNode(name="node-1", tailscale_ip="100.1.2.3")

        mock_proc = AsyncMock()
        mock_proc.returncode = 255
        mock_proc.wait = AsyncMock()
        mock_proc.stderr = AsyncMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"Connection refused")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await monitor._check_ssh(node)
            assert result.healthy is False
            assert "exit code 255" in result.error.lower()

    @pytest.mark.asyncio
    async def test_check_ssh_timeout(self) -> None:
        """SSH check should handle timeout."""
        monitor = NodeMonitor()
        node = MockClusterNode(name="node-1", tailscale_ip="100.1.2.3")

        mock_proc = AsyncMock()
        mock_proc.kill = MagicMock()

        async def wait_timeout():
            raise asyncio.TimeoutError

        mock_proc.wait = wait_timeout

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await monitor._check_ssh(node)
            assert result.healthy is False
            assert "timeout" in result.error.lower()
            mock_proc.kill.assert_called_once()


# --- NodeMonitor GPU Check Tests ---


class TestNodeMonitorGPUCheck:
    """Tests for NodeMonitor._check_gpu()."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_node_monitor()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_node_monitor()

    @pytest.mark.asyncio
    async def test_check_gpu_no_ip(self) -> None:
        """GPU check should fail if node has no IP."""
        monitor = NodeMonitor()
        node = MockClusterNode(name="node-1", is_gpu_node=True)
        result = await monitor._check_gpu(node)
        assert result.healthy is False
        assert result.error == "No IP address"
        assert result.layer == HealthCheckLayer.GPU

    @pytest.mark.asyncio
    async def test_check_gpu_success(self) -> None:
        """GPU check should return healthy on successful nvidia-smi."""
        monitor = NodeMonitor()
        node = MockClusterNode(
            name="node-1",
            tailscale_ip="100.1.2.3",
            is_gpu_node=True,
        )

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(b"NVIDIA A100, 80GB, 50%", b"")
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await monitor._check_gpu(node)
            assert result.healthy is True
            assert result.layer == HealthCheckLayer.GPU
            assert "NVIDIA A100" in result.details["gpu_info"]

    @pytest.mark.asyncio
    async def test_check_gpu_nvidia_smi_failed(self) -> None:
        """GPU check should return unhealthy if nvidia-smi fails."""
        monitor = NodeMonitor()
        node = MockClusterNode(
            name="node-1",
            tailscale_ip="100.1.2.3",
            is_gpu_node=True,
        )

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"nvidia-smi not found"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await monitor._check_gpu(node)
            assert result.healthy is False
            assert "nvidia-smi failed" in result.error


# --- NodeMonitor Provider Check Tests ---


class TestNodeMonitorProviderCheck:
    """Tests for NodeMonitor._check_provider_status()."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_node_monitor()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_node_monitor()

    @pytest.mark.asyncio
    async def test_check_provider_no_provider(self) -> None:
        """Provider check should skip if no provider configured."""
        monitor = NodeMonitor()
        node = MockClusterNode(name="node-1", tailscale_ip="100.1.2.3")
        result = await monitor._check_provider_status(node)
        assert result.healthy is True  # Skipped = healthy
        assert result.details["skipped"] == "no provider configured"

    @pytest.mark.asyncio
    async def test_check_provider_no_instance_id(self) -> None:
        """Provider check should skip if no instance_id."""
        monitor = NodeMonitor()
        node = MockClusterNode(
            name="node-1",
            tailscale_ip="100.1.2.3",
            provider="vast",
            # No instance_id - should result in skip
        )

        # Since the function does a local import, we need to mock it via the module
        # The get_provider function returns a mock provider client
        mock_provider = MagicMock()

        # Patch the import inside _check_provider_status
        with patch.dict(
            "sys.modules",
            {"app.coordination.providers.registry": MagicMock(get_provider=lambda x: mock_provider)},
        ):
            result = await monitor._check_provider_status(node)
            assert result.healthy is True
            assert result.details["skipped"] == "no instance_id"


# --- NodeMonitor Process Health Result Tests ---


class TestNodeMonitorProcessResult:
    """Tests for NodeMonitor._process_health_result()."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_node_monitor()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_node_monitor()

    @pytest.mark.asyncio
    async def test_process_healthy_result(self) -> None:
        """Healthy result should reset failure count."""
        monitor = NodeMonitor()
        node = MockClusterNode(name="node-1", tailscale_ip="100.1.2.3")
        monitor._failure_counts["node-1"] = 5  # Previous failures

        result = NodeHealthResult(
            node_id="node-1",
            layer=HealthCheckLayer.ALL,
            healthy=True,
            latency_ms=50.0,
        )

        await monitor._process_health_result(node, result)

        assert monitor._failure_counts["node-1"] == 0
        assert "node-1" in monitor._last_healthy

    @pytest.mark.asyncio
    async def test_process_unhealthy_result(self) -> None:
        """Unhealthy result should increment failure count."""
        monitor = NodeMonitor()
        node = MockClusterNode(name="node-1", tailscale_ip="100.1.2.3")
        monitor._failure_counts["node-1"] = 1

        result = NodeHealthResult(
            node_id="node-1",
            layer=HealthCheckLayer.P2P,
            healthy=False,
            latency_ms=15000.0,
            error="P2P timeout",
        )

        await monitor._process_health_result(node, result)

        assert monitor._failure_counts["node-1"] == 2

    @pytest.mark.asyncio
    async def test_process_result_tracks_history(self) -> None:
        """Results should be tracked in history."""
        monitor = NodeMonitor()
        node = MockClusterNode(name="node-1", tailscale_ip="100.1.2.3")

        result = NodeHealthResult(
            node_id="node-1",
            layer=HealthCheckLayer.P2P,
            healthy=True,
            latency_ms=50.0,
        )

        await monitor._process_health_result(node, result)

        assert len(monitor._health_history["node-1"]) == 1
        assert monitor._health_history["node-1"][0] == result

    @pytest.mark.asyncio
    async def test_process_result_limits_history(self) -> None:
        """History should be limited to 100 entries."""
        monitor = NodeMonitor()
        node = MockClusterNode(name="node-1", tailscale_ip="100.1.2.3")
        monitor._health_history["node-1"] = [MagicMock()] * 100  # Fill history

        result = NodeHealthResult(
            node_id="node-1",
            layer=HealthCheckLayer.P2P,
            healthy=True,
            latency_ms=50.0,
        )

        await monitor._process_health_result(node, result)

        assert len(monitor._health_history["node-1"]) == 100  # Still 100

    @pytest.mark.asyncio
    async def test_emit_unhealthy_after_threshold(self) -> None:
        """NODE_UNHEALTHY should be emitted after threshold failures."""
        monitor = NodeMonitor()
        node = MockClusterNode(name="node-1", tailscale_ip="100.1.2.3")
        monitor._failure_counts["node-1"] = 2  # One below threshold

        result = NodeHealthResult(
            node_id="node-1",
            layer=HealthCheckLayer.P2P,
            healthy=False,
            latency_ms=15000.0,
            error="P2P timeout",
        )

        with patch.object(monitor, "_emit_node_unhealthy") as mock_emit:
            await monitor._process_health_result(node, result)
            mock_emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_recovery_after_higher_threshold(self) -> None:
        """RECOVERY_INITIATED should be emitted after recovery threshold."""
        monitor = NodeMonitor()
        node = MockClusterNode(name="node-1", tailscale_ip="100.1.2.3")
        monitor._failure_counts["node-1"] = 4  # One below recovery threshold

        result = NodeHealthResult(
            node_id="node-1",
            layer=HealthCheckLayer.P2P,
            healthy=False,
            latency_ms=15000.0,
            error="P2P timeout",
        )

        with patch.object(monitor, "_emit_recovery_needed") as mock_emit:
            await monitor._process_health_result(node, result)
            mock_emit.assert_called_once()


# --- NodeMonitor Run Cycle Tests ---


class TestNodeMonitorRunCycle:
    """Tests for NodeMonitor._run_cycle()."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_node_monitor()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_node_monitor()

    @pytest.mark.asyncio
    async def test_run_cycle_no_nodes(self) -> None:
        """Run cycle should return early if no nodes."""
        monitor = NodeMonitor()

        with patch.object(monitor, "_load_nodes_from_config") as mock_load:
            await monitor._run_cycle()
            mock_load.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_cycle_checks_all_nodes(self) -> None:
        """Run cycle should check all nodes."""
        nodes = [
            MockClusterNode(name="node-1", tailscale_ip="100.1.2.3"),
            MockClusterNode(name="node-2", tailscale_ip="100.1.2.4"),
        ]
        monitor = NodeMonitor(nodes=nodes)

        with patch.object(
            monitor, "_check_node_health", return_value=NodeHealthResult(
                node_id="test",
                layer=HealthCheckLayer.ALL,
                healthy=True,
                latency_ms=50.0,
            )
        ) as mock_check:
            with patch.object(monitor, "_process_health_result"):
                await monitor._run_cycle()
                assert mock_check.call_count == 2


# --- NodeMonitor Status Tests ---


class TestNodeMonitorStatus:
    """Tests for NodeMonitor status methods."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_node_monitor()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_node_monitor()

    def test_get_node_status_nonexistent(self) -> None:
        """get_node_status should handle nonexistent nodes."""
        monitor = NodeMonitor()
        status = monitor.get_node_status("nonexistent")
        assert status["node_id"] == "nonexistent"
        assert status["healthy"] is None
        assert status["consecutive_failures"] == 0
        assert status["last_healthy"] is None
        assert status["last_check"] is None

    def test_get_node_status_with_history(self) -> None:
        """get_node_status should return latest status."""
        monitor = NodeMonitor()
        result = NodeHealthResult(
            node_id="node-1",
            layer=HealthCheckLayer.ALL,
            healthy=True,
            latency_ms=50.0,
        )
        monitor._health_history["node-1"] = [result]
        monitor._failure_counts["node-1"] = 0
        monitor._last_healthy["node-1"] = datetime.now()

        status = monitor.get_node_status("node-1")
        assert status["node_id"] == "node-1"
        assert status["healthy"] is True
        assert status["consecutive_failures"] == 0
        assert status["last_healthy"] is not None
        assert status["last_check"] is not None

    def test_get_all_node_statuses(self) -> None:
        """get_all_node_statuses should return status for all nodes."""
        nodes = [
            MockClusterNode(name="node-1", tailscale_ip="100.1.2.3"),
            MockClusterNode(name="node-2", tailscale_ip="100.1.2.4"),
        ]
        monitor = NodeMonitor(nodes=nodes)

        statuses = monitor.get_all_node_statuses()
        assert len(statuses) == 2
        assert "node-1" in statuses
        assert "node-2" in statuses


# --- NodeMonitor Health Check Tests ---


class TestNodeMonitorHealthCheck:
    """Tests for NodeMonitor.health_check()."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_node_monitor()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_node_monitor()

    def test_health_check_all_healthy(self) -> None:
        """health_check should report healthy when all nodes healthy."""
        nodes = [
            MockClusterNode(name="node-1", tailscale_ip="100.1.2.3"),
            MockClusterNode(name="node-2", tailscale_ip="100.1.2.4"),
        ]
        monitor = NodeMonitor(nodes=nodes)
        monitor._failure_counts["node-1"] = 0
        monitor._failure_counts["node-2"] = 0

        health = monitor.health_check()
        assert health.healthy is True
        assert "2 nodes" in health.message
        assert health.details["unhealthy_count"] == 0

    def test_health_check_some_unhealthy(self) -> None:
        """health_check should report unhealthy when nodes fail."""
        nodes = [
            MockClusterNode(name="node-1", tailscale_ip="100.1.2.3"),
            MockClusterNode(name="node-2", tailscale_ip="100.1.2.4"),
        ]
        monitor = NodeMonitor(nodes=nodes)
        monitor._failure_counts["node-1"] = 5  # Above threshold
        monitor._failure_counts["node-2"] = 0

        health = monitor.health_check()
        assert health.healthy is False
        assert health.details["unhealthy_count"] == 1


# --- Singleton Tests ---


class TestSingleton:
    """Tests for singleton pattern."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_node_monitor()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_node_monitor()

    def test_get_node_monitor_creates_instance(self) -> None:
        """get_node_monitor should create instance."""
        monitor = get_node_monitor()
        assert monitor is not None
        assert isinstance(monitor, NodeMonitor)

    def test_get_node_monitor_returns_same_instance(self) -> None:
        """get_node_monitor should return same instance."""
        monitor1 = get_node_monitor()
        monitor2 = get_node_monitor()
        assert monitor1 is monitor2

    def test_reset_node_monitor(self) -> None:
        """reset_node_monitor should clear singleton."""
        monitor1 = get_node_monitor()
        reset_node_monitor()
        monitor2 = get_node_monitor()
        assert monitor1 is not monitor2


# --- NodeMonitor Check Node Health Tests ---


class TestNodeMonitorCheckNodeHealth:
    """Tests for NodeMonitor._check_node_health()."""

    def setup_method(self) -> None:
        """Reset singleton before each test."""
        reset_node_monitor()

    def teardown_method(self) -> None:
        """Reset singleton after each test."""
        reset_node_monitor()

    @pytest.mark.asyncio
    async def test_check_node_health_p2p_fails(self) -> None:
        """Should return early on P2P failure."""
        monitor = NodeMonitor()
        node = MockClusterNode(name="node-1", tailscale_ip="100.1.2.3")

        p2p_fail = NodeHealthResult(
            node_id="node-1",
            layer=HealthCheckLayer.P2P,
            healthy=False,
            latency_ms=15000.0,
            error="P2P timeout",
        )

        with patch.object(monitor, "_check_p2p", return_value=p2p_fail):
            result = await monitor._check_node_health(node)
            assert result.healthy is False
            assert result.layer == HealthCheckLayer.P2P

    @pytest.mark.asyncio
    async def test_check_node_health_ssh_fails(self) -> None:
        """Should return on SSH failure after P2P passes."""
        monitor = NodeMonitor()
        node = MockClusterNode(name="node-1", tailscale_ip="100.1.2.3")

        p2p_pass = NodeHealthResult(
            node_id="node-1",
            layer=HealthCheckLayer.P2P,
            healthy=True,
            latency_ms=50.0,
        )
        ssh_fail = NodeHealthResult(
            node_id="node-1",
            layer=HealthCheckLayer.SSH,
            healthy=False,
            latency_ms=30000.0,
            error="SSH timeout",
        )

        with patch.object(monitor, "_check_p2p", return_value=p2p_pass):
            with patch.object(monitor, "_check_ssh", return_value=ssh_fail):
                result = await monitor._check_node_health(node)
                assert result.healthy is False
                assert result.layer == HealthCheckLayer.SSH

    @pytest.mark.asyncio
    async def test_check_node_health_all_pass(self) -> None:
        """Should return ALL layer when all checks pass."""
        config = NodeMonitorConfig(
            gpu_check_enabled=False,
            provider_check_enabled=False,
        )
        monitor = NodeMonitor(config=config)
        node = MockClusterNode(name="node-1", tailscale_ip="100.1.2.3")

        p2p_pass = NodeHealthResult(
            node_id="node-1",
            layer=HealthCheckLayer.P2P,
            healthy=True,
            latency_ms=50.0,
        )
        ssh_pass = NodeHealthResult(
            node_id="node-1",
            layer=HealthCheckLayer.SSH,
            healthy=True,
            latency_ms=100.0,
        )

        with patch.object(monitor, "_check_p2p", return_value=p2p_pass):
            with patch.object(monitor, "_check_ssh", return_value=ssh_pass):
                result = await monitor._check_node_health(node)
                assert result.healthy is True
                assert result.layer == HealthCheckLayer.ALL
