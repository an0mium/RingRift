"""Tests for discovery module.

Tests the worker discovery components:
- WorkerInfo: Worker information dataclass
- WorkerDiscovery: Bonjour/mDNS discovery class
- parse_manual_workers: Manual worker configuration parsing
- verify_worker_health: Health check functionality
"""

from __future__ import annotations

import threading
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from app.distributed.discovery import (
    SERVICE_TYPE,
    WorkerDiscovery,
    WorkerInfo,
    filter_healthy_workers,
    parse_manual_workers,
    verify_worker_health,
)


# ============================================================================
# WorkerInfo Tests
# ============================================================================


class TestWorkerInfo:
    """Tests for the WorkerInfo dataclass."""

    def test_basic_creation(self):
        """Test basic worker info creation."""
        worker = WorkerInfo(
            worker_id="worker-1",
            address="192.168.1.10",
            port=8765,
            hostname="mac-mini-1",
        )

        assert worker.worker_id == "worker-1"
        assert worker.address == "192.168.1.10"
        assert worker.port == 8765
        assert worker.hostname == "mac-mini-1"

    def test_default_discovered_at(self):
        """Test discovered_at defaults to now."""
        worker = WorkerInfo(
            worker_id="worker-1",
            address="192.168.1.10",
            port=8765,
            hostname="mac-mini-1",
        )

        assert worker.discovered_at is not None
        assert isinstance(worker.discovered_at, datetime)

    def test_default_properties(self):
        """Test properties defaults to empty dict."""
        worker = WorkerInfo(
            worker_id="worker-1",
            address="192.168.1.10",
            port=8765,
            hostname="mac-mini-1",
        )

        assert worker.properties == {}

    def test_custom_properties(self):
        """Test custom properties."""
        worker = WorkerInfo(
            worker_id="worker-1",
            address="192.168.1.10",
            port=8765,
            hostname="mac-mini-1",
            properties={"gpu": "M1 Max", "memory": "64GB"},
        )

        assert worker.properties["gpu"] == "M1 Max"
        assert worker.properties["memory"] == "64GB"

    def test_url_property(self):
        """Test url property returns address:port."""
        worker = WorkerInfo(
            worker_id="worker-1",
            address="192.168.1.10",
            port=8765,
            hostname="mac-mini-1",
        )

        assert worker.url == "192.168.1.10:8765"

    def test_url_with_different_port(self):
        """Test url with non-default port."""
        worker = WorkerInfo(
            worker_id="worker-1",
            address="10.0.0.5",
            port=9000,
            hostname="node-1",
        )

        assert worker.url == "10.0.0.5:9000"

    def test_hash(self):
        """Test workers can be hashed."""
        worker = WorkerInfo(
            worker_id="worker-1",
            address="192.168.1.10",
            port=8765,
            hostname="mac-mini-1",
        )

        # Should be hashable
        hash_val = hash(worker)
        assert isinstance(hash_val, int)

    def test_hash_based_on_address_port(self):
        """Test hash is based on address and port."""
        worker1 = WorkerInfo(
            worker_id="worker-1",
            address="192.168.1.10",
            port=8765,
            hostname="mac-mini-1",
        )
        worker2 = WorkerInfo(
            worker_id="worker-2",  # Different ID
            address="192.168.1.10",
            port=8765,
            hostname="mac-mini-2",  # Different hostname
        )

        # Same address+port = same hash
        assert hash(worker1) == hash(worker2)

    def test_equality(self):
        """Test equality based on address and port."""
        worker1 = WorkerInfo(
            worker_id="worker-1",
            address="192.168.1.10",
            port=8765,
            hostname="mac-mini-1",
        )
        worker2 = WorkerInfo(
            worker_id="worker-1",
            address="192.168.1.10",
            port=8765,
            hostname="mac-mini-1",
        )

        assert worker1 == worker2

    def test_inequality_different_address(self):
        """Test inequality with different address."""
        worker1 = WorkerInfo(
            worker_id="worker-1",
            address="192.168.1.10",
            port=8765,
            hostname="mac-mini-1",
        )
        worker2 = WorkerInfo(
            worker_id="worker-1",
            address="192.168.1.11",
            port=8765,
            hostname="mac-mini-1",
        )

        assert worker1 != worker2

    def test_inequality_different_port(self):
        """Test inequality with different port."""
        worker1 = WorkerInfo(
            worker_id="worker-1",
            address="192.168.1.10",
            port=8765,
            hostname="mac-mini-1",
        )
        worker2 = WorkerInfo(
            worker_id="worker-1",
            address="192.168.1.10",
            port=9000,
            hostname="mac-mini-1",
        )

        assert worker1 != worker2

    def test_inequality_with_non_worker(self):
        """Test inequality with non-WorkerInfo objects."""
        worker = WorkerInfo(
            worker_id="worker-1",
            address="192.168.1.10",
            port=8765,
            hostname="mac-mini-1",
        )

        assert worker != "not a worker"
        assert worker != 123
        assert worker != None

    def test_workers_in_set(self):
        """Test workers can be used in sets."""
        worker1 = WorkerInfo(
            worker_id="worker-1",
            address="192.168.1.10",
            port=8765,
            hostname="mac-mini-1",
        )
        worker2 = WorkerInfo(
            worker_id="worker-2",
            address="192.168.1.11",
            port=8765,
            hostname="mac-mini-2",
        )

        workers = {worker1, worker2}
        assert len(workers) == 2


# ============================================================================
# WorkerDiscovery Tests
# ============================================================================


class TestWorkerDiscovery:
    """Tests for the WorkerDiscovery class."""

    def test_init_default_service_type(self):
        """Test default service type."""
        discovery = WorkerDiscovery()
        assert discovery.service_type == SERVICE_TYPE

    def test_init_custom_service_type(self):
        """Test custom service type."""
        discovery = WorkerDiscovery(service_type="_custom._tcp.local.")
        assert discovery.service_type == "_custom._tcp.local."

    def test_init_empty_workers(self):
        """Test workers dict starts empty."""
        discovery = WorkerDiscovery()
        assert len(discovery._workers) == 0

    def test_init_not_running(self):
        """Test discovery starts not running."""
        discovery = WorkerDiscovery()
        assert discovery._running is False

    def test_get_workers_empty(self):
        """Test get_workers returns empty list initially."""
        discovery = WorkerDiscovery()
        assert discovery.get_workers() == []

    def test_get_worker_urls_empty(self):
        """Test get_worker_urls returns empty list initially."""
        discovery = WorkerDiscovery()
        assert discovery.get_worker_urls() == []

    def test_get_workers_returns_copy(self):
        """Test get_workers returns a copy of workers list."""
        discovery = WorkerDiscovery()

        # Add a worker manually
        worker = WorkerInfo(
            worker_id="test",
            address="192.168.1.10",
            port=8765,
            hostname="test",
        )
        discovery._workers[worker.url] = worker

        workers = discovery.get_workers()
        assert len(workers) == 1

        # Modifying returned list doesn't affect internal state
        workers.clear()
        assert len(discovery.get_workers()) == 1

    def test_get_worker_urls(self):
        """Test get_worker_urls returns urls."""
        discovery = WorkerDiscovery()

        worker1 = WorkerInfo(
            worker_id="test1",
            address="192.168.1.10",
            port=8765,
            hostname="test1",
        )
        worker2 = WorkerInfo(
            worker_id="test2",
            address="192.168.1.11",
            port=9000,
            hostname="test2",
        )
        discovery._workers[worker1.url] = worker1
        discovery._workers[worker2.url] = worker2

        urls = discovery.get_worker_urls()
        assert "192.168.1.10:8765" in urls
        assert "192.168.1.11:9000" in urls

    def test_start_without_zeroconf(self):
        """Test start returns False when zeroconf not available."""
        discovery = WorkerDiscovery()

        with patch.dict("sys.modules", {"zeroconf": None}):
            # Mock import to raise ImportError
            with patch(
                "app.distributed.discovery.WorkerDiscovery.start",
                return_value=False
            ):
                result = discovery.start()
                assert result is False

    def test_stop_not_running(self):
        """Test stop when not running."""
        discovery = WorkerDiscovery()
        # Should not raise
        discovery.stop()
        assert discovery._running is False

    def test_context_manager(self):
        """Test context manager protocol."""
        discovery = WorkerDiscovery()

        # Mock start to not actually use network
        discovery.start = MagicMock(return_value=True)
        discovery.stop = MagicMock()

        with discovery as d:
            assert d is discovery
            discovery.start.assert_called_once()

        discovery.stop.assert_called_once()

    def test_thread_safety(self):
        """Test worker access is thread-safe."""
        discovery = WorkerDiscovery()

        def add_workers():
            for i in range(100):
                worker = WorkerInfo(
                    worker_id=f"worker-{i}",
                    address=f"192.168.1.{i}",
                    port=8765,
                    hostname=f"host-{i}",
                )
                with discovery._lock:
                    discovery._workers[worker.url] = worker

        def read_workers():
            for _ in range(100):
                discovery.get_workers()

        threads = [
            threading.Thread(target=add_workers),
            threading.Thread(target=read_workers),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have all workers
        assert len(discovery._workers) == 100


# ============================================================================
# parse_manual_workers Tests
# ============================================================================


class TestParseManualWorkers:
    """Tests for the parse_manual_workers function."""

    def test_single_worker_with_port(self):
        """Test parsing single worker with port."""
        workers = parse_manual_workers("192.168.1.10:8765")

        assert len(workers) == 1
        assert workers[0].address == "192.168.1.10"
        assert workers[0].port == 8765

    def test_single_worker_without_port(self):
        """Test parsing single worker without port uses default."""
        workers = parse_manual_workers("192.168.1.10")

        assert len(workers) == 1
        assert workers[0].address == "192.168.1.10"
        assert workers[0].port == 8765  # Default port

    def test_multiple_workers(self):
        """Test parsing multiple workers."""
        workers = parse_manual_workers(
            "192.168.1.10:8765,192.168.1.11:8766,192.168.1.12:8767"
        )

        assert len(workers) == 3
        assert workers[0].address == "192.168.1.10"
        assert workers[0].port == 8765
        assert workers[1].address == "192.168.1.11"
        assert workers[1].port == 8766
        assert workers[2].address == "192.168.1.12"
        assert workers[2].port == 8767

    def test_mixed_with_and_without_port(self):
        """Test parsing mixed workers with and without ports."""
        workers = parse_manual_workers("192.168.1.10:9000,192.168.1.11")

        assert len(workers) == 2
        assert workers[0].port == 9000
        assert workers[1].port == 8765  # Default

    def test_handles_spaces(self):
        """Test parsing handles whitespace."""
        workers = parse_manual_workers(
            "192.168.1.10:8765 , 192.168.1.11:8766 , 192.168.1.12:8767"
        )

        assert len(workers) == 3

    def test_empty_string(self):
        """Test parsing empty string."""
        workers = parse_manual_workers("")
        assert len(workers) == 0

    def test_empty_entries(self):
        """Test parsing with empty entries."""
        workers = parse_manual_workers("192.168.1.10:8765,,192.168.1.11:8766,")

        assert len(workers) == 2

    def test_worker_ids(self):
        """Test worker IDs are generated."""
        workers = parse_manual_workers("192.168.1.10:8765")

        assert workers[0].worker_id == "manual-192.168.1.10"

    def test_hostnames(self):
        """Test hostnames are set to address."""
        workers = parse_manual_workers("192.168.1.10:8765")

        assert workers[0].hostname == "192.168.1.10"

    def test_ipv6_address(self):
        """Test IPv6 addresses with port."""
        workers = parse_manual_workers("[::1]:8765")

        assert len(workers) == 1
        assert workers[0].address == "[::1]"
        assert workers[0].port == 8765


# ============================================================================
# verify_worker_health Tests
# ============================================================================


class TestVerifyWorkerHealth:
    """Tests for the verify_worker_health function."""

    def test_healthy_worker(self):
        """Test healthy worker returns True."""
        worker = WorkerInfo(
            worker_id="test",
            address="localhost",
            port=8765,
            hostname="test",
        )

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"status": "healthy"}'
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = verify_worker_health(worker)
            assert result is True

    def test_unhealthy_worker(self):
        """Test unhealthy worker returns False."""
        worker = WorkerInfo(
            worker_id="test",
            address="localhost",
            port=8765,
            hostname="test",
        )

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"status": "unhealthy"}'
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = verify_worker_health(worker)
            assert result is False

    def test_connection_error(self):
        """Test connection error returns False."""
        worker = WorkerInfo(
            worker_id="test",
            address="192.168.1.10",
            port=8765,
            hostname="test",
        )

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = Exception("Connection refused")

            result = verify_worker_health(worker)
            assert result is False

    def test_timeout_error(self):
        """Test timeout returns False."""
        worker = WorkerInfo(
            worker_id="test",
            address="192.168.1.10",
            port=8765,
            hostname="test",
        )

        with patch("urllib.request.urlopen") as mock_urlopen:
            import socket
            mock_urlopen.side_effect = socket.timeout("timed out")

            result = verify_worker_health(worker, timeout=1.0)
            assert result is False

    def test_invalid_json(self):
        """Test invalid JSON response returns False."""
        worker = WorkerInfo(
            worker_id="test",
            address="localhost",
            port=8765,
            hostname="test",
        )

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b'not valid json'
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = verify_worker_health(worker)
            assert result is False


# ============================================================================
# filter_healthy_workers Tests
# ============================================================================


class TestFilterHealthyWorkers:
    """Tests for the filter_healthy_workers function."""

    def test_all_healthy(self):
        """Test all healthy workers are returned."""
        workers = [
            WorkerInfo(
                worker_id=f"worker-{i}",
                address=f"192.168.1.{i}",
                port=8765,
                hostname=f"host-{i}",
            )
            for i in range(3)
        ]

        with patch(
            "app.distributed.discovery.verify_worker_health",
            return_value=True
        ):
            healthy = filter_healthy_workers(workers)
            assert len(healthy) == 3

    def test_all_unhealthy(self):
        """Test no workers returned when all unhealthy."""
        workers = [
            WorkerInfo(
                worker_id=f"worker-{i}",
                address=f"192.168.1.{i}",
                port=8765,
                hostname=f"host-{i}",
            )
            for i in range(3)
        ]

        with patch(
            "app.distributed.discovery.verify_worker_health",
            return_value=False
        ):
            healthy = filter_healthy_workers(workers)
            assert len(healthy) == 0

    def test_mixed_health(self):
        """Test only healthy workers returned."""
        workers = [
            WorkerInfo(
                worker_id=f"worker-{i}",
                address=f"192.168.1.{i}",
                port=8765,
                hostname=f"host-{i}",
            )
            for i in range(3)
        ]

        # First and third are healthy
        health_results = [True, False, True]

        with patch(
            "app.distributed.discovery.verify_worker_health",
            side_effect=health_results
        ):
            healthy = filter_healthy_workers(workers)
            assert len(healthy) == 2

    def test_empty_list(self):
        """Test empty worker list."""
        healthy = filter_healthy_workers([])
        assert len(healthy) == 0


# ============================================================================
# Constants Tests
# ============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_service_type_format(self):
        """Test service type follows mDNS convention."""
        assert SERVICE_TYPE.startswith("_")
        assert SERVICE_TYPE.endswith(".local.")
        assert "_tcp." in SERVICE_TYPE or "_udp." in SERVICE_TYPE
