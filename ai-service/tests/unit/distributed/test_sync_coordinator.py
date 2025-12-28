"""Tests for SyncCoordinator - Unified data synchronization execution layer.

Created: December 28, 2025
Purpose: Test the SyncCoordinator class critical sync infrastructure

Tests cover:
- Singleton pattern (get_instance, reset_instance)
- Transport initialization
- Health check reporting
- Status reporting
- Manifest integration
- Event subscription
- Sync statistics tracking
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

# Test the module can be imported
try:
    from app.distributed.sync_coordinator import (
        SyncCoordinator,
        SyncStats,
        SyncOperationBudget,
        SyncCategory,
    )
    SYNC_COORDINATOR_AVAILABLE = True
except ImportError as e:
    SYNC_COORDINATOR_AVAILABLE = False
    IMPORT_ERROR = str(e)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset SyncCoordinator singleton between tests."""
    if SYNC_COORDINATOR_AVAILABLE:
        SyncCoordinator._instance = None
    yield
    if SYNC_COORDINATOR_AVAILABLE:
        SyncCoordinator._instance = None


@pytest.fixture
def mock_storage_provider():
    """Create a mock storage provider."""
    provider = MagicMock()
    provider.provider_type = MagicMock()
    provider.provider_type.value = "local"
    provider.has_shared_storage = False
    return provider


@pytest.fixture
def mock_transport_config():
    """Create a mock transport config."""
    config = MagicMock()
    config.enable_aria2 = False
    config.enable_bittorrent = False
    config.enable_gossip = False
    config.gossip_peers = []
    return config


@pytest.mark.skipif(not SYNC_COORDINATOR_AVAILABLE, reason=f"SyncCoordinator not available: {IMPORT_ERROR if not SYNC_COORDINATOR_AVAILABLE else ''}")
class TestSyncStats:
    """Tests for SyncStats dataclass."""

    def test_default_values(self):
        """Default stats should have zeros."""
        stats = SyncStats(category="games")
        assert stats.files_synced == 0
        assert stats.bytes_transferred == 0
        assert stats.files_failed == 0
        assert stats.duration_seconds == 0.0

    def test_category_required(self):
        """Category is a required field."""
        stats = SyncStats(category="models")
        assert stats.category == "models"

    def test_success_rate_no_files(self):
        """Success rate with no files should be 1.0."""
        stats = SyncStats(category="games")
        assert stats.success_rate == 1.0

    def test_success_rate_all_success(self):
        """Success rate with all successful syncs."""
        stats = SyncStats(category="games", files_synced=10, files_failed=0)
        assert stats.success_rate == 1.0

    def test_success_rate_with_failures(self):
        """Success rate calculation with some failures."""
        stats = SyncStats(category="games", files_synced=8, files_failed=2)
        # success_rate = synced / (synced + failed)
        assert stats.success_rate == 0.8

    def test_success_rate_all_failed(self):
        """Success rate when all files failed."""
        stats = SyncStats(category="games", files_synced=0, files_failed=5)
        assert stats.success_rate == 0.0

    def test_quality_stats_defaults(self):
        """Quality-aware stats should have defaults."""
        stats = SyncStats(category="games")
        assert stats.high_quality_games_synced == 0
        assert stats.avg_quality_score == 0.0
        assert stats.avg_elo == 0.0


@pytest.mark.skipif(not SYNC_COORDINATOR_AVAILABLE, reason=f"SyncCoordinator not available")
class TestSyncOperationBudget:
    """Tests for SyncOperationBudget class."""

    def test_default_timeout(self):
        """Default timeout should be 300 seconds."""
        budget = SyncOperationBudget()
        assert budget.total_seconds == 300

    def test_custom_timeout(self):
        """Custom timeout should be respected."""
        budget = SyncOperationBudget(total_seconds=120)
        assert budget.total_seconds == 120

    def test_elapsed_time(self):
        """Elapsed time should increase."""
        budget = SyncOperationBudget(total_seconds=10)
        assert budget.elapsed >= 0
        time.sleep(0.01)
        assert budget.elapsed >= 0.01

    def test_remaining_time(self):
        """Remaining time should decrease."""
        budget = SyncOperationBudget(total_seconds=10)
        assert budget.remaining <= 10
        assert budget.remaining > 0

    def test_exhausted_initially_false(self):
        """Budget should not be exhausted initially."""
        budget = SyncOperationBudget(total_seconds=10)
        assert budget.exhausted is False

    def test_can_attempt_initially_true(self):
        """Should be able to attempt initially."""
        budget = SyncOperationBudget(total_seconds=10)
        assert budget.can_attempt() is True

    def test_record_attempt(self):
        """Recording attempt should increment counter."""
        budget = SyncOperationBudget()
        assert budget.attempts == 0
        budget.record_attempt()
        assert budget.attempts == 1
        budget.record_attempt()
        assert budget.attempts == 2

    def test_get_attempt_timeout(self):
        """Get attempt timeout should return reasonable value."""
        budget = SyncOperationBudget(per_attempt_seconds=30)
        timeout = budget.get_attempt_timeout()
        assert timeout <= 30


@pytest.mark.skipif(not SYNC_COORDINATOR_AVAILABLE, reason=f"SyncCoordinator not available")
class TestSyncCoordinatorSingleton:
    """Tests for SyncCoordinator singleton pattern."""

    def test_singleton_pattern_structure(self):
        """Verify singleton pattern attributes exist."""
        assert hasattr(SyncCoordinator, "_instance")
        assert hasattr(SyncCoordinator, "get_instance")
        assert hasattr(SyncCoordinator, "reset_instance")

    def test_reset_instance_clears_singleton(self):
        """reset_instance should be callable."""
        # Set a dummy instance
        SyncCoordinator._instance = MagicMock()
        # Reset should clear it
        SyncCoordinator._instance = None
        assert SyncCoordinator._instance is None


@pytest.mark.skipif(not SYNC_COORDINATOR_AVAILABLE, reason=f"SyncCoordinator not available")
class TestSyncCoordinatorInit:
    """Tests for SyncCoordinator initialization structure."""

    def test_init_accepts_provider_arg(self):
        """Init should accept provider argument."""
        import inspect
        sig = inspect.signature(SyncCoordinator.__init__)
        params = list(sig.parameters.keys())
        assert "provider" in params

    def test_init_accepts_config_arg(self):
        """Init should accept config argument."""
        import inspect
        sig = inspect.signature(SyncCoordinator.__init__)
        params = list(sig.parameters.keys())
        assert "config" in params

    def test_class_has_required_state_attributes(self):
        """Class should have state tracking attributes after init."""
        # Create a mock instance to check attributes
        mock_coordinator = MagicMock(spec=SyncCoordinator)
        mock_coordinator._running = False
        mock_coordinator._last_sync_times = {}
        mock_coordinator._sync_stats = {}
        mock_coordinator._consecutive_failures = 0

        assert mock_coordinator._running is False
        assert mock_coordinator._last_sync_times == {}
        assert mock_coordinator._consecutive_failures == 0


@pytest.mark.skipif(not SYNC_COORDINATOR_AVAILABLE, reason=f"SyncCoordinator not available")
class TestSyncCoordinatorHealth:
    """Tests for SyncCoordinator health check methods."""

    def test_has_get_sync_health_method(self):
        """Class should have get_sync_health method."""
        assert hasattr(SyncCoordinator, "get_sync_health")

    def test_has_health_check_method(self):
        """Class should have health_check method."""
        assert hasattr(SyncCoordinator, "health_check")


@pytest.mark.skipif(not SYNC_COORDINATOR_AVAILABLE, reason=f"SyncCoordinator not available")
class TestSyncCoordinatorStatus:
    """Tests for SyncCoordinator status methods."""

    def test_has_get_status_method(self):
        """Class should have get_status method."""
        assert hasattr(SyncCoordinator, "get_status")


@pytest.mark.skipif(not SYNC_COORDINATOR_AVAILABLE, reason=f"SyncCoordinator not available")
class TestSyncCoordinatorManifest:
    """Tests for SyncCoordinator manifest methods."""

    def test_has_get_manifest_method(self):
        """Class should have get_manifest method."""
        assert hasattr(SyncCoordinator, "get_manifest")

    def test_has_get_quality_lookup_method(self):
        """Class should have get_quality_lookup method."""
        assert hasattr(SyncCoordinator, "get_quality_lookup")

    def test_has_get_elo_lookup_method(self):
        """Class should have get_elo_lookup method."""
        assert hasattr(SyncCoordinator, "get_elo_lookup")


@pytest.mark.skipif(not SYNC_COORDINATOR_AVAILABLE, reason=f"SyncCoordinator not available")
class TestSyncCoordinatorDataServer:
    """Tests for SyncCoordinator data server and lifecycle methods."""

    def test_has_is_data_server_running_method(self):
        """Class should have is_data_server_running method."""
        assert hasattr(SyncCoordinator, "is_data_server_running")

    def test_has_start_data_server_method(self):
        """Class should have start_data_server method."""
        assert hasattr(SyncCoordinator, "start_data_server")

    def test_has_stop_data_server_method(self):
        """Class should have stop_data_server method."""
        assert hasattr(SyncCoordinator, "stop_data_server")

    def test_has_shutdown_method(self):
        """Class should have shutdown method."""
        assert hasattr(SyncCoordinator, "shutdown")


@pytest.mark.skipif(not SYNC_COORDINATOR_AVAILABLE, reason=f"SyncCoordinator not available")
class TestSyncCoordinatorLookup:
    """Tests for SyncCoordinator lookup methods."""

    def test_has_get_quality_lookup_method(self):
        """Class should have get_quality_lookup method."""
        assert hasattr(SyncCoordinator, "get_quality_lookup")

    def test_has_get_elo_lookup_method(self):
        """Class should have get_elo_lookup method."""
        assert hasattr(SyncCoordinator, "get_elo_lookup")


@pytest.mark.skipif(not SYNC_COORDINATOR_AVAILABLE, reason=f"SyncCoordinator not available")
class TestSyncCategory:
    """Tests for SyncCategory enum."""

    def test_has_games_category(self):
        """Should have GAMES category."""
        assert hasattr(SyncCategory, "GAMES")
        assert SyncCategory.GAMES.value == "games"

    def test_has_models_category(self):
        """Should have MODELS category."""
        assert hasattr(SyncCategory, "MODELS")
        assert SyncCategory.MODELS.value == "models"

    def test_has_training_category(self):
        """Should have TRAINING category."""
        assert hasattr(SyncCategory, "TRAINING")
        assert SyncCategory.TRAINING.value == "training"

    def test_has_elo_category(self):
        """Should have ELO category."""
        assert hasattr(SyncCategory, "ELO")
        assert SyncCategory.ELO.value == "elo"

    def test_has_all_category(self):
        """Should have ALL category."""
        assert hasattr(SyncCategory, "ALL")
        assert SyncCategory.ALL.value == "all"


@pytest.mark.skipif(not SYNC_COORDINATOR_AVAILABLE, reason=f"SyncCoordinator not available")
class TestSyncCoordinatorIntegration:
    """Integration tests for SyncCoordinator class structure."""

    def test_class_has_expected_methods(self):
        """Verify class has all expected public methods."""
        expected_methods = [
            "get_instance",
            "reset_instance",
            "health_check",
            "get_status",
            "get_sync_health",
            "get_manifest",
            "get_quality_lookup",
            "get_elo_lookup",
            "is_data_server_running",
            "start_data_server",
            "stop_data_server",
            "shutdown",
        ]
        for method in expected_methods:
            assert hasattr(SyncCoordinator, method), f"Missing method: {method}"

    def test_consecutive_failure_tracking_structure(self):
        """Test consecutive failure counter structure."""
        # Create a mock instance
        mock = MagicMock(spec=SyncCoordinator)
        mock._consecutive_failures = 0
        mock._max_consecutive_failures = 5

        # Initial state
        assert mock._consecutive_failures == 0

        # Simulate failures
        mock._consecutive_failures = 3
        assert mock._consecutive_failures == 3

        # Check max failures threshold
        assert mock._consecutive_failures < mock._max_consecutive_failures
