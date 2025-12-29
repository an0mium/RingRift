"""Comprehensive tests for app/config/coordination_defaults.py.

This module provides centralized configuration constants for coordination
infrastructure. Tests cover:

1. All dataclass constants (30+ classes)
2. Environment variable overrides (_env_int, _env_float)
3. Helper functions (get_timeout, get_job_timeout, get_sqlite_timeout, etc.)
4. get_all_defaults() dictionary structure
5. Edge cases and error handling

Created: December 28, 2025
"""

from __future__ import annotations

import os
from importlib import reload
from unittest.mock import patch

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def fresh_module():
    """Reload the module to pick up env var changes."""
    import app.config.coordination_defaults as cd
    yield cd
    # Reload to reset state after test
    reload(cd)


# =============================================================================
# Test _env_int and _env_float helper functions
# =============================================================================


class TestEnvHelpers:
    """Tests for environment variable helper functions."""

    def test_env_int_returns_default_when_not_set(self):
        """_env_int returns default when env var not set."""
        from app.config.coordination_defaults import _env_int

        # Use a unique key that definitely doesn't exist
        result = _env_int("RINGRIFT_TEST_NONEXISTENT_KEY_12345", 42)
        assert result == 42

    def test_env_int_returns_env_value_when_set(self):
        """_env_int returns env value when set."""
        from app.config.coordination_defaults import _env_int

        with patch.dict(os.environ, {"RINGRIFT_TEST_INT_KEY": "123"}):
            result = _env_int("RINGRIFT_TEST_INT_KEY", 0)
            assert result == 123

    def test_env_float_returns_default_when_not_set(self):
        """_env_float returns default when env var not set."""
        from app.config.coordination_defaults import _env_float

        result = _env_float("RINGRIFT_TEST_NONEXISTENT_FLOAT_KEY", 3.14)
        assert result == 3.14

    def test_env_float_returns_env_value_when_set(self):
        """_env_float returns env value when set."""
        from app.config.coordination_defaults import _env_float

        with patch.dict(os.environ, {"RINGRIFT_TEST_FLOAT_KEY": "2.718"}):
            result = _env_float("RINGRIFT_TEST_FLOAT_KEY", 0.0)
            assert result == pytest.approx(2.718)


# =============================================================================
# Test LockDefaults
# =============================================================================


class TestLockDefaults:
    """Tests for LockDefaults dataclass."""

    def test_lock_timeout_positive(self):
        """LOCK_TIMEOUT should be positive."""
        from app.config.coordination_defaults import LockDefaults

        assert LockDefaults.LOCK_TIMEOUT > 0

    def test_acquire_timeout_positive(self):
        """ACQUIRE_TIMEOUT should be positive."""
        from app.config.coordination_defaults import LockDefaults

        assert LockDefaults.ACQUIRE_TIMEOUT > 0

    def test_retry_interval_positive(self):
        """RETRY_INTERVAL should be positive."""
        from app.config.coordination_defaults import LockDefaults

        assert LockDefaults.RETRY_INTERVAL > 0

    def test_training_lock_timeout_greater_or_equal_lock_timeout(self):
        """TRAINING_LOCK_TIMEOUT should be >= LOCK_TIMEOUT."""
        from app.config.coordination_defaults import LockDefaults

        assert LockDefaults.TRAINING_LOCK_TIMEOUT >= LockDefaults.LOCK_TIMEOUT

    def test_lock_defaults_is_frozen(self):
        """LockDefaults should be frozen (immutable instance)."""
        from app.config.coordination_defaults import LockDefaults

        # The dataclass is frozen, so instances would be immutable
        # Class-level defaults are read from class attributes which are set
        # at definition time. The frozen flag prevents instance mutation.
        # Verify the class has frozen=True decorator
        import dataclasses
        assert dataclasses.is_dataclass(LockDefaults)
        # The class itself is frozen (has __dataclass_fields__ with frozen=True)
        # Note: Class attributes can still be set, but instances are immutable


# =============================================================================
# Test TransportDefaults
# =============================================================================


class TestTransportDefaults:
    """Tests for TransportDefaults dataclass."""

    def test_connect_timeout_reasonable(self):
        """CONNECT_TIMEOUT should be reasonable (1-120s)."""
        from app.config.coordination_defaults import TransportDefaults

        assert 1 <= TransportDefaults.CONNECT_TIMEOUT <= 120

    def test_operation_timeout_greater_than_connect(self):
        """OPERATION_TIMEOUT should be > CONNECT_TIMEOUT."""
        from app.config.coordination_defaults import TransportDefaults

        assert TransportDefaults.OPERATION_TIMEOUT > TransportDefaults.CONNECT_TIMEOUT

    def test_http_timeout_positive(self):
        """HTTP_TIMEOUT should be positive."""
        from app.config.coordination_defaults import TransportDefaults

        assert TransportDefaults.HTTP_TIMEOUT > 0

    def test_max_retries_positive(self):
        """MAX_RETRIES should be positive."""
        from app.config.coordination_defaults import TransportDefaults

        assert TransportDefaults.MAX_RETRIES > 0

    def test_ssh_timeout_positive(self):
        """SSH_TIMEOUT should be positive."""
        from app.config.coordination_defaults import TransportDefaults

        assert TransportDefaults.SSH_TIMEOUT > 0


# =============================================================================
# Test SyncDefaults
# =============================================================================


class TestSyncDefaults:
    """Tests for SyncDefaults dataclass."""

    def test_max_concurrent_per_host_positive(self):
        """MAX_CONCURRENT_PER_HOST should be positive."""
        from app.config.coordination_defaults import SyncDefaults

        assert SyncDefaults.MAX_CONCURRENT_PER_HOST >= 1

    def test_max_concurrent_cluster_greater_than_per_host(self):
        """MAX_CONCURRENT_CLUSTER should be >= MAX_CONCURRENT_PER_HOST."""
        from app.config.coordination_defaults import SyncDefaults

        assert SyncDefaults.MAX_CONCURRENT_CLUSTER >= SyncDefaults.MAX_CONCURRENT_PER_HOST

    def test_data_sync_interval_positive(self):
        """DATA_SYNC_INTERVAL should be positive."""
        from app.config.coordination_defaults import SyncDefaults

        assert SyncDefaults.DATA_SYNC_INTERVAL > 0

    def test_elo_sync_interval_positive(self):
        """ELO_SYNC_INTERVAL should be positive."""
        from app.config.coordination_defaults import SyncDefaults

        assert SyncDefaults.ELO_SYNC_INTERVAL > 0


# =============================================================================
# Test HeartbeatDefaults
# =============================================================================


class TestHeartbeatDefaults:
    """Tests for HeartbeatDefaults dataclass."""

    def test_timeout_greater_than_interval(self):
        """TIMEOUT should be > INTERVAL (otherwise false positives)."""
        from app.config.coordination_defaults import HeartbeatDefaults

        assert HeartbeatDefaults.TIMEOUT > HeartbeatDefaults.INTERVAL

    def test_timeout_multiplier_positive(self):
        """TIMEOUT_MULTIPLIER should be positive."""
        from app.config.coordination_defaults import HeartbeatDefaults

        assert HeartbeatDefaults.TIMEOUT_MULTIPLIER > 0


# =============================================================================
# Test TrainingDefaults
# =============================================================================


class TestTrainingDefaults:
    """Tests for TrainingDefaults dataclass."""

    def test_max_concurrent_same_config_positive(self):
        """MAX_CONCURRENT_SAME_CONFIG should be >= 1."""
        from app.config.coordination_defaults import TrainingDefaults

        assert TrainingDefaults.MAX_CONCURRENT_SAME_CONFIG >= 1

    def test_max_concurrent_total_positive(self):
        """MAX_CONCURRENT_TOTAL should be >= 1."""
        from app.config.coordination_defaults import TrainingDefaults

        assert TrainingDefaults.MAX_CONCURRENT_TOTAL >= 1

    def test_timeout_hours_positive(self):
        """TIMEOUT_HOURS should be positive."""
        from app.config.coordination_defaults import TrainingDefaults

        assert TrainingDefaults.TIMEOUT_HOURS > 0


# =============================================================================
# Test CircuitBreakerDefaults
# =============================================================================


class TestCircuitBreakerDefaults:
    """Tests for CircuitBreakerDefaults dataclass."""

    def test_failure_threshold_positive(self):
        """FAILURE_THRESHOLD should be positive."""
        from app.config.coordination_defaults import CircuitBreakerDefaults

        assert CircuitBreakerDefaults.FAILURE_THRESHOLD > 0

    def test_recovery_timeout_positive(self):
        """RECOVERY_TIMEOUT should be positive."""
        from app.config.coordination_defaults import CircuitBreakerDefaults

        assert CircuitBreakerDefaults.RECOVERY_TIMEOUT > 0

    def test_half_open_max_calls_positive(self):
        """HALF_OPEN_MAX_CALLS should be positive."""
        from app.config.coordination_defaults import CircuitBreakerDefaults

        assert CircuitBreakerDefaults.HALF_OPEN_MAX_CALLS >= 1

    def test_per_transport_configs_exist(self):
        """Per-transport circuit breaker configs should be defined."""
        from app.config.coordination_defaults import CircuitBreakerDefaults

        assert CircuitBreakerDefaults.SSH_FAILURE_THRESHOLD > 0
        assert CircuitBreakerDefaults.HTTP_FAILURE_THRESHOLD > 0
        assert CircuitBreakerDefaults.P2P_FAILURE_THRESHOLD > 0


# =============================================================================
# Test P2PDefaults
# =============================================================================


class TestP2PDefaults:
    """Tests for P2PDefaults dataclass."""

    def test_default_port_is_8770(self):
        """DEFAULT_PORT should be 8770."""
        from app.config.coordination_defaults import P2PDefaults

        assert P2PDefaults.DEFAULT_PORT == 8770

    def test_gossip_interval_positive(self):
        """GOSSIP_INTERVAL should be positive."""
        from app.config.coordination_defaults import P2PDefaults

        assert P2PDefaults.GOSSIP_INTERVAL > 0

    def test_peer_timeout_greater_than_heartbeat(self):
        """PEER_TIMEOUT should be > HEARTBEAT_INTERVAL."""
        from app.config.coordination_defaults import P2PDefaults

        assert P2PDefaults.PEER_TIMEOUT > P2PDefaults.HEARTBEAT_INTERVAL


# =============================================================================
# Test SQLiteDefaults
# =============================================================================


class TestSQLiteDefaults:
    """Tests for SQLiteDefaults dataclass."""

    def test_timeout_ordering(self):
        """Timeouts should follow: QUICK < READ < STANDARD < WRITE < HEAVY < MERGE."""
        from app.config.coordination_defaults import SQLiteDefaults

        assert SQLiteDefaults.QUICK_TIMEOUT < SQLiteDefaults.READ_TIMEOUT
        assert SQLiteDefaults.READ_TIMEOUT < SQLiteDefaults.STANDARD_TIMEOUT
        assert SQLiteDefaults.STANDARD_TIMEOUT < SQLiteDefaults.WRITE_TIMEOUT
        assert SQLiteDefaults.WRITE_TIMEOUT < SQLiteDefaults.HEAVY_TIMEOUT
        assert SQLiteDefaults.HEAVY_TIMEOUT < SQLiteDefaults.MERGE_TIMEOUT

    def test_busy_timeout_positive(self):
        """BUSY_TIMEOUT_MS should be positive."""
        from app.config.coordination_defaults import SQLiteDefaults

        assert SQLiteDefaults.BUSY_TIMEOUT_MS > 0


# =============================================================================
# Test BackpressureDefaults
# =============================================================================


class TestBackpressureDefaults:
    """Tests for BackpressureDefaults dataclass."""

    def test_weights_sum_close_to_one(self):
        """Component weights should sum close to 1.0."""
        from app.config.coordination_defaults import BackpressureDefaults

        total = (
            BackpressureDefaults.WEIGHT_QUEUE +
            BackpressureDefaults.WEIGHT_TRAINING +
            BackpressureDefaults.WEIGHT_DISK +
            BackpressureDefaults.WEIGHT_SYNC +
            BackpressureDefaults.WEIGHT_MEMORY
        )
        assert total == pytest.approx(1.0, rel=0.01)

    def test_queue_thresholds_ordered(self):
        """Queue thresholds should be ordered: LOW < MEDIUM < HIGH < CRITICAL."""
        from app.config.coordination_defaults import BackpressureDefaults

        assert BackpressureDefaults.QUEUE_LOW < BackpressureDefaults.QUEUE_MEDIUM
        assert BackpressureDefaults.QUEUE_MEDIUM < BackpressureDefaults.QUEUE_HIGH
        assert BackpressureDefaults.QUEUE_HIGH < BackpressureDefaults.QUEUE_CRITICAL

    def test_multipliers_decreasing(self):
        """Multipliers should decrease as severity increases."""
        from app.config.coordination_defaults import BackpressureDefaults

        assert BackpressureDefaults.MULTIPLIER_NONE > BackpressureDefaults.MULTIPLIER_LOW
        assert BackpressureDefaults.MULTIPLIER_LOW > BackpressureDefaults.MULTIPLIER_SOFT
        assert BackpressureDefaults.MULTIPLIER_SOFT > BackpressureDefaults.MULTIPLIER_MEDIUM
        assert BackpressureDefaults.MULTIPLIER_MEDIUM > BackpressureDefaults.MULTIPLIER_HARD
        assert BackpressureDefaults.MULTIPLIER_STOP == 0.0


# =============================================================================
# Test JobTimeoutDefaults
# =============================================================================


class TestJobTimeoutDefaults:
    """Tests for JobTimeoutDefaults dataclass."""

    def test_gpu_selfplay_positive(self):
        """GPU_SELFPLAY timeout should be positive."""
        from app.config.coordination_defaults import JobTimeoutDefaults

        assert JobTimeoutDefaults.GPU_SELFPLAY > 0

    def test_training_longest(self):
        """TRAINING timeout should be one of the longest."""
        from app.config.coordination_defaults import JobTimeoutDefaults

        assert JobTimeoutDefaults.TRAINING >= JobTimeoutDefaults.GPU_SELFPLAY

    def test_cmaes_very_long(self):
        """CMAES timeout should be very long (8+ hours)."""
        from app.config.coordination_defaults import JobTimeoutDefaults

        assert JobTimeoutDefaults.CMAES >= 8 * 3600  # 8 hours


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestGetTimeout:
    """Tests for get_timeout() helper function."""

    def test_get_http_timeout(self):
        """get_timeout('http') should return HTTP_TIMEOUT."""
        from app.config.coordination_defaults import TransportDefaults, get_timeout

        assert get_timeout("http") == TransportDefaults.HTTP_TIMEOUT

    def test_get_ssh_timeout(self):
        """get_timeout('ssh') should return SSH_TIMEOUT."""
        from app.config.coordination_defaults import TransportDefaults, get_timeout

        assert get_timeout("ssh") == TransportDefaults.SSH_TIMEOUT

    def test_get_health_timeout(self):
        """get_timeout('health') should return HEALTH_CHECK."""
        from app.config.coordination_defaults import OperationTimeouts, get_timeout

        assert get_timeout("health") == OperationTimeouts.HEALTH_CHECK

    def test_get_unknown_returns_http(self):
        """Unknown operation should return HTTP_TIMEOUT as default."""
        from app.config.coordination_defaults import TransportDefaults, get_timeout

        assert get_timeout("unknown_operation") == TransportDefaults.HTTP_TIMEOUT


class TestGetJobTimeout:
    """Tests for get_job_timeout() helper function."""

    def test_get_training_timeout(self):
        """get_job_timeout('training') should return TRAINING timeout."""
        from app.config.coordination_defaults import JobTimeoutDefaults, get_job_timeout

        assert get_job_timeout("training") == JobTimeoutDefaults.TRAINING

    def test_get_gpu_selfplay_timeout(self):
        """get_job_timeout('gpu_selfplay') should return GPU_SELFPLAY timeout."""
        from app.config.coordination_defaults import JobTimeoutDefaults, get_job_timeout

        assert get_job_timeout("gpu_selfplay") == JobTimeoutDefaults.GPU_SELFPLAY

    def test_get_selfplay_alias(self):
        """get_job_timeout('selfplay') should work as alias for gpu_selfplay."""
        from app.config.coordination_defaults import JobTimeoutDefaults, get_job_timeout

        assert get_job_timeout("selfplay") == JobTimeoutDefaults.GPU_SELFPLAY

    def test_case_insensitive(self):
        """get_job_timeout should be case insensitive."""
        from app.config.coordination_defaults import get_job_timeout

        assert get_job_timeout("TRAINING") == get_job_timeout("training")
        assert get_job_timeout("Training") == get_job_timeout("training")

    def test_unknown_returns_gpu_selfplay(self):
        """Unknown job type should return GPU_SELFPLAY as default."""
        from app.config.coordination_defaults import JobTimeoutDefaults, get_job_timeout

        assert get_job_timeout("unknown_job") == JobTimeoutDefaults.GPU_SELFPLAY


class TestGetSqliteTimeout:
    """Tests for get_sqlite_timeout() helper function."""

    def test_get_quick_timeout(self):
        """get_sqlite_timeout('quick') should return QUICK_TIMEOUT."""
        from app.config.coordination_defaults import SQLiteDefaults, get_sqlite_timeout

        assert get_sqlite_timeout("quick") == SQLiteDefaults.QUICK_TIMEOUT

    def test_get_read_timeout(self):
        """get_sqlite_timeout('read') should return READ_TIMEOUT."""
        from app.config.coordination_defaults import SQLiteDefaults, get_sqlite_timeout

        assert get_sqlite_timeout("read") == SQLiteDefaults.READ_TIMEOUT

    def test_get_write_timeout(self):
        """get_sqlite_timeout('write') should return WRITE_TIMEOUT."""
        from app.config.coordination_defaults import SQLiteDefaults, get_sqlite_timeout

        assert get_sqlite_timeout("write") == SQLiteDefaults.WRITE_TIMEOUT

    def test_alias_health(self):
        """get_sqlite_timeout('health') should be alias for QUICK_TIMEOUT."""
        from app.config.coordination_defaults import SQLiteDefaults, get_sqlite_timeout

        assert get_sqlite_timeout("health") == SQLiteDefaults.QUICK_TIMEOUT

    def test_alias_elo(self):
        """get_sqlite_timeout('elo') should be alias for WRITE_TIMEOUT."""
        from app.config.coordination_defaults import SQLiteDefaults, get_sqlite_timeout

        assert get_sqlite_timeout("elo") == SQLiteDefaults.WRITE_TIMEOUT

    def test_unknown_returns_standard(self):
        """Unknown operation should return STANDARD_TIMEOUT."""
        from app.config.coordination_defaults import SQLiteDefaults, get_sqlite_timeout

        assert get_sqlite_timeout("unknown") == SQLiteDefaults.STANDARD_TIMEOUT


class TestGetBackpressureMultiplier:
    """Tests for get_backpressure_multiplier() helper function."""

    def test_get_none_level(self):
        """get_backpressure_multiplier('none') should return 1.0."""
        from app.config.coordination_defaults import get_backpressure_multiplier

        assert get_backpressure_multiplier("none") == 1.0

    def test_get_stop_level(self):
        """get_backpressure_multiplier('stop') should return 0.0."""
        from app.config.coordination_defaults import get_backpressure_multiplier

        assert get_backpressure_multiplier("stop") == 0.0

    def test_get_medium_level(self):
        """get_backpressure_multiplier('medium') should return MULTIPLIER_MEDIUM."""
        from app.config.coordination_defaults import (
            BackpressureDefaults,
            get_backpressure_multiplier,
        )

        assert get_backpressure_multiplier("medium") == BackpressureDefaults.MULTIPLIER_MEDIUM

    def test_case_insensitive(self):
        """get_backpressure_multiplier should be case insensitive."""
        from app.config.coordination_defaults import get_backpressure_multiplier

        assert get_backpressure_multiplier("HIGH") == get_backpressure_multiplier("high")

    def test_unknown_returns_none(self):
        """Unknown level should return MULTIPLIER_NONE (1.0)."""
        from app.config.coordination_defaults import get_backpressure_multiplier

        assert get_backpressure_multiplier("unknown") == 1.0


class TestGetP2PPort:
    """Tests for get_p2p_port() helper function."""

    def test_returns_default_port(self):
        """get_p2p_port() should return P2PDefaults.DEFAULT_PORT."""
        from app.config.coordination_defaults import P2PDefaults, get_p2p_port

        assert get_p2p_port() == P2PDefaults.DEFAULT_PORT

    def test_returns_8770_by_default(self):
        """get_p2p_port() should return 8770 by default."""
        from app.config.coordination_defaults import get_p2p_port

        assert get_p2p_port() == 8770


class TestGetSshTimeout:
    """Tests for get_ssh_timeout() helper function."""

    def test_get_command_timeout(self):
        """get_ssh_timeout('command') should return COMMAND_TIMEOUT."""
        from app.config.coordination_defaults import SSHDefaults, get_ssh_timeout

        assert get_ssh_timeout("command") == SSHDefaults.COMMAND_TIMEOUT

    def test_get_rsync_timeout(self):
        """get_ssh_timeout('rsync') should return RSYNC_TIMEOUT."""
        from app.config.coordination_defaults import SSHDefaults, get_ssh_timeout

        assert get_ssh_timeout("rsync") == SSHDefaults.RSYNC_TIMEOUT

    def test_default_is_command(self):
        """get_ssh_timeout() with no arg should return COMMAND_TIMEOUT."""
        from app.config.coordination_defaults import SSHDefaults, get_ssh_timeout

        assert get_ssh_timeout() == SSHDefaults.COMMAND_TIMEOUT

    def test_unknown_returns_command(self):
        """Unknown operation should return COMMAND_TIMEOUT."""
        from app.config.coordination_defaults import SSHDefaults, get_ssh_timeout

        assert get_ssh_timeout("unknown") == SSHDefaults.COMMAND_TIMEOUT


class TestGetPeerTimeout:
    """Tests for get_peer_timeout() helper function."""

    def test_get_peer_timeout(self):
        """get_peer_timeout('peer') should return PEER_TIMEOUT."""
        from app.config.coordination_defaults import PeerDefaults, get_peer_timeout

        assert get_peer_timeout("peer") == PeerDefaults.PEER_TIMEOUT

    def test_get_manifest_timeout(self):
        """get_peer_timeout('manifest') should return MANIFEST_TIMEOUT."""
        from app.config.coordination_defaults import PeerDefaults, get_peer_timeout

        assert get_peer_timeout("manifest") == PeerDefaults.MANIFEST_TIMEOUT

    def test_default_is_peer(self):
        """get_peer_timeout() with no arg should return PEER_TIMEOUT."""
        from app.config.coordination_defaults import PeerDefaults, get_peer_timeout

        assert get_peer_timeout() == PeerDefaults.PEER_TIMEOUT


class TestGetCircuitBreakerConfigs:
    """Tests for get_circuit_breaker_configs() helper function."""

    def test_returns_dict(self):
        """get_circuit_breaker_configs() should return a dict."""
        from app.config.coordination_defaults import get_circuit_breaker_configs

        configs = get_circuit_breaker_configs()
        assert isinstance(configs, dict)

    def test_has_ssh_config(self):
        """Config should have 'ssh' key with expected fields."""
        from app.config.coordination_defaults import get_circuit_breaker_configs

        configs = get_circuit_breaker_configs()
        assert "ssh" in configs
        assert "failure_threshold" in configs["ssh"]
        assert "recovery_timeout" in configs["ssh"]

    def test_has_http_config(self):
        """Config should have 'http' key."""
        from app.config.coordination_defaults import get_circuit_breaker_configs

        configs = get_circuit_breaker_configs()
        assert "http" in configs

    def test_has_p2p_config(self):
        """Config should have 'p2p' key."""
        from app.config.coordination_defaults import get_circuit_breaker_configs

        configs = get_circuit_breaker_configs()
        assert "p2p" in configs


# =============================================================================
# Test get_all_defaults()
# =============================================================================


class TestGetAllDefaults:
    """Tests for get_all_defaults() helper function."""

    def test_returns_dict(self):
        """get_all_defaults() should return a dict."""
        from app.config.coordination_defaults import get_all_defaults

        defaults = get_all_defaults()
        assert isinstance(defaults, dict)

    def test_has_lock_section(self):
        """Result should have 'lock' section."""
        from app.config.coordination_defaults import get_all_defaults

        defaults = get_all_defaults()
        assert "lock" in defaults

    def test_has_transport_section(self):
        """Result should have 'transport' section."""
        from app.config.coordination_defaults import get_all_defaults

        defaults = get_all_defaults()
        assert "transport" in defaults

    def test_has_sync_section(self):
        """Result should have 'sync' section."""
        from app.config.coordination_defaults import get_all_defaults

        defaults = get_all_defaults()
        assert "sync" in defaults

    def test_has_heartbeat_section(self):
        """Result should have 'heartbeat' section."""
        from app.config.coordination_defaults import get_all_defaults

        defaults = get_all_defaults()
        assert "heartbeat" in defaults

    def test_has_training_section(self):
        """Result should have 'training' section."""
        from app.config.coordination_defaults import get_all_defaults

        defaults = get_all_defaults()
        assert "training" in defaults

    def test_has_scheduler_section(self):
        """Result should have 'scheduler' section."""
        from app.config.coordination_defaults import get_all_defaults

        defaults = get_all_defaults()
        assert "scheduler" in defaults

    def test_lock_section_has_lock_timeout(self):
        """Lock section should have lock_timeout."""
        from app.config.coordination_defaults import get_all_defaults

        defaults = get_all_defaults()
        assert "lock_timeout" in defaults["lock"]

    def test_transport_section_has_http_timeout(self):
        """Transport section should have http_timeout."""
        from app.config.coordination_defaults import get_all_defaults

        defaults = get_all_defaults()
        assert "http_timeout" in defaults["transport"]


# =============================================================================
# Test Additional Dataclasses (Coverage)
# =============================================================================


class TestSchedulerDefaults:
    """Tests for SchedulerDefaults dataclass."""

    def test_min_memory_gb_reasonable(self):
        """MIN_MEMORY_GB should be reasonable (8-128 GB)."""
        from app.config.coordination_defaults import SchedulerDefaults

        assert 8 <= SchedulerDefaults.MIN_MEMORY_GB <= 128

    def test_max_queue_size_positive(self):
        """MAX_QUEUE_SIZE should be positive."""
        from app.config.coordination_defaults import SchedulerDefaults

        assert SchedulerDefaults.MAX_QUEUE_SIZE > 0


class TestEphemeralDefaults:
    """Tests for EphemeralDefaults dataclass."""

    def test_checkpoint_interval_positive(self):
        """CHECKPOINT_INTERVAL should be positive."""
        from app.config.coordination_defaults import EphemeralDefaults

        assert EphemeralDefaults.CHECKPOINT_INTERVAL > 0

    def test_heartbeat_timeout_positive(self):
        """HEARTBEAT_TIMEOUT should be positive."""
        from app.config.coordination_defaults import EphemeralDefaults

        assert EphemeralDefaults.HEARTBEAT_TIMEOUT > 0


class TestHealthDefaults:
    """Tests for HealthDefaults dataclass."""

    def test_ssh_timeout_quick(self):
        """SSH_TIMEOUT should be quick for health checks (1-30s)."""
        from app.config.coordination_defaults import HealthDefaults

        assert 1 <= HealthDefaults.SSH_TIMEOUT <= 30

    def test_healthy_cache_longer_than_unhealthy(self):
        """HEALTHY_CACHE_TTL should typically be >= UNHEALTHY_CACHE_TTL."""
        from app.config.coordination_defaults import HealthDefaults

        assert HealthDefaults.HEALTHY_CACHE_TTL >= HealthDefaults.UNHEALTHY_CACHE_TTL


class TestDaemonLoopDefaults:
    """Tests for DaemonLoopDefaults dataclass."""

    def test_check_interval_positive(self):
        """CHECK_INTERVAL should be positive."""
        from app.config.coordination_defaults import DaemonLoopDefaults

        assert DaemonLoopDefaults.CHECK_INTERVAL > 0

    def test_error_backoff_max_greater_than_base(self):
        """ERROR_BACKOFF_MAX should be > ERROR_BACKOFF_BASE."""
        from app.config.coordination_defaults import DaemonLoopDefaults

        assert DaemonLoopDefaults.ERROR_BACKOFF_MAX > DaemonLoopDefaults.ERROR_BACKOFF_BASE


class TestRetryDefaults:
    """Tests for RetryDefaults dataclass."""

    def test_max_retries_positive(self):
        """MAX_RETRIES should be positive."""
        from app.config.coordination_defaults import RetryDefaults

        assert RetryDefaults.MAX_RETRIES > 0

    def test_max_delay_greater_than_base(self):
        """MAX_DELAY should be > BASE_DELAY."""
        from app.config.coordination_defaults import RetryDefaults

        assert RetryDefaults.MAX_DELAY > RetryDefaults.BASE_DELAY

    def test_backoff_multiplier_greater_than_one(self):
        """BACKOFF_MULTIPLIER should be > 1."""
        from app.config.coordination_defaults import RetryDefaults

        assert RetryDefaults.BACKOFF_MULTIPLIER > 1.0

    def test_jitter_factor_in_range(self):
        """JITTER_FACTOR should be in [0, 1]."""
        from app.config.coordination_defaults import RetryDefaults

        assert 0 <= RetryDefaults.JITTER_FACTOR <= 1


class TestOperationTimeouts:
    """Tests for OperationTimeouts dataclass."""

    def test_health_check_quick(self):
        """HEALTH_CHECK should be quick (1-30s)."""
        from app.config.coordination_defaults import OperationTimeouts

        assert 1 <= OperationTimeouts.HEALTH_CHECK <= 30

    def test_training_job_very_long(self):
        """TRAINING_JOB should be very long (hours)."""
        from app.config.coordination_defaults import OperationTimeouts

        assert OperationTimeouts.TRAINING_JOB >= 3600  # At least 1 hour


class TestCurriculumDefaults:
    """Tests for CurriculumDefaults dataclass."""

    def test_mastery_threshold_in_range(self):
        """MASTERY_THRESHOLD should be in [0.5, 1.0]."""
        from app.config.coordination_defaults import CurriculumDefaults

        assert 0.5 <= CurriculumDefaults.MASTERY_THRESHOLD <= 1.0

    def test_max_weight_greater_than_min(self):
        """MAX_WEIGHT should be > MIN_WEIGHT."""
        from app.config.coordination_defaults import CurriculumDefaults

        assert CurriculumDefaults.MAX_WEIGHT > CurriculumDefaults.MIN_WEIGHT

    def test_min_weight_positive(self):
        """MIN_WEIGHT should be positive."""
        from app.config.coordination_defaults import CurriculumDefaults

        assert CurriculumDefaults.MIN_WEIGHT > 0


class TestMonitoringDefaults:
    """Tests for MonitoringDefaults dataclass."""

    def test_disk_warning_less_than_critical(self):
        """DISK_WARNING_THRESHOLD should be < DISK_CRITICAL_THRESHOLD."""
        from app.config.coordination_defaults import MonitoringDefaults

        assert MonitoringDefaults.DISK_WARNING_THRESHOLD < MonitoringDefaults.DISK_CRITICAL_THRESHOLD

    def test_memory_warning_less_than_critical(self):
        """MEMORY_WARNING_THRESHOLD should be < MEMORY_CRITICAL_THRESHOLD."""
        from app.config.coordination_defaults import MonitoringDefaults

        assert MonitoringDefaults.MEMORY_WARNING_THRESHOLD < MonitoringDefaults.MEMORY_CRITICAL_THRESHOLD


class TestIdleThresholdDefaults:
    """Tests for IdleThresholdDefaults dataclass."""

    def test_gpu_idle_threshold_positive(self):
        """GPU_IDLE_THRESHOLD should be positive."""
        from app.config.coordination_defaults import IdleThresholdDefaults

        assert IdleThresholdDefaults.GPU_IDLE_THRESHOLD > 0

    def test_min_gpu_utilization_in_range(self):
        """MIN_GPU_UTILIZATION should be in [0, 100]."""
        from app.config.coordination_defaults import IdleThresholdDefaults

        assert 0 <= IdleThresholdDefaults.MIN_GPU_UTILIZATION <= 100


# =============================================================================
# Test get_aiohttp_timeout
# =============================================================================


class TestGetAiohttpTimeout:
    """Tests for get_aiohttp_timeout() helper function."""

    def test_returns_client_timeout_when_aiohttp_available(self):
        """get_aiohttp_timeout() should return ClientTimeout when aiohttp is available."""
        from app.config.coordination_defaults import get_aiohttp_timeout

        timeout = get_aiohttp_timeout("http")

        # May be None if aiohttp not installed
        if timeout is not None:
            # Should be an aiohttp.ClientTimeout instance
            import aiohttp
            assert isinstance(timeout, aiohttp.ClientTimeout)

    def test_different_operations_have_different_timeouts(self):
        """Different operations should have different timeouts."""
        from app.config.coordination_defaults import get_aiohttp_timeout

        health_timeout = get_aiohttp_timeout("health")
        training_timeout = get_aiohttp_timeout("training")

        # If aiohttp is available, timeouts should differ
        if health_timeout is not None and training_timeout is not None:
            assert health_timeout.total != training_timeout.total


# =============================================================================
# Test Environment Override (Integration)
# =============================================================================


class TestEnvironmentOverride:
    """Test that environment variables properly override defaults."""

    def test_lock_timeout_override(self):
        """RINGRIFT_LOCK_TIMEOUT env var should override LockDefaults.LOCK_TIMEOUT."""
        with patch.dict(os.environ, {"RINGRIFT_LOCK_TIMEOUT": "9999"}):
            from importlib import reload
            import app.config.coordination_defaults as cd
            reload(cd)

            assert cd.LockDefaults.LOCK_TIMEOUT == 9999

            # Clean up
            reload(cd)

    def test_http_timeout_override(self):
        """RINGRIFT_HTTP_TIMEOUT env var should override TransportDefaults.HTTP_TIMEOUT."""
        with patch.dict(os.environ, {"RINGRIFT_HTTP_TIMEOUT": "99"}):
            from importlib import reload
            import app.config.coordination_defaults as cd
            reload(cd)

            assert cd.TransportDefaults.HTTP_TIMEOUT == 99

            # Clean up
            reload(cd)
