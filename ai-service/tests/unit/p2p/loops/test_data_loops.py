"""Tests for P2P data management loops.

Tests cover:
- ModelSyncConfig/Loop: Model synchronization across cluster
- DataAggregationConfig/Loop: Training data aggregation
- DataManagementConfig/Loop: Comprehensive data management
- ModelFetchConfig/Loop: Model fetching from training nodes
"""

import asyncio
import pytest
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from scripts.p2p.loops.data_loops import (
    ModelSyncConfig,
    ModelSyncLoop,
    DataAggregationConfig,
    DataAggregationLoop,
    DataManagementConfig,
    DataManagementLoop,
    ModelFetchConfig,
    ModelFetchLoop,
)


# =============================================================================
# ModelSyncConfig Tests
# =============================================================================


class TestModelSyncConfig:
    """Tests for ModelSyncConfig dataclass."""

    def test_default_values(self):
        """Test ModelSyncConfig has sensible defaults."""
        config = ModelSyncConfig()

        assert config.check_interval_seconds == 120.0
        assert config.max_sync_operations_per_cycle == 5
        assert config.sync_timeout_seconds == 300.0
        assert config.priority_configs == ["hex8_2p", "square8_2p"]

    def test_validation_check_interval_zero(self):
        """Test validation rejects check_interval_seconds <= 0."""
        with pytest.raises(ValueError, match="check_interval_seconds"):
            ModelSyncConfig(check_interval_seconds=0)

    def test_validation_max_sync_zero(self):
        """Test validation rejects max_sync_operations_per_cycle <= 0."""
        with pytest.raises(ValueError, match="max_sync_operations_per_cycle"):
            ModelSyncConfig(max_sync_operations_per_cycle=0)

    def test_validation_sync_timeout_zero(self):
        """Test validation rejects sync_timeout_seconds <= 0."""
        with pytest.raises(ValueError, match="sync_timeout_seconds"):
            ModelSyncConfig(sync_timeout_seconds=0)

    def test_custom_priority_configs(self):
        """Test custom priority configs."""
        config = ModelSyncConfig(priority_configs=["hexagonal_4p"])

        assert config.priority_configs == ["hexagonal_4p"]


# =============================================================================
# ModelSyncLoop Tests
# =============================================================================


class TestModelSyncLoop:
    """Tests for ModelSyncLoop class."""

    def _create_loop(self, **overrides):
        """Create a ModelSyncLoop with defaults."""
        defaults = {
            "get_model_versions": MagicMock(return_value={}),
            "get_node_models": AsyncMock(return_value={}),
            "sync_model": AsyncMock(return_value=True),
            "get_active_nodes": MagicMock(return_value=[]),
            "config": None,
        }
        defaults.update(overrides)
        return ModelSyncLoop(**defaults)

    def test_init(self):
        """Test ModelSyncLoop initialization."""
        loop = self._create_loop()

        assert loop.name == "model_sync"
        assert loop._sync_stats["models_synced"] == 0
        assert loop._sync_stats["sync_failures"] == 0

    @pytest.mark.asyncio
    async def test_run_once_no_versions(self):
        """Test _run_once with no model versions."""
        get_versions = MagicMock(return_value={})
        sync_model = AsyncMock(return_value=True)
        loop = self._create_loop(
            get_model_versions=get_versions,
            sync_model=sync_model,
        )

        await loop._run_once()

        sync_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_once_no_active_nodes(self):
        """Test _run_once with no active nodes."""
        get_versions = MagicMock(return_value={"hex8_2p": {"version": "v1"}})
        get_nodes = MagicMock(return_value=[])
        sync_model = AsyncMock(return_value=True)
        loop = self._create_loop(
            get_model_versions=get_versions,
            get_active_nodes=get_nodes,
            sync_model=sync_model,
        )

        await loop._run_once()

        sync_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_once_syncs_outdated_model(self):
        """Test _run_once syncs outdated models to nodes."""
        get_versions = MagicMock(return_value={
            "hex8_2p": {"version": "v2", "path": "/models/hex8_2p.pth"},
        })
        get_node_models = AsyncMock(return_value={"hex8_2p": "v1"})  # Outdated
        sync_model = AsyncMock(return_value=True)
        get_nodes = MagicMock(return_value=["node-1"])
        loop = self._create_loop(
            get_model_versions=get_versions,
            get_node_models=get_node_models,
            sync_model=sync_model,
            get_active_nodes=get_nodes,
        )

        await loop._run_once()

        sync_model.assert_called_once_with("node-1", "hex8_2p", "/models/hex8_2p.pth")
        assert loop._sync_stats["models_synced"] == 1

    @pytest.mark.asyncio
    async def test_run_once_prioritizes_configs(self):
        """Test _run_once prioritizes certain configs."""
        get_versions = MagicMock(return_value={
            "hexagonal_4p": {"version": "v1", "path": "/models/hex_4p.pth"},
            "hex8_2p": {"version": "v1", "path": "/models/hex8_2p.pth"},
        })
        get_node_models = AsyncMock(return_value={})  # All outdated
        sync_model = AsyncMock(return_value=True)
        get_nodes = MagicMock(return_value=["node-1"])
        config = ModelSyncConfig(
            priority_configs=["hex8_2p"],
            max_sync_operations_per_cycle=1,
        )
        loop = self._create_loop(
            get_model_versions=get_versions,
            get_node_models=get_node_models,
            sync_model=sync_model,
            get_active_nodes=get_nodes,
            config=config,
        )

        await loop._run_once()

        # Should sync priority config first
        sync_model.assert_called_once()
        call_args = sync_model.call_args[0]
        assert call_args[1] == "hex8_2p"

    @pytest.mark.asyncio
    async def test_run_once_handles_sync_failure(self):
        """Test _run_once handles sync failures."""
        get_versions = MagicMock(return_value={
            "hex8_2p": {"version": "v1", "path": "/models/hex8_2p.pth"},
        })
        get_node_models = AsyncMock(return_value={})
        sync_model = AsyncMock(return_value=False)
        get_nodes = MagicMock(return_value=["node-1"])
        loop = self._create_loop(
            get_model_versions=get_versions,
            get_node_models=get_node_models,
            sync_model=sync_model,
            get_active_nodes=get_nodes,
        )

        await loop._run_once()

        assert loop._sync_stats["sync_failures"] == 1

    @pytest.mark.asyncio
    async def test_run_once_handles_sync_timeout(self):
        """Test _run_once handles sync timeout."""
        async def slow_sync(*args):
            await asyncio.sleep(10)
            return True

        get_versions = MagicMock(return_value={
            "hex8_2p": {"version": "v1", "path": "/models/hex8_2p.pth"},
        })
        get_node_models = AsyncMock(return_value={})
        get_nodes = MagicMock(return_value=["node-1"])
        config = ModelSyncConfig(sync_timeout_seconds=0.01)
        loop = self._create_loop(
            get_model_versions=get_versions,
            get_node_models=get_node_models,
            sync_model=slow_sync,
            get_active_nodes=get_nodes,
            config=config,
        )

        await loop._run_once()

        assert loop._sync_stats["sync_failures"] == 1

    def test_get_sync_stats(self):
        """Test get_sync_stats returns correct stats."""
        loop = self._create_loop()
        loop._sync_stats["models_synced"] = 10
        loop._sync_stats["sync_failures"] = 2

        stats = loop.get_sync_stats()

        assert stats["models_synced"] == 10
        assert stats["sync_failures"] == 2
        assert "total_runs" in stats

    def test_health_check_healthy(self):
        """Test health_check when healthy."""
        loop = self._create_loop()
        loop._running = True
        loop._sync_stats["models_synced"] = 10
        loop._sync_stats["sync_failures"] = 1

        health = loop.health_check()
        assert health["status"] == "HEALTHY"

    def test_health_check_degraded(self):
        """Test health_check when degraded."""
        loop = self._create_loop()
        loop._running = True
        loop._sync_stats["models_synced"] = 5
        loop._sync_stats["sync_failures"] = 15  # High failure rate

        health = loop.health_check()
        assert health["status"] == "DEGRADED"


# =============================================================================
# DataAggregationConfig Tests
# =============================================================================


class TestDataAggregationConfig:
    """Tests for DataAggregationConfig dataclass."""

    def test_default_values(self):
        """Test DataAggregationConfig has sensible defaults."""
        config = DataAggregationConfig()

        assert config.check_interval_seconds == 300.0
        assert config.min_games_to_aggregate == 100
        assert config.max_nodes_per_cycle == 10
        assert config.aggregation_timeout_seconds == 600.0

    def test_validation_check_interval_zero(self):
        """Test validation rejects check_interval_seconds <= 0."""
        with pytest.raises(ValueError, match="check_interval_seconds"):
            DataAggregationConfig(check_interval_seconds=0)

    def test_validation_min_games_zero(self):
        """Test validation rejects min_games_to_aggregate <= 0."""
        with pytest.raises(ValueError, match="min_games_to_aggregate"):
            DataAggregationConfig(min_games_to_aggregate=0)

    def test_validation_max_nodes_zero(self):
        """Test validation rejects max_nodes_per_cycle <= 0."""
        with pytest.raises(ValueError, match="max_nodes_per_cycle"):
            DataAggregationConfig(max_nodes_per_cycle=0)


# =============================================================================
# DataAggregationLoop Tests
# =============================================================================


class TestDataAggregationLoop:
    """Tests for DataAggregationLoop class."""

    def _create_loop(self, **overrides):
        """Create a DataAggregationLoop with defaults."""
        defaults = {
            "get_node_game_counts": MagicMock(return_value={}),
            "aggregate_from_node": AsyncMock(return_value={"games_aggregated": 0}),
            "config": None,
        }
        defaults.update(overrides)
        return DataAggregationLoop(**defaults)

    def test_init(self):
        """Test DataAggregationLoop initialization."""
        loop = self._create_loop()

        assert loop.name == "data_aggregation"
        assert loop._aggregation_stats["total_games_aggregated"] == 0

    @pytest.mark.asyncio
    async def test_run_once_no_node_counts(self):
        """Test _run_once with no node game counts."""
        get_counts = MagicMock(return_value={})
        aggregate = AsyncMock()
        loop = self._create_loop(
            get_node_game_counts=get_counts,
            aggregate_from_node=aggregate,
        )

        await loop._run_once()

        aggregate.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_once_aggregates_from_high_count_nodes(self):
        """Test _run_once aggregates from nodes with enough games."""
        get_counts = MagicMock(return_value={
            "node-1": 500,  # High count
            "node-2": 50,   # Low count
        })
        aggregate = AsyncMock(return_value={
            "games_aggregated": 500,
            "bytes_transferred": 10_000_000,
        })
        config = DataAggregationConfig(min_games_to_aggregate=100)
        loop = self._create_loop(
            get_node_game_counts=get_counts,
            aggregate_from_node=aggregate,
            config=config,
        )

        await loop._run_once()

        # Should only aggregate from node-1
        aggregate.assert_called_once_with("node-1")
        assert loop._aggregation_stats["total_games_aggregated"] == 500

    @pytest.mark.asyncio
    async def test_run_once_respects_max_nodes(self):
        """Test _run_once respects max_nodes_per_cycle."""
        get_counts = MagicMock(return_value={
            f"node-{i}": 200 for i in range(5)
        })
        aggregate = AsyncMock(return_value={"games_aggregated": 100})
        config = DataAggregationConfig(max_nodes_per_cycle=2)
        loop = self._create_loop(
            get_node_game_counts=get_counts,
            aggregate_from_node=aggregate,
            config=config,
        )

        await loop._run_once()

        assert aggregate.call_count == 2

    @pytest.mark.asyncio
    async def test_run_once_handles_timeout(self):
        """Test _run_once handles aggregation timeout."""
        async def slow_aggregate(*args):
            await asyncio.sleep(10)
            return {"games_aggregated": 100}

        get_counts = MagicMock(return_value={"node-1": 500})
        config = DataAggregationConfig(aggregation_timeout_seconds=0.01)
        loop = self._create_loop(
            get_node_game_counts=get_counts,
            aggregate_from_node=slow_aggregate,
            config=config,
        )

        await loop._run_once()

        assert loop._aggregation_stats["aggregation_failures"] == 1

    def test_get_aggregation_stats(self):
        """Test get_aggregation_stats returns correct stats."""
        loop = self._create_loop()
        loop._aggregation_stats["total_games_aggregated"] = 1000

        stats = loop.get_aggregation_stats()

        assert stats["total_games_aggregated"] == 1000

    def test_health_check_healthy(self):
        """Test health_check when healthy."""
        loop = self._create_loop()
        loop._running = True
        loop._aggregation_stats["total_games_aggregated"] = 1000

        health = loop.health_check()
        assert health["status"] == "HEALTHY"


# =============================================================================
# DataManagementConfig Tests
# =============================================================================


class TestDataManagementConfig:
    """Tests for DataManagementConfig dataclass."""

    def test_default_values(self):
        """Test DataManagementConfig has sensible defaults."""
        config = DataManagementConfig()

        assert config.interval_seconds == 300.0
        assert config.disk_warning_percent == 70.0
        assert config.disk_critical_percent == 85.0
        assert config.db_export_threshold_mb == 100.0
        assert config.max_concurrent_exports == 3

    def test_validation_interval_zero(self):
        """Test validation rejects interval_seconds <= 0."""
        with pytest.raises(ValueError, match="interval_seconds"):
            DataManagementConfig(interval_seconds=0)

    def test_validation_warning_ge_critical(self):
        """Test validation rejects warning >= critical."""
        with pytest.raises(ValueError, match="disk_warning_percent"):
            DataManagementConfig(
                disk_warning_percent=90.0,
                disk_critical_percent=80.0,
            )

    def test_validation_max_exports_zero(self):
        """Test validation rejects max_concurrent_exports <= 0."""
        with pytest.raises(ValueError, match="max_concurrent_exports"):
            DataManagementConfig(max_concurrent_exports=0)


# =============================================================================
# DataManagementLoop Tests
# =============================================================================


class TestDataManagementLoop:
    """Tests for DataManagementLoop class."""

    def _create_loop(self, **overrides):
        """Create a DataManagementLoop with defaults."""
        defaults = {
            "is_leader": MagicMock(return_value=False),
            "check_disk_capacity": MagicMock(return_value=(True, 50.0)),
            "cleanup_disk": AsyncMock(),
            "convert_jsonl_to_db": AsyncMock(return_value=0),
            "convert_jsonl_to_npz": AsyncMock(return_value=0),
            "config": DataManagementConfig(initial_delay_seconds=0),
        }
        defaults.update(overrides)
        return DataManagementLoop(**defaults)

    def test_init(self):
        """Test DataManagementLoop initialization."""
        loop = self._create_loop()

        assert loop.name == "data_management"
        assert loop._data_stats["disk_cleanups"] == 0

    @pytest.mark.asyncio
    async def test_run_once_checks_disk(self):
        """Test _run_once checks disk capacity."""
        check_disk = MagicMock(return_value=(True, 50.0))
        loop = self._create_loop(check_disk_capacity=check_disk)

        await loop._run_once()

        check_disk.assert_called()

    @pytest.mark.asyncio
    async def test_run_once_triggers_cleanup(self):
        """Test _run_once triggers cleanup when disk full."""
        check_disk = MagicMock(return_value=(False, 80.0))
        cleanup_disk = AsyncMock()
        loop = self._create_loop(
            check_disk_capacity=check_disk,
            cleanup_disk=cleanup_disk,
        )

        await loop._run_once()

        cleanup_disk.assert_called_once()
        assert loop._data_stats["disk_cleanups"] == 1

    @pytest.mark.asyncio
    async def test_run_once_converts_jsonl(self):
        """Test _run_once converts JSONL files."""
        convert_db = AsyncMock(return_value=5)
        convert_npz = AsyncMock(return_value=2)
        loop = self._create_loop(
            convert_jsonl_to_db=convert_db,
            convert_jsonl_to_npz=convert_npz,
        )

        await loop._run_once()

        convert_db.assert_called_once()
        convert_npz.assert_called_once()
        assert loop._data_stats["jsonl_to_db_conversions"] == 5
        assert loop._data_stats["jsonl_to_npz_conversions"] == 2

    @pytest.mark.asyncio
    async def test_run_once_skips_leader_ops_if_not_leader(self):
        """Test _run_once skips leader operations if not leader."""
        is_leader = MagicMock(return_value=False)
        check_integrity = AsyncMock()
        loop = self._create_loop(
            is_leader=is_leader,
            check_db_integrity=check_integrity,
        )

        await loop._run_once()

        check_integrity.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_once_runs_leader_ops_if_leader(self):
        """Test _run_once runs leader operations if leader."""
        is_leader = MagicMock(return_value=True)
        config = DataManagementConfig(
            initial_delay_seconds=0,
            db_integrity_check_frequency=1,  # Check every cycle
        )
        check_integrity = AsyncMock(return_value={"checked": 5, "corrupted": 0})
        loop = self._create_loop(
            is_leader=is_leader,
            check_db_integrity=check_integrity,
            config=config,
        )

        await loop._run_once()

        check_integrity.assert_called_once()
        assert loop._data_stats["integrity_checks"] == 1

    def test_infer_board_type_hex(self):
        """Test _infer_board_type detects hex boards."""
        assert DataManagementLoop._infer_board_type("hex8_2p.db") == "hexagonal"
        assert DataManagementLoop._infer_board_type("canonical_hexagonal.db") == "hexagonal"

    def test_infer_board_type_square19(self):
        """Test _infer_board_type detects square19 boards."""
        assert DataManagementLoop._infer_board_type("square19_2p.db") == "square19"
        assert DataManagementLoop._infer_board_type("sq19_games.db") == "square19"

    def test_infer_board_type_default(self):
        """Test _infer_board_type defaults to square8."""
        assert DataManagementLoop._infer_board_type("games.db") == "square8"
        assert DataManagementLoop._infer_board_type("unknown.db") == "square8"

    def test_get_status(self):
        """Test get_status returns extended status."""
        loop = self._create_loop()
        loop._data_stats["disk_cleanups"] = 3
        loop._active_exports["test.db"] = time.time()

        status = loop.get_status()

        assert "data_management_stats" in status
        assert status["data_management_stats"]["disk_cleanups"] == 3
        assert status["data_management_stats"]["active_exports"] == 1

    def test_health_check_healthy(self):
        """Test health_check when healthy."""
        loop = self._create_loop()
        loop._running = True

        health = loop.health_check()
        assert health["status"] == "HEALTHY"

    def test_health_check_not_running(self):
        """Test health_check when not running."""
        loop = self._create_loop()
        loop._running = False

        health = loop.health_check()
        assert health["status"] == "ERROR"


# =============================================================================
# ModelFetchConfig Tests
# =============================================================================


class TestModelFetchConfig:
    """Tests for ModelFetchConfig dataclass."""

    def test_default_values(self):
        """Test ModelFetchConfig has sensible defaults."""
        config = ModelFetchConfig()

        assert config.check_interval_seconds == 60.0
        assert config.fetch_timeout_seconds == 180.0
        assert config.max_fetch_retries == 3
        assert config.retry_delay_seconds == 30.0

    def test_validation_check_interval_zero(self):
        """Test validation rejects check_interval_seconds <= 0."""
        with pytest.raises(ValueError, match="check_interval_seconds"):
            ModelFetchConfig(check_interval_seconds=0)

    def test_validation_fetch_timeout_zero(self):
        """Test validation rejects fetch_timeout_seconds <= 0."""
        with pytest.raises(ValueError, match="fetch_timeout_seconds"):
            ModelFetchConfig(fetch_timeout_seconds=0)


# =============================================================================
# ModelFetchLoop Tests
# =============================================================================


class TestModelFetchLoop:
    """Tests for ModelFetchLoop class."""

    def _create_mock_job(self, job_id: str, output_path: str = "/model.pth", worker: str = "node-1"):
        """Create a mock training job."""
        job = MagicMock()
        job.job_id = job_id
        job.output_model_path = output_path
        job.worker_node = worker
        return job

    def _create_loop(self, **overrides):
        """Create a ModelFetchLoop with defaults."""
        defaults = {
            "is_leader": MagicMock(return_value=True),
            "get_completed_training_jobs": MagicMock(return_value=[]),
            "fetch_model": AsyncMock(return_value=True),
            "mark_model_fetched": MagicMock(),
            "is_model_fetched": MagicMock(return_value=False),
            "config": None,
        }
        defaults.update(overrides)
        return ModelFetchLoop(**defaults)

    def test_init(self):
        """Test ModelFetchLoop initialization."""
        loop = self._create_loop()

        assert loop.name == "model_fetch"
        assert loop._fetch_stats["models_fetched"] == 0

    @pytest.mark.asyncio
    async def test_run_once_skips_non_leader(self):
        """Test _run_once skips if not leader."""
        is_leader = MagicMock(return_value=False)
        get_jobs = MagicMock(return_value=[self._create_mock_job("job-1")])
        fetch_model = AsyncMock(return_value=True)
        loop = self._create_loop(
            is_leader=is_leader,
            get_completed_training_jobs=get_jobs,
            fetch_model=fetch_model,
        )

        await loop._run_once()

        fetch_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_once_fetches_unfetched_model(self):
        """Test _run_once fetches unfetched models."""
        mock_job = self._create_mock_job("job-1")
        get_jobs = MagicMock(return_value=[mock_job])
        is_fetched = MagicMock(return_value=False)
        fetch_model = AsyncMock(return_value=True)
        mark_fetched = MagicMock()
        loop = self._create_loop(
            get_completed_training_jobs=get_jobs,
            is_model_fetched=is_fetched,
            fetch_model=fetch_model,
            mark_model_fetched=mark_fetched,
        )

        await loop._run_once()

        fetch_model.assert_called_once_with(mock_job)
        mark_fetched.assert_called_once_with("job-1")
        assert loop._fetch_stats["models_fetched"] == 1

    @pytest.mark.asyncio
    async def test_run_once_skips_already_fetched(self):
        """Test _run_once skips already fetched models."""
        mock_job = self._create_mock_job("job-1")
        get_jobs = MagicMock(return_value=[mock_job])
        is_fetched = MagicMock(return_value=True)
        fetch_model = AsyncMock(return_value=True)
        loop = self._create_loop(
            get_completed_training_jobs=get_jobs,
            is_model_fetched=is_fetched,
            fetch_model=fetch_model,
        )

        await loop._run_once()

        fetch_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_once_respects_max_retries(self):
        """Test _run_once respects max retry limit."""
        mock_job = self._create_mock_job("job-1")
        get_jobs = MagicMock(return_value=[mock_job])
        is_fetched = MagicMock(return_value=False)
        fetch_model = AsyncMock(return_value=True)
        config = ModelFetchConfig(max_fetch_retries=3)
        loop = self._create_loop(
            get_completed_training_jobs=get_jobs,
            is_model_fetched=is_fetched,
            fetch_model=fetch_model,
            config=config,
        )
        loop._job_retries["job-1"] = 3  # Already at max

        await loop._run_once()

        fetch_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_once_handles_fetch_failure(self):
        """Test _run_once handles fetch failure and increments retry."""
        mock_job = self._create_mock_job("job-1")
        get_jobs = MagicMock(return_value=[mock_job])
        is_fetched = MagicMock(return_value=False)
        fetch_model = AsyncMock(return_value=False)
        loop = self._create_loop(
            get_completed_training_jobs=get_jobs,
            is_model_fetched=is_fetched,
            fetch_model=fetch_model,
        )

        await loop._run_once()

        assert loop._job_retries.get("job-1") == 1
        assert loop._fetch_stats["fetch_failures"] == 1

    @pytest.mark.asyncio
    async def test_run_once_handles_timeout(self):
        """Test _run_once handles fetch timeout."""
        async def slow_fetch(*args):
            await asyncio.sleep(10)
            return True

        mock_job = self._create_mock_job("job-1")
        get_jobs = MagicMock(return_value=[mock_job])
        is_fetched = MagicMock(return_value=False)
        config = ModelFetchConfig(fetch_timeout_seconds=0.01)
        loop = self._create_loop(
            get_completed_training_jobs=get_jobs,
            is_model_fetched=is_fetched,
            fetch_model=slow_fetch,
            config=config,
        )

        await loop._run_once()

        assert loop._fetch_stats["fetch_failures"] == 1

    def test_get_status(self):
        """Test get_status returns extended status."""
        loop = self._create_loop()
        loop._fetch_stats["models_fetched"] = 5
        loop._job_retries["job-1"] = 2

        status = loop.get_status()

        assert "model_fetch_stats" in status
        assert status["model_fetch_stats"]["models_fetched"] == 5
        assert status["model_fetch_stats"]["pending_retries"] == 1

    def test_health_check_healthy(self):
        """Test health_check when healthy."""
        loop = self._create_loop()
        loop._running = True
        loop._fetch_stats["models_fetched"] = 10

        health = loop.health_check()
        assert health["status"] == "HEALTHY"

    def test_health_check_degraded_high_failures(self):
        """Test health_check when degraded due to failures."""
        loop = self._create_loop()
        loop._running = True
        loop._fetch_stats["models_fetched"] = 5
        loop._fetch_stats["fetch_failures"] = 15

        health = loop.health_check()
        assert health["status"] == "DEGRADED"

    def test_health_check_degraded_pending_retries(self):
        """Test health_check when degraded due to pending retries."""
        loop = self._create_loop()
        loop._running = True
        for i in range(10):
            loop._job_retries[f"job-{i}"] = 1

        health = loop.health_check()
        assert health["status"] == "DEGRADED"


# =============================================================================
# Loop Lifecycle Tests
# =============================================================================


class TestDataLoopsLifecycle:
    """Tests for data loop lifecycle management."""

    @pytest.mark.asyncio
    async def test_model_sync_start_stop(self):
        """Test ModelSyncLoop can start and stop cleanly."""
        config = ModelSyncConfig(check_interval_seconds=0.1)
        loop = ModelSyncLoop(
            get_model_versions=MagicMock(return_value={}),
            get_node_models=AsyncMock(return_value={}),
            sync_model=AsyncMock(return_value=True),
            get_active_nodes=MagicMock(return_value=[]),
            config=config,
        )

        task = loop.start_background()
        await asyncio.sleep(0.05)
        assert loop.running

        loop.stop()
        await asyncio.sleep(0.1)
        assert not loop.running
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_data_aggregation_start_stop(self):
        """Test DataAggregationLoop can start and stop cleanly."""
        config = DataAggregationConfig(check_interval_seconds=0.1)
        loop = DataAggregationLoop(
            get_node_game_counts=MagicMock(return_value={}),
            aggregate_from_node=AsyncMock(return_value={}),
            config=config,
        )

        task = loop.start_background()
        await asyncio.sleep(0.05)
        assert loop.running

        loop.stop()
        await asyncio.sleep(0.1)
        assert not loop.running
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_data_management_start_stop(self):
        """Test DataManagementLoop can start and stop cleanly."""
        config = DataManagementConfig(
            interval_seconds=0.1,
            initial_delay_seconds=0,
        )
        loop = DataManagementLoop(
            is_leader=MagicMock(return_value=False),
            check_disk_capacity=MagicMock(return_value=(True, 50.0)),
            cleanup_disk=AsyncMock(),
            convert_jsonl_to_db=AsyncMock(return_value=0),
            convert_jsonl_to_npz=AsyncMock(return_value=0),
            config=config,
        )

        task = loop.start_background()
        await asyncio.sleep(0.05)
        assert loop.running

        loop.stop()
        await asyncio.sleep(0.1)
        assert not loop.running
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_model_fetch_start_stop(self):
        """Test ModelFetchLoop can start and stop cleanly."""
        config = ModelFetchConfig(check_interval_seconds=0.1)
        loop = ModelFetchLoop(
            is_leader=MagicMock(return_value=False),
            get_completed_training_jobs=MagicMock(return_value=[]),
            fetch_model=AsyncMock(return_value=True),
            mark_model_fetched=MagicMock(),
            is_model_fetched=MagicMock(return_value=False),
            config=config,
        )

        task = loop.start_background()
        await asyncio.sleep(0.05)
        assert loop.running

        loop.stop()
        await asyncio.sleep(0.1)
        assert not loop.running
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
