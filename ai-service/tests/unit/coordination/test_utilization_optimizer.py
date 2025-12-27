"""Tests for utilization_optimizer module.

Tests GPU capability matching, board selection, engine selection,
underutilized node detection, and workload optimization.
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

from app.coordination.utilization_optimizer import (
    # Enums
    WorkloadType,
    # Data classes
    WorkloadConfig,
    NodeWorkload,
    OptimizationResult,
    # Constants
    BOARD_PRIORITIES,
    GPU_CAPABILITIES,
    # Functions
    _infer_capability_from_vram,
    load_gpu_profiles_from_config,
    get_gpu_capabilities,
    # Main class
    UtilizationOptimizer,
    get_utilization_optimizer,
)
from app.coordination.types import BoardType
from app.providers import Provider


# ============================================================================
# WorkloadType Enum Tests
# ============================================================================


class TestWorkloadType:
    """Tests for WorkloadType enum."""

    def test_workload_types_exist(self) -> None:
        """Test all workload types exist."""
        assert WorkloadType.SELFPLAY == "selfplay"
        assert WorkloadType.TRAINING == "training"
        assert WorkloadType.EVALUATION == "evaluation"
        assert WorkloadType.SYNC == "sync"

    def test_workload_type_is_string_enum(self) -> None:
        """Test WorkloadType is a string enum."""
        assert isinstance(WorkloadType.SELFPLAY, str)
        assert WorkloadType.SELFPLAY.value == "selfplay"


# ============================================================================
# WorkloadConfig Dataclass Tests
# ============================================================================


class TestWorkloadConfig:
    """Tests for WorkloadConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = WorkloadConfig(
            workload_type=WorkloadType.SELFPLAY,
            board_type=BoardType.HEX8,
        )
        assert config.num_players == 2
        assert config.engine == "gumbel"
        assert config.num_games == 1000
        assert config.priority == 50

    def test_custom_values(self) -> None:
        """Test custom values are stored correctly."""
        config = WorkloadConfig(
            workload_type=WorkloadType.TRAINING,
            board_type=BoardType.SQUARE19,
            num_players=4,
            engine="heuristic",
            num_games=500,
            priority=80,
        )
        assert config.workload_type == WorkloadType.TRAINING
        assert config.board_type == BoardType.SQUARE19
        assert config.num_players == 4
        assert config.engine == "heuristic"
        assert config.num_games == 500
        assert config.priority == 80


# ============================================================================
# NodeWorkload Dataclass Tests
# ============================================================================


class TestNodeWorkload:
    """Tests for NodeWorkload dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        workload = NodeWorkload(node_id="test-node")
        assert workload.node_id == "test-node"
        assert workload.selfplay_jobs == 0
        assert workload.training_running is False
        assert workload.current_board_type is None
        assert workload.current_engine is None
        assert workload.games_generated == 0
        assert workload.utilization_score == 0.0

    def test_custom_values(self) -> None:
        """Test custom values are stored correctly."""
        workload = NodeWorkload(
            node_id="gpu-node-1",
            selfplay_jobs=3,
            training_running=True,
            current_board_type=BoardType.HEXAGONAL,
            current_engine="gumbel",
            games_generated=1500,
            utilization_score=85.5,
        )
        assert workload.node_id == "gpu-node-1"
        assert workload.selfplay_jobs == 3
        assert workload.training_running is True
        assert workload.current_board_type == BoardType.HEXAGONAL
        assert workload.current_engine == "gumbel"
        assert workload.games_generated == 1500
        assert workload.utilization_score == 85.5


# ============================================================================
# OptimizationResult Dataclass Tests
# ============================================================================


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_success_result(self) -> None:
        """Test creating a success result."""
        result = OptimizationResult(
            node_id="node-1",
            action="spawn_selfplay",
            success=True,
            message="Job spawned successfully",
        )
        assert result.node_id == "node-1"
        assert result.action == "spawn_selfplay"
        assert result.success is True
        assert result.message == "Job spawned successfully"
        assert result.workload is None
        assert isinstance(result.timestamp, datetime)

    def test_failure_result(self) -> None:
        """Test creating a failure result."""
        result = OptimizationResult(
            node_id="node-2",
            action="spawn_training",
            success=False,
            message="Node not available",
        )
        assert result.success is False
        assert result.message == "Node not available"

    def test_result_with_workload(self) -> None:
        """Test result with workload config."""
        workload = WorkloadConfig(
            workload_type=WorkloadType.SELFPLAY,
            board_type=BoardType.SQUARE8,
        )
        result = OptimizationResult(
            node_id="node-1",
            action="spawn_selfplay",
            success=True,
            message="OK",
            workload=workload,
        )
        assert result.workload is not None
        assert result.workload.board_type == BoardType.SQUARE8


# ============================================================================
# GPU Capability Constants Tests
# ============================================================================


class TestGPUCapabilities:
    """Tests for GPU capability constants."""

    def test_gpu_capabilities_has_all_types(self) -> None:
        """Test GPU_CAPABILITIES has small, medium, and large GPUs."""
        # Small GPUs
        assert "RTX 3060" in GPU_CAPABILITIES
        assert "RTX 3070" in GPU_CAPABILITIES

        # Medium GPUs
        assert "RTX 4060 Ti" in GPU_CAPABILITIES
        assert "RTX 4080" in GPU_CAPABILITIES

        # Large GPUs
        assert "H100" in GPU_CAPABILITIES
        assert "A100" in GPU_CAPABILITIES

    def test_capability_structure(self) -> None:
        """Test each capability has max_board and preferred."""
        for gpu_name, caps in GPU_CAPABILITIES.items():
            assert "max_board" in caps, f"{gpu_name} missing max_board"
            assert "preferred" in caps, f"{gpu_name} missing preferred"
            assert isinstance(caps["max_board"], BoardType)
            assert isinstance(caps["preferred"], BoardType)

    def test_board_priorities_exist(self) -> None:
        """Test BOARD_PRIORITIES has all board types."""
        assert BoardType.HEX8 in BOARD_PRIORITIES
        assert BoardType.SQUARE8 in BOARD_PRIORITIES
        assert BoardType.SQUARE19 in BOARD_PRIORITIES
        assert BoardType.HEXAGONAL in BOARD_PRIORITIES


# ============================================================================
# _infer_capability_from_vram Tests
# ============================================================================


class TestInferCapabilityFromVram:
    """Tests for _infer_capability_from_vram function."""

    def test_large_vram_gets_hexagonal(self) -> None:
        """Test 40GB+ VRAM gets hexagonal capability."""
        caps = _infer_capability_from_vram(80.0)  # H100
        assert caps["max_board"] == BoardType.HEXAGONAL
        assert caps["preferred"] == BoardType.HEXAGONAL

    def test_medium_vram_gets_square19(self) -> None:
        """Test 20-40GB VRAM gets square19 capability."""
        caps = _infer_capability_from_vram(24.0)  # RTX 4090
        assert caps["max_board"] == BoardType.SQUARE19
        assert caps["preferred"] == BoardType.SQUARE19

    def test_small_vram_12gb_gets_square19_max(self) -> None:
        """Test 12-20GB VRAM gets square19 max, square8 preferred."""
        caps = _infer_capability_from_vram(12.0)  # RTX 3060
        assert caps["max_board"] == BoardType.SQUARE19
        assert caps["preferred"] == BoardType.SQUARE8

    def test_small_vram_8gb_gets_square8_max(self) -> None:
        """Test 8-12GB VRAM gets square8 max."""
        caps = _infer_capability_from_vram(8.0)
        assert caps["max_board"] == BoardType.SQUARE8
        assert caps["preferred"] == BoardType.HEX8

    def test_tiny_vram_gets_hex8(self) -> None:
        """Test <8GB VRAM gets hex8 only."""
        caps = _infer_capability_from_vram(4.0)
        assert caps["max_board"] == BoardType.HEX8
        assert caps["preferred"] == BoardType.HEX8


# ============================================================================
# load_gpu_profiles_from_config Tests
# ============================================================================


class TestLoadGpuProfilesFromConfig:
    """Tests for load_gpu_profiles_from_config function."""

    def test_loads_from_cluster_config(self) -> None:
        """Test loading GPU profiles from cluster config."""
        with patch("app.config.cluster_config.get_gpu_types") as mock_get:
            mock_get.return_value = {
                "Custom GPU X": 32,  # New GPU with 32GB VRAM
            }
            profiles = load_gpu_profiles_from_config()

            # Should create profile for new GPU
            assert "Custom GPU X" in profiles
            assert profiles["Custom GPU X"]["max_board"] == BoardType.SQUARE19

    def test_skips_existing_gpu_types(self) -> None:
        """Test that static GPU types are not duplicated."""
        with patch("app.config.cluster_config.get_gpu_types") as mock_get:
            mock_get.return_value = {
                "H100": 80,  # Already in GPU_CAPABILITIES
            }
            profiles = load_gpu_profiles_from_config()

            # Should NOT include H100 (already in static)
            assert "H100" not in profiles

    def test_skips_zero_vram(self) -> None:
        """Test GPUs with no VRAM info are skipped."""
        with patch("app.config.cluster_config.get_gpu_types") as mock_get:
            mock_get.return_value = {
                "Unknown GPU": 0,
            }
            profiles = load_gpu_profiles_from_config()

            assert "Unknown GPU" not in profiles

    def test_import_error_returns_empty(self) -> None:
        """Test ImportError returns empty dict."""
        with patch(
            "app.config.cluster_config.get_gpu_types",
            side_effect=ImportError,
        ):
            profiles = load_gpu_profiles_from_config()
            assert profiles == {}


# ============================================================================
# get_gpu_capabilities Tests
# ============================================================================


class TestGetGpuCapabilities:
    """Tests for get_gpu_capabilities function."""

    def test_combines_static_and_dynamic(self) -> None:
        """Test that capabilities combine static and dynamic profiles."""
        # Reset dynamic profiles
        import app.coordination.utilization_optimizer as mod
        mod._DYNAMIC_GPU_PROFILES = {}

        with patch.object(mod, "load_gpu_profiles_from_config") as mock_load:
            mock_load.return_value = {
                "New GPU": {"max_board": BoardType.HEXAGONAL, "preferred": BoardType.HEXAGONAL}
            }
            caps = get_gpu_capabilities()

            # Should have both static and dynamic
            assert "H100" in caps  # Static
            assert "New GPU" in caps  # Dynamic

    def test_static_takes_precedence(self) -> None:
        """Test that static profiles override dynamic ones."""
        import app.coordination.utilization_optimizer as mod
        mod._DYNAMIC_GPU_PROFILES = {
            "H100": {"max_board": BoardType.HEX8, "preferred": BoardType.HEX8}  # Wrong!
        }

        caps = get_gpu_capabilities()

        # Static H100 should take precedence
        assert caps["H100"]["max_board"] == BoardType.HEXAGONAL


# ============================================================================
# UtilizationOptimizer Tests
# ============================================================================


class TestUtilizationOptimizerInit:
    """Tests for UtilizationOptimizer initialization."""

    def test_init_with_default_orchestrator(self) -> None:
        """Test initialization with default health orchestrator."""
        with patch("app.coordination.utilization_optimizer.get_health_orchestrator") as mock_get:
            mock_orchestrator = MagicMock()
            mock_get.return_value = mock_orchestrator

            optimizer = UtilizationOptimizer()

            assert optimizer.health_orchestrator is mock_orchestrator
            mock_get.assert_called_once()

    def test_init_with_custom_orchestrator(self) -> None:
        """Test initialization with custom health orchestrator."""
        mock_orchestrator = MagicMock()
        optimizer = UtilizationOptimizer(health_orchestrator=mock_orchestrator)

        assert optimizer.health_orchestrator is mock_orchestrator

    def test_init_workload_state(self) -> None:
        """Test workload state is initialized correctly."""
        mock_orchestrator = MagicMock()
        optimizer = UtilizationOptimizer(health_orchestrator=mock_orchestrator)

        assert optimizer.node_workloads == {}
        assert BoardType.HEX8 in optimizer.board_data_needs
        assert BoardType.HEXAGONAL in optimizer.board_data_needs


class TestUtilizationOptimizerBoardSizeOrder:
    """Tests for board size ordering."""

    def test_board_size_order(self) -> None:
        """Test board size ordering is correct."""
        mock_orchestrator = MagicMock()
        optimizer = UtilizationOptimizer(health_orchestrator=mock_orchestrator)

        assert optimizer._board_size_order(BoardType.HEX8) == 1
        assert optimizer._board_size_order(BoardType.SQUARE8) == 2
        assert optimizer._board_size_order(BoardType.SQUARE19) == 3
        assert optimizer._board_size_order(BoardType.HEXAGONAL) == 4


class TestUtilizationOptimizerEngineSelection:
    """Tests for engine selection."""

    @pytest.fixture
    def optimizer(self) -> UtilizationOptimizer:
        """Create optimizer with mock orchestrator."""
        mock_orchestrator = MagicMock()
        return UtilizationOptimizer(health_orchestrator=mock_orchestrator)

    def test_lambda_gets_gumbel(self, optimizer: UtilizationOptimizer) -> None:
        """Test Lambda provider gets GPU selfplay engine."""
        health = MagicMock()
        health.provider = Provider.LAMBDA

        engine = optimizer._select_engine_for_node(health)

        assert engine == "gumbel"

    def test_vast_gets_heuristic(self, optimizer: UtilizationOptimizer) -> None:
        """Test Vast provider gets CPU selfplay engine."""
        health = MagicMock()
        health.provider = Provider.VAST

        engine = optimizer._select_engine_for_node(health)

        assert engine == "heuristic"

    def test_hetzner_gets_heuristic(self, optimizer: UtilizationOptimizer) -> None:
        """Test Hetzner provider gets CPU selfplay engine."""
        health = MagicMock()
        health.provider = Provider.HETZNER

        engine = optimizer._select_engine_for_node(health)

        assert engine == "heuristic"

    def test_aws_gets_heuristic(self, optimizer: UtilizationOptimizer) -> None:
        """Test AWS provider gets heuristic (light duty)."""
        health = MagicMock()
        health.provider = Provider.AWS

        engine = optimizer._select_engine_for_node(health)

        assert engine == "heuristic"

    def test_unknown_provider_gets_heuristic(
        self, optimizer: UtilizationOptimizer
    ) -> None:
        """Test unknown provider defaults to heuristic."""
        health = MagicMock()
        health.provider = MagicMock()  # Unknown provider

        engine = optimizer._select_engine_for_node(health)

        assert engine == "heuristic"


class TestUtilizationOptimizerGpuCapability:
    """Tests for GPU capability detection."""

    @pytest.fixture
    def optimizer(self) -> UtilizationOptimizer:
        """Create optimizer with mock orchestrator."""
        mock_orchestrator = MagicMock()
        return UtilizationOptimizer(health_orchestrator=mock_orchestrator)

    def test_no_instance_returns_none(self, optimizer: UtilizationOptimizer) -> None:
        """Test no instance returns None."""
        health = MagicMock()
        health.instance = None

        caps = optimizer._get_gpu_capability(health)

        assert caps is None

    def test_known_gpu_returns_capability(
        self, optimizer: UtilizationOptimizer
    ) -> None:
        """Test known GPU type returns correct capability."""
        health = MagicMock()
        health.instance.gpu_type = "NVIDIA H100 PCIe"
        health.instance.gpu_memory_gb = 80

        caps = optimizer._get_gpu_capability(health)

        assert caps is not None
        assert caps["max_board"] == BoardType.HEXAGONAL

    def test_unknown_gpu_uses_vram_inference(
        self, optimizer: UtilizationOptimizer
    ) -> None:
        """Test unknown GPU falls back to VRAM inference."""
        health = MagicMock()
        health.instance.gpu_type = "Unknown Custom GPU"
        health.instance.gpu_memory_gb = 48

        caps = optimizer._get_gpu_capability(health)

        assert caps is not None
        assert caps["max_board"] == BoardType.HEXAGONAL  # 48GB -> hexagonal


class TestUtilizationOptimizerUnderutilizedNodes:
    """Tests for underutilized node detection."""

    @pytest.fixture
    def optimizer(self) -> UtilizationOptimizer:
        """Create optimizer with mock orchestrator."""
        mock_orchestrator = MagicMock()
        return UtilizationOptimizer(health_orchestrator=mock_orchestrator)

    @pytest.mark.asyncio
    async def test_detects_underutilized_gpu_node(
        self, optimizer: UtilizationOptimizer
    ) -> None:
        """Test detection of underutilized GPU node."""
        health = MagicMock()
        health.is_available.return_value = True
        health.instance.gpu_count = 1
        health.gpu_percent = 10.0  # < 20% threshold

        optimizer.health_orchestrator.node_health = {"gpu-node": health}

        underutilized = await optimizer.get_underutilized_nodes()

        assert len(underutilized) == 1
        assert underutilized[0][0] == "gpu-node"

    @pytest.mark.asyncio
    async def test_detects_underutilized_cpu_node(
        self, optimizer: UtilizationOptimizer
    ) -> None:
        """Test detection of underutilized CPU node."""
        health = MagicMock()
        health.is_available.return_value = True
        health.instance.gpu_count = 0
        health.cpu_percent = 15.0  # < 30% threshold

        optimizer.health_orchestrator.node_health = {"cpu-node": health}

        underutilized = await optimizer.get_underutilized_nodes()

        assert len(underutilized) == 1
        assert underutilized[0][0] == "cpu-node"

    @pytest.mark.asyncio
    async def test_excludes_unavailable_nodes(
        self, optimizer: UtilizationOptimizer
    ) -> None:
        """Test that unavailable nodes are excluded."""
        health = MagicMock()
        health.is_available.return_value = False
        health.instance.gpu_count = 1
        health.gpu_percent = 10.0

        optimizer.health_orchestrator.node_health = {"offline-node": health}

        underutilized = await optimizer.get_underutilized_nodes()

        assert len(underutilized) == 0

    @pytest.mark.asyncio
    async def test_excludes_well_utilized_nodes(
        self, optimizer: UtilizationOptimizer
    ) -> None:
        """Test that well-utilized nodes are excluded."""
        health = MagicMock()
        health.is_available.return_value = True
        health.instance.gpu_count = 1
        health.gpu_percent = 75.0  # > 20% threshold

        optimizer.health_orchestrator.node_health = {"busy-node": health}

        underutilized = await optimizer.get_underutilized_nodes()

        assert len(underutilized) == 0


class TestUtilizationOptimizerSpawnSelfplay:
    """Tests for spawning selfplay jobs."""

    @pytest.fixture
    def optimizer(self) -> UtilizationOptimizer:
        """Create optimizer with mock orchestrator."""
        mock_orchestrator = MagicMock()
        return UtilizationOptimizer(health_orchestrator=mock_orchestrator)

    @pytest.mark.asyncio
    async def test_spawn_fails_node_not_found(
        self, optimizer: UtilizationOptimizer
    ) -> None:
        """Test spawn fails if node not found."""
        optimizer.health_orchestrator.get_node_health.return_value = None

        result = await optimizer.spawn_selfplay_job("missing-node")

        assert result.success is False
        assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_spawn_fails_node_unavailable(
        self, optimizer: UtilizationOptimizer
    ) -> None:
        """Test spawn fails if node unavailable."""
        health = MagicMock()
        health.is_available.return_value = False
        health.state.value = "offline"
        optimizer.health_orchestrator.get_node_health.return_value = health

        result = await optimizer.spawn_selfplay_job("offline-node")

        assert result.success is False
        assert "not available" in result.message.lower()

    @pytest.mark.asyncio
    async def test_spawn_respects_aws_limits(
        self, optimizer: UtilizationOptimizer
    ) -> None:
        """Test spawn respects AWS job limits."""
        health = MagicMock()
        health.is_available.return_value = True
        health.provider = Provider.AWS
        optimizer.health_orchestrator.get_node_health.return_value = health

        # Set node workload at max
        optimizer.node_workloads["aws-node"] = NodeWorkload(
            node_id="aws-node",
            selfplay_jobs=2,  # At AWS_MAX_SELFPLAY_JOBS
        )

        result = await optimizer.spawn_selfplay_job("aws-node")

        assert result.success is False
        assert "max selfplay jobs" in result.message.lower()


# ============================================================================
# Singleton Tests
# ============================================================================


class TestGetUtilizationOptimizer:
    """Tests for get_utilization_optimizer singleton."""

    def test_returns_same_instance(self) -> None:
        """Test singleton returns same instance."""
        # Reset singleton
        import app.coordination.utilization_optimizer as mod
        mod._utilization_optimizer_instance = None

        with patch("app.coordination.utilization_optimizer.get_health_orchestrator"):
            opt1 = get_utilization_optimizer()
            opt2 = get_utilization_optimizer()

            assert opt1 is opt2
